#!/usr/bin/env python3
"""
TRELLIS generator for 12GB GPU: reproduce pipeline.run stage-by-stage so we can
free big flow/decoder models between stages and keep peak VRAM low enough for a
12GB card (TRELLIS officially wants >=16GB).

Strategy (all official methods, no manual dtype juggling):
  - get_cond / sample_sparse_structure / sample_slat / decode_slat
  - override Pipeline.device so noise/coords stay on cuda
  - run decode inside torch.inference_mode() (drops gradient memory)
  - decode gaussian first -> save PLY -> free gs decoder
  - then decode mesh separately (GPU clean for the heavy mesh extraction)
  - to_glb + preview last

Usage:
  python run_gen_glb.py <input.png> <out_dir> [--seed N --ss_cfg F --ss_steps N
    --slat_cfg F --slat_steps N --texture_size N --simplify F]
"""
import os, sys, gc, json, time, torch, traceback, argparse, copy
import numpy as np
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRELLIS_MESH_RES_MULT", "4")
TRELLIS = "/home/Luna/git/qzl/TRELLIS"
sys.path.insert(0, TRELLIS)

from PIL import Image
import trimesh
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.base import Pipeline
from trellis.utils import postprocessing_utils, render_utils

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ss_cfg", type=float, default=7.5)
    ap.add_argument("--ss_steps", type=int, default=12)
    ap.add_argument("--slat_cfg", type=float, default=3.0)
    ap.add_argument("--slat_steps", type=int, default=12)
    ap.add_argument("--texture_size", type=int, default=1024)
    ap.add_argument("--simplify", type=float, default=0.95)
    ap.add_argument("--nviews", type=int, default=64)
    ap.add_argument("--mode", choices=["stochastic", "multidiffusion"], default="stochastic",
                    help="multi-image algorithm when multiple input images are given")
    args = ap.parse_args()

    OUT = args.out_dir; os.makedirs(OUT, exist_ok=True)
    STATUS = os.path.join(OUT, "status.json")
    def set_stat(d):
        with open(STATUS, "w") as f: json.dump(d, f)
    def log(m):
        with open(os.path.join(OUT, "log.txt"), "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")
    def peak():
        return int(torch.cuda.max_memory_allocated()/1e6)
    def free_all(*keys):
        for k in keys:
            m = p.models.get(k)
            if m is not None:
                try: m.cpu().float()
                except Exception: pass
        gc.collect(); torch.cuda.empty_cache()

    def detach_clone(obj):
        """Deep-copy an object (Gaussian/Mesh) cloning every tensor so results are
        normal tensors (not inference tensors) for postprocessing autograd."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone()
        if isinstance(obj, dict):
            return {k: detach_clone(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [detach_clone(v) for v in obj]
        # object with named attributes holding tensors
        d = getattr(obj, "__dict__", None)
        if d is None:
            return obj
        try:
            nc = copy.copy(obj)
        except Exception:
            return obj
        changed = False
        for k, v in d.items():
            nv = detach_clone(v)
            if nv is not v:
                setattr(nc, k, nv); changed = True
        return nc if changed else obj

    # ============ 流式显存优化版 to_glb（bake 观测逐帧 .cuda() 即用即释放）============
    def streaming_bake_texture(vertices, faces, uvs, observations, masks, extrinsics, intrinsics,
                               texture_size, near=0.1, far=10.0, lambda_tv=1e-2, verbose=False,
                               total_steps=2500):
        import utils3d, nvdiffrast.torch as dr, cv2
        from tqdm import tqdm
        # vertices/faces/uvs 是 CPU ndarray，optim 需要 texture 在 GPU，其余保持 CPU 按需上送
        vt = torch.tensor(vertices).cuda()
        ft = torch.tensor(faces.astype(np.int32)).cuda()
        nut = torch.tensor(uvs).cuda()
        views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
        projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]
        # UV 预计算（在 GPU 上做，结果移回 CPU 释放显存）
        rastctx = utils3d.torch.RastContext(backend='cuda')
        _uv_cpu, _uvdr_cpu = [], []
        for i, (view, proj) in enumerate(zip(views, projections)):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vt[None], ft, observations[i].shape[1], observations[i].shape[0],
                    uv=nut[None], view=view, projection=proj)
            _uv_cpu.append(rast['uv'].detach().cpu())
            _uvdr_cpu.append(rast['uv_dr'].detach().cpu())
        # 释放不需要常驻 GPU 的（保留 vt/ft/nut 后续 inpaint 用）
        gc.collect(); torch.cuda.empty_cache()
        # 纹理初始化为白色：任何未被相机观察到/未烘培的 UV 区域默认白（用户要求“没颜色的默认白”），
        # 而不是黑色。损失只在有掩码像素上计算，训练像素不受初始值影响，未训练区保持白。
        texture = torch.nn.Parameter(torch.ones((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)
        def cosine_anealing(step, ts, s, e):
            return e + 0.5 * (s - e) * (1 + np.cos(np.pi * step / ts))
        def tv_loss(t):
            return torch.nn.functional.l1_loss(t[:, :-1, :, :], t[:, 1:, :, :]) + \
                   torch.nn.functional.l1_loss(t[:, :, :-1, :], t[:, :, 1:, :])
        nviews = len(views)
        with tqdm(total=total_steps, disable=not verbose) as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                sel = np.random.randint(0, nviews)
                # 逐帧从 CPU 取：观测+掩码+uv 现在才上 GPU，用完即释放（内存缓冲核心）
                obs = torch.tensor(observations[sel] / 255.0, dtype=torch.float32).cuda().flip(0)
                msk = torch.tensor(masks[sel] > 0, dtype=torch.bool).cuda().flip(0)
                uv = _uv_cpu[sel].cuda()
                uv_dr = _uvdr_cpu[sel].cuda()
                render = dr.texture(texture, uv, uv_dr)[0]
                loss = torch.nn.functional.l1_loss(render[msk], obs[msk])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()
                optimizer.param_groups[0]['lr'] = cosine_anealing(step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()}); pbar.update()
                # 立即释放本帧 GPU 临时量
                del obs, msk, uv, uv_dr, render, loss
                if step % 20 == 0:
                    torch.cuda.empty_cache()
        tex = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        # 释放所有临时 GPU（vt/ft/nut 保留到返回前）
        del texture, optimizer, _uv_cpu, _uvdr_cpu
        gc.collect(); torch.cuda.empty_cache()
        # inpaint 未覆盖区域
        # 1) 小洞/部分遮挡：用邻居颜色平滑填补（避免墙壁出现突兀白斑或黑斑）
        m = np.zeros((texture_size, texture_size), dtype=np.uint8)
        try:
            with torch.no_grad():
                m = 1 - utils3d.torch.rasterize_triangle_faces(
                    rastctx, (nut * 2 - 1)[None], ft, texture_size, texture_size
                )['mask'][0].detach().cpu().numpy().astype(np.uint8)
            tex = cv2.inpaint(tex, m, 3, cv2.INPAINT_TELEA)
        except Exception:
            pass
        # 2) 大洞/完全无颜色区域（如底面、相机从未覆盖的墙面）：默认白，不做死黑
        #    inpaint 对超大洞无效（保持黑/残留），把仍为黑的像素统一置白，符合
        #    “没颜色的默认白”要求；小洞已被 inpaint 用邻居颜色填好，不再覆盖。
        try:
            m = (m > 0)
            dark = tex.sum(axis=-1) < 30   # 近黑(残留黑/大洞未填)的像素视为无颜色
            tex[m & dark] = 255
        except Exception:
            pass
        return tex

    def streaming_to_glb(app_rep, mesh, simplify=0.95, texture_size=1024, nviews=64, fill_holes=True):
        """流式 to_glb：render_multiview 降 nviews + bake 观测逐帧上送，峰值显存大幅下降。"""
        import utils3d
        vertices = mesh.vertices.cpu().numpy()
        faces = mesh.faces.cpu().numpy()
        # 1) mesh 后处理（简化 + 去不可见面 + 补洞）
        vertices, faces = postprocessing_utils.postprocess_mesh(
            vertices, faces, simplify=simplify > 0, simplify_ratio=simplify,
            fill_holes=fill_holes, fill_holes_max_hole_size=0.04,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
            fill_holes_resolution=1024, fill_holes_num_views=1000, verbose=False)
        gc.collect(); torch.cuda.empty_cache()
        # 2) UV 参数化（xatlas，CPU）
        vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)
        # 3) 多视角渲染（GS -> CPU numpy，nviews 可调）
        observations, extrinsics, intrinsics = render_utils.render_multiview(app_rep, resolution=1024, nviews=nviews)
        # 掩码判定“有颜色”：仅亮度(最大通道)足够高的像素算有颜色。
        # 近黑像素（背景0、掠射角高斯轻微泄漏如[4,5,16]、未覆盖底面）视为“无颜色”
        # → 不参与烘培 → 保持默认白，避免死黑。真实深色细节(>25)仍正常烘培。
        masks = [np.max(o, axis=-1) >= 25 for o in observations]
        extr = [e.cpu().numpy() for e in extrinsics]
        intr = [i.cpu().numpy() for i in intrinsics]
        gc.collect(); torch.cuda.empty_cache()
        # 4) 流式 bake（观测逐帧上送 GPU）
        texture = streaming_bake_texture(vertices, faces, uvs, observations, masks, extr, intr,
                                          texture_size=texture_size, lambda_tv=0.01, verbose=False)
        from PIL import Image as _Image
        texture = _Image.fromarray(texture)
        # 5) mesh z-up -> y-up + 材质（含完整 PBR 贴图）
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0, baseColorTexture=texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8))
        return trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))

    def render_mesh_preview(mesh, num_frames=24, resolution=512, fov=40, r=2.2):
        """用 nvdiffrast 从烘焙后的 GLB 网格渲染轨道预览，替代原始高斯 splat 预览。
        原始高斯预览用 view-dependent SH 着色，远处/背面墙在某些角度渲染成黑（
        即便本质是打光/视角伪影，不是模型的真实颜色）；而 GLB 贴图是干净烘焙的。
        从 GLB 网格渲染可让预览如实反映交付物，不再出现死黑墙。"""
        import utils3d, nvdiffrast.torch as dr
        # mesh 是 y-up；相机用 z-up 轨道（look_at 原点, up=(0,0,1)）→ 先把网格转回 z-up
        verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
        verts = verts @ torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32).cuda()
        faces = torch.tensor(mesh.faces.astype(np.int32)).cuda()
        mat = getattr(mesh.visual, 'material', None)
        tex_img = None
        if mat is not None and hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            tex_img = mat.baseColorTexture.convert('RGB')
        if tex_img is None:
            color = np.zeros((1, 3))
        else:
            color = np.array(tex_img) / 255.0
        tex = torch.tensor(color, dtype=torch.float32).cuda()
        if len(tex.shape) == 3:
            tex = tex[None]  # dr.texture 需要 batch 维 (1,H,W,3)
        uvs = getattr(mesh.visual, 'uv', None)
        has_uv = uvs is not None
        if has_uv:
            uv = torch.tensor(uvs.astype(np.float32)).cuda()
            uv_db = (uv * 2 - 1)
        # 轨道相机（与旧预览一致的 yaw/pitch）
        yaws = torch.linspace(0, 2 * 3.1415, num_frames).tolist()
        pitchs = (0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))).tolist()
        extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
        rastctx = utils3d.torch.RastContext(backend='cuda')
        with torch.no_grad():
            frames = []
            for i in range(num_frames):
                view = utils3d.torch.extrinsics_to_view(extr[i])
                proj = utils3d.torch.intrinsics_to_perspective(intr[i], 0.1, 10.0)
                if has_uv:
                    rast = utils3d.torch.rasterize_triangle_faces(
                        rastctx, verts[None], faces, resolution, resolution, uv=uv_db[None],
                        view=view, projection=proj)
                    mask = rast['mask'][0]          # (H,W,1)
                    col = dr.texture(tex, rast['uv'], rast['uv_dr'])[0]  # (H,W,3) 0..1
                else:
                    rast = utils3d.torch.rasterize_triangle_faces(
                        rastctx, verts[None], faces, resolution, resolution, view=view, projection=proj)
                    mask = rast['mask'][0]
                    col = torch.full((resolution, resolution, 3), 0.6, device='cuda')
                # 背景黑、模型区用贴图颜色（mask 广播到 3 通道）
                m = mask.float()
                if m.ndim == 2:
                    m = m.unsqueeze(-1)
                img = col * m
                img = (torch.clamp(img, 0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
                frames.append(np.flip(img, axis=0))
        del verts, faces, tex, rastctx, col
        if has_uv:
            del uv, uv_db
        gc.collect(); torch.cuda.empty_cache()
        return frames

    try:
        set_stat({"status": "running", "stage": "loading", "peakMB": 0})
        log(f"loading seed={args.seed} ss={args.ss_steps}@{args.ss_cfg} slat={args.slat_steps}@{args.slat_cfg} tex={args.texture_size}")
        p = TrellisImageTo3DPipeline.from_pretrained(
            "/home/Luna/git/qzl/trellis-weights/microsoft/TRELLIS-image-large")
        # keep self.device on cuda (official Pipeline.device otherwise reports cpu here)
        def _dev(self):
            for n in ("image_cond_model","slat_flow_model","slat_decoder_gs","slat_decoder_mesh"):
                m = self.models.get(n)
                if m is not None:
                    try:
                        if next(m.parameters()).is_cuda: return torch.device("cuda")
                    except Exception: pass
            return torch.device("cpu")
        Pipeline.device = property(_dev)
        p.cuda()
        log("pipeline loaded")

        # 支持多图：input 可用逗号分隔多个路径
        input_paths = [s.strip() for s in args.input.split(",") if s.strip()]
        images = [Image.open(p).convert("RGBA") for p in input_paths]
        multi = len(images) > 1
        set_stat({"status": "running", "stage": "gen", "peakMB": peak()})
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            imgs_p = [p.preprocess_image(img) for img in images]
            cond = p.get_cond(imgs_p)   # list -> batched (B,518,518,3) features
            cond = {"cond": cond["cond"], "neg_cond": cond["neg_cond"]}
            if multi:
                # 官方 run_multi_image：neg_cond 先切成单张（空条件），stochastic 注入只切 cond
                cond["neg_cond"] = cond["neg_cond"][:1]
                log(f"multi-image mode={args.mode} n={len(images)}")
                with p.inject_sampler_multi_image(
                        "sparse_structure_sampler", len(images), args.ss_steps, mode=args.mode):
                    coords = p.sample_sparse_structure(cond, 1, {"steps": args.ss_steps, "cfg_strength": args.ss_cfg})
            else:
                coords = p.sample_sparse_structure(cond, 1, {"steps": args.ss_steps, "cfg_strength": args.ss_cfg})
            # free sparse-structure models before slat sampling
            free_all("sparse_structure_flow_model", "sparse_structure_decoder")
            log(f"ss done peak={peak()}MB")
            if multi:
                with p.inject_sampler_multi_image(
                        "slat_sampler", len(images), args.slat_steps, mode=args.mode):
                    slat = p.sample_slat(cond, coords, {"steps": args.slat_steps, "cfg_strength": args.slat_cfg})
            else:
                slat = p.sample_slat(cond, coords, {"steps": args.slat_steps, "cfg_strength": args.slat_cfg})
            del cond, coords
            free_all("slat_flow_model")
            log(f"slat done peak={peak()}MB")

            # ---- gaussian decode -> PLY (then free gs decoder) ----
            gs = None
            try:
                set_stat({"status": "running", "stage": "gen", "peakMB": peak()})
                ret = p.decode_slat(slat, formats=["gaussian"])
                gs = detach_clone(ret["gaussian"][0])
                free_all("slat_decoder_gs")
                log(f"gs decode peak={peak()}MB")
                gs_path = os.path.join(OUT, "output.ply")
                gs.save_ply(gs_path)
                log(f"PLY saved peak={peak()}MB")
            except Exception as e:
                log("gs/PLY FAIL: " + repr(e)[:200])

            # ---- mesh decode (GPU clean now) ----
            mesh = None
            try:
                set_stat({"status": "running", "stage": "glb", "peakMB": peak()})
                retm = p.decode_slat(slat, formats=["mesh"])
                mesh = detach_clone(retm["mesh"][0])
                free_all("slat_decoder_mesh")
                log(f"mesh decode peak={peak()}MB (verts={len(mesh.vertices)})")
            except Exception as e:
                log("mesh decode FAIL: " + repr(e)[:300])

        results = {}
        glm = None
        if gs is not None:
            results["ply"] = "output.ply"
        if mesh is not None:
            try:
                set_stat({"status": "running", "stage": "glb", "peakMB": peak()})
                # 流式 to_glb：bake 观测逐帧上送 GPU，峰值显存大幅下降
                torch.cuda.reset_peak_memory_stats()
                glm = streaming_to_glb(gs, mesh, simplify=args.simplify,
                                       texture_size=args.texture_size, nviews=args.nviews)
                glb_peak = peak()
                glb_path = os.path.join(OUT, "output.glb")
                glm.export(glb_path)
                results["glb"] = "output.glb"
                log(f"GLB exported tex={args.texture_size} nviews={args.nviews} bake_peak={glb_peak}MB overall_peak={peak()}MB size={os.path.getsize(glb_path)/1e6:.1f}MB")
            except Exception as e:
                log("GLB FAIL: " + repr(e)[:300])
            torch.cuda.empty_cache()

        if mesh is not None:
            try:
                set_stat({"status": "running", "stage": "preview", "peakMB": peak()})
                import imageio
                # 从烘焙后的 GLB 网格渲染预览（真实反映交付物，无原始高斯 view-dependent 死黑墙）
                if glm is not None:
                    v = render_mesh_preview(glm, num_frames=24)
                else:
                    with torch.inference_mode():
                        v = render_utils.render_video(gs, num_frames=24)["color"]
                imageio.mimsave(OUT + "/preview.gif", v, duration=0.1)
                results["preview"] = "preview.gif"
                log("preview saved (mesh render)")
            except Exception as e:
                log("preview skip: " + repr(e)[:150])

        if not results.get("glb") and not results.get("ply"):
            log("NO OUTPUT; error")
            set_stat({"status": "error", "error": "generation failed (see log.txt)"})
            sys.exit(1)
        final = {"status": "done", "peakMB": peak()}; final.update(results)
        set_stat(final); log("ALL DONE peak=%dMB" % peak())
    except Exception as e:
        tb = traceback.format_exc(); log("FAIL " + tb[-1500:])
        set_stat({"status": "error", "error": tb[-1500:]}); sys.exit(1)

if __name__ == "__main__":
    main()
