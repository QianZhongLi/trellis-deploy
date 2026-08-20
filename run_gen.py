#!/usr/bin/env python3
# TRELLIS core generator (lean, proven): fp16 sampling + GS decode fp16-mixed + PLY + render preview.
# Designed to run as a subprocess per task (fresh process => stable VRAM, avoids gradio's ~4GB overhead).
# Usage: python run_gen.py <input.png> <out_dir>
import os, sys, gc, json, time, traceback
os.environ["ATTN_BACKEND"]="xformers"
os.environ["SPARSE_ATTN_BACKEND"]="xformers"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0,"/home/Luna/git/qzl/TRELLIS")

IN_IMG=sys.argv[1]; OUT=sys.argv[2]
os.makedirs(OUT, exist_ok=True)
STATUS=os.path.join(OUT,"status.json")
def set_stat(d):
    with open(STATUS,"w") as f: json.dump(d,f)
def log(m):
    with open(os.path.join(OUT,"log.txt"),"a") as f: f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")

import torch, imageio, numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.sparse import SparseGroupNorm32
from trellis.pipelines.base import Pipeline
from trellis.utils import render_utils

set_stat({"status":"running","stage":"loading","peakMB":0})
p=TrellisImageTo3DPipeline.from_pretrained("/home/Luna/git/qzl/trellis-weights/microsoft/TRELLIS-image-large")
names=["image_cond_model","sparse_structure_flow_model","sparse_structure_decoder","slat_flow_model","slat_decoder_gs","slat_decoder_mesh"]
def _d(self):
    for n in names:
        m=self.models.get(n)
        if m is not None:
            try:
                if next(m.parameters()).is_cuda: return torch.device("cuda")
            except: pass
    return torch.device("cpu")
Pipeline.device=property(_d)
log("loaded"); set_stat({"status":"running","stage":"cond"})

try:
    # cond fp16
    p.models["image_cond_model"]=p.models["image_cond_model"].float().half().cuda()
    img=Image.open(IN_IMG).convert("RGBA")
    img_p=p.preprocess_image(img)
    with torch.autocast("cuda",dtype=torch.float16):
        cond=p.get_cond([img_p])
    cond={"cond":cond["cond"].half().cuda(),"neg_cond":cond["neg_cond"].half().cuda()}
    p.models["image_cond_model"].cpu().float(); gc.collect(); torch.cuda.empty_cache()
    log("cond done"); set_stat({"status":"running","stage":"ss"})

    # ss sampling fp16
    p.models["sparse_structure_flow_model"]=p.models["sparse_structure_flow_model"].float().half().cuda()
    p.models["sparse_structure_decoder"]=p.models["sparse_structure_decoder"].float().half().cuda()
    ssp=dict(p.sparse_structure_sampler_params); ssp["steps"]=12; ssp["cfg_strength"]=7.5
    with torch.autocast("cuda",dtype=torch.float16):
        coords=p.sample_sparse_structure(cond,1,ssp)
    p.models["sparse_structure_flow_model"].cpu().float(); p.models["sparse_structure_decoder"].cpu().float()
    gc.collect(); torch.cuda.empty_cache()
    log("ss coords="+str(tuple(coords.shape))); set_stat({"status":"running","stage":"slat"})

    # slat sampling fp16
    p.models["slat_flow_model"]=p.models["slat_flow_model"].float().half().cuda()
    slp=dict(p.slat_sampler_params); slp["steps"]=12; slp["cfg_strength"]=3.0
    with torch.autocast("cuda",dtype=torch.float16):
        slat=p.sample_slat(cond,coords,slp)
    del cond, coords
    p.models["slat_flow_model"].cpu().float(); gc.collect(); torch.cuda.empty_cache()
    log("slat feats="+str(slat.feats.dtype)+" "+str(tuple(slat.feats.shape))); set_stat({"status":"running","stage":"decode"})

    # GS decode fp16-mixed
    m=p.models["slat_decoder_gs"]
    m=m.float().cpu(); m.use_fp16=False; m.dtype=torch.float16
    for sub in m.modules():
        if hasattr(sub,"use_fp16"): sub.use_fp16=False
        if isinstance(getattr(sub,"dtype",None),torch.dtype): sub.dtype=torch.float16
    p.models["slat_decoder_gs"]=m
    m.half().cuda()
    for sub in m.modules():
        if type(sub).__name__=="SparseGroupNorm32":
            sub.float(); sub.cuda(); co=getattr(sub,"norm",None)
            if co is not None: co.float(); co.cuda()
    if slat.feats.dtype!=torch.float16:
        slat.data=slat.data.replace_feature(slat.feats.half().contiguous())
    ret=p.decode_slat(slat, formats=["gaussian"])
    gs=ret["gaussian"][0]
    peak=torch.cuda.max_memory_allocated()/1e6
    log("decode gs OK peak=%.0fMB"%peak)
    set_stat({"status":"running","stage":"export","peakMB":int(peak)})

    # export PLY
    gs_path=os.path.join(OUT,"output.ply")
    gs.save_ply(gs_path)
    log("ply saved "+gs_path)

    # render preview (rasterizer needs fp32; Gaussian holds fp16 tensors internally)
    try:
        for _k in ("_xyz","_features_dc","_features_rest","_scaling","_rotation","_opacity","_opacity_bias","scale_bias","rots_bias"):
            _v = getattr(gs, _k, None)
            if _v is not None: setattr(gs, _k, _v.float())
        v=render_utils.render_video(gs, num_frames=24)["color"]
        vpath=os.path.join(OUT,"preview.gif")
        imageio.mimsave(vpath, v, duration=0.1)
        log("preview saved "+vpath)
    except Exception as e:
        log("preview skip: "+repr(e)[:200])

    set_stat({"status":"done","peakMB":int(torch.cuda.max_memory_allocated()/1e6),
              "ply":"output.ply","preview":"preview.gif"})
    log("ALL DONE peak=%.0fMB"%torch.cuda.max_memory_allocated())
except Exception as e:
    tb=traceback.format_exc(); log("FAIL "+tb[-1200:])
    set_stat({"status":"error","error":tb[-1200:]})
    sys.exit(1)
