# TRELLIS Deploy — Image to 3D (GLB) 部署仓库

微软 TRELLIS (Image→3D) 的部署封装，针对 **12GB 显存 GPU（RTX 4070）** 做了分步推理
优化，输出 GLB / PLY / 预览 GIF，提供 FastAPI REST 接口 + Web 前端。

## 目录结构
- `TRELLIS/`       — 官方 TRELLIS 源码（含 12GB 优化运行方式，已去除 .git）
- `app.py`         — FastAPI 服务（任务队列 / API / 日志给 systemd journalctl）
- `run_gen_glb.py` — 12GB 分步生成核心脚本（分步 + 流式 bake）
- `run_gen.py`     — 旧版生成脚本
- `web/index.html` — 前端页面（HuggingFace 风格，model-viewer 预览）
- `requirements.txt`

## 部署

### 1. 环境
```bash
python3 -m venv trellis-venv
source trellis-venv/bin/activate
pip install -r requirements.txt
```

### 2. 编译 TRELLIS 扩展
```bash
cd TRELLIS && bash setup.sh
```

### 3. 运行 API 服务（端口 5801）
```bash
export GPU_QUEUE=1
python app.py
```
systemd 单元 `trellis-api.service` 参考：
```ini
[Service]
WorkingDirectory=<repo>
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=GPU_QUEUE=1
Environment=PORT=5801
ExecStart=<venv>/bin/python app.py
Restart=always
```

## API
- `POST /api/trellis/generate` — multipart 上传图片（+ texture_size/simplify 等）→ 返回 task_id
- `GET  /api/trellis/task/{id}` — 查询状态：`{status, glb_url, ply_url, preview_url}`
- `GET  /api/trellis/task/{id}/file/{name}` — 下载 `output.glb` / `output.ply` / `preview.gif`

## 12GB 显存优化要点
- 复刻官方 pipeline 分步（get_cond → sample_sparse_structure → sample_slat → decode_slat），
  而非一次性 `pipeline.run`（12GB 卡会 OOM）
- 每阶段结束将 flow/decoder 模型 `.cpu()` 卸载到内存 + `gc.collect()` + `empty_cache()`
- decode 用 `torch.no_grad()`（省显存且不破坏 to_glb 的 autograd 要求）
- `TRELLIS_MESH_RES_MULT=4`；GPU_QUEUE=1 串行
- 实测峰值 ~8GB（12GB 卡安全）

## 模型下载
首次运行自动从 HuggingFace 下载权重到 `~/.cache/huggingface`。
