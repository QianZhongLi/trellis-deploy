#!/usr/bin/env python3
"""
TRELLIS 3D 生成 FastAPI 服务(独立进程每任务, GLB + PLY 输出)。
POST /api/trellis/generate  -> 收 multipart 图片 + 可选参数 -> {task_id}
GET  /api/trellis/task/{id} -> {status, glb_url, ply_url, preview_url, peakMB}
GET  /api/trellis/task/{id}/file/{name} -> 下载 output.glb / output.ply / preview.gif

每个任务 spawn 一个独立 python 子进程跑 run_gen_glb.py。
并发队列限制(GPU_QUEUE),避免显存 OOM。
"""
import os, sys, uuid, json, subprocess, shutil, threading, time
from typing import List
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = Path("/home/Luna/git/qzl/trellis-test")
RUN_GEN = BASE / "run_gen_glb.py"   # <- 切换到 GLB 版生成脚本
VENV_PY = "/home/Luna/git/qzl/trellis-venv/bin/python"
TASKS_DIR = Path("/opt/trellis/tasks")
TASKS_DIR.mkdir(parents=True, exist_ok=True)

GPU_QUEUE = int(os.environ.get("GPU_QUEUE", "1"))  # 并发上限, 显存有限保持1
_sem = threading.Semaphore(GPU_QUEUE)
_lock = threading.Lock()
_procs = {}

app = FastAPI(title="TRELLIS 3D API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEFAULTS = {
    "seed": 1,
    "ss_cfg": 7.5,
    "ss_steps": 12,
    "slat_cfg": 3.0,
    "slat_steps": 12,
    "texture_size": 1024,
    "simplify": 0.95,
}

@app.get("/", response_class=HTMLResponse)
def root():
    return open(r"/home/Luna/git/qzl/trellis-web/index.html", encoding="utf-8").read()

@app.get("/api/health")
def health():
    return {
        "service": "trellis-3d", "status": "ok",
        "endpoints": ["/api/trellis/generate", "/api/trellis/task/{id}"],
        "defaults": DEFAULTS,
    }

def _f(v, cast, dflt, lo=None, hi=None):
    """安全转换表单参数(字符串/数字),越界回退默认。"""
    try:
        x = cast(v)
    except Exception:
        return dflt
    if lo is not None and x < lo: return dflt
    if hi is not None and x > hi: return dflt
    return x

@app.post("/api/trellis/generate")
async def generate(
    image: UploadFile = File(None),
    images: List[UploadFile] = File(None),
    seed: str = Form(""),
    ss_cfg: str = Form(""),
    ss_steps: str = Form(""),
    slat_cfg: str = Form(""),
    slat_steps: str = Form(""),
    texture_size: str = Form(""),
    simplify: str = Form(""),
    mode: str = Form("stochastic"),
):
    task_id = uuid.uuid4().hex
    out_dir = TASKS_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 兼容：单图走 image，多图走 images[]；也可 image + images 并存
    files = list(images or [])
    if image is not None:
        files.insert(0, image)
    if not files:
        return JSONResponse({"error": "no image provided"}, status_code=400)
    img_paths = []
    for i, f in enumerate(files, start=1):
        ext = os.path.splitext(f.filename or "img")[1] or ".png"
        p = out_dir / f"input_{i}{ext}"
        p.write_bytes(await f.read())
        img_paths.append(str(p))

    with open(out_dir / "status.json", "w") as f:
        json.dump({"status": "queued"}, f)

    params = [
        ",".join(img_paths), str(out_dir),   # input(多图逗号分隔) out_dir
        "--seed", str(_f(seed, int, DEFAULTS["seed"])),
        "--ss_cfg", str(_f(ss_cfg, float, DEFAULTS["ss_cfg"], 0, 20)),
        "--ss_steps", str(_f(ss_steps, int, DEFAULTS["ss_steps"], 1, 100)),
        "--slat_cfg", str(_f(slat_cfg, float, DEFAULTS["slat_cfg"], 0, 20)),
        "--slat_steps", str(_f(slat_steps, int, DEFAULTS["slat_steps"], 1, 100)),
        "--texture_size", str(_f(texture_size, int, DEFAULTS["texture_size"], 512, 4096)),
        "--simplify", str(_f(simplify, float, DEFAULTS["simplify"], 0.0, 0.99)),
        "--mode", mode if mode in ("stochastic", "multidiffusion") else "stochastic",
    ]

    def _run():
        with _sem:
            if not (out_dir / "status.json").exists():
                return
            with open(out_dir / "status.json", "w") as f:
                json.dump({"status": "running"}, f)
            cmd = [VENV_PY, str(RUN_GEN)] + params
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with _lock: _procs[task_id] = proc
            proc.wait()
            with _lock: _procs.pop(task_id, None)

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "status_url": f"/api/trellis/task/{task_id}"}

@app.get("/api/trellis/task/{task_id}")
def task_status(task_id: str):
    out_dir = TASKS_DIR / task_id
    status_path = out_dir / "status.json"
    if not out_dir.exists():
        return JSONResponse({"error": "task not found"}, status_code=404)
    if not status_path.exists():
        return {"task_id": task_id, "status": "unknown"}
    data = json.loads(status_path.read_text())
    base = f"/api/trellis/task/{task_id}/file/"
    if data.get("status") == "done":
        for name in ("output.glb", "output.ply", "preview.gif"):
            if (out_dir / name).exists():
                key = {"output.glb": "glb_url", "output.ply": "ply_url", "preview.gif": "preview_url"}[name]
                data[key] = base + name
    return data

@app.get("/api/trellis/task/{task_id}/file/{name}")
def task_file(task_id: str, name: str):
    fp = TASKS_DIR / task_id / name
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    media = {
        "output.glb": "model/gltf-binary",
        "output.ply": "application/octet-stream",
        "preview.gif": "image/gif",
    }.get(name, "application/octet-stream")
    return FileResponse(fp, media_type=media, filename=name)

if __name__ == "__main__":
    import socket
    from uvicorn import Config, Server
    PORT = int(os.environ.get("PORT", "5801"))
    socks = []
    s4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s4.bind(("0.0.0.0", PORT)); s4.listen(2048); socks.append(s4)
    s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    s6.bind(("::", PORT)); s6.listen(2048); socks.append(s6)
    config = Config(app=app, log_level="info")
    server = Server(config=config)
    server.run(sockets=socks)
