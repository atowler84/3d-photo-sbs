import os
import uuid
import yaml
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import subprocess
import threading
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Spatial-Photo 3D Photo API")

static_dir = os.path.join(PROJECT_ROOT, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def ensure_weights(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    checkpoints = {
        "depth_edge_model_ckpt": cfg["depth_edge_model_ckpt"],
        "depth_feat_model_ckpt": cfg["depth_feat_model_ckpt"],
        "rgb_feat_model_ckpt": cfg["rgb_feat_model_ckpt"],
    }
    # Only prepare MiDaS checkpoint if explicitly required
    paths = list(checkpoints.values())
    if cfg.get("require_midas") is True:
        midas_path = cfg.get("MiDaS_model_ckpt", "MiDaS/model.pt")
        paths.append(midas_path)
    for path in paths:
        full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)


def preload_depth_anything_v2():
    try:
        # Preload once to populate HF cache for faster first job
        from transformers import pipeline
        pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    except Exception:
        pass


jobs = {}

def _run_job(
    work_id: str,
    out_key: str,
    encoder: Optional[str] = None,
    fast: Optional[bool] = None,
    longer_side: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    fps: Optional[int] = None,
    loop: Optional[bool] = None,
    crop_top_px: Optional[int] = None,
    crop_bottom_px: Optional[int] = None,
    crop_left_px: Optional[int] = None,
    crop_right_px: Optional[int] = None,
    speed: Optional[float] = None,
    debug: Optional[bool] = None,
):
    image_dir = os.path.join(PROJECT_ROOT, "image")
    depth_dir = os.path.join(PROJECT_ROOT, "depth")
    video_dir = os.path.join(PROJECT_ROOT, "video")
    mesh_dir = os.path.join(PROJECT_ROOT, "mesh")
    progress_file = os.path.join(PROJECT_ROOT, f".progress_{work_id}.json")
    env = os.environ.copy()
    env["SPECIFIC_NAME"] = out_key
    env["PROGRESS_FILE"] = progress_file
    # Default processing resolution; allow override
    if longer_side is not None and int(longer_side) > 0:
        env["LONGER_SIDE_LEN"] = str(int(longer_side))
    else:
        env.setdefault("LONGER_SIDE_LEN", "768")
    env.setdefault("SAVE_PLY", "0")
    env["SRC_FOLDER"] = image_dir
    env["DEPTH_FOLDER"] = depth_dir
    env["MESH_FOLDER"] = mesh_dir
    # Stage videos into a per-job temp dir; copy to static only after success
    staged_video_dir = os.path.join(video_dir, work_id)
    os.makedirs(staged_video_dir, exist_ok=True)
    env["VIDEO_FOLDER"] = staged_video_dir
    if encoder:
        env["DEPTH_ANYTHING_ENCODER"] = encoder
    if fast:
        env["FAST_MODE"] = "1"
    if duration_seconds is not None and duration_seconds > 0:
        env["DURATION_SECONDS"] = str(duration_seconds)
    if fps is not None and fps > 0:
        env["FPS_OVERRIDE"] = str(fps)
    if loop:
        env["LOOP_MODE"] = "1"
    if speed is not None:
        env["SPEED_MULTIPLIER"] = str(speed)
    # Always enable debug directory so we can stream intermediate assets in UI
    env["DEBUG_MODE"] = "1"
    debug_subdir = os.path.join(static_dir, f"dbg_{work_id}")
    os.makedirs(debug_subdir, exist_ok=True)
    env["DEBUG_DIR"] = debug_subdir
    env["WORK_ID"] = work_id
    # crop px borders (applied after aspect crop)
    if crop_top_px is not None:
        env["CROP_TOP_PX"] = str(max(0, int(crop_top_px)))
    if crop_bottom_px is not None:
        env["CROP_BOTTOM_PX"] = str(max(0, int(crop_bottom_px)))
    if crop_left_px is not None:
        env["CROP_LEFT_PX"] = str(max(0, int(crop_left_px)))
    if crop_right_px is not None:
        env["CROP_RIGHT_PX"] = str(max(0, int(crop_right_px)))
    cmd = [
        "python",
        os.path.join(PROJECT_ROOT, "main.py"),
        "--config",
        os.path.join(PROJECT_ROOT, "argument.yml"),
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    jobs[work_id]["returncode"] = proc.returncode
    jobs[work_id]["stdout"] = proc.stdout[-4000:]
    jobs[work_id]["stderr"] = proc.stderr[-4000:]
    # Prepare result if success
    if proc.returncode == 0:
        videos = []
        # copy staged videos into static/ now that generation finished
        if os.path.isdir(staged_video_dir):
            for f in os.listdir(staged_video_dir):
                if f.startswith(out_key) and f.lower().endswith((".mp4", ".webm", ".mov")):
                    src = os.path.join(staged_video_dir, f)
                    dst = os.path.join(static_dir, f)
                    try:
                        if not os.path.exists(dst):
                            with open(src, "rb") as rf, open(dst, "wb") as wf:
                                wf.write(rf.read())
                    except Exception:
                        pass
        # collect videos from static after copy
        if os.path.isdir(static_dir):
            for f in os.listdir(static_dir):
                if f.startswith(out_key) and f.lower().endswith((".mp4", ".webm", ".mov")):
                    videos.append(f"/static/{f}")
        debug_assets = []
        # collect debug files if any
        dbg_dir = os.path.join(static_dir, f"dbg_{work_id}")
        if os.path.isdir(dbg_dir):
            for f in sorted(os.listdir(dbg_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    debug_assets.append(f"/static/dbg_{work_id}/{f}")
        # Mesh serving disabled by default
        jobs[work_id]["result"] = {"key": out_key, "videos": videos, "mesh": None, "debug_assets": debug_assets}
    # Mark done
    jobs[work_id]["done"] = True
    if proc.returncode != 0:
        # Print logs to server console for easier debugging
        print(f"[JOB {work_id}] Generation failed with code {proc.returncode}")
        try:
            print(proc.stdout)
        except Exception:
            pass
        try:
            print(proc.stderr)
        except Exception:
            pass


@app.post("/api/generate")
async def generate_3d(
    image: UploadFile = File(...),
    motion: Optional[str] = Form(None),
    encoder: Optional[str] = Form(None),
    longer_side: Optional[int] = Form(None),
    fast: Optional[bool] = Form(False),
    duration: Optional[float] = Form(None),
    fps: Optional[int] = Form(None),
    loop: Optional[bool] = Form(False),
    crop_top: Optional[int] = Form(None),
    crop_bottom: Optional[int] = Form(None),
    crop_left: Optional[int] = Form(None),
    crop_right: Optional[int] = Form(None),
    speed: Optional[float] = Form(None),
    debug: Optional[bool] = Form(False),
):
    try:
        work_id = str(uuid.uuid4())[:8]
        # Prepare workspace
        image_dir = os.path.join(PROJECT_ROOT, "image")
        depth_dir = os.path.join(PROJECT_ROOT, "depth")
        video_dir = os.path.join(PROJECT_ROOT, "video")
        mesh_dir = os.path.join(PROJECT_ROOT, "mesh")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        # Save uploaded file
        base_name = os.path.splitext(image.filename or f"upload_{work_id}.jpg")[0]
        safe_name = f"{base_name}_{work_id}.jpg"
        image_path = os.path.join(image_dir, safe_name)
        content = await image.read()
        with open(image_path, "wb") as f:
            f.write(content)

        # Ensure weights
        config_path = os.path.join(PROJECT_ROOT, "argument.yml")
        ensure_weights(config_path)

        # Create job entry
        out_key = os.path.splitext(os.path.basename(image_path))[0]
        jobs[work_id] = {"done": False, "result": None, "stdout": None, "stderr": None, "returncode": None}
        t = threading.Thread(
            target=_run_job,
            args=(work_id, out_key, encoder, bool(fast), longer_side, duration, fps, bool(loop), crop_top, crop_bottom, crop_left, crop_right, speed, bool(debug)),
            daemon=True,
        )
        t.start()
        return {"job_id": work_id, "key": out_key}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/progress/{job_id}")
def progress(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    progress_file = os.path.join(PROJECT_ROOT, f".progress_{job_id}.json")
    percent = 0
    message = "Queued"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                percent = int(data.get('percent', 0))
                message = data.get('message', message)
        except Exception:
            pass
    # stream any debug assets produced so far
    debug_assets = []
    dbg_dir = os.path.join(static_dir, f"dbg_{job_id}")
    try:
        if os.path.isdir(dbg_dir):
            for f in sorted(os.listdir(dbg_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    debug_assets.append(f"/static/dbg_{job_id}/{f}")
    except Exception:
        pass
    return {"done": jobs[job_id]["done"], "percent": percent, "message": message, "debug_assets": debug_assets}


@app.get("/api/result/{job_id}")
def result(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    job = jobs[job_id]
    if not job["done"]:
        return {"done": False}
    if job["returncode"] != 0:
        return JSONResponse(status_code=500, content={"error": "Generation failed", "stderr": job["stderr"], "stdout": job["stdout"]})
    return {"done": True, **(job["result"] or {})}


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("webapp_server:app", host="0.0.0.0", port=8008, reload=False)


