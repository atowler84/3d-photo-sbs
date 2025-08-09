# 🌌 Spatial-Photo  
**3D Photo from a Single Image**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/github/license/fake-oskars/Spatial-Photo)](LICENSE)

Turn a single image into a short 3D motion video using **state-of-the-art monocular depth estimation** + lightweight mesh rendering.  
Includes a simple **Web UI** and minimal **HTTP API**.

---

## ✨ Features
- 🖤 Depth via **Depth-Anything v2** (Small / Base / Large)
- ⚡ Fast, no-upscale pipeline with optional **Fast Mode**
- 🎥 Multiple motions: dolly-zoom-in, zoom-in, circle, swing
- 👀 Live depth preview (auto-hides when done)
- 💻 Lightweight Web UI + REST API

---

## 🚀 Quickstart

### Requirements
- 🐍 Python **3.10+** (tested on 3.13)
- 🍎 macOS / 🐧 Linux recommended
- Pretrained checkpoints in `checkpoints/`:
  - `edge-model.pth`
  - `depth-model.pth`
  - `color-model.pth`

### Install & Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python webapp_server.py
# open http://localhost:8008
```

---

## 🖥 Using the Web UI
1. Open `http://localhost:8008`
2. Drop an image or choose a file
3. Adjust:
   - Encoder, Resolution, Fast Mode, Duration, Speed, Loop
4. Click **Generate**
5. Watch progress + optional debug previews → videos render below

> **Tip:** Enable **Debug** to see intermediate images; the depth preview hides automatically after completion.

---

## 📡 HTTP API

<details>
<summary><b>Generate 3D</b></summary>

```http
POST /api/generate  (multipart/form-data)
Fields:
  image: file
  encoder: vits | vitb | vitl
  longer_side: px
  fast: 1
  duration: sec
  speed: float
  loop: 1
  debug: 1
Response:
  { "job_id": "xxxxxx", "key": "basename" }
```
</details>

<details>
<summary><b>Progress</b></summary>

```http
GET /api/progress/{job_id}
Response:
  { done: bool, percent: int, message: str, debug_assets: string[] }
```
</details>

<details>
<summary><b>Result</b></summary>

```http
GET /api/result/{job_id}
Response:
  { done: true, key, videos: ["/static/....mp4"], mesh: null, debug_assets: [] }
```
</details>

<details>
<summary><b>Health Check</b></summary>

```http
GET /api/health
Response:
  { "status": "ok" }
```
</details>

---

## 📂 Outputs & Folders
| Type                | Location                        |
|---------------------|---------------------------------|
| Input images        | `image/`                        |
| Depth maps          | `depth/`                        |
| Videos              | `video/` → served via `/static` |
| Mesh cache          | `mesh/`                         |
| Debug previews      | `static/dbg_<jobid>/`           |

---

## ⚙ Configuration
Main config: `argument.yml`  
- `use_depth_anything_v2`: `True` (default)  
- `depth_anything_encoder`: `vits | vitb | vitl`  
- `longer_side_len`: max processing resolution (no upscaling)  
- `fps`, `num_frames`, trajectories/postfixes  

**Performance Tips**
- Enable **Fast Mode**
- Lower resolution (e.g., `640`)
- Shorter duration or higher speed

---

## 🛠 Troubleshooting
- **Stuck at “Building mesh”** → This step is heaviest; use **Fast Mode** and/or lower **Resolution**  
- **Artifacts / memory spikes** → Lower `longer_side_len` in `argument.yml`, or reduce `sparse_iter` / `largest_size`  
- **Missing models** → Ensure all checkpoints exist in `checkpoints/`

---

## 📜 License & Acknowledgments
- See [`LICENSE`](LICENSE)
- Built on ideas from prior depth & inpainting research
- Uses **Depth-Anything v2** via Hugging Face
- Edits: UI hides depth map after completion; **MiDaS** is optional and disabled by default
