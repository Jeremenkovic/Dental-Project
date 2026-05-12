# Dental 3D — Proof of Concept

Intraoral video capture → cloud reconstruction → interactive 3D viewer.

Built against the architecture spec (v1.0, 12 May 2026). This POC covers the full end-to-end data flow: capture client → upload → job queue → reconstruction pipeline → OBJ mesh → Three.js viewer.

---

## What this demonstrates

| Capability | Status |
|---|---|
| WebRTC camera capture with live focus / motion quality bars | ✅ Running |
| On-device keyframe selection (Laplacian variance + greedy farthest-point sampling) | ✅ Running |
| Arch coverage heatmap overlay | ✅ Running |
| Chunked upload → FastAPI backend | ✅ Running |
| Celery job queue (Redis) with per-stage progress | ✅ Running |
| Specular suppression pre-processing | ✅ Running |
| Dense geometry: **MASt3R** foundation model (GPU path) | ✅ Code complete — needs GPU worker |
| Dense geometry: **pycolmap** SfM (CPU fallback) | ✅ Code complete — needs `pycolmap` install |
| Dense geometry: SIFT pairwise SfM (last-resort fallback) | ✅ Runs on any machine |
| **3D Gaussian Splatting** training via `train.py` subprocess | ✅ Code complete — needs CUDA + 3DGS install |
| Poisson surface reconstruction via Open3D (CPU fallback) | ✅ Runs on any machine |
| COLMAP binary format export (MASt3R → 3DGS handoff) | ✅ Running |
| Quality scorecard (coverage %, photometric residual) | ✅ Running |
| OBJ + MTL mesh download | ✅ Running |
| Three.js interactive 3D viewer (orbit, zoom, pan) | ✅ Running |
| **Demo mode** — parametric synthetic dental arch, no hardware needed | ✅ Running |
| Electron desktop app wrapper | ✅ Code complete — run with `npm run electron:dev` |

---

## Architecture

```
┌─────────────────────────────────┐
│  Capture Client                 │
│  React + Vite (web)             │
│  Electron wrapper (desktop)     │
│                                 │
│  • WebRTC camera capture        │
│  • Live quality indicators      │
│  • Arch coverage heatmap        │
│  • Keyframe selection           │
│  • Upload + progress polling    │
│  • Three.js 3D viewer           │
└──────────────┬──────────────────┘
               │ HTTPS multipart upload
               ▼
┌─────────────────────────────────┐
│  Backend (FastAPI)              │
│  port 8000                      │
│                                 │
│  POST /scans          → queue   │
│  POST /scans/demo     → demo    │
│  GET  /scans/{id}     → status  │
│  GET  /scans/{id}/result → OBJ  │
└──────────────┬──────────────────┘
               │ Celery task (Redis)
               ▼
┌─────────────────────────────────┐
│  Reconstruction Worker          │
│  (GPU machine in production)    │
│                                 │
│  Stage 1  Preprocessing         │
│           specular suppression  │
│           frame undistortion    │
│                                 │
│  Stage 2  Dense geometry        │
│           MASt3R  ← primary     │
│           pycolmap ← fallback   │
│           SIFT SfM ← last resort│
│                                 │
│  Stage 3  3D Gaussian Splatting │
│           train.py subprocess   │
│           point cloud ← fallback│
│                                 │
│  Stage 4  Mesh extraction       │
│           Poisson (Open3D)      │
│           dental-specific clean │
│                                 │
│  Stage 5  Quality validation    │
│           coverage % / residual │
└─────────────────────────────────┘
```

---

## Quick start — Demo mode (no GPU, no camera)

Demo mode generates a parametric upper dental arch and runs through all pipeline stages. No hardware or GPU required. Use this to see the full flow.

### Prerequisites

- Python 3.9+
- Node.js 20+
- Redis

**macOS:**
```bash
brew install redis node python@3.11
brew services start redis
```

**Ubuntu:**
```bash
sudo apt install redis-server nodejs python3 python3-pip
sudo systemctl start redis
```

### Install

```bash
git clone <repo>
cd dental-3d-poc

pip3 install -r backend/requirements.txt
pip3 install numpy   # minimum for demo mode (no Open3D needed)

cd client && npm install && cd ..
```

### Run

```bash
make demo
```

This starts:
- Backend at `http://localhost:8000`
- Worker (demo mode, no GPU)
- Frontend at `http://localhost:5173`

Open `http://localhost:5173`, click **Demo Mode**, watch the pipeline run, then interact with the 3D result.

---

## Full pipeline — Real reconstruction (GPU required)

### 1. Install worker dependencies

```bash
pip3 install -r worker/requirements.txt
```

### 2. Install MASt3R (dense geometry foundation model)

```bash
git clone --recursive https://github.com/naver/mast3r /opt/mast3r
pip3 install -e "/opt/mast3r[demo]"

# Download weights (~1.1 GB)
mkdir -p /opt/mast3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
     -P /opt/mast3r/checkpoints
```

### 3. Install 3D Gaussian Splatting

Requires CUDA 11.8+ and a compatible GPU (RTX 3080 or better for reasonable training times).

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive /opt/gaussian-splatting
pip3 install -r /opt/gaussian-splatting/requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
# Edit .env:
#   GAUSSIAN_SPLATTING_DIR=/opt/gaussian-splatting
#   DEMO_MODE=false
```

### 5. Run

```bash
make dev
```

Open `http://localhost:5173`, connect an intraoral UVC camera, capture a scan, and wait ~2 minutes for reconstruction.

---

## Docker (all-in-one)

```bash
# Demo mode
DEMO_MODE=true docker compose up --build

# Real pipeline (GPU)
GAUSSIAN_SPLATTING_DIR=/opt/gaussian-splatting docker compose up --build
```

Uncomment the `deploy.resources` block in `docker-compose.yml` to enable GPU passthrough.

---

## Electron desktop app

The Electron wrapper gives the capture client native OS integration (camera permissions dialog, file system access, no browser required).

```bash
cd client
npm run electron:dev      # development (connects to localhost:8000)
npm run electron:build    # production build (macOS .dmg / Windows .exe)
```

---

## Project structure

```
dental-3d-poc/
├── backend/                  FastAPI API server
│   ├── main.py               Endpoints: /scans, /scans/demo, /scans/{id}
│   ├── db.py                 SQLAlchemy models (SQLite → Postgres in prod)
│   └── celery_app.py         Task dispatch (broker only, no result backend)
│
├── worker/                   Celery reconstruction worker
│   ├── tasks.py              Task entry point, demo/real mode routing
│   ├── demo_mode.py          Parametric dental arch generator
│   └── pipeline/
│       ├── preprocess.py     Specular suppression, undistortion
│       ├── mastery.py        MASt3R → pycolmap → SIFT SfM fallback chain
│       ├── colmap_export.py  MASt3R output → COLMAP binary format for 3DGS
│       ├── gaussian_splatting.py  3DGS training subprocess + point cloud fallback
│       ├── mesh_extraction.py     Poisson reconstruction, dental cleanup, OBJ export
│       └── quality.py        Coverage %, density, photometric residual
│
├── client/                   React + Vite frontend
│   ├── electron/
│   │   ├── main.js           Electron main process
│   │   └── preload.js        Secure IPC bridge (contextIsolation)
│   └── src/
│       ├── App.jsx           Capture / Viewer tab layout
│       ├── components/
│       │   ├── CaptureView.jsx      Camera, quality bars, upload flow
│       │   ├── QualityBar.jsx       Live focus + steadiness indicators
│       │   ├── CoverageHeatmap.jsx  Arch segment coverage SVG
│       │   └── ModelViewer.jsx      Three.js OBJ viewer, orbit controls
│       └── lib/
│           ├── keyframeSelector.js  Laplacian + farthest-point sampling
│           └── uploader.js          XHR upload, status polling
│
├── docker-compose.yml
├── Makefile                  make demo / make dev / make docker / make test-demo
├── .env.example
└── scripts/setup.sh          One-shot dependency installer
```

---

## Key technical decisions

**Why MASt3R instead of COLMAP?**
COLMAP uses SIFT feature matching, which fails on dental imagery due to specular highlights, repetitive tooth geometry, and texture-poor enamel surfaces. MASt3R is a 2024 foundation model that directly regresses dense 3D structure without feature matching — it handles exactly these failure modes.

**Why 3D Gaussian Splatting?**
3DGS renders at hundreds of FPS in the viewer and captures view-dependent effects (highlights, enamel translucency) that polygon meshes miss. It also trains in 3–8 minutes on a single A10G GPU for the scene sizes involved.

**Why hybrid cloud architecture?**
The client runs on any clinic PC (€600–1,200 hardware). The GPU reconstruction worker scales on demand and costs ~€0.06–0.14 per scan at spot pricing on Azure NV6ads A10 v5. A self-hosted RTX 4090 breaks even above ~2,000 scans/month.

**Specular reflection**
The dominant failure mode in dental photogrammetry. The recommended fix is optical (cross-polarized LEDs + camera filter, ~€15–40 hardware change). The software fallback — highlight inpainting — is implemented in `preprocess.py` and runs on every frame before reconstruction.

---

## What's not in this POC

Per the spec scope (Section 9.1):

- Multi-clinic portal, user accounts, Stripe billing
- STL export (OBJ is exported; STL conversion is a one-line Open3D call)
- Real-time 3D preview during capture
- Multi-arch registration
- IEC 62304 / ISO 14971 regulatory documentation
- Sub-millimetre accuracy validation (requires a calibrated reference scanner)

---

## Contact

Nemanja Jeremenković — architecture & implementation  
Spec version 1.0, 12 May 2026
