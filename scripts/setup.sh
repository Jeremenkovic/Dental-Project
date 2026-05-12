#!/usr/bin/env bash
# setup.sh — one-shot setup for Dental 3D POC development environment
# Run from the repo root: bash scripts/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-/opt/dental3d}"

echo "==> Dental 3D POC setup"
echo "    Repo:   $REPO_ROOT"
echo "    Models: $MODELS_DIR"
echo ""

# ── Python deps ───────────────────────────────────────────────────────────────
echo "[1/5] Installing Python deps (backend)…"
pip install -r "$REPO_ROOT/backend/requirements.txt"

echo "[2/5] Installing Python deps (worker)…"
pip install -r "$REPO_ROOT/worker/requirements.txt"

# ── MASt3R (optional, GPU only) ───────────────────────────────────────────────
if [ "${INSTALL_MAST3R:-false}" = "true" ]; then
  echo "[3/5] Cloning MASt3R…"
  if [ ! -d "$MODELS_DIR/mast3r" ]; then
    git clone --recursive https://github.com/naver/mast3r "$MODELS_DIR/mast3r"
  fi
  pip install -e "$MODELS_DIR/mast3r[demo]"

  echo "      Downloading MASt3R weights (~1.1 GB)…"
  mkdir -p "$MODELS_DIR/mast3r/checkpoints"
  wget -q --show-progress -nc \
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    -P "$MODELS_DIR/mast3r/checkpoints"
else
  echo "[3/5] Skipping MASt3R (set INSTALL_MAST3R=true to install, GPU required)"
fi

# ── 3DGS (optional, GPU + CUDA only) ─────────────────────────────────────────
if [ "${INSTALL_3DGS:-false}" = "true" ]; then
  echo "[4/5] Cloning gaussian-splatting…"
  if [ ! -d "$MODELS_DIR/gaussian-splatting" ]; then
    git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive \
      "$MODELS_DIR/gaussian-splatting"
  fi
  pip install -r "$MODELS_DIR/gaussian-splatting/requirements.txt"
  export GAUSSIAN_SPLATTING_DIR="$MODELS_DIR/gaussian-splatting"
  echo "      Set GAUSSIAN_SPLATTING_DIR=$GAUSSIAN_SPLATTING_DIR in your .env"
else
  echo "[4/5] Skipping 3DGS (set INSTALL_3DGS=true to install, requires CUDA 11.8+)"
fi

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[5/5] Installing frontend deps…"
cd "$REPO_ROOT/client" && npm install
cd "$REPO_ROOT"

# ── .env ──────────────────────────────────────────────────────────────────────
if [ ! -f "$REPO_ROOT/.env" ]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  echo ""
  echo "  Created .env from .env.example — review it before running."
fi

echo ""
echo "==> Setup complete."
echo ""
echo "    Web (Docker):    docker compose up --build"
echo "    Dev (local):     make dev"
echo "    Demo mode:       DEMO_MODE=true make dev"
echo "    Electron:        cd client && npm run electron:dev"
