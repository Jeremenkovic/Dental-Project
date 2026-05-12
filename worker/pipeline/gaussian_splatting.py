"""
Stage 3: 3D Gaussian Splatting

Production path: calls gaussian-splatting/train.py (requires CUDA 11.8+).
Fallback: Open3D point cloud .ply (CPU-runnable, used for dev/demo).

Install 3DGS:
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive /opt/gaussian-splatting
  cd /opt/gaussian-splatting && pip install -r requirements.txt
  # Installs custom CUDA submodules automatically

Set env var:  GAUSSIAN_SPLATTING_DIR=/opt/gaussian-splatting
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mastery import GeometryResult
from .colmap_export import export_colmap_scene

log = logging.getLogger(__name__)

GAUSSIAN_SPLATTING_DIR = Path(os.getenv("GAUSSIAN_SPLATTING_DIR", "/opt/gaussian-splatting"))
GS_TRAIN_ITERATIONS = int(os.getenv("GS_TRAIN_ITERATIONS", "7000"))


@dataclass
class SplatScene:
    ply_path: Path
    point_count: int
    method: str   # "3dgs" | "pointcloud"


def train_or_proxy(
    geometry: GeometryResult,
    frames: list[tuple[str, np.ndarray]],
    output_dir: Path,
    progress_cb=None,
) -> SplatScene:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_py = GAUSSIAN_SPLATTING_DIR / "train.py"
    if train_py.exists():
        try:
            return _run_3dgs(geometry, frames, output_dir, progress_cb)
        except Exception as e:
            log.warning("3DGS training failed (%s), falling back to point cloud", e)
    else:
        log.info("3DGS not found at %s — using point cloud proxy", GAUSSIAN_SPLATTING_DIR)

    return _write_pointcloud_ply(geometry, output_dir, progress_cb)


# ──────────────────────────────────────────────────────────────────────────────
# Path A: real 3DGS training
# ──────────────────────────────────────────────────────────────────────────────

def _run_3dgs(
    geometry: GeometryResult,
    frames: list[tuple[str, np.ndarray]],
    output_dir: Path,
    progress_cb,
) -> SplatScene:
    scene_dir = output_dir / "colmap_scene"
    gs_output = output_dir / "gs_model"

    # Write COLMAP-format scene so train.py can read it
    export_colmap_scene(geometry, frames, scene_dir)

    if progress_cb:
        progress_cb(0.10)

    cmd = [
        "python",
        str(GAUSSIAN_SPLATTING_DIR / "train.py"),
        "-s", str(scene_dir),
        "--model_path", str(gs_output),
        "--iterations", str(GS_TRAIN_ITERATIONS),
        "--test_iterations", str(GS_TRAIN_ITERATIONS),
        "--save_iterations", str(GS_TRAIN_ITERATIONS),
        "--checkpoint_iterations", str(GS_TRAIN_ITERATIONS),
        # Dental-specific: smaller Gaussians for fine tooth geometry
        "--densify_until_iter", "5000",
        "--densification_interval", "100",
    ]

    log.info("Starting 3DGS training: %s iterations", GS_TRAIN_ITERATIONS)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(GAUSSIAN_SPLATTING_DIR),
    )

    for line in proc.stdout:
        line = line.strip()
        if line:
            log.debug("[3dgs] %s", line)
            # Parse progress from training log lines like "[7000/7000]"
            if "/" in line and "[" in line:
                try:
                    parts = line.split("[")[1].split("]")[0].split("/")
                    current, total = int(parts[0]), int(parts[1])
                    if progress_cb:
                        progress_cb(0.10 + (current / total) * 0.85)
                except (ValueError, IndexError):
                    pass

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    ply_path = gs_output / "point_cloud" / f"iteration_{GS_TRAIN_ITERATIONS}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"3DGS output not found at {ply_path}")

    if progress_cb:
        progress_cb(1.0)

    pts = np.asarray(geometry.points)
    log.info("3DGS training complete: %s", ply_path)
    return SplatScene(ply_path=ply_path, point_count=len(pts), method="3dgs")


# ──────────────────────────────────────────────────────────────────────────────
# Path B: dense point cloud .ply (CPU fallback / dev mode)
# ──────────────────────────────────────────────────────────────────────────────

def _write_pointcloud_ply(
    geometry: GeometryResult,
    output_dir: Path,
    progress_cb,
) -> SplatScene:
    ply_path = output_dir / "splat.ply"
    points = geometry.points
    colors = np.clip(geometry.colors, 0, 1)
    colors_u8 = (colors * 255).astype(np.uint8)

    if progress_cb:
        progress_cb(0.5)

    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        data = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data["x"] = points[:, 0].astype(np.float32)
        data["y"] = points[:, 1].astype(np.float32)
        data["z"] = points[:, 2].astype(np.float32)
        data["red"] = colors_u8[:, 0]
        data["green"] = colors_u8[:, 1]
        data["blue"] = colors_u8[:, 2]
        f.write(data.tobytes())

    if progress_cb:
        progress_cb(1.0)

    return SplatScene(ply_path=ply_path, point_count=n, method="pointcloud")
