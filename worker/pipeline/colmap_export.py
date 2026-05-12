"""
Convert MASt3R/SfM output to COLMAP binary format so 3DGS train.py can consume it.

COLMAP binary spec:
  cameras.bin  — camera intrinsics
  images.bin   — camera extrinsics + 2D point observations
  points3D.bin — 3D points with colour and track info
"""
from __future__ import annotations

import shutil
import struct
from pathlib import Path

import cv2
import numpy as np

from .mastery import GeometryResult


def export_colmap_scene(
    geometry: GeometryResult,
    frames: list[tuple[str, np.ndarray]],
    scene_dir: Path,
) -> Path:
    """
    Write a COLMAP-format scene directory consumable by 3DGS train.py.

    Returns the scene_dir path.
    """
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Copy keyframes into images/
    image_names: list[str] = []
    for idx, (src_path, _) in enumerate(frames):
        name = f"frame_{idx:04d}.jpg"
        shutil.copy(src_path, images_dir / name)
        image_names.append(name)

    h, w = frames[0][1].shape[:2]
    f = max(h, w) * 1.2

    _write_cameras_bin(sparse_dir / "cameras.bin", w, h, f)
    _write_images_bin(sparse_dir / "images.bin", geometry.camera_poses, image_names)
    _write_points3d_bin(sparse_dir / "points3D.bin", geometry.points, geometry.colors)

    return scene_dir


# ──────────────────────────────────────────────────────────────────────────────
# Binary writers
# ──────────────────────────────────────────────────────────────────────────────

def _write_cameras_bin(path: Path, w: int, h: int, f: float):
    """Single shared PINHOLE camera for all images."""
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", 1))       # num_cameras
        fp.write(struct.pack("<i", 1))       # camera_id = 1
        fp.write(struct.pack("<i", 1))       # model_id = PINHOLE
        fp.write(struct.pack("<QQ", w, h))   # width, height
        # params: fx fy cx cy
        for v in [f, f, w / 2.0, h / 2.0]:
            fp.write(struct.pack("<d", v))


def _write_images_bin(
    path: Path,
    poses: list[np.ndarray],
    image_names: list[str],
):
    """One entry per image with extrinsic pose."""
    n = min(len(poses), len(image_names))
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", n))
        for img_id, (pose, name) in enumerate(zip(poses[:n], image_names[:n]), start=1):
            R = pose[:3, :3]
            t = pose[:3, 3]
            qvec = _rotmat_to_qvec(R)

            fp.write(struct.pack("<i", img_id))
            for q in qvec:
                fp.write(struct.pack("<d", q))
            for tv in t:
                fp.write(struct.pack("<d", tv))
            fp.write(struct.pack("<i", 1))   # camera_id = 1 (shared)
            for c in name.encode("ascii"):
                fp.write(struct.pack("<B", c))
            fp.write(struct.pack("<B", 0))   # null terminator
            fp.write(struct.pack("<Q", 0))   # no 2D point observations


def _write_points3d_bin(path: Path, points: np.ndarray, colors: np.ndarray):
    """3D point cloud with colour; no track info required by 3DGS."""
    n = len(points)
    colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", n))
        for pid, (pt, col) in enumerate(zip(points, colors_u8), start=1):
            fp.write(struct.pack("<Q", pid))
            for v in pt:
                fp.write(struct.pack("<d", float(v)))
            for c in col[:3]:
                fp.write(struct.pack("<B", int(c)))
            fp.write(struct.pack("<d", 0.0))  # reprojection error
            fp.write(struct.pack("<Q", 0))    # track length 0


def _rotmat_to_qvec(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3×3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (w, x, y, z)
