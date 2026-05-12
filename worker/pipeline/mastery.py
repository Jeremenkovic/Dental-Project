"""
Stage 2: Dense geometry estimation.

Priority order:
  1. MASt3R  — foundation model (GPU required, ~30-90 s for full arch)
  2. pycolmap — good CPU SfM, handles dental imagery better than hand-rolled SIFT
  3. SIFT pairwise — last-resort fallback, drift-prone on dental scenes

Install MASt3R (GPU worker only):
  git clone --recursive https://github.com/naver/mast3r /opt/mast3r
  pip install -e "/opt/mast3r[demo]"
  wget -P /opt/mast3r/checkpoints \
    https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

Install pycolmap (CPU fallback):
  pip install pycolmap
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GeometryResult:
    points: np.ndarray           # (N, 3) world-space XYZ
    colors: np.ndarray           # (N, 3) RGB float 0-1
    camera_poses: list[np.ndarray]  # list of (4, 4) extrinsic matrices (cam-to-world)
    method: str                  # which backend ran


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def estimate_geometry(
    frames: list[tuple[str, np.ndarray]],
    K: np.ndarray | None = None,
    imu_poses: list[np.ndarray] | None = None,
    progress_cb=None,
) -> GeometryResult:
    if not frames:
        raise ValueError("No frames provided")

    paths = [p for p, _ in frames]

    try:
        return _run_mast3r(paths, imu_poses, progress_cb)
    except ImportError:
        log.warning("MASt3R not available, falling back to pycolmap")
    except Exception as e:
        log.warning("MASt3R failed (%s), falling back to pycolmap", e)

    try:
        return _run_pycolmap(paths, frames, progress_cb)
    except ImportError:
        log.warning("pycolmap not available, falling back to SIFT SfM")
    except Exception as e:
        log.warning("pycolmap failed (%s), falling back to SIFT SfM", e)

    return _run_sift_sfm(frames, K, progress_cb)


# ──────────────────────────────────────────────────────────────────────────────
# Backend 1: MASt3R
# ──────────────────────────────────────────────────────────────────────────────

def _run_mast3r(
    image_paths: list[str],
    imu_poses: list[np.ndarray] | None,
    progress_cb,
) -> GeometryResult:
    import torch
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("MASt3R running on %s with %d images", device, len(image_paths))

    # Subsample to ≤120 images — MASt3R scales O(N²) in pair count
    paths = _subsample(image_paths, max_count=120)

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    if progress_cb:
        progress_cb(0.05)

    images = load_images(paths, size=512)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    if progress_cb:
        progress_cb(0.15)

    output = inference(pairs, model, device, batch_size=1, verbose=False)

    if progress_cb:
        progress_cb(0.50)

    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False,
    )

    # Inject IMU priors as soft pose constraints if available
    if imu_poses and len(imu_poses) == len(paths):
        _inject_imu_priors(scene, imu_poses)

    loss = scene.compute_global_alignment(
        init="mst", niter=300, schedule="cosine", lr=0.01, verbose=False
    )
    log.info("MASt3R alignment loss: %.4f", float(loss))

    if progress_cb:
        progress_cb(0.90)

    pts3d_list = scene.get_pts3d()
    masks = scene.get_masks()
    imgs = scene.imgs
    poses_t = scene.get_im_poses()

    all_pts, all_colors = [], []
    for pts, img, mask in zip(pts3d_list, imgs, masks):
        m = to_numpy(mask).astype(bool)
        all_pts.append(to_numpy(pts)[m])
        all_colors.append(to_numpy(img)[m])

    points = np.concatenate(all_pts, axis=0)
    colors = np.clip(np.concatenate(all_colors, axis=0), 0, 1)
    poses = [to_numpy(p) for p in poses_t]

    if progress_cb:
        progress_cb(1.0)

    return GeometryResult(points=points, colors=colors, camera_poses=poses, method="mast3r")


def _inject_imu_priors(scene, imu_poses: list[np.ndarray]):
    """
    Apply IMU rotation priors as initial values to the scene optimizer.
    This dramatically reduces banana-effect drift along the dental arch.
    Only sets the initial value — the optimizer may deviate if visual evidence contradicts.
    """
    try:
        import torch
        for i, pose in enumerate(imu_poses):
            if i >= len(scene.im_poses):
                break
            R = torch.tensor(pose[:3, :3], dtype=torch.float32, device=scene.im_poses[i].device)
            # Set rotation component of the pose (leave translation for optimizer)
            with torch.no_grad():
                scene.im_poses[i][:3, :3] = R
    except Exception as e:
        log.warning("IMU prior injection failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Backend 2: pycolmap
# ──────────────────────────────────────────────────────────────────────────────

def _run_pycolmap(
    image_paths: list[str],
    frames: list[tuple[str, np.ndarray]],
    progress_cb,
) -> GeometryResult:
    import pycolmap

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img_dir = tmp / "images"
        img_dir.mkdir()
        sparse_dir = tmp / "sparse"
        sparse_dir.mkdir()
        db_path = tmp / "database.db"

        # Copy images
        for src, _ in frames[:120]:
            shutil.copy(src, img_dir / Path(src).name)

        if progress_cb:
            progress_cb(0.10)

        pycolmap.extract_features(db_path, img_dir)

        if progress_cb:
            progress_cb(0.30)

        pycolmap.match_exhaustive(db_path)

        if progress_cb:
            progress_cb(0.55)

        maps = pycolmap.incremental_mapping(db_path, img_dir, sparse_dir)
        if not maps:
            raise RuntimeError("pycolmap produced no reconstruction")

        rec = maps[0]
        if progress_cb:
            progress_cb(0.85)

        pts = np.array([p.xyz for p in rec.points3D.values()])
        colors = np.array([p.color / 255.0 for p in rec.points3D.values()])

        poses = []
        for img in rec.images.values():
            T = np.eye(4)
            T[:3, :3] = img.rotmat()
            T[:3, 3] = img.tvec
            poses.append(T)

        if progress_cb:
            progress_cb(1.0)

        return GeometryResult(points=pts, colors=colors, camera_poses=poses, method="pycolmap")


# ──────────────────────────────────────────────────────────────────────────────
# Backend 3: SIFT pairwise SfM (last resort)
# ──────────────────────────────────────────────────────────────────────────────

def _run_sift_sfm(
    frames: list[tuple[str, np.ndarray]],
    K: np.ndarray | None,
    progress_cb,
) -> GeometryResult:
    log.warning("Using SIFT SfM fallback — quality will be limited on dental imagery")

    ref_img = frames[0][1]
    if K is None:
        K = _build_default_K(ref_img)

    all_pts3d: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    poses: list[np.ndarray] = [np.eye(4)]
    R_accum, t_accum = np.eye(3), np.zeros((3, 1))

    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    for i in range(len(frames) - 1):
        if progress_cb:
            progress_cb(i / (len(frames) - 1))

        _, img1 = frames[i]
        _, img2 = frames[i + 1]

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 8:
            poses.append(poses[-1].copy())
            continue

        matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:400]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append(poses[-1].copy())
            continue

        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        pts1_in = pts1[mask.ravel() == 255]
        pts2_in = pts2[mask.ravel() == 255]
        if len(pts1_in) < 4:
            poses.append(poses[-1].copy())
            continue

        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        valid = (pts3d[:, 2] > 0) & (np.linalg.norm(pts3d, axis=1) < 500)
        pts3d = pts3d[valid]

        for pt2d, pt3d in zip(pts1_in[valid], pts3d):
            x = int(np.clip(pt2d[0], 0, img1.shape[1] - 1))
            y = int(np.clip(pt2d[1], 0, img1.shape[0] - 1))
            all_colors.append(img1[y, x, ::-1] / 255.0)
        all_pts3d.extend(pts3d)

        R_accum = R @ R_accum
        t_accum = R @ t_accum + t
        pose = np.eye(4)
        pose[:3, :3] = R_accum
        pose[:3, 3] = t_accum.ravel()
        poses.append(pose)

    if not all_pts3d:
        raise RuntimeError("SIFT SfM produced no points — capture quality too low")

    return GeometryResult(
        points=np.array(all_pts3d),
        colors=np.array(all_colors),
        camera_poses=poses,
        method="sift-sfm",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_default_K(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    f = max(h, w) * 1.2
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _subsample(paths: list[str], max_count: int) -> list[str]:
    if len(paths) <= max_count:
        return paths
    idx = np.round(np.linspace(0, len(paths) - 1, max_count)).astype(int)
    return [paths[i] for i in idx]
