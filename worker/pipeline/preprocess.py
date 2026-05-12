"""
Stage 1: frame undistortion and optional learned specular removal.
In POC we apply basic specular suppression via highlight desaturation.
Replace with a learned CNN (e.g. Specular-Diffuse network) for production.
"""
import cv2
import numpy as np
from pathlib import Path


def load_frames(frames_dir: Path) -> list[tuple[str, np.ndarray]]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    return [(str(p), cv2.imread(str(p))) for p in paths if cv2.imread(str(p)) is not None]


def suppress_specular(img: np.ndarray, threshold: int = 230) -> np.ndarray:
    """
    POC-grade specular suppression: inpaint bright highlights.
    Production: replace with cross-polarization optics (Section 4.1.1)
    or a learned Specular-Diffuse CNN (Section 4.1.2).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel)
    return cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def undistort(img: np.ndarray, K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    if K is None or dist is None:
        return img
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
    return cv2.undistort(img, K, dist, None, new_K)


def preprocess_frames(
    frames_dir: Path,
    camera_K: np.ndarray | None = None,
    camera_dist: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    frames = load_frames(frames_dir)
    result = []
    for path, img in frames:
        img = suppress_specular(img)
        img = undistort(img, camera_K, camera_dist)
        result.append((path, img))
    return result
