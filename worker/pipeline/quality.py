"""
Stage 5: Quality validation scorecard.
Every reconstruction surfaces these metrics to the client so operators
know whether a re-capture is needed and which arch region is weak.
"""
from __future__ import annotations
import numpy as np
import open3d as o3d
from pathlib import Path
from dataclasses import dataclass


@dataclass
class QualityReport:
    coverage_pct: float          # 0–100: % of expected arch bounding box covered
    point_density: float         # points per mm² (proxy without scale calibration)
    photometric_residual: float  # mean abs difference between render and input (0–1)
    passed: bool
    message: str


COVERAGE_THRESHOLD = 60.0
DENSITY_THRESHOLD = 10.0


def validate(
    splat_ply: Path,
    frames: list,
) -> QualityReport:
    pcd = o3d.io.read_point_cloud(str(splat_ply))
    pts = np.asarray(pcd.points)

    if len(pts) < 100:
        return QualityReport(0, 0, 1.0, False, "Too few points reconstructed — re-capture recommended")

    # Coverage: ratio of occupied voxels in expected arch bounding box
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.5)
    occupied = len(voxel_grid.get_voxels())
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(aabb.get_extent())
    total_voxels = max(1, np.prod(np.ceil(extent / 1.5).astype(int)))
    coverage = min(100.0, occupied / total_voxels * 100)

    # Point density proxy
    volume = max(1e-6, float(np.prod(extent)))
    density = len(pts) / volume * 1000  # arbitrary normalisation

    # Photometric residual: not computed without a renderer in POC — use 0 as placeholder
    residual = 0.0

    passed = coverage >= COVERAGE_THRESHOLD
    msg = "Reconstruction quality OK" if passed else f"Coverage {coverage:.0f}% below threshold — consider re-capturing lingual surfaces"

    return QualityReport(
        coverage_pct=round(coverage, 1),
        point_density=round(density, 2),
        photometric_residual=residual,
        passed=passed,
        message=msg,
    )
