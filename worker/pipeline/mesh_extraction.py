"""
Stage 4: Mesh extraction — Poisson surface reconstruction with dental-specific cleanup.

SuGaR (production, requires trained 3DGS) is the preferred path when 3DGS ran.
Poisson via Open3D is the fallback and the POC path — it works on the dense point cloud.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


@dataclass
class MeshResult:
    obj_path: Path
    mtl_path: Path
    vertex_count: int
    face_count: int


def extract_mesh(
    splat_ply: Path,
    output_dir: Path,
    progress_cb=None,
) -> MeshResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(splat_ply))
    log.info("Loaded point cloud: %d points", len(pcd.points))

    if progress_cb:
        progress_cb(0.05)

    pcd = _remove_background(pcd)
    pcd = _denoise(pcd)
    log.info("After cleanup: %d points", len(pcd.points))

    if progress_cb:
        progress_cb(0.25)

    if len(pcd.points) < 50:
        raise ValueError("Too few points after cleanup for mesh reconstruction")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(10)

    if progress_cb:
        progress_cb(0.40)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    densities = np.asarray(densities)
    # Remove low-density outer hull (artefacts on scan boundary)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.08))

    if progress_cb:
        progress_cb(0.65)

    mesh = _dental_cleanup(mesh)

    if progress_cb:
        progress_cb(0.85)

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    vc = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else _ivory(len(verts))

    obj_path = output_dir / "mesh.obj"
    mtl_path = output_dir / "mesh.mtl"
    _write_obj(obj_path, mtl_path, verts, tris, vc)

    if progress_cb:
        progress_cb(1.0)

    log.info("Mesh: %d vertices, %d faces", len(verts), len(tris))
    return MeshResult(obj_path=obj_path, mtl_path=mtl_path, vertex_count=len(verts), face_count=len(tris))


# ──────────────────────────────────────────────────────────────────────────────
# Dental-specific cleanup
# ──────────────────────────────────────────────────────────────────────────────

def _remove_background(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Keep the 85th-percentile central cluster — removes cheek tissue, tongue,
    and reflection artefacts that sit outside the arch bounding volume.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    centroid = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    keep = dists < np.percentile(dists, 85)
    return pcd.select_by_index(np.where(keep)[0])


def _denoise(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if len(pcd.points) < 20:
        return pcd
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)
    return pcd


def _dental_cleanup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # Fill small holes left by occlusal specularity
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=5, lambda_filter=0.5)

    # Remove disconnected fragments (keep only the largest connected component)
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) > 1:
        largest = cluster_n_triangles.argmax()
        remove_mask = triangle_clusters != largest
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh


def _ivory(n: int) -> np.ndarray:
    return np.tile([0.94, 0.90, 0.84], (n, 1))


# ──────────────────────────────────────────────────────────────────────────────
# OBJ export
# ──────────────────────────────────────────────────────────────────────────────

def _write_obj(
    obj_path: Path,
    mtl_path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
):
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl dental\n")
        for v, c in zip(verts, np.clip(colors, 0, 1)):
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    with open(mtl_path, "w") as f:
        f.write(
            "newmtl dental\n"
            "Ka 0.94 0.90 0.84\n"
            "Kd 0.94 0.90 0.84\n"
            "Ks 0.40 0.40 0.40\n"
            "Ns 60\n"
            "d 1.0\n"
        )
