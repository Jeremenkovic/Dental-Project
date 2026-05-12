"""
Demo mode: generate a parametric upper dental arch model.

Used when DEMO_MODE=true in env or when called with --demo flag.
Bypasses capture and reconstruction entirely, producing a realistic-looking
OBJ + quality report so the full viewer / export / portal flow can be demonstrated
without dental hardware or a GPU.

The arch is anatomically proportioned (ISO dental numbering, permanent dentition).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Tooth catalogue  (width mm, height mm, depth mm)
# Upper arch, right→left: 18 17 16 15 14 13 12 11 | 21 22 23 24 25 26 27 28
# POC: 14 teeth (no wisdom teeth)
# ──────────────────────────────────────────────────────────────────────────────

UPPER_TEETH = [
    # (name,        w,    h,    d)
    ("UR2_molar",   10.5, 7.5,  11.0),
    ("UR1_molar",   10.5, 7.5,  11.5),
    ("UR2_premol",   7.0, 8.5,   8.5),
    ("UR1_premol",   7.0, 8.5,   9.0),
    ("UR_canine",    8.0, 11.0,  8.0),
    ("UR_lat_inc",   6.5, 9.5,   6.5),
    ("UR_cen_inc",   8.5, 10.5,  6.5),
    ("UL_cen_inc",   8.5, 10.5,  6.5),
    ("UL_lat_inc",   6.5, 9.5,   6.5),
    ("UL_canine",    8.0, 11.0,  8.0),
    ("UL1_premol",   7.0, 8.5,   9.0),
    ("UL2_premol",   7.0, 8.5,   8.5),
    ("UL1_molar",   10.5, 7.5,  11.5),
    ("UL2_molar",   10.5, 7.5,  11.0),
]

# Arch ellipse parameters (mm)
ARCH_A = 30.0   # half-width
ARCH_B = 24.0   # half-depth
GINGIVAL_Y = -3.0  # gum plane relative to tooth bases


def generate_arch(output_dir: Path) -> Path:
    """Write mesh.obj + mesh.mtl for a synthetic upper dental arch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / "mesh.obj"
    mtl_path = output_dir / "mesh.mtl"

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    vert_offset = 0

    n = len(UPPER_TEETH)
    angles = np.linspace(0, math.pi, n)

    for i, (name, tw, th, td) in enumerate(UPPER_TEETH):
        t = angles[i]
        cx = ARCH_A * math.cos(math.pi - t)
        cy = ARCH_B * math.sin(math.pi - t)

        # Tooth orientation: tangent to arch ellipse
        dx = -ARCH_A * math.sin(math.pi - t)
        dy = ARCH_B * math.cos(math.pi - t)
        angle = math.atan2(dy, dx)

        verts, faces, colors = _tooth_mesh(cx, 0.0, cy, tw, th, td, angle, name)
        all_faces.append(faces + vert_offset)
        all_verts.append(verts)
        all_colors.append(colors)
        vert_offset += len(verts)

    # Gingiva (pink arch band)
    g_verts, g_faces, g_colors = _gingiva_mesh(angles)
    all_faces.append(g_faces + vert_offset)
    all_verts.append(g_verts)
    all_colors.append(g_colors)

    verts = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)
    colors = np.concatenate(all_colors)

    _write_obj(obj_path, mtl_path, verts, faces, colors)
    return obj_path


# ──────────────────────────────────────────────────────────────────────────────
# Geometry builders
# ──────────────────────────────────────────────────────────────────────────────

def _tooth_mesh(
    cx: float, cy: float, cz: float,
    tw: float, th: float, td: float,
    yaw: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A single tooth crown: tapered box (wider at gingival margin, narrower at occlusal).
    4 corners per level, segs+1 levels, capped top and bottom.
    """
    taper = 0.75  # occlusal face is taper×base size
    segs = 4      # height subdivisions
    n_pts = 4     # corners per ring (from _rect_pts)
    levels = segs + 1

    all_v = []
    for li in range(levels):
        frac = li / segs
        alpha = 1.0 - frac * (1.0 - taper)
        y = frac * th
        pts = _rect_pts(tw * alpha, td * alpha)
        for p in pts:
            all_v.append([p[0], y, p[1]])

    verts = np.array(all_v, dtype=np.float32)  # shape: (levels * n_pts, 3)

    faces = []

    # Side bands
    for li in range(levels - 1):
        for pi in range(n_pts):
            a = li * n_pts + pi
            b = li * n_pts + (pi + 1) % n_pts
            c = (li + 1) * n_pts + (pi + 1) % n_pts
            d = (li + 1) * n_pts + pi
            faces.append([a, b, c])
            faces.append([a, c, d])

    # Top cap (occlusal surface)
    top_start = (levels - 1) * n_pts
    top_ring = list(range(top_start, top_start + n_pts))
    top_center_idx = len(verts)
    top_center = np.mean(verts[top_ring], axis=0)
    verts = np.vstack([verts, top_center[None]])
    for pi in range(n_pts):
        faces.append([top_ring[pi], top_ring[(pi + 1) % n_pts], top_center_idx])

    # Bottom cap (cervical margin)
    bot_ring = list(range(0, n_pts))
    bot_center_idx = len(verts)
    bot_center = np.mean(verts[bot_ring], axis=0)
    verts = np.vstack([verts, bot_center[None]])
    for pi in range(n_pts):
        faces.append([bot_ring[(pi + 1) % n_pts], bot_ring[pi], bot_center_idx])

    faces = np.array(faces, dtype=np.int32)

    # Rotate in XZ plane by yaw, then translate to arch position
    R = _rot_y(yaw)
    verts[:, [0, 2]] = verts[:, [0, 2]] @ R.T
    verts[:, 0] += cx
    verts[:, 2] += cz

    # Enamel ivory colour with slight per-tooth variation
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    base_col = np.array([0.94, 0.91, 0.86]) + rng.uniform(-0.02, 0.02, 3)
    colors = np.tile(np.clip(base_col, 0, 1), (len(verts), 1)).astype(np.float32)

    return verts, faces, colors


def _rect_pts(w: float, d: float) -> np.ndarray:
    """4 rectangle corners CCW viewed from above."""
    hw, hd = w / 2, d / 2
    return np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]], dtype=np.float32)


def _gingiva_mesh(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple flat gingival (gum) band connecting tooth bases."""
    n = len(angles)
    outer_r, inner_r = 1.12, 0.88
    verts, faces = [], []
    for i, t in enumerate(angles):
        for r in [inner_r, outer_r]:
            x = ARCH_A * r * math.cos(math.pi - t)
            z = ARCH_B * r * math.sin(math.pi - t)
            verts.append([x, GINGIVAL_Y, z])

    verts = np.array(verts, dtype=np.float32)
    for i in range(n - 1):
        a, b = i * 2, i * 2 + 1
        c, d = (i + 1) * 2 + 1, (i + 1) * 2
        faces.append([a, b, c])
        faces.append([a, c, d])

    faces = np.array(faces, dtype=np.int32)
    gum_pink = np.tile([0.85, 0.50, 0.55], (len(verts), 1)).astype(np.float32)
    return verts, faces, gum_pink


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s], [-s, c]], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# OBJ writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_obj(obj_path: Path, mtl_path: Path, verts, faces, colors):
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.name}\nusemtl dental\n")
        for v, c in zip(verts, colors):
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    with open(mtl_path, "w") as f:
        f.write(
            "newmtl dental\n"
            "Ka 0.94 0.90 0.84\n"
            "Kd 0.94 0.90 0.84\n"
            "Ks 0.40 0.40 0.40\n"
            "Ns 60\n"
        )


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/demo_arch")
    path = generate_arch(out)
    print(f"Demo arch written to {path}")
