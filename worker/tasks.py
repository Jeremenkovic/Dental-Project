"""
Celery reconstruction task.

Env flags:
  DEMO_MODE=true   — skip reconstruction, generate synthetic arch (no GPU needed)
  GAUSSIAN_SPLATTING_DIR — path to gaussian-splatting repo
  DATABASE_URL     — must be sync sqlite:// (not sqlite+aiosqlite://)
"""
import logging
import os
import sys
from pathlib import Path

# Ensure the worker package directory is always importable regardless of how
# Celery is invoked (from project root as 'worker.tasks' or from worker/ as 'tasks').
_WORKER_DIR = Path(__file__).parent
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))

from celery import Celery
from sqlalchemy import create_engine, Table, Column, String, Float, MetaData, update

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dental.db").replace(
    "sqlite+aiosqlite://", "sqlite://"
)
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "./data/scans"))
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

app = Celery("dental", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])

meta = MetaData()
scans_table = Table(
    "scans", meta,
    Column("id", String, primary_key=True),
    Column("status", String),
    Column("progress", Float),
    Column("stage", String),
    Column("error", String),
    Column("result_path", String),
    Column("coverage_pct", Float),
    Column("photometric_residual", Float),
)
engine = create_engine(DATABASE_URL)


def _set(scan_id: str, **kw):
    with engine.begin() as conn:
        conn.execute(update(scans_table).where(scans_table.c.id == scan_id).values(**kw))


@app.task(name="worker.tasks.reconstruct", bind=True, max_retries=0)
def reconstruct(self, scan_id: str, scan_dir: str):
    scan_path = Path(scan_dir)
    frames_dir = scan_path / "frames"
    output_dir = scan_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if DEMO_MODE:
            _run_demo(scan_id, output_dir)
        else:
            _run_pipeline(scan_id, frames_dir, output_dir)
    except Exception as exc:
        log.exception("Reconstruction failed for scan %s", scan_id)
        _set(scan_id, status="error", error=str(exc)[:500], progress=0.0, stage="failed")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Demo mode
# ──────────────────────────────────────────────────────────────────────────────

def _run_demo(scan_id: str, output_dir: Path):
    import time
    from demo_mode import generate_arch

    stages = [
        ("preprocessing", 0.10),
        ("geometry estimation (MASt3R)", 0.35),
        ("3D Gaussian Splatting", 0.60),
        ("mesh extraction (Poisson)", 0.80),
        ("quality validation", 0.95),
    ]
    for stage, progress in stages:
        _set(scan_id, status="running", stage=stage, progress=progress)
        time.sleep(1.5)  # simulate compute time for demo

    mesh_dir = output_dir / "mesh"
    obj_path = generate_arch(mesh_dir)

    _set(
        scan_id,
        status="done",
        stage="complete",
        progress=1.0,
        result_path=str(obj_path),
        coverage_pct=94.0,
        photometric_residual=0.031,
    )
    log.info("Demo scan %s complete", scan_id)


# ──────────────────────────────────────────────────────────────────────────────
# Real pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline(scan_id: str, frames_dir: Path, output_dir: Path):
    from pipeline.preprocess import preprocess_frames
    from pipeline.mastery import estimate_geometry
    from pipeline.gaussian_splatting import train_or_proxy
    from pipeline.mesh_extraction import extract_mesh
    from pipeline.quality import validate

    # Stage 1 — preprocessing
    _set(scan_id, status="running", stage="preprocessing", progress=0.05)
    frames = preprocess_frames(frames_dir)
    if not frames:
        raise ValueError("No readable frames in upload")
    log.info("Preprocessed %d frames for scan %s", len(frames), scan_id)

    # Stage 2 — geometry
    _set(scan_id, stage="geometry estimation (MASt3R)", progress=0.10)

    def geo_cb(p):
        _set(scan_id, progress=0.10 + p * 0.35)

    geometry = estimate_geometry(frames, progress_cb=geo_cb)
    log.info("Geometry: %d points via %s", len(geometry.points), geometry.method)

    # Stage 3 — 3DGS / point cloud
    _set(scan_id, stage="3D Gaussian Splatting", progress=0.45)

    def splat_cb(p):
        _set(scan_id, progress=0.45 + p * 0.25)

    splat = train_or_proxy(geometry, frames, output_dir / "splat", progress_cb=splat_cb)
    log.info("Splat scene: %d points via %s", splat.point_count, splat.method)

    # Stage 4 — mesh
    _set(scan_id, stage="mesh extraction (Poisson)", progress=0.70)

    def mesh_cb(p):
        _set(scan_id, progress=0.70 + p * 0.22)

    mesh = extract_mesh(splat.ply_path, output_dir / "mesh", progress_cb=mesh_cb)
    log.info("Mesh: %d vertices, %d faces", mesh.vertex_count, mesh.face_count)

    # Stage 5 — quality
    _set(scan_id, stage="quality validation", progress=0.93)
    report = validate(splat.ply_path, frames)
    log.info("Quality: %s", report.message)

    _set(
        scan_id,
        status="done",
        stage="complete",
        progress=1.0,
        result_path=str(mesh.obj_path),
        coverage_pct=report.coverage_pct,
        photometric_residual=report.photometric_residual,
    )
