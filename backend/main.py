import os
import uuid
import aiofiles
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db import init_db, get_db, Scan
from celery_app import app as celery_app

STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "./data/scans"))
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Dental 3D POC", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/scans", status_code=201)
async def create_scan(
    frames: list[UploadFile] = File(default=[]),
    demo: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    scan_id = str(uuid.uuid4())
    scan_dir = STORAGE_PATH / scan_id
    frames_dir = scan_dir / "frames"
    frames_dir.mkdir(parents=True)

    saved_count = 0
    for frame in frames:
        dest = frames_dir / (frame.filename or f"frame_{saved_count:04d}.jpg")
        async with aiofiles.open(dest, "wb") as f:
            await f.write(await frame.read())
        saved_count += 1

    scan = Scan(id=scan_id, status="queued", frame_count=saved_count)
    db.add(scan)
    await db.commit()

    use_demo = demo or DEMO_MODE or saved_count == 0
    celery_app.send_task(
        "worker.tasks.reconstruct",
        args=[scan_id, str(scan_dir)],
        kwargs={},
        headers={"demo": use_demo},
    )

    return {"scan_id": scan_id, "frame_count": saved_count, "status": "queued", "demo": use_demo}


@app.get("/scans/{scan_id}")
async def get_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    scan = await db.get(Scan, scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found")
    return {
        "scan_id": scan.id,
        "status": scan.status,
        "progress": scan.progress,
        "stage": scan.stage,
        "frame_count": scan.frame_count,
        "error": scan.error,
        "quality": {
            "coverage_pct": scan.coverage_pct,
            "photometric_residual": scan.photometric_residual,
        } if scan.status == "done" else None,
    }


@app.get("/scans/{scan_id}/result")
async def get_result(scan_id: str, db: AsyncSession = Depends(get_db)):
    scan = await db.get(Scan, scan_id)
    if not scan or scan.status != "done":
        raise HTTPException(404, "Result not ready")
    mesh_path = Path(scan.result_path)
    if not mesh_path.exists():
        raise HTTPException(500, "Result file missing")
    return FileResponse(
        mesh_path,
        media_type="model/obj",
        filename=f"scan_{scan_id}.obj",
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.get("/scans/{scan_id}/ply")
async def get_ply(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Download raw point cloud PLY (for debugging / advanced viewer)."""
    scan = await db.get(Scan, scan_id)
    if not scan or scan.status != "done":
        raise HTTPException(404, "Result not ready")
    ply_path = Path(scan.result_path).parent.parent / "splat" / "splat.ply"
    if not ply_path.exists():
        raise HTTPException(404, "PLY not available")
    return FileResponse(ply_path, media_type="application/octet-stream", filename=f"scan_{scan_id}.ply")


@app.post("/scans/demo", status_code=201)
async def create_demo_scan(db: AsyncSession = Depends(get_db)):
    """One-click demo: no frames needed, generates synthetic arch."""
    scan_id = str(uuid.uuid4())
    scan_dir = STORAGE_PATH / scan_id
    (scan_dir / "frames").mkdir(parents=True)

    scan = Scan(id=scan_id, status="queued", frame_count=0)
    db.add(scan)
    await db.commit()

    celery_app.send_task("worker.tasks.reconstruct", args=[scan_id, str(scan_dir)])
    return {"scan_id": scan_id, "frame_count": 0, "status": "queued", "demo": True}


@app.get("/health")
async def health():
    return {"ok": True, "demo_mode": DEMO_MODE}
