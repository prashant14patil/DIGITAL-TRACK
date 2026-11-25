"""API: quick recording check for depth uploads, and mesh-vs-tape comparison for finished depth jobs.

Both only READ uploads; results are derived files under data/real/processing/ (quick check) or the job's run
directory (comparison.json).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .storage import UploadError, UploadNotFound, UploadStore


class QuickCheckRequest(BaseModel):
    file: str | None = Field(default=None, max_length=1024)  # which .zip in the upload (default: the only one)
    background_upload_id: str | None = Field(default=None, max_length=64)


def _stray_views(store: UploadStore, upload_id: str, file: str | None, target_fps: float = 6.0):
    from ..rgbd.stray import StrayFormatError, find_stray_dirs, load_stray
    from .archive import inspect_archives

    up = store.get(upload_id)
    if up["status"] != "complete":
        raise HTTPException(409, "upload is not finalized yet")
    zips = [f["path"] for f in up["files"] if f["path"].lower().endswith(".zip")]
    if file is None:
        if len(zips) != 1:
            raise HTTPException(409, f"upload has {len(zips)} .zip files; choose one")
        file = zips[0]
    elif file not in zips:
        raise HTTPException(404, "no such .zip in this upload")
    upload_dir = store.data_dir / upload_id
    inspect_archives(upload_dir)  # derived unpacked copy; the zip itself is never changed
    dirs = find_stray_dirs(upload_dir / "extracted" / file.replace("/", "__")[:-4])
    if len(dirs) != 1:
        raise HTTPException(409, f"expected one Stray Scanner recording in {file}, found {len(dirs)}")
    try:
        return load_stray(dirs[0], target_fps=target_fps), file
    except StrayFormatError as e:
        raise HTTPException(409, f"not the expected Stray Scanner format: {e}") from e


def make_validation_router(store: UploadStore, jobs) -> APIRouter:
    r = APIRouter()

    @r.post("/api/uploads/{upload_id}/quickcheck")
    def quickcheck(upload_id: str, req: QuickCheckRequest | None = None):
        from ..validation.quickcheck import quick_check, write

        req = req or QuickCheckRequest()
        try:
            rec, file = _stray_views(store, upload_id, req.file)
            bg = None
            if req.background_upload_id:
                bg_rec, _ = _stray_views(store, req.background_upload_id, None, target_fps=3.0)
                bg = bg_rec.views[:30]
        except UploadNotFound:
            raise HTTPException(404, "upload not found")
        except UploadError as e:
            raise HTTPException(409, str(e))
        result = quick_check(rec.views, info=rec.info, background_views=bg)
        result.update(upload_id=upload_id, file=file, background_upload_id=req.background_upload_id)
        p = write(result, store.data_dir.parent / "processing" / upload_id)
        result["saved_as"] = p.name
        return result

    @r.get("/api/jobs/{job_id}/comparison")
    def comparison(job_id: str):
        from ..validation.compare import compare_run

        try:
            job = jobs.get(job_id)
        except (KeyError, FileNotFoundError, ValueError):
            raise HTTPException(404, "job not found")
        if job.get("kind") != "rgbd_depth":
            raise HTTPException(409, "comparison is only defined for depth (LiDAR) reconstructions")
        if not job.get("run_dir") or not Path(job["run_dir"]).is_dir():
            raise HTTPException(409, "job has no results yet")
        try:
            meta = store.get(job["upload_id"])["metadata"]
        except UploadNotFound:
            raise HTTPException(404, "the job's upload no longer exists")
        return compare_run(job["run_dir"], meta)

    return r
