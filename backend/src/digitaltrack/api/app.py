"""FastAPI app: capture upload (local use only).

Run:  dt serve            (http://127.0.0.1:8000; serves the built frontend if frontend/dist exists)
Dev:  dt serve + `npm run dev` in frontend/ (Vite proxies /api to port 8000)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from .. import __version__
from .archive import inspect_archives
from .jobs import JobManager, make_jobs_router
from .readings import ReadingStore, make_readings_router
from .storage import UploadError, UploadNotFound, UploadStore

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "real" / "uploads"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
MAX_FILE_BYTES = 20 * 1024**3  # 20 GiB per file; local machine

RecordingType = Literal["R0", "R1", "R2", "R3", "R4", "R5", "other"]


class Measurement(BaseModel):
    """A hand-measured reference dimension. Several readings so their spread is visible."""
    name: str = Field(min_length=1, max_length=60)  # e.g. "length", "diameter"
    readings_mm: list[float] = Field(min_length=1, max_length=10)
    instrument: str = Field(default="", max_length=60)
    level_mm: float | None = Field(default=None, gt=0, lt=2500,
                                   description="circumferences: height above the floor where the tape was held")

    @field_validator("readings_mm")
    @classmethod
    def positive(cls, v: list[float]) -> list[float]:
        if any(x <= 0 for x in v):
            raise ValueError("readings must be positive millimetres")
        return v


class UploadMetadata(BaseModel):
    recording_type: RecordingType
    label: str = Field(min_length=1, max_length=80, description="short description; no personal names")
    contains_person: bool = Field(description="a person is visible (or audible) in the recording")
    consent_confirmed: bool = Field(default=False, description="required when contains_person: consent obtained "
                                                                "under the team's process")
    subject_id: str = Field(default="", max_length=40, pattern=r"^[A-Za-z0-9_-]*$",
                            description="pseudonymous subject ID only; never a name")
    device_model: str = Field(default="", max_length=80)
    os_version: str = Field(default="", max_length=40)
    capture_app: str = Field(default="", max_length=60)
    capture_app_version: str = Field(default="", max_length=40)
    capture_settings: str = Field(default="", max_length=1000)
    object_kind: Literal["none", "plane", "box", "cylinder", "other"] = "none"
    object_textured: bool | None = None
    on_marker_disc: bool | None = None
    measurements: list[Measurement] = Field(default_factory=list, max_length=12)
    camera_distance_m: float | None = Field(default=None, gt=0, lt=20)
    camera_height_m: float | None = Field(default=None, gt=0, lt=5)
    step_deg: float | None = Field(default=None, gt=0, le=360)
    n_stops: int | None = Field(default=None, gt=0, le=1000)
    pause_s: float | None = Field(default=None, gt=0, le=600)
    ipad_touched_during_recording: bool | None = None
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def consent_for_people(self) -> "UploadMetadata":
        if self.contains_person and not self.consent_confirmed:
            raise ValueError("recordings with a person need consent confirmed under your team's process")
        return self


class ExpectedFile(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0)


class FinalizeRequest(BaseModel):
    files: list[ExpectedFile] = Field(min_length=1, max_length=200_000)


def create_app(data_dir: str | Path | None = None, *, serve_frontend: bool = True,
               job_runner=None, jobs_inline: bool = False) -> FastAPI:
    store = UploadStore(Path(data_dir or os.environ.get("DIGITALTRACK_UPLOAD_DIR", DEFAULT_DATA_DIR)))
    app = FastAPI(title="DIGITAL TRACK capture upload", version=__version__)
    app.state.store = store
    # processing jobs (experimental RGB mesh demo): data/real/jobs + data/real/sessions next to the uploads
    app.state.jobs = JobManager(store, store.data_dir.parent / "jobs", store.data_dir.parent / "sessions",
                                runner=job_runner, inline=jobs_inline)
    app.include_router(make_jobs_router(app.state.jobs))
    app.include_router(make_readings_router(ReadingStore(store.data_dir.parent / "readings.json")))
    from .validation import make_validation_router  # quick recording check + mesh-vs-tape comparison
    app.include_router(make_validation_router(store, app.state.jobs))
    from .avatar import make_avatar_router  # send a LiDAR depth mesh to the Body avatar page (display only)
    app.include_router(make_avatar_router(app.state.jobs))

    def _call(fn, *a):
        try:
            return fn(*a)
        except UploadNotFound:
            raise HTTPException(404, "upload not found")
        except UploadError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    @app.get("/api/uploads")
    def list_uploads():
        return store.list()

    @app.post("/api/uploads", status_code=201)
    def create_upload(meta: UploadMetadata):
        return {"id": store.create(meta.model_dump())}

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str):
        return _call(store.get, upload_id)

    @app.post("/api/uploads/{upload_id}/inspect")
    def inspect(upload_id: str):
        """Unpack the upload's zip files into a derived copy and describe them (no format interpretation)."""
        if _call(store.get, upload_id)["status"] != "complete":
            raise HTTPException(409, "upload is not finalized yet")
        return _call(inspect_archives, store.data_dir / upload_id)

    @app.put("/api/uploads/{upload_id}/files")
    async def put_file(upload_id: str, request: Request, path: str = Query(min_length=1, max_length=1024)):
        try:
            return await store.write_file(upload_id, path, request.stream(), MAX_FILE_BYTES)
        except UploadNotFound:
            raise HTTPException(404, "upload not found")
        except UploadError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/uploads/{upload_id}/finalize")
    def finalize(upload_id: str, req: FinalizeRequest):
        return _call(store.finalize, upload_id, [f.model_dump() for f in req.files])

    @app.delete("/api/uploads/{upload_id}", status_code=204)
    def discard(upload_id: str):
        _call(store.discard, upload_id)

    if serve_frontend and FRONTEND_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{_path:path}", include_in_schema=False)
        def spa(_path: str):
            return FileResponse(FRONTEND_DIST / "index.html", headers={"Cache-Control": "no-cache"})  # new builds show up

    return app
