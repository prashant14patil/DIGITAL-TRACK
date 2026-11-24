"""Processing jobs for the web app: finalized upload (Camera-app video) -> session -> EXPERIMENTAL mesh.

Pipeline per job (one job at a time, background thread):
  1. ingest:   digitaltrack.ingest.ingest_video_from_upload (size + SHA-256 checked against the upload manifest;
               the upload is only read) -> data/real/sessions/<session_id>/
  2. mesh:     digitaltrack.experimental.rgb_visual_hull.run_visual_hull -> <session>/processing/<run_id>/
Every result is labelled `unreliable` (non-metric, uncalibrated, estimated turn angles). No measurements.

Job state lives in <jobs_dir>/<job_id>/job.json so it survives restarts; jobs that were queued or running when the
server stopped are marked failed ("interrupted"), never left hanging. Only whitelisted result files are served.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..ingest.errors import IngestError
from ..ingest.metadata import validate_capture_metadata
from .storage import UploadNotFound, UploadStore

VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v")
TARGET_FPS = 10.0  # frames kept per second of video (every n-th decoded frame)
WEB_VOTE_FRACTION = 0.8  # keeps more of the (moving) arms than the CLI default 0.9; documented in the report
RESULT_FILES = {  # public name -> file in the run directory
    "mesh.glb": "mesh_height_normalised.glb",
    "mesh.ply": "mesh_height_normalised.ply",
    "preview_mesh.png": "preview_mesh.png",
    "preview_silhouettes.png": "preview_silhouettes.png",
    "preview_angle_estimation.png": "preview_angle_estimation.png",
    "report.json": "report.json",
    "mesh_regions.glb": "mesh_regions.glb",  # display regions (experimental.body_regions), avatar view only
    "regions.json": "regions.json",
}
# overall progress = start + frac * width, per stage
STAGES = {"queued": (0.0, 0.0), "ingesting": (0.0, 0.15), "orienting": (0.15, 0.02), "segmenting": (0.17, 0.63),
          "estimating turn angles": (0.80, 0.04), "carving": (0.84, 0.08), "meshing": (0.92, 0.04),
          "rendering previews": (0.96, 0.03), "done": (1.0, 0.0)}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProcessRequest(BaseModel):
    file: str | None = Field(default=None, max_length=1024,
                             description="relative path of the video inside the upload; optional if there is exactly one")


class JobManager:
    def __init__(self, store: UploadStore, jobs_dir: Path, sessions_dir: Path, *,
                 runner: Callable[..., tuple[Path, dict]] | None = None, inline: bool = False) -> None:
        self.store, self.jobs_dir, self.sessions_dir = store, Path(jobs_dir), Path(sessions_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._inline = inline
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._recover()
        if not inline:
            self._queue.put("__backfill__")  # region files for older completed jobs, done by the worker thread
            threading.Thread(target=self._loop, name="digitaltrack-jobs", daemon=True).start()

    # ---- persistence ----------------------------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        if not (job_id.startswith("job_") and job_id.replace("_", "").isalnum() and len(job_id) < 64):
            raise KeyError(job_id)
        return self.jobs_dir / job_id / "job.json"

    def _save(self, job: dict[str, Any]) -> None:
        p = self._path(job["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)

    def get(self, job_id: str) -> dict[str, Any]:
        p = self._path(job_id)
        if not p.is_file():
            raise KeyError(job_id)
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        jobs = [json.loads(p.read_text(encoding="utf-8")) for p in self.jobs_dir.glob("job_*/job.json")]
        return sorted(jobs, key=lambda j: j["created_utc"], reverse=True)

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            job.update(changes)
            self._save(job)
            return job

    def _recover(self) -> None:
        for job in self.list():
            if job["status"] in ("queued", "running"):
                job.update(status="failed", finished_utc=utc_now(),
                           error="interrupted: the server stopped before this job finished; start it again")
                self._save(job)

    # ---- submission -----------------------------------------------------------------------------

    def submit(self, upload_id: str, file: str | None) -> dict[str, Any]:
        up = self.store.get(upload_id)  # raises UploadNotFound
        if up["status"] != "complete":
            raise ValueError("upload is not finalized yet")
        videos = [f["path"] for f in up["files"] if f["path"].lower().endswith(VIDEO_SUFFIXES)]
        if file is None:
            if len(videos) != 1:
                raise ValueError(f"upload has {len(videos)} video files; choose one: {videos}" if videos
                                 else "upload contains no .mov/.mp4/.m4v video")
            file = videos[0]
        elif file not in videos:
            raise ValueError(f"{file!r} is not a video file of this upload")
        for j in self.list():
            if j["upload_id"] == upload_id and j["file"] == file and j["status"] in ("queued", "running"):
                raise ValueError(f"this video is already being processed (job {j['id']})")
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        job = {"id": job_id, "upload_id": upload_id, "file": file, "label": up["metadata"].get("label", ""),
               "kind": "rgb_visual_hull_demo", "status": "queued", "stage": "queued", "progress": 0.0,
               "created_utc": utc_now(), "started_utc": None, "finished_utc": None, "error": None,
               "session_id": None, "run_dir": None, "result": None}
        with self._lock:
            self._save(job)
        if self._inline:
            self._run(job_id)
        else:
            self._queue.put(job_id)
        return self.get(job_id)

    # ---- execution ------------------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id == "__backfill__":
                    self.backfill_regions()
                    continue
                self._run(job_id)
            finally:
                self._queue.task_done()

    def _progress(self, job_id: str, stage: str, frac: float) -> None:
        start, width = STAGES.get(stage, (None, None))
        changes: dict[str, Any] = {"stage": stage}
        if start is not None:
            changes["progress"] = round(min(1.0, start + width * max(0.0, min(1.0, frac))), 3)
        self._update(job_id, **changes)

    def _run(self, job_id: str) -> None:
        from ..experimental.rgb_visual_hull import run_visual_hull
        from ..ingest.session import ingest_video_from_upload
        from ..ingest.video import probe_video

        job = self._update(job_id, status="running", started_utc=utc_now())
        try:
            up = self.store.get(job["upload_id"])
            meta = up["metadata"]
            md = validate_capture_metadata({
                "subject_id": meta.get("subject_id") or "unspecified",
                "setup_id": "web-upload",
                "protocol_version": "web-demo-unvalidated",
                "capture_type": "rgb_video",
                **({"recording_app": meta["capture_app"]} if meta.get("capture_app") else {}),
                **({"device_model": meta["device_model"]} if meta.get("device_model") else {}),
            })
            upload_dir = self.store.data_dir / job["upload_id"]
            src = upload_dir / "export" / Path(*job["file"].split("/"))
            self._progress(job_id, "ingesting", 0.0)
            probe = probe_video(src)
            fps = float(Fraction(probe["avg_frame_rate"])) if probe.get("avg_frame_rate") not in (None, "0/0") else 30.0
            step = max(1, round(fps / TARGET_FPS))
            session_id = f"ses_{job['upload_id']}_{job_id[-6:]}"
            session_dir = self.sessions_dir / session_id
            ingest_video_from_upload(upload_dir, job["file"], session_dir, md, session_id=session_id, frame_step=step)
            self._update(job_id, session_id=session_id)
            self._progress(job_id, "ingesting", 1.0)
            runner = self._runner or run_visual_hull
            run_dir, report = runner(session_dir, vote_fraction=WEB_VOTE_FRACTION,
                                     progress=lambda s, f: self._progress(job_id, s, f))
            self._add_regions(Path(run_dir))
            self._update(job_id, status="complete", stage="done", progress=1.0, finished_utc=utc_now(),
                         run_dir=str(run_dir), result={
                             "status": report.get("status"), "status_reasons": report.get("status_reasons", []),
                             "measurements": report.get("measurements"), "mesh": report.get("mesh"),
                             "angle_anchors": report.get("angles", {}).get("anchors", []),
                             "frames_used": report.get("carving", {}).get("frames"),
                             "frame_step": step, "runtime_s": report.get("runtime_s"),
                             "files": [n for n, f in RESULT_FILES.items() if (Path(run_dir) / f).is_file()]})
        except (IngestError, ValueError, RuntimeError) as e:
            self._update(job_id, status="failed", finished_utc=utc_now(), error=str(e))
        except Exception as e:  # unexpected: keep the traceback next to the job for diagnosis
            (self.jobs_dir / job_id / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self._update(job_id, status="failed", finished_utc=utc_now(),
                         error=f"internal error ({type(e).__name__}): {e}; see traceback.txt in the job folder")

    @staticmethod
    def _add_regions(run_dir: Path) -> None:
        """Display regions for the avatar view. Optional: a failure here never fails the mesh job."""
        if not (run_dir / "mesh_height_normalised.ply").is_file() or (run_dir / "regions.json").is_file():
            return
        try:
            from ..experimental.body_regions import write_region_files
            write_region_files(run_dir)
        except Exception as e:  # noqa: BLE001 - record and continue
            (run_dir / "regions_error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")

    def backfill_regions(self) -> int:
        """Add region files to completed jobs made before the avatar view existed. Returns how many were added."""
        n = 0
        for job in self.list():
            if job["status"] == "complete" and job.get("run_dir") and not (Path(job["run_dir"]) / "regions.json").is_file():
                self._add_regions(Path(job["run_dir"]))
                if (Path(job["run_dir"]) / "regions.json").is_file():
                    files = [k for k, f in RESULT_FILES.items() if (Path(job["run_dir"]) / f).is_file()]
                    self._update(job["id"], result={**(job.get("result") or {}), "files": files})
                    n += 1
        return n

    def result_file(self, job_id: str, name: str) -> Path:
        job = self.get(job_id)
        if name not in RESULT_FILES:
            raise KeyError(name)
        if job["status"] != "complete" or not job["run_dir"]:
            raise FileNotFoundError("job has no results")
        p = Path(job["run_dir"]) / RESULT_FILES[name]
        if not p.is_file():
            raise FileNotFoundError(name)
        return p


def make_jobs_router(manager: JobManager) -> APIRouter:
    r = APIRouter()

    @r.post("/api/uploads/{upload_id}/process", status_code=202)
    def process(upload_id: str, req: ProcessRequest | None = None):
        try:
            return manager.submit(upload_id, req.file if req else None)
        except UploadNotFound:
            raise HTTPException(404, "upload not found")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @r.get("/api/jobs")
    def list_jobs():
        return manager.list()

    @r.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "job not found")

    @r.get("/api/jobs/{job_id}/files/{name}")
    def get_file(job_id: str, name: str):
        try:
            p = manager.result_file(job_id, name)
        except KeyError:
            raise HTTPException(404, "job or file not found")
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        media = {".glb": "model/gltf-binary", ".png": "image/png", ".json": "application/json",
                 ".ply": "application/octet-stream"}[p.suffix]
        return FileResponse(p, media_type=media, headers={"Cache-Control": "no-store"})

    return r
