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
import re
import threading
import traceback
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..ingest.errors import IngestError
from ..ingest.metadata import validate_capture_metadata
from .storage import UploadNotFound, UploadStore

VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v")
ZIP_SUFFIXES = (".zip",)  # Stray Scanner export (LiDAR depth) -> rgbd self-rotation pipeline
DEPTH_RESULT_FILES = {  # depth jobs: public name -> file in the run directory (plus any *.png the run wrote)
    "mesh.glb": "fused_mesh.glb",
    "mesh.ply": "fused_mesh.ply",
    "run_log.json": "run_log.json",
    "report.html": "report.html",
    "mesh.obj": "fused_mesh.obj",  # tracked pipeline only
    "quality_report.json": "quality_report.json",  # tracked pipeline only
    "body_measurements.json": "body_measurements.json",  # tracked pipeline only (proposed definitions)
    "refined.glb": "refined_mesh.glb",  # tracked pipeline only: closed display mesh fitted to the measuring mesh
    "refined.ply": "refined_mesh.ply",
    "refined_quality.json": "refined_quality.json",
}
DEPTH_CONTINUOUS_VIEWS = 48  # fallback when no pauses are found: evenly spaced frames
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
    "quality.json": "quality.json",  # recording warnings, rough slices (scan_review)
    "slices.json": "slices.json",
}
# overall progress = start + frac * width, per stage
STAGES = {"queued": (0.0, 0.0), "ingesting": (0.0, 0.15), "orienting": (0.15, 0.02), "segmenting": (0.17, 0.63),
          "estimating turn angles": (0.80, 0.04), "carving": (0.84, 0.08), "meshing": (0.92, 0.04),
          "rendering previews": (0.96, 0.03), "done": (1.0, 0.0),
          "unpacking": (0.0, 0.05), "loading depth": (0.05, 0.10), "reconstructing": (0.15, 0.80),
          "reconstructing (no pauses found, evenly spaced frames)": (0.55, 0.40), "writing results": (0.95, 0.05)}


def _clear_readonly(func, path, _exc) -> None:
    """rmtree helper: Windows refuses to delete read-only files; make writable and retry once."""
    import os
    import stat
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    func(path)


def _body_summary(run_dir: Path) -> dict | None:
    """Compact copy of body_measurements.json for the job result: every value keeps its status and reasons."""
    p = run_dir / "body_measurements.json"
    if not p.is_file():
        return None
    b = json.loads(p.read_text(encoding="utf-8"))
    return {"definition_version": b.get("definition_version"), "status": b.get("status"), "flags": b.get("flags", []),
            "landmarks": b.get("landmarks", {}),
            "measurements": [{k: m.get(k) for k in ("name", "value", "unit", "status", "reasons", "level_z",
                                                     "convex_hull_m")} for m in b.get("measurements", [])]}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProcessRequest(BaseModel):
    file: str | None = Field(default=None, max_length=1024,
                             description="relative path of the video inside the upload; optional if there is exactly one")
    background_upload_id: str | None = Field(default=None, max_length=64,
                                             description="depth jobs: finalized upload of the empty scene (one .zip)")


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

    def submit(self, upload_id: str, file: str | None, background_upload_id: str | None = None) -> dict[str, Any]:
        up = self.store.get(upload_id)  # raises UploadNotFound
        if up["status"] != "complete":
            raise ValueError("upload is not finalized yet")
        videos = [f["path"] for f in up["files"] if f["path"].lower().endswith(VIDEO_SUFFIXES + ZIP_SUFFIXES)]
        if file is None:
            if len(videos) != 1:
                raise ValueError(f"upload has {len(videos)} video/zip files; choose one: {videos}" if videos
                                 else "upload contains no .mov/.mp4/.m4v video or .zip export")
            file = videos[0]
        elif file not in videos:
            raise ValueError(f"{file!r} is not a video or zip file of this upload")
        for j in self.list():
            if j["upload_id"] == upload_id and j["file"] == file and j["status"] in ("queued", "running"):
                raise ValueError(f"this video is already being processed (job {j['id']})")
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        job = {"id": job_id, "upload_id": upload_id, "file": file, "label": up["metadata"].get("label", ""),
               "kind": "rgbd_depth" if file.lower().endswith(ZIP_SUFFIXES) else "rgb_visual_hull_demo", "status": "queued", "stage": "queued", "progress": 0.0,
               "created_utc": utc_now(), "started_utc": None, "finished_utc": None, "error": None,
               "session_id": None, "run_dir": None, "result": None}
        if background_upload_id:
            if not file.lower().endswith(ZIP_SUFFIXES):
                raise ValueError("an empty-scene recording can only be used with a Stray Scanner .zip")
            if background_upload_id == upload_id:
                raise ValueError("the empty-scene recording must be a different upload")
            bg = self.store.get(background_upload_id)
            bg_zips = [f["path"] for f in bg["files"] if f["path"].lower().endswith(ZIP_SUFFIXES)]
            if bg["status"] != "complete" or len(bg_zips) != 1:
                raise ValueError("the empty-scene upload must be finalized and contain exactly one .zip")
            job["background"] = {"upload_id": background_upload_id, "file": bg_zips[0]}
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
        if job.get("kind") == "rgbd_depth":
            return self._run_depth(job)
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
            self._add_review(Path(run_dir))
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

    def _run_depth(self, job: dict[str, Any]) -> None:
        """Stray Scanner zip -> unpacked copy -> RGBDViews -> self-rotation TSDF mesh (metric, not validated)."""
        job_id = job["id"]
        try:
            from ..rgbd.pipeline import PipelineParams, run_self_rotation
            from ..rgbd.pose import PoseParams
            from ..rgbd.stops import StopParams
            from ..rgbd.stray import StrayFormatError, find_stray_dirs, load_stray
            from .archive import inspect_archives

            upload_dir = self.store.data_dir / job["upload_id"]
            self._progress(job_id, "unpacking", 0.0)
            inspect_archives(upload_dir)  # derived copy under <upload>/extracted/; the zip is never changed
            extracted = upload_dir / "extracted" / job["file"].replace("/", "__")[:-4]
            dirs = find_stray_dirs(extracted)
            if len(dirs) != 1:
                top = sorted(p.relative_to(extracted).as_posix() for p in extracted.rglob("*"))[:15]
                raise ValueError(f"expected one Stray Scanner recording (a folder with depth/ and odometry.csv) "
                                 f"in the zip, found {len(dirs)}. Contents start with: {top}")
            self._progress(job_id, "loading depth", 0.0)
            try:
                rec = load_stray(dirs[0])
            except StrayFormatError as e:
                raise ValueError(f"the zip does not match the expected Stray Scanner format: {e}") from e
            out_root = self.store.data_dir.parent / "processing" / job["upload_id"]
            source = {"upload_id": job["upload_id"], "file": job["file"], "label": job.get("label", ""),
                      "stray": rec.info}
            bg_views = None
            if job.get("background"):
                b = job["background"]
                bg_dir = self.store.data_dir / b["upload_id"]
                inspect_archives(bg_dir)
                bg_dirs = find_stray_dirs(bg_dir / "extracted" / b["file"].replace("/", "__")[:-4])
                if len(bg_dirs) != 1:
                    raise ValueError(f"the empty-scene zip holds {len(bg_dirs)} Stray Scanner recordings, expected 1")
                try:
                    bg_rec = load_stray(bg_dirs[0], target_fps=3.0)
                except StrayFormatError as e:
                    raise ValueError(f"empty-scene zip: {e}") from e
                bg_views = bg_rec.views[:30]
                source["background"] = {**b, "stray": bg_rec.info, "views_used": len(bg_views)}
            if job.get("background") and (dirs[0] / "rgb.mp4").is_file():
                # RGB-D tracked pipeline (rgbd.tracked): needs the empty scene and the RGB video. The first real
                # export never closed the loop with the shape-only path below. On failure: fall back, reason kept.
                tracked_error = self._run_tracked(job, dirs[0], bg_dirs[0], out_root, source, rec.info)
                if tracked_error is None:
                    return
                source["tracked_pipeline_failed"] = tracked_error
            self._progress(job_id, "reconstructing", 0.0)
            res = run_self_rotation(rec.views, out_root, background_views=bg_views, source=source,
                                    run_id=f"selfrot_{job_id}", params=PipelineParams(stops=StopParams(adaptive=True, min_frames=6),
                                                                          pose=PoseParams(closure_gap_deg=90.0)))
            mode = "pauses (stop detection)"
            if res.mesh is None and res.log.get("stops", {}).get("n", 1) == 0:
                self._progress(job_id, "reconstructing (no pauses found, evenly spaced frames)", 0.0)
                idx = np.linspace(0, len(rec.views) - 1, min(DEPTH_CONTINUOUS_VIEWS, len(rec.views))).round()
                views = [rec.views[int(i)] for i in sorted(set(idx))]
                res = run_self_rotation(views, out_root, background_views=bg_views,
                                        source={**source, "fallback": "evenly spaced frames"},
                                        params=PipelineParams(detect_stops=False),
                                        run_id=f"selfrot_{job_id}_continuous")
                mode = f"continuous turn: {len(views)} evenly spaced frames (no pauses found)"
            self._progress(job_id, "writing results", 0.0)
            run_dir = Path(res.run_dir)
            if res.mesh is not None and len(res.mesh.vertices):
                import open3d as o3d
                m = o3d.geometry.TriangleMesh(res.mesh)
                m.compute_vertex_normals()
                o3d.io.write_triangle_mesh(str(run_dir / "fused_mesh.glb"), m)
            log = res.log
            bounds = (log.get("mesh") or {}).get("bounds_m")
            files = [n for n, f in DEPTH_RESULT_FILES.items() if (run_dir / f).is_file()]
            files += sorted(p.name for p in run_dir.glob("*.png"))
            result = {
                "status": res.status, "status_note": log.get("status_note"), "warnings": log.get("warnings", []),
                "mode": mode, "n_views": log.get("n_views"), "stray": rec.info,
                "background_upload_id": (job.get("background") or {}).get("upload_id"),
                "pose_checks": (log.get("pose") or {}).get("checks"),
                "mesh": {"vertices": (log.get("mesh") or {}).get("vertices"),
                         "triangles": (log.get("mesh") or {}).get("triangles"),
                         "height_extent_m": round(bounds[1][2] - bounds[0][2], 3) if bounds else None,
                         "units": "metres (LiDAR depth; not validated against tape)"},
                "files": files,
            }
            if "mesh.glb" not in files:
                self._update(job_id, status="failed", finished_utc=utc_now(), run_dir=str(run_dir), result=result,
                             error=f"no mesh: {log.get('status_note')}")
                return
            self._update(job_id, status="complete", stage="done", progress=1.0, finished_utc=utc_now(),
                         run_dir=str(run_dir), result=result)
        except (ValueError, RuntimeError, FileExistsError) as e:
            self._update(job_id, status="failed", finished_utc=utc_now(), error=str(e))
        except Exception as e:  # noqa: BLE001
            (self.jobs_dir / job_id / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self._update(job_id, status="failed", finished_utc=utc_now(),
                         error=f"internal error ({type(e).__name__}): {e}; see traceback.txt in the job folder")

    def _run_tracked(self, job: dict[str, Any], recording: Path, background: Path, out_root: Path, source: dict,
                     stray_info: dict) -> str | None:
        """Run rgbd.tracked for a depth job. Returns None when the job was completed, else the failure reason
        (the caller then falls back to the shape-only self-rotation pipeline)."""
        from ..rgbd.tracked import run_tracked

        job_id = job["id"]
        order = ["loading depth", "segmenting", "RGB features", "registering view pairs", "loop-closure search",
                 "optimising pose graph", "fusing", "fusing (pass 1)", "consistency vote", "fusing (pass 2)",
                 "exporting", "quality report", "split-half check"]

        def tell(stage: str, frac: float) -> None:
            k = order.index(stage) if stage in order else 0
            self._progress(job_id, "reconstructing", (k + max(0.0, min(1.0, frac))) / len(order))

        try:
            res = run_tracked(recording, background, out_root, run_id=f"tracked_{job_id}", source=source,
                              progress=tell)
        except Exception as e:  # noqa: BLE001 - any failure: fall back, keep the reason
            return f"{type(e).__name__}: {e}"
        q, run_dir = res.quality, Path(res.run_dir)
        files = [n for n, f in DEPTH_RESULT_FILES.items() if (run_dir / f).is_file()]
        files += sorted(p.name for p in run_dir.glob("*.png"))
        mesh = q.get("mesh") or {}
        bounds = mesh.get("bounds_m")
        result = {
            "status": res.status,
            "status_note": "Metric mesh from the RGB-D tracked pipeline. Not validated against tape or a known "
                           "object: every dimension is diagnostic only.",
            "warnings": q.get("warnings", []),
            "mode": f"RGB-D tracked: feature pose graph, {q['views']['fused']} of {q['views']['posed']} views fused",
            "n_views": q["views"]["fused"], "stray": stray_info,
            "background_upload_id": (job.get("background") or {}).get("upload_id"),
            "pose_checks": q.get("checks"),
            "body_measurements": _body_summary(run_dir),
            "refined_mesh": q.get("refined_mesh"),
            "mesh": {"vertices": mesh.get("vertices"), "triangles": mesh.get("triangles"),
                     "holes": mesh.get("holes"),
                     "height_extent_m": round(bounds[1][2] - bounds[0][2], 3) if bounds else None,
                     "units": "metres (LiDAR depth; not validated against tape)"},
            "files": files,
        }
        if "mesh.glb" not in files:
            return "tracked pipeline wrote no mesh"
        self._update(job_id, status="complete", stage="done", progress=1.0, finished_utc=utc_now(),
                     run_dir=str(run_dir), result=result)
        return None

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

    @staticmethod
    def _add_review(run_dir: Path) -> None:
        """Recording warnings and rough slice readings. Optional, like the regions."""
        if (run_dir / "quality.json").is_file() and (run_dir / "slices.json").is_file():
            return
        try:
            from ..experimental.scan_review import write_review_files
            write_review_files(run_dir)
        except Exception as e:  # noqa: BLE001 - record and continue
            (run_dir / "review_error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")

    def backfill_regions(self) -> int:
        """Add region/review files to completed jobs made before those existed. Returns how many jobs changed."""
        n = 0
        for job in self.list():
            if job["status"] != "complete" or not job.get("run_dir") or job.get("kind") == "rgbd_depth":
                continue
            run_dir = Path(job["run_dir"])
            before = {k for k, f in RESULT_FILES.items() if (run_dir / f).is_file()}
            listed = set((job.get("result") or {}).get("files", []))
            self._add_regions(run_dir)
            self._add_review(run_dir)
            files = [k for k, f in RESULT_FILES.items() if (run_dir / f).is_file()]
            if set(files) - before or set(files) - listed:
                self._update(job["id"], result={**(job.get("result") or {}), "files": files})
                n += 1
        return n

    def delete(self, job_id: str) -> dict[str, Any]:
        """Remove a finished/failed job and its DERIVED outputs (mesh run folder, ingested session copy, avatar files).

        The uploaded recording is raw data and is never touched: the mesh can be made again from it. Only folders
        inside this manager's own derived-data roots are removed. Returns what was deleted (for the server log).
        """
        import shutil

        with self._lock:
            job = self.get(job_id)  # raises KeyError
            if job["status"] in ("queued", "running"):
                raise ValueError("this job is still being processed; wait until it finishes or fails")
            processing_root = self.store.data_dir.parent / "processing"
            targets: list[Path] = []
            if job.get("session_id"):  # RGB jobs: ingested frames + processing/<run_id> live in the session copy
                targets.append(self.sessions_dir / job["session_id"])
            if job.get("run_dir"):
                targets.append(Path(job["run_dir"]))
            if job.get("kind") == "rgbd_depth":  # first attempt + "_continuous" fallback run
                targets += (processing_root / job["upload_id"]).glob(f"selfrot_{job_id}*")
                targets += (processing_root / job["upload_id"]).glob(f"tracked_{job_id}*")  # incl. failed attempts
            targets.append(self.jobs_dir / job_id)
            allowed = [r.resolve() for r in (self.sessions_dir, processing_root, self.jobs_dir)]
            removed = []
            for t in dict.fromkeys(p.resolve() for p in targets):
                if not t.exists():
                    continue
                if not any(t != r and t.is_relative_to(r) for r in allowed):
                    raise RuntimeError(f"refusing to delete {t}: outside the derived-data folders")
                shutil.rmtree(t, onerror=_clear_readonly)  # session copies keep source files read-only
                removed.append(str(t))
            return {"id": job_id, "removed": removed}

    def delete_upload(self, upload_id: str) -> dict[str, Any]:
        """Permanently delete an upload AND every mesh made from it (user request from the web page).

        Refused while any job that reads this upload (as recording or as empty-scene background) is queued/running.
        Meshes that only used it as background keep their results (they no longer need it).
        """
        import shutil

        self.store.get(upload_id)  # raises UploadNotFound
        jobs = self.list()
        if any(j["status"] in ("queued", "running")
               and upload_id in (j["upload_id"], (j.get("background") or {}).get("upload_id")) for j in jobs):
            raise ValueError("a mesh is still being made from this recording; wait until it finishes or fails")
        removed = [self.delete(j["id"])["id"] for j in jobs if j["upload_id"] == upload_id]
        processing = self.store.data_dir.parent / "processing" / upload_id  # leftovers not tied to a listed job
        if processing.is_dir():
            shutil.rmtree(processing, onerror=_clear_readonly)
        self.store.delete(upload_id)
        return {"id": upload_id, "meshes_removed": removed}

    def result_file(self, job_id: str, name: str) -> Path:
        job = self.get(job_id)
        if not job.get("run_dir"):
            raise FileNotFoundError("job has no results")
        if job.get("kind") == "rgbd_depth":  # failed depth runs still have diagnostic images
            if name in DEPTH_RESULT_FILES:
                p = Path(job["run_dir"]) / DEPTH_RESULT_FILES[name]
            elif re.fullmatch(r"[a-z0-9_]{1,40}\.png", name):
                p = Path(job["run_dir"]) / name
            else:
                raise KeyError(name)
            if not p.is_file():
                raise FileNotFoundError(name)
            return p
        if name not in RESULT_FILES:
            raise KeyError(name)
        if job["status"] != "complete":
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
            return manager.submit(upload_id, req.file if req else None, req.background_upload_id if req else None)
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

    @r.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        """Remove a mesh (job + derived outputs). The uploaded recording stays."""
        try:
            manager.delete(job_id)
        except KeyError:
            raise HTTPException(404, "job not found")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @r.get("/api/jobs/{job_id}/files/{name}")
    def get_file(job_id: str, name: str):
        try:
            p = manager.result_file(job_id, name)
        except KeyError:
            raise HTTPException(404, "job or file not found")
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        media = {".glb": "model/gltf-binary", ".png": "image/png", ".json": "application/json",
                 ".ply": "application/octet-stream", ".obj": "model/obj", ".html": "text/html"}[p.suffix]
        return FileResponse(p, media_type=media, headers={"Cache-Control": "no-store"})

    return r
