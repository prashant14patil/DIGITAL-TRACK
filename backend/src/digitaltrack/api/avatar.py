"""API: send a LiDAR depth mesh to the Body avatar page (display copy + regions; nothing is measured)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

AVATAR_FILES = {"mesh_regions.glb": "model/gltf-binary", "regions.json": "application/json",
                "avatar_display.json": "application/json"}


def _depth_job(jobs, job_id: str) -> dict:
    try:
        job = jobs.get(job_id)
    except (KeyError, FileNotFoundError, ValueError):
        raise HTTPException(404, "job not found")
    if job.get("kind") != "rgbd_depth":
        raise HTTPException(409, "only LiDAR depth meshes are sent this way (RGB demo meshes appear automatically)")
    if job.get("status") != "complete" or not job.get("run_dir") or not (Path(job["run_dir"]) / "fused_mesh.ply").is_file():
        raise HTTPException(409, "this job has no finished mesh")
    return job


def _summary(job: dict) -> dict:
    run = Path(job["run_dir"])
    disp = json.loads((run / "avatar_display.json").read_text(encoding="utf-8"))
    return {"id": job["id"], "upload_id": job["upload_id"], "label": job.get("label", ""),
            "created_utc": job["created_utc"], "kind": "rgbd_depth", "run_status": disp.get("run_status"),
            "run_status_note": disp.get("run_status_note"), "display_only": True,
            "note": "LiDAR depth mesh, display only: nothing on the avatar is measured from it"}


def make_avatar_router(jobs) -> APIRouter:
    r = APIRouter()

    @r.post("/api/jobs/{job_id}/avatar")
    def prepare(job_id: str):
        from ..experimental.depth_avatar import prepare_depth_avatar

        job = _depth_job(jobs, job_id)
        try:
            prepare_depth_avatar(job["run_dir"])
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(409, f"cannot prepare this mesh for the avatar: {e}")
        return _summary(job)

    @r.get("/api/jobs/{job_id}/avatar/{name}")
    def avatar_file(job_id: str, name: str):
        if name not in AVATAR_FILES:
            raise HTTPException(404, "unknown file")
        job = _depth_job(jobs, job_id)
        p = Path(job["run_dir"]) / name
        if not p.is_file():
            raise HTTPException(404, "not prepared yet: send the mesh to the avatar first")
        return FileResponse(p, media_type=AVATAR_FILES[name])

    @r.get("/api/avatar/scans")
    def scans():
        out = []
        for job in jobs.list():
            run = Path(job.get("run_dir") or "")
            if job.get("kind") == "rgbd_depth" and job.get("status") == "complete" and \
                    (run / "mesh_regions.glb").is_file() and (run / "avatar_display.json").is_file():
                out.append(_summary(job))
        return sorted(out, key=lambda s: s["created_utc"], reverse=True)

    return r
