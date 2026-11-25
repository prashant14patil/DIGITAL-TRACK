"""API wiring of the quick recording check and the mesh-vs-tape comparison (error paths + a synthetic job)."""

from __future__ import annotations

import asyncio
import json

import open3d as o3d
import pytest
from fastapi.testclient import TestClient

from digitaltrack.api.app import create_app


async def _stream(b):
    yield b


@pytest.fixture()
def app(tmp_path):
    return create_app(tmp_path / "uploads", serve_frontend=False, jobs_inline=True)


def _upload(app, files, meta=None):
    store = app.state.store
    uid = store.create(meta or {"recording_type": "R2", "label": "t", "contains_person": False,
                                "object_kind": "box", "measurements": []})
    for p, d in files.items():
        asyncio.run(store.write_file(uid, p, _stream(d), 10**9))
    store.finalize(uid, [{"path": p, "size": len(d)} for p, d in files.items()])
    return uid


def test_quickcheck_errors(app):
    c = TestClient(app)
    assert c.post("/api/uploads/up_20260101T000000Z_abcdef/quickcheck").status_code == 404
    uid = _upload(app, {"a.txt": b"x"})
    r = c.post(f"/api/uploads/{uid}/quickcheck")
    assert r.status_code == 409 and ".zip" in r.json()["detail"]


def test_level_mm_is_accepted_and_validated(app):
    c = TestClient(app)
    base = {"recording_type": "R2", "label": "t", "contains_person": False, "object_kind": "box"}
    ok = {**base, "measurements": [{"name": "waist", "readings_mm": [900], "level_mm": 1050}]}
    assert c.post("/api/uploads", json=ok).status_code == 201
    bad = {**base, "measurements": [{"name": "waist", "readings_mm": [900], "level_mm": -5}]}
    assert c.post("/api/uploads", json=bad).status_code == 422


def test_comparison_for_a_depth_job(app, tmp_path):
    meta = {"recording_type": "R2", "label": "box", "contains_person": False, "object_kind": "box",
            "measurements": [{"name": "height", "readings_mm": [500, 501, 499]}]}
    uid = _upload(app, {"rec.zip": b"PK"}, meta)
    run = tmp_path / "run"
    run.mkdir()
    o3d.io.write_triangle_mesh(str(run / "fused_mesh.ply"),
                               o3d.geometry.TriangleMesh.create_box(0.4, 0.3, 0.5).subdivide_midpoint(3))
    (run / "run_log.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    jobs = app.state.jobs
    job = {"id": "job_20260101T000000Z_abcdef", "upload_id": uid, "file": "rec.zip", "kind": "rgbd_depth",
           "status": "complete", "run_dir": str(run)}
    jobs._path(job["id"]).parent.mkdir(parents=True, exist_ok=True)
    jobs._save(job)
    c = TestClient(app)
    r = c.get(f"/api/jobs/{job['id']}/comparison")
    assert r.status_code == 200, r.text
    row = r.json()["rows"][0]
    assert row["status"] == "measured" and abs(row["error_mm"]) < 3
    assert c.get("/api/jobs/job_20260101T000000Z_000000/comparison").status_code == 404
