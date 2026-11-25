"""Sending a depth mesh to the Body avatar: display copy orientation, raw mesh untouched, API wiring.

Synthetic figure in a body frame where the camera stands at +X looking toward -X, so the subject (facing the
camera) faces +X and its LEFT is +Y. It has a left arm only, so a left/right mix-up is visible."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import open3d as o3d
import pytest
import trimesh
from fastapi.testclient import TestClient

from digitaltrack.api.app import create_app
from digitaltrack.experimental.body_regions import ID
from digitaltrack.experimental.depth_avatar import prepare_depth_avatar


def _box(size, centre):
    return o3d.geometry.TriangleMesh.create_box(*size).translate(np.asarray(centre) - np.asarray(size) / 2)


def _figure():
    m = _box((0.22, 0.34, 0.60), (0, 0, 1.15))  # trunk (depth along x, width along y)
    m += _box((0.18, 0.16, 0.22), (0, 0, 1.58))  # head
    m += _box((0.14, 0.13, 0.82), (0, 0.09, 0.42))  # leg on +y (subject's left)
    m += _box((0.14, 0.13, 0.82), (0, -0.09, 0.42))  # leg on -y
    m += _box((0.10, 0.45, 0.10), (0, 0.38, 1.12))  # LEFT arm only, out on +y, below the neck search band
    return m.subdivide_midpoint(2)


def _floor_from_camera():
    t = np.eye(4)  # camera axes (right, down, forward) expressed in the body frame
    t[:3, :3] = np.column_stack([(0, 1, 0), (0, 0, -1), (-1, 0, 0)])
    t[:3, 3] = (2.5, 0, 1.2)
    return t


def _run(tmp_path, status="warn"):
    d = tmp_path / "run"
    d.mkdir()
    o3d.io.write_triangle_mesh(str(d / "fused_mesh.ply"), _figure())
    (d / "run_log.json").write_text(json.dumps({"status": status, "status_note": "test",
                                                "floor_from_camera": _floor_from_camera().tolist()}), encoding="utf-8")
    return d


def test_display_copy_faces_minus_y_with_left_on_plus_x(tmp_path):
    run = _run(tmp_path)
    before = hashlib.sha256((run / "fused_mesh.ply").read_bytes()).hexdigest()
    out = prepare_depth_avatar(run)
    assert hashlib.sha256((run / "fused_mesh.ply").read_bytes()).hexdigest() == before  # metric mesh untouched
    d = out["display"]
    t = np.array(d["transform_body_to_display"])
    cam = t @ np.r_[d["camera_position_body_m"], 1.0]
    assert cam[1] < -2 and abs(cam[0]) < 0.05  # camera in front of the subject, on -Y
    assert np.allclose(d["forward_after"], [0, 1, 0], atol=1e-9) and np.allclose(d["camera_right_after"], [1, 0, 0], atol=1e-9)
    assert abs(d["yaw_deg"]) == pytest.approx(90.0)
    m = trimesh.load(run / "mesh_regions.glb", force="mesh")
    v, ids = np.asarray(m.vertices), (np.asarray(m.visual.vertex_colors)[:, 0].astype(int) + 20) // 40
    arm = v[:, 0] > 0.3  # far out on +X at arm height: the left arm after the turn
    assert arm.sum() > 20 and np.mean(ids[arm] == ID["left_arm"]) > 0.9
    assert not np.any(ids[arm] == ID["right_arm"])  # a mirrored turn would put the left arm on the right
    rec = json.loads((run / "avatar_display.json").read_text(encoding="utf-8"))
    assert rec["run_status"] == "warn" and "nothing is measured" in rec["units"]


def test_no_floor_frame_refuses(tmp_path):
    run = _run(tmp_path)
    (run / "run_log.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    with pytest.raises(ValueError, match="floor_from_camera"):
        prepare_depth_avatar(run)


def test_api_send_list_and_serve(tmp_path):
    app = create_app(tmp_path / "uploads", serve_frontend=False, jobs_inline=True)
    run = _run(tmp_path)
    jobs = app.state.jobs
    job = {"id": "job_20260101T000000Z_abcdef", "upload_id": "up_x", "file": "rec.zip", "kind": "rgbd_depth",
           "status": "complete", "run_dir": str(run), "created_utc": "2026-01-01T00:00:00Z", "label": "scan"}
    jobs._path(job["id"]).parent.mkdir(parents=True, exist_ok=True)
    jobs._save(job)
    c = TestClient(app)
    assert c.get("/api/avatar/scans").json() == []
    assert c.get(f"/api/jobs/{job['id']}/avatar/regions.json").status_code == 404  # not prepared yet
    r = c.post(f"/api/jobs/{job['id']}/avatar")
    assert r.status_code == 200 and r.json()["run_status"] == "warn" and r.json()["display_only"]
    assert [s["id"] for s in c.get("/api/avatar/scans").json()] == [job["id"]]
    assert c.get(f"/api/jobs/{job['id']}/avatar/mesh_regions.glb").status_code == 200
    assert c.get(f"/api/jobs/{job['id']}/avatar/fused_mesh.ply").status_code == 404  # only avatar files
    rgb = {**job, "id": "job_20260101T000000Z_bbbbbb", "kind": "rgb_visual_hull_demo"}
    jobs._save(rgb)
    assert c.post(f"/api/jobs/{rgb['id']}/avatar").status_code == 409
