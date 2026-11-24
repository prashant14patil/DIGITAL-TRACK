"""Web processing jobs: finalized upload -> ingest (real) -> mesh step (stubbed here; the real experimental
visual hull takes minutes and is exercised manually on a real recording)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from digitaltrack.api.app import create_app
from digitaltrack.api.jobs import JobManager
from digitaltrack.dataset.validate import validate_session

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not on PATH")

META = {"recording_type": "other", "label": "person turning, demo", "contains_person": True,
        "consent_confirmed": True, "subject_id": "S001", "capture_app": "Camera", "device_model": "iPad (test)"}


def _video(path: Path, n=12) -> bytes:
    rng = np.random.default_rng(0)
    frames = b"".join(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8).tobytes() for _ in range(n))
    subprocess.run([FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48", "-r", "30",
                    "-i", "pipe:0", "-c:v", "png", str(path)], input=frames, check=True, capture_output=True)
    return path.read_bytes()


def fake_runner(session_dir, *, vote_fraction, progress):
    progress("segmenting", 0.5)
    out = Path(session_dir) / "processing" / "demo_vhull_TEST"
    out.mkdir(parents=True)
    (out / "mesh_height_normalised.glb").write_bytes(b"glTF-fake")
    (out / "preview_mesh.png").write_bytes(b"\x89PNG-fake")
    report = {"status": "unreliable", "status_reasons": ["experimental demo"], "measurements": "none computed",
              "mesh": {"triangles": 10}, "angles": {"anchors": []}, "carving": {"frames": 3}, "runtime_s": 0.1,
              "vote_fraction_used": vote_fraction}
    (out / "report.json").write_text(json.dumps(report))
    progress("done", 1.0)
    return out, report


def failing_runner(session_dir, *, vote_fraction, progress):
    raise RuntimeError("leg-width cycle does not show one full turn")


def _client(tmp_path, runner=fake_runner):
    return TestClient(create_app(tmp_path / "uploads", serve_frontend=False, job_runner=runner, jobs_inline=True))


def _finalized_upload(c, tmp_path, files: dict[str, bytes]) -> str:
    uid = c.post("/api/uploads", json=META).json()["id"]
    for name, data in files.items():
        assert c.put(f"/api/uploads/{uid}/files", params={"path": name}, content=data).status_code == 200
    r = c.post(f"/api/uploads/{uid}/finalize", json={"files": [{"path": n, "size": len(d)} for n, d in files.items()]})
    assert r.status_code == 200, r.text
    return uid


def test_process_creates_valid_session_and_serves_results(tmp_path):
    c = _client(tmp_path)
    uid = _finalized_upload(c, tmp_path, {"rec/IMG_0001.MOV": _video(tmp_path / "v.mov")})
    r = c.post(f"/api/uploads/{uid}/process", json={})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "complete" and job["progress"] == 1.0 and job["error"] is None
    assert job["result"]["status"] == "unreliable" and job["result"]["frame_step"] == 3  # 30 fps -> ~10 fps
    assert set(job["result"]["files"]) == {"mesh.glb", "preview_mesh.png", "report.json"}
    session = tmp_path / "sessions" / job["session_id"]
    assert validate_session(session).ok  # real ingest ran; upload only read
    assert c.get(f"/api/jobs/{job['id']}/files/mesh.glb").content == b"glTF-fake"
    assert c.get(f"/api/jobs/{job['id']}/files/report.json").json()["vote_fraction_used"] == 0.8
    assert c.get("/api/jobs").json()[0]["id"] == job["id"]
    assert c.get(f"/api/jobs/{job['id']}").json()["status"] == "complete"


def test_only_whitelisted_files_are_served(tmp_path):
    c = _client(tmp_path)
    uid = _finalized_upload(c, tmp_path, {"a.mov": _video(tmp_path / "v.mov")})
    job = c.post(f"/api/uploads/{uid}/process").json()
    for name in ("session.json", "..%2F..%2Fsession.json", "mesh.ply", "preview_silhouettes.png"):
        assert c.get(f"/api/jobs/{job['id']}/files/{name}").status_code == 404, name
    assert c.get("/api/jobs/job_nope/files/mesh.glb").status_code == 404
    assert c.get("/api/jobs/not-a-job").status_code == 404


def test_refusals_are_clear(tmp_path):
    c = _client(tmp_path)
    assert c.post("/api/uploads/up_20260101T000000Z_abcdef/process").status_code == 404
    uid = _finalized_upload(c, tmp_path, {"notes.txt": b"hello"})
    r = c.post(f"/api/uploads/{uid}/process")
    assert r.status_code == 409 and "no .mov" in r.json()["detail"]
    v = _video(tmp_path / "v.mov")
    uid2 = _finalized_upload(c, tmp_path, {"a.mov": v, "b.mov": v})
    r = c.post(f"/api/uploads/{uid2}/process")
    assert r.status_code == 409 and "choose one" in r.json()["detail"]
    assert c.post(f"/api/uploads/{uid2}/process", json={"file": "c.mov"}).status_code == 409
    assert c.post(f"/api/uploads/{uid2}/process", json={"file": "b.mov"}).json()["status"] == "complete"
    pending = c.post("/api/uploads", json=META).json()["id"]  # never finalized
    assert c.post(f"/api/uploads/{pending}/process").status_code == 409


def test_failures_are_recorded_not_raised(tmp_path):
    c = _client(tmp_path, runner=failing_runner)
    uid = _finalized_upload(c, tmp_path, {"a.mov": _video(tmp_path / "v.mov")})
    job = c.post(f"/api/uploads/{uid}/process").json()
    assert job["status"] == "failed" and "full turn" in job["error"]
    assert c.get(f"/api/jobs/{job['id']}/files/mesh.glb").status_code == 404


def test_tampered_upload_fails_integrity_check(tmp_path):
    import os
    import stat
    c = _client(tmp_path)
    uid = _finalized_upload(c, tmp_path, {"a.mov": _video(tmp_path / "v.mov")})
    f = tmp_path / "uploads" / uid / "export" / "a.mov"
    os.chmod(f, stat.S_IWRITE | stat.S_IREAD)
    f.write_bytes(f.read_bytes()[:-1] + b"\x01")
    job = c.post(f"/api/uploads/{uid}/process").json()
    assert job["status"] == "failed" and "checksum_mismatch" in job["error"]


def test_interrupted_jobs_are_marked_failed_on_restart(tmp_path):
    c = _client(tmp_path)
    uid = _finalized_upload(c, tmp_path, {"a.mov": _video(tmp_path / "v.mov")})
    job = c.post(f"/api/uploads/{uid}/process").json()
    p = tmp_path / "jobs" / job["id"] / "job.json"
    j = json.loads(p.read_text())
    j.update(status="running")
    p.write_text(json.dumps(j))
    mgr = JobManager(c.app.state.store, tmp_path / "jobs", tmp_path / "sessions", inline=True)
    after = mgr.get(job["id"])
    assert after["status"] == "failed" and after["error"].startswith("interrupted")


def figure_runner(session_dir, *, vote_fraction, progress):
    """Writes a real (synthetic stick-figure) mesh so the region step runs for real."""
    import open3d as o3d
    import trimesh

    from test_body_regions import _cyl

    out = Path(session_dir) / "processing" / "demo_vhull_FIG"
    out.mkdir(parents=True)
    parts = [_cyl([0.08, 0, 0], [0.08, 0, 0.47], 0.05), _cyl([-0.08, 0, 0], [-0.08, 0, 0.47], 0.05),
             _cyl([0, 0, 0.46], [0, 0, 0.82], 0.13), _cyl([0, 0, 0.87], [0, 0, 1.0], 0.07),
             _cyl([0.16, 0, 0.78], [0.38, 0, 0.5], 0.03), _cyl([-0.16, 0, 0.78], [-0.38, 0, 0.5], 0.03)]
    tm = trimesh.util.concatenate([trimesh.Trimesh(v, f) for v, f in parts])
    o3d.io.write_triangle_mesh(str(out / "mesh_height_normalised.ply"), o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces)))
    report = {"status": "unreliable", "status_reasons": [], "mesh": {}, "angles": {"anchors": []}, "carving": {}}
    (out / "report.json").write_text(json.dumps(report))
    return out, report


def test_completed_jobs_get_display_regions(tmp_path):
    c = _client(tmp_path, runner=figure_runner)
    uid = _finalized_upload(c, tmp_path, {"a.mov": _video(tmp_path / "v.mov")})
    job = c.post(f"/api/uploads/{uid}/process").json()
    assert job["status"] == "complete"
    assert {"mesh_regions.glb", "regions.json"} <= set(job["result"]["files"])
    reg = c.get(f"/api/jobs/{job['id']}/files/regions.json").json()
    assert reg["status"] == "provisional" and reg["crotch_method"] == "detected leg gap"
    assert c.get(f"/api/jobs/{job['id']}/files/mesh_regions.glb").content[:4] == b"glTF"
    # backfill: remove the region files, the manager regenerates them
    (Path(job["run_dir"]) / "regions.json").unlink()
    (Path(job["run_dir"]) / "mesh_regions.glb").unlink()
    assert c.app.state.jobs.backfill_regions() == 1
    assert c.get(f"/api/jobs/{job['id']}/files/regions.json").status_code == 200
