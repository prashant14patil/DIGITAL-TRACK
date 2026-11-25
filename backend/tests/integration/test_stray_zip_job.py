"""Stray Scanner zip -> upload -> depth job -> mesh, on a SYNTHETIC scene written in the published Stray format.

This checks the plumbing and the format checks only. It says nothing about accuracy on real iPad data.
"""

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from digitaltrack.api.app import create_app
from digitaltrack.rgbd.stray import StrayFormatError, load_stray
from digitaltrack.synthetic.rgbd_scene import SceneConfig, make_sequence


def write_stray(folder: Path, views, rgb_scale: int = 2, fps: float = 6.0) -> None:
    (folder / "depth").mkdir(parents=True)
    (folder / "confidence").mkdir()
    k = views[0].intrinsics
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
    for i, v in enumerate(views):
        mm = np.nan_to_num(v.depth_m * 1000.0).round().clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(folder / "depth" / f"{i:06d}.png"), mm)
        cv2.imwrite(str(folder / "confidence" / f"{i:06d}.png"), np.full(mm.shape, 2, np.uint8))
        s = rgb_scale
        rows.append(f"{i / fps:.4f}, {i:06d}, 0, 0, 0, 0, 0, 0, 1, {k.fx * s}, {k.fy * s}, {k.cx * s}, {k.cy * s}")
    (folder / "odometry.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    w, h = k.width * rgb_scale, k.height * rgb_scale
    vw = cv2.VideoWriter(str(folder / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(3):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()


@pytest.fixture(scope="module")
def seq():
    return make_sequence(SceneConfig(n_stops=24, step_deg=15.0, noise_sigma_m=0.002, frames_per_stop=8,
                                     transition_frames=4, seed=1))


def test_load_stray_converts_units_and_scales_intrinsics(tmp_path, seq):
    write_stray(tmp_path / "rec", seq.views[:6])
    rec = load_stray(tmp_path / "rec")
    v0, src = rec.views[0], seq.views[0]
    assert len(rec.views) == 6
    assert rec.views[0].intrinsics == src.intrinsics  # rgb 2x larger -> scaled back to the depth size
    m = src.depth_m > 0
    assert np.abs(v0.depth_m[m] - src.depth_m[m]).max() < 0.0006  # mm rounding only
    assert rec.info["ipad_moved_m"] == 0.0


def test_load_stray_rejects_wrong_units(tmp_path, seq):
    write_stray(tmp_path / "rec", seq.views[:3])
    p = tmp_path / "rec" / "depth" / "000001.png"
    cv2.imwrite(str(p), np.zeros((192, 256), np.uint8))  # 8-bit, not 16-bit mm
    with pytest.raises(StrayFormatError, match="uint16"):
        load_stray(tmp_path / "rec")


def _zip_upload(c, tmp_path: Path, name: str, views) -> tuple[str, bytes]:
    write_stray(tmp_path / name / "a1b2c3d4", views)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in (tmp_path / name).rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(tmp_path / name).as_posix())
    payload = buf.getvalue()
    uid = c.post("/api/uploads", json={"recording_type": "other", "label": name,
                                       "contains_person": False}).json()["id"]
    assert c.put(f"/api/uploads/{uid}/files", params={"path": "scan.zip"}, content=payload).status_code == 200
    assert c.post(f"/api/uploads/{uid}/finalize", json={"files": [{"path": "scan.zip", "size": len(payload)}]}).status_code == 200
    return uid, payload


@pytest.mark.parametrize("with_background", [True, False])
def test_zip_upload_to_depth_mesh(tmp_path, seq, with_background):
    c = TestClient(create_app(tmp_path / "data" / "uploads", serve_frontend=False, jobs_inline=True))
    uid, payload = _zip_upload(c, tmp_path, "subject", seq.views)
    body = {}
    if with_background:
        body["background_upload_id"] = _zip_upload(c, tmp_path, "empty", seq.background_views)[0]

    job = c.post(f"/api/uploads/{uid}/process", json=body).json()
    assert job["kind"] == "rgbd_depth"
    job = c.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "complete", job["error"]
    r = job["result"]
    assert "mesh.glb" in r["files"] and r["mode"].startswith("pauses")
    assert r["n_views"] >= 20
    assert 0.45 < r["mesh"]["height_extent_m"] < 0.55  # synthetic box is 0.50 m tall (floor band removed)
    assert ("no background plate" in " ".join(r["warnings"])) != with_background
    assert c.get(f"/api/jobs/{job['id']}/files/mesh.glb").status_code == 200
    assert c.get(f"/api/jobs/{job['id']}/files/mesh_render.png").status_code == 200
    assert c.get(f"/api/jobs/{job['id']}/files/..%2Fsecret.png").status_code == 404
    assert (tmp_path / "data" / "uploads" / uid / "export" / "scan.zip").read_bytes() == payload


def test_continuous_turn_without_pauses_falls_back_to_spaced_frames(tmp_path):
    cont = make_sequence(SceneConfig(n_stops=36, step_deg=10.0, noise_sigma_m=0.002, frames_per_stop=1, seed=2))
    c = TestClient(create_app(tmp_path / "data" / "uploads", serve_frontend=False, jobs_inline=True))
    uid, _ = _zip_upload(c, tmp_path, "subject", cont.views)
    bg = _zip_upload(c, tmp_path, "empty", cont.background_views)[0]
    job = c.post(f"/api/uploads/{uid}/process", json={"background_upload_id": bg}).json()
    job = c.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "complete", job["error"]
    assert job["result"]["mode"].startswith("continuous")
