"""End to end: SYNTHETIC textured Stray-format recording -> poses, mesh, exports and report, against exact truth.

Checks the plumbing and the conventions. It says nothing about accuracy on real iPad data.
"""

import hashlib
import json

import numpy as np
import pytest

from digitaltrack.cli import main
from digitaltrack.rgbd.pose import wrap
from digitaltrack.rgbd.textured.pipeline import ReconParams, reconstruct
from digitaltrack.synthetic.stray_scene import StraySceneConfig, make_stray_recording


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("textured")
    truth = make_stray_recording(root / "rec", StraySceneConfig())
    before = _snapshot(root / "rec")
    result = reconstruct(root / "rec", root / "out", run_id="r1",
                         params=ReconParams(sensitivity_intervals_s=(0.75,)))
    return root, truth, result, before


def test_pipeline_produces_mesh_and_does_not_touch_the_recording(run):
    root, _, result, before = run
    assert result.mesh is not None, result.report.get("status_note")
    assert result.status in ("pass", "warn"), [c for c in result.report["checks"] if c["status"] == "fail"]
    assert _snapshot(root / "rec") == before
    for name in ("mesh.ply", "mesh.obj", "mesh.glb", "mesh_display.glb", "pointcloud_registered.ply",
                 "quality_report.json", "report.html"):
        assert (result.run_dir / name).stat().st_size > 0, name
    with pytest.raises(FileExistsError):
        reconstruct(root / "rec", root / "out", run_id="r1")


def test_poses_match_truth(run):
    _, truth, result, _ = run
    keys = np.array(truth["key_frames"])
    assert len(result.pause_frames) >= len(keys) - 1
    nearest = [int(np.argmin(np.abs(keys - f))) for f in result.pause_frames]
    assert nearest[0] == 0
    # body frames differ only by where the floor origin was put: compare motion relative to the first view
    g = result.floor_from_camera @ np.linalg.inv(truth["floor_from_camera"])  # true floor -> estimated floor
    for est, k in zip(result.floor_from_body, nearest):
        true = g @ truth["floor_from_body"][k] @ np.linalg.inv(truth["floor_from_body"][0]) @ np.linalg.inv(g)
        d_theta = np.degrees(abs(float(wrap(np.arctan2(est[1, 0], est[0, 0]) - np.arctan2(true[1, 0], true[0, 0])))))
        point = np.linalg.inv(true) @ (g @ [0.0, 0.0, 0.8, 1.0])  # a point on the body axis
        assert d_theta < 1.5, (k, d_theta)
        assert np.linalg.norm(((est - true) @ point)[:3]) < 0.01, k


def test_mesh_keeps_metric_size(run):
    _, truth, result, _ = run
    cfg = truth["config"]
    v = np.asarray(result.mesh.vertices)
    assert v[:, 2].max() == pytest.approx(cfg.height_m, abs=0.02)
    # a band without side bars: the elliptic cross-section, measured about its own centre
    band = v[(v[:, 2] > 0.3) & (v[:, 2] < 0.6)][:, :2]
    centred = band - (band.max(0) + band.min(0)) / 2
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    extent = np.ptp(centred @ axes.T, axis=0)
    assert extent[0] == pytest.approx(2 * cfg.semi_x_m, abs=0.02)
    assert extent[1] == pytest.approx(2 * cfg.semi_y_m, abs=0.02)


def test_quality_report_content(run):
    _, _, result, _ = run
    rep = json.loads((result.run_dir / "quality_report.json").read_text(encoding="utf-8"))
    checks = {c["name"]: c for c in rep["checks"]}
    assert checks["rgb_depth_sync_lag"]["value"] in (0, None)
    assert checks["exports_verified"]["status"] == "pass"
    assert checks["camera_static"]["status"] in ("pass", "warn")
    assert rep["gravity"]["imu"]["disagreement_deg"] < 0.1
    assert rep["mesh_result_status"] == "estimated"
    assert rep["mesh"]["refinement"]["smoothing"] == "none"
    assert "NOT validated" in rep["note"]
    assert rep["pose"]["sensitivity_to_interval"]["0.75"]["compared"] >= 10
    assert "comparison_depth_only_poses" in rep and "measurements" not in rep
    for name, digest in rep["outputs_sha256"].items():
        assert hashlib.sha256((result.run_dir / name).read_bytes()).hexdigest() == digest


def test_cli_reports_a_bad_export(tmp_path, capsys):
    (tmp_path / "bad").mkdir()
    assert main(["reconstruct-stray", str(tmp_path / "bad"), "--out", str(tmp_path / "out")]) == 1
    assert "cannot reconstruct" in capsys.readouterr().out
