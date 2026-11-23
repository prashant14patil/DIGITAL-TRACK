"""TSDF reconstruction checked against analytic ground truth on the VIRTUAL sensor.

IMPORTANT: thresholds below are REGRESSION BOUNDS for the synthetic pipeline, chosen from
observed behaviour with margin. They are not accuracy claims for any real capture or real bodies.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.schemas import schema_errors
from digitaltrack.dataset.validate import validate_session
from digitaltrack.dataset.writer import sha256_file
from digitaltrack.geometry.transforms import camera_from_body
from digitaltrack.measurement.basic import circumference_profile, measurements_document, o3d_to_trimesh
from digitaltrack.reconstruction.tsdf import TSDFParams, reconstruct_tsdf
from digitaltrack.synthetic.evaluate import evaluate_against_truth
from digitaltrack.synthetic.renderer import RenderConfig, render_session
from digitaltrack.synthetic.shapes import cylinder, elliptic_cylinder, procedural_torso

from conftest import make_writable

pytestmark = pytest.mark.slow
VOXEL = 0.004


@pytest.fixture(scope="module")
def cyl(full_cylinder_session, cylinder_body, tmp_path_factory):
    out = tmp_path_factory.mktemp("cyl_run") / "run"
    res = reconstruct_tsdf(load_session(full_cylinder_session), TSDFParams(), run_id="test_run", out_dir=out)
    return res, evaluate_against_truth(res.mesh, cylinder_body), out


def _reconstruct_new(body, tmp_path, cfg=None, params=None):
    root = render_session(body, tmp_path / "sess", cfg or RenderConfig())
    assert validate_session(root).ok
    res = reconstruct_tsdf(load_session(root), params or TSDFParams(), write=False)
    return res, evaluate_against_truth(res.mesh, body)


# ---- known-dimension checks ---------------------------------------------------------------------

def test_cylinder_circumference_and_height_against_truth(cyl):
    res, ev, _ = cyl
    assert ev["n_slices_failed"] == 0
    assert all(r["slice_status"] == "pass" for r in ev["rows"])  # exactly one closed contour per slice
    assert ev["circumference_max_abs_error_pct"] < 1.5
    # the flat top/bottom caps are never observed by a level camera, so the mesh may stop up to ~1 voxel short
    assert abs(ev["height_error_m"]) <= 2 * VOXEL
    assert res.quality["passed"]


def test_cylinder_surface_distance_to_true_radius(cyl):
    res, _, _ = cyl
    v = np.asarray(res.mesh.vertices)
    side = v[(v[:, 2] > 0.03) & (v[:, 2] < 0.57)]
    radial = np.hypot(side[:, 0], side[:, 1]) - 0.15
    assert abs(radial.mean()) < 0.002  # < 2 mm mean radial bias
    assert radial.std() < 0.001


def test_elliptic_cylinder_against_truth(tmp_path):
    _, ev = _reconstruct_new(elliptic_cylinder(0.18, 0.11, 0.6), tmp_path)
    assert ev["n_slices_failed"] == 0 and ev["circumference_max_abs_error_pct"] < 1.5
    assert abs(ev["height_error_m"]) <= 2 * VOXEL


def test_procedural_torso_against_truth(tmp_path):
    body = procedural_torso()
    _, ev = _reconstruct_new(body, tmp_path)
    assert ev["n_slices_failed"] == 0 and ev["circumference_max_abs_error_pct"] < 2.0
    assert abs(ev["height_error_m"]) <= 2 * VOXEL
    # the reconstruction must preserve the waist being narrower than hips/chest
    by_z = {round(r["z_m"], 3): r["measured_circumference_m"] for r in ev["rows"]}
    assert min(by_z.values()) < max(by_z.values()) - 0.05


def test_noisy_scan_with_missing_depth_stays_within_regression_bound(tmp_path):
    cfg = RenderConfig(noise_sigma_m=0.003, dropout_prob=0.05, seed=1)
    _, ev = _reconstruct_new(procedural_torso(), tmp_path, cfg)
    assert ev["n_slices_failed"] == 0 and ev["circumference_max_abs_error_pct"] < 3.0


def test_high_confidence_only_reduces_grazing_angle_bias(tmp_path):
    body = cylinder(0.15, 0.6)
    root = render_session(body, tmp_path / "s", RenderConfig())
    s = load_session(root)
    lo = evaluate_against_truth(reconstruct_tsdf(s, TSDFParams(min_confidence=1), write=False).mesh, body)
    hi = evaluate_against_truth(reconstruct_tsdf(s, TSDFParams(min_confidence=2), write=False).mesh, body)
    assert 0 < hi["circumference_bias_m"] < lo["circumference_bias_m"]


# ---- sensitivity: the pipeline must expose, not hide, calibration problems -----------------------

def test_axis_centre_error_produces_large_measurement_error(tmp_path):
    cfg = RenderConfig(calibration_center_offset_m=(0.01, 0.0, 0.0))
    _, ev = _reconstruct_new(procedural_torso(), tmp_path, cfg)
    assert ev["circumference_max_abs_error_pct"] > 2.5  # 1 cm axis error => several % circumference error


def test_wrong_turntable_direction_sign_is_catastrophic(tmp_path):
    from dataclasses import replace
    body = cylinder(0.12, 0.6).with_center(0.06, -0.03)
    root = render_session(body, tmp_path / "s", RenderConfig())
    s = load_session(root)
    flipped = replace(s, turntable={**s.turntable, "angle_direction_sign": -1})
    good = evaluate_against_truth(reconstruct_tsdf(s, write=False).mesh, body)
    bad = evaluate_against_truth(reconstruct_tsdf(flipped, write=False).mesh, body)
    assert good["circumference_max_abs_error_pct"] < 2.0
    assert bad["n_slices_failed"] > 0 or bad["circumference_max_abs_error_pct"] > 10.0


def test_quality_gate_flags_sparse_angular_coverage(tmp_path):
    cfg = RenderConfig(n_views=12, total_rotation_deg=120.0)  # frames cover only 120 of 360 degrees
    res, _ = _reconstruct_new(cylinder(0.15, 0.6), tmp_path, cfg)
    assert res.quality["passed"] is False
    gap = next(c for c in res.quality["checks"] if c["name"] == "angular_gap")
    assert gap["status"] == "fail" and gap["value"] == pytest.approx(250.0)


def test_quality_gate_detects_bad_turntable_calibration(tmp_path):
    # CHANGED in QC v0.2: this was a strict xfail ("QC cannot detect calibration errors"). It began
    # XPASS-ing once the view-consistency check existed, so it is now a normal test. The lateral
    # (x) centre error used here is detectable; line-of-sight errors are not (test_calibration_qc.py).
    cfg = RenderConfig(calibration_center_offset_m=(0.01, 0.0, 0.0))
    res, _ = _reconstruct_new(procedural_torso(), tmp_path, cfg)
    assert res.quality["passed"] is False


# ---- auditability and raw-data immutability ------------------------------------------------------

def _hash_tree(root: Path, exclude: str = "processing") -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and exclude not in p.relative_to(root).parts}


def test_reconstruction_leaves_raw_data_untouched(full_cylinder_session, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(full_cylinder_session, dst)
    before = _hash_tree(dst)
    res = reconstruct_tsdf(load_session(dst), TSDFParams(), run_id="immut")
    assert res.run_dir == dst / "processing" / "immut"
    assert _hash_tree(dst) == before
    rep = validate_session(dst)
    assert rep.ok and rep.issues == []
    with pytest.raises(FileExistsError):
        reconstruct_tsdf(load_session(dst), TSDFParams(), run_id="immut")  # runs are never overwritten
    make_writable(dst)


def test_run_log_records_every_parameter_and_transform(cyl, full_cylinder_session):
    res, _, out = cyl
    log = json.loads((out / "run_log.json").read_text())
    assert schema_errors(log, "processing_log") == []
    assert log["parameters"] == asdict(TSDFParams())
    s = load_session(full_cylinder_session)
    assert len(log["transforms"]["per_frame"]) == len(s.frames) == 36
    fr5 = log["transforms"]["per_frame"][4]
    expected = camera_from_body(s.camera_from_turntable, s.frames[4].angle_deg, s.angle_direction_sign)
    assert np.allclose(fr5["camera_from_body"], expected)
    assert fr5["recorded_angle_deg"] == pytest.approx(s.frames[4].angle_deg)
    assert log["input"]["session_json_sha256"] == sha256_file(full_cylinder_session / "session.json")
    assert log["input"]["calibration_id"] == s.calibration_id
    assert {st["name"] for st in log["steps"]} >= {"depth_filter", "tsdf_integration", "remove_small_components",
                                                    "fused_pointcloud_downsample"}
    for name, digest in log["outputs"].items():
        assert sha256_file(out / name) == digest
    assert {"open3d", "numpy", "python", "digitaltrack"} <= set(log["software"])
    px = next(st for st in log["steps"] if st["name"] == "depth_filter")
    assert 0 < px["pixels_after"] < px["pixels_before"]


def test_quality_report_is_valid_and_reports_unavailable_metrics_as_null(cyl):
    # CHANGED in QC v0.2: registration_residual_m used to be asserted null (not computed in v0.1).
    # The view-consistency check now computes it, so it must be a real, small number.
    res, _, out = cyl
    qr = json.loads((out / "quality_report.json").read_text())
    assert schema_errors(qr, "quality_report") == []
    m = qr["metrics"]
    assert m["valid_frames"] == 36 and m["max_angular_gap_deg"] == pytest.approx(10.0)
    assert m["unobserved_region_fraction"] is None and m["outlier_fraction"] is None  # still not computed
    assert 0 < m["registration_residual_m"] < 0.003
    assert m["view_consistency"]["applied_to_geometry"] is False
    assert len(m["view_consistency"]["per_frame"]) == 36
    assert m["mesh_components"] == 1


def test_saved_point_cloud_and_mesh_are_nonempty(cyl):
    _, _, out = cyl
    assert (out / "fused_mesh.ply").stat().st_size > 1000 and (out / "pointcloud_filtered.ply").stat().st_size > 1000


def test_measurements_document_validates_and_marks_failed_scans(cyl):
    res, ev, _ = cyl
    slices = circumference_profile(o3d_to_trimesh(res.mesh), [r["z_m"] for r in ev["rows"]])
    ok_doc = measurements_document("s", "r", slices, quality_passed=True)
    assert schema_errors(ok_doc, "measurements") == []
    assert {m["quality_status"] for m in ok_doc["measurements"]} == {"pass"}
    assert all(m["source"] == "measured" for m in ok_doc["measurements"])
    bad_doc = measurements_document("s", "r", slices, quality_passed=False)
    assert {m["quality_status"] for m in bad_doc["measurements"]} == {"fail"}
    missing = measurements_document("s", "r", circumference_profile(o3d_to_trimesh(res.mesh), [5.0]), True)
    m0 = missing["measurements"][0]
    assert m0["value"] is None and m0["source"] == "unavailable" and m0["quality_status"] == "unavailable"
    assert schema_errors(missing, "measurements") == []
