"""Regression tests: can the quality gate detect wrong turntable calibration?

Thresholds were derived from CORRECTLY calibrated synthetic scans (false-positive baseline),
never from the error cases. The baseline cases below must keep passing; the error cases must
keep failing. The line-of-sight blind spot is pinned down explicitly so it cannot be forgotten.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import open3d as o3d
import pytest

from digitaltrack.dataset.loader import load_session
from digitaltrack.quality.basic import QualityThresholds, consistency_checks
from digitaltrack.quality.registration import view_consistency
from digitaltrack.reconstruction.tsdf import TSDFParams, reconstruct_tsdf
from digitaltrack.synthetic.evaluate import evaluate_against_truth
from digitaltrack.synthetic.renderer import RenderConfig, render_session
from digitaltrack.synthetic.shapes import cylinder, elliptic_cylinder, procedural_torso

pytestmark = pytest.mark.slow


def _run(tmp_path, body, cfg, sign_override=None):
    s = load_session(render_session(body, tmp_path / "s", cfg))
    if sign_override is not None:
        s = replace(s, turntable={**s.turntable, "angle_direction_sign": sign_override})
    res = reconstruct_tsdf(s, TSDFParams(), write=False)
    checks = {c["name"]: c for c in res.quality["checks"]}
    return res, checks, evaluate_against_truth(res.mesh, body)


# ---- false-positive baseline: correct calibration must pass -------------------------------------

@pytest.mark.parametrize("name,body,cfg", [
    ("cylinder_clean", cylinder(0.15, 0.6), RenderConfig()),
    ("ellipse_clean", elliptic_cylinder(0.18, 0.11, 0.6), RenderConfig()),
    ("torso_clean", procedural_torso(), RenderConfig()),
    ("torso_noise3mm_dropout5pct", procedural_torso(), RenderConfig(noise_sigma_m=0.003, dropout_prob=0.05, seed=1)),
    ("torso_noise5mm", procedural_torso(), RenderConfig(noise_sigma_m=0.005, seed=3)),
    ("torso_12_views", procedural_torso(), RenderConfig(n_views=12)),
    ("torso_pitched_camera", procedural_torso(), RenderConfig(camera_pitch_deg=12.0, camera_height_m=0.9)),
], ids=lambda v: v if isinstance(v, str) else "")
def test_correct_calibration_passes_consistency_checks(tmp_path, name, body, cfg):
    res, checks, _ = _run(tmp_path, body, cfg)
    for k in ("view_consistency_fitness", "view_consistency_translation", "view_consistency_tilt"):
        assert checks[k]["status"] == "pass", (name, k, checks[k]["value"])
    assert res.quality["passed"]


# ---- turntable centre error ---------------------------------------------------------------------

@pytest.mark.parametrize("offset", [(0.006, 0, 0), (0.010, 0, 0), (-0.008, 0, 0)])
def test_lateral_centre_error_fails_qc(tmp_path, offset):
    res, checks, _ = _run(tmp_path, procedural_torso(), RenderConfig(calibration_center_offset_m=offset))
    assert checks["view_consistency_translation"]["status"] == "fail"
    assert res.quality["passed"] is False


def test_centre_error_detection_survives_depth_noise(tmp_path):
    cfg = RenderConfig(calibration_center_offset_m=(0.006, 0, 0), noise_sigma_m=0.003, seed=5)
    res, checks, _ = _run(tmp_path, procedural_torso(), cfg)
    assert checks["view_consistency_translation"]["status"] == "fail" and not res.quality["passed"]


def test_small_lateral_centre_error_is_at_least_a_warning(tmp_path):
    _, checks, _ = _run(tmp_path, procedural_torso(), RenderConfig(calibration_center_offset_m=(0.004, 0, 0)))
    assert checks["view_consistency_translation"]["status"] in ("warn", "fail")


def test_line_of_sight_centre_error_is_a_known_blind_spot(tmp_path):
    """A centre error along the camera's viewing direction moves every view along its own line of sight by
    the same amount. Views stay mutually consistent (a uniformly larger body), so the self-consistency QC
    passes, yet circumferences are badly wrong. This test PASSES while the blind spot exists, and documents
    its size. Mitigation needed: a known-dimension reference-object check (Phase 2 calibration)."""
    res, checks, ev = _run(tmp_path, procedural_torso(), RenderConfig(calibration_center_offset_m=(0, 0.006, 0)))
    assert checks["view_consistency_translation"]["status"] == "pass"
    assert ev["circumference_bias_m"] > 0.025  # > 2.5 cm circumference error, undetected


@pytest.mark.xfail(strict=True, reason="KNOWN BLIND SPOT: self-consistency cannot detect calibration errors along "
                   "the camera line of sight; requires a reference-object check (not implemented).")
def test_quality_gate_detects_line_of_sight_centre_error(tmp_path):
    res, _, _ = _run(tmp_path, procedural_torso(), RenderConfig(calibration_center_offset_m=(0, 0.006, 0)))
    assert res.quality["passed"] is False


# ---- axis tilt ----------------------------------------------------------------------------------

@pytest.mark.parametrize("tilt", [0.25, 0.5, 1.0])
def test_sideways_axis_tilt_fails_qc(tmp_path, tilt):
    cfg = RenderConfig(calibration_axis_tilt_deg=tilt, calibration_axis_tilt_azimuth_deg=90.0)
    res, checks, _ = _run(tmp_path, procedural_torso(), cfg)
    assert checks["view_consistency_tilt"]["status"] == "fail"
    assert checks["view_consistency_tilt"]["value"] == pytest.approx(tilt, rel=0.3)  # measured ~= injected
    assert res.quality["passed"] is False


def test_axis_tilt_toward_camera_is_under_read_but_flagged_at_one_degree(tmp_path):
    res, checks, _ = _run(tmp_path, procedural_torso(), RenderConfig(calibration_axis_tilt_deg=1.0))
    assert checks["view_consistency_tilt"]["status"] == "fail"
    assert checks["view_consistency_tilt"]["value"] < 0.5  # only ~20% of the true 1 deg is observable
    assert res.quality["passed"] is False


def test_small_axis_tilt_toward_camera_is_only_a_warning(tmp_path):
    """Documents reduced sensitivity: 0.5 deg toward the camera reads ~0.12 deg (warn, not fail)."""
    _, checks, _ = _run(tmp_path, procedural_torso(), RenderConfig(calibration_axis_tilt_deg=0.5))
    assert checks["view_consistency_tilt"]["status"] == "warn"


# ---- rotation direction -------------------------------------------------------------------------

@pytest.mark.parametrize("name,body", [
    ("torso", procedural_torso()),
    ("ellipse", elliptic_cylinder(0.18, 0.11, 0.6)),
    ("off_axis_cylinder", cylinder(0.12, 0.6).with_center(0.06, -0.03)),
], ids=lambda v: v if isinstance(v, str) else "")
def test_wrong_rotation_direction_fails_qc(tmp_path, name, body):
    res, checks, _ = _run(tmp_path, body, RenderConfig(), sign_override=-1)
    assert checks["view_consistency_translation"]["status"] == "fail", (name, checks["view_consistency_translation"])
    assert res.quality["passed"] is False


def test_wrong_rotation_direction_is_invisible_and_harmless_for_a_centred_cylinder(tmp_path):
    """A centred circular cylinder is identical under the flip, so no geometric check can (or needs to) see
    it. Documented so nobody mistakes this for a QC failure: humans are never rotationally symmetric."""
    res, checks, ev = _run(tmp_path, cylinder(0.15, 0.6), RenderConfig(), sign_override=-1)
    assert checks["view_consistency_translation"]["status"] == "pass"
    assert ev["circumference_max_abs_error_pct"] < 1.5


# ---- unit-level behaviour of the consistency check ----------------------------------------------

def _plane_patch(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.2, 0.2, (n, 2))
    z = 0.05 * np.sin(8 * xy[:, 0]) * np.cos(6 * xy[:, 1])  # curved so ICP is well constrained
    return np.column_stack([xy, z])


def _mesh_from(pts):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pc, o3d.utility.DoubleVector([0.01, 0.02]))
    assert len(mesh.triangles) > 1000
    return mesh


def test_consistency_measures_a_known_view_offset():
    pts = _plane_patch()
    mesh = _mesh_from(pts)
    shifted = pts + np.array([0.0, 0.0, 0.008])
    out = view_consistency([pts, shifted], [1, 2], mesh)
    per = {f["frame_id"]: f for f in out["per_frame"]}
    assert per[1]["translation_m"] < 0.001
    assert per[2]["translation_m"] == pytest.approx(0.008, abs=0.0015)
    assert out["applied_to_geometry"] is False


def test_views_with_no_correspondence_fail_on_fitness_not_pass_on_zero_correction():
    pts = _plane_patch()
    far = pts + np.array([0.5, 0.0, 0.0])  # beyond the correspondence window: ICP cannot move it
    out = view_consistency([far, far + 0.001], [1, 2], _mesh_from(pts))
    checks = {c["name"]: c for c in consistency_checks(out, QualityThresholds())}
    assert out["summary"]["median_fitness"] < 0.8
    assert checks["view_consistency_fitness"]["status"] == "fail"


def test_missing_or_uncomputable_consistency_is_never_a_silent_pass():
    th = QualityThresholds()
    assert consistency_checks(None, th)[0]["status"] == "warn"
    empty = view_consistency([np.zeros((10, 3))], [1], o3d.geometry.TriangleMesh())
    assert consistency_checks(empty, th)[0]["status"] == "fail"
