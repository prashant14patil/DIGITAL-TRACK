import json

import numpy as np
import pytest

from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.validate import validate_session
from digitaltrack.synthetic.renderer import RenderConfig, render_frame, render_session
from digitaltrack.synthetic.shapes import cylinder, elliptic_cylinder


def _frame(body, cfg, angle=0.0, seed=0):
    return render_frame(body, cfg, angle, np.random.default_rng(seed))


def test_principal_ray_depth_is_distance_minus_radius():
    cfg = RenderConfig()
    depth, conf = _frame(cylinder(0.15, 0.6), cfg)
    assert depth.dtype == np.float32 and conf.dtype == np.uint8
    assert depth.shape == (cfg.image_height, cfg.image_width)
    # camera height 0.3 is mid-body; principal ray is horizontal
    v, u = cfg.image_height // 2, cfg.image_width // 2
    assert depth[v, u] == pytest.approx(cfg.camera_distance_m - 0.15, abs=1e-3)


def test_depth_is_invalid_outside_the_body_and_confidence_marks_it():
    depth, conf = _frame(cylinder(0.15, 0.6), RenderConfig())
    assert depth[0, 0] == 0.0 and conf[0, 0] == 0
    assert (depth > 0).any() and set(np.unique(conf)) <= {0, 1, 2}
    assert np.all(conf[depth == 0] == 0)


def test_rotation_changes_the_view_of_an_elliptic_body():
    cfg = RenderConfig()
    body = elliptic_cylinder(a=0.20, b=0.10, height=0.6)
    v, u = cfg.image_height // 2, cfg.image_width // 2
    d0 = _frame(body, cfg, 0.0)[0][v, u]
    d90 = _frame(body, cfg, 90.0)[0][v, u]
    assert d0 == pytest.approx(cfg.camera_distance_m - 0.10, abs=1e-3)  # short semi-axis faces camera
    assert d90 == pytest.approx(cfg.camera_distance_m - 0.20, abs=1e-3)  # long semi-axis faces camera
    d180 = _frame(body, cfg, 180.0)[0][v, u]
    assert d180 == pytest.approx(d0, abs=1e-3)


def test_depth_noise_has_configured_sigma():
    body = cylinder(0.15, 0.6)
    clean, _ = _frame(body, RenderConfig())
    noisy, _ = _frame(body, RenderConfig(noise_sigma_m=0.003), seed=5)
    m = clean > 0
    assert (noisy[m] > 0).mean() > 0.99
    assert np.std((noisy - clean)[m]) == pytest.approx(0.003, rel=0.1)
    assert np.mean((noisy - clean)[m]) == pytest.approx(0.0, abs=2e-4)


def test_dropout_removes_the_requested_fraction_of_valid_pixels():
    body = cylinder(0.15, 0.6)
    clean, _ = _frame(body, RenderConfig())
    dropped, conf = _frame(body, RenderConfig(dropout_prob=0.3), seed=2)
    m = clean > 0
    frac_missing = float(((dropped == 0) & m).sum() / m.sum())
    assert frac_missing == pytest.approx(0.3, abs=0.03)
    assert np.all(conf[dropped == 0] == 0)
    kept = (dropped > 0)
    assert np.array_equal(dropped[kept], clean[kept])  # dropout never alters surviving depth


def test_rendering_is_deterministic_for_a_seed():
    body = cylinder(0.15, 0.6)
    cfg = RenderConfig(noise_sigma_m=0.002, dropout_prob=0.1)
    a, _ = _frame(body, cfg, seed=7)
    b, _ = _frame(body, cfg, seed=7)
    c, _ = _frame(body, cfg, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_angle_errors_are_recorded_separately_from_truth(tmp_path):
    cfg = RenderConfig(n_views=8, angle_bias_deg=2.0, angle_noise_deg=0.5, seed=3)
    root = render_session(cylinder(), tmp_path / "s", cfg)
    assert validate_session(root).ok
    truth = json.loads((root / "ground_truth.json").read_text())["true_angles_deg"]
    rec = np.array([f.angle_deg for f in load_session(root).frames])
    err = rec - np.array(truth)
    assert np.mean(err) == pytest.approx(2.0, abs=0.8)
    assert np.std(err) > 0.1
    assert np.allclose(truth, np.arange(8) * 45.0)


def test_session_is_flagged_synthetic_and_truth_is_outside_raw_frames(small_cylinder_session):
    s = load_session(small_cylinder_session)
    assert s.meta["synthetic"] is True and s.meta["capture"]["rotation_source"] == "synthetic"
    assert (small_cylinder_session / "ground_truth.json").is_file()
    assert not list((small_cylinder_session / "frames").rglob("*truth*"))


def test_injected_calibration_error_changes_only_the_reported_calibration(tmp_path):
    clean = render_session(cylinder(), tmp_path / "c", RenderConfig(n_views=2))
    bad = render_session(cylinder(), tmp_path / "b", RenderConfig(n_views=2, calibration_axis_tilt_deg=1.5,
                                                                  calibration_center_offset_m=(0.01, 0, 0)))
    tc = json.loads((clean / "calibration/turntable_calibration.json").read_text())
    tb = json.loads((bad / "calibration/turntable_calibration.json").read_text())
    assert tc["method"] == "synthetic_truth" and tb["method"] == "synthetic_truth_with_injected_error"
    assert tc["camera_from_turntable"] != tb["camera_from_turntable"]
    f_clean = load_session(clean).frames[0].load_depth()
    f_bad = load_session(bad).frames[0].load_depth()
    assert np.array_equal(f_clean, f_bad)  # the sensor data is the same; only the reported rig differs
