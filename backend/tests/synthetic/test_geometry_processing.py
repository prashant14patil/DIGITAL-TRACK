"""Point-cloud level checks of the camera/turntable conventions against exact geometry.

If these pass, every depth pixel lands on the true surface to numerical precision, so any
reconstruction error seen later comes from fusion/meshing, not from a convention bug.
"""

from dataclasses import replace

import numpy as np
import pytest

from digitaltrack.dataset.loader import Intrinsics, load_session
from digitaltrack.geometry.pointcloud import (
    DepthFilter, depth_to_camera_points, session_points_in_body_frame, valid_depth_mask,
)
from digitaltrack.synthetic.renderer import RenderConfig, render_session
from digitaltrack.synthetic.shapes import cylinder, elliptic_cylinder


def _radial_error(pts, center, radius):
    r = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    return r - radius


def test_noise_free_points_lie_on_the_true_cylinder(full_cylinder_session):
    pts, stats = session_points_in_body_frame(load_session(full_cylinder_session))
    err = _radial_error(pts, (0, 0), 0.15)
    assert len(pts) > 100_000
    assert np.sqrt(np.mean(err**2)) < 1e-5 and np.abs(err).max() < 1e-4
    assert pts[:, 2].min() >= -1e-3 and pts[:, 2].max() <= 0.6 + 1e-3


def test_pitched_camera_still_lands_on_the_surface(tmp_path):
    root = render_session(cylinder(0.15, 0.6), tmp_path / "p", RenderConfig(n_views=12, camera_pitch_deg=12.0,
                                                                             camera_height_m=0.9))
    pts, _ = session_points_in_body_frame(load_session(root))
    on_cap = pts[:, 2] > 0.6 - 1e-4  # a camera above the body legitimately sees the flat top cap
    assert on_cap.sum() > 1000, "pitched camera should observe the top cap"
    assert np.all(np.hypot(pts[on_cap, 0], pts[on_cap, 1]) <= 0.15 + 1e-4)
    side = pts[~on_cap]
    assert np.sqrt(np.mean(_radial_error(side, (0, 0), 0.15) ** 2)) < 1e-5


def test_correct_turntable_sign_is_distinguishable_from_the_wrong_one(tmp_path):
    body = cylinder(0.12, 0.6).with_center(0.06, -0.03)  # off-axis so mirrored fusion is not symmetric
    s = load_session(render_session(body, tmp_path / "off", RenderConfig(n_views=24)))
    good, _ = session_points_in_body_frame(s)
    bad, _ = session_points_in_body_frame(s, sign=-1)
    rms_good = np.sqrt(np.mean(_radial_error(good, body.center_xy, 0.12) ** 2))
    rms_bad = np.sqrt(np.mean(_radial_error(bad, body.center_xy, 0.12) ** 2))
    assert rms_good < 1e-5
    assert rms_bad > 0.02  # >2 cm RMS: a flipped sign is not subtle


def test_axis_centre_error_degrades_the_fused_surface(tmp_path):
    body = cylinder(0.15, 0.6)
    s = load_session(render_session(body, tmp_path / "c",
                                    RenderConfig(n_views=24, calibration_center_offset_m=(0.01, 0.0, 0.0))))
    pts, _ = session_points_in_body_frame(s)
    rms = np.sqrt(np.mean(_radial_error(pts, (0, 0), 0.15) ** 2))
    assert rms > 2e-3  # a 1 cm axis-centre error is visible as millimetre-scale surface smear


def test_axis_tilt_error_degrades_the_fused_surface(tmp_path):
    s = load_session(render_session(cylinder(0.15, 0.6), tmp_path / "t",
                                    RenderConfig(n_views=24, calibration_axis_tilt_deg=2.0)))
    pts, _ = session_points_in_body_frame(s)
    assert np.sqrt(np.mean(_radial_error(pts, (0, 0), 0.15) ** 2)) > 1e-3


def test_random_angle_error_degrades_an_off_axis_body(tmp_path):
    body = cylinder(0.12, 0.6).with_center(0.06, 0.0)
    clean = load_session(render_session(body, tmp_path / "a", RenderConfig(n_views=24)))
    noisy = load_session(render_session(body, tmp_path / "b", RenderConfig(n_views=24, angle_noise_deg=3.0, seed=4)))
    rms = []
    for s in (clean, noisy):
        pts, _ = session_points_in_body_frame(s)
        rms.append(np.sqrt(np.mean(_radial_error(pts, body.center_xy, 0.12) ** 2)))
    assert rms[0] < 1e-5 and rms[1] > 2e-3


def test_elliptic_cross_section_points_satisfy_the_ellipse_equation(tmp_path):
    body = elliptic_cylinder(0.18, 0.10, 0.6)
    s = load_session(render_session(body, tmp_path / "e", RenderConfig(n_views=24)))
    pts, _ = session_points_in_body_frame(s)
    f = (pts[:, 0] / 0.18) ** 2 + (pts[:, 1] / 0.10) ** 2
    assert np.abs(np.sqrt(f) - 1).max() < 2e-3


def test_valid_depth_mask_applies_confidence_and_range():
    depth = np.array([[0.0, 0.05, 1.0, 1.0, 5.0, np.nan]], dtype=np.float32)
    conf = np.array([[2, 2, 0, 1, 2, 2]], dtype=np.uint8)
    m = valid_depth_mask(depth, conf, DepthFilter(min_confidence=1, depth_min_m=0.1, depth_max_m=3.0))
    assert m.tolist() == [[False, False, False, True, False, False]]
    m2 = valid_depth_mask(depth, conf, DepthFilter(min_confidence=2, depth_min_m=0.1, depth_max_m=3.0))
    assert not m2.any()


def test_back_projection_uses_intrinsics_of_the_depth_image():
    k = Intrinsics(fx=100.0, fy=200.0, cx=1.0, cy=2.0, width=3, height=4)
    depth = np.full((4, 3), 2.0, dtype=np.float32)
    pts = depth_to_camera_points(depth, k, np.ones((4, 3), bool))
    # pixel (u=2, v=0): x=(2-1)/100*2, y=(0-2)/200*2
    first_row = pts[:3]
    assert np.allclose(first_row[2], [0.02, -0.02, 2.0])
    with pytest.raises(ValueError):
        depth_to_camera_points(np.zeros((5, 3), np.float32), k, np.ones((5, 3), bool))
