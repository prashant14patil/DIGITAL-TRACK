import numpy as np
import pytest

from digitaltrack.calibration.floor import FloorFitError, fit_floor_plane, floor_from_camera
from digitaltrack.calibration.scale import ScaleError, scale_from_known_lengths
from digitaltrack.geometry.transforms import transform_points

# ---- scale ---------------------------------------------------------------------------------------


def test_exact_scale_is_recovered():
    est = scale_from_known_lengths([10.0, 20.0, 40.0], [0.1, 0.2, 0.4])
    assert est.scale == pytest.approx(0.01) and est.rms_residual_m == pytest.approx(0, abs=1e-12)
    assert est.status() == "pass" and est.n == 3


def test_inconsistent_lengths_show_up_in_residuals_and_status():
    est = scale_from_known_lengths([100.0, 100.0], [1.0, 1.015])
    assert est.scale == pytest.approx(0.010075)
    assert est.residuals_m == pytest.approx((0.0075, -0.0075))
    assert est.max_relative_residual == pytest.approx(0.0075)  # 0.0075 / 1.0
    assert est.status() == "warn"  # experimental thresholds: warn > 0.5 %, fail > 1 %
    assert est.status(warn_rel=0.001, fail_rel=0.005) == "fail"


@pytest.mark.parametrize("m,r", [([], []), ([1.0], [1.0, 2.0]), ([0.0], [1.0]), ([1.0], [-1.0]),
                                 ([np.nan], [1.0]), ([[1.0]], [[1.0]])])
def test_scale_rejects_bad_input(m, r):
    with pytest.raises(ScaleError):
        scale_from_known_lengths(m, r)


# ---- floor plane ---------------------------------------------------------------------------------

def _floor_scene(seed=0, noise=0.003, n_floor=4000, n_clutter=2000):
    """Camera frame (y down). Floor plane y = 1.2 m below the camera, tilted 8 degrees; a box of clutter above."""
    rng = np.random.default_rng(seed)
    tilt = np.deg2rad(8)
    up = np.array([0, -np.cos(tilt), np.sin(tilt)])  # true floor normal pointing up (towards the camera side)
    u = np.cross(up, [1, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(up, u)
    p0 = np.array([0, 1.2, 2.0])
    a, b = rng.uniform(-1.5, 1.5, (2, n_floor))
    floor = p0 + a[:, None] * u + b[:, None] * v + rng.normal(0, noise, n_floor)[:, None] * up
    clutter = p0 + rng.uniform(-0.3, 0.3, (n_clutter, 3)) + up * rng.uniform(0.1, 1.7, n_clutter)[:, None]
    return np.vstack([floor, clutter]), up, p0, n_floor


def test_floor_plane_is_recovered_despite_clutter():
    pts, up, p0, n_floor = _floor_scene()
    plane = fit_floor_plane(pts, distance_threshold_m=0.01)
    n = np.array(plane.normal)
    assert np.degrees(np.arccos(np.clip(n @ up, -1, 1))) < 0.3
    assert abs(plane.signed_distance(p0[None])[0]) < 0.002
    assert plane.n_inliers == pytest.approx(n_floor, rel=0.02) and plane.inlier_rms_m < 0.004


def test_normal_orientation_follows_up_hint():
    pts, up, _, _ = _floor_scene()
    assert np.array(fit_floor_plane(pts, up_hint=(0, 1, 0)).normal) @ up < 0


def test_fit_is_reproducible_for_a_seed():
    pts, *_ = _floor_scene()
    assert fit_floor_plane(pts, seed=3) == fit_floor_plane(pts, seed=3)


def test_degenerate_or_floorless_input_is_refused():
    with pytest.raises(FloorFitError):
        fit_floor_plane(np.zeros((2, 3)))
    with pytest.raises(FloorFitError):
        fit_floor_plane(np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)]))  # collinear
    rng = np.random.default_rng(0)
    with pytest.raises(FloorFitError, match="is the floor visible"):
        fit_floor_plane(rng.uniform(-1, 1, (3000, 3)), distance_threshold_m=0.005)
    with pytest.raises(FloorFitError):
        fit_floor_plane(np.zeros((5, 2)))


def test_floor_frame_puts_floor_at_z0_and_up_at_plus_z():
    pts, up, p0, n_floor = _floor_scene(noise=0.0, n_clutter=0)
    plane = fit_floor_plane(pts)
    t = floor_from_camera(plane, origin_hint=p0 + 0.5 * up)
    assert np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.allclose(transform_points(t, pts)[:, 2], 0, atol=1e-6)
    assert np.allclose(t[:3, :3] @ np.array(plane.normal), [0, 0, 1], atol=1e-9)
    assert np.allclose(transform_points(t, (p0 + 0.5 * up)[None]), [[0, 0, 0.5]], atol=1e-6)
