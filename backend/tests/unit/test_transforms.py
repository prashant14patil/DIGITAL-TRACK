import numpy as np
import pytest

from digitaltrack.geometry.transforms import camera_from_body, invert, rot_x, rot_z, transform_points


def test_rot_z_is_right_handed():
    assert np.allclose(transform_points(rot_z(90), np.array([[1.0, 0, 0]])), [[0, 1, 0]], atol=1e-12)


def test_rot_x_is_right_handed():
    assert np.allclose(transform_points(rot_x(90), np.array([[0, 1.0, 0]])), [[0, 0, 1]], atol=1e-12)


def test_invert_roundtrip():
    t = rot_z(33) @ rot_x(12)
    t[:3, 3] = [0.1, -0.4, 1.3]
    assert np.allclose(invert(t) @ t, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("sign", [1, -1])
def test_camera_from_body_composition(sign):
    t_ct = rot_x(-80)
    t_ct[:3, 3] = [0, 0.3, 1.2]
    p = np.array([[0.1, 0.2, 0.3]])
    got = transform_points(camera_from_body(t_ct, 40.0, sign), p)
    want = transform_points(t_ct, transform_points(rot_z(sign * 40.0), p))
    assert np.allclose(got, want)


def test_zero_angle_is_just_the_calibration():
    t_ct = rot_x(-80)
    assert np.allclose(camera_from_body(t_ct, 0.0, 1), t_ct)
