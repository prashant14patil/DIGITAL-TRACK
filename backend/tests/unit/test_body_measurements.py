"""measurement.body on a SYNTHETIC body built from primitives with known dimensions."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from digitaltrack.measurement.body import BodyParams, measure_body, trim_floor_rim
from digitaltrack.synthetic.shapes import ellipse_perimeter

TORSO_A, TORSO_B = 0.16, 0.11  # lateral (y) and depth (x) semi-axes
LEG_R, ARM_R = 0.07, 0.04


def _cyl(r, z0, z1, x=0.0, y=0.0, sx=1.0, sy=1.0, res=180):
    m = o3d.geometry.TriangleMesh.create_cylinder(1.0, z1 - z0, res, 4)
    v = np.asarray(m.vertices) * [r * sx, r * sy, 1.0] + [x, y, (z0 + z1) / 2]
    m.vertices = o3d.utility.Vector3dVector(v)
    return m


def _box(x, y, z0, z1):
    m = o3d.geometry.TriangleMesh.create_box(2 * x, 2 * y, z1 - z0)
    m.translate((-x, -y, z0))
    return m


def synthetic_body(with_rim: bool = False) -> o3d.geometry.TriangleMesh:
    head = o3d.geometry.TriangleMesh.create_sphere(0.10, 60)
    head.translate((0, 0, 1.64))
    parts = [
        _cyl(LEG_R, 0.02, 0.80, y=-0.10), _cyl(LEG_R, 0.02, 0.80, y=0.10),  # legs, 6 cm apart
        _cyl(1.0, 0.805, 1.35, sx=TORSO_B, sy=TORSO_A),  # elliptic torso
        _cyl(ARM_R, 0.85, 1.35, y=-0.30), _cyl(ARM_R, 0.85, 1.35, y=0.30),  # hanging arms, 10 cm from the torso
        _box(0.10, 0.34, 1.35, 1.45), _cyl(0.055, 1.45, 1.55), head,  # shoulders, neck, head (top 1.74)
    ]
    if with_rim:
        parts.append(_box(0.5, 0.5, 0.0, 0.03))  # floor-level fusion residue
    m = parts[0]
    for q in parts[1:]:
        m += q
    return m


@pytest.fixture(scope="module")
def result():
    return measure_body(synthetic_body())


def _m(res, name):
    return next(x for x in res["measurements"] if x["name"] == name)


def test_landmarks(result):
    lm = result["landmarks"]
    assert abs(lm["head_top_z"] - 1.74) < 0.015
    assert 1.45 <= lm["neck_z"] <= 1.56
    assert 1.30 <= lm["arm_separation_z"] < 1.35
    assert 0.79 <= lm["crotch_z"] <= 0.83
    assert result["flags"] == []


def test_torso_excludes_arms_and_matches_ellipse(result):
    truth = float(ellipse_perimeter(TORSO_A, TORSO_B))
    for name in ("chest_circumference", "waist_circumference", "hip_circumference"):
        m = _m(result, name)
        assert m["status"] == "unreliable" and m["unit"] == "m" and m["definition_version"]
        assert abs(m["value"] - truth) / truth < 0.005, (name, m["value"], truth)  # arms would add ~+0.5 m


def test_thighs(result):
    thighs = [x for x in result["measurements"] if x["name"].startswith("thigh_circumference_side")]
    assert len(thighs) == 2
    for t in thighs:
        assert abs(t["value"] - 2 * np.pi * LEG_R) < 0.004


def test_limbs_perpendicular_to_axis(result):
    arms = [x for x in result["measurements"] if x["name"].startswith("upper_arm_circumference_side")]
    calves = [x for x in result["measurements"] if x["name"].startswith("calf_circumference_side")]
    assert len(arms) == 2 and len(calves) == 2
    for a in arms:
        assert abs(a["value"] - 2 * np.pi * ARM_R) < 0.003, a
    for c in calves:
        assert abs(c["value"] - 2 * np.pi * LEG_R) < 0.003, c


def test_tilted_leg_is_measured_perpendicular():
    """A leg tilted 15 deg: a horizontal cut would read ~3.5 % long; the perpendicular cut must not."""
    from digitaltrack.measurement.body import BodyParams, limb_circumference
    from digitaltrack.measurement.basic import o3d_to_trimesh

    leg = _cyl(LEG_R, 0.0, 0.8)
    r = np.radians(15)
    leg.rotate(np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]), center=(0, 0, 0.4))
    d = np.array([0.0, -np.sin(r), np.cos(r)])
    c = limb_circumference(o3d_to_trimesh(leg), np.array([0.0, 0.0, 0.4]), d, BodyParams())
    assert abs(c["value_m"] - 2 * np.pi * LEG_R) < 0.002, c


def test_unavailable_limb_is_not_guessed():
    """Arms that wander (not a straight centre line) must give no value rather than a guess."""
    from digitaltrack.measurement.body import measure_body as mb

    m = synthetic_body().subdivide_midpoint(2)  # dense enough to carry the wiggle
    v = np.asarray(m.vertices).copy()
    arm = (np.abs(v[:, 1]) > 0.25) & (v[:, 2] < 1.35)
    v[arm, 0] += 0.05 * np.sin(2 * np.pi * (1.35 - v[arm, 2]) / 0.2)  # wiggling arms: centre line not straight
    m.vertices = o3d.utility.Vector3dVector(v)
    res = mb(m)
    arms = [x for x in res["measurements"] if x["name"].startswith("upper_arm")]
    assert arms and all(x["value"] is None and x["status"] == "unavailable" for x in arms), arms


def test_floor_rim_trim_keeps_feet():
    body = synthetic_body(with_rim=True)
    trimmed, log = trim_floor_rim(body)
    x = np.asarray(trimmed.vertices)
    low = x[x[:, 2] < 0.04]
    assert log["removed_vertices"] > 0
    assert np.abs(low[:, :2]).max() < 0.10 + LEG_R + BodyParams().footprint_margin_m + 0.01  # only under the feet
    assert x[:, 2].max() > 1.7  # body untouched


def test_left_right_assignment():
    """Figure with a thicker LEFT leg (on +y); the subject's left given as +y must label that leg 'left'."""
    from digitaltrack.measurement.body import measure_body as mb

    m = synthetic_body()
    v = np.asarray(m.vertices).copy()
    left_leg = (v[:, 1] > 0.02) & (v[:, 2] < 0.801)  # include the top ring at 0.80 m
    c = np.array([0.0, 0.10])
    v[left_leg, :2] = c + (v[left_leg, :2] - c) * 1.2  # left leg radius 7 -> 8.4 cm
    m.vertices = o3d.utility.Vector3dVector(v)
    res = mb(m, subject_left_xy=(0.0, 1.0))
    t = {x["name"]: x["value"] for x in res["measurements"] if x["name"].startswith("thigh_circumference")}
    assert set(t) == {"thigh_circumference_left", "thigh_circumference_right"}, t
    assert abs(t["thigh_circumference_left"] - 2 * np.pi * LEG_R * 1.2) < 0.004
    assert abs(t["thigh_circumference_right"] - 2 * np.pi * LEG_R) < 0.004
