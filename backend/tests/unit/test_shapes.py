import numpy as np
import pytest

from digitaltrack.synthetic.shapes import (
    cylinder, elliptic_cylinder, ellipse_perimeter, procedural_torso,
)


def test_circle_perimeter():
    assert ellipse_perimeter(0.15, 0.15) == pytest.approx(2 * np.pi * 0.15, rel=1e-12)


def test_known_ellipse_perimeter():
    # a=2, b=1 -> 9.688448220547672 (reference value of the complete elliptic integral formula)
    assert float(ellipse_perimeter(2.0, 1.0)) == pytest.approx(9.688448220547672, rel=1e-12)
    assert float(ellipse_perimeter(1.0, 2.0)) == pytest.approx(9.688448220547672, rel=1e-12)  # symmetric


def test_ellipse_perimeter_matches_numeric_polyline():
    a, b = 0.18, 0.11
    th = np.linspace(0, 2 * np.pi, 200001)
    poly = np.sum(np.hypot(np.diff(a * np.cos(th)), np.diff(b * np.sin(th))))
    assert float(ellipse_perimeter(a, b)) == pytest.approx(poly, rel=1e-9)


def test_cylinder_analytics():
    body = cylinder(0.15, 0.6)
    assert body.height == pytest.approx(0.6)
    assert body.volume() == pytest.approx(np.pi * 0.15**2 * 0.6, rel=1e-6)
    assert np.allclose(body.circumference(np.linspace(0, 0.6, 7)), 2 * np.pi * 0.15)


def test_elliptic_cylinder_analytics():
    body = elliptic_cylinder(0.18, 0.11, 0.5)
    assert body.circumference(0.25) == pytest.approx(float(ellipse_perimeter(0.18, 0.11)))
    assert body.volume() == pytest.approx(np.pi * 0.18 * 0.11 * 0.5, rel=1e-6)


def test_inside_respects_radius_height_and_center_offset():
    body = cylinder(0.15, 0.6)
    pts = np.array([[0, 0, 0.3], [0.149, 0, 0.3], [0.151, 0, 0.3], [0, 0, 0.61], [0, 0, -0.01]])
    assert body.inside(pts).tolist() == [True, True, False, False, False]
    moved = body.with_center(0.05, 0.0)
    assert moved.inside(np.array([[0.19, 0, 0.3], [-0.11, 0, 0.3]])).tolist() == [True, False]
    assert moved.bounding_radius == pytest.approx(0.20)


def test_torso_profile_is_not_a_cylinder_and_has_waist():
    t = procedural_torso()
    hip, waist, chest = (float(t.circumference(z)) for z in (0.10, 0.25, 0.40))
    assert waist < hip and waist < chest
    assert 0.55 < t.height < 0.65
    assert t.volume() > 0


def test_ground_truth_serialisation_is_json_safe():
    import json
    json.dumps(procedural_torso().with_center(0.01, 0.02).to_dict())
