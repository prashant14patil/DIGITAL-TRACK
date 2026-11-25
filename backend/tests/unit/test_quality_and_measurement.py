import numpy as np
import pytest
import trimesh

from digitaltrack.measurement.basic import slice_at_height, vertical_extent
from digitaltrack.quality.basic import max_angular_gap_deg


@pytest.mark.parametrize("angles,expected", [
    ([0, 90, 180, 270], 90.0),
    ([270, 0, 180, 90], 90.0),  # order independent
    (list(np.arange(0, 360, 10)), 10.0),
    ([0, 10, 20], 340.0),  # wrap-around gap counts
    ([10.0], 360.0),
    ([0, 360, 720], 360.0),  # modulo 360: all the same direction
    ([-10, 10], 340.0),
])
def test_max_angular_gap(angles, expected):
    assert max_angular_gap_deg(angles) == pytest.approx(expected)


def _cyl(r=0.15, h=0.6, x=0.0):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=720)
    m.apply_translation([x, 0, h / 2])
    return m


def test_slice_perimeter_of_a_known_cylinder():
    s = slice_at_height(_cyl(), 0.3)
    assert s.perimeter_m == pytest.approx(2 * np.pi * 0.15, rel=1e-4)
    assert s.area_m2 == pytest.approx(np.pi * 0.15**2, rel=1e-3)
    assert (s.n_closed_contours, s.n_open_polylines, s.status) == (1, 0, "pass")


def test_slice_perimeter_of_a_box_is_exact():
    box = trimesh.creation.box(extents=[0.4, 0.2, 0.6])
    box.apply_translation([0, 0, 0.3])
    assert slice_at_height(box, 0.2).perimeter_m == pytest.approx(2 * (0.4 + 0.2), rel=1e-9)


def test_slice_outside_the_mesh_is_a_failure_not_a_number():
    s = slice_at_height(_cyl(), 5.0)
    assert s.perimeter_m is None and s.status == "fail"


def test_open_surface_is_not_reported_as_a_confident_contour():
    m = _cyl()
    keep = ~np.isclose(np.arctan2(m.triangles_center[:, 1], m.triangles_center[:, 0]), 0.0, atol=0.4)
    cut = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)  # remove a vertical strip of triangles
    s = slice_at_height(cut, 0.3)
    assert s.perimeter_m is None or s.status != "pass"


def test_multiple_contours_pick_largest_and_warn():
    both = trimesh.util.concatenate([_cyl(0.15, 0.6, 0.0), _cyl(0.05, 0.6, 0.5)])
    s = slice_at_height(both, 0.3)
    assert s.n_closed_contours == 2 and s.status == "warn"
    assert s.perimeter_m == pytest.approx(2 * np.pi * 0.15, rel=1e-3)


def test_broken_main_outline_is_not_replaced_by_a_small_closed_fragment():
    m = _cyl()
    keep = ~np.isclose(np.arctan2(m.triangles_center[:, 1], m.triangles_center[:, 0]), 0.0, atol=0.4)
    cut = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    s = slice_at_height(trimesh.util.concatenate([cut, _cyl(0.02, 0.6, 0.5)]), 0.3)
    assert s.perimeter_m is None and s.status == "fail"
    assert (s.n_closed_contours, s.n_open_polylines) == (1, 1)


def test_small_open_scrap_beside_a_good_contour_warns_but_keeps_the_value():
    m = _cyl(0.02, 0.6, 0.5)
    keep = ~np.isclose(np.arctan2(m.triangles_center[:, 1], m.triangles_center[:, 0] - 0.5), 0.0, atol=0.4)
    scrap = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    s = slice_at_height(trimesh.util.concatenate([_cyl(), scrap]), 0.3)
    assert s.status == "warn" and s.n_open_polylines == 1
    assert s.perimeter_m == pytest.approx(2 * np.pi * 0.15, rel=1e-4)


def test_vertical_extent():
    assert vertical_extent(_cyl(h=0.6)) == pytest.approx((0.0, 0.6))
