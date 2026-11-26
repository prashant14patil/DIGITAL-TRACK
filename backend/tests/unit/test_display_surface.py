"""experimental.display_mesh.build_display_surface on SYNTHETIC shapes: gap filling is flagged, fidelity is checked."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from digitaltrack.experimental.display_mesh import SurfaceDisplayParams, build_display_surface


def _sphere_with_hole(r=0.15, hole_z=0.10):
    m = o3d.geometry.TriangleMesh.create_sphere(r, 120)
    m.translate((0, 0, 1.0))
    v = np.asarray(m.vertices)
    m.remove_vertices_by_mask(v[:, 2] > 1.0 + hole_z)  # open cap: a hole of radius ~11 cm
    return m


def test_hole_is_bridged_and_flagged_estimated():
    src = _sphere_with_hole()
    disp, est, rep = build_display_surface(src, SurfaceDisplayParams(samples=150_000, poisson_depth=8,
                                                                     max_estimated_fraction=0.3))
    assert disp is not None, rep
    v = np.asarray(disp.vertices)
    cap = v[:, 2] > 1.0 + 0.12
    assert cap.any() and est[cap].mean() > 0.9  # filled cap = estimated
    assert not est[v[:, 2] < 1.0].any()  # the observed lower half is measured
    assert rep["deviation_measured_part_to_source"]["p95_m"] < 0.004
    assert rep["source_covered_within_5mm"] > 0.95


def test_large_gaps_are_not_invented():
    """More than half missing: the fitted cap would lie far from any data, so it is dropped, not shown."""
    from digitaltrack.rgbd.consistency import pixel_residuals

    src = _sphere_with_hole(hole_z=-0.05)
    p = SurfaceDisplayParams(samples=100_000, poisson_depth=8)
    disp, est, rep = build_display_surface(src, p)
    assert disp is not None, rep
    d = pixel_residuals(src, np.asarray(disp.vertices))
    assert d.max() <= p.drop_beyond_m + 1e-3
    assert np.asarray(disp.vertices)[:, 2].max() < 1.0 + 0.15 - 0.03  # no full dome on top


def test_rejects_when_measured_surface_would_vanish():
    src = _sphere_with_hole()
    p = SurfaceDisplayParams(samples=150_000, poisson_depth=8, min_source_covered=1.01)  # impossible to meet
    disp, est, rep = build_display_surface(src, p)
    assert disp is None and rep["fidelity_check"] == "fail" and "reason" in rep
