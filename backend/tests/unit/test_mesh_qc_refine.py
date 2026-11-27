"""reconstruction.mesh_qc and reconstruction.refine on SYNTHETIC meshes with injected defects / known dimensions."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from digitaltrack.experimental.display_mesh import SurfaceDisplayParams
from digitaltrack.reconstruction.mesh_qc import inspect_mesh, repair_mesh
from digitaltrack.reconstruction.refine import RefineParams, refine
from digitaltrack.synthetic.body_figure import Figure, scan_copy


def _defective_sphere():
    m = o3d.geometry.TriangleMesh.create_sphere(0.2, 80)  # fine sphere: the speck is ~0.5 % of the triangles
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    speck = o3d.geometry.TriangleMesh.create_sphere(0.01, 6).translate((1.0, 0, 0))  # floating fragment
    t2 = np.vstack([t, t[:3], [[0, 0, 1]]])  # duplicated + degenerate triangles
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t2)) + speck
    return m, len(t)


def test_inspect_finds_and_repair_fixes_defects_without_moving_the_surface():
    m, n_clean = _defective_sphere()
    before = inspect_mesh(m)
    assert before["components"] >= 2 and before["duplicated_triangles"] >= 3 and before["degenerate_triangles"] >= 1
    fixed, log = repair_mesh(m)
    after = log["after"]
    assert after["components"] == 1 and after["degenerate_triangles"] == 0 and after["duplicated_triangles"] == 0
    assert after["non_manifold_edges"] == 0 and after["self_intersecting_pairs"] == 0
    assert len(fixed.triangles) == n_clean  # exactly the sphere is left
    # no vertex moved: every kept vertex is an original vertex
    orig = {tuple(np.round(x, 9)) for x in np.asarray(m.vertices)}
    assert all(tuple(np.round(x, 9)) in orig for x in np.asarray(fixed.vertices))


def test_refine_accepts_a_body_whose_measurements_it_reproduces():
    mesh, _ = scan_copy(Figure(), np.random.default_rng(0), noise_m=0.001, scale_sd=0.0)
    p = RefineParams(surface=SurfaceDisplayParams(poisson_depth=9, drop_beyond_m=0.01, samples=200_000,
                                                  target_triangles=150_000))
    ref, est, rep = refine(mesh, p)
    assert ref is not None, rep
    names = {r["name"]: r for r in rep["measurement_agreement"]}
    for n in ("height_floor_to_head_top", "chest_circumference", "waist_circumference", "hip_circumference"):
        assert names[n]["status"] == "pass", names[n]
    assert rep["status"] == "accepted", rep["reasons"]


def test_refine_flags_when_tolerance_is_impossible():
    mesh, _ = scan_copy(Figure(), np.random.default_rng(0), noise_m=0.001, scale_sd=0.0)
    p = RefineParams(surface=SurfaceDisplayParams(poisson_depth=8, drop_beyond_m=0.01, samples=100_000,
                                                  target_triangles=80_000), abs_tol_m=0.0, rel_tol=0.0)
    ref, est, rep = refine(mesh, p)
    assert rep["status"] == "flagged" and "measurements_agree" in rep["reasons"]
