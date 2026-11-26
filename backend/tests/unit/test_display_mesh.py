"""Display-mesh cleanup must not change the shape: radius kept, debris and floor spill removed, guard falls back."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from digitaltrack.experimental.display_mesh import DisplayMeshParams, clean_display_mesh

R = 0.15  # cylinder radius (m), similar to a torso half-width


def _noisy_cylinder(seed=0):
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=R, height=1.6, resolution=160, split=160)
    m.translate((0, 0, 0.8 + 0.005))  # bottom cap at z = 5 mm: "spill" below the 15 mm floor cut
    rng = np.random.default_rng(seed)
    v = np.asarray(m.vertices)
    side = np.abs(np.hypot(v[:, 0], v[:, 1]) - R) < 1e-6
    radial = np.zeros_like(v)
    radial[side, :2] = v[side, :2] / R
    v = v + radial * rng.normal(0, 0.0015, (len(v), 1))  # 1.5 mm radial sensor-like noise
    m.vertices = o3d.utility.Vector3dVector(v)
    debris = o3d.geometry.TriangleMesh.create_sphere(0.01, resolution=6).translate((0.5, 0.0, 1.0))  # loose fusion blob
    return m + debris


def _side_radius(mesh, z0=0.3, z1=1.3):
    v = np.asarray(mesh.vertices)
    v = v[(v[:, 2] > z0) & (v[:, 2] < z1)]
    return np.hypot(v[:, 0], v[:, 1])


def test_cleanup_keeps_shape_and_removes_debris():
    src = _noisy_cylinder()
    n_src = len(src.vertices)
    out, rep = clean_display_mesh(src, DisplayMeshParams(target_triangles=20_000))
    assert len(src.vertices) == n_src  # source untouched
    assert rep["fidelity_check"] == "pass" and rep["used"] == "cleaned"
    r = _side_radius(out)
    assert abs(np.mean(r) - R) < 0.0005  # Taubin: no shrinkage (plain Laplacian would shrink)
    assert np.std(r) < np.std(_side_radius(src))  # noise reduced
    v = np.asarray(out.vertices)
    assert v[:, 2].min() >= 0.015  # floor spill removed
    assert np.hypot(v[:, 0] - 0.5, v[:, 2] - 1.0).min() > 0.05  # debris removed
    assert rep["deviation_display_to_source"]["rms_m"] < 0.002
    assert rep["source_uncovered_fraction"] < 0.02  # apart from the debris, everything observed is still shown


def test_guard_falls_back_to_untouched_mesh():
    src = _noisy_cylinder()
    out, rep = clean_display_mesh(src, DisplayMeshParams(max_rms_deviation_m=1e-6, max_p95_deviation_m=1e-6))
    assert rep["fidelity_check"] == "fail" and rep["used"] == "source_unchanged" and "reason" in rep
    assert np.array_equal(np.asarray(out.vertices), np.asarray(src.vertices))
