"""Internal consistency metrics (no ground truth): the same functions are applied to every pose method compared.

These say how well the registered views agree with each other and with the fused surface. They do NOT measure
accuracy: a wrong but self-consistent result scores well.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def cross_view_consistency(clouds: list[np.ndarray], *, search_m: float = 0.05, close_m: float = 0.01,
                           max_points: int = 4000, seed: int = 0) -> dict:
    """Distance from each view's points to the nearest point of any OTHER view (only pairs within `search_m`)."""
    if len(clouds) < 2:
        return {"views": len(clouds), "median_mm": None}
    rng = np.random.default_rng(seed)
    dists, matched, total = [], 0, 0
    for k, c in enumerate(clouds):
        sample = c if len(c) <= max_points else c[rng.choice(len(c), max_points, replace=False)]
        others = np.concatenate([o for j, o in enumerate(clouds) if j != k])
        d, _ = cKDTree(others).query(sample, distance_upper_bound=search_m)
        ok = np.isfinite(d)
        dists.append(d[ok])
        matched += int(ok.sum())
        total += len(sample)
    d = np.concatenate(dists)
    return {"views": len(clouds), "median_mm": float(np.median(d)) * 1000 if d.size else None,
            "p90_mm": float(np.percentile(d, 90)) * 1000 if d.size else None,
            "within_10mm_fraction": float((d < close_m).sum() / total) if total else None,
            "without_neighbour_fraction": float(1 - matched / total) if total else None}


def distances_to_mesh(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, np.float32))).numpy().astype(float)


def distance_stats(d: np.ndarray) -> dict:
    if not d.size:
        return {"rms_mm": None, "median_mm": None, "p95_mm": None, "max_mm": None}
    return {"rms_mm": float(np.sqrt(np.mean(d ** 2))) * 1000, "median_mm": float(np.median(d)) * 1000,
            "p95_mm": float(np.percentile(d, 95)) * 1000, "max_mm": float(d.max()) * 1000}


def mesh_topology(mesh: o3d.geometry.TriangleMesh) -> dict:
    if not len(mesh.triangles):
        return {"vertices": 0, "triangles": 0}
    tri = np.sort(np.asarray(mesh.triangles)[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(tri, axis=0, return_counts=True)
    _, sizes, _ = mesh.cluster_connected_triangles()
    return {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles), "components": len(sizes),
            "boundary_edges": int((counts == 1).sum()), "non_manifold_edges": int((counts > 2).sum()),
            "watertight": bool((counts == 2).all())}


def observation_coverage(mesh: o3d.geometry.TriangleMesh, clouds: list[np.ndarray], *, radius_m: float = 0.01,
                         samples: int = 40000) -> dict:
    """Share of the mesh surface seen by 0, 1, 2+ views. Unmeshed (never observed) regions are NOT in the mesh."""
    o3d.utility.random.seed(0)
    pts = np.asarray(mesh.sample_points_uniformly(samples).points)
    count = np.zeros(len(pts), int)
    for c in clouds:
        d, _ = cKDTree(c).query(pts, distance_upper_bound=radius_m)
        count += np.isfinite(d)
    return {"radius_mm": radius_m * 1000, "seen_by_0": float((count == 0).mean()),
            "seen_by_1": float((count == 1).mean()), "seen_by_3_or_more": float((count >= 3).mean())}
