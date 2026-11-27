"""Mesh defect inspection and topology-only repair (no surface movement).

`inspect_mesh` counts every defect class that makes a mesh unfit for measurement or display: disconnected
components, isolated vertices, degenerate and duplicated triangles, non-manifold edges/vertices, self-intersections,
open boundaries (holes), spikes (vertices far from their neighbours' mean) and orientation/watertightness.

`repair_mesh` fixes only what can be fixed WITHOUT moving the surface: it deletes degenerate / duplicated triangles,
triangles on non-manifold edges, triangles at non-manifold vertices, self-intersecting triangles, small loose
components and unreferenced vertices. Every removal is counted, and the distance of the original vertices to the
repaired surface is reported, so it is visible that the measured surface did not change. Holes are NOT filled here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class RepairParams:
    min_component_fraction: float = 0.01  # loose pieces smaller than this share of all triangles are removed
    degenerate_area_m2: float = 1e-10
    spike_factor: float = 4.0  # vertex farther than this x median edge from its neighbours' mean = spike
    self_intersections: bool = True


def _boundary_loops(tri: np.ndarray, x: np.ndarray) -> tuple[int, float]:
    if len(tri) == 0:
        return 0, 0.0
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    b = u[c == 1]
    if not len(b):
        return 0, 0.0
    import networkx as nx

    g = nx.Graph()
    g.add_edges_from(map(tuple, b))
    return nx.number_connected_components(g), float(np.linalg.norm(x[b[:, 0]] - x[b[:, 1]], axis=1).sum())


def _spikes(mesh: o3d.geometry.TriangleMesh, factor: float) -> np.ndarray:
    x, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(t) == 0:
        return np.zeros(len(x), bool)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    med = float(np.median(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)))
    s = np.zeros_like(x)
    n = np.zeros(len(x))
    for a, b in ((0, 1), (1, 0)):
        np.add.at(s, e[:, a], x[e[:, b]])
        np.add.at(n, e[:, a], 1)
    ok = n > 0
    d = np.zeros(len(x))
    d[ok] = np.linalg.norm(x[ok] - s[ok] / n[ok, None], axis=1)
    return d > factor * med


def inspect_mesh(mesh: o3d.geometry.TriangleMesh, *, self_intersections: bool = True, spike_factor: float = 4.0) -> dict:
    x, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    out: dict = {"vertices": int(len(x)), "triangles": int(len(t))}
    if len(t) == 0:
        return out
    _, counts, _ = mesh.cluster_connected_triangles()
    counts = np.sort(np.asarray(counts))[::-1]
    used = np.zeros(len(x), bool)
    used[t.ravel()] = True
    area = 0.5 * np.linalg.norm(np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]), axis=1)
    holes, blen = _boundary_loops(t, x)
    out.update({
        "components": int(len(counts)), "largest_component_share": float(counts[0] / counts.sum()),
        "isolated_vertices": int((~used).sum()), "degenerate_triangles": int((area < 1e-10).sum()),
        "duplicated_triangles": int(len(t) - len(np.unique(np.sort(t, axis=1), axis=0))),
        "non_manifold_edges": int(len(mesh.get_non_manifold_edges())),
        "non_manifold_vertices": int(len(mesh.get_non_manifold_vertices())),
        "holes": holes, "boundary_length_m": blen, "watertight": bool(mesh.is_watertight()),
        "orientable": bool(mesh.is_orientable()), "spikes": int(_spikes(mesh, spike_factor).sum()),
        "surface_area_m2": float(area.sum()),
    })
    if self_intersections:
        out["self_intersecting_pairs"] = int(len(np.asarray(mesh.get_self_intersecting_triangles())))
    return out


def repair_mesh(mesh: o3d.geometry.TriangleMesh, p: RepairParams | None = None) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Topology-only repair; the input is not modified. Returns (repaired mesh, log)."""
    p = p or RepairParams()
    m = o3d.geometry.TriangleMesh(mesh)
    before = inspect_mesh(m, self_intersections=p.self_intersections, spike_factor=p.spike_factor)
    steps = {}
    n = len(m.triangles)
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    x, t = np.asarray(m.vertices), np.asarray(m.triangles)
    area = 0.5 * np.linalg.norm(np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]), axis=1)
    m.remove_triangles_by_mask(area < p.degenerate_area_m2)
    steps["duplicate_or_degenerate_triangles"] = n - len(m.triangles)
    if p.self_intersections:
        si = np.asarray(m.get_self_intersecting_triangles())
        if len(si):
            bad = np.zeros(len(m.triangles), bool)
            bad[np.unique(si.ravel())] = True
            m.remove_triangles_by_mask(bad)
        steps["self_intersecting_triangles"] = int(len(np.unique(si.ravel()))) if len(si) else 0
    n = len(m.triangles)
    m.remove_non_manifold_edges()
    steps["non_manifold_edge_triangles"] = n - len(m.triangles)
    for _ in range(3):  # removing triangles can create new non-manifold vertices; a few passes settle it
        nmv = np.asarray(m.get_non_manifold_vertices())
        if not len(nmv):
            break
        tri = np.asarray(m.triangles)
        m.remove_triangles_by_mask(np.isin(tri, nmv).any(axis=1))
        steps["non_manifold_vertex_triangles"] = steps.get("non_manifold_vertex_triangles", 0) + int(np.isin(tri, nmv).any(axis=1).sum())
    labels, counts, _ = m.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    small = counts[labels] < p.min_component_fraction * max(counts.sum(), 1) if counts.size else np.zeros(0, bool)
    m.remove_triangles_by_mask(small)
    steps["small_component_triangles"] = int(small.sum())
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    after = inspect_mesh(m, self_intersections=p.self_intersections, spike_factor=p.spike_factor)
    # the kept surface is the original surface: report how far original vertices are from it (holes show up here)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    d = scene.compute_distance(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32))).numpy()
    log = {"params": asdict(p), "before": before, "after": after, "removed": steps,
           "original_vertices_off_surface_p99_m": float(np.percentile(d, 99)),
           "note": "topology-only repair: no vertex was moved, no hole was filled"}
    return m, log
