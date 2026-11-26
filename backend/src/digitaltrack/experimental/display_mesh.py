"""EXPERIMENTAL: shape-preserving cleanup of a fused depth mesh for DISPLAY on the Body avatar page.

Nothing is measured from the result; measurements always use the unmodified metric mesh (fused_mesh.ply).
The display copy must still look like the person's real shape, so every step is chosen not to move the surface
and the result is checked against the source:

  1. floor cut: drop faces with a vertex below `floor_cut_m` (same value as the segmentation floor band, so only
     TSDF spill below the level that is already classed as floor is removed)
  2. drop loose pieces smaller than `min_component_fraction` of the faces (fusion debris, not body)
  3. Taubin smoothing (lambda/mu): removes voxel stair-steps without the shrinkage of plain Laplacian smoothing
  4. quadric decimation to `target_triangles` (browser size)
  5. fidelity check: distance from every display vertex to the source surface (RMS, P95, max) and the share of
     the source surface that the display copy no longer covers. If RMS or P95 exceed the limits, the cleanup is
     discarded and the untouched source mesh is used instead, with the reason recorded.

`clean_display_mesh` fills no holes. `build_display_surface` (screened Poisson) does bridge gaps, and therefore
returns which display vertices are NOT on the measured surface so they are shown as `estimated`; its fidelity is
checked on the measured part. Errors that are in the data (e.g. a ghost copy of a moved arm) stay visible.
Thresholds are PROPOSED engineering defaults, checked on synthetic data and one real recording.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d

ALGORITHM = "display-mesh-cleanup-0.1 (EXPERIMENTAL, display only)"


@dataclass(frozen=True)
class DisplayMeshParams:
    floor_cut_m: float = 0.015  # = rgbd.segment.SegmentParams.floor_band_m
    min_component_fraction: float = 0.01
    taubin_iterations: int = 10
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    target_triangles: int = 150_000
    max_rms_deviation_m: float = 0.002  # display -> source surface
    max_p95_deviation_m: float = 0.004
    uncovered_distance_m: float = 0.005  # a source vertex farther than this from the display copy is "not covered"


def _distances(to_mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(to_mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, np.float32))).numpy().astype(float)


def _stats(d: np.ndarray) -> dict:
    return {"rms_m": float(np.sqrt(np.mean(d ** 2))) if d.size else None,
            "p95_m": float(np.percentile(d, 95)) if d.size else None,
            "max_m": float(d.max()) if d.size else None}


@dataclass(frozen=True)
class SurfaceDisplayParams:
    """Closed, smooth DISPLAY surface (screened Poisson) built from the measuring mesh.

    Checked on the first real tracked RGB-D scan: depth 9 kept 96 % of the display surface within 5 mm of the
    measured surface (RMS 0.65 mm, P95 0.84 mm there); the rest bridges holes and is flagged `estimated`."""
    samples: int = 300_000
    poisson_depth: int = 9
    poisson_scale: float = 1.05
    measured_within_m: float = 0.005  # display vertex this close to the measured surface = "measured"
    drop_beyond_m: float = 0.03  # farther than this from any measured surface = Poisson bubble: removed
    target_triangles: int = 200_000
    max_rms_deviation_m: float = 0.002  # on the measured part
    max_p95_deviation_m: float = 0.004
    max_estimated_fraction: float = 0.15
    min_source_covered: float = 0.95  # share of the measured surface the display surface must cover (within 5 mm)
    min_component_fraction: float = 0.01
    seed: int = 0


def build_display_surface(source: o3d.geometry.TriangleMesh, params: SurfaceDisplayParams | None = None
                          ) -> tuple[o3d.geometry.TriangleMesh | None, np.ndarray | None, dict]:
    """Return (display mesh, estimated-vertex mask, report); (None, None, report) when a fidelity limit fails.

    Gap-filled vertices (farther than measured_within_m from the measured surface) are returned in the mask so the
    viewer can show them as estimated. The source mesh is not modified; nothing is measured from the result."""
    p = params or SurfaceDisplayParams()
    src = o3d.geometry.TriangleMesh(source)
    src.compute_vertex_normals()
    o3d.utility.random.seed(p.seed)
    pcd = src.sample_points_uniformly(p.samples, use_triangle_normal=True)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        m, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=p.poisson_depth,
                                                                          scale=p.poisson_scale, linear_fit=True)
    d = _distances(source, np.asarray(m.vertices))
    m.remove_vertices_by_mask(d > p.drop_beyond_m)
    labels, counts, _ = m.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if counts.size:  # loose specks only; every substantial piece (legs, head, ...) stays
        m.remove_triangles_by_mask(counts[labels] < p.min_component_fraction * counts.sum())
        m.remove_unreferenced_vertices()
    if len(m.triangles) > p.target_triangles:
        m = m.simplify_quadric_decimation(p.target_triangles)
        m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    d = _distances(source, np.asarray(m.vertices))
    est = d > p.measured_within_m
    meas = d[~est]
    dev = _stats(meas)
    back = _distances(m, np.asarray(source.vertices))
    covered = float(np.mean(back < 0.005))
    ok = (dev["rms_m"] is not None and dev["rms_m"] <= p.max_rms_deviation_m
          and dev["p95_m"] <= p.max_p95_deviation_m and est.mean() <= p.max_estimated_fraction
          and covered >= p.min_source_covered)  # nothing measured may silently disappear
    report = {
        "algorithm": "display-surface-poisson-0.1 (EXPERIMENTAL, display only)", "params": asdict(p),
        "display_triangles": len(m.triangles),
        "deviation_measured_part_to_source": dev,
        "estimated_vertex_fraction": float(est.mean()),
        "estimated_rule": f"display vertex farther than {p.measured_within_m * 1000:.0f} mm from the measured "
                          "surface (gap bridged by the surface fit): shown as estimated, never measured",
        "source_covered_within_5mm": covered,
        "fidelity_check": "pass" if ok else "fail",
        "note": "display only; measurements always use fused_mesh.ply",
    }
    if not ok:
        report["reason"] = "fidelity limit exceeded; the shape-preserving cleanup of the measured mesh is used instead"
        return None, None, report
    return m, est, report


def clean_display_mesh(source: o3d.geometry.TriangleMesh, params: DisplayMeshParams | None = None
                       ) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Return (display mesh, report). `source` is not modified. Mesh must be Z-up with the floor at z = 0."""
    p = params or DisplayMeshParams()
    m = o3d.geometry.TriangleMesh(source)
    if m.has_vertex_colors():  # plain grey display; (remove_vertex_colors() is missing in some Open3D builds)
        m.vertex_colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    n0 = len(m.triangles)
    steps = []

    v, f = np.asarray(m.vertices), np.asarray(m.triangles)
    low = (v[f, 2] < p.floor_cut_m).any(axis=1)
    m.remove_triangles_by_mask(low)
    m.remove_unreferenced_vertices()
    steps.append({"step": "floor_cut", "below_m": p.floor_cut_m, "faces_removed": int(low.sum())})

    cid, counts, _ = m.cluster_connected_triangles()
    cid, counts = np.asarray(cid), np.asarray(counts)
    tiny = counts < p.min_component_fraction * max(1, counts.sum())
    small = tiny[cid] if counts.size else np.zeros(0, bool)
    m.remove_triangles_by_mask(small)
    m.remove_unreferenced_vertices()
    steps.append({"step": "drop_small_components", "min_fraction": p.min_component_fraction,
                  "components": int(counts.size), "components_removed": int(tiny.sum()),
                  "faces_removed": int(small.sum())})

    m = m.filter_smooth_taubin(number_of_iterations=p.taubin_iterations, lambda_filter=p.taubin_lambda, mu=p.taubin_mu)
    steps.append({"step": "taubin_smoothing", "iterations": p.taubin_iterations, "lambda": p.taubin_lambda,
                  "mu": p.taubin_mu})

    if len(m.triangles) > p.target_triangles:
        m = m.simplify_quadric_decimation(p.target_triangles)
        m.remove_unreferenced_vertices()
        steps.append({"step": "quadric_decimation", "target_triangles": p.target_triangles})
    m.compute_vertex_normals()

    if len(m.triangles) == 0:
        raise ValueError("nothing left after the display cleanup")
    dev = _stats(_distances(source, np.asarray(m.vertices)))
    back = _distances(m, np.asarray(source.vertices))
    src_v = np.asarray(source.vertices)
    above = src_v[:, 2] >= p.floor_cut_m  # spill below the floor cut is removed on purpose; don't count it
    uncovered = float(np.mean(back[above] > p.uncovered_distance_m)) if above.any() else 0.0
    ok = dev["rms_m"] <= p.max_rms_deviation_m and dev["p95_m"] <= p.max_p95_deviation_m
    report = {
        "algorithm": ALGORITHM, "params": asdict(p), "steps": steps,
        "source_triangles": n0, "display_triangles": len(m.triangles),
        "deviation_display_to_source": dev,
        "source_uncovered_fraction": uncovered,
        "fidelity_check": "pass" if ok else "fail",
        "used": "cleaned" if ok else "source_unchanged",
        "note": "display only; no measurement uses this mesh. No holes were filled.",
    }
    if not ok:
        report["reason"] = (f"cleanup moved the surface too far (RMS {dev['rms_m'] * 1e3:.2f} mm, P95 "
                            f"{dev['p95_m'] * 1e3:.2f} mm; limits {p.max_rms_deviation_m * 1e3:.1f} / "
                            f"{p.max_p95_deviation_m * 1e3:.1f} mm): the untouched mesh is shown instead")
        out = o3d.geometry.TriangleMesh(source)
        out.compute_vertex_normals()
        return out, report
    return m, report
