"""Multi-view consistency of a fused surface: free-space violations (ghost removal) and per-pixel residuals.

Why: only SUBJECT pixels are integrated (the static background would smear around the turning body), and
projective TSDF only updates voxels near observed surfaces. Nothing therefore erases surfaces that existed in a few
views only: an arm held in different positions leaves several "ghost" arms, and static floor/wall pixels that slip
into a mask leave arcs. Those surfaces are CONTRADICTED by other views, which see straight through them.

For a surface point X and a view (camera_from_body, depth, subject mask), with z = depth of X in that camera and
d = measured depth at X's pixel:
  support    : subject pixel and |d - z| <= margin
  violation  : d > z + margin (the view sees something BEHIND X, so X is empty space), on a subject pixel or a
               background pixel (the subject mask is dilated by `edge_px` so silhouette edges never count)
  no evidence: d < z - margin (X is occluded) or no valid depth.
A point is removed when violations > violation_ratio * support + min_violations (a clear majority: a thin limb
that LiDAR misses in a few views keeps its support). This only deletes geometry; nothing is added or moved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import open3d as o3d

from .views import RGBDView


@dataclass(frozen=True)
class ConsistencyParams:
    # Tolerant on purpose: culling exists to remove GROSS artefacts (ghost arms, floor/wall arcs tens of cm away).
    # On the first real recording side views scatter ~2-3 cm (yaw ~ +-5 deg at the arms); a 3 cm margin with a
    # simple majority removed the true body side together with its scattered copies.
    margin_m: float = 0.05
    edge_px: int = 2
    min_violations: int = 3
    violation_ratio: float = 2.0  # cull when violations > violation_ratio * support + min_violations
    view_stride: int = 2  # use every n-th fused view for the vote (speed)
    pixel_max_residual_m: float = 0.025  # pass-2 pixel exclusion: point farther than this from the consensus mesh
    min_view_inlier_fraction: float = 0.6  # views below this are dropped from pass 2 (pose or segmentation error)
    view_inlier_m: float = 0.015


def free_space_votes(points: np.ndarray, views: list[RGBDView], subject_masks: list[np.ndarray],
                     camera_from_body: list[np.ndarray], p: ConsistencyParams) -> tuple[np.ndarray, np.ndarray]:
    sup = np.zeros(len(points), np.int32)
    vio = np.zeros(len(points), np.int32)
    k_dil = np.ones((2 * p.edge_px + 1, 2 * p.edge_px + 1), np.uint8)
    for v, m, t in list(zip(views, subject_masks, camera_from_body))[::p.view_stride]:
        k = v.intrinsics
        x = points @ t[:3, :3].T + t[:3, 3]
        z = x[:, 2]
        ok = z > 0.1
        u = np.round(k.fx * x[:, 0] / np.where(ok, z, 1) + k.cx).astype(int)
        r = np.round(k.fy * x[:, 1] / np.where(ok, z, 1) + k.cy).astype(int)
        ok &= (u >= 0) & (u < k.width) & (r >= 0) & (r < k.height)
        idx = np.flatnonzero(ok)
        d = v.depth_m[r[idx], u[idx]]
        valid = np.isfinite(d) & (d > 0)
        if v.confidence is not None:
            valid &= v.confidence[r[idx], u[idx]] >= 1
        subj = m[r[idx], u[idx]]
        near_subj = cv2.dilate(m.astype(np.uint8), k_dil).astype(bool)[r[idx], u[idx]]
        zz = z[idx]
        s = valid & subj & (np.abs(d - zz) <= p.margin_m)
        behind = valid & (d > zz + p.margin_m)
        vi = behind & (subj | ~near_subj)
        sup[idx[s]] += 1
        vio[idx[vi]] += 1
    return sup, vio


def cull_mesh(mesh: o3d.geometry.TriangleMesh, sup: np.ndarray, vio: np.ndarray, p: ConsistencyParams,
              min_cluster_fraction: float = 0.05) -> tuple[o3d.geometry.TriangleMesh, dict]:
    bad = vio > p.violation_ratio * sup + p.min_violations
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_vertices_by_mask(bad)
    n_before = len(out.triangles)
    if n_before:
        labels, counts, _ = out.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        keep = counts >= min_cluster_fraction * counts.max()
        out.remove_triangles_by_mask(~keep[labels])
        out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out, {"vertices_culled": int(bad.sum()), "vertices_before": int(len(sup)),
                 "triangles_after_cull": n_before, "triangles_after_components": len(out.triangles),
                 "rule": f"violations > {p.violation_ratio} x support + {p.min_violations}"}


def pixel_residuals(mesh: o3d.geometry.TriangleMesh, points_body: np.ndarray) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points_body.astype(np.float32))).numpy()


def params_dict(p: ConsistencyParams) -> dict:
    return asdict(p)
