"""Multi-view consistency check (registration residual), DIAGNOSTIC ONLY.

Each frame's body-frame points are rigidly ICP-aligned (point-to-plane) to the fused mesh.
With correct turntable calibration every view already agrees with the fused surface, so the
correction is small. A wrong calibration makes views disagree and the correction grows.
Corrections are reported, never applied to the geometry.

Known blind spot (see docs/phase0_verification_report.md): calibration errors along the
camera's line of sight move every view along its own viewing direction by the same amount.
The views remain mutually consistent and describe a uniformly larger or smaller body, so no
self-consistency check can detect them. They require a known-dimension reference object.

Rotation about the body Z axis (yaw) is reported but not checked: for rotationally symmetric
bodies yaw is unconstrained by geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

ALGORITHM_VERSION = "view-consistency-icp-0.1"


@dataclass(frozen=True)
class ViewConsistencyParams:
    source_voxel_m: float = 0.005
    max_correspondence_m: float = 0.03
    max_iterations: int = 50
    min_points: int = 200


def _mesh_target(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    m = o3d.geometry.TriangleMesh(mesh)  # copy: never add normals to the saved mesh
    m.compute_vertex_normals()
    tgt = o3d.geometry.PointCloud(m.vertices)
    tgt.normals = m.vertex_normals
    return tgt


def view_consistency(frame_points: list[np.ndarray], frame_ids: list[int], mesh: o3d.geometry.TriangleMesh,
                     params: ViewConsistencyParams | None = None) -> dict:
    p = params or ViewConsistencyParams()
    per_frame, skipped = [], []
    if len(mesh.triangles) == 0:
        return {"algorithm_version": ALGORITHM_VERSION, "parameters": asdict(p), "per_frame": [],
                "skipped_frames": list(frame_ids), "summary": None}
    tgt = _mesh_target(mesh)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=p.max_iterations)
    for fid, pts in zip(frame_ids, frame_points):
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(p.source_voxel_m)
        if len(src.points) < p.min_points:
            skipped.append(fid)
            continue
        r = o3d.pipelines.registration.registration_icp(src, tgt, p.max_correspondence_m, np.eye(4), est, crit)
        t = np.asarray(r.transformation)
        rv = Rotation.from_matrix(t[:3, :3]).as_rotvec()
        per_frame.append({
            "frame_id": int(fid), "n_points": len(src.points),
            "translation_m": float(np.linalg.norm(t[:3, 3])),
            "tilt_deg": float(np.degrees(np.hypot(rv[0], rv[1]))),
            "yaw_deg": float(np.degrees(abs(rv[2]))),
            "inlier_rmse_m": float(r.inlier_rmse), "fitness": float(r.fitness),
            "correction": t.tolist(),
        })
    summary = None
    if per_frame:
        col = lambda k: np.array([f[k] for f in per_frame])
        summary = {
            "n_frames": len(per_frame),
            "median_translation_m": float(np.median(col("translation_m"))),
            "max_translation_m": float(col("translation_m").max()),
            "median_tilt_deg": float(np.median(col("tilt_deg"))),
            "max_tilt_deg": float(col("tilt_deg").max()),
            "median_yaw_deg": float(np.median(col("yaw_deg"))),
            "median_inlier_rmse_m": float(np.median(col("inlier_rmse_m"))),
            "median_fitness": float(np.median(col("fitness"))),
            "min_fitness": float(col("fitness").min()),
        }
    return {"algorithm_version": ALGORITHM_VERSION, "parameters": asdict(p), "applied_to_geometry": False,
            "per_frame": per_frame, "skipped_frames": skipped, "summary": summary}
