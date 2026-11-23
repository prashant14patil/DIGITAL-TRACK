"""Open3D TSDF fusion (primary reconstruction baseline).

Raw session data is only read. All outputs and a full parameter/transform log go to
<session>/processing/<run_id>/ (or a caller-supplied directory).
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

from .. import OUTPUT_SCHEMA_VERSION, __version__
from ..dataset.loader import Session
from ..dataset.schemas import schema_errors
from ..dataset.writer import sha256_file, utc_now
from ..geometry.pointcloud import DepthFilter, valid_depth_mask, depth_to_camera_points
from ..geometry.transforms import camera_from_body, invert, transform_points
from ..quality.basic import QualityThresholds, compute_quality
from ..quality.registration import ViewConsistencyParams, view_consistency

ALGORITHM_VERSION = "tsdf-open3d-0.1"


@dataclass(frozen=True)
class TSDFParams:
    voxel_length_m: float = 0.004
    sdf_trunc_m: float = 0.02
    depth_trunc_m: float = 3.0
    depth_min_m: float = 0.1
    min_confidence: int = 1
    weight_threshold: float = 1.0  # minimum accumulated TSDF weight for a voxel to be meshed
    block_resolution: int = 16
    block_count: int = 200000
    min_cluster_fraction: float = 0.05  # drop mesh components smaller than this fraction of the largest
    pointcloud_voxel_m: float = 0.004  # downsampling of the saved fused point cloud


@dataclass
class ReconstructionResult:
    mesh: o3d.geometry.TriangleMesh
    point_cloud: o3d.geometry.PointCloud
    run_id: str
    run_dir: Path | None
    log: dict
    quality: dict


def _clean_mesh(mesh: o3d.geometry.TriangleMesh, min_fraction: float) -> tuple[o3d.geometry.TriangleMesh, dict]:
    n_before = len(mesh.triangles)
    if n_before == 0:
        return mesh, {"name": "remove_small_components", "triangles_before": 0, "triangles_after": 0}
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    keep = counts >= min_fraction * counts.max()
    mesh.remove_triangles_by_mask(~keep[labels])
    mesh.remove_unreferenced_vertices()
    return mesh, {"name": "remove_small_components", "min_cluster_fraction": min_fraction,
                  "components_before": int(counts.size), "components_after": int(keep.sum()),
                  "triangles_before": n_before, "triangles_after": len(mesh.triangles)}


def reconstruct_tsdf(session: Session, params: TSDFParams | None = None, *, run_id: str | None = None,
                     out_dir: str | Path | None = None, write: bool = True,
                     quality_thresholds: QualityThresholds | None = None,
                     consistency_params: ViewConsistencyParams | None = None) -> ReconstructionResult:
    if not session.has_depth:
        raise ValueError("TSDF reconstruction needs depth in every frame; this session is "
                         f"{session.input_type!r} (use an RGB reconstruction method)")
    if not session.angles_complete:
        raise ValueError("every frame needs a recorded turntable angle before reconstruction")
    params = params or TSDFParams()
    run_id = run_id or "tsdf_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dfilt = DepthFilter(params.min_confidence, params.depth_min_m, params.depth_trunc_m)
    k = session.intrinsics
    sign = session.angle_direction_sign
    t_ct = session.camera_from_turntable

    # Sparse hashed voxel-block TSDF (Open3D tensor API). The legacy ScalableTSDFVolume returned
    # empty meshes for every input in Open3D 0.20.0 on Windows, so it is not used.
    dev = o3d.core.Device("CPU:0")
    volume = o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight"), (o3d.core.float32, o3d.core.float32), ((1), (1)),
        params.voxel_length_m, params.block_resolution, params.block_count, dev)
    k_t = o3d.core.Tensor([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]], o3d.core.float64)
    trunc_mult = params.sdf_trunc_m / params.voxel_length_m

    per_frame, frame_stats, cloud_chunks = [], [], []
    for fr in session.frames:
        depth, conf = fr.load_depth(), fr.load_confidence()
        mask = valid_depth_mask(depth, conf, dfilt)
        depth_f = np.where(mask, depth, 0.0).astype(np.float32)
        t = camera_from_body(t_ct, fr.angle_deg, sign)
        depth_t = o3d.t.geometry.Image(o3d.core.Tensor(depth_f[:, :, None].copy(), o3d.core.float32))
        ext_t = o3d.core.Tensor(t, o3d.core.float64)
        coords = volume.compute_unique_block_coordinates(
            depth_t, k_t, ext_t, 1.0, params.depth_trunc_m, trunc_mult)
        volume.integrate(coords, depth_t, k_t, ext_t, 1.0, params.depth_trunc_m, trunc_mult)
        cloud_chunks.append(transform_points(invert(t), depth_to_camera_points(depth, k, mask)))
        frame_stats.append({"frame_id": fr.frame_id, "pixels_total": int(depth.size),
                            "pixels_valid": int(mask.sum()), "pixels_zero_or_invalid": int(depth.size - mask.sum())})
        per_frame.append({"frame_id": fr.frame_id, "recorded_angle_deg": float(fr.angle_deg),
                          "camera_from_body": t.tolist()})

    mesh = volume.extract_triangle_mesh(weight_threshold=params.weight_threshold).to_legacy()
    mesh, cleanup_step = _clean_mesh(mesh, params.min_cluster_fraction)

    pts = np.concatenate(cloud_chunks) if cloud_chunks else np.zeros((0, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    n_pts_raw = len(pts)
    pcd = pcd.voxel_down_sample(params.pointcloud_voxel_m)

    steps = [
        {"name": "depth_filter", "min_confidence": dfilt.min_confidence, "depth_min_m": dfilt.depth_min_m,
         "depth_max_m": dfilt.depth_max_m, "pixels_before": sum(s["pixels_total"] for s in frame_stats),
         "pixels_after": sum(s["pixels_valid"] for s in frame_stats)},
        {"name": "tsdf_integration", "backend": "open3d.t.geometry.VoxelBlockGrid",
         "voxel_length_m": params.voxel_length_m, "sdf_trunc_m": params.sdf_trunc_m,
         "trunc_voxel_multiplier": trunc_mult, "weight_threshold": params.weight_threshold,
         "n_frames": len(session.frames), "color": False},
        {"name": "extract_triangle_mesh", "triangles": None},  # filled below
        cleanup_step,
        {"name": "fused_pointcloud_downsample", "voxel_m": params.pointcloud_voxel_m,
         "points_before": n_pts_raw, "points_after": len(pcd.points)},
    ]
    steps[2]["triangles"] = cleanup_step["triangles_before"]

    # Diagnostic only: corrections are measured, never applied to the mesh or point cloud.
    consistency = view_consistency(cloud_chunks, [fr.frame_id for fr in session.frames], mesh, consistency_params)
    steps.append({"name": "view_consistency_icp", "applied_to_geometry": False,
                  "algorithm_version": consistency["algorithm_version"], **consistency["parameters"],
                  "summary": consistency["summary"]})

    quality = compute_quality(session, frame_stats, mesh, run_id, quality_thresholds, consistency)

    session_json = session.root / "session.json"
    log = {
        "schema_version": OUTPUT_SCHEMA_VERSION, "run_id": run_id, "stage": "tsdf_reconstruction",
        "created_utc": utc_now(),
        "software": {"digitaltrack": __version__, "algorithm_version": ALGORITHM_VERSION,
                     "open3d": o3d.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0], "platform": platform.platform()},
        "input": {"session_id": session.session_id, "calibration_id": session.calibration_id,
                  "dataset_schema_version_on_disk": session.schema_version_on_disk,
                  "session_json_sha256": sha256_file(session_json)},
        "parameters": asdict(params),
        "transforms": {"camera_from_turntable": t_ct.tolist(), "angle_direction_sign": sign,
                       "per_frame": per_frame},
        "steps": steps, "outputs": {},
    }

    run_dir = None
    if write:
        run_dir = Path(out_dir) if out_dir else session.root / "processing" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        o3d.io.write_triangle_mesh(str(run_dir / "fused_mesh.ply"), mesh)
        o3d.io.write_point_cloud(str(run_dir / "pointcloud_filtered.ply"), pcd)
        (run_dir / "quality_report.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
        log["outputs"] = {n: sha256_file(run_dir / n)
                          for n in ("fused_mesh.ply", "pointcloud_filtered.ply", "quality_report.json")}
        errs = schema_errors(log, "processing_log")
        if errs:
            raise ValueError(f"processing log violates schema: {errs}")
        (run_dir / "run_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return ReconstructionResult(mesh, pcd, run_id, run_dir, log, quality)
