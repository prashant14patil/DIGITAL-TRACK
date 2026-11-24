"""Open3D TSDF fusion with EXPLICIT per-view poses (camera_from_body), integrating SUBJECT pixels only.

The turntable path (`reconstruction.tsdf.reconstruct_tsdf`) is left untouched; this module only differs in where
poses come from and in masking: static floor and background are excluded, otherwise they would be smeared around
the subject when the body-frame poses change from view to view. Same VoxelBlockGrid backend and mesh cleanup.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import open3d as o3d

from ..geometry.transforms import invert, transform_points
from ..reconstruction.tsdf import TSDFParams, _clean_mesh
from .views import RGBDView, backproject


def fuse_views(views: list[RGBDView], masks: list[np.ndarray], camera_from_body: list[np.ndarray],
               params: TSDFParams | None = None) -> tuple[o3d.geometry.TriangleMesh, list[np.ndarray], dict]:
    """Returns (cleaned mesh in the body frame, per-view subject points in the body frame, log)."""
    if not (len(views) == len(masks) == len(camera_from_body)):
        raise ValueError("views, masks and poses must have the same length")
    p = params or TSDFParams()
    dev = o3d.core.Device("CPU:0")
    vol = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight"), (o3d.core.float32, o3d.core.float32), ((1), (1)),
                                        p.voxel_length_m, p.block_resolution, p.block_count, dev)
    trunc_mult = p.sdf_trunc_m / p.voxel_length_m
    per_view_pts, per_frame = [], []
    for v, m, t in zip(views, masks, camera_from_body):
        k = v.intrinsics
        k_t = o3d.core.Tensor([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]], o3d.core.float64)
        depth = np.where(m, v.depth_m, 0.0).astype(np.float32)
        d_t = o3d.t.geometry.Image(o3d.core.Tensor(depth[:, :, None].copy(), o3d.core.float32))
        ext = o3d.core.Tensor(np.asarray(t, float), o3d.core.float64)
        coords = vol.compute_unique_block_coordinates(d_t, k_t, ext, 1.0, p.depth_trunc_m, trunc_mult)
        vol.integrate(coords, d_t, k_t, ext, 1.0, p.depth_trunc_m, trunc_mult)
        per_view_pts.append(transform_points(invert(t), backproject(v, m)))
        per_frame.append({"view_index": v.index, "subject_pixels": int(m.sum()), "camera_from_body": np.asarray(t).tolist()})
    mesh = vol.extract_triangle_mesh(weight_threshold=p.weight_threshold).to_legacy()
    n_raw = len(mesh.triangles)
    mesh, cleanup = _clean_mesh(mesh, p.min_cluster_fraction)
    mesh.compute_vertex_normals()
    log = {"backend": "open3d.t.geometry.VoxelBlockGrid", "open3d": o3d.__version__, "parameters": asdict(p),
           "integrated": "subject pixels only (floor and background masked out)",
           "triangles_extracted": n_raw, "cleanup": cleanup, "per_frame": per_frame}
    return mesh, per_view_pts, log
