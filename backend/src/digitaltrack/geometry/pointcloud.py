"""Depth -> confidence-filtered camera-space points -> common body frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset.loader import Intrinsics, Session
from .transforms import camera_from_body, invert, transform_points


@dataclass(frozen=True)
class DepthFilter:
    min_confidence: int = 1  # depth-branch confidence levels: 0 low, 1 medium, 2 high
    depth_min_m: float = 0.1
    depth_max_m: float = 3.0


def valid_depth_mask(depth: np.ndarray, confidence: np.ndarray | None, f: DepthFilter) -> np.ndarray:
    m = np.isfinite(depth) & (depth > f.depth_min_m) & (depth < f.depth_max_m)
    if confidence is not None:
        m &= confidence >= f.min_confidence
    return m


def depth_to_camera_points(depth: np.ndarray, k: Intrinsics, mask: np.ndarray) -> np.ndarray:
    """Back-project masked depth pixels using intrinsics of the DEPTH image (index convention)."""
    if depth.shape != (k.height, k.width):
        raise ValueError(f"depth {depth.shape} does not match intrinsics {(k.height, k.width)}")
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    return np.stack([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z], axis=1)


def frame_points_in_body_frame(session: Session, frame_index: int, f: DepthFilter,
                               sign: int | None = None) -> tuple[np.ndarray, dict]:
    fr = session.frames[frame_index]
    depth, conf = fr.load_depth(), fr.load_confidence()
    mask = valid_depth_mask(depth, conf, f)
    pts_cam = depth_to_camera_points(depth, session.intrinsics, mask)
    s = session.angle_direction_sign if sign is None else sign
    t = camera_from_body(session.camera_from_turntable, fr.angle_deg, s)
    pts = transform_points(invert(t), pts_cam)
    stats = {"frame_id": fr.frame_id, "pixels_total": int(depth.size),
             "pixels_valid": int(mask.sum()), "pixels_zero_or_invalid": int(depth.size - mask.sum())}
    return pts, stats


def session_points_in_body_frame(session: Session, f: DepthFilter | None = None,
                                 sign: int | None = None) -> tuple[np.ndarray, list[dict]]:
    f = f or DepthFilter()
    chunks, stats = [], []
    for i in range(len(session.frames)):
        pts, st = frame_points_in_body_frame(session, i, f, sign)
        chunks.append(pts)
        stats.append(st)
    return (np.concatenate(chunks) if chunks else np.zeros((0, 3))), stats
