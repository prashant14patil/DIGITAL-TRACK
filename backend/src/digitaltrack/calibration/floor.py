"""Floor-plane fit from 3-D points (e.g. depth back-projected from an empty-scene frame) and a floor frame.

RANSAC (seeded, reproducible) finds the dominant plane; the inliers are refit by least squares (SVD).
The plane is n . x + d = 0 with |n| = 1 and n oriented towards `up_hint` (default -y: "up" in the OpenCV
camera frame, whose y axis points down).

Library only: schema 0.2.0 has no field for a floor plane, so results are returned, not persisted.
The floor-aligned frame fixes origin and +Z (floor normal); its horizontal axes are provisional because the
body frame for manual self-rotation depends on the open pose model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FloorFitError(ValueError):
    pass


@dataclass(frozen=True)
class FloorPlane:
    normal: tuple[float, float, float]
    d: float
    n_points: int
    n_inliers: int
    inlier_rms_m: float

    @property
    def inlier_fraction(self) -> float:
        return self.n_inliers / self.n_points

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, float) @ np.asarray(self.normal) + self.d


def _plane_from(p: np.ndarray) -> tuple[np.ndarray, float] | None:
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    n /= norm
    return n, -float(n @ p[0])


def fit_floor_plane(points: np.ndarray, *, distance_threshold_m: float = 0.01, iterations: int = 500,
                    seed: int = 0, up_hint: tuple[float, float, float] = (0.0, -1.0, 0.0),
                    min_inlier_fraction: float = 0.2, max_normal_angle_deg: float | None = None) -> FloorPlane:
    """RANSAC plane. With `max_normal_angle_deg` (use only when `up_hint` is a measured gravity direction),
    candidate planes whose normal is further than that from `up_hint` are ignored, so a wall cannot win."""
    up = np.asarray(up_hint, float)
    cos_min = None if max_normal_angle_deg is None else float(np.cos(np.radians(max_normal_angle_deg)))
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise FloorFitError(f"points must be (N, 3), got {pts.shape}")
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 3:
        raise FloorFitError("at least 3 finite points are required")
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, 0
    for _ in range(iterations):
        cand = _plane_from(pts[rng.choice(len(pts), 3, replace=False)])
        if cand is None:
            continue
        n, d = cand
        if cos_min is not None and abs(n @ up) < cos_min:
            continue
        mask = np.abs(pts @ n + d) < distance_threshold_m
        c = int(mask.sum())
        if c > best_count:
            best_mask, best_count = mask, c
    if best_mask is None or best_count < 3:
        raise FloorFitError("no plane found (points are degenerate, e.g. collinear)")
    inl = pts[best_mask]
    centroid = inl.mean(axis=0)
    _, s, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    if s[1] < 1e-9:
        raise FloorFitError("inliers are collinear; plane is undefined")
    n = vt[2]
    if n @ np.asarray(up_hint, float) < 0:
        n = -n
    d = -float(n @ centroid)
    # final inlier set w.r.t. the refined plane
    mask = np.abs(pts @ n + d) < distance_threshold_m
    res = inl @ n + d
    plane = FloorPlane(tuple(float(x) for x in n), d, int(len(pts)), int(mask.sum()),
                       float(np.sqrt(np.mean(res ** 2))))
    if plane.inlier_fraction < min_inlier_fraction:
        raise FloorFitError(f"dominant plane holds only {plane.inlier_fraction:.1%} of points "
                            f"(minimum {min_inlier_fraction:.0%}); is the floor visible?")
    return plane


def floor_from_camera(plane: FloorPlane, origin_hint: np.ndarray) -> np.ndarray:
    """4x4 transform camera -> floor frame: origin = projection of `origin_hint` onto the floor, +Z = floor
    normal (up). +X = camera x axis projected onto the floor (PROVISIONAL horizontal axes)."""
    n = np.asarray(plane.normal, float)
    o = np.asarray(origin_hint, float)
    origin = o - plane.signed_distance(o[None])[0] * n
    x = np.array([1.0, 0.0, 0.0]) - n * n[0]
    if np.linalg.norm(x) < 0.3:  # camera x is close to the normal (device held portrait): use the view direction
        x = np.array([0.0, 0.0, 1.0]) - n * n[2]
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    r = np.stack([x, y, n])  # rows: floor axes expressed in camera coordinates
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = -r @ origin
    return t
