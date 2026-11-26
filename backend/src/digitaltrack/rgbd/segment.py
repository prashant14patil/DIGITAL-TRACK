"""Floor plane and subject segmentation for a FIXED camera.

Everything static (floor, walls, furniture) must be removed before registration: static content looks identical
in every view and would pull registration toward "no motion".

* Floor: dominant plane (calibration.floor.fit_floor_plane, other workstream) of points pooled from the
  empty-scene views (preferred) or the subject views, and it must face up (angle to `up_hint` checked).
* Subject: valid depth -> drop floor band -> drop background (background depth plate, if given) -> drop points
  outside the height/range limits -> largest connected cluster (DBSCAN). Returns pixel mask + floor-frame points.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d

from ..calibration.floor import FloorFitError, FloorPlane, fit_floor_plane, floor_from_camera
from ..geometry.transforms import transform_points
from .views import RGBDView, backproject, valid_mask


class SegmentationError(ValueError):
    pass


@dataclass(frozen=True)
class SegmentParams:
    min_confidence: int | None = 1  # applied only if the view has a confidence map
    depth_min_m: float = 0.2
    depth_max_m: float = 4.0
    floor_fit_threshold_m: float = 0.01
    floor_max_tilt_deg: float = 35.0  # floor normal vs camera "up"; larger => the dominant plane is probably a wall
    floor_band_m: float = 0.015  # points closer than this to the floor are floor
    max_height_m: float = 2.3
    background_margin_m: float = 0.04  # foreground = closer to the camera than the background plate by this much
    roi_radius_m: float | None = None  # optional horizontal radius around the floor origin
    dbscan_eps_m: float = 0.03
    dbscan_min_points: int = 10
    min_subject_points: int = 300
    # which connected cluster is the subject: "largest" (default), "widest" (largest horizontal extent; first view
    # without a background plate), "nearest_origin" (closest to the floor origin; later views without a plate)
    cluster_select: str = "largest"
    union_radius_m: float = 0.5  # non-"largest" modes: clusters with a centroid this close to the centre are joined


@dataclass
class FloorEstimate:
    plane: FloorPlane
    tilt_deg: float  # angle between floor normal and up_hint
    source: str  # "background_views" | "subject_views"


@dataclass
class SubjectCloud:
    view_index: int
    mask: np.ndarray  # (H, W) bool, pixels belonging to the subject
    points_floor: np.ndarray  # (N, 3) in the floor frame
    stats: dict


def estimate_floor(views: list[RGBDView], params: SegmentParams, *, source: str,
                   up_hint=(0.0, -1.0, 0.0), max_points: int = 60000, seed: int = 0,
                   gravity: bool = False) -> FloorEstimate:
    """`gravity=True`: `up_hint` is a measured gravity direction, so only near-horizontal planes are candidates
    and the floor may be a smaller share of the points (portrait framing shows little floor)."""
    rng = np.random.default_rng(seed)
    chunks = []
    for v in views:
        m = valid_mask(v, min_confidence=params.min_confidence, depth_min_m=params.depth_min_m,
                       depth_max_m=params.depth_max_m)
        chunks.append(backproject(v, m))
    pts = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    try:
        extra = {"max_normal_angle_deg": params.floor_max_tilt_deg, "min_inlier_fraction": 0.05} if gravity else {}
        plane = fit_floor_plane(pts, distance_threshold_m=params.floor_fit_threshold_m, seed=seed, up_hint=up_hint,
                                **extra)
    except FloorFitError as e:
        raise SegmentationError(f"floor plane not found: {e}") from e
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(plane.normal, up_hint), -1, 1))))
    if tilt > params.floor_max_tilt_deg:
        raise SegmentationError(f"dominant plane is tilted {tilt:.1f} deg from {'gravity' if gravity else 'camera'}"
                                "-up; it is probably a wall, "
                                "not the floor. Frame the floor around the subject, or record an empty scene.")
    return FloorEstimate(plane, tilt, source)


def floor_with_vertical(floor: FloorEstimate, views: list[RGBDView], up_camera, params: SegmentParams,
                        max_points: int = 60000, seed: int = 0) -> tuple[FloorEstimate, dict]:
    """Replace the fitted floor NORMAL by a measured vertical (e.g. IMU gravity) and re-estimate only the floor
    HEIGHT along it: median of up . p over the points within `floor_fit_threshold_m` of the fitted plane.

    Reason: in portrait framing the floor is a thin strip at the image edge, so its fitted normal is weakly
    constrained (first real export: 1.5 deg from IMU gravity, while the person's rotation axis matched gravity
    within 0.7 deg). Returns the new estimate and a log of both normals."""
    up = np.asarray(up_camera, float)
    up = up / np.linalg.norm(up)
    rng = np.random.default_rng(seed)
    pts = np.concatenate([backproject(v, valid_mask(v, min_confidence=params.min_confidence,
                                                     depth_min_m=params.depth_min_m, depth_max_m=params.depth_max_m))
                          for v in views])
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    on = np.abs(floor.plane.signed_distance(pts)) < params.floor_fit_threshold_m
    if on.sum() < 100:
        raise SegmentationError("too few floor points to place the floor along the measured vertical")
    h = pts[on] @ up
    level = float(np.median(h))
    n_fit = np.asarray(floor.plane.normal, float)
    angle = float(np.degrees(np.arccos(np.clip(abs(n_fit @ up), -1, 1))))
    plane = FloorPlane(tuple(float(x) for x in up), -level, floor.plane.n_points, int(on.sum()),
                       float(np.sqrt(np.mean((h - level) ** 2))))
    log = {"vertical_source": "measured (IMU gravity)", "fitted_normal_camera": n_fit.tolist(),
           "vertical_camera": up.tolist(), "angle_fitted_vs_vertical_deg": angle,
           "floor_points": int(on.sum()), "floor_rms_along_vertical_m": plane.inlier_rms_m}
    return FloorEstimate(plane, angle, floor.source + " + measured vertical"), log


def background_plate(views: list[RGBDView], params: SegmentParams) -> np.ndarray:
    """Per-pixel median depth of empty-scene views (0 where fewer than half the views are valid)."""
    stack = []
    for v in views:
        m = valid_mask(v, min_confidence=params.min_confidence, depth_min_m=params.depth_min_m,
                       depth_max_m=params.depth_max_m)
        stack.append(np.where(m, v.depth_m, np.nan))
    s = np.stack(stack)
    n_valid = np.isfinite(s).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(s, axis=0)
    return np.where(n_valid * 2 >= len(views), med, 0.0).astype(np.float32)


def segment_subject(view: RGBDView, floor_from_cam: np.ndarray, params: SegmentParams,
                    background: np.ndarray | None = None) -> SubjectCloud:
    m = valid_mask(view, min_confidence=params.min_confidence, depth_min_m=params.depth_min_m,
                   depth_max_m=params.depth_max_m)
    stats = {"pixels_valid": int(m.sum())}
    if background is not None:
        if background.shape != view.depth_m.shape:
            raise SegmentationError("background plate size differs from the view")
        fg = (background <= 0) | (view.depth_m < background - params.background_margin_m)
        m &= fg
        stats["pixels_after_background"] = int(m.sum())
    v_idx, u_idx = np.nonzero(m)
    pts = transform_points(floor_from_cam, backproject(view, m))
    keep = (pts[:, 2] > params.floor_band_m) & (pts[:, 2] < params.max_height_m)
    if params.roi_radius_m is not None:
        keep &= np.hypot(pts[:, 0], pts[:, 1]) < params.roi_radius_m
    pts, v_idx, u_idx = pts[keep], v_idx[keep], u_idx[keep]
    stats["points_after_floor_height_roi"] = int(len(pts))
    if len(pts) < params.min_subject_points:
        raise SegmentationError(f"view {view.index}: only {len(pts)} subject points after removing floor/background")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.asarray(pcd.cluster_dbscan(params.dbscan_eps_m, params.dbscan_min_points))
    if labels.max() < 0:
        raise SegmentationError(f"view {view.index}: no connected subject cluster")
    counts = np.bincount(labels[labels >= 0])
    best = int(np.argmax(counts))
    sel = labels == best
    if params.cluster_select != "largest":
        # A subject can fall apart into several clusters (depth gaps at a moving arm, median holes), so the
        # subject is the UNION of every cluster whose centroid lies within `union_radius_m` (horizontally) of a
        # centre: the floor origin ("nearest_origin": it turns in place) or the widest cluster ("widest": first
        # view; poles, door frames and wall edges are thin). Clusters further away (other objects) stay out.
        cent = np.array([pts[labels == l][:, :2].mean(0) for l in range(counts.size)])
        big = [l for l in range(counts.size) if counts[l] >= params.min_subject_points]
        if params.cluster_select == "nearest_origin":
            centre = np.zeros(2)
        elif params.cluster_select == "widest":
            wid = lambda l: float(np.ptp(pts[labels == l][:, :2], axis=0).max())  # noqa: E731
            centre = cent[max(big or [best], key=wid)]
        else:
            raise ValueError(f"unknown cluster_select {params.cluster_select!r}")
        near = np.flatnonzero(np.hypot(*(cent - centre).T) < params.union_radius_m)
        if near.size:
            sel = np.isin(labels, near)
        stats["clusters_joined"] = int(near.size)
    stats["cluster_select"] = params.cluster_select
    stats.update({"clusters": int(counts.size), "subject_points": int(sel.sum()),
                  "second_cluster_points": int(np.sort(counts)[-2]) if counts.size > 1 else 0})
    if sel.sum() < params.min_subject_points:
        raise SegmentationError(f"view {view.index}: largest cluster has only {int(sel.sum())} points")
    mask = np.zeros_like(m)
    mask[v_idx[sel], u_idx[sel]] = True
    return SubjectCloud(view.index, mask, pts[sel], stats)


def segment_subject_plate(view: RGBDView, floor_from_cam: np.ndarray, background: np.ndarray,
                          params: SegmentParams, *, min_component_fraction: float = 0.02,
                          join_radius_m: float = 0.6) -> SubjectCloud:
    """Subject segmentation when an empty-scene background plate EXISTS (image-space connected components).

    The DBSCAN path (`segment_subject`) keeps one 3-D cluster; on the first real recording the person often fell
    apart into torso / legs / arm clusters and only one survived (masks of ~1500 instead of ~4500 pixels). With a
    plate, foreground is well defined, so: foreground (closer than the plate by `background_margin_m`) ->
    floor band / height limits -> 8-connected components in the image -> keep the largest component and every
    component of at least `min_component_fraction` of it whose floor position is within `join_radius_m`
    (horizontal) of it. Small specks (plate noise) are dropped, nothing is added.
    """
    import cv2

    if background.shape != view.depth_m.shape:
        raise SegmentationError("background plate size differs from the view")
    m = valid_mask(view, min_confidence=params.min_confidence, depth_min_m=params.depth_min_m,
                   depth_max_m=params.depth_max_m)
    stats = {"pixels_valid": int(m.sum())}
    m &= (background <= 0) | (view.depth_m < background - params.background_margin_m)
    stats["pixels_after_background"] = int(m.sum())
    v_idx, u_idx = np.nonzero(m)
    pts = transform_points(floor_from_cam, backproject(view, m))
    keep = (pts[:, 2] > params.floor_band_m) & (pts[:, 2] < params.max_height_m)
    if params.roi_radius_m is not None:
        keep &= np.hypot(pts[:, 0], pts[:, 1]) < params.roi_radius_m
    fg = np.zeros_like(m)
    fg[v_idx[keep], u_idx[keep]] = True
    n, lab, cc_stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise SegmentationError(f"view {view.index}: no foreground in front of the background plate")
    pts_img = np.zeros(m.shape + (3,), np.float64)
    pts_img[v_idx[keep], u_idx[keep]] = pts[keep]
    areas = cc_stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    c_big = pts_img[lab == big][:, :2].mean(0)
    sel = [big]
    for c in range(1, n):
        if c != big and areas[c - 1] >= min_component_fraction * areas[big - 1]:
            if np.hypot(*(pts_img[lab == c][:, :2].mean(0) - c_big)) < join_radius_m:
                sel.append(c)
    mask = np.isin(lab, sel)
    stats.update({"components": int(n - 1), "components_joined": len(sel), "subject_points": int(mask.sum()),
                  "method": "background plate + image connected components"})
    if mask.sum() < params.min_subject_points:
        raise SegmentationError(f"view {view.index}: subject has only {int(mask.sum())} pixels")
    return SubjectCloud(view.index, mask, pts_img[mask], stats)


def make_floor_frame(floor: FloorEstimate, origin_hint_cam: np.ndarray) -> np.ndarray:
    return floor_from_camera(floor.plane, origin_hint_cam)


def params_dict(p: SegmentParams) -> dict:
    return asdict(p)
