"""Self-rotation reconstruction: views -> stops -> floor -> segmentation -> poses -> TSDF -> visual outputs.

Inputs are in-memory `RGBDView`s (from an export adapter or the synthetic generator); nothing is read from or
written to raw data. Every output goes to a NEW directory `<out_root>/<run_id>/`. If poses cannot be estimated
reliably the run is written with status "fail" and no mesh: poses are never fabricated.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .. import __version__
from ..geometry.transforms import invert, transform_points
from ..reconstruction.tsdf import TSDFParams
from . import visualize as vis
from .fuse import fuse_views
from .pose import PoseParams, estimate_poses, se2_params
from .segment import (SegmentationError, SegmentParams, background_plate, estimate_floor, make_floor_frame,
                      segment_subject)
from .stops import StopParams, detect_stops, median_view, stop_threshold
from .views import RGBDView, backproject, valid_mask

PIPELINE_VERSION = "selfrot-pipeline-0.2"  # 0.2: subject ROI from the image centre when there is no background
NO_BACKGROUND_ROI_M = 0.75  # proposed: half an A-pose arm span plus margin; unvalidated
PARTIAL_VIEW_DROP_M = 0.10  # proposed: drop a stop whose subject top is this far below the median top


def default_tsdf() -> TSDFParams:
    return TSDFParams(voxel_length_m=0.005, sdf_trunc_m=0.02, depth_trunc_m=4.0, depth_min_m=0.2, min_confidence=1)


@dataclass(frozen=True)
class PipelineParams:
    segment: SegmentParams = field(default_factory=SegmentParams)
    stops: StopParams = field(default_factory=StopParams)
    pose: PoseParams = field(default_factory=PoseParams)
    tsdf: TSDFParams = field(default_factory=default_tsdf)
    detect_stops: bool = True  # False when every input view is already one stop
    stop_median: bool = True  # temporal median depth per stop (derived) instead of the single keyframe


@dataclass
class RunResult:
    run_dir: Path
    status: str  # pass | warn | fail
    pose: Any
    mesh: o3d.geometry.TriangleMesh | None
    camera_from_body: list[np.ndarray]
    key_views: list[RGBDView]
    log: dict


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_self_rotation(views: list[RGBDView], out_root: str | Path, *, background_views: list[RGBDView] | None = None,
                      params: PipelineParams | None = None, source: dict | None = None, run_id: str | None = None,
                      write_visuals: bool = True) -> RunResult:
    p = params or PipelineParams()
    run_id = run_id or f"selfrot_{_now()}"
    run_dir = Path(out_root) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for v in list(views) + list(background_views or []):
        v.validate()
    warnings: list[str] = []
    log: dict[str, Any] = {
        "run_id": run_id, "pipeline_version": PIPELINE_VERSION, "created_utc": _now(),
        "software": {"digitaltrack": __version__, "open3d": o3d.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0], "platform": platform.platform()},
        "source": source or {}, "parameters": vis.to_jsonable(asdict(p)), "n_input_views": len(views),
        "note": "Thresholds are proposed values checked on synthetic data only. Mesh dimensions are not validated "
                "measurements.",
    }

    # 1. stops
    if p.detect_stops and len(views) > 1:
        stops, scores = detect_stops(views, p.stops)
        log["stops"] = {"n": len(stops), "changed_fraction": scores.tolist(), "threshold": stop_threshold(scores, p.stops),
                        "stops": [asdict(s) for s in stops]}
        if not stops:
            return _fail(run_dir, log, "no stationary stops found (changed-pixel fraction never below "
                         f"{p.stops.max_changed_fraction} for {p.stops.min_frames} frames)", warnings)
        key = [median_view(views, s, p.stops) if p.stop_median else views[s.keyframe] for s in stops]
    else:
        key = list(views)
        log["stops"] = {"n": len(key), "note": "stop detection disabled: each input view is one stop"}

    # 2. floor
    # gravity from the device (if every view has it): required when the device was held portrait
    ups = [v.up_camera for v in list(key) + list(background_views or [])]
    up_kw: dict[str, Any] = {}
    if ups and all(u is not None for u in ups):
        u = np.mean(ups, axis=0)
        up_kw = {"up_hint": tuple(float(x) for x in u / np.linalg.norm(u)), "gravity": True}
        log["gravity_up_camera"] = list(up_kw["up_hint"])
    try:
        if background_views:
            floor = estimate_floor(background_views, p.segment, source="background_views", **up_kw)
        else:
            warnings.append("no empty-scene recording: floor fitted from subject views; background not subtracted")
            floor = estimate_floor(key, p.segment, source="subject_views", **up_kw)
    except SegmentationError as e:
        return _fail(run_dir, log, str(e), warnings)
    plate = background_plate(background_views, p.segment) if background_views else None
    log["floor"] = {"normal_camera": list(floor.plane.normal), "d_m": floor.plane.d, "tilt_deg": floor.tilt_deg,
                    "inlier_rms_m": floor.plane.inlier_rms_m, "inlier_fraction": floor.plane.inlier_fraction,
                    "source": floor.source}

    # 3. body-frame origin: floor projection of the subject centroid in the first stop
    k0 = key[0]
    seg = p.segment
    m0 = valid_mask(k0, min_confidence=seg.min_confidence, depth_min_m=seg.depth_min_m, depth_max_m=seg.depth_max_m)
    hint_z = float(np.median(k0.depth_m[m0]))
    if plate is None:
        # No background plate: walls and furniture would win the largest-cluster test. Assume the subject is
        # centred in the image (capture protocol), take its depth from the central patch and keep only points
        # within a horizontal radius of it. Logged as a warning; an empty-scene recording is the proper fix.
        h, w = k0.depth_m.shape
        centre = np.zeros_like(m0)
        centre[h // 4:3 * h // 4, 2 * w // 5:3 * w // 5] = True
        if (m0 & centre).sum() >= 50:
            hint_z = float(np.percentile(k0.depth_m[m0 & centre], 25))  # nearer quartile: the subject, not gaps
        if seg.roi_radius_m is None:
            seg = replace(seg, roi_radius_m=NO_BACKGROUND_ROI_M)
            warnings.append(f"no background plate: only points within {NO_BACKGROUND_ROI_M} m (horizontal) of "
                            "the subject at the image centre are used")
        log["segmentation_roi"] = {"radius_m": seg.roi_radius_m, "centre_depth_m": hint_z,
                                   "source": "central image patch of the first stop"}
    provisional = make_floor_frame(floor, np.array([0.0, 0.0, hint_z]))
    if plate is None and seg.cluster_select == "largest":
        # static thin objects (poles, door frames) can out-number the person: first view picks the widest
        # cluster, every later view the cluster nearest the subject's floor position (it turns in place)
        seg0, seg = replace(seg, cluster_select="widest"), replace(seg, cluster_select="nearest_origin")
    else:
        seg0 = seg
    try:
        s0 = segment_subject(k0, provisional, seg0, plate)
    except SegmentationError as e:
        return _fail(run_dir, log, f"first stop: {e}", warnings)
    c_floor = np.array([*s0.points_floor[:, :2].mean(0), 0.0])
    floor_from_cam = make_floor_frame(floor, transform_points(invert(provisional), c_floor[None])[0])
    log["floor_from_camera"] = floor_from_cam.tolist()

    # 4. segmentation of every stop
    clouds, masks, kept, seg_log = [], [], [], []
    for k, v in enumerate(key):
        try:
            s = segment_subject(v, floor_from_cam, seg, plate)
            clouds.append(s.points_floor)
            masks.append(s.mask)
            kept.append(k)
            seg_log.append({"stop": k, "view_index": v.index, **s.stats})
        except SegmentationError as e:
            warnings.append(f"stop {k} dropped: {e}")
            seg_log.append({"stop": k, "view_index": v.index, "error": str(e)})
    # partial views (subject still moving during the "stop", so the median depth has holes): the top of the
    # subject is clearly lower than in the other stops. Dropped and logged, never patched.
    if len(clouds) >= 5:
        tops = np.array([float(np.percentile(c[:, 2], 99)) for c in clouds])
        med_top = float(np.median(tops))
        ok = tops >= med_top - PARTIAL_VIEW_DROP_M
        for k in np.flatnonzero(~ok):
            warnings.append(f"stop {kept[k]} dropped: subject top {tops[k]:.2f} m vs median {med_top:.2f} m "
                            "(partial view; subject probably moving)")
            seg_log[kept[k]]["error"] = "partial view (top too low)"
        clouds = [c for c, o in zip(clouds, ok) if o]
        masks = [m for m, o in zip(masks, ok) if o]
        kept = [k for k, o in zip(kept, ok) if o]
        log["subject_top_m"] = {"median": med_top, "per_stop": tops.tolist(), "drop_below_median_m": PARTIAL_VIEW_DROP_M}
    log["segmentation"] = seg_log
    if len(kept) < 2:
        return _fail(run_dir, log, "fewer than 2 stops could be segmented", warnings)
    key = [key[k] for k in kept]

    # 5. poses
    pose = estimate_poses(clouds, p.pose)
    log["pose"] = {
        "algorithm_version": pose.algorithm_version, "status": pose.status, "checks": pose.checks,
        "loop_closure": pose.loop_closure, "edges": [e.summary() for e in pose.edges],
        "views": [{"stop": kept[k], "view_index": key[k].index, "theta_deg": float(np.degrees(t)), "x_m": x, "y_m": y,
                   "std": pose.view_std[k]} for k, (t, x, y) in enumerate(pose.se2_params)],
    }
    if pose.status == "fail" and not pose.floor_from_body:
        return _fail(run_dir, log, "poses could not be estimated reliably (see pose checks)", warnings, pose, clouds)

    cam_from_floor = invert(floor_from_cam)
    camera_from_body = [cam_from_floor @ fb for fb in pose.floor_from_body]

    # 6. fusion
    mesh, per_view, fuse_log = fuse_views(key, masks, camera_from_body, p.tsdf)
    log["fusion"] = fuse_log
    o3d.io.write_triangle_mesh(str(run_dir / "fused_mesh.ply"), mesh)
    (run_dir / "poses.json").write_text(json.dumps(log["pose"]["views"], indent=2) + "\n", encoding="utf-8")

    status = pose.status
    if warnings and status == "pass":
        status = "warn"
    note = {"pass": "All pose checks passed (proposed thresholds).",
            "warn": "Mesh produced, but some checks raised warnings: inspect the images before using it.",
            "fail": "Mesh produced but marked UNRELIABLE: a pose check failed."}[status]
    images = {}
    if write_visuals:
        images.update(vis.write_point_views(per_view, run_dir))
        images["slices"] = vis.write_slices(per_view, mesh, run_dir)
        images["poses"] = vis.write_pose_plot(pose, run_dir)
        images.update(vis.write_mesh_renders(mesh, per_view, run_dir))
    log.update({"status": status, "status_note": note, "warnings": warnings, "n_views": len(key),
                "mesh": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                         "bounds_m": [np.asarray(mesh.get_min_bound()).tolist(), np.asarray(mesh.get_max_bound()).tolist()]
                         if len(mesh.vertices) else None,
                         "note": "extent is a visual check only, not a measurement"},
                "images": images})
    if write_visuals:
        vis.write_report(run_dir, log, images)
    log["outputs"] = {f.name: _sha(f) for f in sorted(run_dir.iterdir()) if f.is_file()}
    (run_dir / "run_log.json").write_text(json.dumps(vis.to_jsonable(log), indent=2) + "\n", encoding="utf-8")
    return RunResult(run_dir, status, pose, mesh, camera_from_body, key, log)


def _fail(run_dir: Path, log: dict, reason: str, warnings: list[str], pose=None, clouds=None) -> RunResult:
    log.update({"status": "fail", "status_note": f"No mesh: {reason}", "warnings": warnings, "n_views": len(clouds or [])})
    images = {}
    if pose is not None:
        images["poses"] = vis.write_pose_plot(pose, run_dir)
    if clouds:
        images.update(vis.write_point_views(clouds, run_dir))  # floor frame, unaligned: shows what was segmented
    log["images"] = images
    vis.write_report(run_dir, log, images)
    log["outputs"] = {f.name: _sha(f) for f in sorted(run_dir.iterdir()) if f.is_file()}
    (run_dir / "run_log.json").write_text(json.dumps(vis.to_jsonable(log), indent=2) + "\n", encoding="utf-8")
    return RunResult(run_dir, "fail", pose, None, [], [], log)
