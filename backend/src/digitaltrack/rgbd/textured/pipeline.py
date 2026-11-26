"""Stray Scanner recording of a self-rotating person -> metric mesh, exports and quality report.

Steps: audit + RGB/depth sync check -> gravity (odometry, IMU cross-check) -> pauses -> confidence-filtered
temporal-median depth per pause -> floor height -> subject segmentation -> camera-motion check -> texture
tracking + pose graph -> registered point clouds -> TSDF -> topology-only mesh cleanup -> GLB/PLY/OBJ -> report.

Raw files are only read. Outputs go to a NEW directory <out_root>/<run_id>/. If poses cannot be estimated no mesh
is written. The mesh is in metres in a Z-up body frame (floor at z = 0); it is NOT converted to glTF's Y-up
convention (same as the existing pipeline). No body measurement is computed and no accuracy is claimed.
"""

from __future__ import annotations

import hashlib
import html
import json
import platform
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import open3d as o3d
import trimesh

from ... import __version__
from ...calibration.floor import FloorFitError, FloorPlane, fit_floor_plane, floor_from_camera
from ...experimental.display_mesh import clean_display_mesh
from ...geometry.transforms import invert, transform_points
from ...quality.registration import view_consistency
from ...reconstruction.tsdf import TSDFParams
from .. import visualize as vis
from ..fuse import fuse_views
from ..pose import PoseParams, estimate_poses, wrap
from ..segment import SegmentationError, SegmentParams, SubjectCloud, segment_subject
from ..stops import StopParams, detect_stops, stop_threshold
from ..views import RGBDView, backproject
from . import evaluate as ev
from .depth import DepthFilterParams, filter_depth, temporal_median
from .sensors import Export, camera_motion, check_sync, gravity, iter_frames, read_export
from .tracking import ALGORITHM_VERSION as TRACKING_VERSION
from .tracking import Features, PoseSolution, TrackingParams, extract_features, track

PIPELINE_VERSION = "textured-rgbd-0.1"


def _default_tsdf() -> TSDFParams:
    return TSDFParams(voxel_length_m=0.005, sdf_trunc_m=0.02, depth_trunc_m=4.0, depth_min_m=0.2)


def _default_stops() -> StopParams:
    return StopParams(adaptive=True, min_frames=6)


@dataclass(frozen=True)
class ReconParams:
    depth: DepthFilterParams = field(default_factory=DepthFilterParams)
    tracking: TrackingParams = field(default_factory=TrackingParams)
    tsdf: TSDFParams = field(default_factory=_default_tsdf)
    stops: StopParams = field(default_factory=_default_stops)
    stop_scan_fps: float = 6.0
    roi_radius_m: float = 0.75
    floor_removal_m: float = 0.02  # above the fitted depth floor; this also trims the lowest 2 cm of the shoes
    # LiDAR depth bridges the gap between the legs, which joins floor/wall behind to the subject cluster. Below
    # lower_roi_height_m only points near the torso axis are kept (assumes a normal stance; counted in the report).
    lower_roi_height_m: float = 0.5
    lower_roi_radius_m: float = 0.35
    partial_view_drop_m: float = 0.10
    sensitivity_intervals_s: tuple[float, ...] = (0.35, 0.75)
    compare_depth_only: bool = True
    check_sync: bool = True
    # proposed status limits; unvalidated
    max_loop_error_deg: float = 10.0
    max_pose_sensitivity_deg: float = 3.0
    max_pose_sensitivity_mm: float = 15.0
    max_gravity_disagreement_deg: float = 2.0
    max_coverage_gap_deg: float = 45.0
    max_cross_view_median_mm: float = 10.0


@dataclass
class ReconResult:
    run_dir: Path
    status: str  # pass | warn | fail
    report: dict
    mesh: o3d.geometry.TriangleMesh | None = None
    floor_from_body: list[np.ndarray] = field(default_factory=list)
    pause_frames: list[int] = field(default_factory=list)
    floor_from_camera: np.ndarray | None = None


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(name: str, status: str, value: Any, limit: Any, message: str) -> dict:
    return {"name": name, "status": status, "value": value, "threshold": limit, "message": message}


# ---- stages -----------------------------------------------------------------------------------

def find_pauses(exp: Export, p: ReconParams) -> tuple[list[tuple[int, int, int]], dict]:
    """[(first frame, last frame, key frame)] of the still periods, from raw depth sampled at stop_scan_fps."""
    fps = 1.0 / float(np.median(np.diff(exp.t)))
    step = max(1, round(fps / p.stop_scan_fps))
    idx = list(range(0, len(exp), step))
    views = [RGBDView(i, exp.depth_raw(i).astype(np.float32) * 0.001, exp.intrinsics_depth(i)) for i in idx]
    stops, scores = detect_stops(views, p.stops)
    pauses = [(idx[s.start], idx[s.end], idx[s.keyframe]) for s in stops]
    info = {"n": len(stops), "frame_step": step, "changed_fraction_threshold": stop_threshold(scores, p.stops),
            "pauses": [{"frames": [a, b], "key_frame": k, "seconds": round(float(exp.t[b] - exp.t[a]), 2)}
                       for a, b, k in pauses]}
    return pauses, info


def _single_view(exp: Export, i: int, p: DepthFilterParams) -> tuple[RGBDView, dict]:
    depth, stats = filter_depth(exp.depth_raw(i), exp.confidence(i), p)
    return RGBDView(i, depth, exp.intrinsics_depth(i), timestamp_s=float(exp.t[i])), stats


def pause_view(exp: Export, first: int, last: int, key: int, p: DepthFilterParams) -> tuple[RGBDView, dict]:
    frames = np.unique(np.linspace(first, last, min(p.median_max_frames, last - first + 1)).round().astype(int))
    depths, counts = [], {"pixels": 0, "out_of_range": 0, "low_confidence": 0, "flying": 0, "kept": 0}
    for i in frames:
        d, s = filter_depth(exp.depth_raw(int(i)), exp.confidence(int(i)), p)
        depths.append(d)
        for k in counts:
            counts[k] += s[k]
    med, ms = temporal_median(depths, p)
    view = RGBDView(key, med, exp.intrinsics_depth(key), timestamp_s=float(exp.t[key]), derived=True,
                    source={"method": "temporal_median", "frames": [int(frames[0]), int(frames[-1])], "n": len(frames)})
    return view, {**counts, "median": ms}


def fit_floor(views: list[RGBDView], up: np.ndarray) -> tuple[float, FloorPlane, dict]:
    """Floor height along gravity (h = up . x, camera at 0) and the plane fitted to the depth floor points."""
    pts = np.concatenate([backproject(v, v.depth_m > 0) for v in views])
    pts = pts[pts @ up < -0.5]
    rng = np.random.default_rng(0)
    if len(pts) > 60000:
        pts = pts[rng.choice(len(pts), 60000, replace=False)]
    plane = fit_floor_plane(pts, distance_threshold_m=0.015, up_hint=tuple(up), max_normal_angle_deg=15.0,
                            min_inlier_fraction=0.03, iterations=1500)
    inl = pts[np.abs(plane.signed_distance(pts)) < 0.015]
    h = inl @ up
    height = float(np.median(h))
    return height, plane, {"camera_height_m": round(-height, 4), "inliers": int(len(inl)), "inlier_rms_m": plane.inlier_rms_m,
                    "fitted_normal_vs_gravity_deg": float(np.degrees(np.arccos(np.clip(np.dot(plane.normal, up), -1, 1)))),
                    "note": "the vertical axis is gravity; only the height is taken from the depth fit"}


def remove_floor(view: RGBDView, plane: FloorPlane, height_m: float) -> RGBDView:
    """Drop pixels lower than height_m above the fitted floor plane (DERIVED view)."""
    k = view.intrinsics
    u, v = np.meshgrid(np.arange(k.width), np.arange(k.height))
    z = view.depth_m.astype(np.float64)
    pts = np.stack([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z], axis=-1)
    keep = (z > 0) & (plane.signed_distance(pts.reshape(-1, 3)).reshape(z.shape) > height_m)
    return replace(view, depth_m=np.where(keep, view.depth_m, 0.0).astype(np.float32), derived=True)


def trim_lower_background(view: RGBDView, seg: SubjectCloud, floor_from_cam: np.ndarray, p: ReconParams) -> SubjectCloud:
    """Drop low points far (horizontally) from the torso axis: background seen between or behind the legs."""
    v_idx, u_idx = np.nonzero(seg.mask)
    pts = transform_points(floor_from_cam, backproject(view, seg.mask))  # same pixel order as np.nonzero
    z = pts[:, 2]
    torso = pts[(z > 0.55 * np.percentile(z, 99)) & (z < 0.75 * np.percentile(z, 99))]
    if len(torso) < 50:
        return seg
    axis = np.median(torso[:, :2], axis=0)
    drop = (z < p.lower_roi_height_m) & (np.linalg.norm(pts[:, :2] - axis, axis=1) > p.lower_roi_radius_m)
    mask = np.zeros_like(seg.mask)
    mask[v_idx[~drop], u_idx[~drop]] = True
    return SubjectCloud(seg.view_index, mask, pts[~drop],
                        {**seg.stats, "lower_background_removed": int(drop.sum()), "subject_points": int((~drop).sum())})


def segment_all(views: list[RGBDView], base: np.ndarray, p: ReconParams) -> tuple[list, list[str]]:
    """Segment every view (None where it fails). `base[k]`: floor_from_camera of view k."""
    common = SegmentParams(min_confidence=None, roi_radius_m=p.roi_radius_m, depth_max_m=p.depth.depth_max_m)
    out, errors = [], []
    for k, v in enumerate(views):
        params = SegmentParams(**{**asdict(common), "cluster_select": "widest" if k == 0 else "nearest_origin"})
        try:
            out.append(trim_lower_background(v, segment_subject(v, base[k], params), base[k], p))
        except SegmentationError as e:
            out.append(None)
            errors.append(str(e))
    return out, errors


def _node_frames(exp: Export, pauses: list[tuple[int, int, int]], interval_s: float) -> list[int]:
    """Frames every interval_s from the first to the last pause key frame; pause key frames replace near ones."""
    keys = [k for _, _, k in pauses]
    fps = 1.0 / float(np.median(np.diff(exp.t)))
    step = max(1, round(interval_s * fps))
    grid = range(keys[0], keys[-1] + 1, step)
    return sorted(set(keys) | {g for g in grid if min(abs(g - k) for k in keys) > step // 2})


class _FeatureStore:
    """Features per frame, computed once. Pause key frames use the pause median depth."""

    def __init__(self, exp: Export, p: ReconParams, floor_cam0: np.ndarray, rel: Callable[[int], np.ndarray],
                 pause_data: dict[int, tuple[RGBDView, np.ndarray]], floor_plane: FloorPlane):
        self.floor_plane = floor_plane
        self.exp, self.p, self.floor_cam0, self.rel, self.pause_data = exp, p, floor_cam0, rel, pause_data
        self.cache: dict[int, Features] = {}

    def get(self, frames: list[int]) -> list[Features]:
        todo = [f for f in frames if f not in self.cache]
        seg = SegmentParams(min_confidence=None, roi_radius_m=self.p.roi_radius_m, cluster_select="nearest_origin",
                            depth_max_m=self.p.depth.depth_max_m)
        for i, gray in iter_frames(self.exp.root / "rgb.mp4", todo, self.exp.rgb_size):
            floor_i = self.floor_cam0 @ self.rel(i)
            if i in self.pause_data:
                view, mask = self.pause_data[i]
            else:
                view = remove_floor(_single_view(self.exp, i, self.p.depth)[0], self.floor_plane, self.p.floor_removal_m)
                try:
                    mask = trim_lower_background(view, segment_subject(view, floor_i, seg), floor_i, self.p).mask
                except SegmentationError:
                    mask = np.zeros(view.depth_m.shape, bool)
            self.cache[i] = extract_features(gray, mask, view.depth_m, self.exp.intrinsics_rgb(i), floor_i,
                                             self.p.tracking)
        return [self.cache[f] for f in frames]


def _pose_differences(a: PoseSolution, fa: list[int], b: PoseSolution, fb: list[int], pause_frames: list[int],
                      centres: dict[int, np.ndarray]) -> dict:
    deg, mm, missing = [], [], 0
    for f in pause_frames:
        i, j = fa.index(f), fb.index(f)
        if i not in a.floor_from_body or j not in b.floor_from_body:
            missing += 1
            continue
        pa, pb = a.floor_from_body[i], b.floor_from_body[j]
        deg.append(abs(np.degrees(float(wrap(a.se2[i][0] - b.se2[j][0])))))
        body = invert(pa) @ np.r_[centres[f], 1.0]  # where does the other solution put the same body point?
        mm.append(float(np.linalg.norm(((pa - pb) @ body)[:2])) * 1000)
    return {"compared": len(deg), "missing": missing,
            "turn_deg": {"median": float(np.median(deg)), "max": float(np.max(deg))} if deg else None,
            "position_mm": {"median": float(np.median(mm)), "max": float(np.max(mm))} if mm else None}


def _coverage_gap(angles_rad: list[float]) -> float:
    a = np.sort(np.degrees(np.asarray(angles_rad)) % 360)
    return float(np.max(np.diff(np.r_[a, a[0] + 360])))


# ---- exports ----------------------------------------------------------------------------------

def write_glb(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    """Standard binary GLB (separate BIN chunk). Open3D's writer embeds base64 buffers that its own reader rejects."""
    trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False).export(str(path))


def export_mesh(mesh: o3d.geometry.TriangleMesh, out: Path, stem: str = "mesh") -> dict:
    """Write PLY, OBJ and GLB (same metric vertices, no smoothing) and read each back to verify it."""
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    bounds = np.stack([m.get_min_bound(), m.get_max_bound()])
    result = {}
    for ext in ("ply", "obj", "glb"):
        path = out / f"{stem}.{ext}"
        if ext == "glb":
            write_glb(m, path)
        elif not o3d.io.write_triangle_mesh(str(path), m, write_triangle_uvs=False):
            raise RuntimeError(f"could not write {path.name}")
        back = trimesh.load(str(path), force="mesh", process=False)
        ok = len(back.faces) == len(m.triangles)
        dev = float(np.abs(np.asarray(back.bounds) - bounds).max()) if ok else None
        result[ext] = {"file": path.name, "bytes": path.stat().st_size, "triangles_read_back": int(len(back.faces)),
                       "bounds_error_mm": None if dev is None else dev * 1000,
                       "verified": bool(ok and dev is not None and dev < 1e-3)}
    return result


def _refine(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Topology-only cleanup: no vertex is moved, nothing is smoothed, no hole is filled."""
    m = o3d.geometry.TriangleMesh(mesh)
    before = (len(m.vertices), len(m.triangles))
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_non_manifold_edges()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m, {"steps": ["remove degenerate/duplicated triangles", "merge duplicated vertices",
                         "remove non-manifold edges"], "vertices_before": before[0], "triangles_before": before[1],
               "vertices_after": len(m.vertices), "triangles_after": len(m.triangles),
               "vertices_moved": False, "holes_filled": False, "smoothing": "none"}


def _colored_cloud(clouds_body: list[np.ndarray], colours: list[np.ndarray]) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(clouds_body)))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(colours))
    return pc


# ---- report -----------------------------------------------------------------------------------

def _write_html(run_dir: Path, report: dict, images: list[str]) -> None:
    esc = lambda s: html.escape(str(s))
    rows = "".join(f"<tr class='{esc(c['status'])}'><td>{esc(c['name'])}</td><td>{esc(c['status'])}</td>"
                   f"<td>{esc(_short(c['value']))}</td><td>{esc(_short(c['threshold']))}</td><td>{esc(c['message'])}</td></tr>"
                   for c in report.get("checks", []))
    figs = "".join(f"<figure><img src='{esc(i)}' alt='{esc(i)}'><figcaption>{esc(i)}</figcaption></figure>" for i in images)
    sections = "".join(f"<h2>{esc(k)}</h2><pre>{esc(json.dumps(report[k], indent=1))}</pre>"
                       for k in ("consistency", "pose", "comparison_depth_only_poses", "mesh", "exports", "inputs")
                       if k in report)
    (run_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>RGB-D reconstruction quality</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:24px}table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;"
        "padding:4px 8px;text-align:left}tr.fail td{color:#b42318}tr.warn td{color:#8a5a00}img{max-width:100%}"
        "pre{background:#f6f7f9;padding:8px;overflow:auto;max-height:360px}</style>"
        f"<h1>RGB-D reconstruction quality report</h1><p><b>Status: {esc(report['status'])}</b>; mesh result status: "
        f"<b>{esc(report.get('mesh_result_status', 'unavailable'))}</b>. {esc(report.get('note', ''))}</p>"
        f"<p>{esc(report.get('status_note', ''))}</p>"
        f"<table><tr><th>check</th><th>status</th><th>value</th><th>limit</th><th>meaning</th></tr>{rows}</table>"
        f"{figs}{sections}", encoding="utf-8")


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def _finish(run_dir: Path, report: dict, images: list[str]) -> None:
    _write_html(run_dir, vis.to_jsonable(report), images)
    report["outputs_sha256"] = {f.name: _sha(f) for f in sorted(run_dir.iterdir()) if f.is_file()}
    (run_dir / "quality_report.json").write_text(json.dumps(vis.to_jsonable(report), indent=2) + "\n", encoding="utf-8")


# ---- entry point ------------------------------------------------------------------------------

def reconstruct(recording_dir: str | Path, out_root: str | Path, *, params: ReconParams | None = None,
                run_id: str | None = None, progress: Callable[[str], None] | None = None) -> ReconResult:
    p = params or ReconParams()
    say = progress or (lambda s: None)
    run_id = run_id or "textured_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_root) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
    say("reading export")
    exp, audit = read_export(recording_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    report: dict[str, Any] = {
        "run_id": run_id, "pipeline_version": PIPELINE_VERSION, "tracking_version": TRACKING_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software": {"digitaltrack": __version__, "open3d": o3d.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0], "platform": platform.platform()},
        "note": "No physical reference was used: accuracy is NOT validated and no body measurement is computed. "
                "All limits are proposed, unvalidated values.",
        "parameters": asdict(p), "inputs": {"recording": exp.root.name, "audit": audit}, "checks": checks,
    }

    def fail(reason: str) -> ReconResult:
        report.update(status="fail", status_note=f"No mesh: {reason}", mesh_result_status="unavailable")
        _finish(run_dir, report, [])
        return ReconResult(run_dir, "fail", report)

    # 1. synchronisation
    if p.check_sync:
        say("checking RGB/depth synchronisation")
        sync = check_sync(exp)
        report["inputs"]["sync"] = sync
        lag = sync.get("best_lag")
        checks.append(_check("rgb_depth_sync_lag", "warn" if lag is None else "pass" if lag == 0 else "fail", lag, 0,
                             sync.get("reason") or "frame lag with the best RGB/depth motion correlation"))
        if lag not in (None, 0):
            return fail(f"RGB and depth appear offset by {lag} frame(s)")
    else:
        checks.append(_check("rgb_depth_sync_lag", "warn", None, 0, "not checked"))
    checks.append(_check("timestamp_gaps", "pass" if not audit["timestamp_gaps_over_1_5x"] else "warn",
                         audit["timestamp_gaps_over_1_5x"], 0, "dropped frames in the odometry clock"))

    # 2. gravity
    up, ginfo = gravity(exp)
    report["gravity"] = ginfo
    dis = (ginfo["imu"] or {}).get("disagreement_deg")
    checks.append(_check("gravity_imu_vs_odometry_deg", "warn" if dis is None or dis > p.max_gravity_disagreement_deg
                         else "pass", dis, p.max_gravity_disagreement_deg, "accelerometer vs odometry 'up'"))

    # 3. pauses and filtered median depth
    say("finding pauses")
    pauses, report["pauses"] = find_pauses(exp, p)
    if len(pauses) < 3:
        return fail(f"only {len(pauses)} pauses found (the protocol needs the person to turn in steps with pauses)")
    say("filtering depth")
    views, filt = [], []
    for first, last, key in pauses:
        v, s = pause_view(exp, first, last, key, p.depth)
        views.append(v)
        filt.append(s)
    total = {k: int(sum(s[k] for s in filt)) for k in ("pixels", "out_of_range", "low_confidence", "flying", "kept")}
    mads = [s["median"]["p90_abs_deviation_mm"] for s in filt if s["median"]["p90_abs_deviation_mm"] is not None]
    report["depth_filter"] = {**total, "low_confidence_fraction": total["low_confidence"] / total["pixels"],
                              "flying_fraction": total["flying"] / total["pixels"],
                              "temporal_mad_mm_p90_median_over_pauses": float(np.median(mads)) if mads else None,
                              "spatial_smoothing": "none"}

    # 4. floor
    try:
        floor_h, depth_plane, finfo = fit_floor(views, up)
    except FloorFitError as e:
        return fail(f"floor not found: {e}")
    report["floor"] = finfo
    # The depth floor can be tilted against gravity by a degree or two, which at 1 m is more than the floor band:
    # remove floor pixels using the plane that was actually fitted to the depth.
    views = [remove_floor(v, depth_plane, p.floor_removal_m) for v in views]
    finfo["floor_pixels_removed_below_m"] = p.floor_removal_m
    plane = FloorPlane(tuple(float(x) for x in up), -floor_h, finfo["inliers"], finfo["inliers"], finfo["inlier_rms_m"])
    # origin under the subject: nearer quartile of the central image patch of the first pause
    d0 = views[0].depth_m
    h, w = d0.shape
    centre = np.zeros_like(d0, bool)
    centre[h // 4:3 * h // 4, 2 * w // 5:3 * w // 5] = True
    if (centre & (d0 > 0)).sum() < 50:
        return fail("no depth at the image centre in the first pause (the subject should stand there)")
    z0 = float(np.percentile(d0[centre & (d0 > 0)], 25))
    k0 = views[0].intrinsics
    hint = np.array([(w / 2 - k0.cx) / k0.fx * z0, (h / 2 - k0.cy) / k0.fy * z0, z0])
    floor0 = floor_from_camera(plane, hint)
    segs, _ = segment_all(views[:1], [floor0], p)
    if segs[0] is None:
        return fail("the subject could not be segmented in the first pause")
    c = np.r_[segs[0].points_floor[:, :2].mean(0), 0.0]
    floor_cam0 = floor_from_camera(plane, transform_points(invert(floor0), c[None])[0])
    report["floor_from_camera"] = floor_cam0.tolist()

    # 5. segmentation (camera assumed fixed for the masks; re-done below if odometry has to be applied)
    say("segmenting")
    segs, errors = segment_all(views, [floor_cam0] * len(views), p)
    usable = [k for k, s in enumerate(segs) if s is not None]
    if len(usable) < 3:
        return fail("fewer than 3 pauses could be segmented: " + "; ".join(errors[:3]))

    # 6. camera motion
    say("checking the camera did not move")
    keys = [pauses[k][2] for k in usable]
    _, cam = camera_motion(exp, keys, {pauses[k][2]: segs[k].mask for k in usable})
    report["camera_motion"] = cam
    checks.append(_check("camera_static", cam["status"], cam["max_measured_px"], cam["static_limit_px"],
                         f"{cam['decision']}: {cam['note']}"))
    if cam["status"] == "fail":
        return fail(cam["note"])
    if cam["decision"] == "odometry_applied":
        ref_inv = invert(exp.world_from_camera(keys[0]))
        rel = lambda i: ref_inv @ exp.world_from_camera(i)  # ref_camera_from_camera_i
        segs, errors = segment_all(views, [floor_cam0 @ rel(pk[2]) for pk in pauses], p)
        usable = [k for k, s in enumerate(segs) if s is not None]
    else:
        rel = lambda i: np.eye(4)

    # partial views: subject top clearly below the others (moving during the "pause")
    tops = np.array([np.percentile(segs[k].points_floor[:, 2], 99) for k in usable])
    keep = tops >= np.median(tops) - p.partial_view_drop_m
    dropped = [{"pause": k, "reason": "partial view (subject top too low)"} for k, ok in zip(usable, keep) if not ok]
    dropped += [{"pause": k, "reason": "segmentation failed"} for k, s in enumerate(segs) if s is None]
    usable = [k for k, ok in zip(usable, keep) if ok]
    report["segmentation"] = {"pauses_used": usable, "dropped": dropped,
                              "points_per_view": [int(len(segs[k].points_floor)) for k in usable],
                              "lower_background_removed_per_view": [segs[k].stats.get("lower_background_removed", 0)
                                                                    for k in usable]}
    if len(usable) < 3:
        return fail("fewer than 3 usable pauses")

    # 7. tracking and pose graph
    pause_frames = [pauses[k][2] for k in usable]
    use_pauses = [pauses[k] for k in usable]
    pause_data = {pauses[k][2]: (views[k], segs[k].mask) for k in usable}
    centres = {pauses[k][2]: np.r_[segs[k].points_floor[:, :2].mean(0), 1.0] for k in usable}
    store = _FeatureStore(exp, p, floor_cam0, rel, pause_data, depth_plane)

    def solve(interval: float) -> tuple[PoseSolution, list[int]]:
        frames = _node_frames(exp, use_pauses, interval)
        nodes = [frames.index(f) for f in pause_frames]
        return track(store.get(frames), p.tracking, pause_nodes=nodes), frames

    say("tracking the subject through the turn")
    sol, node_frames = solve(p.tracking.interval_s)
    placed = [f for f in pause_frames if node_frames.index(f) in sol.floor_from_body]
    report["pose"] = {"interval_s": p.tracking.interval_s, **sol.checks, "pause_views_placed": len(placed),
                      "pause_views_total": len(pause_frames),
                      "edges": [e.summary() for e in sol.edges]}
    checks.append(_check("pause_views_placed", "pass" if len(placed) == len(pause_frames) else
                         "warn" if len(placed) >= 3 else "fail", len(placed), len(pause_frames),
                         "pause views linked to the first view by texture correspondences"))
    if len(placed) < 3:
        return fail("subject texture could not be tracked (plain clothing, motion blur or large steps)")
    loop = sol.checks["loop_error_before_optimisation_deg"]
    checks.append(_check("loop_closure_error_deg", "warn" if loop is None or loop["median"] > p.max_loop_error_deg
                         else "pass", None if loop is None else loop["median"], p.max_loop_error_deg,
                         "drift of the neighbour-only chain against direct loop matches, before optimisation"
                         if loop else "no loop pair found: drift cannot be checked"))

    say("pose sensitivity to the tracking interval")
    sens = {}
    for interval in p.sensitivity_intervals_s:
        other, frames = solve(interval)
        sens[str(interval)] = {**_pose_differences(sol, node_frames, other, frames, pause_frames, centres),
                               "loop_error_before_optimisation_deg": other.checks["loop_error_before_optimisation_deg"]}
    report["pose"]["sensitivity_to_interval"] = sens
    worst_deg = max((s["turn_deg"]["max"] for s in sens.values() if s["turn_deg"]), default=None)
    worst_mm = max((s["position_mm"]["max"] for s in sens.values() if s["position_mm"]), default=None)
    checks.append(_check("pose_sensitivity_turn_deg", "warn" if worst_deg is None or worst_deg > p.max_pose_sensitivity_deg
                         else "pass", worst_deg, p.max_pose_sensitivity_deg,
                         "largest change of a pause view's turn angle when the tracking interval changes"))
    checks.append(_check("pose_sensitivity_position_mm", "warn" if worst_mm is None or worst_mm > p.max_pose_sensitivity_mm
                         else "pass", worst_mm, p.max_pose_sensitivity_mm,
                         "largest change of a pause view's position when the tracking interval changes"))

    # 8. registered clouds and fusion
    say("fusing")
    fused_k, masks, poses_cam, poses_floor, clouds = [], [], [], [], []
    for k in usable:
        f = pauses[k][2]
        node = node_frames.index(f)
        if node not in sol.floor_from_body:
            continue
        pose = sol.floor_from_body[node]
        cam_from_body = invert(floor_cam0 @ rel(f)) @ pose
        fused_k.append(k)
        masks.append(segs[k].mask)
        poses_floor.append(pose)
        poses_cam.append(cam_from_body)
        clouds.append(transform_points(invert(pose), segs[k].points_floor))
    angles = [sol.se2[node_frames.index(pauses[k][2])][0] for k in fused_k]
    gap = _coverage_gap(angles)
    checks.append(_check("max_angular_gap_deg", "pass" if gap <= p.max_coverage_gap_deg else "warn" if gap <= 90 else "fail",
                         gap, p.max_coverage_gap_deg, "largest gap between neighbouring view headings"))
    fuse_in = [views[k] for k in fused_k]
    raw_mesh, per_view, fuse_log = fuse_views(fuse_in, masks, poses_cam, p.tsdf)
    if not len(raw_mesh.triangles):
        return fail("TSDF produced no surface")
    mesh, refine_log = _refine(raw_mesh)
    report["fusion"] = {k: v for k, v in fuse_log.items() if k != "per_frame"}
    report["views"] = [{"pause": k, "frame": int(exp.frames[pauses[k][2]]), "turn_deg": float(np.degrees(a)),
                        "floor_from_body": m.tolist()} for k, a, m in zip(fused_k, angles, poses_floor)]

    colour_images = dict(iter_frames(exp.root / "rgb.mp4", [pauses[k][2] for k in fused_k], exp.rgb_size,
                                     gray=False, scale_to=exp.depth_size))
    colours = []
    for k, v, m, t in zip(fused_k, fuse_in, masks, poses_cam):
        # same pixel order as backproject(v, m)
        colours.append(colour_images[pauses[k][2]][m][:, ::-1].astype(float) / 255.0)
    registered = [transform_points(invert(t), backproject(v, m)) for v, m, t in zip(fuse_in, masks, poses_cam)]
    o3d.io.write_point_cloud(str(run_dir / "pointcloud_registered.ply"), _colored_cloud(registered, colours))

    # 9. exports
    say("exporting")
    exports = export_mesh(mesh, run_dir)
    display, dinfo = clean_display_mesh(mesh)
    write_glb(display, run_dir / "mesh_display.glb")
    report["exports"] = {"metric_mesh": exports, "display_mesh": {"file": "mesh_display.glb", **dinfo},
                         "units": "metres", "axes": "Z up, floor at z = 0 (not glTF Y-up)"}
    checks.append(_check("exports_verified", "pass" if all(e["verified"] for e in exports.values()) else "fail",
                         sum(e["verified"] for e in exports.values()), 3, "PLY, OBJ and GLB read back identically"))

    # 10. quality
    say("quality checks")
    consistency = ev.cross_view_consistency(registered)
    residual = ev.distance_stats(ev.distances_to_mesh(mesh, np.concatenate(registered)))
    icp = view_consistency(registered, [pauses[k][2] for k in fused_k], mesh)
    report["consistency"] = {"cross_view": consistency, "view_points_to_mesh": residual,
                             "icp_diagnostic_only": {"applied": False, "algorithm_version": icp["algorithm_version"],
                                                     "summary": icp["summary"]}}
    med = consistency["median_mm"]
    checks.append(_check("cross_view_median_mm", "pass" if med is not None and med <= p.max_cross_view_median_mm else "warn",
                         med, p.max_cross_view_median_mm, "distance between overlapping views after registration"))
    topo = ev.mesh_topology(mesh)
    lo, hi = np.asarray(mesh.get_min_bound()), np.asarray(mesh.get_max_bound())
    report["mesh"] = {**topo, "refinement": refine_log, "coverage": ev.observation_coverage(mesh, registered),
                      "bounds_m": [lo.tolist(), hi.tolist()],
                      "bounds_note": "bounding box for visual sanity only; not a height or girth measurement"}
    checks.append(_check("mesh_watertight", "pass" if topo["watertight"] else "warn", topo["boundary_edges"], 0,
                         "open boundary edges (unobserved areas are left open, not filled): volume is unavailable"))

    # 11. the existing depth-shape pose method on the SAME clouds, scored with the SAME metrics
    if p.compare_depth_only:
        say("comparing with depth-shape poses (existing method)")
        floor_clouds = [segs[k].points_floor for k in fused_k]
        old = estimate_poses(floor_clouds, PoseParams(closure_gap_deg=90.0))
        entry: dict[str, Any] = {"method": old.algorithm_version, "status": old.status,
                                 "loop_closure": old.loop_closure,
                                 "net_turn_check": next((c["value"] for c in old.checks if c["name"] == "turn_coverage_deg"), None)}
        if old.floor_from_body:
            old_cam = [invert(floor_cam0 @ rel(pauses[k][2])) @ m for k, m in zip(fused_k, old.floor_from_body)]
            old_mesh, _, _ = fuse_views(fuse_in, masks, old_cam, p.tsdf)
            old_reg = [transform_points(invert(t), backproject(v, m)) for v, m, t in zip(fuse_in, masks, old_cam)]
            entry.update(cross_view=ev.cross_view_consistency(old_reg),
                         view_points_to_mesh=ev.distance_stats(ev.distances_to_mesh(old_mesh, np.concatenate(old_reg))),
                         mesh=ev.mesh_topology(old_mesh),
                         bounds_m=[old_mesh.get_min_bound().tolist(), old_mesh.get_max_bound().tolist()])
            o3d.io.write_triangle_mesh(str(run_dir / "mesh_depth_only_poses.ply"), old_mesh)
        report["comparison_depth_only_poses"] = entry

    images = vis.write_point_views(registered, run_dir)
    images["slices"] = vis.write_slices(registered, mesh, run_dir)
    images.update(vis.write_mesh_renders(mesh, registered, run_dir))
    image_files = [v for v in images.values() if isinstance(v, str)]

    status = "fail" if any(c["status"] == "fail" for c in checks) else \
        "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    report.update(status=status, mesh_result_status="unreliable" if status == "fail" else "estimated",
                  status_note={"pass": "All proposed internal-consistency checks passed. Accuracy is still unvalidated.",
                               "warn": "Mesh produced; some checks raised warnings. Accuracy is unvalidated.",
                               "fail": "Mesh produced but UNRELIABLE: a check failed."}[status])
    _finish(run_dir, report, image_files)
    return ReconResult(run_dir, status, report, mesh, poses_floor, [pauses[k][2] for k in fused_k], floor_cam0)
