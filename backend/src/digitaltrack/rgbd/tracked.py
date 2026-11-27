"""Stray Scanner RGB-D recording of a person turning in front of a FIXED iPad -> metric body mesh + quality report.

Pipeline (all steps logged in run_log.json; raw files are only read):
  1. load depth / confidence / intrinsics (rgbd.stray, every `frame_step`-th frame) and the empty-scene recording
  2. vertical = IMU gravity (imu.csv), floor height from the empty scene's floor plane (rgbd.segment)
  3. background plate (empty scene) -> subject masks by image connected components
  4. RGB keypoints inside the masks, lifted to 3-D by high-confidence depth -> pairwise 4-DOF registration ->
     loop-closing pose graph (rgbd.track)
  5. view selection: turn speed, pose corroboration
  6. fusion masks: confidence, silhouette erosion, depth discontinuities, grazing angle, floor band (rgbd.depthfilter)
  7. TSDF pass 1 -> free-space vote culling -> drop views / pixels inconsistent with it -> TSDF pass 2 -> culling
     (rgbd.consistency). Nothing is filled, smoothed or moved; only unsupported geometry is removed.
  8. exports: fused_mesh.ply (measurement mesh, metres, body frame: floor z = 0, +z up), fused_mesh.obj,
     fused_mesh.glb (same geometry, vertex colours from the RGB video; display), fused_points.ply
  9. quality report (quality_report.json, report.html, PNG previews), including an internal repeatability check:
     the selected views are split into two interleaved halves, each fused on its own, and the two meshes and their
     circumference profiles are compared.

Nothing here is a validated measurement: the circumference profile is labelled `unreliable` until the pipeline is
validated against tape / known objects on this rig (CLAUDE.md, Validation).
"""

from __future__ import annotations

import hashlib
import html
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import open3d as o3d

from .. import __version__
from ..geometry.transforms import invert, transform_points
from ..reconstruction.tsdf import TSDFParams
from . import visualize as vis
from .consistency import ConsistencyParams, cull_mesh, free_space_votes, pixel_residuals
from .depthfilter import DepthFilterParams, fusion_mask
from .fuse import fuse_views
from .rgbmask import RGBMaskParams, build_background, outline_overlay, refine_mask
from .segment import (SegmentationError, SegmentParams, background_plate, estimate_floor, floor_with_vertical,
                      make_floor_frame, segment_subject_plate)
from .stray import imu_gravity_up_camera, iter_rgb_frames, load_stray
from .track import TrackParams, extract_features, estimate_poses_tracked
from .views import RGBDView, backproject

PIPELINE_VERSION = "rgbd-tracked-0.1"


def _default_tsdf() -> TSDFParams:
    # Truncation 3 cm (not 2): where two view groups place a surface ~2 cm apart (side views of the first real
    # recording), 2 cm left a fold and an open seam; 3 cm averages them (holes 229 -> 119) while the outline where
    # views agree did not move (cross-sections identical at 2/3/4 cm). Arms (~8 cm) stay above 2x truncation.
    return TSDFParams(voxel_length_m=0.005, sdf_trunc_m=0.03, depth_trunc_m=4.0, depth_min_m=0.2, min_confidence=2,
                      weight_threshold=3.0, min_cluster_fraction=0.05)


@dataclass(frozen=True)
class TrackedParams:
    frame_step: int = 6  # 60 fps recording -> 10 views per second
    background_views: int = 30
    segment: SegmentParams = field(default_factory=lambda: SegmentParams(background_margin_m=0.05))
    track: TrackParams = field(default_factory=TrackParams)
    depth_filter: DepthFilterParams = field(default_factory=DepthFilterParams)
    consistency: ConsistencyParams = field(default_factory=ConsistencyParams)
    tsdf: TSDFParams = field(default_factory=_default_tsdf)
    use_imu_vertical: bool = True
    # sharpen the depth mask with the RGB video vs the empty scene (rgbd.rgbmask): removes floor leak at the feet
    # and the depth halo; only ever removes pixels
    rgb_mask: bool = True
    refine: bool = True  # refined closed mesh + measurement-eligibility gate (reconstruction.refine)
    rgb_mask_params: RGBMaskParams = field(default_factory=RGBMaskParams)
    max_vertical_disagreement_deg: float = 5.0  # IMU vertical vs fitted floor normal; above => warn, use the floor
    fusion_floor_cut_m: float = 0.03  # floor noise was 7 mm RMS; height is measured from the plane, not the mesh
    max_turn_speed_deg_s: float = 15.0  # fast steps: motion blur and ARKit depth lag
    max_view_outlier_fraction: float = 0.3  # share of a view's edges contradicted by the solved graph
    split_check: bool = True
    profile_step_m: float = 0.01
    thumb_size: tuple[int, int] = (320, 240)
    thumb_every: int = 3


@dataclass
class TrackedResult:
    run_dir: Path
    status: str
    mesh: o3d.geometry.TriangleMesh | None
    quality: dict
    log: dict


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _check(name: str, status: str, value, threshold, message: str) -> dict:
    return {"name": name, "status": status, "value": value, "threshold": threshold, "message": message}


# ---- fusion helpers ---------------------------------------------------------------------------------------------

def _fusion_inputs(views, subj_masks, floor_from_cam, p: TrackedParams):
    masks, stats = [], []
    for v, m in zip(views, subj_masks):
        fm, st = fusion_mask(v, m, p.depth_filter)
        vv, uu = np.nonzero(fm)
        h = transform_points(floor_from_cam, backproject(v, fm))[:, 2]
        low = h <= p.fusion_floor_cut_m
        fm[vv[low], uu[low]] = False
        st["after_floor_cut"] = int(fm.sum())
        masks.append(fm)
        stats.append(st)
    return masks, stats


def _two_pass(views, masks, subj, cam_from_body, p: TrackedParams, tell=None):
    """TSDF pass 1 -> cull -> per-view / per-pixel consistency -> TSDF pass 2 -> cull."""
    tell = tell or (lambda s, f: None)
    cp = p.consistency
    tell("fusing (pass 1)", 0.0)
    m1, pv1, flog1 = fuse_views(views, masks, cam_from_body, p.tsdf)
    tell("consistency vote", 0.3)
    sup, vio = free_space_votes(np.asarray(m1.vertices), views, subj, cam_from_body, cp)
    c1, cull1 = cull_mesh(m1, sup, vio, cp, p.tsdf.min_cluster_fraction)
    keep, masks2, view_frac, px_dropped = [], [], [], 0
    for k, (m, pts) in enumerate(zip(masks, pv1)):
        r = pixel_residuals(c1, pts) if len(pts) else np.zeros(0)
        frac = float(np.mean(r < cp.view_inlier_m)) if len(r) else 0.0
        view_frac.append(frac)
        if frac < cp.min_view_inlier_fraction:
            continue
        vv, uu = np.nonzero(m)
        bad = r > cp.pixel_max_residual_m
        m2 = m.copy()
        m2[vv[bad], uu[bad]] = False
        px_dropped += int(bad.sum())
        keep.append(k)
        masks2.append(m2)
    if len(keep) < 3:
        raise RuntimeError("fewer than 3 views are consistent with the first fusion pass")
    v2 = [views[k] for k in keep]
    t2 = [cam_from_body[k] for k in keep]
    s2 = [subj[k] for k in keep]
    tell("fusing (pass 2)", 0.6)
    m2, pv2, flog2 = fuse_views(v2, masks2, t2, p.tsdf)
    sup2, vio2 = free_space_votes(np.asarray(m2.vertices), v2, s2, t2, cp)
    final, cull2 = cull_mesh(m2, sup2, vio2, cp, p.tsdf.min_cluster_fraction)
    log = {"pass1": {"triangles": len(m1.triangles), "cull": cull1, "views": len(views)},
           "view_consistency": {"inlier_m": cp.view_inlier_m, "min_fraction": cp.min_view_inlier_fraction,
                                "fraction_p05_p50": [float(np.percentile(view_frac, 5)), float(np.median(view_frac))],
                                "views_dropped": len(views) - len(keep), "pixels_dropped": px_dropped},
           "pass2": {"triangles": len(m2.triangles), "cull": cull2, "views": len(keep),
                     "tsdf_backend": flog2["backend"], "open3d": flog2["open3d"]}}
    return final, keep, pv2, view_frac, log


def _colourise(mesh, thumbs: dict[int, np.ndarray], views_by_pos: dict[int, RGBDView], cam_from_body: dict[int, np.ndarray]):
    """Vertex colour from the RGB frame that sees the vertex most frontally (depth-checked visibility). DISPLAY ONLY."""
    x = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    n = np.asarray(mesh.vertex_normals)
    best = np.full(len(x), -1.0)
    col = np.full((len(x), 3), 0.6)
    for pos, img in thumbs.items():
        v, t = views_by_pos[pos], cam_from_body[pos]
        k = v.intrinsics
        xc = x @ t[:3, :3].T + t[:3, 3]
        z = xc[:, 2]
        ok = z > 0.1
        u = k.fx * xc[:, 0] / np.where(ok, z, 1) + k.cx
        r = k.fy * xc[:, 1] / np.where(ok, z, 1) + k.cy
        ui, ri = np.round(u).astype(int), np.round(r).astype(int)
        ok &= (ui >= 0) & (ui < k.width) & (ri >= 0) & (ri < k.height)
        d = np.where(ok, v.depth_m[np.clip(ri, 0, k.height - 1), np.clip(ui, 0, k.width - 1)], 0)
        ok &= np.abs(d - z) < 0.02
        nc = n @ t[:3, :3].T
        score = -np.sum(nc * xc, 1) / np.maximum(np.linalg.norm(xc, axis=1), 1e-9)
        upd = ok & (score > best) & (score > 0.2)
        if not upd.any():
            continue
        h, w = img.shape[:2]
        su, sv = w / k.width, h / k.height
        px = img[np.clip(((r[upd] + 0.5) * sv).astype(int), 0, h - 1), np.clip(((u[upd] + 0.5) * su).astype(int), 0, w - 1)]
        col[upd] = px[:, ::-1] / 255.0
        best[upd] = score[upd]
    mesh.vertex_colors = o3d.utility.Vector3dVector(col)
    return float(np.mean(best > 0))


# ---- mesh quality and profile -----------------------------------------------------------------------------------

def mesh_quality(mesh: o3d.geometry.TriangleMesh) -> dict:
    tri = np.asarray(mesh.triangles)
    if len(tri) == 0:
        return {"triangles": 0}
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    ue, inv = np.unique(e, axis=0, return_inverse=True)
    boundary = ue[cnt == 1]
    x = np.asarray(mesh.vertices)
    blen = float(np.linalg.norm(x[boundary[:, 0]] - x[boundary[:, 1]], axis=1).sum()) if len(boundary) else 0.0
    # boundary loops (holes) via connected components of boundary edges
    n_loops, big = 0, []
    if len(boundary):
        import networkx as nx
        g = nx.Graph()
        g.add_edges_from(map(tuple, boundary))
        for comp in nx.connected_components(g):
            n_loops += 1
            sub = g.subgraph(comp)
            big.append(float(sum(np.linalg.norm(x[a] - x[b]) for a, b in sub.edges)))
    labels, counts, _ = mesh.cluster_connected_triangles()
    watertight = bool(mesh.is_watertight())
    out = {"vertices": len(x), "triangles": int(len(tri)), "components": int(len(counts)),
           "surface_area_m2": float(mesh.get_surface_area()), "watertight": watertight,
           "boundary_edges": int(len(boundary)), "boundary_length_m": blen, "holes": n_loops,
           "largest_holes_perimeter_m": sorted(big, reverse=True)[:5],
           "non_manifold_edges": int(len(mesh.get_non_manifold_edges())),
           "bounds_m": [np.asarray(mesh.get_min_bound()).tolist(), np.asarray(mesh.get_max_bound()).tolist()]}
    out["volume_m3"] = float(mesh.get_volume()) if watertight else None
    out["volume_status"] = "estimated (closed mesh, not validated)" if watertight else \
        "unavailable: mesh is not closed and no hole filling is applied"
    return out


def circumference_profile(mesh: o3d.geometry.TriangleMesh, step: float) -> list[dict]:
    from ..measurement.basic import circumference_profile as prof, o3d_to_trimesh

    tm = o3d_to_trimesh(mesh)
    top = float(np.asarray(mesh.vertices)[:, 2].max())
    zs = np.arange(0.05, top - 0.02, step)
    return [{"z_m": round(float(s.z), 4), "perimeter_m": s.perimeter_m, "closed_contours": s.n_closed_contours,
             "open_polylines": s.n_open_polylines} for s in prof(tm, zs)]


def _hausdorff(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh, n: int = 100000) -> dict:
    pa = np.asarray(a.sample_points_uniformly(n).points)
    pb = np.asarray(b.sample_points_uniformly(n).points)
    dab, dba = pixel_residuals(b, pa), pixel_residuals(a, pb)
    d = np.concatenate([dab, dba])
    return {"p50_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)), "p95_m": float(np.percentile(d, 95)),
            "rms_m": float(np.sqrt(np.mean(d ** 2))), "max_m": float(d.max())}


# ---- images -----------------------------------------------------------------------------------------------------

def _plot(series: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]], path: Path, title: str,
          xlabel: str, ylabel: str, size=(900, 300), hlines=()) -> str:
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    xs = np.concatenate([s[0] for s in series if len(s[0])])
    ys = np.concatenate([s[1][np.isfinite(s[1])] for s in series if len(s[1])])
    if not len(xs) or not len(ys):
        return ""
    x0, x1 = float(xs.min()), float(xs.max()) or 1.0
    y0, y1 = float(ys.min()), float(ys.max())
    for y in hlines:
        y0, y1 = min(y0, y), max(y1, y)
    pad = 0.05 * (y1 - y0 or 1)
    y0, y1 = y0 - pad, y1 + pad
    l, r, t, b = 60, 15, 28, 35
    tx = lambda x: (l + (np.asarray(x) - x0) / max(x1 - x0, 1e-9) * (w - l - r)).astype(int)  # noqa: E731
    ty = lambda y: (h - b - (np.asarray(y) - y0) / max(y1 - y0, 1e-9) * (h - t - b)).astype(int)  # noqa: E731
    cv2.rectangle(img, (l, t), (w - r, h - b), (200, 200, 200), 1)
    for y in hlines:
        cv2.line(img, (l, int(ty(y))), (w - r, int(ty(y))), (180, 180, 180), 1)
    for k in range(5):
        yv = y0 + k * (y1 - y0) / 4
        cv2.putText(img, f"{yv:.3g}", (4, int(ty(yv)) + 4), 0, 0.38, (80, 80, 80), 1)
        xv = x0 + k * (x1 - x0) / 4
        cv2.putText(img, f"{xv:.3g}", (int(tx(xv)) - 10, h - b + 15), 0, 0.38, (80, 80, 80), 1)
    for xs_, ys_, colr, name in series:
        okm = np.isfinite(ys_)
        pts = np.stack([tx(xs_[okm]), ty(ys_[okm])], 1).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, colr, 1, cv2.LINE_AA)
    cv2.putText(img, title, (l, 18), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, xlabel, (w // 2 - 40, h - 4), 0, 0.4, (60, 60, 60), 1)
    cv2.putText(img, ylabel, (4, 18), 0, 0.4, (60, 60, 60), 1)
    for k, (_, _, colr, name) in enumerate(series):
        cv2.putText(img, name, (w - r - 230, t + 16 + 15 * k), 0, 0.42, colr, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return path.name


def _slice_image(meshes: list[tuple[o3d.geometry.TriangleMesh, tuple[int, int, int]]], heights, path: Path) -> str:
    tiles = []
    allx = np.asarray(meshes[0][0].vertices)
    c = np.median(allx[:, :2], 0)
    for hgt in heights:
        img = np.full((300, 300, 3), 255, np.uint8)
        for mesh, colr in meshes:
            x = np.asarray(mesh.sample_points_uniformly(300000).points)
            s = x[np.abs(x[:, 2] - hgt) < 0.003]
            uv = ((s[:, :2] - c) * 500 + 150).astype(int)
            ok = (uv >= 0).all(1) & (uv < 300).all(1)
            img[uv[ok, 1], uv[ok, 0]] = colr
        cv2.line(img, (10, 285), (60, 285), (0, 0, 0), 2)
        cv2.putText(img, "10 cm", (14, 280), 0, 0.38, (0, 0, 0), 1)
        cv2.putText(img, f"z = {hgt:.2f} m", (6, 16), 0, 0.45, (0, 0, 0), 1)
        tiles.append(img)
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1))
    return path.name


def _alignment_image(rgb: np.ndarray, mask: np.ndarray, path: Path) -> str:
    h, w = rgb.shape[:2]
    up = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    edge = cv2.dilate(cv2.Canny(up, 50, 100), np.ones((2, 2), np.uint8))
    o = rgb.copy()
    o[edge > 0] = (0, 0, 255)
    cv2.imwrite(str(path), cv2.rotate(o, cv2.ROTATE_90_CLOCKWISE) if w > h else o)
    return path.name


# ---- main -------------------------------------------------------------------------------------------------------

def run_tracked(recording: str | Path, background: str | Path, out_root: str | Path, *,
                params: TrackedParams | None = None, run_id: str | None = None, source: dict | None = None,
                progress: Callable[[str, float], None] | None = None) -> TrackedResult:
    p = params or TrackedParams()
    tell = progress or (lambda s, f: None)
    t_start = time.time()
    run_id = run_id or f"tracked_{_now()}"
    run_dir = Path(out_root) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    checks: list[dict] = []
    log: dict[str, Any] = {
        "run_id": run_id, "pipeline_version": PIPELINE_VERSION, "created_utc": _now(),
        "software": {"digitaltrack": __version__, "open3d": o3d.__version__, "opencv": cv2.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0], "platform": platform.platform()},
        "source": source or {}, "parameters": vis.to_jsonable(asdict(p)),
        "note": "Thresholds are proposed values tuned on ONE real recording. Mesh dimensions are not validated.",
    }

    # 1. load
    tell("loading depth", 0.0)
    fps_target = 60.0 / p.frame_step
    rec = load_stray(recording, target_fps=fps_target)
    bg = load_stray(background, target_fps=3.0)
    bg_views = bg.views[:p.background_views]
    views = rec.views
    log["recording"] = rec.info
    log["background"] = {**bg.info, "views_used": len(bg_views)}
    still = rec.info.get("ipad_moved_m", 0.0)
    checks.append(_check("camera_static", "pass" if still <= 0.02 else "fail", still, 0.02,
                         "iPad translation over the recording from ARKit odometry (fixed mount required)"))

    # 2. vertical + floor
    up_hint = None
    try:
        up_imu, imu_info = imu_gravity_up_camera(recording)
        log["imu"] = imu_info
        up_hint = up_imu
    except Exception as e:  # noqa: BLE001 - IMU is optional: fall back to the floor normal
        warnings.append(f"IMU gravity unavailable ({e}); vertical = fitted floor normal")
    seg = p.segment
    try:
        floor = estimate_floor(bg_views, seg, source="background_views",
                               **({"up_hint": tuple(up_hint), "gravity": True} if up_hint is not None else {}))
    except SegmentationError as e:
        raise RuntimeError(f"floor: {e}") from e
    log["floor_fit"] = {"normal_camera": list(floor.plane.normal), "inlier_rms_m": floor.plane.inlier_rms_m,
                        "inlier_fraction": floor.plane.inlier_fraction}
    if up_hint is not None and p.use_imu_vertical:
        ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(floor.plane.normal, up_hint)), -1, 1))))
        if ang <= p.max_vertical_disagreement_deg:
            floor, vlog = floor_with_vertical(floor, bg_views, up_hint, seg)
            log["vertical"] = vlog
        else:
            warnings.append(f"IMU vertical and floor normal differ by {ang:.1f} deg; using the floor normal")
            log["vertical"] = {"vertical_source": "fitted floor normal", "imu_vs_floor_deg": ang}
        checks.append(_check("vertical_agreement_deg", "pass" if ang <= 2 else "warn", ang, 2.0,
                             "IMU gravity vs fitted floor normal (the RANSAC fit is seeded with the IMU direction, so "
                             "this confirms consistency, not independence; the person's rotation axis is the "
                             "independent check, see docs/architecture.md)"))
    plate = background_plate(bg_views, seg)

    # background offset between the two recordings (camera bumped? depth bias?)
    offs = []
    for v in views[::max(1, len(views) // 20)]:
        okp = (plate > 0) & np.isfinite(v.depth_m) & (v.depth_m > 0)
        dd = v.depth_m[okp] - plate[okp]
        dd = dd[np.abs(dd) < 0.05]
        if len(dd):
            offs.append(float(np.median(dd)))
    bg_off = float(np.median(offs)) if offs else None
    log["background_offset_m"] = bg_off
    if bg_off is not None:
        checks.append(_check("background_depth_offset_m", "pass" if abs(bg_off) <= 0.01 else "warn", bg_off, 0.01,
                             "median depth difference of static background between the subject and empty-scene "
                             "recordings (camera moved between clips, or a depth bias)"))

    # 3. segmentation
    tell("segmenting", 0.0)
    prov = make_floor_frame(floor, np.array([0.0, 0.0, 2.0]))
    s0 = segment_subject_plate(views[0], prov, plate, seg)
    c0 = np.array([*s0.points_floor[:, :2].mean(0), 0.0])
    floor_from_cam = make_floor_frame(floor, transform_points(invert(prov), c0[None])[0])
    log["floor_from_camera"] = floor_from_cam.tolist()
    subj, ok_pos, seg_fail = {}, [], []
    for k, v in enumerate(views):
        try:
            subj[k] = segment_subject_plate(v, floor_from_cam, plate, seg).mask
            ok_pos.append(k)
        except SegmentationError as e:
            seg_fail.append({"view": k, "error": str(e)})
    log["segmentation"] = {"views": len(views), "segmented": len(ok_pos), "failed": seg_fail[:50],
                           "pixels_p05_p50_p95": np.percentile([subj[k].sum() for k in ok_pos], [5, 50, 95]).tolist()}

    # 4. features (sequential RGB decode) + poses
    tell("RGB features", 0.0)
    frame_of = {k: int(views[k].source["frame"]) for k in ok_pos}
    pos_of = {f: k for k, f in frame_of.items()}
    sift = cv2.SIFT_create(nfeatures=p.track.sift_features, contrastThreshold=p.track.sift_contrast)
    feats, thumbs, align_img = {}, {}, None
    kk = views[0].intrinsics
    bgrgb, mask_stats, overlays = None, [], {}
    if p.rgb_mask:
        try:
            bg_frames = [f for _, f in iter_rgb_frames(background, [int(v.source["frame"]) for v in bg_views])]
            if not bg_frames:
                raise ValueError("no decodable frames in the empty-scene rgb.mp4")
            plate_depth = np.median(np.stack([v.depth_m for v in bg_views]), 0).astype(np.float32)
            bgrgb = build_background(bg_frames, plate_depth, bg_views[0], floor_from_cam, p.rgb_mask_params)
            log["rgb_mask"] = {"background": bgrgb.info}
        except Exception as e:  # noqa: BLE001 - optional refinement: never fail the run for it
            warnings.append(f"RGB mask refinement skipped: {e}")
    overlay_at = set(ok_pos[:: max(1, len(ok_pos) // 3)][:3])
    for i, (fr, img) in enumerate(iter_rgb_frames(recording, frame_of.values())):
        k = pos_of[fr]
        v = views[k]
        if bgrgb is not None:
            before = subj[k]
            subj[k], st = refine_mask(before, v, img, floor_from_cam, bgrgb, p.rgb_mask_params)
            mask_stats.append(st)
            if k in overlay_at:
                overlays[k] = Path(outline_overlay(img, before, subj[k], run_dir / f"mask_rgb_refined_{len(overlays)}.png")).name
        feats[k] = extract_features(img, v.depth_m, v.confidence, subj[k], (kk.fx, kk.fy, kk.cx, kk.cy),
                                    floor_from_cam, p.track, sift, k)
        if k % p.thumb_every == 0:
            thumbs[k] = cv2.resize(img, p.thumb_size, interpolation=cv2.INTER_AREA)
        if align_img is None:
            align_img = _alignment_image(cv2.resize(img, (960, 720)), subj[k], run_dir / "alignment_rgb_depth.png")
        if i % 50 == 0:
            tell("RGB features", i / max(len(frame_of), 1))
    if mask_stats:
        tot = sum(s_["pixels_in"] for s_ in mask_stats)
        log["rgb_mask"].update({
            "views": len(mask_stats), "pixels_in": tot, "floor_veto": sum(s_["floor_veto"] for s_ in mask_stats),
            "wall_veto": sum(s_["wall_veto"] for s_ in mask_stats),
            "removed_fraction": 1 - sum(s_["pixels_out"] for s_ in mask_stats) / max(tot, 1)})
    posed_pos = [k for k in ok_pos if k in feats]
    if len(posed_pos) < len(ok_pos):
        warnings.append(f"{len(ok_pos) - len(posed_pos)} views have no decodable RGB frame and are left out")
    fl = [feats[k] for k in posed_pos]
    tr = estimate_poses_tracked(fl, p.track, progress=tell)
    log["pose"] = {**tr.log, "status": tr.status, "checks": tr.checks,
                   "edges": [e.summary() for e in tr.edges if e.kind != "loop" or e.ok]}
    checks += [{**c, "name": "pose_" + c["name"]} for c in tr.checks]
    # body_from_floor per view position; camera_from_body = camera_from_floor @ floor_from_body
    cam_from_floor = invert(floor_from_cam)
    body_from_floor = {posed_pos[i]: m for i, m in tr.body_from_floor.items()}
    yaw = {posed_pos[i]: y for i, y in tr.yaw_deg.items()}
    cam_from_body = {k: cam_from_floor @ invert(m) for k, m in body_from_floor.items()}

    # 5. view selection
    inc: dict[int, list[int]] = {}
    for e in tr.edges:
        qc_ok = e.ok or any(r.startswith("outlier") for r in e.reasons)
        if qc_ok and e.kind != "loop":
            for n in (e.i, e.j):
                a = inc.setdefault(posed_pos[n], [0, 0])
                a[0] += 1
                a[1] += int(not e.ok)
    ks = sorted(yaw)
    yy = np.array([yaw[k] for k in ks])
    ts = np.array([views[k].timestamp_s for k in ks])
    speed = np.abs(np.gradient(yy, ts)) if len(ks) > 2 else np.zeros(len(ks))
    outfrac = np.array([inc.get(k, [1, 1])[1] / max(inc.get(k, [1, 1])[0], 1) for k in ks])
    sel = [k for k, s, o in zip(ks, speed, outfrac) if s <= p.max_turn_speed_deg_s and o <= p.max_view_outlier_fraction]
    log["view_selection"] = {"posed": len(ks), "selected": len(sel),
                             "too_fast": int((speed > p.max_turn_speed_deg_s).sum()),
                             "low_pose_corroboration": int((outfrac > p.max_view_outlier_fraction).sum())}
    if len(sel) < 10:
        raise RuntimeError(f"only {len(sel)} views passed view selection")

    # 6./7. fusion
    tell("fusing", 0.0)
    fviews = [views[k] for k in sel]
    fmasks, fstats = _fusion_inputs(fviews, [subj[k] for k in sel], floor_from_cam, p)
    log["depth_filter"] = {k: int(np.median([s[k] for s in fstats])) for k in fstats[0]}
    fposes = [cam_from_body[k] for k in sel]
    mesh, keep, pv2, view_frac, flog = _two_pass(fviews, fmasks, [subj[k] for k in sel], fposes, p, tell)
    from ..measurement.body import trim_floor_rim

    mesh, log["floor_rim_trim"] = trim_floor_rim(mesh)  # floor-level fusion residue outside the feet; logged
    from ..reconstruction.mesh_qc import repair_mesh

    tell("mesh repair", 0.0)
    mesh, log["mesh_repair"] = repair_mesh(mesh)  # topology only (degenerate, non-manifold, self-intersecting); no vertex moved
    log["fusion"] = flog
    used = [sel[i] for i in keep]
    vf = flog["view_consistency"]["fraction_p05_p50"][1]
    checks.append(_check("view_consistency_median", "pass" if vf >= 0.8 else "warn" if vf >= 0.65 else "fail", vf, 0.8,
                         "median share of a view's depth pixels within 15 mm of the first-pass fused surface"))

    # residual of the fused points vs the final surface (how well the mesh explains the measured depth)
    res = np.concatenate([pixel_residuals(mesh, q) for q in pv2[::5] if len(q)])
    log["depth_residual_m"] = {"p50": float(np.median(res)), "p90": float(np.percentile(res, 90)),
                               "p95": float(np.percentile(res, 95))}
    checks.append(_check("depth_residual_p50_m", "pass" if np.median(res) <= 0.006 else "warn", float(np.median(res)), 0.006,
                         "median distance of the integrated depth points to the final mesh"))

    # 8. exports
    tell("exporting", 0.0)
    disp = o3d.geometry.TriangleMesh(mesh)
    coloured = _colourise(disp, {k: v for k, v in thumbs.items() if k in cam_from_body},
                          {k: views[k] for k in thumbs}, cam_from_body)
    o3d.io.write_triangle_mesh(str(run_dir / "fused_mesh.ply"), disp)  # same geometry; colours are display only
    o3d.io.write_triangle_mesh(str(run_dir / "fused_mesh.obj"), disp)
    glb = o3d.geometry.TriangleMesh(disp)
    glb.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(run_dir / "fused_mesh.glb"), glb)
    pts = np.concatenate([q for q in pv2[::3] if len(q)])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.004)
    o3d.io.write_point_cloud(str(run_dir / "fused_points.ply"), pcd)
    (run_dir / "poses.json").write_text(json.dumps([
        {"view": k, "frame": frame_of[k], "timestamp_s": views[k].timestamp_s, "yaw_deg": yaw[k],
         "body_from_floor": body_from_floor[k].tolist(), "used_in_fusion": k in used}
        for k in ks], indent=1) + "\n", encoding="utf-8")

    # 9. quality
    tell("quality report", 0.0)
    mq = mesh_quality(mesh)
    top = mq["bounds_m"][1][2] if mq.get("bounds_m") else None
    q: dict[str, Any] = {"mesh": mq, "vertex_colour_coverage": coloured,
                         "highest_point_above_floor_m": top,
                         "highest_point_note": "top of the fused surface above the floor plane (hair included); "
                                               "not a validated stature"}
    checks.append(_check("mesh_components", "pass" if mq["components"] == 1 else "warn", mq["components"], 1,
                         "connected components after cleanup"))
    checks.append(_check("mesh_holes", "pass" if mq["holes"] <= 5 else "warn", mq["holes"], 5,
                         f"open boundary loops (total boundary {mq['boundary_length_m']:.2f} m); not filled"))
    prof = circumference_profile(mesh, p.profile_step_m)
    from ..measurement.basic import ALGORITHM_VERSION as SLICE_ALG, DEFINITION_VERSION as SLICE_DEF

    q["circumference_profile"] = {"status": "unreliable", "units": "m", "definition_version": SLICE_DEF,
                                  "algorithm_version": f"{SLICE_ALG} on {PIPELINE_VERSION}",
                                  "source_method": "rgbd_tracked_tsdf", "result_kind": "measured geometry (not filled)",
                                  "scale_uncertainty": "not quantified: LiDAR metric scale has no known-size "
                                                       "reference in this capture; background depth offset between "
                                                       "clips was logged as a hint of depth bias",
                                  "reason": "pipeline not validated against tape / known objects; LiDAR scale "
                                            "unverified on this rig; the value is the largest closed contour per slice, "
                                            "which includes arms wherever they touched the torso in some views; no "
                                            "anatomical landmarks are detected",
                                  "slices_with_value": sum(1 for s_ in prof if s_["perimeter_m"]),
                                  "slices": prof}
    from ..measurement.body import measure_body, write_landmark_image

    # the subject starts facing the iPad (protocol), and view 0 defines the body frame: subject's left = camera right
    cam_right = floor_from_cam[:3, :3] @ np.array([1.0, 0.0, 0.0])
    body = measure_body(mesh, source_method="rgbd_tracked_tsdf", subject_left_xy=cam_right[:2])
    (run_dir / "body_measurements.json").write_text(json.dumps(vis.to_jsonable(body), indent=2) + "\n",
                                                    encoding="utf-8")
    q["body_measurements"] = {"file": "body_measurements.json", "status": body["status"], "flags": body["flags"],
                              "landmarks": body["landmarks"],
                              "values": {m_["name"]: m_["value"] for m_ in body["measurements"]}}
    # refined (closed, smooth) mesh for display, accepted as measurement-eligible only if it reproduces the values
    if p.refine:
        tell("refined mesh", 0.0)
        from ..reconstruction.refine import refine

        refined, est, rrep = refine(mesh)
        q["refined_mesh"] = {k: rrep.get(k) for k in ("status", "reasons", "eligibility", "checks",
                                                        "estimated_vertex_fraction")}
        (run_dir / "refined_quality.json").write_text(json.dumps(vis.to_jsonable(rrep), indent=2) + "\n",
                                                      encoding="utf-8")
        checks.append(_check("refined_mesh", "pass" if rrep["status"] == "accepted" else "warn", rrep["status"],
                             "accepted", rrep.get("eligibility", "")))
        if refined is not None:
            import trimesh

            o3d.io.write_triangle_mesh(str(run_dir / "refined_mesh.ply"), refined)
            rgba = np.zeros((len(refined.vertices), 4), np.uint8)
            rgba[:, 1] = np.where(est, 255, 0)  # green channel = estimated (gap bridged, not measured)
            rgba[:, 3] = 255
            tm_ = trimesh.Trimesh(np.asarray(refined.vertices), np.asarray(refined.triangles), vertex_colors=rgba,
                                  process=False)
            (run_dir / "refined_mesh.glb").write_bytes(tm_.export(file_type="glb"))
            sub = run_dir / "_refined_render"
            sub.mkdir(exist_ok=True)
            vis.write_mesh_renders(refined, [np.asarray(refined.vertices)], sub)
            (sub / "mesh_render.png").replace(run_dir / "refined_render.png")
            for f_ in sub.iterdir():
                f_.unlink()
            sub.rmdir()
    images: dict[str, str] = {}
    if (run_dir / "refined_render.png").is_file():
        images[f"Refined mesh ({q['refined_mesh']['status']}): closed surface fitted to the measuring mesh"] =             "refined_render.png"
    for n_ in overlays.values():
        images[f"Subject mask: depth only (red) vs RGB-sharpened (green) [{n_}]"] = n_
    images["Landmarks and measurement levels (proposed definitions, not validated)"] = Path(
        write_landmark_image(mesh, body, run_dir / "landmarks.png")).name
    if align_img:
        images["RGB / depth alignment (red = depth subject outline)"] = align_img
    if p.split_check and len(used) >= 20:
        tell("split-half check", 0.0)
        halves = []
        for h_ in (0, 1):
            idx = [i for i, k in enumerate(sel) if k in set(used[h_::2])]
            mh, _, _, _, _ = _two_pass([fviews[i] for i in idx], [fmasks[i] for i in idx],
                                       [subj[sel[i]] for i in idx], [fposes[i] for i in idx], p)
            halves.append(mh)
        hd = _hausdorff(*halves)
        pa = circumference_profile(halves[0], p.profile_step_m)
        pb = circumference_profile(halves[1], p.profile_step_m)
        # every height where BOTH halves give a perimeter for their largest closed contour (the contour may be
        # torso + touching arms: same selection rule in both halves, so this measures repeatability, not anatomy)
        diffs = [(a["z_m"], a["perimeter_m"] - b["perimeter_m"]) for a, b in zip(pa, pb)
                 if a["perimeter_m"] and b["perimeter_m"] and a["closed_contours"] == b["closed_contours"]]
        dd = np.array([d for _, d in diffs]) if diffs else np.zeros(0)
        q["split_half"] = {"surface_distance": hd, "profile_slices_compared": len(dd),
                           "perimeter_diff_abs_p50_m": float(np.median(np.abs(dd))) if len(dd) else None,
                           "perimeter_diff_abs_p95_m": float(np.percentile(np.abs(dd), 95)) if len(dd) else None,
                           "note": "two interleaved halves of the SAME recording; a lower bound on repeatability, "
                                   "not a repeat scan"}
        checks.append(_check("split_half_surface_p95_m", "pass" if hd["p95_m"] <= 0.01 else "warn", hd["p95_m"], 0.01,
                             "95th percentile distance between meshes fused from interleaved halves of the views"))
        images["Cross-sections: full (black), half A (red), half B (blue)"] = _slice_image(
            [(halves[0], (0, 0, 220)), (halves[1], (220, 80, 0)), (mesh, (0, 0, 0))], (0.6, 0.95, 1.1, 1.3, 1.45),
            run_dir / "cross_sections.png")
        z = np.array([s["z_m"] for s in prof])
        per = lambda pr: np.array([s["perimeter_m"] if s["perimeter_m"] else np.nan for s in pr])  # noqa: E731
        images["Perimeter of the largest closed contour per height (NOT anatomical: may include touching arms)"] = _plot(
            [(z, per(prof), (0, 0, 0), "full"), (z, per(pa), (0, 0, 220), "half A"), (z, per(pb), (220, 80, 0), "half B")],
            run_dir / "circumference_profile.png", "Circumference vs height (UNRELIABLE: not validated)",
            "height above floor (m)", "perimeter (m)")
    images["Turn angle per view"] = _plot([(ts - ts[0], yy, (200, 0, 0), "estimated yaw (deg)")],
                                          run_dir / "turn_angle.png", "Body turn angle from the pose graph",
                                          "time (s)", "deg")
    vc = np.array(view_frac)
    images["Per-view consistency"] = _plot([(np.array([views[k].timestamp_s for k in sel]) - ts[0], vc, (0, 120, 0),
                                             "share within 15 mm")], run_dir / "view_consistency.png",
                                           "View consistency with the first-pass surface", "time (s)", "fraction",
                                           hlines=(p.consistency.min_view_inlier_fraction,))
    r = vis.write_mesh_renders(mesh, pv2[::10], run_dir)
    images["Mesh (shaded)"] = r.get("mesh_render", "")
    images["Coverage (views supporting each surface point)"] = r.get("coverage", "")
    q["coverage"] = r.get("coverage_stats")

    status = "fail" if any(c["status"] == "fail" for c in checks) else \
        "warn" if any(c["status"] == "warn" for c in checks) or warnings else "pass"
    q.update({"status": status, "checks": checks, "warnings": warnings,
              "views": {"recorded_frames": rec.info.get("odometry_rows"), "loaded": len(views),
                        "posed": len(ks), "fused": len(used)},
              "pose_summary": {"turn_range_deg": tr.log["turn_range_deg"], "status": tr.status},
              "depth_residual_m": log["depth_residual_m"], "background_offset_m": bg_off,
              "measurement_status_rule": "every value here is diagnostic (unreliable) until validated on this rig"})
    (run_dir / "quality_report.json").write_text(json.dumps(vis.to_jsonable(q), indent=2) + "\n", encoding="utf-8")
    log.update({"status": status, "warnings": warnings, "n_views": len(used), "quality": "quality_report.json",
                "images": images, "runtime_s": round(time.time() - t_start, 1),
                "mesh": {"vertices": mq.get("vertices"), "triangles": mq.get("triangles"), "bounds_m": mq.get("bounds_m"),
                         "note": "metres, body frame (floor z = 0, +z = IMU vertical); not validated"}})
    _write_report(run_dir, log, q, images)
    log["outputs"] = {f.name: _sha(f) for f in sorted(run_dir.iterdir()) if f.is_file()}
    (run_dir / "run_log.json").write_text(json.dumps(vis.to_jsonable(log), indent=2) + "\n", encoding="utf-8")
    return TrackedResult(run_dir, status, mesh, q, log)


def _write_report(run_dir: Path, log: dict, q: dict, images: dict) -> None:
    esc = lambda s: html.escape(str(s))  # noqa: E731
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    rows = "".join(f"<tr class='{esc(c['status'])}'><td>{esc(c['name'])}</td><td>{esc(c['status'])}</td>"
                   f"<td>{esc(fmt(c['value']))}</td><td>{esc(fmt(c['threshold']))}</td><td>{esc(c['message'])}</td></tr>"
                   for c in q["checks"])
    m = q["mesh"]
    sh = q.get("split_half") or {}
    facts = [("views recorded / posed / fused", f"{q['views']['recorded_frames']} / {q['views']['posed']} / {q['views']['fused']}"),
             ("turn range", f"{q['pose_summary']['turn_range_deg']:.0f} deg"),
             ("depth residual p50 / p90", f"{q['depth_residual_m']['p50'] * 1000:.1f} / {q['depth_residual_m']['p90'] * 1000:.1f} mm"),
             ("mesh", f"{m.get('vertices')} vertices, {m.get('triangles')} triangles, {m.get('components')} component(s)"),
             ("holes (open boundary loops)", f"{m.get('holes')} (boundary {m.get('boundary_length_m', 0):.2f} m), not filled"),
             ("volume", m.get("volume_status")),
             ("highest point above floor", f"{(q.get('highest_point_above_floor_m') or 0):.3f} m (hair included; not validated)"),
             ("split-half surface distance p50 / p95",
              f"{sh['surface_distance']['p50_m'] * 1000:.1f} / {sh['surface_distance']['p95_m'] * 1000:.1f} mm" if sh else "n/a"),
             ("split-half perimeter difference |p50| / |p95|",
              f"{(sh.get('perimeter_diff_abs_p50_m') or 0) * 1000:.1f} / {(sh.get('perimeter_diff_abs_p95_m') or 0) * 1000:.1f} mm"
              if sh else "n/a")]
    fact_rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in facts)
    body = {}
    bp = run_dir / "body_measurements.json"
    if bp.is_file():
        body = json.loads(bp.read_text(encoding="utf-8"))
    def metres(v, digits):
        return "" if v is None else f"{v:.{digits}f} m"

    meas_rows = "".join(
        f"<tr class='{'fail' if x['value'] is None else 'warn'}'><td>{esc(x['name'])}</td>"
        f"<td>{esc(metres(x['value'], 3))}</td><td>{esc(metres(x.get('level_z'), 2))}</td>"
        f"<td>{esc(x['status'])}</td><td>{esc('; '.join(x['reasons']))}</td></tr>"
        for x in body.get("measurements", []))
    lm = ", ".join(f"{k.replace('_z', '')} {v:.2f} m" for k, v in (body.get("landmarks") or {}).items() if v is not None)
    flags = "".join(f"<li>{esc(f)}</li>" for f in body.get("flags", []))
    img_html = "".join(f"<figure><img src='{esc(v)}' alt='{esc(k)}'><figcaption>{esc(k)}</figcaption></figure>"
                       for k, v in images.items() if v)
    warn = "".join(f"<li>{esc(w)}</li>" for w in q["warnings"])
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>RGB-D body reconstruction report</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;max-width:1300px;color:#1b1f24}}
.banner{{padding:10px 14px;border-radius:8px;margin:12px 0;background:#fbf0d9}} .status-pass{{background:#e3f4ea}}
.status-fail{{background:#fdeceb}} table{{border-collapse:collapse;margin:10px 0 24px;font-size:13px}}
td,th{{border-bottom:1px solid #e2e5ea;padding:4px 8px;text-align:left;vertical-align:top}} tr.fail td{{color:#b42318}}
tr.warn td{{color:#8a5a00}} img{{max-width:100%;border:1px solid #e2e5ea}} figure{{margin:0 0 18px}}</style></head><body>
<h1>RGB-D body reconstruction (Stray Scanner, fixed iPad, subject turning)</h1>
<div class='banner status-{esc(q['status'])}'><b>Status: {esc(q['status'])}</b>. Geometry is metric (LiDAR depth) but
<b>not validated</b>: no tape or known-object reference was captured, so no accuracy is claimed. All circumference
values are diagnostic (<code>unreliable</code>).</div>
<p>Run <code>{esc(log['run_id'])}</code>, pipeline <code>{esc(log['pipeline_version'])}</code>, pose algorithm
<code>{esc(log['pose']['algorithm_version'])}</code>, runtime {esc(log.get('runtime_s'))} s.</p>
<h2>Summary</h2><table>{fact_rows}</table>
<h2>Body measurements (definitions {esc(body.get('definition_version', ''))}, proposed, NOT validated)</h2>
<p>Landmarks: {esc(lm or 'none found')}. Torso values use the torso part of each slice only (arms excluded where they
are separated). Every value is <code>unreliable</code> until compared with tape on this rig.</p>
{('<ul>' + flags + '</ul>') if flags else ''}
<table><tr><th>measurement</th><th>value</th><th>level</th><th>status</th><th>reasons</th></tr>{meas_rows}</table>
<h2>Quality checks</h2><table><tr><th>check</th><th>status</th><th>value</th><th>limit</th><th>meaning</th></tr>{rows}</table>
{('<h2>Warnings</h2><ul>' + warn + '</ul>') if warn else ''}
<h2>Previews</h2>{img_html}
<h2>Files</h2><p><code>fused_mesh.ply</code> (measurement mesh, metres, floor z = 0, +z up), <code>fused_mesh.obj</code>,
<code>fused_mesh.glb</code> (same geometry, vertex colours from the video: display only), <code>fused_points.ply</code>,
<code>poses.json</code>, <code>quality_report.json</code>, <code>run_log.json</code> (every parameter and transform).</p>
</body></html>"""
    (run_dir / "report.html").write_text(doc, encoding="utf-8")
