"""EXPERIMENTAL DEMO: approximate body mesh from ONE Camera-app RGB video of a person turning in place.

Lives in `digitaltrack.experimental` (optional experiments, never part of the primary pipeline). It is NOT the
product pipeline and produces NO measurements. It shows the RGB-only fallback idea (silhouette carving) on a
real recording that has none of the required references:
  * no camera calibration (orthographic projection is assumed),
  * no scale reference (the mesh is height-normalised: standing height = 1.0, or an operator-assumed height),
  * no measured turn angle (angles are ESTIMATED from the leg-width cycle and interpolated in time),
  * no fixed rotation axis (the axis is taken as the leg-band silhouette centre in every frame),
  * a non-rigid subject (arms move between views; a robust vote keeps voxels seen in >= vote_fraction of views).
Everything it writes is labelled `unreliable`. Input: a validated `video_rgb` session (see digitaltrack.ingest).
Used by tools/demo_rgb_visual_hull.py (CLI) and the web app's job runner (api/jobs.py).
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from digitaltrack import __version__
from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.validate import validate_session
from digitaltrack.ingest.video import find_tool

ALGORITHM = "demo-rgb-visual-hull-0.1 (EXPERIMENTAL)"


# ---- orientation ---------------------------------------------------------------------------------

def upright_rotation(session, frame0_rgb: np.ndarray) -> int:
    """cv2.rotate code that turns stored frames upright, decided by comparing with ffmpeg's own autorotation."""
    src = session.root / session.meta["source_video"]["file"]
    out = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-i", str(src), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True)
    best, best_err = None, np.inf
    for code in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        cand = frame0_rgb if code is None else cv2.rotate(frame0_rgb, code)
        h, w = cand.shape[:2]
        if len(out.stdout) != h * w * 3:
            continue
        ref = np.frombuffer(out.stdout, np.uint8).reshape(h, w, 3)
        err = float(np.mean(np.abs(ref.astype(int) - cand.astype(int))))
        if err < best_err:
            best, best_err = code, err
    if best_err > 20:
        raise RuntimeError(f"could not determine upright orientation (best mean abs diff {best_err:.1f})")
    return -1 if best is None else best


def rotate(img: np.ndarray, code: int) -> np.ndarray:
    return img if code == -1 else cv2.rotate(img, code)


# ---- segmentation --------------------------------------------------------------------------------

def segment_person(rgb: np.ndarray, iterations: int = 5) -> np.ndarray:
    """GrabCut with a generic person-in-the-middle prior; largest component, holes filled, thin bits removed."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[:, :int(0.06 * w)] = mask[:, int(0.94 * w):] = cv2.GC_BGD
    mask[:int(0.10 * h)] = cv2.GC_BGD
    mask[int(0.25 * h):int(0.85 * h), int(0.42 * w):int(0.58 * w)] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return np.zeros((h, w), bool)
    fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    flood = fg.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (fg | (1 - flood)).astype(bool)  # fill enclosed holes


def silhouette_features(m: np.ndarray) -> dict:
    rows = np.nonzero(m.any(axis=1))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    hgt = bottom - top
    band = m[top + int(0.62 * hgt): top + int(0.72 * hgt)]  # thigh band: below the hands in an A-pose
    cols = np.nonzero(band.any(axis=0))[0]
    return {"top": top, "bottom": bottom, "height": hgt, "area": int(m.sum()),
            "leg_width": float(cols[-1] - cols[0]) if cols.size else 0.0,
            "leg_centre": float((cols[-1] + cols[0]) / 2) if cols.size else float(m.shape[1] / 2)}


# ---- angles --------------------------------------------------------------------------------------

def estimate_angles(times: np.ndarray, width: np.ndarray, smooth_s: float = 0.4, min_prominence: float = 0.3,
                    min_separation_s: float = 3.0, plateau_tol: float = 0.05):
    """Anchor 0/90/180/270/360 deg at the alternating MAJOR leg-width extrema (max-min-max-min-max) and
    interpolate linearly in time. Extrema must swing by >= min_prominence of the signal range and be
    >= min_separation_s apart; each anchor is the centre of its plateau (within plateau_tol of the range)."""
    from scipy.signal import find_peaks

    dt = float(np.median(np.diff(times)))
    w = gaussian_filter1d(width, max(1.0, smooth_s / dt), mode="nearest")
    rng = float(w.max() - w.min())
    if rng <= 0:
        raise RuntimeError("leg-width signal is flat; angles cannot be estimated")
    pad = np.concatenate([[w.min() - rng], w, [w.min() - rng]])  # lets plateaus at the ends count as peaks
    dist = max(1, int(round(min_separation_s / dt)))
    pk, _ = find_peaks(pad, prominence=min_prominence * rng, distance=dist)
    tr, _ = find_peaks(-np.concatenate([[w.max() + rng], w, [w.max() + rng]]), prominence=min_prominence * rng,
                       distance=dist)
    ext = sorted([(int(i) - 1, "max") for i in pk] + [(int(i) - 1, "min") for i in tr])

    def plateau_centre(i: int, kind: str) -> int:
        ok = (w >= w[i] - plateau_tol * rng) if kind == "max" else (w <= w[i] + plateau_tol * rng)
        lo = hi = i
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(w) - 1 and ok[hi + 1]:
            hi += 1
        return (lo + hi) // 2

    seq = [(plateau_centre(i, k), k) for i, k in ext]
    while seq and seq[0][1] != "max":
        seq.pop(0)
    if len(seq) < 5 or [k for _, k in seq[:5]] != ["max", "min", "max", "min", "max"]:
        raise RuntimeError(f"leg-width cycle does not show one full turn (major extrema: {seq}); "
                           "angles cannot be estimated")
    anchors = [(seq[j][0], 90.0 * j) for j in range(5)]
    idx = np.array([a[0] for a in anchors])
    ang = np.interp(np.arange(len(w)), idx, [a[1] for a in anchors])  # clamps before/after the turn
    return ang, anchors, w


def turn_direction(rgbs, masks, i0: int, i1: int):
    """+1 if the visible front surface moves to image-right while turning from front (i0) to side (i1), else -1.
    Sums the median horizontal Farneback flow of the torso band over every consecutive frame pair."""
    total = 0.0
    for a, b in zip(range(i0, i1), range(i0 + 1, i1 + 1)):
        g0 = cv2.cvtColor(rgbs[a], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(rgbs[b], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        m = masks[a] & masks[b]
        rows = np.nonzero(m.any(axis=1))[0]
        torso = np.zeros_like(m)
        torso[rows[0] + int(0.2 * (rows[-1] - rows[0])): rows[0] + int(0.5 * (rows[-1] - rows[0]))] = True
        sel = m & torso
        if sel.any():
            total += float(np.median(flow[..., 0][sel]))
    return (1 if total > 0 else -1), total


# ---- carving -------------------------------------------------------------------------------------

def carve(masks, feats, angles_deg, sign: int, voxel_px: float, h_ref: float):
    """Vote count per voxel. Orthographic: column = c_k + (x cos t - y sin t)/s_k, row = floor_k - z/s_k."""
    r = 0.5 * max(max(f["leg_width"] for f in feats) * 2.2, max(m.any(axis=0).sum() for m in masks) * 1.1)
    xs = np.arange(-r, r, voxel_px)
    zs = np.arange(0, h_ref * 1.03, voxel_px)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    votes = np.zeros((len(xs), len(xs), len(zs)), np.uint16)
    for m, f, a in zip(masks, feats, angles_deg):
        s = h_ref / f["height"]  # per-frame scale: assumes constant standing height
        t = np.deg2rad(sign * a)
        col = np.round(f["leg_centre"] + (gx * np.cos(t) - gy * np.sin(t)) / s).astype(int)
        row = np.round(f["bottom"] - zs / s).astype(int)
        okc = (col >= 0) & (col < m.shape[1])
        okr = (row >= 0) & (row < m.shape[0])
        hit = np.zeros(votes.shape, bool)
        sub = m[np.clip(row, 0, m.shape[0] - 1)][:, np.clip(col, 0, m.shape[1] - 1)]  # (z, x, y)
        hit[:] = np.moveaxis(sub, 0, -1) & okc[..., None] & okr[None, None, :]
        votes += hit
    return votes, xs, zs


def mesh_from_votes(frac: np.ndarray, xs, zs, voxel_px, h_ref, threshold: float):
    f = gaussian_filter(frac.astype(np.float32), 1.0)
    inside = f >= threshold
    pad = np.pad(inside, 1)
    boundary = inside & ~(pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] & pad[1:-1, :-2, 1:-1] &
                          pad[1:-1, 2:, 1:-1] & pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:])
    gxg, gyg, gzg = np.gradient(f)
    ii = np.nonzero(boundary)
    pts = np.column_stack([xs[ii[0]], xs[ii[1]], zs[ii[2]]]) / h_ref  # height-normalised units
    nrm = -np.column_stack([gxg[ii], gyg[ii], gzg[ii]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.normals = o3d.utility.Vector3dVector(nrm)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.03))
    mesh = mesh.crop(pc.get_axis_aligned_bounding_box().scale(1.02, pc.get_center()))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, int(inside.sum())


# ---- previews ------------------------------------------------------------------------------------

def render(mesh, azimuth_deg: float, size=(480, 800)) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    bb = mesh.get_axis_aligned_bounding_box()
    c = bb.get_center()
    a = np.deg2rad(azimuth_deg)
    eye = c + 3.0 * np.array([np.sin(a), -np.cos(a), 0.0])
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=25, center=c.tolist(), eye=eye.tolist(), up=[0, 0, 1], width_px=size[0], height_px=size[1])
    ans = scene.cast_rays(rays)
    hit = np.isfinite(ans["t_hit"].numpy())
    n = ans["primitive_normals"].numpy()
    light = (c - eye) / np.linalg.norm(c - eye)
    shade = np.clip(-(n @ light), 0, 1) * 0.8 + 0.2
    img = np.where(hit, (shade * 230).astype(np.uint8), 245).astype(np.uint8)
    return np.flipud(img)  # ray-image rows run bottom-up for up=+Z; flip so the head is at the top


def width_plot(times, raw, smooth, anchors, angles, path):
    W, H, pad = 900, 360, 40
    img = np.full((H, W, 3), 255, np.uint8)
    def xy(t, v, lo, hi):
        return int(pad + (t - times[0]) / (times[-1] - times[0]) * (W - 2 * pad)), int(H - pad - (v - lo) / (hi - lo + 1e-9) * (H - 2 * pad))
    lo, hi = raw.min(), raw.max()
    for series, col in ((raw, (200, 200, 200)), (smooth, (40, 40, 200))):
        pts = np.array([xy(t, v, lo, hi) for t, v in zip(times, series)], np.int32)
        cv2.polylines(img, [pts], False, col, 2)
    for i, a in anchors:
        x, _ = xy(times[i], 0, 0, 1)
        cv2.line(img, (x, pad), (x, H - pad), (0, 150, 0), 1)
        cv2.putText(img, f"{a:.0f}deg", (x + 3, pad + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)
    cv2.putText(img, "leg-band silhouette width (grey raw, blue smoothed) and angle anchors", (pad, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


# ---- entry point -----------------------------------------------------------------------------------

def run_visual_hull(session_dir: str | Path, *, downscale: int = 2, voxel_px: float = 3.0, vote_fraction: float = 0.9,
                    reuse_masks: str | Path | None = None, assume_height_m: float | None = None,
                    progress=None) -> tuple[Path, dict]:
    """Run the experimental RGB visual-hull demo on a validated `video_rgb` session.

    Writes processing/<run_id>/ with mesh_height_normalised.{ply,glb}, previews, silhouettes and report.json.
    `progress(stage: str, fraction: float)` is called as the run advances (optional). Returns (run_dir, report).
    Raises RuntimeError/ValueError with a human-readable reason when the input cannot be processed.
    """
    def tell(stage: str, frac: float) -> None:
        if progress:
            progress(stage, float(frac))

    params = {"downscale": downscale, "voxel_px": voxel_px, "vote_fraction": vote_fraction,
              "reuse_masks": str(reuse_masks) if reuse_masks else None, "assume_height_m": assume_height_m}
    t0 = time.time()
    root = Path(session_dir)
    rep = validate_session(root)
    if not rep.ok:
        raise ValueError("session invalid: " + "; ".join(str(i) for i in rep.errors))
    s = load_session(root)
    if s.input_type != "video_rgb":
        raise ValueError("this demo needs a video_rgb session")
    run_id = "demo_vhull_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / "processing" / run_id
    (out / "silhouettes").mkdir(parents=True)

    tell("orienting", 0.0)
    first = s.frames[0].load_rgb()
    code = upright_rotation(s, first)
    rgbs, masks, feats, times = [], [], [], []
    n = len(s.frames)
    for k, fr in enumerate(s.frames):
        rgb = rotate(fr.load_rgb(), code)
        rgb = cv2.resize(rgb, (rgb.shape[1] // downscale, rgb.shape[0] // downscale), interpolation=cv2.INTER_AREA)
        if reuse_masks:
            m = cv2.imread(str(Path(reuse_masks) / "silhouettes" / f"mask_{fr.frame_id:06d}.png"), 0) > 0
            if m.shape != rgb.shape[:2]:
                raise RuntimeError("reused mask size differs (different downscale?)")
        else:
            m = segment_person(rgb)
        if not m.any():
            raise RuntimeError(f"no person found in frame {fr.frame_id}")
        rgbs.append(rgb); masks.append(m); feats.append(silhouette_features(m)); times.append(fr.timestamp_s)
        cv2.imwrite(str(out / "silhouettes" / f"mask_{fr.frame_id:06d}.png"), m.astype(np.uint8) * 255)
        tell("segmenting", (k + 1) / n)
    times = np.array(times)
    widths = np.array([f["leg_width"] for f in feats])
    tell("estimating turn angles", 0.0)
    angles, anchors, smooth = estimate_angles(times, widths)
    sign, dx = turn_direction(rgbs, masks, anchors[0][0], anchors[1][0])
    # front surface moving to image-right == body turning so it faces image-right == +rotation about +Z (camera
    # looks along +Y, image-right = +X); see geometry/transforms.py conventions
    heights = np.array([f["height"] for f in feats])
    h_ref = float(np.median(heights))
    tell("carving", 0.0)
    votes, xs, zs = carve(masks, feats, angles, sign, voxel_px, h_ref)
    frac = votes / len(masks)
    sens = {str(th): int((gaussian_filter(frac.astype(np.float32), 1.0) >= th).sum()) for th in (0.8, 0.9, 1.0)}
    tell("meshing", 0.0)
    mesh, n_inside = mesh_from_votes(frac, xs, zs, voxel_px, h_ref, vote_fraction)
    o3d.io.write_triangle_mesh(str(out / "mesh_height_normalised.ply"), mesh)
    o3d.io.write_triangle_mesh(str(out / "mesh_height_normalised.glb"), mesh)
    if assume_height_m:
        scaled = o3d.geometry.TriangleMesh(mesh).scale(assume_height_m, center=np.zeros(3))
        o3d.io.write_triangle_mesh(str(out / f"mesh_assumed_height_{assume_height_m:.2f}m.ply"), scaled)

    tell("rendering previews", 0.0)
    views = [render(mesh, a) for a in (0, 90, 180, 270, 45)]
    cv2.imwrite(str(out / "preview_mesh.png"), np.hstack(views))
    pick = np.linspace(0, len(masks) - 1, 8).astype(int)
    sil = [cv2.resize(np.where(masks[i][..., None], rgbs[i], (rgbs[i] * 0.2).astype(np.uint8)), (216, 384)) for i in pick]
    cv2.imwrite(str(out / "preview_silhouettes.png"), cv2.cvtColor(np.hstack(sil), cv2.COLOR_RGB2BGR))
    width_plot(times, widths, smooth, anchors, angles, out / "preview_angle_estimation.png")

    area = np.array([f["area"] for f in feats])
    area_outliers = [int(s.frames[i].frame_id) for i in np.nonzero(np.abs(area - np.median(area)) > 0.25 * np.median(area))[0]]
    report = {
        "run_id": run_id, "algorithm": ALGORITHM, "digitaltrack": __version__, "open3d": o3d.__version__,
        "opencv": cv2.__version__, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": s.session_id, "status": "unreliable",
        "status_reasons": [
            "experimental demo; not a validated pipeline",
            "no camera calibration: orthographic projection assumed",
            "no scale reference: mesh is height-normalised (standing height = 1.0); NOT metric",
            "turn angles estimated from the leg-width cycle and linearly interpolated, not measured",
            "rotation axis assumed to pass through the leg-band silhouette centre in every frame",
            "subject is non-rigid (arms move between views); robust vote used instead of strict intersection",
            "visual hull over-estimates concave regions (between legs, under arms, lumbar curve)",
        ],
        "measurements": "none computed: no metric scale, no calibration, unvalidated method",
        "parameters": params,
        "orientation": {"cv2_rotate_code": code, "method": "matched to ffmpeg autorotation of frame 0"},
        "angles": {"anchors": [{"frame_id": int(s.frames[i].frame_id), "time_s": float(times[i]), "angle_deg": a}
                               for i, a in anchors],
                   "direction_sign": sign,
                   "direction_method": "summed Farneback torso flow over the front->side quarter turn",
                   "torso_flow_dx_px": dx, "per_frame_deg": [float(a) for a in angles]},
        "segmentation": {"method": "OpenCV GrabCut, generic centred prior, largest component, holes filled",
                         "median_area_px": float(np.median(area)), "area_outlier_frame_ids": area_outliers},
        "carving": {"frames": len(masks), "voxel_px": voxel_px, "grid": list(votes.shape),
                    "height_ref_px": h_ref, "height_px_min_max": [int(heights.min()), int(heights.max())],
                    "vote_fraction": vote_fraction, "voxels_inside": n_inside,
                    "voxels_inside_by_threshold": sens},
        "mesh": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                 "watertight": bool(mesh.is_watertight()), "units": "height-normalised (not metres)"},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tell("done", 1.0)
    return out, report
