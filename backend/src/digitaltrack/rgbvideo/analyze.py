"""Analyse a Camera-app video of a person turning in place (RGB-only fallback route). Read-only on the video.

Steps (all logged in analysis.json):
  1. probe (ingest.video.probe_video) + SHA-256 of the source; the source is never modified or copied
  2. one decoding pass: motion score (changed-pixel fraction between consecutive frames, downscaled grey),
     sharpness (variance of the Laplacian), motion-energy map (where anything ever moved)
  3. stops = runs of still frames lasting at least `min_stop_s`
  4. keyframe per stop = sharpest frame in the central part of the stop; saved as lossless PNG (display-rotated)
  5. silhouette per keyframe: GrabCut initialised with the motion-energy bounding box; largest component, holes
     filled; quality flags (touches the image border, fragmented, size outlier vs other stops)
  6. report.html with the timeline, keyframes and silhouette overlays, and what is still missing for 3-D

Silhouette sizes are in PIXELS. Nothing here is metric and nothing here is a body measurement.
"""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .. import __version__
from ..ingest.video import decode_frames, frame_timestamps, probe_video

ANALYSIS_VERSION = "rgbvideo-analysis-0.1"


@dataclass(frozen=True)
class VideoAnalysisParams:
    analysis_width: int = 270  # downscaled width (display orientation) for motion analysis
    pixel_change: int = 12  # grey-level change counted as motion
    max_changed_fraction: float = 0.0015  # below this a frame pair is "still"
    min_stop_s: float = 0.25
    keyframe_central_fraction: float = 0.6  # choose the sharpest frame within the middle 60 % of a stop
    motion_energy_fraction: float = 0.005  # moving region: pixel changed in >= 0.5 % of frame pairs (and >= 2 times)
    bbox_percentile: float = 0.5  # robust bounds of the moving region (ignores stray noise pixels)
    bbox_margin: float = 0.06  # horizontal margin, fraction of the region size
    bbox_margin_vertical: float = 0.12  # larger: a still head may not show up as motion
    segment_height: int = 960  # GrabCut runs at this image height (display orientation)
    grabcut_iterations: int = 6
    background_sample_every: int = 20  # frames sampled for the background model
    diff_probable_fg: float = 10.0  # Lab distance to the predicted empty scene
    diff_sure_fg: float = 30.0
    open_px: int = 5  # morphological opening at segment resolution
    size_outlier_fraction: float = 0.25  # silhouette area differing from the median by more than this is flagged


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def display_rotate(img: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Apply the container's display rotation (what players show). ffmpeg's -90 means 90 deg clockwise."""
    k = int(round(-rotation_deg / 90.0)) % 4  # number of clockwise quarter turns
    return np.ascontiguousarray(np.rot90(img, k=-k)) if k else img


def _stops(still: np.ndarray, fps: float, min_stop_s: float) -> list[tuple[int, int]]:
    """Frame index ranges [start, end] where consecutive frame pairs are still for at least min_stop_s."""
    member = np.zeros(len(still) + 1, bool)  # frame i is still if pair (i-1, i) or (i, i+1) is still
    member[1:] |= still
    member[:-1] |= still
    runs, start = [], None
    for i, m in enumerate(member):
        if m and start is None:
            start = i
        if (not m or i == len(member) - 1) and start is not None:
            end = i if m else i - 1
            if (end - start + 1) / fps >= min_stop_s:
                runs.append((start, end))
            start = None
    return runs


@dataclass
class BackgroundModel:
    """Predicted empty-scene colour (Lab, at segment resolution) built from the video itself (no empty-scene clip).

    Wall rows: a smooth quadratic colour model fitted to wall pixels OUTSIDE the subject region (never covered).
    Floor rows: per-pixel temporal median over sampled frames (the feet turn and move, so the carpet shows through).
    """
    lab: np.ndarray  # (sh, sw, 3) float32
    floor_row: int
    info: dict


def build_background(samples: list[np.ndarray], rect_frac, ) -> BackgroundModel:
    stack = np.stack([cv2.cvtColor(s, cv2.COLOR_RGB2LAB).astype(np.float32) for s in samples])
    sh, sw = stack.shape[1:3]
    x0, y0, x1, y1 = (int(rect_frac[0] * sw), int(rect_frac[1] * sh), int(rect_frac[2] * sw), int(rect_frac[3] * sh))
    median = np.median(stack, axis=0)
    outside = np.ones((sh, sw), bool)
    outside[:, x0:x1] = False
    # floor line: the carpet is much bluer (lower b*) than the wall; use columns outside the subject region
    b_rows = np.array([np.median(median[r, outside[r], 2]) if outside[r].any() else np.nan for r in range(sh)])
    lo, hi = np.nanpercentile(b_rows, 5), np.nanpercentile(b_rows, 95)
    below = np.nonzero(b_rows < (lo + hi) / 2)[0]
    floor_row = int(below[below > sh // 3].min()) if (below > sh // 3).any() else sh
    # wall model: quadratic in (x, y) per Lab channel, fitted on outside pixels above the floor line
    ys, xs = np.nonzero(outside[: max(floor_row - 4, 1)])
    sel = np.random.default_rng(0).choice(len(ys), min(20000, len(ys)), replace=False)
    ys, xs = ys[sel], xs[sel]
    design = lambda x, y: np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    xn, yn = xs / sw, ys / sh
    coef = np.linalg.lstsq(design(xn, yn), median[ys, xs], rcond=None)[0]
    gy, gx = np.mgrid[0:sh, 0:sw]
    wall = (design(gx.ravel() / sw, gy.ravel() / sh) @ coef).reshape(sh, sw, 3)
    resid = np.linalg.norm(median[ys, xs] - (design(xn, yn) @ coef), axis=1)
    # skirting board / wall-floor junction is not smooth: use the temporal median from just above the floor line
    junction = max(0, floor_row - int(round(0.03 * sh)))
    lab = np.where((np.arange(sh) < junction)[:, None, None], wall, median).astype(np.float32)
    return BackgroundModel(lab, floor_row, {"method": "wall quadratic colour model + floor temporal median",
                                            "n_frames": len(samples), "floor_row_fraction": floor_row / sh,
                                            "wall_fit_residual_median_lab": float(np.median(resid))})


def _segment(frame_rgb: np.ndarray, rect_frac: tuple[float, float, float, float], p: VideoAnalysisParams,
             bg: BackgroundModel | None = None) -> tuple[np.ndarray, dict]:
    """GrabCut silhouette at `segment_height`, returned at full resolution (bool mask) plus metrics.

    With a background model, GrabCut is seeded from the colour difference to the predicted empty scene:
    small difference -> probable background, large -> probable / sure foreground; outside the subject region ->
    sure background. Without one it falls back to plain rectangle initialisation.
    """
    h, w = frame_rgb.shape[:2]
    s = p.segment_height / h
    small = cv2.resize(frame_rgb, (int(round(w * s)), p.segment_height), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    x0, y0, x1, y1 = rect_frac
    rect = (int(x0 * sw), int(y0 * sh), max(1, int((x1 - x0) * sw)), max(1, int((y1 - y0) * sh)))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    if bg is not None and bg.lab.shape[:2] == (sh, sw):
        de = np.linalg.norm(cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32) - bg.lab, axis=2)
        mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
        inside = np.zeros((sh, sw), bool)
        inside[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = True
        mask[inside] = cv2.GC_PR_BGD
        mask[inside & (de > p.diff_probable_fg)] = cv2.GC_PR_FGD
        mask[inside & (de > p.diff_sure_fg)] = cv2.GC_FGD
        if not (mask == cv2.GC_FGD).any() and not (mask == cv2.GC_PR_FGD).any():
            return np.zeros((h, w), bool), {"area_px": 0, "flags": ["no_foreground_difference"]}
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, p.grabcut_iterations,
                    cv2.GC_INIT_WITH_MASK)
    else:
        mask = np.zeros((sh, sw), np.uint8)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, p.grabcut_iterations,
                    cv2.GC_INIT_WITH_RECT)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    if p.open_px > 1:  # remove 1-2 px slivers (e.g. along the skirting board); arms are far thicker
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_px, p.open_px)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    metrics = {"components_before_cleanup": int(n - 1)}
    if n <= 1:
        return np.zeros((h, w), bool), {**metrics, "area_px": 0, "flags": ["empty_silhouette"]}
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    fg = (labels == best).astype(np.uint8)
    metrics["largest_component_share"] = float(areas.max() / areas.sum())
    filled = fg.copy()  # fill holes: flood the background from the border
    ff = np.zeros((sh + 2, sw + 2), np.uint8)
    inv = (1 - fg).astype(np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 2)
    filled[inv == 1] = 1
    full = cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(full)
    flags = []
    if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
        flags.append("touches_image_border")
    if metrics["largest_component_share"] < 0.9:
        flags.append("fragmented")
    if bg is not None:
        fl = int(round(bg.floor_row / s))  # floor line at full resolution
        below = full[fl:]
        if below.any():
            metrics["pixels_below_floor_line"] = int(below.sum())
            flags.append("feet_floor_contact_unreliable")  # shoes vs carpet and contact shadows
    metrics.update({"area_px": int(full.sum()), "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "height_px": int(ys.max() - ys.min() + 1), "width_px": int(xs.max() - xs.min() + 1),
                    "holes_filled_px": int((filled - fg).sum() / s ** 2), "flags": flags})
    return full, metrics


def analyze_video(video: str | Path, out_root: str | Path, *, params: VideoAnalysisParams | None = None,
                  label: str = "", run_id: str | None = None) -> dict:
    p = params or VideoAnalysisParams()
    video = Path(video)
    run_id = run_id or "rgbvideo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_root) / run_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    (out / "silhouettes").mkdir(exist_ok=True)

    sha_before = _sha256(video)
    probe = probe_video(video)
    rot = probe["display_rotation_deg"]
    ts = frame_timestamps(video, probe["video_stream_index"])
    fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 30.0

    # pass 1: motion, sharpness, motion energy
    motion, sharp, prev, energy, samples = [], [], None, None, []
    for idx, fr in enumerate(decode_frames(video, probe)):
        img = display_rotate(fr, rot)
        if img.dtype != np.uint8:
            img = (img >> 8).astype(np.uint8)
        if idx % p.background_sample_every == 0:
            s = p.segment_height / img.shape[0]
            samples.append(cv2.resize(img, (int(round(img.shape[1] * s)), p.segment_height), interpolation=cv2.INTER_AREA))
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(g, (p.analysis_width, int(round(g.shape[0] * p.analysis_width / g.shape[1]))),
                           interpolation=cv2.INTER_AREA).astype(np.int16)
        hh, ww = g.shape
        centre = g[hh // 4: 3 * hh // 4, ww // 4: 3 * ww // 4]
        sharp.append(float(cv2.Laplacian(centre, cv2.CV_64F).var()))
        if prev is not None:
            changed = np.abs(small - prev) > p.pixel_change
            motion.append(float(changed.mean()))
            energy = changed.astype(np.int32) if energy is None else energy + changed
        prev = small
    n = len(sharp)
    if n != len(ts):
        raise RuntimeError(f"decoded {n} frames but ffprobe listed {len(ts)} timestamps")
    motion_arr = np.array(motion)
    stops = _stops(motion_arr < p.max_changed_fraction, fps, p.min_stop_s)

    # moving region -> GrabCut rectangle (fractions of the display frame)
    min_count = max(2, int(np.ceil(p.motion_energy_fraction * max(n - 1, 1))))
    moving = energy >= min_count if energy is not None else None
    if moving is None or not moving.any():
        raise RuntimeError("nothing moved in the video; cannot locate the subject")
    ys, xs = np.nonzero(moving)
    eh, ew = moving.shape
    q = p.bbox_percentile
    bx0, bx1 = np.percentile(xs, [q, 100 - q])
    by0, by1 = np.percentile(ys, [q, 100 - q])
    mx, my = (bx1 - bx0) * p.bbox_margin, (by1 - by0) * p.bbox_margin_vertical
    rect = (max(0, (bx0 - mx) / ew), max(0, (by0 - my) / eh), min(1, (bx1 + mx) / ew), min(1, (by1 + my) / eh))

    bgm = build_background(samples, rect)
    del samples
    keys = []
    for k, (a, b) in enumerate(stops):
        span = b - a + 1
        lo = a + int(span * (1 - p.keyframe_central_fraction) / 2)
        hi = max(lo, b - int(span * (1 - p.keyframe_central_fraction) / 2))
        keys.append(int(lo + np.argmax(sharp[lo:hi + 1])))

    # pass 2: grab keyframes, save losslessly, segment
    stop_rows, wanted = [], {f: k for k, f in enumerate(keys)}
    for idx, fr in enumerate(decode_frames(video, probe)):
        if idx not in wanted:
            continue
        k = wanted[idx]
        img = display_rotate(fr, rot)
        kf = out / "keyframes" / f"stop_{k:02d}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        kf.write_bytes(buf.tobytes())
        mask, m = _segment(img if img.dtype == np.uint8 else (img >> 8).astype(np.uint8), rect, p, bgm)
        mp = out / "silhouettes" / f"stop_{k:02d}_mask.png"
        mp.write_bytes(cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes())
        a, b = stops[k]
        stop_rows.append({"stop": k, "start_frame": a, "end_frame": b, "start_s": ts[a] - ts[0], "end_s": ts[b] - ts[0],
                          "duration_s": ts[b] - ts[a], "keyframe": idx, "keyframe_time_s": ts[idx] - ts[0],
                          "keyframe_sharpness": sharp[idx], "keyframe_png": f"keyframes/{kf.name}",
                          "keyframe_sha256": hashlib.sha256(kf.read_bytes()).hexdigest(),
                          "silhouette_png": f"silhouettes/{mp.name}", "silhouette": m})
    stop_rows.sort(key=lambda r: r["stop"])
    areas = np.array([r["silhouette"]["area_px"] for r in stop_rows], float)
    if len(areas) and np.median(areas) > 0:
        for r in stop_rows:
            if abs(r["silhouette"]["area_px"] / np.median(areas) - 1) > p.size_outlier_fraction:
                r["silhouette"]["flags"].append("area_outlier_vs_other_stops")

    sha_after = _sha256(video)
    if sha_after != sha_before:
        raise RuntimeError("source video changed during analysis")
    result = {
        "analysis_version": ANALYSIS_VERSION, "run_id": run_id, "label": label,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software": {"digitaltrack": __version__, "opencv": cv2.__version__, "numpy": np.__version__},
        "source": {"path_name": video.name, "sha256": sha_before, "size_bytes": video.stat().st_size,
                   "unchanged_after_analysis": True},
        "probe": probe, "parameters": asdict(p),
        "frames": n, "fps_measured": fps, "duration_s": ts[-1] - ts[0],
        "subject_region_fraction": list(rect),
        "background_model": bgm.info,
        "stops": stop_rows,
        "motion_changed_fraction": motion_arr.tolist(),
        "sharpness": sharp,
        "depth_present": False,
        "missing_for_3d": [
            "per-stop turn angle (no turntable or markers; front/back silhouettes are nearly symmetric)",
            "metric scale (no object of known size in view)",
            "camera intrinsics for this video mode (no calibration recording)",
            "depth (Camera-app video does not store LiDAR depth)",
        ],
        "note": "Silhouette sizes are pixels in the display-rotated frame; nothing here is metric or a measurement.",
    }
    _timeline_png(out / "timeline.png", ts, motion_arr, stop_rows, p)
    _overlay_sheet(out, stop_rows)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _report(out, result)
    return result


def _timeline_png(path: Path, ts, motion, stops, p: VideoAnalysisParams, w: int = 1200, h: int = 260) -> None:
    img = np.full((h, w, 3), 255, np.uint8)
    t = np.array(ts[1:]) - ts[0]
    dur = max(t[-1], 1e-6)
    top = max(float(np.percentile(motion, 99)), p.max_changed_fraction * 4)
    for r in stops:
        cv2.rectangle(img, (int(r["start_s"] / dur * (w - 1)), 20), (int(r["end_s"] / dur * (w - 1)), h - 30), (220, 245, 225), -1)
        x = int(r["keyframe_time_s"] / dur * (w - 1))
        cv2.line(img, (x, 20), (x, h - 30), (60, 160, 60), 1)
        cv2.putText(img, str(r["stop"]), (x + 3, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 120, 40), 1, cv2.LINE_AA)
    pts = np.stack([t / dur * (w - 1), (h - 30) - np.clip(motion / top, 0, 1) * (h - 60)], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, (180, 80, 30), 1, cv2.LINE_AA)
    yth = int((h - 30) - p.max_changed_fraction / top * (h - 60))
    cv2.line(img, (0, yth), (w - 1, yth), (40, 40, 200), 1)
    cv2.putText(img, "motion (changed-pixel fraction) vs time; green = detected stops; red = still threshold",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, f"0 s", (4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, f"{dur:.1f} s", (w - 60, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def _overlay_sheet(out: Path, stops: list[dict], tile_h: int = 480) -> None:
    tiles = []
    for r in stops:
        img = cv2.imdecode(np.fromfile(out / r["keyframe_png"], np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(out / r["silhouette_png"], np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        ov = img.copy()
        ov[mask] = (0.55 * ov[mask] + 0.45 * np.array([60, 200, 60])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, (0, 0, 255), 3)
        s = tile_h / ov.shape[0]
        t = cv2.resize(ov, (int(ov.shape[1] * s), tile_h), interpolation=cv2.INTER_AREA)
        flags = r["silhouette"]["flags"]
        cv2.putText(t, f"stop {r['stop']}  {r['keyframe_time_s']:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)
        if flags:
            cv2.putText(t, ",".join(flags)[:40], (6, tile_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(t)
    if tiles:
        cv2.imwrite(str(out / "silhouettes_overlay.png"), np.concatenate(tiles, axis=1))


def _report(out: Path, r: dict) -> None:
    e = lambda s: html.escape(str(s))
    rows = "".join(
        f"<tr><td>{s['stop']}</td><td>{s['start_s']:.2f}-{s['end_s']:.2f}</td><td>{s['duration_s']:.2f}</td>"
        f"<td>{s['keyframe']}</td><td>{s['silhouette'].get('height_px', '')}</td><td>{s['silhouette'].get('width_px', '')}</td>"
        f"<td>{e(', '.join(s['silhouette']['flags']) or 'none')}</td></tr>" for s in r["stops"])
    pr = r["probe"]
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Video analysis</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#1b1f24}} table{{border-collapse:collapse;margin:8px 0 22px}}
td,th{{border-bottom:1px solid #e2e5ea;padding:4px 10px;text-align:left}} img{{max-width:100%;border:1px solid #e2e5ea}}
.box{{background:#fbf0d9;padding:10px 14px;border-radius:8px}} code{{font-family:Consolas,monospace}}</style></head><body>
<h1>RGB video analysis: {e(r['label'] or r['source']['path_name'])}</h1>
<div class='box'><b>No 3-D reconstruction and no measurements from this video.</b> Still missing:
<ul>{''.join(f'<li>{e(m)}</li>' for m in r['missing_for_3d'])}</ul></div>
<p><code>{e(r['source']['path_name'])}</code>, SHA-256 <code>{e(r['source']['sha256'][:16])}…</code> (unchanged after
analysis). {e(pr['codec'])} {pr['width']}x{pr['height']}, display rotation {pr['display_rotation_deg']:.0f} deg,
{r['frames']} frames, {r['fps_measured']:.2f} fps measured, {r['duration_s']:.1f} s. Probe flags: {e(', '.join(pr['flags']) or 'none')}.</p>
<h2>Stops ({len(r['stops'])})</h2><img src='timeline.png' alt='timeline'>
<table><tr><th>stop</th><th>time (s)</th><th>length (s)</th><th>keyframe</th><th>silhouette height (px)</th>
<th>width (px)</th><th>flags</th></tr>{rows}</table>
<h2>Keyframes with silhouettes</h2><p>Green fill + red outline = segmented person (GrabCut, classical; check it by eye).</p>
<img src='silhouettes_overlay.png' alt='silhouettes'>
<p>Lossless keyframes: <code>keyframes/</code>; masks: <code>silhouettes/</code>; everything: <code>analysis.json</code>.
Pixel sizes are not metric.</p></body></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")
