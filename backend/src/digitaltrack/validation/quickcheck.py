"""Quick recording check (about half a minute): is this depth recording good enough to reconstruct?

Meant to run right after upload, before leaving the room. It does NOT build a mesh. It reuses the pipeline's
public steps with the website job's parameters (stop detection, gravity floor, no-plate ROI and cluster rules,
partial-view drop) on the FIRST and LAST stops only, then registers the last stops DIRECTLY against the first.

Why direct: summing the stop-to-stop turns under-counts the turn on real bodies (~13 % on the first real
recording; a smooth body front matches itself near zero rotation), so "summed turn" is not trusted as coverage.
The direct last->first registration says how far the subject still was from the starting direction.

Every limit is a PROPOSED value (one real recording so far). Each check says what to do differently.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.transforms import invert, transform_points
from ..rgbd.pipeline import NO_BACKGROUND_ROI_M, PARTIAL_VIEW_DROP_M
from ..rgbd.pose import PoseParams, register_pair, se2_params
from ..rgbd.segment import (SegmentationError, SegmentParams, background_plate, estimate_floor, make_floor_frame,
                            segment_subject)
from ..rgbd.stops import StopParams, detect_stops, median_view, stop_threshold
from ..rgbd.views import RGBDView, valid_mask

QUICKCHECK_VERSION = "quickcheck-0.2"  # 0.2: body-capture checks (validation/capture_checks.py)
LIMITS = {  # proposed, from one real recording; unvalidated
    "min_stops": 16,  # 15 deg steps over a full turn give ~24
    "min_pause_s": 1.5,
    "max_ipad_moved_m": 0.02,
    "full_turn_short_deg": 8.0,  # last stop more than this short of the starting direction -> not a full turn
    "full_turn_max_ambiguity": 0.5,  # the decisive last->first match must be clearly unambiguous
}


def _check(cid: str, status: str, message: str, value=None, limit=None) -> dict:
    return {"id": cid, "status": status, "message": message, "value": value, "limit": limit}


def quick_check(views: list[RGBDView], *, info: dict | None = None, background_views: list[RGBDView] | None = None,
                stops_params: StopParams | None = None, pose_params: PoseParams | None = None) -> dict[str, Any]:
    """Stop/turn checks (below) + body-capture checks (validation.capture_checks) when an empty scene is given."""
    out = _stop_checks(views, info=info, background_views=background_views, stops_params=stops_params,
                       pose_params=pose_params)
    if background_views:
        from .capture_checks import body_capture_checks

        body, det = body_capture_checks(views, background_views, info)
        out["body_capture"] = det
        out = _finish(out, out["checks"] + body)
    else:
        out = _finish(out, out["checks"] + [_check("body_capture", "warn", "body checks (distance, legs apart, arms "
                                                   "away and still) need the empty-scene recording: select it and "
                                                   "check again")])
    return out


def _stop_checks(views: list[RGBDView], *, info: dict | None = None, background_views: list[RGBDView] | None = None,
                 stops_params: StopParams | None = None, pose_params: PoseParams | None = None) -> dict[str, Any]:
    info = info or {}
    sp = stops_params or StopParams(adaptive=True, min_frames=6)  # same as the website job
    pp = pose_params or PoseParams()
    checks: list[dict] = []
    out: dict[str, Any] = {"version": QUICKCHECK_VERSION, "limits": LIMITS,
                           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "recording": {k: info.get(k) for k in ("duration_s", "depth_size", "ipad_moved_m", "held")}}

    # recording basics
    moved = info.get("ipad_moved_m")
    if moved is not None:
        bad = moved > LIMITS["max_ipad_moved_m"]
        checks.append(_check("ipad_still", "warn" if bad else "ok",
                             f"iPad moved {moved * 100:.1f} cm during the recording" +
                             (": fix the mount so it cannot move" if bad else ""), moved, LIMITS["max_ipad_moved_m"]))
    checks.append(_check("empty_room", "ok" if background_views else "warn",
                         "empty-room recording given" if background_views else
                         "no empty-room recording: record 10 s of the empty scene (separates you from walls)"))

    # stops
    stops, scores = detect_stops(views, sp)
    times = [v.timestamp_s for v in views]
    pauses = [float(times[s.end] - times[s.start]) for s in stops] if all(t is not None for t in times) else []
    out["stops"] = {"n": len(stops), "threshold": stop_threshold(scores, sp), "pause_s": pauses}
    if len(stops) < 2:
        checks.append(_check("stops", "fail", f"only {len(stops)} pauses found: turn in small steps and hold still "
                             "about 3 s at each step", len(stops), LIMITS["min_stops"]))
        return _finish(out, checks)
    checks.append(_check("stops", "ok" if len(stops) >= LIMITS["min_stops"] else "warn",
                         f"{len(stops)} pauses found" + ("" if len(stops) >= LIMITS["min_stops"] else
                                                         ": use smaller steps (about 15 deg, ~24 stops per turn)"),
                         len(stops), LIMITS["min_stops"]))
    if pauses:
        med = float(np.median(pauses))
        checks.append(_check("pause_length", "ok" if med >= LIMITS["min_pause_s"] else "warn",
                             f"median pause {med:.1f} s" + ("" if med >= LIMITS["min_pause_s"] else
                                                            ": hold each stop about 3 s"), med, LIMITS["min_pause_s"]))

    # first and last stops only, segmented exactly as the pipeline does
    pick = sorted({0, 1, len(stops) - 3, len(stops) - 2, len(stops) - 1} & set(range(len(stops))))
    key = {i: median_view(views, stops[i], sp) for i in pick}
    ups = [v.up_camera for v in list(key.values()) + list(background_views or [])]
    up_kw = {}
    if ups and all(u is not None for u in ups):
        u = np.mean(ups, axis=0)
        up_kw = {"up_hint": tuple(float(x) for x in u / np.linalg.norm(u)), "gravity": True}
    seg = SegmentParams()
    try:
        floor = estimate_floor(list(background_views) if background_views else list(key.values()), seg,
                               source="background_views" if background_views else "subject_views", **up_kw)
    except SegmentationError as e:
        checks.append(_check("floor", "fail", f"floor not found ({e}); keep the floor around the feet in view"))
        return _finish(out, checks)
    checks.append(_check("floor", "ok", f"floor found (tilt {floor.tilt_deg:.1f} deg from "
                         f"{'gravity' if up_kw else 'image'} up)", floor.tilt_deg))
    plate = background_plate(background_views, seg) if background_views else None
    k0 = key[pick[0]]
    m0 = valid_mask(k0, min_confidence=seg.min_confidence, depth_min_m=seg.depth_min_m, depth_max_m=seg.depth_max_m)
    hint_z = float(np.median(k0.depth_m[m0]))
    if plate is None:
        h, w = k0.depth_m.shape
        centre = np.zeros_like(m0)
        centre[h // 4:3 * h // 4, 2 * w // 5:3 * w // 5] = True
        if (m0 & centre).sum() >= 50:
            hint_z = float(np.percentile(k0.depth_m[m0 & centre], 25))
        seg = replace(seg, roi_radius_m=NO_BACKGROUND_ROI_M)
    provisional = make_floor_frame(floor, np.array([0.0, 0.0, hint_z]))
    seg0, segn = (replace(seg, cluster_select="widest"), replace(seg, cluster_select="nearest_origin")) \
        if plate is None else (seg, seg)
    try:
        s0 = segment_subject(k0, provisional, seg0, plate)
    except SegmentationError as e:
        checks.append(_check("subject", "fail", f"could not find you in the first stop ({e}); stand centred, "
                             "whole body in view"))
        return _finish(out, checks)
    c = np.array([*s0.points_floor[:, :2].mean(0), 0.0])
    f_from_cam = make_floor_frame(floor, transform_points(invert(provisional), c[None])[0])
    clouds = {}
    for i in pick:
        try:
            clouds[i] = segment_subject(key[i], f_from_cam, segn, plate).points_floor
        except SegmentationError:
            pass
    tops = {i: float(np.percentile(p[:, 2], 99)) for i, p in clouds.items()}
    out["subject_top_m"] = tops
    if tops:
        med_top = float(np.median(list(tops.values())))
        clouds = {i: p for i, p in clouds.items() if tops[i] >= med_top - PARTIAL_VIEW_DROP_M}
    first = [i for i in pick[:2] if i in clouds]
    last = [i for i in reversed(pick[2:]) if i in clouds and i not in first]
    if not first or not last:
        checks.append(_check("subject", "fail", "could not segment you in the first or last stops"))
        return _finish(out, checks)
    checks.append(_check("subject", "ok", f"found you in {len(clouds)} of {len(pick)} checked stops "
                         f"(height about {np.median(list(tops.values())):.2f} m incl. hair; not a measurement)"))

    # turning direction from the first step, then last -> first directly
    e01 = register_pair(clouds[first[0]], clouds[first[1]], first[0], first[1], "sequential", pp) \
        if len(first) > 1 else None
    direction = np.sign(se2_params(e01.transform)[0]) if e01 is not None and e01.ok else 0.0
    out["first_step_deg"] = float(np.degrees(se2_params(e01.transform)[0])) if e01 is not None and e01.transform is not None else None
    # Only the FINAL stop answers "did the turn come back to the start?". An earlier stop would need the missing
    # steps added (bias-prone), so it is not used. The match must be clearly unambiguous; otherwise: unconfirmed.
    i = len(stops) - 1
    e = register_pair(clouds[i], clouds[first[0]], i, first[0], "closure", pp) if i in clouds else None
    clear = e is not None and e.ok and e.ambiguity_ratio is not None and \
        e.ambiguity_ratio <= LIMITS["full_turn_max_ambiguity"]
    if not clear or direction == 0:
        why = "the last stop was dropped" if e is None else ("; ".join(e.reasons) if not e.ok else
                                                             f"match ambiguous ({e.ambiguity_ratio:.2f})")
        out["last_to_first"] = {"from_stop": i, "confirmed": False, "why": why}
        checks.append(_check("full_turn", "warn", "could not confirm a full turn (the last stop does not clearly match "
                             "the first): end facing the same way you started, then one more step"))
        return _finish(out, checks)
    th = float(np.degrees(se2_params(e.transform)[0]))
    short = float(th * direction)  # >0: still turning in the same direction to reach the start -> stopped early
    out["last_to_first"] = {"from_stop": i, "turn_deg": th, "short_of_start_deg": short, "overlap": e.overlap,
                            "ambiguity": e.ambiguity_ratio}
    if short > LIMITS["full_turn_short_deg"]:
        checks.append(_check("full_turn", "warn", f"the turn stopped about {short:.0f} deg short of a full circle: "
                             "keep going until you face the start again, then one extra step", short,
                             LIMITS["full_turn_short_deg"]))
    else:
        checks.append(_check("full_turn", "ok", "full turn completed (last stop back at or past the start, "
                             f"{abs(short):.0f} deg {'past' if short < 0 else 'from'} it)", short,
                             LIMITS["full_turn_short_deg"]))
    return _finish(out, checks)


def _finish(out: dict, checks: list[dict]) -> dict:
    out["checks"] = checks
    worst = "fail" if any(c["status"] == "fail" for c in checks) else \
        "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    out["verdict"] = worst
    out["verdict_text"] = {"ok": "Good to process.",
                           "warn": "Usable, but fix the warnings next time (or re-record now if you can).",
                           "fail": "Re-record: this recording cannot be reconstructed."}[worst]
    return out


def write(result: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"quickcheck_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    p.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    return p
