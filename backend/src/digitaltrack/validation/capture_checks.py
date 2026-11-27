"""Body-capture checks for a recording (seconds, no mesh): will this recording give MEASURABLE geometry?

Added after the first real person scan, where every remaining defect came from the capture, not the processing:
the camera was 2.2 m away (~1 cm per depth pixel: hands, the gap between the thighs and the crotch were not
resolved), the thighs touched (crotch and thighs unmeasurable), the arms changed position during a 112 s turn
(ghost arm surfaces) and hung against the body in side views (arms merged into waist/hip outlines).

Needs the empty-scene recording (background plate). Uses ~`n_frames` frames spread over the recording, the depth
subject mask (rgbd.segment.segment_subject_plate) and the camera's horizontal axis in the floor frame. Analysis uses
FRONT/BACK frames only (body widest across the image), at fixed fractions of the body height:
  * resolution: metres per depth pixel at the subject (distance / focal length)
  * in_frame: subject mask touching the image border
  * legs_apart: thighs (40 % of height) separated by >= `min_gap_m` across the image
  * arms_away: at 60 % of height, separate parts on both sides of the torso (arms not against the body)
  * arms_still: spread (SD) of each arm's distance from the torso centre over the turn
  * turn_duration: long turns make holding a pose harder
All limits are PROPOSED from one recording; each message says what to change.
"""

from __future__ import annotations

import numpy as np

from ..geometry.transforms import invert, transform_points
from ..rgbd.segment import (SegmentationError, SegmentParams, background_plate, estimate_floor, make_floor_frame,
                            segment_subject_plate)
from ..rgbd.views import RGBDView

LIMITS = {
    "max_m_per_px": 0.0085,  # ~1.8 m with the iPad's 256 x 192 depth
    "max_border_share": 0.2,
    "min_gap_m": 0.025,
    "min_legs_apart_share": 0.7,
    "min_arms_away_share": 0.7,
    "max_arm_sd_m": 0.03,
    "max_turn_s": 60.0,
}


def _chk(cid, status, message, value=None, limit=None):
    return {"id": cid, "status": status, "message": message, "value": value, "limit": limit}


def _clusters_1d(t: np.ndarray, gap: float) -> list[tuple[float, float, int]]:
    if len(t) == 0:
        return []
    s = np.sort(t)
    cuts = np.flatnonzero(np.diff(s) > gap)
    starts, ends = np.r_[0, cuts + 1], np.r_[cuts, len(s) - 1]
    return [(float(s[a]), float(s[b]), int(b - a + 1)) for a, b in zip(starts, ends) if b - a + 1 >= 5]


def body_capture_checks(views: list[RGBDView], background_views: list[RGBDView], info: dict | None = None,
                        n_frames: int = 40) -> tuple[list[dict], dict]:
    info = info or {}
    seg = SegmentParams(background_margin_m=0.05)
    checks: list[dict] = []
    det: dict = {"limits": LIMITS}
    dur = info.get("duration_s")
    if dur is not None:
        checks.append(_chk("turn_duration", "warn" if dur > LIMITS["max_turn_s"] else "ok",
                           f"recording lasts {dur:.0f} s" + (": turn steadily in 30-40 s; the longer the turn, the more "
                                                             "the arms and posture drift" if dur > LIMITS["max_turn_s"] else ""),
                           dur, LIMITS["max_turn_s"]))
    ups = [v.up_camera for v in background_views if v.up_camera is not None]
    up_kw = {"up_hint": tuple(np.mean(ups, 0) / np.linalg.norm(np.mean(ups, 0))), "gravity": True} if ups else {}
    try:
        floor = estimate_floor(background_views, seg, source="background_views", **up_kw)
    except SegmentationError as e:
        checks.append(_chk("body_capture", "warn", f"body checks skipped: {e}"))
        return checks, det
    plate = background_plate(background_views, seg)
    idx = np.linspace(0, len(views) - 1, min(n_frames, len(views))).round().astype(int)
    prov = make_floor_frame(floor, np.array([0.0, 0.0, 2.0]))
    frames = []
    for i in idx:
        try:
            s = segment_subject_plate(views[i], prov, plate, seg)
            frames.append((views[i], s))
        except SegmentationError:
            continue
    if len(frames) < 5:
        checks.append(_chk("body_capture", "warn", "the person was not found in front of the empty scene in most frames"))
        return checks, det
    c0 = np.median(np.concatenate([s.points_floor[:, :2] for _, s in frames[:3]]), 0)
    f_from_c = make_floor_frame(floor, transform_points(invert(prov), np.array([[*c0, 0.0]]))[0])
    lat = f_from_c[:3, :3] @ np.array([1.0, 0.0, 0.0])  # camera x (across the image) in the floor frame
    lat = lat[:2] / np.linalg.norm(lat[:2])

    # resolution and framing
    k = frames[0][0].intrinsics
    dist = float(np.median([np.median(v.depth_m[s.mask]) for v, s in frames]))
    mpp = dist / ((k.fx + k.fy) / 2)
    det["subject_distance_m"], det["m_per_depth_px"] = dist, mpp
    checks.append(_chk("resolution", "warn" if mpp > LIMITS["max_m_per_px"] else "ok",
                       f"subject {dist:.1f} m away: {mpp * 1000:.1f} mm per depth pixel" +
                       (f". Move the iPad closer (about {LIMITS['max_m_per_px'] * (k.fx + k.fy) / 2:.1f} m) so hands, "
                        "the gap between the legs and the crotch are resolved" if mpp > LIMITS["max_m_per_px"] else ""),
                       mpp, LIMITS["max_m_per_px"]))
    border = np.mean([s.mask[0].any() or s.mask[-1].any() or s.mask[:, 0].any() or s.mask[:, -1].any()
                      for _, s in frames])
    checks.append(_chk("in_frame", "warn" if border > LIMITS["max_border_share"] else "ok",
                       "whole body inside the picture" if border <= LIMITS["max_border_share"] else
                       f"the body touches the picture edge in {border:.0%} of frames: step back or re-aim so head, "
                       "hands and feet stay inside", float(border), LIMITS["max_border_share"]))

    # front/back frames: body widest across the image at 60 % of its height
    to_body = f_from_c @ invert(prov)  # provisional floor frame -> floor frame centred on the person
    per = []
    for v, s in frames:
        p = transform_points(to_body, s.points_floor)
        h = float(np.percentile(p[:, 2], 99.5))
        t = p[:, :2] @ lat  # position across the image, 0 = the person's standing spot
        per.append((p, h, t))
    hs = np.median([h for _, h, _ in per])
    det["body_top_m"] = float(hs)
    widths = []
    for p, h, t in per:
        band = np.abs(p[:, 2] - 0.6 * hs) < 0.015
        widths.append(float(np.ptp(t[band])) if band.sum() > 10 else 0.0)
    widths = np.array(widths)
    fb = widths >= np.percentile(widths, 60)
    det["front_back_frames"] = int(fb.sum())
    gap = LIMITS["min_gap_m"]

    legs_ok, arms_ok, offs = [], [], []
    for (p, h, t), use in zip(per, fb):
        if not use:
            continue
        thigh = np.abs(p[:, 2] - 0.40 * hs) < 0.015
        cl = [c for c in _clusters_1d(t[thigh], gap) if abs(0.5 * (c[0] + c[1])) < 0.25]
        legs_ok.append(len(cl) >= 2)
        waist = np.abs(p[:, 2] - 0.60 * hs) < 0.015
        cl = _clusters_1d(t[waist], gap)
        if cl:
            torso = max(cl, key=lambda c: c[2])
            tc = 0.5 * (torso[0] + torso[1])
            left = [c for c in cl if c[1] < torso[0]]
            right = [c for c in cl if c[0] > torso[1]]
            arms_ok.append(bool(left and right))
            if left and right:
                offs.append((tc - 0.5 * (left[-1][0] + left[-1][1]), 0.5 * (right[0][0] + right[0][1]) - tc))
    if legs_ok:
        share = float(np.mean(legs_ok))
        checks.append(_chk("legs_apart", "ok" if share >= LIMITS["min_legs_apart_share"] else "warn",
                           f"thighs separated in {share:.0%} of front/back views" +
                           ("" if share >= LIMITS["min_legs_apart_share"] else
                            ": stand with feet about 20-25 cm apart (tape marks) so the thighs do not touch; otherwise "
                            "crotch, hip and thigh cannot be measured"), share, LIMITS["min_legs_apart_share"]))
    if arms_ok:
        share = float(np.mean(arms_ok))
        checks.append(_chk("arms_away", "ok" if share >= LIMITS["min_arms_away_share"] else "warn",
                           f"both arms clear of the body at waist height in {share:.0%} of front/back views" +
                           ("" if share >= LIMITS["min_arms_away_share"] else
                            ": hold the arms about 30 deg away from the body for the whole turn"), share,
                           LIMITS["min_arms_away_share"]))
    if len(offs) >= 4:
        o = np.array(offs)
        sd = float(np.max(np.std(np.sort(o, axis=1), axis=0)))  # sorted: front and back views mirror left/right
        det["arm_offset_sd_m"] = sd
        checks.append(_chk("arms_still", "ok" if sd <= LIMITS["max_arm_sd_m"] else "warn",
                           f"arm position varied by {sd * 100:.1f} cm (SD) over the turn" +
                           ("" if sd <= LIMITS["max_arm_sd_m"] else
                            ": keep the arms frozen while turning (moving arms leave ghost surfaces)"), sd,
                           LIMITS["max_arm_sd_m"]))
    return checks, det
