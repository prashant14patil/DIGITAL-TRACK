"""EXPERIMENTAL review of a finished RGB visual-hull run: recording-quality warnings and rough slice readings.

Works only from files already in the run directory (report.json, silhouettes/, mesh), so older runs can be
reviewed without reprocessing. Thresholds are untested heuristics: "ok" means no problem was detected, NOT that
the scan is accurate. Slice readings are fractions of body height (the mesh has no real size) and `unreliable`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..measurement.basic import o3d_to_trimesh, slice_at_height

ALGORITHM = "scan-review-0.1 (EXPERIMENTAL)"
# forced-background margins used by rgb_visual_hull.segment_person: a mask cannot extend past them
TOP_MARGIN, SIDE_MARGIN = 0.10, 0.06
SLICE_LEVELS = (("about chest", 0.72), ("about waist", 0.62), ("about abdomen", 0.57), ("about hip", 0.50),
                ("about thigh (both legs together if touching)", 0.38), ("about calf (one outline only)", 0.18))


def _check(cid: str, bad: bool, ok_msg: str, warn_msg: str, value, limit) -> dict:
    return {"id": cid, "status": "warn" if bad else "ok", "message": warn_msg if bad else ok_msg,
            "value": value, "limit": limit}


def _aligned(mask: np.ndarray) -> np.ndarray:
    """Silhouette cropped to its bounding box and resized, to compare shape independently of position."""
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    crop = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].astype(np.uint8)
    return cv2.resize(crop, (128, 384), interpolation=cv2.INTER_NEAREST).astype(bool)


def quality_checks(run_dir: Path) -> dict | None:
    """Warnings about the recording. Returns None if the run has no silhouettes to look at."""
    run_dir = Path(run_dir)
    paths = sorted((run_dir / "silhouettes").glob("mask_*.png"))
    if not paths or not (run_dir / "report.json").is_file():
        return None
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    masks = {}
    for p in paths:
        m = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        if m.any():
            masks[int(p.stem.split("_")[1])] = m
    if not masks:
        return None
    top = side = bottom = 0
    heights = []
    for m in masks.values():
        h, w = m.shape
        rows, cols = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        top += int(rows[0] <= TOP_MARGIN * h + 3)
        bottom += int(rows[-1] >= h - 4)
        side += int(cols[0] <= SIDE_MARGIN * w + 3 or cols[-1] >= (1 - SIDE_MARGIN) * w - 4)
        heights.append((rows[-1] - rows[0]) / h)
    n = len(masks)
    checks = []
    for cid, count, where, advice in (
            ("head_room", top, "top", "move the camera back or tilt it so there is space above the head"),
            ("feet_room", bottom, "bottom", "keep both feet fully in the picture"),
            ("side_room", side, "side", "stand in the middle and move the camera back")):
        frac = round(count / n, 3)
        checks.append(_check(cid, frac > 0.05, f"Body stays clear of the {where} edge.",
                             f"Body reaches the {where} edge in {frac:.0%} of frames and may be cut off: {advice}.",
                             frac, 0.05))
    size = round(float(np.median(heights)), 3)
    checks.append(_check("body_size", size < 0.55, "Body is large enough in the picture.",
                         f"Body fills only {size:.0%} of the picture height: bring the camera closer.", size, 0.55))
    checks.append(_check("frame_count", n < 60, f"{n} frames used.",
                         f"Only {n} frames used: record a slower turn.", n, 60))

    outliers = round(len(report.get("segmentation", {}).get("area_outlier_frame_ids", [])) / n, 3)
    checks.append(_check("cutout_consistency", outliers > 0.10, "Person cut-out size is steady.",
                         f"Cut-out size jumps in {outliers:.0%} of frames: use a plain background and even light, "
                         "and check the cut-out preview.", outliers, 0.10))

    anchors = report.get("angles", {}).get("anchors", [])
    if len(anchors) == 5:
        quarters = np.diff([a["time_s"] for a in anchors])
        ratio = round(float(quarters.max() / max(quarters.min(), 1e-6)), 2)
        checks.append(_check("turn_speed", ratio > 1.5, "The turn looks fairly even.",
                             f"The turn speed is uneven (slowest quarter took {ratio}x the fastest): turn in equal "
                             "steps, pausing the same time at each.", ratio, 1.5))
        i0, i1 = anchors[0].get("frame_id"), anchors[-1].get("frame_id")
        if i0 in masks and i1 in masks:
            a, b = _aligned(masks[i0]), _aligned(masks[i1])
            iou = round(float((a & b).sum() / max((a | b).sum(), 1)), 3)
            checks.append(_check("start_end_match", iou < 0.85, "Start and end poses look alike.",
                                 f"Start and end poses differ (overlap {iou:.0%}): the arms, feet or posture moved "
                                 "during the turn, or the turn was not a full circle.", iou, 0.85))
    return {"algorithm": ALGORITHM, "warnings": sum(c["status"] == "warn" for c in checks), "checks": checks,
            "note": "Untested heuristic thresholds. No warning does not mean the scan is accurate."}


def rough_slices(mesh_path: Path) -> dict:
    """Outline lengths of horizontal slices as a percentage of body height."""
    import open3d as o3d

    tm = o3d_to_trimesh(o3d.io.read_triangle_mesh(str(mesh_path)))
    z0, z1 = float(tm.vertices[:, 2].min()), float(tm.vertices[:, 2].max())
    items = []
    for name, level in SLICE_LEVELS:
        cut = slice_at_height(tm, z0 + level * (z1 - z0))
        length = cut.perimeter_m  # in mesh units here: the mesh has no real size
        reason = ("closed outline" if length is not None else
                  "outline has a gap (hole in the mesh)" if cut.n_open_polylines else "no surface at this height")
        value = None if length is None else round(100.0 * length / (z1 - z0), 1)
        items.append({"name": name, "height_fraction": level, "value": value, "unit": "% of body height",
                      "status": "unavailable" if value is None else "unreliable", "reason": reason})
    return {"algorithm": ALGORITHM, "source_method": "visual_hull", "scale": "none (mesh has no real size)",
            "definition_version": "rough-levels-0", "items": items,
            "note": "Not body measurements. Fixed height levels, not anatomical landmarks; arms may be merged into "
                    "the trunk and hollow areas are filled in. For diagnosis only."}


def write_review_files(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    q = quality_checks(run_dir)
    if q is not None:
        (run_dir / "quality.json").write_text(json.dumps(q, indent=2) + "\n", encoding="utf-8")
    mesh = run_dir / "mesh_height_normalised.ply"
    if mesh.is_file():
        (run_dir / "slices.json").write_text(json.dumps(rough_slices(mesh), indent=2) + "\n", encoding="utf-8")
