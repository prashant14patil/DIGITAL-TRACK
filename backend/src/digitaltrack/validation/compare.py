"""Mesh vs tape: compare a depth reconstruction with the reference measurements typed on the upload page.

Inputs: the fused mesh of a self-rotation run (body frame, METRES, floor at z = 0, +z up) and the upload's
`measurements` ([{name, readings_mm, instrument, level_mm?}]). Nothing here adjusts or calibrates the mesh.

Definitions (version COMPARE_VERSION; every result records it):
  box     length / width: sides of the minimum-area rectangle around the mesh between 10 % and 90 % of its height
          height: 99.5th percentile of vertex height above the floor
  person  height: 99.9th percentile of vertex height above the floor (top of head, INCLUDES hair)
          circumference "<name>": perimeter of the largest closed contour of the horizontal mesh section at the
          tape level `level_mm` above the floor. Without a level there is no value (the slice level would be a
          guess). Torso names may be `measured`; limb names are always `estimated` (a horizontal cut is not
          perpendicular to the limb).

Status per value (CLAUDE.md): measured only if the run passed every check and the slice is clean;
`unreliable` if the run status is warn/fail (shown for diagnosis only); `unavailable` if no value can be computed.
One scan gives one error per measurement; MAE, bias, Bland-Altman and repeatability need several scans.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..measurement.basic import o3d_to_trimesh, slice_at_height

COMPARE_VERSION = "compare-0.1"
TORSO = ("chest", "bust", "waist", "abdomen", "belly", "hip", "neck")
LIMB = ("thigh", "calf", "arm", "bicep", "forearm", "knee", "ankle", "wrist")
CIRC_HINTS = TORSO + LIMB + ("circumference", "girth")


def _kind(name: str) -> str:
    n = name.lower()
    if re.search(r"\bheight\b|stature|tall", n):
        return "height"
    if "length" in n or n == "l":
        return "length"
    if "width" in n or n == "w":
        return "width"
    if any(h in n for h in CIRC_HINTS):
        return "circumference"
    return "unknown"


def _tape(readings: list[float]) -> dict:
    a = np.asarray(readings, float)
    return {"n": int(a.size), "mean_mm": float(a.mean()), "sd_mm": float(a.std(ddof=1)) if a.size > 1 else None,
            "range_mm": float(np.ptp(a)), "readings_mm": a.tolist()}


def _row(m: dict, kind: str, mesh_mm: float | None, status: str, why: str, extra: dict | None = None) -> dict:
    tape = _tape(m["readings_mm"])
    row = {"name": m["name"], "kind": kind, "tape": tape, "instrument": m.get("instrument", ""),
           "mesh_mm": None if mesh_mm is None else round(float(mesh_mm), 1), "status": status, "why": why,
           "unit": "mm", "definition_version": COMPARE_VERSION}
    if mesh_mm is not None:
        err = float(mesh_mm) - tape["mean_mm"]
        row.update(error_mm=round(err, 1), error_pct=round(100 * err / tape["mean_mm"], 2))
    if extra:
        row.update(extra)
    return row


def box_dimensions(vertices: np.ndarray) -> dict:
    z = vertices[:, 2]
    h = float(np.percentile(z, 99.5))
    band = vertices[(z > 0.1 * h) & (z < 0.9 * h)][:, :2].astype(np.float32)
    if len(band) < 10:
        return {"height_m": h, "length_m": None, "width_m": None}
    (_, _), (a, b), _ = cv2.minAreaRect(band)
    return {"height_m": h, "length_m": float(max(a, b)), "width_m": float(min(a, b))}


def compare_run(run_dir: str | Path, metadata: dict, *, write: bool = True) -> dict:
    import open3d as o3d

    run_dir = Path(run_dir)
    log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8")) if (run_dir / "run_log.json").is_file() else {}
    run_status = log.get("status", "unknown")
    tape = [m for m in metadata.get("measurements") or [] if m.get("readings_mm")]
    out: dict[str, Any] = {
        "compare_version": COMPARE_VERSION, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": log.get("run_id", run_dir.name), "run_status": run_status, "object_kind": metadata.get("object_kind"),
        "contains_person": metadata.get("contains_person"), "rows": [],
        "note": "One scan gives one error per measurement. Accuracy claims need several scans and repeat "
                "measurements (MAE, bias, Bland-Altman, repeatability). Units: mm.",
    }
    mesh_path = run_dir / "fused_mesh.ply"
    if not mesh_path.is_file():
        out["error"] = "this run has no mesh"
        for m in tape:
            out["rows"].append(_row(m, _kind(m["name"]), None, "unavailable", "no mesh"))
        return _finish(out, run_dir, write)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        out["error"] = "the mesh is empty"
        return _finish(out, run_dir, write)
    run_ok = run_status == "pass"
    base_status = "measured" if run_ok else "unreliable"
    base_why = "run passed all checks" if run_ok else f"run status is {run_status}: shown for diagnosis only"
    tm = o3d_to_trimesh(mesh)
    is_box = metadata.get("object_kind") == "box"
    dims = box_dimensions(v) if is_box else None
    out["mesh_extent_m"] = {"z_top_p999": float(np.percentile(v[:, 2], 99.9)), "z_min": float(v[:, 2].min())}
    if dims:
        out["box_from_mesh_m"] = dims

    # runs of the tracked RGB-D pipeline carry landmark-based body measurements (torso part only, arms excluded)
    bp = run_dir / "body_measurements.json"
    body = json.loads(bp.read_text(encoding="utf-8")) if bp.is_file() and not is_box else None
    if body is not None:
        out["body_definitions"] = body.get("definition_version")
    for m in tape:
        k = _kind(m["name"])
        if k == "height":
            if is_box:
                out["rows"].append(_row(m, k, dims["height_m"] * 1000, base_status, base_why))
            else:
                hz = float(np.percentile(v[:, 2], 99.9)) * 1000
                out["rows"].append(_row(m, k, hz, base_status, base_why + "; includes hair"))
        elif k in ("length", "width") and is_box:
            val = dims[f"{k}_m"]
            out["rows"].append(_row(m, k, None if val is None else val * 1000,
                                    base_status if val is not None else "unavailable",
                                    base_why if val is not None else "too few mesh points at mid-height"))
        elif k == "circumference" and body is not None:
            out["rows"].append(_body_row(m, k, mesh, body, base_status, base_why))
        elif k == "circumference":
            lvl = m.get("level_mm")
            if not lvl:
                out["rows"].append(_row(m, k, None, "unavailable",
                                        "no tape level given: enter the height above the floor where the tape was held"))
                continue
            s = slice_at_height(tm, float(lvl) / 1000)
            limb = any(h in m["name"].lower() for h in LIMB)
            if s.perimeter_m is None:
                out["rows"].append(_row(m, k, None, "unavailable", f"no closed contour at {lvl:.0f} mm",
                                        {"level_mm": lvl}))
                continue
            status, why = base_status, base_why
            if status == "measured" and (limb or s.status != "pass"):
                status = "estimated"
                why = "horizontal cut is not perpendicular to the limb" if limb else \
                    f"slice not clean ({s.n_closed_contours} closed contours, {s.n_open_polylines} open)"
            out["rows"].append(_row(m, k, s.perimeter_m * 1000, status, why,
                                    {"level_mm": lvl, "closed_contours": s.n_closed_contours,
                                     "open_polylines": s.n_open_polylines, "slice_status": s.status}))
        else:
            out["rows"].append(_row(m, k, None, "unavailable", "name not recognised (use length, width, height, "
                                    "or a body part such as chest, waist, hip, thigh)"))
    return _finish(out, run_dir, write)


BODY_NAMES = {"chest": "chest_circumference", "bust": "chest_circumference", "waist": "waist_circumference",
              "hip": "hip_circumference"}


def _body_row(m: dict, k: str, mesh, body: dict, base_status: str, base_why: str) -> dict:
    """Circumference from the body-measurement definitions: at the tape level if given (torso part only), else the
    landmark-based named value. Status never exceeds the body file's own status (unvalidated => unreliable)."""
    from ..measurement.body import torso_circumference_at

    name = m["name"].lower()
    limb = any(h in name for h in LIMB)
    body_status = body.get("status", "unreliable")
    status = base_status if body_status == "measured" else "unreliable"
    why = base_why if status == base_status else "body-measurement definitions are not validated yet"
    lvl = m.get("level_mm")
    if lvl and not limb:
        c = torso_circumference_at(mesh, float(lvl) / 1000)
        if c["value_m"] is None:
            return _row(m, k, None, "unavailable", c["reason"], {"level_mm": lvl, "method": "torso part at tape level"})
        return _row(m, k, c["value_m"] * 1000, status, why, {"level_mm": lvl, "method": "torso part at tape level",
                                                            "convex_hull_mm": round(c["convex_hull_m"] * 1000, 1)})
    key = next((v for t, v in BODY_NAMES.items() if t in name), None)
    if key is None and "thigh" in name:
        vals = [x for x in body["measurements"] if x["name"].startswith("thigh_circumference_side") and x["value"]]
        if vals:  # left/right are not assigned in the mesh: compare with the mean of both thighs
            v = float(np.mean([x["value"] for x in vals]))
            return _row(m, k, v * 1000, "unreliable", "mean of both thighs (left/right not assigned); " + why,
                        {"method": "landmark thigh level"})
    if key is None:
        return _row(m, k, None, "unavailable", "no body definition for this name; give the tape level to compare "
                    "the torso at that height")
    x = next((x for x in body["measurements"] if x["name"] == key), None)
    if not x or x["value"] is None:
        return _row(m, k, None, "unavailable", "; ".join((x or {}).get("reasons", ["not computed"])))
    return _row(m, k, x["value"] * 1000, status, why + "; level from landmarks (no tape level given)",
                {"level_mm": None if x.get("level_z") is None else round(x["level_z"] * 1000),
                 "method": f"{key} ({body.get('definition_version')})"})


def _finish(out: dict, run_dir: Path, write: bool) -> dict:
    vals = [r for r in out["rows"] if r.get("error_mm") is not None]
    out["summary"] = {"compared": len(vals), "unavailable": sum(r["status"] == "unavailable" for r in out["rows"]),
                      "max_abs_error_mm": max((abs(r["error_mm"]) for r in vals), default=None),
                      "any_measured": any(r["status"] == "measured" for r in out["rows"])}
    if write:
        (run_dir / "comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
