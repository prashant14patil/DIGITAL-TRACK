"""Per-scan quality checks. Poor scans are flagged, not forced into confident results.

Thresholds are provisional engineering defaults, to be set properly during validation.

v0.2 adds the multi-view consistency checks (quality/registration.py). Their thresholds were
set from CORRECTLY calibrated synthetic scans only (worst median translation 3.1 mm, worst
median tilt 0.04 deg; see docs/phase0_verification_report.md) and must be re-derived from
real correctly-calibrated scans before use on real data. They cannot detect calibration
errors along the camera line of sight (a known blind spot).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import OUTPUT_SCHEMA_VERSION
from ..dataset.schemas import schema_errors
from ..dataset.writer import utc_now

ALGORITHM_VERSION = "quality-basic-0.2"


@dataclass(frozen=True)
class QualityThresholds:
    min_valid_frames: int = 12
    min_valid_pixels_per_frame: int = 500
    max_gap_fail_deg: float = 30.0
    max_gap_warn_deg: float = 15.0
    min_mesh_triangles: int = 100
    min_valid_pixel_fraction_warn: float = 0.02
    consistency_translation_warn_m: float = 0.004
    consistency_translation_fail_m: float = 0.005
    consistency_tilt_warn_deg: float = 0.08
    consistency_tilt_fail_deg: float = 0.15
    consistency_min_fitness: float = 0.8


def max_angular_gap_deg(angles_deg: list[float]) -> float:
    if len(angles_deg) < 2:
        return 360.0
    a = np.sort(np.mod(np.asarray(angles_deg, dtype=float), 360.0))
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(gaps.max())


def _check(name, status, value, threshold, message):
    return {"name": name, "status": status, "value": value, "threshold": threshold, "message": message}


def _level(value: float, warn: float, fail: float) -> str:
    return "fail" if value > fail else "warn" if value > warn else "pass"


def consistency_checks(consistency: dict | None, th: QualityThresholds) -> list[dict]:
    if consistency is None:
        return [_check("view_consistency", "warn", None, None, "multi-view consistency was not computed")]
    s = consistency.get("summary")
    if s is None:
        return [_check("view_consistency", "fail", None, None,
                       "multi-view consistency could not be computed (empty mesh or too few points)")]
    return [
        _check("view_consistency_fitness", "fail" if s["median_fitness"] < th.consistency_min_fitness else "pass",
               s["median_fitness"], th.consistency_min_fitness,
               "median fraction of view points with a fused-surface correspondence; low = views do not overlap "
               "the fused surface (corrections below would be meaningless)"),
        _check("view_consistency_translation",
               _level(s["median_translation_m"], th.consistency_translation_warn_m, th.consistency_translation_fail_m),
               s["median_translation_m"],
               {"warn": th.consistency_translation_warn_m, "fail": th.consistency_translation_fail_m},
               "median rigid correction needed to align a view to the fused surface; large = turntable centre, "
               "direction-sign or angle errors. Blind to line-of-sight calibration errors."),
        _check("view_consistency_tilt",
               _level(s["median_tilt_deg"], th.consistency_tilt_warn_deg, th.consistency_tilt_fail_deg),
               s["median_tilt_deg"], {"warn": th.consistency_tilt_warn_deg, "fail": th.consistency_tilt_fail_deg},
               "median out-of-axis rotation correction per view; large = rotation-axis tilt error. "
               "Under-reads tilt toward/away from the camera."),
    ]


def compute_quality(session, frame_stats: list[dict], mesh, run_id: str,
                    thresholds: QualityThresholds | None = None, consistency: dict | None = None) -> dict:
    th = thresholds or QualityThresholds()
    n_valid_frames = sum(1 for s in frame_stats if s["pixels_valid"] >= th.min_valid_pixels_per_frame)
    gap = max_angular_gap_deg([f.angle_deg for f in session.frames])
    total_px = sum(s["pixels_total"] for s in frame_stats)
    valid_px = sum(s["pixels_valid"] for s in frame_stats)
    n_tri = len(mesh.triangles)
    comps, largest_frac = 0, 0.0
    if n_tri:
        _, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        comps, largest_frac = int(counts.size), float(counts.max() / counts.sum())

    metrics = {
        "valid_frames": n_valid_frames, "total_frames": len(frame_stats),
        "max_angular_gap_deg": gap,
        "depth_pixels_before_filtering": int(total_px), "depth_pixels_after_filtering": int(valid_px),
        "valid_pixel_fraction": float(valid_px / total_px) if total_px else 0.0,
        "mesh_triangles": n_tri, "mesh_components": comps, "largest_component_fraction": largest_frac,
        "mesh_is_watertight": bool(mesh.is_watertight()) if n_tri else False,
        "outlier_fraction": None,
        "registration_residual_m": (consistency["summary"]["median_inlier_rmse_m"]
                                    if consistency and consistency.get("summary") else None),
        "unobserved_region_fraction": None,
        "view_consistency": consistency,
    }
    checks = [
        _check("valid_frames", "pass" if n_valid_frames >= th.min_valid_frames else "fail",
               n_valid_frames, th.min_valid_frames, "frames with enough valid depth"),
        _check("angular_gap",
               "fail" if gap > th.max_gap_fail_deg else "warn" if gap > th.max_gap_warn_deg else "pass",
               gap, {"warn": th.max_gap_warn_deg, "fail": th.max_gap_fail_deg},
               "largest uncovered turntable arc between recorded angles"),
        _check("valid_pixel_fraction",
               "warn" if metrics["valid_pixel_fraction"] < th.min_valid_pixel_fraction_warn else "pass",
               metrics["valid_pixel_fraction"], th.min_valid_pixel_fraction_warn,
               "valid depth pixels / all pixels (the subject only fills part of the image)"),
        _check("mesh_nonempty", "pass" if n_tri >= th.min_mesh_triangles else "fail",
               n_tri, th.min_mesh_triangles, "reconstructed mesh has enough triangles"),
        _check("single_component", "pass" if comps <= 1 else "warn", comps, 1,
               "connected components remaining after conservative cleanup"),
        *consistency_checks(consistency, th),
    ]
    report = {
        "schema_version": OUTPUT_SCHEMA_VERSION, "session_id": session.session_id, "run_id": run_id,
        "algorithm_version": ALGORITHM_VERSION, "generated_utc": utc_now(),
        "passed": all(c["status"] != "fail" for c in checks),
        "checks": checks, "metrics": metrics, "thresholds": asdict(th),
    }
    errs = schema_errors(report, "quality_report")
    if errs:
        raise ValueError(f"quality report violates schema: {errs}")
    return report
