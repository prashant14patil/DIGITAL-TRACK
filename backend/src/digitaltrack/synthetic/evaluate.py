"""Compare a reconstructed mesh against the analytic synthetic truth.

Synthetic-only: it needs the ground-truth body, which never exists for real scans.
"""

from __future__ import annotations

import numpy as np

from ..measurement.basic import circumference_profile, o3d_to_trimesh, vertical_extent
from .shapes import ProfileBody


def evaluate_against_truth(mesh, body: ProfileBody, fractions=(0.15, 0.3, 0.45, 0.6, 0.75, 0.9)) -> dict:
    """Per-height circumference error and vertical-extent error (all in metres / percent)."""
    tm = o3d_to_trimesh(mesh)
    heights = body.z_min + np.asarray(fractions) * body.height
    slices = circumference_profile(tm, heights)
    rows = []
    for z, s in zip(heights, slices):
        truth = float(body.circumference(z))
        meas = s.perimeter_m
        err = None if meas is None else meas - truth
        rows.append({"z_m": float(z), "true_circumference_m": truth, "measured_circumference_m": meas,
                     "error_m": err, "error_pct": None if err is None else 100.0 * err / truth,
                     "slice_status": s.status})
    errs = np.array([r["error_m"] for r in rows if r["error_m"] is not None])
    zmin, zmax = vertical_extent(tm)
    return {
        "rows": rows,
        "n_slices_failed": sum(r["error_m"] is None for r in rows),
        "circumference_mae_m": float(np.abs(errs).mean()) if errs.size else None,
        "circumference_bias_m": float(errs.mean()) if errs.size else None,
        "circumference_max_abs_error_m": float(np.abs(errs).max()) if errs.size else None,
        "circumference_max_abs_error_pct": max((abs(r["error_pct"]) for r in rows if r["error_pct"] is not None),
                                               default=None),
        "true_height_m": body.height, "mesh_vertical_extent_m": zmax - zmin,
        "height_error_m": (zmax - zmin) - body.height,
        "mesh_z_min_m": zmin, "mesh_z_max_m": zmax,
    }
