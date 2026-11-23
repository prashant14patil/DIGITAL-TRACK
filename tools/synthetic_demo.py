"""End-to-end SYNTHETIC demonstration: render -> validate -> TSDF -> measure -> compare to truth.

Usage (from repo root, venv active):
    python tools/synthetic_demo.py

Writes sessions to data/synthetic/demo/ (git-ignored) and results to results.json there.
Everything is synthetic. The errors printed are errors against ANALYTIC ground truth on a
virtual sensor. They say nothing about real-capture accuracy on real bodies.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import time
from dataclasses import replace
from pathlib import Path

from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.validate import validate_session
from digitaltrack.measurement.basic import circumference_profile, measurements_document, o3d_to_trimesh
from digitaltrack.dataset.schemas import schema_errors
from digitaltrack.reconstruction.tsdf import TSDFParams, reconstruct_tsdf
from digitaltrack.synthetic.evaluate import evaluate_against_truth
from digitaltrack.synthetic.renderer import RenderConfig, render_session
from digitaltrack.synthetic.shapes import cylinder, elliptic_cylinder, procedural_torso

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic" / "demo"


def _rmtree(path: Path) -> None:
    def onerror(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        shutil.rmtree(path, onerror=onerror)


SCENARIOS = [
    ("cylinder_clean", cylinder(0.15, 0.6), RenderConfig(), TSDFParams()),
    ("ellipse_clean", elliptic_cylinder(0.18, 0.11, 0.6), RenderConfig(), TSDFParams()),
    ("torso_clean", procedural_torso(), RenderConfig(), TSDFParams()),
    ("torso_noise3mm_dropout5pct", procedural_torso(),
     RenderConfig(noise_sigma_m=0.003, dropout_prob=0.05, seed=1), TSDFParams()),
    ("torso_angle_noise_1deg", procedural_torso(), RenderConfig(angle_noise_deg=1.0, seed=2), TSDFParams()),
    ("torso_angle_noise_3deg", procedural_torso(), RenderConfig(angle_noise_deg=3.0, seed=2), TSDFParams()),
    ("torso_calib_center_err_1cm", procedural_torso(),
     RenderConfig(calibration_center_offset_m=(0.01, 0.0, 0.0)), TSDFParams()),
    ("torso_calib_axis_tilt_1deg", procedural_torso(), RenderConfig(calibration_axis_tilt_deg=1.0), TSDFParams()),
    ("torso_12_views_only", procedural_torso(), RenderConfig(n_views=12), TSDFParams()),
    # added with QC v0.2
    ("torso_calib_center_err_LOS_6mm", procedural_torso(),
     RenderConfig(calibration_center_offset_m=(0.0, 0.006, 0.0)), TSDFParams()),
    ("torso_calib_tilt_sideways_0.5deg", procedural_torso(),
     RenderConfig(calibration_axis_tilt_deg=0.5, calibration_axis_tilt_azimuth_deg=90.0), TSDFParams()),
    ("torso_rotation_sign_flipped", procedural_torso(), RenderConfig(), TSDFParams()),
]
SIGN_FLIPPED = {"torso_rotation_sign_flipped"}  # session written correctly; backend told the wrong sign


def main() -> int:
    _rmtree(OUT)
    OUT.mkdir(parents=True)
    results = []
    print(f"{'scenario':32s} {'circ bias':>10s} {'circ MAE':>9s} {'max |err|':>10s} {'height err':>11s}  QC   slices")
    for name, body, cfg, params in SCENARIOS:
        t0 = time.time()
        root = render_session(body, OUT / name, cfg, session_id=name)
        rep = validate_session(root)
        assert rep.ok, [str(i) for i in rep.issues]
        session = load_session(root)
        if name in SIGN_FLIPPED:
            session = replace(session, turntable={**session.turntable, "angle_direction_sign": -1})
        res = reconstruct_tsdf(session, params)
        ev = evaluate_against_truth(res.mesh, body)
        slices = circumference_profile(o3d_to_trimesh(res.mesh), [r["z_m"] for r in ev["rows"]])
        doc = measurements_document(name, res.run_id, slices, res.quality["passed"])
        assert not schema_errors(doc, "measurements")
        (res.run_dir / "measurements.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
        ok = ev["n_slices_failed"] == 0
        mm = lambda x: "   n/a" if x is None else f"{x * 1000:+7.2f} mm"
        print(f"{name:32s} {mm(ev['circumference_bias_m']):>10s} {mm(ev['circumference_mae_m']):>9s} "
              f"{mm(ev['circumference_max_abs_error_m']):>10s} {mm(ev['height_error_m']):>11s}  "
              f"{'pass' if res.quality['passed'] else 'FAIL'} {len(ev['rows']) - ev['n_slices_failed']}/{len(ev['rows'])}"
              f"  ({time.time() - t0:.1f}s)")
        flagged = [f"{c['name']}={c['status']}" for c in res.quality["checks"] if c["status"] != "pass"]
        s = res.quality["metrics"]["view_consistency"]["summary"] or {}
        print(f"{'':32s} consistency: translation {s.get('median_translation_m', float('nan')) * 1000:.2f} mm, "
              f"tilt {s.get('median_tilt_deg', float('nan')):.2f} deg; flagged: {', '.join(flagged) or 'none'}")
        results.append({"scenario": name, "render_config": cfg.__dict__ | {"calibration_center_offset_m": list(cfg.calibration_center_offset_m)},
                        "tsdf_params": params.__dict__, "quality_passed": res.quality["passed"],
                        "quality_checks": res.quality["checks"], "sign_flipped": name in SIGN_FLIPPED,
                        "evaluation": ev, "run_dir": str(res.run_dir.relative_to(ROOT))})
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults: {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
