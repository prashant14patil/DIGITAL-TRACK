"""SYNTHETIC validation study: exercises measurement/body.py + validation/compare.py + validation/stats.py end to end.

    python tools/synthetic_validation_study.py --out data/synthetic/validation_study

Creates synthetic subjects of different sizes, three simulated repeat scans each (surface noise, scale error, arm
position, random turn), writes run folders exactly like a real tracked run (fused_mesh.ply, run_log.json,
body_measurements.json), a manifest with the TRUE dimensions as the "tape", and the validation report.
The result tells how the measurement definitions behave on clean known geometry. It says NOTHING about the accuracy
of real iPad scans (no capture, no reconstruction, no clothing); the report title says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from digitaltrack.measurement.body import measure_body  # noqa: E402
from digitaltrack.synthetic.body_figure import Figure, scan_copy  # noqa: E402
from digitaltrack.validation.study import run_study  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 1
    rng = np.random.default_rng(a.seed)
    scans = []
    for i in range(a.subjects):
        k = rng.uniform(0.85, 1.2)  # body size factor
        f = Figure(hip=(0.17 * k, 0.12 * k), waist=(0.145 * k ** 1.3, 0.10 * k ** 1.3), chest=(0.165 * k, 0.115 * k),
                   leg_r=0.07 * k, leg_y=max(0.10 * k, 0.07 * k + 0.03))
        truth = f.truth_mm()
        for r in range(a.repeats):
            mesh, pert = scan_copy(f, rng)
            run = out / "runs" / f"S{i + 1:02d}_r{r + 1}"
            run.mkdir(parents=True)
            o3d.io.write_triangle_mesh(str(run / "fused_mesh.ply"), mesh)
            (run / "run_log.json").write_text(json.dumps({"run_id": run.name, "status": "pass",
                                                          "synthetic": True, "perturbation": pert}), encoding="utf-8")
            (run / "body_measurements.json").write_text(json.dumps(measure_body(mesh), default=float), encoding="utf-8")
            meas = [{"name": n, "readings_mm": [truth[n]]} for n in ("height", "chest", "waist", "hip", "thigh")]
            meas += [{"name": f"{n} at tape level", "readings_mm": [truth[n]], "level_mm": lv}
                     for n, lv in f.levels_mm().items()]
            scans.append({"subject": f"S{i + 1:02d}", "scan": run.name, "run_dir": str(run), "measurements": meas})
    manifest = {"title": "SYNTHETIC validation study (measurement engine only, NOT iPad accuracy)",
                "conditions": f"{a.subjects} synthetic subjects x {a.repeats} simulated scans: 2 mm surface noise, "
                              "0.3 % scale SD, arm gap 10 +- 2 cm, random turn. Reference = exact geometry.",
                "scans": scans}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run_study(manifest, out)
    print((out / "validation_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
