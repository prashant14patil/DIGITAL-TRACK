"""validation.stats / validation.study on SYNTHETIC data with known answers."""

from __future__ import annotations

import json

import numpy as np
import open3d as o3d
import pytest

from digitaltrack.measurement.body import measure_body
from digitaltrack.synthetic.body_figure import Figure, scan_copy
from digitaltrack.validation.stats import _icc_oneway, summarise
from digitaltrack.validation.study import run_study


def test_stats_known_values():
    rows = [{"subject": s, "scan": f"{s}{k}", "measurement": "waist", "mesh_mm": ref + e, "ref_mm": ref, "status": "x"}
            for s, ref, errs in (("A", 800.0, (10, 12)), ("B", 900.0, (8, 10))) for k, e in enumerate(errs)]
    rows.append({"subject": "C", "scan": "C0", "measurement": "waist", "mesh_mm": None, "ref_mm": 850.0, "status": "unavailable"})
    s = summarise(rows)["measurements"]["waist"]
    assert s["n_compared"] == 4 and s["n_unavailable"] == 1
    assert s["bias_mm"] == pytest.approx(10.0) and s["mae_mm"] == pytest.approx(10.0)
    assert s["within_subject_sd_mm"] == pytest.approx(np.sqrt(2.0))  # each subject: two values 2 mm apart -> var 2
    assert s["sdc95_mm"] == pytest.approx(1.96 * np.sqrt(2) * np.sqrt(2.0))
    assert s["within_target"] == 1.0


def test_icc_perfect_and_none():
    assert _icc_oneway([np.array([1.0, 1.0]), np.array([5.0, 5.0])]) == pytest.approx(1.0)
    assert _icc_oneway([np.array([1.0])]) is None


def test_study_end_to_end_on_synthetic_figures(tmp_path):
    rng = np.random.default_rng(3)
    scans = []
    for i, k in enumerate((0.9, 1.1)):
        f = Figure(hip=(0.17 * k, 0.12 * k), waist=(0.145 * k, 0.10 * k), chest=(0.165 * k, 0.115 * k))
        truth = f.truth_mm()
        for r in range(2):
            mesh, _ = scan_copy(f, rng, noise_m=0.001, scale_sd=0.0)
            run = tmp_path / f"S{i}r{r}"
            run.mkdir()
            o3d.io.write_triangle_mesh(str(run / "fused_mesh.ply"), mesh)
            (run / "run_log.json").write_text(json.dumps({"status": "pass"}))
            (run / "body_measurements.json").write_text(json.dumps(measure_body(mesh), default=float))
            scans.append({"subject": f"S{i}", "scan": run.name, "run_dir": str(run), "measurements": [
                {"name": "waist", "readings_mm": [truth["waist"]]},
                {"name": "chest", "readings_mm": [truth["chest"]], "level_mm": f.levels_mm()["chest"]}]})
    out = run_study({"title": "t", "conditions": "synthetic", "scans": scans}, tmp_path / "report")
    for name in ("waist", "chest"):
        s = out["measurements"][name]
        assert s["n_compared"] == 4 and abs(s["bias_mm"]) < 5 and s["mae_mm"] < 5, (name, s)
    assert (tmp_path / "report" / "validation_report.md").read_text(encoding="utf-8").startswith("# t")
    rows = json.loads((tmp_path / "report" / "validation_rows.json").read_text())
    assert all(r["status"] == "unreliable" for r in rows)  # unvalidated definitions never report "measured"
