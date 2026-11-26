"""Validation study: many scans + reference (tape) values -> per-scan comparison -> accuracy / repeatability report.

Manifest (JSON): {"title": ..., "conditions": "clothing, distance, protocol, calibration ...",
                  "scans": [{"subject": "P01", "scan": "P01-s1", "run_dir": ".../tracked_...",
                             "measurements": [{"name": "waist", "readings_mm": [812, 815], "level_mm": 1040}, ...]}]}
`level_mm` is optional (height of the tape above the floor; without it the landmark-based definition is used).
Writes <out>/validation_rows.json, validation_summary.json and validation_report.md. Read-only on every run.
"""

from __future__ import annotations

import json
from pathlib import Path

from .compare import compare_run
from .stats import markdown, summarise


def run_study(manifest: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []
    for s in manifest["scans"]:
        res = compare_run(s["run_dir"], {"measurements": s["measurements"], "contains_person": True}, write=False)
        if res.get("error"):
            excluded.append({"scan": s["scan"], "reason": res["error"]})
        for r in res["rows"]:
            rows.append({"subject": s["subject"], "scan": s["scan"], "measurement": r["name"].lower(),
                         "mesh_mm": r.get("mesh_mm"), "ref_mm": r["tape"]["mean_mm"], "status": r["status"],
                         "why": r["why"], "method": r.get("method"), "run_status": res.get("run_status")})
    summary = summarise(rows)
    summary["excluded_scans"] = excluded
    summary["n_scans"] = len(manifest["scans"])
    (out / "validation_rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = markdown(summary, title=manifest.get("title", "Validation report"),
                  conditions=manifest.get("conditions", "conditions not recorded"))
    if excluded:
        md += "\nExcluded scans:\n" + "".join(f"- {e['scan']}: {e['reason']}\n" for e in excluded)
    (out / "validation_report.md").write_text(md, encoding="utf-8")
    return summary
