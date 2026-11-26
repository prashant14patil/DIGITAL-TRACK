"""Accuracy and repeatability statistics over many scans (validation Stages 3-4, CLAUDE.md).

Input rows: {"subject", "scan", "measurement", "mesh_mm", "ref_mm", "status"}; rows without both values are
counted as unavailable and excluded from the statistics. Per measurement:

  accuracy      n, MAE, bias (mean of mesh - reference), SD of the differences, mean absolute % error,
                Bland-Altman 95 % limits of agreement (bias +- 1.96 SD; with repeat scans per subject the limits are
                computed on all scans and are somewhat narrow, as stated in the output)
  repeatability within-subject SD Sw from subjects with >= 2 scans, CV % (Sw / mean), repeatability coefficient
                and smallest detectable change SDC95 = 1.96 * sqrt(2) * Sw (= 2.77 Sw), one-way ICC(1,1) with the
                unbalanced-design n0
  target        share of scans within the PROVISIONAL target (20 mm or 4 % for circumferences): an evaluation
                target, never an accuracy claim

No correlation coefficient is reported on its own (CLAUDE.md). Nothing here changes any measurement.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

STATS_VERSION = "validation-stats-0.1"


def _icc_oneway(groups: list[np.ndarray]) -> float | None:
    groups = [g for g in groups if len(g) >= 1]
    k = len(groups)
    n_i = np.array([len(g) for g in groups], float)
    N = n_i.sum()
    if k < 2 or N - k < 1:
        return None
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else None


def summarise(rows: list[dict], *, target_mm: float = 20.0, target_pct: float = 4.0) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["measurement"]].append(r)
    out = {"stats_version": STATS_VERSION, "measurements": {}, "notes": [
        "Bland-Altman limits use every scan; repeat scans of one subject are not independent, so the limits are "
        "somewhat narrow.",
        f"Target ({target_mm:.0f} mm or {target_pct:.0f} %) is a provisional evaluation target, not a claim."]}
    for name, rs in sorted(by.items()):
        ok = [r for r in rs if r.get("mesh_mm") is not None and r.get("ref_mm") is not None]
        s: dict = {"n_scans": len(rs), "n_compared": len(ok), "n_unavailable": len(rs) - len(ok),
                   "subjects": len({r["subject"] for r in ok}),
                   "statuses": {st: sum(r.get("status") == st for r in rs) for st in sorted({r.get("status") for r in rs})}}
        if ok:
            m = np.array([r["mesh_mm"] for r in ok], float)
            ref = np.array([r["ref_mm"] for r in ok], float)
            d = m - ref
            sd = float(d.std(ddof=1)) if len(d) > 1 else None
            s.update({"mae_mm": float(np.abs(d).mean()), "bias_mm": float(d.mean()), "sd_diff_mm": sd,
                      "mape_pct": float(np.mean(np.abs(d) / ref) * 100),
                      "loa_mm": [float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)] if sd is not None else None,
                      "within_target": float(np.mean((np.abs(d) <= target_mm) | (np.abs(d) / ref * 100 <= target_pct)))})
            groups = defaultdict(list)
            for r in ok:
                groups[r["subject"]].append(r["mesh_mm"])
            rep = [np.array(v, float) for v in groups.values() if len(v) >= 2]
            if rep:
                sw = float(np.sqrt(np.mean([g.var(ddof=1) for g in rep])))
                s.update({"repeat_subjects": len(rep), "within_subject_sd_mm": sw,
                          "cv_pct": float(100 * sw / np.mean(np.concatenate(rep))),
                          "sdc95_mm": float(1.96 * np.sqrt(2) * sw),
                          "icc_1_1": _icc_oneway([np.array(v, float) for v in groups.values()])})
        out["measurements"][name] = s
    return out


def markdown(summary: dict, *, title: str, conditions: str) -> str:
    f = lambda v, d=1: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = [f"# {title}", "", conditions, "",
             "| measurement | scans (subjects) | MAE mm | bias mm | LoA mm | MAPE % | within target | Sw mm | SDC95 mm | ICC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary["measurements"].items():
        loa = s.get("loa_mm")
        loa_txt = "–" if not loa else f"{loa[0]:.1f} to {loa[1]:.1f}"
        wt = s.get("within_target")
        wt_txt = "–" if wt is None else f"{100 * wt:.0f} %"
        lines.append(f"| {name} | {s['n_compared']} ({s['subjects']}) | {f(s.get('mae_mm'))} | {f(s.get('bias_mm'))} | "
                     f"{loa_txt} | {f(s.get('mape_pct'), 2)} | {wt_txt} | "
                     f"{f(s.get('within_subject_sd_mm'))} | {f(s.get('sdc95_mm'))} | {f(s.get('icc_1_1'), 3)} |")
    lines += ["", *[f"- {n}" for n in summary["notes"]], ""]
    return "\n".join(lines)
