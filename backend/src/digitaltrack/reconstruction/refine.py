"""Refined (closed, smooth) body mesh + automatic acceptance gate, built FROM the measuring mesh.

The measuring mesh (fused, topology-repaired, never smoothed or filled) stays the source of every reported number.
The refined mesh is what is shown. It is accepted as MEASUREMENT-ELIGIBLE only if, measured with the same
definitions, it reproduces the measuring mesh's values; otherwise it is flagged and the page says so.

Steps: screened-Poisson closed surface fitted to the measuring mesh (experimental.display_mesh.build_display_surface:
fill farther than 3 cm from data dropped, gap-filled vertices flagged `estimated`, fidelity limits on the measured
part) -> topology repair (reconstruction.mesh_qc) -> gate:
  * surface: one component, no non-manifold geometry, no self-intersections, deviation RMS <= 2 mm and P95 <= 4 mm on
    the measured part, estimated share <= 15 %, measured surface covered >= 95 %
  * measurements: every value available on the measuring mesh is also available on the refined mesh and agrees within
    max(abs_tol, rel_tol x value) (height 5 mm; circumferences 5 mm or 0.5 %); landmark differences are reported
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d

from ..experimental.display_mesh import SurfaceDisplayParams, _distances, build_display_surface
from .mesh_qc import RepairParams, inspect_mesh, repair_mesh


@dataclass(frozen=True)
class RefineParams:
    # depth 10 / 1 cm: on the first real scan depth 9 / 3 cm domed the unseen head top (+24 mm height) and bridged the
    # inner thighs (crotch lost); depth 10 with fill limited to 1 cm from data reproduced every measurement
    surface: SurfaceDisplayParams = SurfaceDisplayParams(poisson_depth=10, drop_beyond_m=0.01, samples=400_000,
                                                         target_triangles=300_000)
    abs_tol_m: float = 0.005
    rel_tol: float = 0.005
    landmark_tol_m: float = 0.02
    # limbs: small outlines are noise-sensitive on the MEASURING mesh itself (synthetic upper arm, 1 mm white noise:
    # +2.3 % on the measuring mesh, refined within 0.2 % of truth), so 0.5 % would flag the measuring mesh's own noise
    limb_rel_tol: float = 0.025


def _agreement(meas: dict, ref: dict, p: RefineParams) -> tuple[list[dict], bool]:
    rows, ok = [], True
    rv = {x["name"]: x for x in ref["measurements"]}
    for x in meas["measurements"]:
        y = rv.get(x["name"])
        a, b = x.get("value"), (y or {}).get("value")
        if a is None:
            rows.append({"name": x["name"], "measuring_m": None, "refined_m": b, "status": "not compared"})
            continue
        if b is None:
            rows.append({"name": x["name"], "measuring_m": a, "refined_m": None, "status": "fail",
                         "why": "available on the measuring mesh but not on the refined mesh"})
            ok = False
            continue
        limb = any(k in x["name"] for k in ("arm", "thigh", "calf"))
        tol = max(p.abs_tol_m, (p.limb_rel_tol if limb else p.rel_tol) * abs(a))
        good = abs(b - a) <= tol
        ok &= good
        rows.append({"name": x["name"], "measuring_m": a, "refined_m": b, "difference_m": b - a, "tolerance_m": tol,
                     "status": "pass" if good else "fail"})
    # landmarks are intermediate steps: reported, but eligibility rests on the measurement VALUES (height is itself
    # a value above). The crotch level in particular is ill-defined when the thighs merge gradually.
    for k, v in meas.get("landmarks", {}).items():
        w = ref.get("landmarks", {}).get(k)
        if v is None:
            continue
        good = w is not None and abs(w - v) <= p.landmark_tol_m
        rows.append({"name": f"landmark {k}", "measuring_m": v, "refined_m": w,
                     "difference_m": None if w is None else w - v, "tolerance_m": p.landmark_tol_m,
                     "status": "info: within tolerance" if good else "info: differs (not used for eligibility)"})
    return rows, ok


def refine(measuring: o3d.geometry.TriangleMesh, p: RefineParams | None = None, *, measure=None
           ) -> tuple[o3d.geometry.TriangleMesh | None, np.ndarray | None, dict]:
    """Return (refined mesh or None, estimated-vertex mask, report). `measure(mesh) -> dict` defaults to
    measurement.body.measure_body."""
    if measure is None:
        from ..measurement.body import measure_body as measure
    p = p or RefineParams()
    surf, est, srep = build_display_surface(measuring, p.surface)
    report: dict = {"algorithm": "refined-mesh-0.1", "params": asdict(p), "surface_fit": srep}
    if surf is None:
        report.update(status="rejected", reasons=[srep.get("reason", "surface fit failed its fidelity limits")])
        return None, None, report
    refined, rep = repair_mesh(surf, RepairParams(min_component_fraction=0.01))
    report["repair"] = {"removed": rep["removed"], "before": rep["before"]}
    d = _distances(measuring, np.asarray(refined.vertices))
    est = d > p.surface.measured_within_m
    meas_d = d[~est]
    qa = inspect_mesh(refined)
    report["mesh"] = qa
    m_meas, m_ref = measure(measuring), measure(refined)
    rows, agree = _agreement(m_meas, m_ref, p)
    checks = [
        ("one_component", qa["components"] == 1, qa["components"]),
        ("no_non_manifold", qa["non_manifold_edges"] == 0 and qa["non_manifold_vertices"] == 0,
         qa["non_manifold_edges"] + qa["non_manifold_vertices"]),
        ("no_self_intersections", qa.get("self_intersecting_pairs", 0) == 0, qa.get("self_intersecting_pairs")),
        ("deviation_rms_m", float(np.sqrt(np.mean(meas_d ** 2))) <= p.surface.max_rms_deviation_m,
         float(np.sqrt(np.mean(meas_d ** 2)))),
        ("deviation_p95_m", float(np.percentile(meas_d, 95)) <= p.surface.max_p95_deviation_m,
         float(np.percentile(meas_d, 95))),
        ("estimated_share", float(est.mean()) <= p.surface.max_estimated_fraction, float(est.mean())),
        ("measured_surface_covered", srep["source_covered_within_5mm"] >= p.surface.min_source_covered,
         srep["source_covered_within_5mm"]),
        ("measurements_agree", agree, sum(r["status"] == "fail" for r in rows)),
    ]
    report["checks"] = [{"name": n, "status": "pass" if ok else "fail", "value": v} for n, ok, v in checks]
    report["measurement_agreement"] = rows
    report["estimated_vertex_fraction"] = float(est.mean())
    failed = [c["name"] for c in report["checks"] if c["status"] == "fail"]
    report["status"] = "accepted" if not failed else "flagged"
    report["reasons"] = failed
    report["eligibility"] = ("measurement-eligible: reproduces the measuring mesh's values within tolerance; reported "
                             "numbers still come from the measuring mesh") if not failed else \
        "NOT measurement-eligible: " + ", ".join(failed)
    return refined, est, report
