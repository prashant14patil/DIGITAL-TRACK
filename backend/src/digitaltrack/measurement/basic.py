"""PROVISIONAL minimal measurements used to check reconstruction against known geometry.

This is not the Phase 3 measurement engine: no landmarks, no ground plane, no limb logic.
It slices a mesh with a horizontal plane (body frame, Z up) and reports the closed-contour
perimeter. Definitions are versioned so results are traceable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .. import OUTPUT_SCHEMA_VERSION

ALGORITHM_VERSION = "basic-slice-0.1"
DEFINITION_VERSION = "provisional-0"


@dataclass(frozen=True)
class SliceResult:
    z: float
    perimeter_m: float | None
    area_m2: float | None
    n_closed_contours: int
    n_open_polylines: int

    @property
    def status(self) -> str:
        if self.perimeter_m is None:
            return "fail"
        return "pass" if self.n_closed_contours == 1 and self.n_open_polylines == 0 else "warn"


def o3d_to_trimesh(mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=False)


def _shoelace(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def slice_at_height(tm: trimesh.Trimesh, z: float) -> SliceResult:
    """Perimeter of the largest closed contour of the horizontal section at height z.

    Largest-by-area is a provisional rule for single-torso test shapes; contour-selection
    policy for real bodies (arms, multiple contours) is a Phase 3 item.
    """
    sec = tm.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    if sec is None:
        return SliceResult(z, None, None, 0, 0)
    closed, n_open = [], 0
    for poly in sec.discrete:
        if len(poly) >= 4 and np.allclose(poly[0], poly[-1], atol=1e-9):
            closed.append(poly)
        else:
            n_open += 1
    if not closed:
        return SliceResult(z, None, None, 0, n_open)
    best = max(closed, key=lambda p: _shoelace(p[:, :2]))
    perim = float(np.linalg.norm(np.diff(best[:, :2], axis=0), axis=1).sum())
    return SliceResult(z, perim, _shoelace(best[:, :2]), len(closed), n_open)


def vertical_extent(tm: trimesh.Trimesh) -> tuple[float, float]:
    """(z_min, z_max) of mesh vertices. Not a body height: no ground plane handling."""
    z = tm.vertices[:, 2]
    return float(z.min()), float(z.max())


def measurements_document(session_id: str, run_id: str, slices: list[SliceResult], quality_passed: bool) -> dict:
    """Build a measurements.json document (schema: measurements.schema.json).

    Values are marked 'fail' when the scan itself failed quality checks, so a poor scan never
    yields a confident-looking number.
    """
    items = []
    for s in slices:
        if s.perimeter_m is None:
            source, status = "unavailable", "unavailable"
        else:
            source, status = "measured", (s.status if quality_passed else "fail")
        items.append({
            "name": f"circumference_at_z_{s.z:.3f}m", "value": s.perimeter_m, "unit": "m",
            "source": source, "quality_status": status,
            "definition_version": DEFINITION_VERSION, "algorithm_version": ALGORITHM_VERSION,
            "notes": "provisional horizontal-slice perimeter; not an anatomical measurement",
            "details": {"n_closed_contours": s.n_closed_contours, "n_open_polylines": s.n_open_polylines},
        })
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION, "session_id": session_id, "run_id": run_id,
        "definition_version": DEFINITION_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "measurements": items,
        "profile": {"heights_m": [s.z for s in slices], "circumference_m": [s.perimeter_m for s in slices]},
    }


def circumference_profile(tm: trimesh.Trimesh, heights: np.ndarray) -> list[SliceResult]:
    return [slice_at_height(tm, float(z)) for z in heights]
