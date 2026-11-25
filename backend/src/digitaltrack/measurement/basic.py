"""PROVISIONAL minimal measurements used to check reconstruction against known geometry.

This is not the Phase 3 measurement engine: no landmarks, no ground plane, no limb logic.
It slices a mesh with a horizontal plane (body frame, Z up) and reports the closed-contour
perimeter. Definitions are versioned so results are traceable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh.intersections import mesh_plane

from .. import OUTPUT_SCHEMA_VERSION

ALGORITHM_VERSION = "basic-slice-0.2"
DEFINITION_VERSION = "provisional-1"


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


def _section_chains(tm: trimesh.Trimesh, z: float) -> list[tuple[np.ndarray, bool]]:
    """Connected outlines of the horizontal section as (points, is_closed). Closed points are in walking order.

    Built from raw plane/triangle segments because trimesh's Path discretisation silently drops outlines that
    have a gap, which let a small closed fragment be reported as the section."""
    seg = mesh_plane(tm, [0.0, 0.0, 1.0], [0.0, 0.0, z])
    if not len(seg):
        return []
    pts, idx = np.unique(np.round(seg.reshape(-1, 3), 8), axis=0, return_inverse=True)
    edges = np.sort(idx.reshape(-1, 2), axis=1)
    edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)  # an edge lying in the plane appears twice
    if not len(edges):
        return []
    adj: dict[int, list[int]] = {}
    for i, j in edges.tolist():
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    seen: set[int] = set()
    chains = []
    for start in adj:
        if start in seen:
            continue
        order, stack = [], [start]  # depth-first: walks a simple loop in order
        seen.add(start)
        while stack:
            node = stack.pop()
            order.append(node)
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        closed = len(order) >= 3 and all(len(adj[n]) == 2 for n in order)
        if closed:
            order, prev, cur = [start], None, start
            while True:
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                if nxt == start:
                    break
                order.append(nxt)
                prev, cur = cur, nxt
            chains.append((pts[order + [start]], True))
        else:
            lengths = [np.linalg.norm(pts[i] - pts[j]) for i in order for j in adj[i] if i < j]
            chains.append((np.array([[sum(lengths), 0.0, 0.0]]), False))  # open: only its total length matters
    return chains


def _perimeter(loop: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(loop[:, :2], axis=0), axis=1).sum())


def slice_at_height(tm: trimesh.Trimesh, z: float) -> SliceResult:
    """Perimeter of the largest closed contour of the horizontal section at height z.

    No value is returned when an open (broken) outline is longer than that contour: the main outline then has
    a gap and the closed contour is only a fragment or a minor part. Largest-by-area is a provisional rule for
    single-torso test shapes; contour-selection policy for real bodies is a Phase 3 item.
    """
    chains = _section_chains(tm, z)
    closed = [p for p, is_closed in chains if is_closed]
    open_lengths = [float(p[0, 0]) for p, is_closed in chains if not is_closed]
    if not closed:
        return SliceResult(z, None, None, 0, len(open_lengths))
    best = max(closed, key=lambda p: _shoelace(p[:, :2]))
    perim = _perimeter(best)
    if open_lengths and max(open_lengths) >= perim:
        return SliceResult(z, None, None, len(closed), len(open_lengths))
    return SliceResult(z, perim, _shoelace(best[:, :2]), len(closed), len(open_lengths))


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
