"""EXPERIMENTAL: split an upright body mesh into display regions (head, trunk, arms, legs).

For the avatar view only: regions colour the mesh and anchor the callout cards. Nothing is measured from them.
Heuristic, geometry-only, for an upright person in an A-pose, mesh Z-up (feet at the bottom):
  * neck   = narrowest central horizontal section between 75 % and 93 % of the height -> above it: head
  * arms   = vertices outside the central (trunk) cluster of each horizontal section between the crotch (at least
             35 % of the height) and the neck, plus shoulder-level vertices further out than 1.15 x the trunk
             half-width. Hands hanging below the crotch are therefore labelled as legs.
  * crotch = highest level (25-55 % of height) where the central section splits into two legs; if the legs are
             merged (visual hull) a fixed 47 % of height is used and reported as an anthropometric default
  * legs   = below the crotch, split at the body centre line; trunk = the rest between crotch and neck.
Left/right: the subject faces -Y at turn angle 0, so +X is the subject's LEFT. This relies on the estimated turn
direction being correct and is reported as an assumption.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

ALGORITHM = "body-regions-heuristic-0.1 (EXPERIMENTAL)"
REGIONS = ["head", "trunk", "left_arm", "right_arm", "left_leg", "right_leg"]
ID = {n: i for i, n in enumerate(REGIONS)}
COLOUR_STEP = 40  # region id encoded in the red vertex channel as id * 40 / 255 (robust to 8-bit rounding)


def _clusters(x: np.ndarray, gap: float) -> list[tuple[float, float]]:
    """Intervals of sorted x separated by gaps > gap."""
    xs = np.sort(x)
    cuts = np.nonzero(np.diff(xs) > gap)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts, [len(xs) - 1]])
    return [(float(xs[s]), float(xs[e])) for s, e in zip(starts, ends)]


def _central(intervals, xc):
    for lo, hi in intervals:
        if lo <= xc <= hi:
            return lo, hi
    return min(intervals, key=lambda iv: min(abs(iv[0] - xc), abs(iv[1] - xc)))


def segment_regions(vertices: np.ndarray, *, slice_frac: float = 0.01, gap_frac: float = 0.02) -> tuple[np.ndarray, dict]:
    v = np.asarray(vertices, float)
    z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
    h = z1 - z0
    if h <= 0 or len(v) < 100:
        raise ValueError("mesh too small or flat to segment")
    zn = (v[:, 2] - z0) / h
    gap = gap_frac * h
    legband = (zn > 0.1) & (zn < 0.3)
    xc = float(np.median(v[legband, 0])) if legband.any() else float(np.median(v[:, 0]))

    bins = np.arange(0.0, 1.0 + slice_frac, slice_frac)
    central = {}
    for b0 in bins[:-1]:
        sel = (zn >= b0) & (zn < b0 + slice_frac)
        if sel.sum() >= 5:
            central[round(b0, 4)] = _central(_clusters(v[sel, 0], gap), xc)

    def width_at(b):
        lo, hi = central[b]
        return hi - lo

    neck_cands = [b for b in central if 0.75 <= b <= 0.93]
    neck = min(neck_cands, key=width_at) + slice_frac / 2 if neck_cands else 0.86

    crotch, crotch_method = None, "detected leg gap"
    for b in sorted((b for b in central if 0.25 <= b <= 0.55), reverse=True):
        sel = (zn >= b) & (zn < b + slice_frac)
        xs = v[sel, 0]
        # legs separated at this level: >= 2 clusters and no surface on the centre line
        if len(_clusters(xs, gap)) >= 2 and not (np.abs(xs - xc) < 0.01 * h).any():
            crotch = b
            break
    if crotch is None:
        crotch, crotch_method = 0.47, "anthropometric default (legs not separated in this mesh)"

    ref = [central[b][1] - central[b][0] for b in central if 0.55 <= b <= 0.68]
    torso_half = 0.5 * float(np.median(ref)) if ref else 0.12 * h

    labels = np.full(len(v), ID["trunk"], np.uint8)
    labels[zn > neck] = ID["head"]
    arm_zone = (zn >= max(0.35, float(crotch))) & (zn <= neck)  # below the crotch the legs form two clusters
    for i in np.nonzero(arm_zone)[0]:
        b = round(np.floor(zn[i] / slice_frac) * slice_frac, 4)
        lo, hi = central.get(b, (xc - torso_half, xc + torso_half))
        outside = v[i, 0] < lo - 1e-9 or v[i, 0] > hi + 1e-9
        shoulder = zn[i] > 0.68 and abs(v[i, 0] - xc) > 1.15 * torso_half
        if outside or shoulder:
            labels[i] = ID["left_arm"] if v[i, 0] > xc else ID["right_arm"]
    legs = (zn < crotch) & ~np.isin(labels, [ID["left_arm"], ID["right_arm"]])
    labels[legs & (v[:, 0] > xc)] = ID["left_leg"]
    labels[legs & (v[:, 0] <= xc)] = ID["right_leg"]

    anchors = {}
    for name, rid in ID.items():
        pts = v[labels == rid]
        if not len(pts):
            continue
        if name.endswith("_arm"):
            pts = pts[pts[:, 2] >= np.median(pts[:, 2])]  # upper arm
        elif name.endswith("_leg"):
            pts = pts[pts[:, 2] >= np.quantile(pts[:, 2], 0.6)]  # thigh
        anchors[name] = [float(c) for c in pts.mean(axis=0)]
    info = {
        "algorithm": ALGORITHM, "status": "provisional",
        "levels_height_fraction": {"neck": round(neck, 3), "crotch": round(float(crotch), 3)},
        "crotch_method": crotch_method, "centre_x": xc,
        "left_right": "+X is the subject's left (subject faces -Y at turn angle 0); relies on the estimated turn "
                      "direction",
        "regions": [{"id": rid, "name": n, "vertex_count": int((labels == rid).sum()), "anchor": anchors.get(n)}
                    for n, rid in ID.items()],
        "colour_encoding": f"vertex red channel = id * {COLOUR_STEP} / 255",
        "note": "display regions only; no measurement is derived from them",
    }
    return labels, info


def write_region_files(run_dir: str | Path, mesh_name: str = "mesh_height_normalised.ply") -> dict:
    """Write mesh_regions.glb (region id in vertex colour) and regions.json next to the run's mesh."""
    import trimesh

    run = Path(run_dir)
    mesh = o3d.io.read_triangle_mesh(str(run / mesh_name))
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    labels, info = segment_regions(verts)
    rgba = np.zeros((len(labels), 4), np.uint8)
    rgba[:, 0] = labels * COLOUR_STEP
    rgba[:, 3] = 255
    # trimesh, not Open3D: Open3D 0.20 writes a corrupt .glb when vertex colours are present
    tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)
    (run / "mesh_regions.glb").write_bytes(tm.export(file_type="glb"))
    (run / "regions.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info
