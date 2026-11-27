"""Body landmarks and torso / leg circumferences from a metric body mesh (PROPOSED definitions, not validated).

Input: a mesh in the body frame (metres, floor plane z = 0, +z up), e.g. rgbd.tracked's fused_mesh.ply.
Definitions: docs/measurement_definitions.md, version `body-provisional-1`.

Method: horizontal slices every `step_m`. Each slice is split into connected parts (the slice's points are
rasterised and labelled), so a slice through the waist with hanging arms has three parts (torso, two arms) and
only the TORSO part is measured: the arms are never included in a torso circumference. A part's circumference is
the perimeter of its largest closed section polygon (exact mesh section, shapely); when that polygon covers clearly
less than the part's filled area, the outline is open (a hole) and the value is `unavailable`, never guessed.

Landmarks (all slice-based, see the definitions file for exact rules): head top, neck, arm separation, crotch.
Every value carries units, definition / algorithm version, status and reasons. Status is `unreliable` for every
number until this definition set is validated against tape on the rig (CLAUDE.md), `unavailable` when the
geometry does not allow a value. No body composition is inferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import open3d as o3d
from scipy import ndimage

DEFINITION_VERSION = "body-provisional-2"  # 2: chest window 2-10 cm below arm separation
ALGORITHM_VERSION = "body-slices-0.4"  # 0.4: limbs perpendicular to their axes;  # 0.3: crotch = full-depth surface between the legs; 0.2: smoothed perimeter


@dataclass(frozen=True)
class BodyParams:
    step_m: float = 0.01
    slab_m: float = 0.003  # half-thickness of the point slab used to find a slice's parts
    cell_m: float = 0.005
    dilate_cells: int = 2  # parts closer than ~1 cm are one part
    samples: int = 600000
    min_part_fraction: float = 0.03  # parts with fewer points than this share of the slice are ignored
    min_leg_separation_m: float = 0.08
    crotch_depth_ratio: float = 0.6  # surface between the legs must be this deep relative to a leg (see crotch)
    closed_area_ratio: float = 0.6  # closed polygon area / part's filled area below this => outline open
    # perimeter of the outline after a Gaussian low-pass along the outline (a tape follows a smooth path; surface
    # noise makes the raw outline zig-zag: synthetic +19 % at 1 mm, +56 % at 2 mm noise). sigma 10 mm: within
    # +-3 mm up to 2 mm noise, -0.35 % (about -3 mm) on noise-free shapes (known shrink of a smoothed closed curve)
    contour_sigma_m: float = 0.010
    contour_step_m: float = 0.002
    level_median_slices: int = 5  # chest/waist/hip level chosen on a running median of the profile
    # floor rim (trim_floor_rim): geometry below floor_band_m outside the feet footprint is removed
    floor_band_m: float = 0.08
    footprint_ref_m: tuple[float, float] = (0.08, 0.14)
    footprint_margin_m: float = 0.03
    seed: int = 0


@dataclass
class Part:
    n: int
    centroid: np.ndarray
    extent: np.ndarray
    label: int
    pts: np.ndarray | None = None  # the part's slice points (x, y)


@dataclass
class Slice:
    z: float
    parts: list[Part] = field(default_factory=list)
    lab: np.ndarray | None = None  # raster labels
    lo: np.ndarray | None = None


# ---- floor rim --------------------------------------------------------------------------------------------------

def trim_floor_rim(mesh: o3d.geometry.TriangleMesh, p: BodyParams | None = None) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Remove geometry near the floor (z < floor_band_m) that lies outside the footprint of the feet measured just
    above it (footprint_ref_m band), dilated by footprint_margin_m. Floor-level fusion residue, not body."""
    p = p or BodyParams()
    x = np.asarray(mesh.vertices)
    o3d.utility.random.seed(p.seed)
    surf = np.asarray(mesh.sample_points_uniformly(200000).points)  # surface samples: vertex density may be coarse
    ref = surf[(surf[:, 2] >= p.footprint_ref_m[0]) & (surf[:, 2] < p.footprint_ref_m[1])][:, :2]
    if len(ref) < 20:
        return mesh, {"removed_vertices": 0, "note": "no feet found in the reference band; nothing removed"}
    lo = ref.min(0) - p.footprint_margin_m - p.cell_m
    shape = np.ceil((ref.max(0) + p.footprint_margin_m + p.cell_m - lo) / p.cell_m).astype(int) + 1
    g = np.zeros(shape, bool)
    ij = ((ref - lo) / p.cell_m).astype(int)
    g[ij[:, 0], ij[:, 1]] = True
    g = ndimage.binary_fill_holes(ndimage.binary_dilation(g, iterations=int(np.ceil(p.footprint_margin_m / p.cell_m))))
    low = x[:, 2] < p.floor_band_m
    ij = ((x[:, :2] - lo) / p.cell_m).astype(int)
    inside = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    foot = np.zeros(len(x), bool)
    foot[inside] = g[ij[inside, 0], ij[inside, 1]]
    bad = low & ~foot
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_vertices_by_mask(bad)
    out.remove_unreferenced_vertices()
    return out, {"removed_vertices": int(bad.sum()), "band_m": p.floor_band_m, "footprint_ref_m": list(p.footprint_ref_m),
                 "margin_m": p.footprint_margin_m}


# ---- slices -----------------------------------------------------------------------------------------------------

def _slice_parts(pts: np.ndarray, z: float, p: BodyParams) -> Slice:
    s = pts[np.abs(pts[:, 2] - z) < p.slab_m][:, :2]
    sl = Slice(float(z))
    if len(s) < 20:
        return sl
    lo = s.min(0) - (p.dilate_cells + 2) * p.cell_m
    ij = ((s - lo) / p.cell_m).astype(int)
    g = np.zeros(ij.max(0) + p.dilate_cells + 3, bool)
    g[ij[:, 0], ij[:, 1]] = True
    g = ndimage.binary_dilation(g, iterations=p.dilate_cells)
    lab, n = ndimage.label(g)
    l = lab[ij[:, 0], ij[:, 1]]
    for k in range(1, n + 1):
        q = s[l == k]
        if len(q) >= p.min_part_fraction * len(s):
            sl.parts.append(Part(len(q), q.mean(0), np.ptp(q, 0), k, q))
    sl.parts.sort(key=lambda a: -a.n)
    sl.lab, sl.lo = lab, lo
    return sl


def _part_area(sl: Slice, part: Part, p: BodyParams) -> float:
    return float(ndimage.binary_fill_holes(sl.lab == part.label).sum() * p.cell_m ** 2)


def smoothed_perimeter(coords, sigma_m: float, step_m: float) -> float:
    """Length of a closed outline after uniform resampling and a circular Gaussian low-pass of width sigma_m."""
    from scipy.ndimage import gaussian_filter1d

    c = np.asarray(coords, float)[:, :2]
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    cc = np.vstack([c, c[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))])
    total = s[-1]
    n = max(int(total / step_m), 16)
    t = np.linspace(0.0, total, n, endpoint=False)
    x, y = np.interp(t, s, cc[:, 0]), np.interp(t, s, cc[:, 1])
    k = sigma_m / (total / n)
    x, y = gaussian_filter1d(x, k, mode="wrap"), gaussian_filter1d(y, k, mode="wrap")
    length = float(np.sum(np.hypot(np.diff(np.r_[x, x[:1]]), np.diff(np.r_[y, y[:1]]))))
    # first-order shrink correction: the outline's fundamental (radius R ~ length / 2 pi) is attenuated by
    # exp(-sigma^2 / 2 R^2) by the low-pass (exact for a circle); without it a 7 cm thigh reads ~1 % short
    r = length / (2 * np.pi)
    return length * float(np.exp(sigma_m ** 2 / (2 * r ** 2))) if r > 2 * sigma_m else length


def _section_polygons(tm, z: float):
    import trimesh

    sec = tm.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return []
    try:
        path2d, _ = sec.to_2D(to_2D=trimesh.transformations.translation_matrix([0, 0, -z]))
    except Exception:  # noqa: BLE001 - degenerate section
        return []
    return [q for q in path2d.polygons_closed if q is not None and q.is_valid and q.area > 0]


def part_circumference(tm, sl: Slice, part: Part, p: BodyParams) -> dict:
    """Perimeter of the part's largest closed section polygon; `unavailable` if its outline is open."""
    polys = []
    for q in _section_polygons(tm, sl.z):
        rp = np.array(q.representative_point().coords[0])
        ij = ((rp - sl.lo) / p.cell_m).astype(int)
        if (ij >= 0).all() and ij[0] < sl.lab.shape[0] and ij[1] < sl.lab.shape[1]:
            # a polygon belongs to the part if its interior point or its boundary falls on the part's cells
            b = np.asarray(q.exterior.coords)
            bij = np.clip(((b - sl.lo) / p.cell_m).astype(int), 0, np.array(sl.lab.shape) - 1)
            if np.mean(sl.lab[bij[:, 0], bij[:, 1]] == part.label) > 0.5:
                polys.append(q)
    area = _part_area(sl, part, p)
    if not polys:
        return {"value_m": None, "reason": "no closed outline for this part (hole in the mesh at this height)"}
    best = max(polys, key=lambda q: q.area)
    if best.area < p.closed_area_ratio * area:
        return {"value_m": None, "reason": f"largest closed outline covers {best.area / area:.0%} of the part: the "
                                           "outline is open (hole), value not computed"}
    return {"value_m": smoothed_perimeter(best.exterior.coords, p.contour_sigma_m, p.contour_step_m),
            "raw_outline_m": float(best.exterior.length),
            "convex_hull_m": float(best.convex_hull.exterior.length),
            "area_m2": float(best.area), "closed_polygons_in_part": len(polys)}


# ---- landmarks and measurements -----------------------------------------------------------------------------------

def _value(name, value, status, reasons, **extra):
    return {"name": name, "value": value, "unit": "m", "status": status, "reasons": reasons,
            "definition_version": DEFINITION_VERSION, "algorithm_version": ALGORITHM_VERSION, **extra}


def measure_body(mesh: o3d.geometry.TriangleMesh, p: BodyParams | None = None, *, source_method: str = "mesh",
                 subject_left_xy=None,
                 validated: bool = False) -> dict:
    from .basic import o3d_to_trimesh

    p = p or BodyParams()
    o3d.utility.random.seed(p.seed)
    pts = np.asarray(mesh.sample_points_uniformly(p.samples).points)
    tm = o3d_to_trimesh(mesh)
    base = ["definitions are proposed and not yet validated against tape on this rig"] if not validated else []
    st = "unreliable" if not validated else "measured"
    out: dict = {"definition_version": DEFINITION_VERSION, "algorithm_version": ALGORITHM_VERSION,
                 "source_method": source_method, "units": "m", "frame": "body frame: floor z = 0, +z up",
                 "scale_uncertainty": "not quantified (no known-size reference)", "parameters": asdict(p),
                 "landmarks": {}, "measurements": [], "flags": []}

    # body axis and head top
    mid = pts[(pts[:, 2] > 0.3 * pts[:, 2].max()) & (pts[:, 2] < 0.7 * pts[:, 2].max())]
    axis = np.median(mid[:, :2], 0)
    near = np.linalg.norm(pts[:, :2] - axis, axis=1) < 0.25
    top = float(np.percentile(pts[near, 2], 99.95))
    out["landmarks"]["head_top_z"] = top
    out["measurements"].append(_value("height_floor_to_head_top", top, st, base + [
        "includes hair; the floor plane is z = 0 of the fusion frame"], level_z=None))

    zs = np.arange(0.05, top - 0.01, p.step_m)
    slices = [_slice_parts(pts, z, p) for z in zs]

    def torso_part(sl: Slice) -> Part | None:
        if not sl.parts:
            return None
        return min(sl.parts, key=lambda a: np.linalg.norm(a.centroid - axis))

    # lateral axis: direction of the separated side parts (arms) in the middle of the body
    side = [a.centroid - axis for sl in slices if 0.45 * top < sl.z < 0.70 * top
            for a in sl.parts if a is not torso_part(sl) and np.linalg.norm(a.centroid - axis) > 0.15]
    if len(side) >= 4:
        v = np.array(side)
        lat = np.linalg.svd(v - 0 * v.mean(0), full_matrices=False)[2][0]
    else:
        widths = [torso_part(sl).extent for sl in slices if 0.5 * top < sl.z < 0.65 * top and torso_part(sl)]
        lat = np.array([1.0, 0.0]) if not widths or np.median(widths, 0)[0] >= np.median(widths, 0)[1] else np.array([0.0, 1.0])
    out["lateral_axis_xy"] = lat.tolist()

    def width(a: Part) -> float:  # lateral extent of a part
        return float(abs(a.extent @ np.abs(lat)))

    # neck: narrowest torso-part width in the upper body
    cand = [(width(torso_part(sl)), sl.z) for sl in slices if 0.78 * top < sl.z < 0.95 * top and torso_part(sl)]
    neck_z = None
    if cand:
        w = np.array([c[0] for c in cand])
        w = np.convolve(w, np.ones(3) / 3, mode="same")
        neck_z = float(cand[int(np.argmin(w[1:-1])) + 1][1]) if len(w) > 2 else float(cand[0][1])
    out["landmarks"]["neck_z"] = neck_z

    # arm separation: highest z below the neck where separated parts exist on BOTH lateral sides of the torso
    arm_z = None
    for sl in reversed(slices):
        if neck_z is None or sl.z >= neck_z - 0.03:
            continue
        t = torso_part(sl)
        sides = {np.sign((a.centroid - axis) @ lat) for a in sl.parts if a is not t
                 and abs((a.centroid - axis) @ lat) > 0.5 * width(t)}
        if len(sides) == 2:
            arm_z = float(sl.z)
            break
    out["landmarks"]["arm_separation_z"] = arm_z
    if arm_z is None:
        out["flags"].append("arms never separate from the torso in the slices: torso circumferences would include "
                            "the arms and are not computed")

    # crotch: lowest z above which the legs are no longer two parts separated along the lateral axis (3 slices)
    def legs_apart(sl: Slice) -> tuple[Part, Part] | None:
        big = [a for a in sl.parts if np.linalg.norm(a.centroid - axis) < 0.25]
        for i, a in enumerate(big):
            for b in big[i + 1:]:
                if abs((a.centroid - b.centroid) @ lat) > p.min_leg_separation_m:
                    return a, b
        return None

    # crotch = lowest height at which the surface BETWEEN the legs is as deep (front-back) as the legs themselves:
    # at the midpoint of the leg positions tracked from below, the depth extent of the surface must reach
    # crotch_depth_ratio x the legs' own depth for 3 consecutive slices. On the first real scan the inner thighs were
    # joined by a narrow 2-4 cm bridge (LiDAR merges them) and by debris strands; "one part spans both legs" fired there
    # (0.38-0.47 m), the pelvis only became full depth at ~0.86 m.
    from shapely.geometry import LineString

    fwd = np.array([-lat[1], lat[0]])
    crotch_z = None
    low = [sl for sl in slices if 0.2 * top < sl.z < (arm_z or 0.7 * top)]
    legs, leg_depth, run = None, None, 0
    for k, sl in enumerate(low):
        pair = legs_apart(sl)
        if pair and all(a.pts is not None for a in pair):
            legs = [float((a.centroid - axis) @ lat) for a in pair]
            leg_depth = float(np.mean([np.ptp((a.pts - axis) @ fwd) for a in pair]))
        if legs is None or not leg_depth:
            continue
        mid = 0.5 * (legs[0] + legs[1])
        # chord of the CLOSED section outline along the front-back line through the midpoint between the legs:
        # loose debris strands are not closed outlines, and the narrow inner-thigh bridge gives a short chord
        c0 = axis + mid * lat
        line = LineString([tuple(c0 - 0.6 * fwd), tuple(c0 + 0.6 * fwd)])
        depth_mid = 0.0
        for q in _section_polygons(tm, sl.z):
            if q.area > 0.003:
                depth_mid = max(depth_mid, float(q.intersection(line).length))
        run = run + 1 if depth_mid >= p.crotch_depth_ratio * leg_depth else 0
        if run == 3:
            crotch_z = float(low[k - 2].z)
            break
    if crotch_z is None:
        out["flags"].append("legs never appear as two separate parts: crotch not found")
    out["landmarks"]["crotch_z"] = crotch_z
    reasons_crotch = []
    if crotch_z is not None and not 0.40 <= crotch_z / top <= 0.52:
        reasons_crotch.append(f"crotch at {crotch_z / top:.2f} of the top height is outside 0.40-0.52: the thighs "
                              "probably touch in the mesh, so the crotch level is biased")
        out["flags"].append(reasons_crotch[-1])

    def circ_at(sl: Slice, part: Part, name: str, extra_reasons: list[str]):
        c = part_circumference(tm, sl, part, p)
        if c["value_m"] is None:
            return _value(name, None, "unavailable", [c["reason"]], level_z=sl.z)
        return _value(name, c["value_m"], st, base + extra_reasons, level_z=sl.z,
                      convex_hull_m=c["convex_hull_m"], raw_outline_m=c["raw_outline_m"], area_m2=c["area_m2"],
                      note="perimeter of the smoothed section outline (Gaussian, sigma "
                           f"{p.contour_sigma_m * 1000:.0f} mm); raw_outline_m and convex_hull_m are diagnostics")

    # torso profile between crotch and arm separation (torso part only)
    profile = []
    if arm_z is not None:
        lo_z = crotch_z if crotch_z is not None else 0.4 * top
        for sl in slices:
            if lo_z <= sl.z <= arm_z:
                t = torso_part(sl)
                c = part_circumference(tm, sl, t, p) if t else {"value_m": None, "reason": "no torso part"}
                profile.append({"z_m": round(sl.z, 4), "value_m": c["value_m"], "parts_in_slice": len(sl.parts),
                                **({"reason": c["reason"]} if c["value_m"] is None else {})})
    out["torso_profile"] = profile

    # running median of the profile: the level choice must not follow one bad slice
    vals = np.array([r["value_m"] if r["value_m"] else np.nan for r in profile], float)
    half = p.level_median_slices // 2
    med = np.array([np.nanmedian(vals[max(0, i - half):i + half + 1]) if np.isfinite(vals[max(0, i - half):i + half + 1]).any()
                    else np.nan for i in range(len(vals))]) if len(vals) else vals

    def pick(lo, hi, mode, name, why):
        rows = [(r, m_) for r, m_ in zip(profile, med) if lo <= r["z_m"] <= hi and r["value_m"] and np.isfinite(m_)]
        if not rows:
            out["measurements"].append(_value(name, None, "unavailable",
                                              [f"no closed torso outline between {lo:.2f} and {hi:.2f} m"]))
            return
        r = (max if mode == "max" else min)(rows, key=lambda rm: rm[1])[0]
        sl = next(s for s in slices if abs(s.z - r["z_m"]) < 1e-6)
        out["measurements"].append(circ_at(sl, torso_part(sl), name, why))

    if arm_z is not None and crotch_z is not None:
        span = arm_z - crotch_z
        pick(arm_z - 0.10, arm_z - 0.02, "max", "chest_circumference",
             ["chest level = largest torso outline 2-10 cm below arm separation (the top 2 cm can clip the "
              "shoulders; arm separation is below the true armpit when arms hang close)"])
        pick(crotch_z + 0.35 * span, crotch_z + 0.75 * span, "min", "waist_circumference",
             ["waist = smallest torso outline in 35-75 % of crotch..arm-separation"] + reasons_crotch)
        pick(crotch_z, crotch_z + 0.35 * span, "max", "hip_circumference",
             ["hip = largest torso/pelvis outline in 0-35 % of crotch..arm-separation"] + reasons_crotch)
    else:
        for n in ("chest_circumference", "waist_circumference", "hip_circumference"):
            out["measurements"].append(_value(n, None, "unavailable", ["crotch or arm separation not found"]))

    # limbs: perpendicular to each limb's own axis (CLAUDE.md). Axis = line fitted through the centroids of that
    # limb's separated slice parts; a poor fit (moving / ghost limb) makes the value unavailable instead of guessed.
    def side_parts(lo_z, hi_z, pick):
        pts = {-1: [], 1: []}
        for sl in slices:
            if lo_z <= sl.z <= hi_z:
                for a, sgn in pick(sl):
                    if sgn in pts:
                        pts[sgn].append([*a.centroid, sl.z])
        return {k: np.array(v) for k, v in pts.items()}

    def fit_axis(c):
        if len(c) < 5:
            return None, None, None
        m0 = c.mean(0)
        _, _, vt = np.linalg.svd(c - m0)
        d = vt[0] if vt[0][2] >= 0 else -vt[0]
        e = (c - m0) - np.outer((c - m0) @ d, d)
        return m0, d, float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))

    def at_z(m0, d, z):
        return m0 + d * (z - m0[2]) / d[2]

    def limb_value(name, fit, z, reasons, max_resid=0.02):
        m0, d, resid = fit
        if m0 is None:
            return _value(name, None, "unavailable", ["too few separated slices to find the limb axis"])
        if resid > max_resid:
            return _value(name, None, "unavailable", [f"limb centre line is not straight (residual {resid * 100:.1f} cm):"
                                                      " the limb moved or has ghost surfaces"])
        c = limb_circumference(tm, at_z(m0, d, z), d, p)
        if c["value_m"] is None:
            return _value(name, None, "unavailable", [c["reason"]], level_z=z)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(d[2])))))
        return _value(name, c["value_m"], st, base + reasons + [f"slice perpendicular to the limb axis (tilt {tilt:.0f} deg)"],
                      level_z=z, convex_hull_m=c["convex_hull_m"], raw_outline_m=c["raw_outline_m"])

    # left / right: given the subject's LEFT direction in the body frame (e.g. camera right at the first view when
    # the subject started facing the camera), parts on that side of the lateral axis are "left"
    if subject_left_xy is not None:
        sl_ = np.asarray(subject_left_xy, float)[:2]
        left_sign = 1 if float(sl_ @ lat) >= 0 else -1
        side_tag = {left_sign: "left", -left_sign: "right"}
        lr = "left/right from the start direction (assumes the subject faced the camera at the start)"
        out["left_right"] = {"subject_left_xy": sl_.tolist(), "assumption": lr}
    else:
        side_tag = {-1: "side_neg", 1: "side_pos"}
        lr = "left/right not assigned: the body's facing direction is not determined"
    if crotch_z is not None:
        def legs_pick(sl):
            pr = legs_apart(sl)
            return [(a, int(np.sign((a.centroid - axis) @ lat))) for a in pr] if pr else []
        legs = side_parts(0.12 * top, crotch_z - 0.02, legs_pick)
        for sgn, tag in ((-1, side_tag[-1]), (1, side_tag[1])):
            fit = fit_axis(legs[sgn])
            out["measurements"].append(limb_value(f"thigh_circumference_{tag}", fit, crotch_z - 0.03,
                                                  reasons_crotch + [lr, "3 cm below the crotch"]))
            m0, d, r_ = fit
            if m0 is not None and r_ <= 0.02:
                vals = [(z, limb_circumference(tm, at_z(m0, d, z), d, p)) for z in np.arange(0.10 * top, 0.30 * top, 0.01)]
                vals = [(z, c) for z, c in vals if c["value_m"]]
                if vals:
                    zb = max(vals, key=lambda zc: zc[1]["value_m"])[0]
                    out["measurements"].append(limb_value(f"calf_circumference_{tag}", fit, zb,
                                                          [lr, "largest perpendicular leg outline at 10-30 % of height"]))
                else:
                    out["measurements"].append(_value(f"calf_circumference_{tag}", None, "unavailable",
                                                      ["no closed leg outline at 10-30 % of height"]))
    else:
        out["measurements"].append(_value("thigh_circumference", None, "unavailable", ["crotch not found"]))
    if arm_z is not None:
        def arms_pick(sl):
            t = torso_part(sl)
            if t is None:
                return []
            return [(a, int(np.sign((a.centroid - axis) @ lat))) for a in sl.parts
                    if a is not t and abs((a.centroid - axis) @ lat) > 0.5 * width(t)]
        arms = side_parts(arm_z - 0.25, arm_z, arms_pick)
        for sgn, tag in ((-1, side_tag[-1]), (1, side_tag[1])):
            out["measurements"].append(limb_value(f"upper_arm_circumference_{tag}", fit_axis(arms[sgn]), arm_z - 0.03,
                                                  [lr, "3 cm below arm separation (provisional level: no shoulder or"
                                                   " elbow landmarks)"], max_resid=0.015))
    else:
        out["measurements"].append(_value("upper_arm_circumference", None, "unavailable", ["arm separation not found"]))
    out["status"] = "unreliable" if not validated else "measured"
    return out


def limb_circumference(tm, centre, direction, p):
    """Circumference of the closed section outline around `centre` in the plane perpendicular to `direction`."""
    import trimesh
    from shapely.geometry import Point

    d = np.asarray(direction, float) / np.linalg.norm(direction)
    sec = tm.section(plane_origin=centre, plane_normal=d)
    if sec is None:
        return {"value_m": None, "reason": "no surface in the perpendicular plane"}
    try:
        path2d, _ = sec.to_2D(to_2D=trimesh.geometry.plane_transform(centre, d))
    except Exception:  # noqa: BLE001 - degenerate section
        return {"value_m": None, "reason": "degenerate perpendicular section"}
    polys = [q for q in path2d.polygons_closed if q is not None and q.is_valid and q.area > 0]
    inside = [q for q in polys if q.contains(Point(0.0, 0.0))]
    if not inside:
        return {"value_m": None, "reason": "no closed outline around the limb centre (hole or merged limb)"}
    q = min(inside, key=lambda q: q.area)  # the limb itself, not a larger outline that also encloses it
    return {"value_m": smoothed_perimeter(q.exterior.coords, p.contour_sigma_m, p.contour_step_m),
            "raw_outline_m": float(q.exterior.length), "convex_hull_m": float(q.convex_hull.exterior.length)}


def torso_circumference_at(mesh: o3d.geometry.TriangleMesh, z: float, p: BodyParams | None = None) -> dict:
    """Circumference of the TORSO part (the slice part nearest the body axis; separated arms excluded) at height z.
    For comparison with a tape held at a known level. Returns {"value_m": None, "reason": ...} when unavailable."""
    from .basic import o3d_to_trimesh

    p = p or BodyParams()
    o3d.utility.random.seed(p.seed)
    pts = np.asarray(mesh.sample_points_uniformly(p.samples).points)
    top = float(pts[:, 2].max())
    mid = pts[(pts[:, 2] > 0.3 * top) & (pts[:, 2] < 0.7 * top)]
    axis = np.median(mid[:, :2], 0)
    sl = _slice_parts(pts, z, p)
    if not sl.parts:
        return {"value_m": None, "reason": f"no surface at {z:.3f} m"}
    t = min(sl.parts, key=lambda a: np.linalg.norm(a.centroid - axis))
    out = part_circumference(o3d_to_trimesh(mesh), sl, t, p)
    out["parts_in_slice"] = len(sl.parts)
    return out


def write_landmark_image(mesh: o3d.geometry.TriangleMesh, result: dict, path, size: int = 520) -> str:
    """Front and side point projections with landmark and measurement levels (display only)."""
    import cv2

    o3d.utility.random.seed(0)
    pts = np.asarray(mesh.sample_points_uniformly(150000).points)
    lat = np.array(result.get("lateral_axis_xy", [1.0, 0.0]))
    fwd = np.array([-lat[1], lat[0]])
    top = result["landmarks"]["head_top_z"]
    tiles = []
    for name, ax in (("front", lat), ("side", fwd)):
        img = np.full((size, size // 2 + 40, 3), 255, np.uint8)
        h = pts[:, :2] @ ax
        sc = (size - 40) / max(top, 1e-3)
        u = ((h - np.median(h)) * sc + img.shape[1] / 2).astype(int)
        v = (size - 20 - pts[:, 2] * sc).astype(int)
        ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < size)
        img[v[ok], u[ok]] = (150, 150, 150)
        for key, col in (("head_top_z", (0, 0, 0)), ("neck_z", (200, 0, 0)), ("arm_separation_z", (0, 140, 0)),
                         ("crotch_z", (0, 0, 220))):
            z = result["landmarks"].get(key)
            if z is not None:
                y = int(size - 20 - z * sc)
                cv2.line(img, (0, y), (img.shape[1], y), col, 1)
                cv2.putText(img, key.replace("_z", ""), (2, y - 3), 0, 0.35, col, 1)
        for m in result["measurements"]:
            if m.get("level_z") is not None and m["value"] is not None:
                y = int(size - 20 - m["level_z"] * sc)
                cv2.line(img, (img.shape[1] // 4, y), (3 * img.shape[1] // 4, y), (0, 120, 220), 2)
        cv2.putText(img, name, (4, 14), 0, 0.45, (0, 0, 0), 1)
        tiles.append(img)
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1))
    return str(path)
