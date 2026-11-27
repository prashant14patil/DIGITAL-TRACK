"""EXPERIMENTAL, DISPLAY ONLY: an open-licensed template body (MakeHuman base mesh, CC0) fitted to a person's scan.

Decided by the user on 2026-11-27: the Body avatar may show a template body fitted to the scan, for the reference look
(face, hands, smooth limbs). Nothing is ever measured from it; every number comes from the raw measuring mesh.

Frames: the scan is the avatar display frame (metres, Z up, floor z = 0, subject faces -Y, +X = subject's LEFT).
The template (experimental/assets/makehuman_base.obj, CC0; Y up, decimetres, faces +Z, +X = character's left) is converted to
that frame. Steps (all logged):
  1. heights: piecewise-linear map of the template's height so floor, crotch, neck and head top land on the scan's
     landmarks (measurement.body on both meshes)
  2. width/depth: per height band, the template torso/leg extents are scaled to the scan's (smooth in height)
  3. arms: each template arm is rotated about its shoulder joint to the scan's measured arm direction
  4. surface fit: Laplacian deformation; every template vertex is pulled to its closest scan point (closer than
     `max_corr_m`, normals agreeing) while the template's own surface detail (Laplacian coordinates) is kept.
     Face, hands and feet get a low data weight: their shape stays the template's (scan detail there is ~1 cm/pixel).
  5. regions from the template skeleton (head, trunk, arms, legs); vertices whose shape comes from the template
     rather than the scan (low weight, or no scan surface within `measured_m`) are flagged "template / not measured".
Deviation of the fitted surface from the scan is reported; the fit is refused (caller falls back to the scan surface)
if it exceeds the limits.
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

ALGORITHM = "template-fit-0.1 (EXPERIMENTAL, display only)"
TEMPLATE = Path(__file__).resolve().parent / "assets" / "makehuman_base.obj"  # CC0, see assets/README.md
REGIONS = ["head", "trunk", "left_arm", "right_arm", "left_leg", "right_leg"]


@dataclass(frozen=True)
class TemplateFitParams:
    iterations: int = 4
    stiffness: tuple[float, ...] = (60.0, 40.0, 30.0, 25.0)  # stiff: smooth corrections only, no scan noise
    blend_m: float = 0.06  # torso girth change fades out over this distance into neck, shoulders and hips
    limb_weight: float = 0.0  # limbs keep the template's clean shape (posed and lengthened to the scan)
    torso_weight: float = 0.0  # torso size comes from the girth morph; a pull copied the loose shirt's folds
    # "chest": one girth factor from the scan's chest (default, user decision 2026-11-27: loose clothing makes the
    # waist/hip girth the shirt's volume); "profile": girth at every height from the scan (tight clothing)
    torso_mode: str = "chest"
    girth_smooth_slices: int = 15  # arms/legs/head follow the scan only weakly (ghost arm, merged legs, 1 cm pixels)
    max_corr_m: float = 0.08
    normal_dot: float = 0.3
    low_weight: float = 0.08  # face, hands, feet: template shape kept
    measured_m: float = 0.01  # fitted vertex within this of the scan (and full weight) = follows the measurement
    # torso deviation from the scan is REPORTED; refused only when gross (wrong alignment). With loose clothing and
    # torso_mode "chest", a few cm at the waist are expected and are the shirt, not an error.
    max_rms_m: float = 0.05
    max_p95_m: float = 0.10


# ---- template ---------------------------------------------------------------------------------------------------

BUILD_TARGETS = {  # target file -> weight (MakeHuman macro system: gender and build are offsets on the base)
    "african-male-young.target": 1 / 3, "caucasian-male-young.target": 1 / 3, "asian-male-young.target": 1 / 3,
    "universal-male-young-maxmuscle-averageweight.target": 0.5,
}


def _apply_targets(v: np.ndarray, folder: Path, targets: dict) -> list[str]:
    used = []
    for name, w in targets.items():
        f = folder / name
        if not f.is_file():
            continue
        d = np.loadtxt(f, comments="#")
        if d.ndim == 2 and d.shape[1] == 4:
            v[d[:, 0].astype(int)] += w * d[:, 1:4]
            used.append(f"{name} x {w:.2f}")
    return used


def load_template(path: Path = TEMPLATE, build: dict | None = BUILD_TARGETS) -> dict:
    """Body (+ eyes) of the MakeHuman base mesh in the display frame (metres), with joints and triangle faces.
    `build`: CC0 shape targets applied to the base (default: young male, athletic)."""
    if not path.is_file():
        raise FileNotFoundError(f"template not found: {path} (MakeHuman base.obj, CC0)")
    v, faces, g = [], collections.defaultdict(list), None
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            g = line.split(maxsplit=1)[1].strip()
        elif line.startswith("f "):
            faces[g].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    v = np.array(v)
    used_targets = _apply_targets(v, path.parent, build) if build else []
    joints = {k[6:]: v[np.unique(np.concatenate(f))].mean(0) for k, f in faces.items() if k.startswith("joint-")}
    keep = ["body", "helper-l-eye", "helper-r-eye"]
    tris, part = [], []
    for k in keep:
        for f in faces.get(k, []):
            for a in range(1, len(f) - 1):
                tris.append([f[0], f[a], f[a + 1]])
                part.append(k)
    tris = np.array(tris)
    used = np.unique(tris)
    remap = -np.ones(len(v), int)
    remap[used] = np.arange(len(used))
    floor = v[used, 1].min()
    conv = lambda p: np.stack([p[..., 0], -p[..., 2], p[..., 1] - floor], -1) * 0.1  # noqa: E731
    is_eye = np.zeros(len(used), bool)
    eye_tris = np.array([t for t, k in zip(tris, part) if k != "body"])
    if len(eye_tris):
        is_eye[remap[np.unique(eye_tris)]] = True
    return {"vertices": conv(v[used]), "triangles": remap[tris], "joints": {k: conv(p) for k, p in joints.items()},
            "is_eye": is_eye, "license": "MakeHuman base mesh and targets, CC0 1.0 (released September 2020)",
            "targets": used_targets}


def _laplacian(n: int, tris: np.ndarray):
    from scipy.sparse import coo_matrix, diags

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    a = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    deg = np.asarray(a.sum(1)).ravel()
    return (diags(np.ones(n)) - diags(1.0 / np.maximum(deg, 1)) @ a).tocsr()


def _segment_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def _region_labels(x: np.ndarray, j: dict) -> np.ndarray:
    """Nearest skeleton segment -> region (head, trunk, arms, legs)."""
    segs = {
        "head": [("neck", "head")],
        "trunk": [("pelvis", "spine-3"), ("spine-3", "neck"), ("l-clavicle", "r-clavicle")],
        "left_arm": [("l-shoulder", "l-elbow"), ("l-elbow", "l-hand")],
        "right_arm": [("r-shoulder", "r-elbow"), ("r-elbow", "r-hand")],
        "left_leg": [("l-upper-leg", "l-knee"), ("l-knee", "l-ankle"), ("l-ankle", "l-foot-1")],
        "right_leg": [("r-upper-leg", "r-knee"), ("r-knee", "r-ankle"), ("r-ankle", "r-foot-1")],
    }
    best = np.full(len(x), np.inf)
    lab = np.ones(len(x), np.uint8)
    for rid, name in enumerate(REGIONS):
        for a, b in segs[name]:
            if a in j and b in j:
                d, _ = _segment_dist(x, j[a], j[b])
                upd = d < best
                best[upd], lab[upd] = d[upd], rid
    lab[(x[:, 2] > j["neck"][2]) & (lab != 2) & (lab != 3)] = 0
    return lab


def _detail_weight(x: np.ndarray, j: dict, p: TemplateFitParams) -> np.ndarray:
    """Low data weight on face, hands and feet (template shape kept), smooth transition."""
    w = np.ones(len(x))
    face = (x[:, 2] > j["neck"][2] + 0.03) & (x[:, 1] < j["head"][1] - 0.02)  # front of the head
    w[face] = p.low_weight
    for s in ("l", "r"):
        for jn, r in ((f"{s}-hand", 0.11), (f"{s}-ankle", 0.0)):
            if jn not in j:
                continue
            if jn.endswith("hand"):
                d = np.linalg.norm(x - j[jn], axis=1)
                w = np.minimum(w, np.clip((d - r * 0.6) / (r * 0.4), p.low_weight, 1.0))
            else:
                w[x[:, 2] < j[jn][2] + 0.02] = np.minimum(w[x[:, 2] < j[jn][2] + 0.02], p.low_weight)
    return w


# ---- fit --------------------------------------------------------------------------------------------------------

def _rot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def _scan_wrists(scan_pts: np.ndarray, crotch: float, arm_z: float, axis: np.ndarray) -> dict:
    """Lowest separated arm slice on each side (about the wrist / hand): (x, y, z) of its centroid."""
    from ..measurement.body import BodyParams, _slice_parts

    p = BodyParams()
    out = {}
    cents = {"l": [], "r": []}
    for z in np.arange(max(crotch, arm_z - 0.30), arm_z, 0.01):
        sl = _slice_parts(scan_pts, z, p)
        if len(sl.parts) < 2:
            continue
        torso = min(sl.parts, key=lambda a: np.linalg.norm(a.centroid - axis))
        for a in sl.parts:
            if a is torso:
                continue
            dx = a.centroid[0] - axis[0]
            if abs(dx) > 0.5 * torso.extent[0] + 0.03:
                cents["l" if dx > 0 else "r"].append([*a.centroid, z])
    for s, c in cents.items():
        c = np.array(c)
        if len(c) >= 5:
            low = c[np.argsort(c[:, 2])[:5]]
            out[s] = np.median(low, 0)
    return out


def fit_template(scan: o3d.geometry.TriangleMesh, p: TemplateFitParams | None = None, template: dict | None = None):
    """Return (fitted o3d mesh, region labels, template-only mask, report) or raise ValueError when refused."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import factorized

    from ..measurement.body import measure_body

    p = p or TemplateFitParams()
    t = template or load_template()
    x = t["vertices"].copy()
    tris = t["triangles"]
    j = {k: v.copy() for k, v in t["joints"].items()}
    report: dict = {"algorithm": ALGORITHM, "params": asdict(p), "template": t["license"], "steps": []}

    tm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x), o3d.utility.Vector3iVector(tris))
    ms, mt = measure_body(scan)["landmarks"], measure_body(tm)["landmarks"]
    keys = ["crotch_z", "neck_z", "head_top_z"]
    if any(ms.get(k) is None or mt.get(k) is None for k in keys):
        raise ValueError(f"landmarks missing (scan {ms}, template {mt})")
    src = np.r_[0.0, [mt[k] for k in keys]]
    dst = np.r_[0.0, [ms[k] for k in keys]]
    zmap = lambda z: np.interp(z, src, dst, left=None, right=None) + np.where(z > src[-1], (z - src[-1]) * dst[-1] / src[-1], 0)  # noqa: E731
    x[:, 2] = zmap(x[:, 2])
    j = {k: np.r_[v[:2], zmap(np.array([v[2]]))[0]] for k, v in j.items()}
    report["steps"].append({"step": "height map", "template_landmarks": mt, "scan_landmarks": ms})

    # 2. width / depth per height band (centred on each mesh's axis), smoothed over height
    o3d.utility.random.seed(0)
    sp = np.asarray(scan.sample_points_uniformly(300000).points)
    tp = np.asarray(tm.sample_points_uniformly(300000).points)
    tp[:, 2] = zmap(tp[:, 2])
    ax_s = np.median(sp[(sp[:, 2] > 0.3 * dst[-1]) & (sp[:, 2] < 0.7 * dst[-1])][:, :2], 0)
    ax_t = np.median(tp[(tp[:, 2] > 0.3 * dst[-1]) & (tp[:, 2] < 0.7 * dst[-1])][:, :2], 0)
    x[:, :2] += ax_s - ax_t
    for k in j:
        j[k][:2] += ax_s - ax_t
    tp[:, :2] += ax_s - ax_t
    report["steps"].append({"step": "axis alignment", "shift_xy_m": (ax_s - ax_t).tolist(),
                            "note": "no banded width scaling (it pushed separate template legs apart); the surface "
                                    "fit does the widening"})

    # 3. arms to the scan's arm directions (arm vertices = nearest skeleton bone is an arm bone)
    arm_lab = _region_labels(x, j)
    arm_z = ms.get("arm_separation_z") or dst[2] - 0.3
    wrists = _scan_wrists(sp, dst[1], arm_z, ax_s)
    # symmetric display pose: the mean of both scan arm angles (on the first real scan the two differed by 12 deg
    # because one arm had ghost surfaces; an asymmetric avatar pose only showed that noise)
    angs = []
    for s in ("l", "r"):
        sh, wr = j.get(f"{s}-shoulder"), wrists.get(s)
        if sh is not None and wr is not None:
            angs.append(np.arctan2(abs(wr[0] - sh[0]), max(sh[2] - wr[2], 1e-3)))
    a_sym = float(np.mean(angs)) if angs else None
    for s in ("l", "r"):
        sh, hand = j.get(f"{s}-shoulder"), j.get(f"{s}-hand")
        if s not in wrists or sh is None or hand is None:
            report["steps"].append({"step": f"arm {s}", "rotated": False, "why": "scan arm direction not found"})
            continue
        # rotate only within the frontal (x-z) plane about the shoulder: the arm's sideways angle comes from the scan,
        # its forward/backward lean stays the template's (scan arm lines lean forward when the arm moved / ghosts)
        a_t = np.arctan2(abs(hand[0] - sh[0]), sh[2] - hand[2])  # template: angle from straight down
        wr = wrists[s]
        a_s = a_sym  # scan: shoulder joint -> wrist slice, averaged over both arms
        sgn = 1 if s == "l" else -1
        da = sgn * (a_s - a_t)
        r = np.array([[np.cos(da), 0, -np.sin(da)], [0, 1, 0], [np.sin(da), 0, np.cos(da)]])
        u = (hand - sh) / np.linalg.norm(hand - sh)
        sproj = (x - sh) @ u
        is_arm = arm_lab == (2 if s == "l" else 3)
        # rotation weight ramps from 3 cm to 12 cm beyond the shoulder joint, so the armpit and chest are not dragged
        w = np.clip((sproj - 0.03) / 0.09, 0, 1) * is_arm
        w = w * w * (3 - 2 * w)  # smoothstep
        rx = (x - sh) @ r.T + sh
        x = x + w[:, None] * (rx - x)
        for k in (f"{s}-elbow", f"{s}-hand"):
            j[k] = (j[k] - sh) @ r.T + sh
        report["steps"].append({"step": f"arm {s}", "rotated": True, "template_angle_deg": float(np.degrees(a_t)),
                                "scan_angle_deg": float(np.degrees(a_s))})

    # 3b. measurement-driven torso morph: template torso girth at each height scaled to the scan's measured torso
    #     circumference profile (measurement.body torso_profile). Large-scale shape from the scan, none of its noise.
    lab0 = _region_labels(x, j)
    tm2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x), o3d.utility.Vector3iVector(tris))
    ps = {r["z_m"]: r["value_m"] for r in measure_body(scan)["torso_profile"] if r["value_m"]}
    pt = {r["z_m"]: r["value_m"] for r in measure_body(tm2)["torso_profile"] if r["value_m"]}
    common = sorted(set(round(z, 2) for z in ps) & set(round(z, 2) for z in pt))
    ps = {round(z, 2): v for z, v in ps.items()}
    pt = {round(z, 2): v for z, v in pt.items()}
    if len(common) >= 5:
        zc = np.array(common)
        ratio = np.array([ps[z] / pt[z] for z in common])
        k_ = p.girth_smooth_slices
        ratio = np.clip(np.convolve(np.pad(ratio, k_ // 2, mode="edge"), np.ones(k_) / k_, mode="valid"), 0.8, 1.5)
        if p.torso_mode == "chest":
            # one factor from the chest (user decision 2026-11-27): with loose clothing the waist/hip girth is the
            # shirt's volume; the chest is where it sits closest. Waist/hip SHAPE stays the template's.
            cs = {x_["name"]: x_["value"] for x_ in measure_body(scan)["measurements"]}.get("chest_circumference")
            ct = {x_["name"]: x_["value"] for x_ in measure_body(tm2)["measurements"]}.get("chest_circumference")
            if cs and ct:
                ratio = np.full_like(ratio, float(np.clip(cs / ct, 0.8, 1.5)))
        neck_cut = (ms.get("neck_z") or j["neck"][2]) - 0.03
        trunk = (lab0 == 1) & (x[:, 2] < neck_cut)  # never the jaw / chin (their nearest bone can be the spine top)
        ztr = x[trunk, 2]
        r_v = np.interp(ztr, zc, ratio)  # constant beyond the measured range
        # torso centre per height: mean of trunk vertices in a 2 cm band
        cen = np.array([x[trunk][np.abs(ztr - z) < 0.02][:, :2].mean(0) if np.any(np.abs(ztr - z) < 0.02)
                        else ax_s for z in ztr])
        disp = np.zeros_like(x)
        disp[trunk, :2] = (x[trunk, :2] - cen) * (r_v[:, None] - 1.0)
        # neighbouring vertices (neck, shoulders, upper arms, hips, thighs) take the displacement of their nearest
        # trunk vertex, fading out over `blend_m`, so no step appears at the region borders
        from scipy.spatial import cKDTree

        tr_idx = np.flatnonzero(trunk)
        other = np.flatnonzero(~trunk)
        dist, nn = cKDTree(x[tr_idx]).query(x[other], k=8)
        wgt = np.exp(-(dist / p.blend_m) ** 2)
        disp[other] = (disp[tr_idx[nn]] * wgt[..., None]).sum(1) / np.maximum(wgt.sum(1, keepdims=True), 1e-9)             * np.exp(-(dist[:, :1] / p.blend_m) ** 2)
        disp[x[:, 2] > neck_cut + 0.03] = 0.0  # head untouched by the girth change
        x = x + disp
        report["steps"].append({"step": "torso morph", "mode": p.torso_mode, "slices": len(common),
                                "girth_ratio_min_max": [float(ratio.min()), float(ratio.max())]})
    else:
        report["steps"].append({"step": "torso morph", "skipped": "too few torso slices on scan and template"})

    # 4. Laplacian surface fit (stiff; torso only by default: limbs keep the template's clean shape)
    lap = _laplacian(len(x), tris)
    delta = lap @ x
    wdet = _detail_weight(x, j, p)
    detail = wdet < 0.99  # face, hands, feet: template detail (shown pale)
    lab_fit = _region_labels(x, j)
    neck_z = ms.get("neck_z") or j["neck"][2]
    head = (x[:, 2] > neck_z + 0.01) & (np.linalg.norm(x[:, :2] - ax_s, axis=1) < 0.16)
    detail |= head
    wdet[head] = 0.0  # head is pure template (scaled by the height map): the scan head is hair at 1 cm pixels
    wdet = np.where(lab_fit == 1, np.minimum(wdet, p.torso_weight), np.minimum(wdet, p.limb_weight))
    eye = t["is_eye"]
    x_coarse = x.copy()
    wdet[eye] = 1.0  # eyes are separate pieces: anchored to their coarse place here, carried with the head below
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scan))
    scan_n = o3d.geometry.TriangleMesh(scan)
    scan_n.compute_triangle_normals()
    tn = np.asarray(scan_n.triangle_normals)
    lt = (lap.T @ lap).tocsc()
    n_iter = p.iterations if wdet[~eye].max() > 0 else 0  # nothing to fit to: skip (an unanchored solve drifts)
    if n_iter == 0:
        report["steps"].append({"step": "surface fit", "skipped": "all data weights are zero (shape from the morph)"})
    for it in range(n_iter):
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x), o3d.utility.Vector3iVector(tris))
        m.compute_vertex_normals()
        vn = np.asarray(m.vertex_normals)
        q = scene.compute_closest_points(o3d.core.Tensor(x.astype(np.float32)))
        cp = q["points"].numpy().astype(float)
        pid = q["primitive_ids"].numpy()
        d = np.linalg.norm(cp - x, axis=1)
        ok = (d < p.max_corr_m) & (np.sum(vn * tn[pid], 1) > p.normal_dot)
        wd = np.where(ok, wdet, 0.0)
        wd[eye] = 1.0
        cp[eye] = x_coarse[eye]
        lam = p.stiffness[min(it, len(p.stiffness) - 1)]
        solve = factorized((lam * lt + diags(wd)).tocsc())
        rhs = lam * (lap.T @ delta) + wd[:, None] * cp
        x = np.stack([solve(rhs[:, c]) for c in range(3)], 1)
        report["steps"].append({"step": f"fit {it + 1}", "stiffness": lam, "corresponded": float(ok.mean())})

    if eye.any():  # move the eyes with the surrounding head surface (mean displacement of the nearest head vertices)
        from scipy.spatial import cKDTree

        body_idx = np.flatnonzero(~eye)
        _, nn = cKDTree(x_coarse[body_idx]).query(x_coarse[eye], k=30)
        x[eye] = x_coarse[eye] + (x[body_idx[nn]] - x_coarse[body_idx[nn]]).mean(1)
    fitted = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x), o3d.utility.Vector3iVector(tris))
    fitted.compute_vertex_normals()
    d = scene.compute_distance(o3d.core.Tensor(x.astype(np.float32))).numpy()
    full = (lab_fit == 1) & ~detail  # torso: its size follows the scan (deviation reported on these)
    template_only = detail  # head, hands, feet: template detail, not measured (shown pale)
    rms = float(np.sqrt(np.mean(d[full] ** 2)))
    p95 = float(np.percentile(d[full], 95))
    labels = _region_labels(x, j)
    labels[head] = 0
    labels[(labels == 0) & ~head] = 1  # below the neck line: trunk
    # display band like the reference's "shorts" (pelvis, crotch - 9 cm .. crotch + 11 cm, trunk and legs only)
    cz = ms.get("crotch_z") or j["pelvis"][2]
    report["pelvis_band"] = ((x[:, 2] > cz - 0.09) & (x[:, 2] < cz + 0.11) & np.isin(labels, [1, 4, 5])).tolist()
    report.update({"deviation_full_weight_m": {"rms": rms, "p95": p95, "max": float(d[full].max())},
                   "template_only_fraction": float(template_only.mean()),
                   "fitted_within_measured_m_fraction": float(np.mean(d[full] <= p.measured_m)),
                   "regions": {n: int((labels == i).sum()) for i, n in enumerate(REGIONS)}})
    if rms > p.max_rms_m or p95 > p.max_p95_m:
        raise ValueError(f"template fit too far from the scan (RMS {rms * 1000:.1f} mm, P95 {p95 * 1000:.1f} mm)")
    report["status"] = "fitted"
    return fitted, labels, template_only, report
