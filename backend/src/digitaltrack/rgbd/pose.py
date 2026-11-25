"""Per-view poses for a subject turning in place in front of a fixed camera (floor-constrained, no ICP).

Model (options C + B of the pose-estimation plan). The floor is static and measured once, so for a subject
standing upright on the floor each stop k has 3 unknowns, a turn angle and a floor position:

    floor_from_body_k = SE2(theta_k, x_k, y_k)       (height and tilt fixed by the floor)
    camera_from_body_k = inv(floor_from_camera) @ floor_from_body_k

Body frame = floor frame at stop 0 (stop 0 is the identity).

Pairwise registration i -> j: EXHAUSTIVE 3-DOF CORRELATION SEARCH (no ICP, no correspondence re-association).
  * Both subject clouds are binned into height slices of a 2-D occupancy grid; the target grid is blurred.
  * For every candidate turn angle within +-max_step_deg (protocol prior: small steps), the source is rotated
    about its centroid and the best floor shift is found by FFT cross-correlation summed over height slices.
  * Coarse grid -> fine grid around the best angle -> parabolic interpolation of the peak in angle and shift.
  QC per edge (evaluation only, never changes the transform): overlap and RMS distance of overlapping points,
  ambiguity (best score vs the best score at a clearly different angle), and SHARPNESS: how much the overlap
  drops when the solution is perturbed by +-angle / +-shift. A flat response means the geometry does not fix
  the pose (e.g. a cylinder's spin) and the edge is rejected, never silently trusted.

Global: SE(2) pose graph over all accepted edges (sequential, skip-one, closing edge last -> first), solved by
non-linear least squares (scipy). The loop-closure error is reported BEFORE optimisation.

All thresholds are PROPOSED engineering defaults, checked only on synthetic data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
from scipy import fft as sfft
from scipy.fft import next_fast_len
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "selfrot-se2-correlation-0.2"  # 0.2: ambiguity = separate local maxima only


class PoseEstimationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PoseParams:
    max_step_deg: float = 60.0  # prior: consecutive stops turn less than this
    max_shift_m: float = 0.30  # prior: the subject moves less than this on the floor between compared stops
    coarse_cell_m: float = 0.02
    coarse_step_deg: float = 2.0
    fine_cell_m: float = 0.005
    fine_step_deg: float = 0.25
    z_bin_m: float = 0.02
    blur_cells: float = 1.0
    overlap_threshold_m: float = 0.015
    min_overlap: float = 0.3
    min_overlap_points: int = 150
    max_rms_m: float = 0.010
    ambiguity_exclusion_deg: float = 10.0
    max_ambiguity_ratio: float = 0.92  # best score at a clearly different angle / best score
    sharpness_angle_deg: float = 5.0  # diagnostics only (reported, not used to reject)
    sharpness_shift_m: float = 0.02
    robust_c_m: float = 0.01  # Cauchy scale of the alignment cost used for the edge information matrix
    hessian_step_m: float = 0.005  # at least the point spacing is used
    hessian_step_deg: float = 0.5
    n_eff_cap: int = 300  # dense points from one surface are not independent: cap the effective count
    chain_tolerance_deg: float = 5.0  # non-sequential edges must agree with the sequential chain
    chain_tolerance_m: float = 0.05
    skip_edges: bool = True
    loop_closure: bool = True
    closure_gap_deg: float | None = None  # try the closing edge if the summed turn is within this of 360 (default max_step_deg); real self-rotation under-counts steps
    max_loop_rotation_deg: float = 3.0
    max_loop_translation_m: float = 0.02  # floor of the limit; the limit grows as per-edge * sqrt(n)
    max_loop_translation_per_edge_m: float = 0.005
    max_edge_residual_sigma: float = 4.0
    max_coverage_gap_deg: float = 45.0
    max_view_std_m: float = 0.005  # per-view position 1-sigma from the pose graph
    max_view_std_deg: float = 0.5


# ---- SE(2) helpers ----------------------------------------------------------------------------

def se2(theta: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def se2_params(m: np.ndarray) -> tuple[float, float, float]:
    return float(np.arctan2(m[1, 0], m[0, 0])), float(m[0, 2]), float(m[1, 2])


def se2_to_4x4(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:2, :2] = m[:2, :2]
    out[:2, 3] = m[:2, 2]
    return out


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def apply_se2(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = np.array(pts, dtype=float, copy=True)
    out[:, :2] = pts[:, :2] @ m[:2, :2].T + m[:2, 2]
    return out


# ---- correlation search -----------------------------------------------------------------------

@dataclass
class Grid:
    lo: np.ndarray  # (2,) xy of cell (0, 0)
    cell: float
    nx: int
    ny: int
    nz: int
    z_bin: float


def _grid(a: np.ndarray, b: np.ndarray, cell: float, pad: float, z_bin: float) -> Grid:
    ra = np.max(np.linalg.norm(a[:, :2] - a[:, :2].mean(0), axis=1))  # a is rotated about its centroid
    lo = np.minimum(b[:, :2].min(0), a[:, :2].mean(0) - ra) - pad
    hi = np.maximum(b[:, :2].max(0), a[:, :2].mean(0) + ra) + pad
    n = [next_fast_len(int(v)) for v in np.ceil((hi - lo) / cell)]  # FFT-friendly sizes
    zmax = max(a[:, 2].max(), b[:, 2].max())
    return Grid(lo, cell, int(n[0]), int(n[1]), int(np.ceil(zmax / z_bin)) + 1, z_bin)


def _occupancy(pts: np.ndarray, g: Grid) -> np.ndarray:
    ij = np.floor((pts[:, :2] - g.lo) / g.cell).astype(int)
    k = np.clip(np.floor(pts[:, 2] / g.z_bin).astype(int), 0, g.nz - 1)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < g.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < g.ny)
    flat = (k[ok] * g.nx + ij[ok, 0]) * g.ny + ij[ok, 1]
    return np.bincount(flat, minlength=g.nz * g.nx * g.ny).reshape(g.nz, g.nx, g.ny).astype(np.float32)


def _parabolic(ym: float, y0: float, yp: float) -> float:
    """Sub-sample peak offset; 0 when a neighbour lies outside the allowed window (non-finite)."""
    if not (np.isfinite(ym) and np.isfinite(y0) and np.isfinite(yp)):
        return 0.0
    d = ym - 2 * y0 + yp
    return 0.0 if d >= 0 else float(np.clip(0.5 * (ym - yp) / d, -0.5, 0.5))


class _Searcher:
    def __init__(self, src: np.ndarray, dst: np.ndarray, cell: float, p: PoseParams, max_shift_m: float):
        self.src, self.p = src, p
        self.c = src[:, :2].mean(0)
        self.g = _grid(src, dst, cell, max_shift_m, p.z_bin_m)
        b = gaussian_filter(_occupancy(dst, self.g), sigma=(0, p.blur_cells, p.blur_cells))
        self.fb = sfft.rfft2(b, workers=-1)  # all cores; same result as numpy
        self.max_shift_cells = max_shift_m / cell

    def score(self, theta: float) -> tuple[float, np.ndarray]:
        """Best correlation score at this angle and the floor shift (metres) that achieves it."""
        r = se2(0, *self.c) @ se2(theta, 0, 0) @ se2(0, *(-self.c))
        a = _occupancy(apply_se2(r, self.src), self.g)
        corr = sfft.irfft2(np.sum(np.conj(sfft.rfft2(a, workers=-1)) * self.fb, axis=0), s=(self.g.nx, self.g.ny),
                           workers=-1)
        sx = np.fft.fftfreq(self.g.nx, 1 / self.g.nx)
        sy = np.fft.fftfreq(self.g.ny, 1 / self.g.ny)
        allowed = np.hypot(sx[:, None], sy[None, :]) <= self.max_shift_cells
        corr = np.where(allowed, corr, -np.inf)
        i, j = np.unravel_index(int(np.argmax(corr)), corr.shape)
        nxt = lambda arr, k, n: arr[k % n]
        di = _parabolic(corr[(i - 1) % self.g.nx, j], corr[i, j], corr[(i + 1) % self.g.nx, j])
        dj = _parabolic(corr[i, (j - 1) % self.g.ny], corr[i, j], corr[i, (j + 1) % self.g.ny])
        shift = np.array([nxt(sx, i, self.g.nx) + di, nxt(sy, j, self.g.ny) + dj]) * self.g.cell
        return float(corr[i, j]), shift

    def transform(self, theta: float, shift: np.ndarray) -> np.ndarray:
        return se2(0, *shift) @ se2(0, *self.c) @ se2(theta, 0, 0) @ se2(0, *(-self.c))


def search_se2(src: np.ndarray, dst: np.ndarray, p: PoseParams) -> tuple[np.ndarray, dict]:
    lim = np.radians(p.max_step_deg)
    coarse = _Searcher(src, dst, p.coarse_cell_m, p, p.max_shift_m)
    thetas = np.arange(-lim, lim + 1e-9, np.radians(p.coarse_step_deg))
    scores = np.array([coarse.score(t)[0] for t in thetas])
    k = int(np.argmax(scores))
    far = np.abs(wrap(thetas - thetas[k])) > np.radians(p.ambiguity_exclusion_deg)
    # Only a SEPARATE local maximum is a competing solution. The shoulder of one broad peak (smooth surfaces such
    # as a human torso) is not: its width is angular uncertainty, reported via the information matrix. A flat
    # response (every angle ties, e.g. a cylinder's spin) still counts, because ties are local maxima.
    peak = np.ones_like(far)
    peak[1:] &= scores[1:] >= scores[:-1]
    peak[:-1] &= scores[:-1] >= scores[1:]
    rivals = far & peak
    ambiguity = float(scores[rivals].max() / scores[k]) if rivals.any() and scores[k] > 0 else 0.0

    # fine stage on the coarse-aligned source: small residual angle and shift window, finer grid
    m_coarse = coarse.transform(thetas[k], coarse.score(thetas[k])[1])
    fine = _Searcher(apply_se2(m_coarse, src), dst, p.fine_cell_m, p, 2.5 * p.coarse_cell_m)
    half = np.radians(1.5 * p.coarse_step_deg)
    ft = np.arange(-half, half + 1e-9, np.radians(p.fine_step_deg))
    fs = np.array([fine.score(t)[0] for t in ft])
    m = int(np.argmax(fs))
    dtheta = ft[m]
    if 0 < m < len(ft) - 1:
        dtheta += _parabolic(fs[m - 1], fs[m], fs[m + 1]) * np.radians(p.fine_step_deg)
    _, shift = fine.score(dtheta)
    total = fine.transform(dtheta, shift) @ m_coarse
    theta = se2_params(total)[0]
    info = {"coarse_best_deg": float(np.degrees(thetas[k])), "ambiguity_ratio": ambiguity,
            "at_search_limit": bool(abs(theta) > lim - np.radians(p.coarse_step_deg)),
            "n_angles_evaluated": int(len(thetas) + len(ft) + 2)}
    return total, info


# ---- edges ------------------------------------------------------------------------------------

@dataclass
class Edge:
    i: int
    j: int
    kind: str  # sequential | skip | closure
    transform: np.ndarray | None  # 3x3: floor coords of view i -> floor coords of view j
    ok: bool
    reasons: list[str] = field(default_factory=list)
    overlap: float | None = None
    overlap_points: int = 0
    rms_m: float | None = None
    ambiguity_ratio: float | None = None
    sharpness: dict | None = None
    information: np.ndarray | None = None  # 3x3 for a left perturbation (dtheta, dx, dy) in floor_j coordinates
    std: dict | None = None  # 1-sigma implied by `information` (theta deg, x m, y m)
    weak_directions: list[str] = field(default_factory=list)
    cost_minimum_offset: dict | None = None  # where the fitted cost quadratic has its minimum (diagnostic, not applied)
    post_residual_sigma: float | None = None

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("transform", "information")}
        if self.transform is not None:
            th, tx, ty = se2_params(self.transform)
            d.update(theta_deg=float(np.degrees(th)), tx_m=tx, ty_m=ty)
        return d


def _overlap(src: np.ndarray, tree: cKDTree, m: np.ndarray, thr: float) -> tuple[float, np.ndarray]:
    d, _ = tree.query(apply_se2(m, src), distance_upper_bound=thr)
    ok = np.isfinite(d)
    return float(ok.mean()), d[ok]


def _information(src: np.ndarray, tree: cKDTree, m: np.ndarray, p: PoseParams, n_eff: int,
                 sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Curvature of a smooth (Cauchy) alignment cost around the solution -> 3x3 information. EVALUATION ONLY.

    A quadratic is fitted to the cost sampled on a 5x5x5 grid of perturbations whose spacing is at least the point
    spacing: finite differences finer than the sampling measure nearest-neighbour ripple, not the shape of the
    valley, and give falsely confident (tiny) uncertainties. Returns (information, offset of the fitted minimum);
    the offset is reported as a diagnostic and is NOT applied to the transform.
    """
    c = p.robust_c_m
    spacing = float(np.median(cKDTree(src).query(src, k=2)[0][:, 1]))
    h = np.array([np.radians(p.hessian_step_deg), max(p.hessian_step_m, spacing), max(p.hessian_step_m, spacing)])

    def cost(d3):
        d, _ = tree.query(apply_se2(se2(*d3) @ m, src), distance_upper_bound=0.1)
        d = np.where(np.isfinite(d), d, 0.1)
        return float(np.mean(c ** 2 * np.log1p((d / c) ** 2)))

    g = np.array(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij")).reshape(3, -1).T * h  # (125, 3)
    f = np.array([cost(d) for d in g])
    u = g / h  # well-conditioned design in step units
    a = np.column_stack([np.ones(len(u)), u, u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2,
                         u[:, 0] * u[:, 1], u[:, 0] * u[:, 2], u[:, 1] * u[:, 2]])
    coef = np.linalg.lstsq(a, f, rcond=None)[0]
    hu = np.array([[2 * coef[4], coef[7], coef[8]], [coef[7], 2 * coef[5], coef[9]], [coef[8], coef[9], 2 * coef[6]]])
    hess = hu / np.outer(h, h)
    w, v = np.linalg.eigh(hess)
    hess = (v * np.clip(w, 0, None)) @ v.T  # negative curvature => no information in that direction
    grad = coef[1:4] / h
    offset = -np.linalg.pinv(hess) @ grad
    return hess * 0.5 * n_eff / sigma ** 2, offset  # Gauss-Newton: cost = mean r^2  =>  info = (N/2) H / sigma^2


def register_pair(src: np.ndarray, dst: np.ndarray, i: int, j: int, kind: str, p: PoseParams,
                  dst_tree: cKDTree | None = None) -> Edge:
    e = Edge(i, j, kind, None, False)
    if len(src) < p.min_overlap_points or len(dst) < p.min_overlap_points:
        e.reasons.append("too few subject points")
        return e
    m, info = search_se2(src, dst, p)
    tree = dst_tree or cKDTree(dst)
    base, d = _overlap(src, tree, m, p.overlap_threshold_m)
    e.transform, e.overlap, e.overlap_points = m, base, int(len(d))
    e.rms_m = float(np.sqrt(np.mean(d ** 2))) if len(d) else None
    e.ambiguity_ratio = info["ambiguity_ratio"]
    ctr = src[:, :2].mean(0)
    rot_about = lambda a: se2(0, *ctr) @ se2(a, 0, 0) @ se2(0, *(-ctr))
    a, s = np.radians(p.sharpness_angle_deg), p.sharpness_shift_m
    drop = lambda mm: base - _overlap(src, tree, mm, p.overlap_threshold_m)[0]
    e.sharpness = {k: (float(v / base) if base > 0 else 0.0) for k, v in {
        "rotation": min(drop(m @ rot_about(a)), drop(m @ rot_about(-a))),
        "shift_x": min(drop(se2(0, s, 0) @ m), drop(se2(0, -s, 0) @ m)),
        "shift_y": min(drop(se2(0, 0, s) @ m), drop(se2(0, 0, -s) @ m)),
    }.items()}

    if info["at_search_limit"]:
        e.reasons.append(f"best turn is at the +-{p.max_step_deg} deg search limit")
    if base < p.min_overlap or e.overlap_points < p.min_overlap_points:
        e.reasons.append(f"overlap {base:.2f} ({e.overlap_points} points) below {p.min_overlap}")
    if e.rms_m is not None and e.rms_m > p.max_rms_m:
        e.reasons.append(f"RMS distance {e.rms_m * 1000:.1f} mm > {p.max_rms_m * 1000:.1f} mm")
    if e.ambiguity_ratio > p.max_ambiguity_ratio:
        e.reasons.append(f"turn angle ambiguous (another angle scores {e.ambiguity_ratio:.2f} of the best)")
    if e.reasons:
        return e

    sigma = max(e.rms_m or p.overlap_threshold_m, 0.002)
    e.information, off = _information(src, tree, m, p, min(e.overlap_points, p.n_eff_cap), sigma)
    e.cost_minimum_offset = {"theta_deg": float(np.degrees(off[0])), "x_m": float(off[1]), "y_m": float(off[2])}
    if not np.any(e.information):
        e.reasons.append("alignment cost is flat in every direction")
        return e
    sd = np.sqrt(np.clip(np.diag(np.linalg.pinv(e.information)), 0, None))
    e.std = {"theta_deg": float(np.degrees(sd[0])), "x_m": float(sd[1]), "y_m": float(sd[2])}
    e.weak_directions = [k for k, bad in (("theta", e.std["theta_deg"] > p.max_view_std_deg),
                                          ("x", e.std["x_m"] > p.max_view_std_m),
                                          ("y", e.std["y_m"] > p.max_view_std_m)) if bad]
    e.ok = True
    return e


# ---- pose graph -------------------------------------------------------------------------------

@dataclass
class PoseResult:
    floor_from_body: list[np.ndarray]  # 4x4 per view (view 0 = identity); EMPTY when status == fail
    se2_params: list[tuple[float, float, float]]
    view_std: list[dict]  # per-view 1-sigma (theta deg, x m, y m) from the pose graph
    edges: list[Edge]
    loop_closure: dict | None
    checks: list[dict]
    status: str  # pass | warn | fail
    params: dict
    algorithm_version: str = ALGORITHM_VERSION


def _check(name, status, value, threshold, message):
    return {"name": name, "status": status, "value": value, "threshold": threshold, "message": message,
            "threshold_status": "proposed; checked on synthetic data only"}


def _chain(by_pair: dict, a: int, b: int, n: int) -> np.ndarray | None:
    """Compose accepted forward edges from a to b (indices modulo n), stepping +1 (sequential) or +2 (skip) so a
    single rejected sequential edge can be bridged. Prefers +1 steps. None if no forward path exists."""
    steps = (b - a) % n
    best: list[np.ndarray | None] = [None] * (steps + 1)
    best[0] = np.eye(3)
    for s in range(1, steps + 1):
        for d in (1, 2):
            if s - d < 0 or best[s - d] is None:
                continue
            e = by_pair.get(((a + s - d) % n, (a + s) % n))
            if e is not None and e.ok:
                best[s] = e.transform @ best[s - d]
                break
    return best[steps]


def estimate_poses(clouds: list[np.ndarray], p: PoseParams | None = None) -> PoseResult:
    """clouds: subject points per view in the FLOOR frame (z up, floor at z = 0), in capture order."""
    p = p or PoseParams()
    n = len(clouds)
    if n < 2:
        raise PoseEstimationError("need at least 2 views")
    trees = [cKDTree(c) for c in clouds]
    edges = [register_pair(clouds[k], clouds[k + 1], k, k + 1, "sequential", p, trees[k + 1]) for k in range(n - 1)]
    by_pair = {(e.i, e.j): e for e in edges}
    seq_only = dict(by_pair)  # skip edges are checked against the SEQUENTIAL chain only (independent evidence)
    checks = []

    def add(i: int, j: int, kind: str) -> None:
        e = register_pair(clouds[i], clouds[j], i, j, kind, p, trees[j])
        if e.ok and kind != "closure":  # the closing edge is judged by the loop-closure error instead
            pred = _chain(seq_only, i, j, n)
            if pred is not None:
                th, tx, ty = se2_params(e.transform @ np.linalg.inv(pred))
                if abs(np.degrees(th)) > p.chain_tolerance_deg or np.hypot(tx, ty) > p.chain_tolerance_m:
                    e.ok = False
                    e.reasons.append(f"disagrees with the sequential chain ({np.degrees(th):.1f} deg, "
                                     f"{np.hypot(tx, ty) * 1000:.0f} mm)")
        edges.append(e)
        by_pair[(i, j)] = e

    if p.skip_edges and n > 3:
        for k in range(n - 2):
            add(k, k + 2, "skip")

    # only a sequence that closes a full turn gets closing edges; a rejected sequential edge may be bridged by a skip edge
    full_turn = False
    chain_all = _chain(by_pair, 0, n - 1, n)
    if chain_all is not None and n > 2:
        # a view bridged by a skip edge has no chain of its own from view 0: leave it out of the running sum
        chained = [c for k in range(1, n) if (c := _chain(by_pair, 0, k, n)) is not None]
        summed = float(np.degrees(np.sum(np.diff(np.unwrap([0.0] + [se2_params(c)[0] for c in chained])))))
        full_turn = abs(360.0 - abs(summed)) <= (p.closure_gap_deg if p.closure_gap_deg is not None else p.max_step_deg)
        checks.append(_check("turn_coverage_deg", "pass" if full_turn else "warn", summed, 360.0,
                             "summed turn from view 0 to the last view" + ("" if full_turn else "; not a full turn, no loop closure")))
    else:
        checks.append(_check("turn_coverage_deg", "warn", None, 360.0, "no forward chain from the first to the last view"))
    if p.loop_closure and full_turn:
        add(n - 1, 0, "closure")
        if p.skip_edges and n > 3:
            add(n - 2, 0, "skip")
            add(n - 1, 1, "skip")
    good = [e for e in edges if e.ok]

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((e.i, e.j) for e in good)
    if not nx.is_connected(g):
        comp = nx.node_connected_component(g, 0)
        missing = [k for k in range(n) if k not in comp]
        checks.append(_check("pose_graph_connected", "fail", len(missing), 0,
                             f"views {missing} have no reliable registration to view 0; they cannot be placed"))
        return PoseResult([], [], [], edges, None, checks, "fail", asdict(p))  # nothing is fabricated
    checks.append(_check("pose_graph_connected", "pass", 0, 0, "every view is linked by reliable edges"))

    init = {0: np.eye(3)}
    for a, b in nx.bfs_edges(g, 0):
        e = by_pair.get((a, b))
        init[b] = e.transform @ init[a] if (e is not None and e.ok) else np.linalg.inv(by_pair[(b, a)].transform) @ init[a]
    x0 = np.array([v for k in range(1, n) for v in se2_params(init[k])])
    sqrt_info = {id(e): np.linalg.cholesky(e.information + np.eye(3) * 1e-9).T for e in good}

    def poses_from(x):
        return [np.eye(3)] + [se2(*x[3 * (k - 1):3 * k]) for k in range(1, n)]

    def edge_error(e, ps):  # left error in floor_j coordinates, matching the information matrix
        return np.array(se2_params(e.transform @ ps[e.i] @ np.linalg.inv(ps[e.j])))

    def residuals(x):
        ps = poses_from(x)
        return np.concatenate([sqrt_info[id(e)] @ edge_error(e, ps) for e in good])

    sol = least_squares(residuals, x0, method="lm" if 3 * len(good) >= len(x0) else "trf", x_scale="jac")
    final = poses_from(sol.x)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    view_std = [{"theta_deg": 0.0, "x_m": 0.0, "y_m": 0.0}]  # view 0 defines the body frame
    for k in range(1, n):
        sd = np.sqrt(np.clip(np.diag(cov)[3 * (k - 1):3 * k], 0, None))
        view_std.append({"theta_deg": float(np.degrees(sd[0])), "x_m": float(sd[1]), "y_m": float(sd[2])})
    for e in good:
        e.post_residual_sigma = float(np.linalg.norm(sqrt_info[id(e)] @ edge_error(e, final)) / np.sqrt(3))

    loop = None
    if p.loop_closure and full_turn:
        closing = by_pair.get((n - 1, 0))
        if closing is not None and closing.ok and chain_all is not None:
            th, tx, ty = se2_params(closing.transform @ chain_all)
            loop = {"rotation_deg": float(np.degrees(abs(th))), "translation_m": float(np.hypot(tx, ty)), "n_edges": n}
    seq_bad = [f"{e.i}->{e.j}" for e in edges if e.kind in ("sequential", "closure") and not e.ok]
    checks.append(_check("rejected_edges", "pass" if not seq_bad else "warn", sum(not e.ok for e in edges), 0,
                         "rejected sequential/closing edges: " + (", ".join(seq_bad) or "none")))
    if p.loop_closure:
        if loop is None:
            checks.append(_check("loop_closure", "warn", None, None, "loop not measurable: not a full turn, or a "
                                 "sequential/closing edge was rejected"))
        else:
            checks.append(_check("loop_closure_rotation", "pass" if loop["rotation_deg"] <= p.max_loop_rotation_deg else "fail",
                                 loop["rotation_deg"], p.max_loop_rotation_deg, "turn error accumulated around the loop (before optimisation)"))
            lim_t = max(p.max_loop_translation_m, p.max_loop_translation_per_edge_m * np.sqrt(n))
            checks.append(_check("loop_closure_translation", "pass" if loop["translation_m"] <= lim_t else "fail",
                                 loop["translation_m"], float(lim_t), "position error accumulated around the loop "
                                 "(before optimisation); limit = max(floor, per-edge x sqrt(n)), random-walk scaling"))
    worst = max((e.post_residual_sigma for e in good), default=0.0)
    checks.append(_check("edge_consistency", "pass" if worst <= p.max_edge_residual_sigma else "warn", worst,
                         p.max_edge_residual_sigma, "largest disagreement of an accepted edge with the optimised poses (sigma units)"))
    uncertain = [k for k, s in enumerate(view_std)
                 if s["x_m"] > p.max_view_std_m or s["y_m"] > p.max_view_std_m or s["theta_deg"] > p.max_view_std_deg]
    worst_pos = max(max(s["x_m"], s["y_m"]) for s in view_std)
    checks.append(_check("view_pose_uncertainty", "pass" if not uncertain else "warn", worst_pos, p.max_view_std_m,
                         "views with pose 1-sigma above limits: " + (str(uncertain) if uncertain else "none")
                         + " (heuristic: dense points are not independent)"))
    th = np.sort(np.degrees(np.array([se2_params(mm)[0] for mm in final])) % 360)
    gap = float(np.max(np.diff(np.concatenate([th, [th[0] + 360]]))))
    checks.append(_check("max_angular_gap_deg", "pass" if gap <= p.max_coverage_gap_deg else "warn", gap,
                         p.max_coverage_gap_deg, "largest turn-angle gap between neighbouring views (coverage)"))
    status = "fail" if any(c["status"] == "fail" for c in checks) else \
        "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    return PoseResult([se2_to_4x4(mm) for mm in final], [se2_params(mm) for mm in final], view_std, edges, loop,
                      checks, status, asdict(p))
