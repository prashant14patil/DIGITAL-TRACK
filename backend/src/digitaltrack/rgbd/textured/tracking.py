"""Subject poses from RGB texture lifted with LiDAR depth, solved as a floor-constrained SE(2) pose graph.

Each node has SIFT features on the subject's pixels with a 3-D position in the floor frame (z = gravity up).
A person standing and turning has 3 unknowns per node: turn angle and floor position. For a pair i -> j
(floor coordinates of node i -> floor coordinates of node j):

  1. mutual ratio-test matches, 2-point RANSAC on SE(2) with a 3-D distance threshold
  2. refinement with both reprojection errors (RGB pixels, sub-pixel) and the 3-D error (LiDAR), robust loss
  3. 3x3 information from the Jacobian, with the effective point count capped (neighbouring matches are not
     independent)

Pairs: neighbours in time (1, 2, 3 apart) plus loop pairs far apart in time that face a similar way. The loop
error is reported BEFORE optimisation. No ICP and no depth-shape registration moves any pose.
All thresholds are PROPOSED engineering defaults; none is validated against a physical reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ...dataset.loader import Intrinsics
from ..pose import se2, se2_params, se2_to_4x4, wrap

ALGORITHM_VERSION = "textured-se2-posegraph-0.1"


@dataclass(frozen=True)
class TrackingParams:
    interval_s: float = 0.5  # tracking-node spacing; pause views are added as nodes too
    neighbour_offsets: tuple[int, ...] = (1, 2, 3, 6, 12)  # longer baselines limit accumulated drift
    pause_offsets: tuple[int, ...] = (1, 2)  # direct pairs between the pause views (the ones that are fused)
    ratio: float = 0.8
    ransac_threshold_m: float = 0.02
    ransac_iterations: int = 400
    min_inliers: int = 30
    min_inlier_fraction: float = 0.2
    sigma_px: float = 2.0
    sigma_3d_m: float = 0.01
    max_effective_matches: int = 60
    max_step_deg: float = 60.0  # a neighbour pair turning more than this is rejected
    loop_min_gap_fraction: float = 0.4  # loop pairs: at least this far apart (fraction of the nodes)
    loop_search_deg: float = 60.0  # ... and predicted headings this close (wide: the chain drifts)
    loop_stride: int = 3
    sift_contrast: float = 0.02
    sift_max_features: int = 1500
    seed: int = 0


@dataclass
class Features:
    points: np.ndarray  # (N, 3) floor frame
    uv: np.ndarray  # (N, 2) rgb pixels
    desc: np.ndarray  # (N, 128)
    camera: Intrinsics  # rgb intrinsics
    camera_from_floor: np.ndarray  # 4x4


@dataclass
class PairEstimate:
    i: int
    j: int
    kind: str  # sequential | pause | loop
    transform: np.ndarray  # 3x3 SE2, floor_i -> floor_j
    information: np.ndarray  # 3x3 for (dtheta, dx, dy)
    inliers: int
    matches: int
    rms_3d_m: float
    median_px: float

    def summary(self) -> dict:
        th, x, y = se2_params(self.transform)
        return {"i": self.i, "j": self.j, "kind": self.kind, "theta_deg": float(np.degrees(th)), "tx_m": x, "ty_m": y,
                "inliers": self.inliers, "matches": self.matches, "rms_3d_mm": self.rms_3d_m * 1000,
                "median_reprojection_px": self.median_px}


def extract_features(gray: np.ndarray, mask: np.ndarray, depth_m: np.ndarray, camera: Intrinsics,
                     floor_from_camera: np.ndarray, p: TrackingParams) -> Features:
    """SIFT inside the subject mask; depth is bilinear and only accepted where the 4 neighbours are consistent."""
    dh, dw = depth_m.shape
    empty = Features(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 128), np.float32), camera,
                     np.linalg.inv(floor_from_camera))
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    if not inner.any():
        return empty
    big = cv2.resize(inner * 255, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    x, y, w, h = cv2.boundingRect(big)
    kp, desc = cv2.SIFT_create(nfeatures=p.sift_max_features, contrastThreshold=p.sift_contrast).detectAndCompute(gray[y:y + h, x:x + w],
                                                                                  big[y:y + h, x:x + w])
    if not kp:
        return empty
    uv = np.array([k.pt for k in kp]) + [x, y]
    du = uv * [dw / camera.width, dh / camera.height]
    x0 = np.clip(np.floor(du[:, 0]).astype(int), 0, dw - 2)
    y0 = np.clip(np.floor(du[:, 1]).astype(int), 0, dh - 2)
    fx = np.clip(du[:, 0] - x0, 0, 1)
    fy = np.clip(du[:, 1] - y0, 0, 1)
    q = np.stack([depth_m[y0, x0], depth_m[y0, x0 + 1], depth_m[y0 + 1, x0], depth_m[y0 + 1, x0 + 1]], 1).astype(float)
    ok = (q > 0).all(1) & (np.ptp(q, axis=1) < 0.05)
    z = (q[:, 0] * (1 - fx) * (1 - fy) + q[:, 1] * fx * (1 - fy) + q[:, 2] * (1 - fx) * fy + q[:, 3] * fx * fy)
    pc = np.stack([(uv[:, 0] - camera.cx) / camera.fx * z, (uv[:, 1] - camera.cy) / camera.fy * z, z], 1)
    pf = pc @ floor_from_camera[:3, :3].T + floor_from_camera[:3, 3]
    return Features(pf[ok], uv[ok], desc[ok].astype(np.float32), camera, np.linalg.inv(floor_from_camera))


def _fit_se2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ca, cb = a[:, :2].mean(0), b[:, :2].mean(0)
    x, y = a[:, :2] - ca, b[:, :2] - cb
    th = np.arctan2(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]), np.sum(x * y))
    t = cb - se2(th, 0, 0)[:2, :2] @ ca
    return se2(th, *t)


def _move(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, :2] = points[:, :2] @ m[:2, :2].T + m[:2, 2]
    return out


def _project(points_floor: np.ndarray, f: Features) -> np.ndarray:
    pc = points_floor @ f.camera_from_floor[:3, :3].T + f.camera_from_floor[:3, 3]
    z = np.maximum(pc[:, 2], 0.05)
    return np.stack([f.camera.fx * pc[:, 0] / z + f.camera.cx, f.camera.fy * pc[:, 1] / z + f.camera.cy], 1)


def match_features(a: Features, b: Features, ratio: float) -> np.ndarray:
    """(M, 2) indices of mutual nearest neighbours that pass the ratio test."""
    if len(a.desc) < 2 or len(b.desc) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher()
    fwd = {m.queryIdx: m.trainIdx for m, n in bf.knnMatch(a.desc, b.desc, k=2) if m.distance < ratio * n.distance}
    back = {m.queryIdx: m.trainIdx for m, n in bf.knnMatch(b.desc, a.desc, k=2) if m.distance < ratio * n.distance}
    return np.array([(i, j) for i, j in fwd.items() if back.get(j) == i], int).reshape(-1, 2)


def estimate_pair(a: Features, b: Features, i: int, j: int, kind: str, p: TrackingParams) -> PairEstimate | None:
    m = match_features(a, b, p.ratio)
    if len(m) < p.min_inliers:
        return None
    pa, pb = a.points[m[:, 0]], b.points[m[:, 1]]
    ua, ub = a.uv[m[:, 0]], b.uv[m[:, 1]]
    rng = np.random.default_rng(p.seed + 100003 * i + j)
    s = rng.integers(0, len(m), (p.ransac_iterations, 2))
    s = s[s[:, 0] != s[:, 1]]
    best = np.zeros(len(m), bool)
    for pick in s:
        t = _fit_se2(pa[pick], pb[pick])
        inl = np.linalg.norm(_move(pa, t) - pb, axis=1) < p.ransac_threshold_m
        if inl.sum() > best.sum():
            best = inl
    for _ in range(3):
        if best.sum() < 2:
            return None
        t = _fit_se2(pa[best], pb[best])
        best = np.linalg.norm(_move(pa, t) - pb, axis=1) < p.ransac_threshold_m
    n = int(best.sum())
    if n < p.min_inliers or n < p.min_inlier_fraction * len(m):
        return None
    qa, qb, va, vb = pa[best], pb[best], ua[best], ub[best]
    # parametrised about the source centroid, so that the angle and shift are not artificially coupled
    c = qa[:, :2].mean(0)
    th0 = se2_params(_fit_se2(qa, qb))[0]
    shift0 = qb[:, :2].mean(0) - c

    def to_matrix(x):
        return se2(0, *(c + x[1:])) @ se2(x[0], 0, 0) @ se2(0, *(-c))

    def residuals(x):
        t = to_matrix(x)
        fwd = _move(qa, t)
        back = _move(qb, np.linalg.inv(t))
        return np.concatenate([((_project(fwd, b) - vb) / p.sigma_px).ravel(),
                               ((_project(back, a) - va) / p.sigma_px).ravel(),
                               ((fwd - qb) / p.sigma_3d_m).ravel()])

    sol = least_squares(residuals, np.r_[th0, shift0], loss="cauchy", x_scale=[0.05, 0.02, 0.02])
    t = to_matrix(sol.x)
    th, tx, ty = se2_params(t)
    # information for (dtheta, dtx, dty) of `t`: numerical Jacobian in that parametrisation
    base = residuals(sol.x)
    jac = np.empty((len(base), 3))
    for col, step in enumerate((1e-4, 1e-4, 1e-4)):
        d = [th, tx, ty]
        d[col] += step
        tm = se2(*d)
        jac[:, col] = (residuals(np.r_[d[0], tm[:2, :2] @ c + tm[:2, 2] - c]) - base) / step
    w = 1.0 / (1.0 + base ** 2)  # Cauchy weights at the solution
    info = (jac * w[:, None]).T @ jac * min(1.0, p.max_effective_matches / n)
    moved = _move(qa, t)
    return PairEstimate(i, j, kind, t, info, n, int(len(m)),
                        float(np.sqrt(np.mean(np.sum((moved - qb) ** 2, axis=1)))),
                        float(np.median(np.linalg.norm(_project(moved, b) - vb, axis=1))))


@dataclass
class PoseSolution:
    floor_from_body: dict[int, np.ndarray]  # node -> 4x4; only nodes linked to node 0
    se2: dict[int, tuple[float, float, float]]
    edges: list[PairEstimate]
    unplaced: list[int]
    checks: dict
    algorithm_version: str = ALGORITHM_VERSION


def solve_pose_graph(n: int, edges: list[PairEstimate]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Optimise SE2 floor_from_body for the nodes linked to node 0. -> (poses, per-edge whitened residual norm)."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for k, e in enumerate(edges):
        if not g.has_edge(e.i, e.j):
            g.add_edge(e.i, e.j, k=k)
    comp = nx.node_connected_component(g, 0)
    use = [e for e in edges if e.i in comp and e.j in comp]
    init = {0: np.eye(3)}
    # initialise along sequential edges first: loop edges must not seed the chain
    order = nx.Graph()
    order.add_nodes_from(comp)
    for e in use:
        w = abs(e.i - e.j) + (1000 if e.kind == "loop" else 0)
        if not order.has_edge(e.i, e.j) or order[e.i][e.j]["weight"] > w:
            order.add_edge(e.i, e.j, weight=w, e=e)
    tree = nx.minimum_spanning_tree(order)
    for a, b in nx.bfs_edges(tree, 0):
        e = tree[a][b]["e"]
        init[b] = (e.transform if e.i == a else np.linalg.inv(e.transform)) @ init[a]
    nodes = sorted(k for k in comp if k)
    col = {k: 3 * c for c, k in enumerate(nodes)}
    if not nodes:
        return {0: init[0]}, np.zeros(0)
    x0 = np.array([v for k in nodes for v in se2_params(init[k])])
    sqrt_info = [np.linalg.cholesky(e.information + np.eye(3) * 1e-6).T for e in use]

    def pose(x, k):
        return np.eye(3) if k == 0 else se2(*x[col[k]:col[k] + 3])

    def residuals(x):
        out = np.empty(3 * len(use))
        for r, e in enumerate(use):
            th, tx, ty = se2_params(e.transform @ pose(x, e.i) @ np.linalg.inv(pose(x, e.j)))
            out[3 * r:3 * r + 3] = sqrt_info[r] @ np.array([float(wrap(th)), tx, ty])
        return out

    sparsity = lil_matrix((3 * len(use), len(x0)), dtype=int)
    for r, e in enumerate(use):
        for k in (e.i, e.j):
            if k:
                sparsity[3 * r:3 * r + 3, col[k]:col[k] + 3] = 1
    # plain least squares first: it spreads accumulated drift over the loop instead of discarding the loop edges
    sol = least_squares(residuals, x0, jac_sparsity=sparsity, x_scale="jac", max_nfev=200)
    sol = least_squares(residuals, sol.x, jac_sparsity=sparsity, x_scale="jac", loss="cauchy", f_scale=3.0, max_nfev=200)
    poses = {0: np.eye(3), **{k: pose(sol.x, k) for k in nodes}}
    res = residuals(sol.x).reshape(-1, 3)
    full = np.full(len(edges), np.nan)
    idx = [k for k, e in enumerate(edges) if e.i in comp and e.j in comp]
    full[idx] = np.linalg.norm(res, axis=1)
    return poses, full


def _chain_yaw(n: int, edges: list[PairEstimate]) -> tuple[np.ndarray, list[int]]:
    """Heading of every node from neighbour pairs only (widest-step fallback bridges a missing pair), unwrapped."""
    by = {(e.i, e.j): e for e in edges if e.kind == "sequential"}
    yaw = np.full(n, np.nan)
    yaw[0] = 0.0
    broken = []
    for k in range(1, n):
        for d in range(1, 4):
            e = by.get((k - d, k))
            if e is not None and np.isfinite(yaw[k - d]):
                yaw[k] = yaw[k - d] + se2_params(e.transform)[0]
                break
        if not np.isfinite(yaw[k]):
            broken.append(k)
            yaw[k] = yaw[k - 1]
    return yaw, broken


def track(features: list[Features], p: TrackingParams | None = None, *, pause_nodes: list[int] | None = None) -> PoseSolution:
    """features: one entry per node in time order. Node 0 defines the body frame."""
    p = p or TrackingParams()
    n = len(features)
    edges: list[PairEstimate] = []
    rejected = 0
    pairs = {(i, i + d): "sequential" for i in range(n) for d in p.neighbour_offsets if i + d < n}
    pauses = sorted(pause_nodes or [])
    for d in p.pause_offsets:
        for a, b in zip(pauses, pauses[d:]):
            pairs.setdefault((a, b), "pause")
    for (i, j), kind in sorted(pairs.items()):
        e = estimate_pair(features[i], features[j], i, j, kind, p)
        if e is None or abs(np.degrees(se2_params(e.transform)[0])) > p.max_step_deg:
            rejected += 1
            continue
        edges.append(e)
    yaw, broken = _chain_yaw(n, edges)
    gap = max(max(p.neighbour_offsets) + 1, int(p.loop_min_gap_fraction * n))
    for i in range(0, n, p.loop_stride):
        for j in range(n - 1, i + gap - 1, -p.loop_stride):
            if abs(np.degrees(float(wrap(yaw[j] - yaw[i])))) <= p.loop_search_deg:
                e = estimate_pair(features[i], features[j], i, j, "loop", p)
                if e is not None:
                    edges.append(e)
    poses, resid = solve_pose_graph(n, edges)
    loops = [e for e in edges if e.kind == "loop"]
    # loop error BEFORE optimisation: neighbour-only chain heading against each direct loop measurement
    errs = [abs(np.degrees(float(wrap(se2_params(e.transform)[0] - (yaw[e.j] - yaw[e.i]))))) for e in loops]
    ang = np.array([e_[0] for e_ in (se2_params(m) for _, m in sorted(poses.items()))])
    total = float(np.degrees(np.sum(wrap(np.diff(ang))))) if len(ang) > 1 else 0.0
    r_deg, r_mm = [], []
    for e in edges:
        if e.i in poses and e.j in poses:
            th, tx, ty = se2_params(e.transform @ poses[e.i] @ np.linalg.inv(poses[e.j]))
            r_deg.append(abs(np.degrees(float(wrap(th)))))
            # how far the two pose chains place the same body point (the subject's centre in view j)
            centre = features[e.j].points[:, :2].mean(0) if len(features[e.j].points) else np.zeros(2)
            body = np.linalg.inv(poses[e.j]) @ np.r_[centre, 1.0]
            r_mm.append(1000 * float(np.linalg.norm(((e.transform @ poses[e.i] - poses[e.j]) @ body)[:2])))
    finite = resid[np.isfinite(resid)]
    checks = {
        "nodes": n, "nodes_placed": len(poses), "edges_sequential": sum(e.kind == "sequential" for e in edges),
        "edges_pause": sum(e.kind == "pause" for e in edges), "edges_loop": len(loops), "pairs_rejected": rejected, "chain_breaks": broken,
        "chain_only_turn_deg": float(np.degrees(yaw[-1] - yaw[0])), "optimised_net_turn_deg": total,
        "loop_error_before_optimisation_deg": None if not errs else {"median": float(np.median(errs)), "max": float(np.max(errs))},
        "edge_residual_after_deg": None if not r_deg else {"median": float(np.median(r_deg)), "p95": float(np.percentile(r_deg, 95))},
        "edge_residual_after_mm_at_subject": None if not r_mm else {"median": float(np.median(r_mm)), "p95": float(np.percentile(r_mm, 95))},
        "edges_downweighted_fraction": float(np.mean(finite > 3 * 3.0)) if finite.size else None,
        "median_inliers": float(np.median([e.inliers for e in edges])) if edges else 0.0,
        "median_reprojection_px": float(np.median([e.median_px for e in edges])) if edges else None,
    }
    return PoseSolution({k: se2_to_4x4(m) for k, m in poses.items()}, {k: se2_params(m) for k, m in poses.items()},
                        edges, [k for k in range(n) if k not in poses], checks)
