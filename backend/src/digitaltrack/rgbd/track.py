"""Subject poses from RGB features lifted to 3-D by confidence-filtered depth, solved as a loop-closing pose graph.

Why this exists (first real Stray Scanner export, 2026-11-25): the iPad was fixed (ARKit odometry moved < 2 cm and
< 0.3 deg over 112 s), so the device poses say nothing about the person, who turned in place. The shape-only
SE(2) correlation registration (`rgbd.pose`) never closed the loop on that recording (best: 48 deg closure error):
a clothed torso is smooth and nearly symmetric, and the arms move between views. The RGB video, however, is
pixel-registered with the depth (ARKit scene depth is produced in the colour camera's frame; verified by overlay)
and a printed shirt carries hundreds of stable features per frame.

Per view (subject pixels only):
  SIFT keypoints in the RGB frame inside the (dilated) subject mask -> depth at the corresponding depth pixel,
  kept only if confidence >= `min_confidence`, the pixel is inside the subject mask and its 3x3 depth neighbourhood
  spans < `max_local_depth_range_m` (no silhouette / mixed pixels) -> 3-D point in the FLOOR frame.
Per pair (i, j): ratio-test matching -> 6-DOF RANSAC (Kabsch) -> refit on inliers. QC: inlier count, inlier RMS,
  rotation size. A rejected pair contributes nothing; poses are never invented.
Graph: edges at several frame offsets (multi-baseline: long baselines carry the angle with better SNR) + loop
  closures between views whose current yaw estimates are within `loop_window_deg` (mod 360) but far apart in time
  -> Open3D global pose-graph optimisation (Levenberg-Marquardt with line-process pruning of inconsistent
  non-sequential edges).

Convention: node pose k = body_from_floor_k (maps subject points observed in view k, floor coordinates, into the
body frame = floor frame of view 0). Edge (i -> j) transformation T_ij maps view-i floor points onto view j:
X_j = T_ij X_i, hence inv(pose_j) @ pose_i = T_ij (Open3D's convention).

All thresholds are PROPOSED values, tuned on ONE real recording. They are logged with every run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import open3d as o3d

ALGORITHM_VERSION = "rgbd-feature-posegraph-0.1"


@dataclass(frozen=True)
class TrackParams:
    feature_width: int = 960  # RGB is resized to this width for feature detection
    sift_features: int = 4000
    sift_contrast: float = 0.02
    ratio: float = 0.75
    min_confidence: int = 2  # Stray confidence 0..2; only the highest level lifts a feature to 3-D
    max_local_depth_range_m: float = 0.03
    ransac_threshold_m: float = 0.015
    ransac_iterations: int = 400
    min_inliers: int = 30
    max_rms_m: float = 0.010
    max_rotation_deg: float = 60.0
    offsets: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21)
    loop_stride: int = 4  # loop-closure candidates: every n-th view on both sides
    loop_min_separation: int = 120  # views apart in time
    loop_window_deg: float = 45.0
    loop_max_candidates: int = 4000
    max_correspondence_distance_m: float = 0.015
    edge_prune_threshold: float = 0.25
    preference_loop_closure: float = 1.0
    # "yaw_xyz" (default): the subject turns about the VERTICAL (gravity) axis of the floor frame and may shift in
    # x, y, z: 4 DOF per view, 2-point RANSAC, robust sparse least-squares pose graph. Physically motivated: on the
    # first real recording the free 6-DOF solution's rotation axis agreed with IMU gravity within 0.5-0.7 deg.
    # "se3": free 6-DOF per edge + Open3D global optimisation (kept for comparison).
    model: str = "yaw_xyz"
    min_pair_spread_m: float = 0.05  # 2-point RANSAC samples closer than this (horizontally) are skipped
    robust_scale: float = 2.0  # Huber threshold of the normalised edge residuals
    nominal_sigma_m: float = 0.003  # residual of this size (at 100 inliers) = 1 unit
    n_eff_cap: int = 300  # dense features of one surface are not independent
    reject_rotation_deg: float = 3.0  # after the robust solve, edges disagreeing more than this are dropped
    reject_translation_m: float = 0.03
    seed: int = 0


@dataclass
class ViewFeatures:
    view_index: int
    points: np.ndarray  # (N, 3) floor frame
    descriptors: np.ndarray | None  # (N, 128) float32
    n_keypoints: int  # before depth filtering


@dataclass
class PairResult:
    i: int
    j: int
    kind: str  # sequential | skip | loop
    ok: bool
    transform: np.ndarray | None = None  # 4x4, view-i floor -> view-j floor
    information: np.ndarray | None = None
    matches: int = 0
    inliers: int = 0
    rms_m: float | None = None
    rotation_deg: float | None = None
    lever_m: float | None = None  # RMS horizontal radius of the inliers (turns an angle error into metres)
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("transform", "information")}
        if self.transform is not None:
            d["yaw_deg"] = float(np.degrees(np.arctan2(self.transform[1, 0], self.transform[0, 0])))
            d["translation_m"] = [float(x) for x in self.transform[:3, 3]]
        return d


@dataclass
class TrackResult:
    body_from_floor: dict[int, np.ndarray]  # position in the input list -> 4x4; views without a pose are absent
    edges: list[PairResult]
    checks: list[dict]
    status: str
    yaw_deg: dict[int, float]  # unwrapped turn angle per posed view
    log: dict
    algorithm_version: str = ALGORITHM_VERSION


# ---- features -------------------------------------------------------------------------------------------------

def extract_features(rgb: np.ndarray, depth_m: np.ndarray, confidence: np.ndarray | None, mask: np.ndarray,
                     k: tuple[float, float, float, float], floor_from_cam: np.ndarray, p: TrackParams,
                     sift=None, view_index: int = -1) -> ViewFeatures:
    """`k` = depth-image intrinsics (fx, fy, cx, cy). The RGB image must cover the same field of view as the depth
    image (checked by aspect ratio); pixel centres are mapped by scaling."""
    dh, dw = depth_m.shape
    h0, w0 = rgb.shape[:2]
    if abs(w0 / h0 - dw / dh) > 0.01:
        raise ValueError(f"RGB {w0}x{h0} and depth {dw}x{dh} have different aspect ratios")
    w = p.feature_width
    h = int(round(h0 * w / w0))
    img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA) if w != w0 else rgb
    s = w / dw
    sift = sift or cv2.SIFT_create(nfeatures=p.sift_features, contrastThreshold=p.sift_contrast)
    m_rgb = cv2.resize(cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)), (w, h),
                       interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    kp, des = sift.detectAndCompute(gray, m_rgb)
    if not kp or des is None:
        return ViewFeatures(view_index, np.zeros((0, 3)), None, 0)
    uv = np.array([q.pt for q in kp]) / s - 0.5 + 0.5 / s  # RGB pixel centre -> depth pixel coordinates
    du = np.clip(np.round(uv[:, 0]).astype(int), 0, dw - 1)
    dv = np.clip(np.round(uv[:, 1]).astype(int), 0, dh - 1)
    ok = mask[dv, du] & np.isfinite(depth_m[dv, du]) & (depth_m[dv, du] > 0)
    if confidence is not None:
        ok &= confidence[dv, du] >= p.min_confidence
    pad = np.pad(depth_m, 1, mode="edge")
    win = np.stack([pad[dv + a, du + b] for a in range(3) for b in range(3)])
    ok &= (win.max(0) - win.min(0)) < p.max_local_depth_range_m
    z = depth_m[dv, du].astype(np.float64)
    fx, fy, cx, cy = k
    pts = np.stack([(uv[:, 0] - cx) / fx * z, (uv[:, 1] - cy) / fy * z, z], axis=1)
    pts = pts @ floor_from_cam[:3, :3].T + floor_from_cam[:3, 3]
    return ViewFeatures(view_index, pts[ok], des[ok].astype(np.float32), len(kp))


# ---- pairwise registration ------------------------------------------------------------------------------------

def _kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def _yaw_fit(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation about z + translation in x, y, z mapping a onto b."""
    ca, cb = a.mean(0), b.mean(0)
    a0, b0 = a[:, :2] - ca[:2], b[:, :2] - cb[:2]
    th = np.arctan2(np.sum(a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]), np.sum(a0[:, 0] * b0[:, 0] + a0[:, 1] * b0[:, 1]))
    r = _rz(th)
    return r, cb - r @ ca


def _rz(th: float) -> np.ndarray:
    c, s_ = np.cos(th), np.sin(th)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _information(target_pts: np.ndarray) -> np.ndarray:
    """Point-to-point information matrix in Open3D's (rotation, translation) ordering, one row block per point."""
    x, y, z = target_pts.T
    g = np.zeros((len(target_pts), 3, 6))
    g[:, 0, 1], g[:, 0, 2], g[:, 0, 3] = z, -y, 1
    g[:, 1, 0], g[:, 1, 2], g[:, 1, 4] = -z, x, 1
    g[:, 2, 0], g[:, 2, 1], g[:, 2, 5] = y, -x, 1
    return np.einsum("nki,nkj->ij", g, g)


_BF = cv2.BFMatcher(cv2.NORM_L2)


def register_features(a: ViewFeatures, b: ViewFeatures, i: int, j: int, kind: str, p: TrackParams) -> PairResult:
    r = PairResult(i, j, kind, False)
    if a.descriptors is None or b.descriptors is None or len(a.points) < p.min_inliers or len(b.points) < p.min_inliers:
        r.reasons.append("too few 3-D features")
        return r
    knn = _BF.knnMatch(a.descriptors, b.descriptors, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < p.ratio * m[1].distance]
    r.matches = len(good)
    if len(good) < p.min_inliers:
        r.reasons.append(f"{len(good)} descriptor matches < {p.min_inliers}")
        return r
    pa = a.points[[m.queryIdx for m in good]]
    pb = b.points[[m.trainIdx for m in good]]
    rng = np.random.default_rng(p.seed + 7919 * i + j)
    fit = _kabsch if p.model == "se3" else _yaw_fit
    k_min = 3 if p.model == "se3" else 2
    best = None
    for _ in range(p.ransac_iterations):
        s = rng.choice(len(pa), k_min, replace=False)
        if k_min == 2 and np.hypot(*(pa[s[1], :2] - pa[s[0], :2])) < p.min_pair_spread_m:
            continue
        rot, t = fit(pa[s], pb[s])
        inl = np.linalg.norm(pa @ rot.T + t - pb, axis=1) < p.ransac_threshold_m
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < k_min:
        r.reasons.append("RANSAC found no model")
        return r
    rot, t = fit(pa[best], pb[best])
    for _ in range(3):
        inl = np.linalg.norm(pa @ rot.T + t - pb, axis=1) < p.ransac_threshold_m
        if inl.sum() < k_min:
            break
        rot, t = fit(pa[inl], pb[inl])
    res = np.linalg.norm(pa[inl] @ rot.T + t - pb[inl], axis=1)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, t
    r.transform, r.inliers = m, int(inl.sum())
    r.rms_m = float(np.sqrt(np.mean(res ** 2))) if len(res) else None
    r.rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    if r.inliers < p.min_inliers:
        r.reasons.append(f"{r.inliers} RANSAC inliers < {p.min_inliers}")
    if r.rms_m is not None and r.rms_m > p.max_rms_m:
        r.reasons.append(f"inlier RMS {r.rms_m * 1000:.1f} mm > {p.max_rms_m * 1000:.0f} mm")
    if r.rotation_deg > p.max_rotation_deg:
        r.reasons.append(f"rotation {r.rotation_deg:.0f} deg > {p.max_rotation_deg:.0f} deg")
    if not r.reasons:
        r.ok = True
        r.information = _information(pb[inl])
        q = pb[inl]
        r.lever_m = float(np.sqrt(np.mean(np.sum((q[:, :2] - q[:, :2].mean(0)) ** 2, axis=1))))
    return r


# ---- pose graph -----------------------------------------------------------------------------------------------

def _yaw(m: np.ndarray) -> float:
    return float(np.arctan2(m[1, 0], m[0, 0]))


def _chain_init(g, root: int, start: dict[int, np.ndarray] | None = None) -> dict[int, np.ndarray]:
    """Initial poses by chaining accepted edges IN TIME ORDER, preferring the shortest time offset (the most
    reliable edge), so the start differs from the solution only by slow drift. (A BFS spanning tree mixes long
    skip edges from different branches; on real data the solver then left junction residuals every 21 views.)
    Nodes not reachable from an earlier node are placed by a BFS pass afterwards."""
    import networkx as nx

    init = dict(start or {})
    init.setdefault(root, np.eye(4))
    comp = sorted(nx.node_connected_component(g, root))
    for n in comp:
        if n in init:
            continue
        prev = sorted((abs(n - m), m) for m in g.neighbors(n) if m in init and m < n)
        if prev:
            m = prev[0][1]
            e = g.edges[m, n]["e"]
            init[n] = init[m] @ np.linalg.inv(e.transform) if e.i == m else init[m] @ e.transform
    for a, b in nx.bfs_edges(g, root):
        if b not in init:
            e = g.edges[a, b]["e"]
            init[b] = init[a] @ np.linalg.inv(e.transform) if e.i == a else init[a] @ e.transform
    return {n: init[n] for n in comp}


def _optimise(nodes: list[int], edges: list[PairResult], p: TrackParams,
              solver_log: list | None = None) -> dict[int, np.ndarray]:
    """Spanning-tree initialisation from accepted edges, then Open3D global optimisation. Returns poses for the
    connected component of nodes[0] only."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(nodes)
    for e in edges:
        if e.ok:
            g.add_edge(e.i, e.j, e=e)
    init = _chain_init(g, nodes[0])  # inv(pose_j) @ pose_i = T_ij
    order = sorted(init)
    pos = {n: k for k, n in enumerate(order)}
    pg = o3d.pipelines.registration.PoseGraph()
    for n in order:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(init[n]))
    for e in edges:
        if e.ok and e.i in pos and e.j in pos:
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                pos[e.i], pos[e.j], e.transform, e.information, uncertain=e.kind != "sequential"))
    opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=p.max_correspondence_distance_m, edge_prune_threshold=p.edge_prune_threshold,
        preference_loop_closure=p.preference_loop_closure, reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        o3d.pipelines.registration.global_optimization(
            pg, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), opt)
    kept = {(order[e.source_node_id], order[e.target_node_id]) for e in pg.edges}
    for e in edges:  # Open3D drops pruned edges from the graph: mark them
        if e.ok and e.kind != "sequential" and e.i in pos and e.j in pos and (e.i, e.j) not in kept:
            e.ok = False
            e.reasons.append("pruned by the pose graph (inconsistent with the other edges)")
    return {n: np.asarray(pg.nodes[pos[n]].pose) for n in order}


def _optimise_yaw(nodes: list[int], edges: list[PairResult], p: TrackParams,
                  solver_log: list | None = None) -> dict[int, np.ndarray]:
    """Solve, flag edges that still disagree beyond reject_rotation_deg / reject_translation_m (never so many that
    a view loses its connection to the graph), and solve once more without them: under a convex loss a gross
    outlier keeps a small linear pull, so it is removed rather than only down-weighted."""
    import networkx as nx

    poses, info = _irls_yaw(nodes, edges, p)
    flagged = []
    for e in edges:
        if e.ok and e.i in poses and e.j in poses:
            ang, tr = _edge_residual(e, poses)
            if ang > p.reject_rotation_deg or tr > p.reject_translation_m:
                flagged.append((max(ang / p.reject_rotation_deg, tr / p.reject_translation_m), e, ang, tr))
    for _, e, _, _ in flagged:
        e.ok = False
    g = nx.Graph()
    g.add_nodes_from(poses)
    g.add_edges_from((e.i, e.j) for e in edges if e.ok and e.i in poses and e.j in poses)
    restored = 0
    for _, e, _, _ in sorted(flagged, key=lambda f: f[0]):  # least-bad first
        if not nx.has_path(g, e.i, nodes[0]) or not nx.has_path(g, e.j, nodes[0]):
            e.ok = True
            g.add_edge(e.i, e.j)
            restored += 1
    n_out = 0
    for _, e, ang, tr in flagged:
        if not e.ok:
            e.reasons.append(f"outlier: disagrees with the optimised pose graph ({ang:.1f} deg, {tr * 1000:.0f} mm)")
            n_out += 1
    if n_out:
        poses, info2 = _irls_yaw(nodes, edges, p, start=poses)
        info["second_solve"] = info2
    if solver_log is not None:
        solver_log.append({**info, "outliers_removed": n_out, "kept_for_connectivity": restored})
    return poses


def _irls_yaw(nodes: list[int], edges: list[PairResult], p: TrackParams,
              start: dict[int, np.ndarray] | None = None) -> tuple[dict[int, np.ndarray], dict]:
    """Robust 4-DOF pose graph (yaw, x, y, z per view; nodes[0] fixed), solved by decoupled IRLS.

    For a rotation about one fixed axis the problem separates:
      yaw:          theta_i - theta_j = phi_ij  (+ 2 pi k, k chosen from the current estimate)   -> linear
      translation:  t_i - t_j = Rz(theta_j) s_ij                     (given the yaws)               -> linear
    Each edge is weighted by its own quality, sqrt(min(inliers, cap) / 100) / max(rms, nominal_sigma), with the
    yaw residual expressed in metres through the edge's lever arm. Huber IRLS (convex; threshold `robust_scale` in those
    units) down-weights inconsistent edges."""
    import networkx as nx
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    g = nx.Graph()
    g.add_nodes_from(nodes)
    for e in edges:
        if e.ok:
            g.add_edge(e.i, e.j, e=e)
    init = _chain_init(g, nodes[0], start)
    comp = sorted(init)
    free = [n for n in comp if n != nodes[0]]
    col = {n: k for k, n in enumerate(free)}
    act = [e for e in edges if e.ok and e.i in init and e.j in init]
    if not free or not act:
        return init, {"edges": len(act), "nodes": len(comp)}
    ci = np.array([col.get(e.i, -1) for e in act])
    cj = np.array([col.get(e.j, -1) for e in act])
    w0 = np.array([np.sqrt(min(e.inliers, p.n_eff_cap) / 100.0) / max(e.rms_m or p.ransac_threshold_m, p.nominal_sigma_m)
                   for e in act])
    lev = np.array([e.lever_m or 0.1 for e in act])
    phi = np.array([_yaw(e.transform) for e in act])
    s_ij = np.array([e.transform[:3, 3] for e in act])
    ne, nf = len(act), len(free)

    # unwrapped initial yaws (chain init is continuous in time order)
    th = np.zeros(nf)
    prev = 0.0
    for n in free:
        y = _yaw(init[n])
        prev = prev + ((y - prev + np.pi) % (2 * np.pi) - np.pi)
        th[col[n]] = prev
    t = np.array([init[n][:3, 3] for n in free])

    def a_matrix(wts):
        r = np.concatenate([np.arange(ne)[ci >= 0], np.arange(ne)[cj >= 0]])
        c = np.concatenate([ci[ci >= 0], cj[cj >= 0]])
        v = np.concatenate([np.ones((ci >= 0).sum()), -np.ones((cj >= 0).sum())])
        a = coo_matrix((v, (r, c)), shape=(ne, nf)).tocsr()
        return a, a.T.multiply(wts).tocsr()  # A, A^T W

    def get(arr, idx, fill):
        return np.where((idx >= 0)[(...,) + (None,) * (arr.ndim - 1)], arr[np.maximum(idx, 0)], fill)

    robust = np.ones(ne)
    it = 0
    for it in range(60):
        # --- yaw step
        thi, thj = get(th, ci, 0.0), get(th, cj, 0.0)
        k = np.round((thi - thj - phi) / (2 * np.pi))
        b = phi + 2 * np.pi * k
        wy = (w0 * lev) ** 2 * robust
        a, atw = a_matrix(wy)
        th_new = spsolve((atw @ a).tocsc(), atw @ b)
        # --- translation step (given yaws): t_i - t_j = Rz(theta_j) s_ij
        thj = get(th_new, cj, 0.0)
        c, sn = np.cos(thj), np.sin(thj)
        rhs = np.stack([c * s_ij[:, 0] - sn * s_ij[:, 1], sn * s_ij[:, 0] + c * s_ij[:, 1], s_ij[:, 2]], 1)
        wt = w0 ** 2 * robust
        a, atw = a_matrix(wt)
        lhs = (atw @ a).tocsc()
        t_new = np.stack([spsolve(lhs, atw @ rhs[:, d]) for d in range(3)], 1)
        # --- robust weights from the normalised residuals
        thi, thj = get(th_new, ci, 0.0), get(th_new, cj, 0.0)
        ry = ((thi - thj - phi + np.pi) % (2 * np.pi) - np.pi) * lev * w0
        ti, tj = get(t_new, ci, 0.0), get(t_new, cj, 0.0)
        rt = np.linalg.norm((ti - tj) - rhs, axis=1) * w0
        r = np.sqrt(ry ** 2 + rt ** 2)
        # Huber (convex): a non-convex loss (Cauchy) was seen to absorb loop-closure drift by BREAKING the chain
        # at one weakly connected view (a 22 deg jump contradicted by every edge across it) instead of spreading it
        robust = np.where(r <= p.robust_scale, 1.0, p.robust_scale / np.maximum(r, 1e-12))
        step = max(np.abs(th_new - th).max(), np.abs(t_new - t).max())
        th, t = th_new, t_new
        if step < 1e-6:
            break
    poses = {nodes[0]: np.eye(4)}
    for n in free:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = _rz(th[col[n]]), t[col[n]]
        poses[n] = m
    return poses, {"solver": "decoupled yaw/translation IRLS (Huber)", "edges": ne, "nodes": nf + 1,
                   "iterations": it + 1, "last_step": float(step), "median_robust_weight": float(np.median(robust))}


def _edge_residual(e: PairResult, poses: dict[int, np.ndarray]) -> tuple[float, float]:
    d = np.linalg.inv(e.transform) @ np.linalg.inv(poses[e.j]) @ poses[e.i]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
    return ang, float(np.linalg.norm(d[:3, 3]))


def _check(name, status, value, threshold, message):
    return {"name": name, "status": status, "value": value, "threshold": threshold, "message": message,
            "threshold_status": "proposed; tuned on one real recording"}


def estimate_poses_tracked(features: list[ViewFeatures], p: TrackParams | None = None, progress=None) -> TrackResult:
    p = p or TrackParams()
    n = len(features)
    if n < 2:
        raise ValueError("need at least 2 views")
    tell = progress or (lambda s, f: None)
    edges: list[PairResult] = []
    done: set[tuple[int, int]] = set()
    for i in range(n):
        for off in p.offsets:
            j = i + off
            if j < n:
                edges.append(register_features(features[i], features[j], i, j, "sequential" if off == 1 else "skip", p))
                done.add((i, j))
        if i % 50 == 0:
            tell("registering view pairs", i / n)
    solve = _optimise if p.model == "se3" else _optimise_yaw
    solver_log: list = []
    seq_poses = solve(list(range(n)), edges, p, solver_log)
    yaw0 = {k: _yaw(m) for k, m in seq_poses.items()}

    # loop closures: views far apart in time whose yaw (from the loop-free graph) is within the window
    cand = []
    idx = sorted(yaw0)[::p.loop_stride]
    for a in idx:
        for b in idx:
            if b - a >= p.loop_min_separation and (a, b) not in done:
                dyaw = abs(np.degrees((yaw0[b] - yaw0[a] + np.pi) % (2 * np.pi) - np.pi))
                if dyaw <= p.loop_window_deg:
                    cand.append((dyaw, a, b))
    cand.sort()
    cand = cand[:p.loop_max_candidates]
    loop_edges = []
    for c, (_, a, b) in enumerate(cand):
        e = register_features(features[a], features[b], a, b, "loop", p)
        loop_edges.append(e)
        if c % 200 == 0:
            tell("loop-closure search", c / max(len(cand), 1))
    # loop-closure error BEFORE optimisation: accepted loop edges vs the loop-free solution
    pre = [_edge_residual(e, seq_poses) for e in loop_edges if e.ok and e.i in seq_poses and e.j in seq_poses]
    edges += loop_edges
    tell("optimising pose graph", 0.0)
    poses = solve(list(range(n)), edges, p, solver_log)

    checks = []
    missing = [k for k in range(n) if k not in poses]
    checks.append(_check("views_posed", "pass" if not missing else "warn", n - len(missing), n,
                         f"{len(missing)} views have no accepted edge to the main graph and are left out"))
    yaw_u = {}
    prev = None
    for k in sorted(poses):
        y = _yaw(poses[k])
        yaw_u[k] = y if prev is None else yaw_u[prev] + float((y - yaw_u[prev] + np.pi) % (2 * np.pi) - np.pi)
        prev = k
    turn = float(np.degrees(max(yaw_u.values()) - min(yaw_u.values())))
    ang = np.sort(np.degrees(np.array(list(yaw_u.values()))) % 360)
    gap = float(np.max(np.diff(np.concatenate([ang, [ang[0] + 360]])))) if len(ang) > 1 else 360.0
    checks.append(_check("angular_coverage_gap_deg", "pass" if gap <= 20 else "warn" if gap <= 45 else "fail", gap, 20.0,
                         f"largest gap in body orientation between posed views (turn range {turn:.0f} deg)"))
    n_loop_ok = sum(e.ok for e in loop_edges)
    if pre:
        pre_a = np.array(pre)
        checks.append(_check("loop_closure_before_opt_deg", "pass" if np.median(pre_a[:, 0]) <= 10 else "warn",
                             float(np.median(pre_a[:, 0])), 10.0,
                             f"median rotation disagreement of {len(pre)} accepted loop edges with the loop-free "
                             f"chain (max {pre_a[:, 0].max():.1f} deg): drift that loop closure corrects"))
    checks.append(_check("loop_edges_accepted", "pass" if n_loop_ok >= 3 else "fail", n_loop_ok, 3,
                         f"accepted loop-closure edges out of {len(loop_edges)} candidates"))
    res = np.array([_edge_residual(e, poses) for e in edges if e.ok and e.i in poses and e.j in poses])
    if len(res):
        p95a, p95t = float(np.percentile(res[:, 0], 95)), float(np.percentile(res[:, 1], 95))
        checks.append(_check("edge_residual_p95_deg", "pass" if p95a <= 1.5 else "warn" if p95a <= 3 else "fail", p95a, 1.5,
                             "95th percentile rotation disagreement of accepted edges with the optimised poses"))
        checks.append(_check("edge_residual_p95_m", "pass" if p95t <= 0.008 else "warn" if p95t <= 0.015 else "fail",
                             p95t, 0.008, "95th percentile translation disagreement of accepted edges (metres)"))
    tilt = np.array([np.degrees(np.arccos(np.clip(m[2, 2], -1, 1))) for m in poses.values()])
    checks.append(_check("body_tilt_max_deg", "pass" if tilt.max() <= 3 else "warn", float(tilt.max()), 3.0,
                         "largest tilt of the subject relative to view 0 (sway / lean while turning, or pose error)"))
    status = "fail" if any(c["status"] == "fail" for c in checks) else \
        "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    kinds = {}
    for e in edges:
        d = kinds.setdefault(e.kind, {"tried": 0, "accepted": 0})
        d["tried"] += 1
        d["accepted"] += int(e.ok)
    log = {"algorithm_version": ALGORITHM_VERSION, "parameters": asdict(p), "edges_by_kind": kinds,
           "loop_candidates": len(cand), "turn_range_deg": turn, "solver": solver_log,
           "features_3d_median": float(np.median([len(f.points) for f in features])),
           "edge_residuals": {"rotation_deg_p50": float(np.median(res[:, 0])) if len(res) else None,
                              "translation_m_p50": float(np.median(res[:, 1])) if len(res) else None}}
    return TrackResult(poses, edges, checks, status, {k: float(np.degrees(v)) for k, v in yaw_u.items()}, log)
