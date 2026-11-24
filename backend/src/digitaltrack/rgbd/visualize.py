"""Visual outputs for inspecting a self-rotation reconstruction (no GPU or display needed).

Files written to the run directory:
  fused_mesh.ply                 the TSDF mesh (body frame, metres)
  mesh_coverage.ply              same mesh, vertex colour = number of views that observed that surface
  views_colored.ply              every view's subject points in the body frame, one colour per view
  views_front.png / _side / _top per-view coloured points, orthographic: misalignment shows as colour ghosting
  slices.png                     horizontal slices: each view's points + the mesh contour; doubled contours = misalignment
  poses.png                      top view of estimated view positions/headings, accepted/rejected edges, step angles
  mesh_render.png                shaded mesh from 4 directions
  coverage.png                   mesh coloured by observation count (red = seen by 1 view, missing = never meshed)
  report.html                    all of the above with the checks and per-edge table
"""

from __future__ import annotations

import colorsys
import html
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

BG = (255, 255, 255)


def palette(n: int) -> np.ndarray:
    return np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.85)] for i in range(n)], np.uint8)


def _project(points: np.ndarray, colors: np.ndarray, axes: tuple[int, int], depth_axis: int, depth_sign: float,
             size: int, bounds: tuple[np.ndarray, np.ndarray], splat: int = 1) -> np.ndarray:
    """Orthographic z-buffered splat. Image x = points[:, axes[0]], image up = points[:, axes[1]]."""
    lo, hi = bounds
    span = max(hi[axes[0]] - lo[axes[0]], hi[axes[1]] - lo[axes[1]]) * 1.08 + 1e-9
    cx, cy = (lo[axes[0]] + hi[axes[0]]) / 2, (lo[axes[1]] + hi[axes[1]]) / 2
    u = ((points[:, axes[0]] - cx) / span + 0.5) * (size - 1)
    v = (0.5 - (points[:, axes[1]] - cy) / span) * (size - 1)
    order = np.argsort(depth_sign * points[:, depth_axis])  # far first, near overwrites
    img = np.full((size, size, 3), BG, np.uint8)
    ui, vi, col = np.round(u[order]).astype(int), np.round(v[order]).astype(int), colors[order]
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            uu, vv = ui + du, vi + dv
            ok = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
            img[vv[ok], uu[ok]] = col[ok][:, ::-1]  # RGB -> BGR for OpenCV
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def _bounds(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)


def write_point_views(per_view: list[np.ndarray], out: Path, size: int = 640) -> dict:
    cols = palette(len(per_view))
    pts = np.concatenate(per_view)
    colors = np.concatenate([np.repeat(cols[k][None], len(p), 0) for k, p in enumerate(per_view)])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(str(out / "views_colored.ply"), pcd)
    b = _bounds(pts)
    files = {}
    for name, axes, daxis, sign in (("front", (0, 2), 1, -1.0), ("side", (1, 2), 0, 1.0), ("top", (0, 1), 2, 1.0)):
        img = _label(_project(pts, colors, axes, daxis, sign, size, b), f"{name}: one colour per view")
        cv2.imwrite(str(out / f"views_{name}.png"), img)
        files[name] = f"views_{name}.png"
    return files


def write_slices(per_view: list[np.ndarray], mesh: o3d.geometry.TriangleMesh, out: Path, size: int = 360,
                 fractions=(0.15, 0.4, 0.65, 0.9), band_m: float = 0.004) -> str:
    import trimesh

    cols = palette(len(per_view))
    pts = np.concatenate(per_view)
    zmin, zmax = np.percentile(pts[:, 2], 1), np.percentile(pts[:, 2], 99)
    b = _bounds(pts)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False) if len(mesh.triangles) else None
    tiles = []
    for f in fractions:
        z = zmin + f * (zmax - zmin)
        sel = [(p[np.abs(p[:, 2] - z) < band_m], cols[k]) for k, p in enumerate(per_view)]
        sp = np.concatenate([s for s, _ in sel]) if sel else np.zeros((0, 3))
        sc = np.concatenate([np.repeat(c[None], len(s), 0) for s, c in sel]) if sel else np.zeros((0, 3), np.uint8)
        img = _project(sp, sc, (0, 1), 2, 1.0, size, b, splat=1) if len(sp) else np.full((size, size, 3), BG, np.uint8)
        if tm is not None:
            sec = tm.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if sec is not None:
                lo, hi = b
                span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.08
                cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
                for path in sec.discrete:
                    uv = np.stack([((path[:, 0] - cx) / span + 0.5) * (size - 1),
                                   (0.5 - (path[:, 1] - cy) / span) * (size - 1)], 1).round().astype(np.int32)
                    cv2.polylines(img, [uv], False, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(_label(img, f"z = {z:.3f} m"))
    grid = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(out / "slices.png"), grid)
    return "slices.png"


def write_pose_plot(pose_result, out: Path, size: int = 640) -> str:
    img = np.full((size, size + 320, 3), BG, np.uint8)
    params = np.array(pose_result.se2_params) if pose_result.se2_params else np.zeros((0, 3))
    n = len(params)
    cols = palette(max(n, 1))
    if n:
        xy = params[:, 1:]
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 0.05) * 1.6
        c = xy.mean(0)
        to_px = lambda q: (int((q[0] - c[0]) / span * (size - 80) + size / 2), int(size / 2 - (q[1] - c[1]) / span * (size - 80)))
        for e in pose_result.edges:
            if e.i < n and e.j < n:
                cv2.line(img, to_px(xy[e.i]), to_px(xy[e.j]), (60, 170, 60) if e.ok else (40, 40, 220), 1, cv2.LINE_AA)
        for k in range(n):
            pt = to_px(xy[k])
            th = params[k, 0]
            tip = (int(pt[0] + 22 * np.cos(th)), int(pt[1] - 22 * np.sin(th)))
            cv2.arrowedLine(img, pt, tip, tuple(int(x) for x in cols[k][::-1]), 2, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(img, str(k), (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
        # step angles
        steps = np.degrees(np.diff(np.unwrap(params[:, 0])))
        x0, y0, h = size + 20, size - 40, size - 120
        smax = max(np.abs(steps).max() if len(steps) else 1, 1)
        for k, s in enumerate(steps):
            bh = int(abs(s) / smax * h)
            x = x0 + k * max(1, 280 // max(len(steps), 1))
            cv2.rectangle(img, (x, y0 - bh), (x + max(1, 280 // max(len(steps), 1)) - 2, y0), tuple(int(v) for v in cols[k][::-1]), -1)
        cv2.putText(img, f"step angles (max {smax:.1f} deg)", (x0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    _label(img, f"view positions/headings (floor plane); green = accepted edge, red = rejected; status {pose_result.status}")
    cv2.imwrite(str(out / "poses.png"), img)
    return "poses.png"


def _shade(mesh: o3d.geometry.TriangleMesh, n_points: int = 150000) -> tuple[np.ndarray, np.ndarray]:
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    pcd = m.sample_points_uniformly(n_points, use_triangle_normal=True)
    return np.asarray(pcd.points), np.asarray(pcd.normals)


def write_mesh_renders(mesh: o3d.geometry.TriangleMesh, per_view: list[np.ndarray], out: Path, size: int = 360,
                       coverage_radius_m: float = 0.01) -> dict:
    if len(mesh.triangles) == 0:
        return {}
    pts, nrm = _shade(mesh)
    trees = [cKDTree(p) for p in per_view if len(p)]
    count = np.zeros(len(pts), int)
    for t in trees:
        d, _ = t.query(pts, distance_upper_bound=coverage_radius_m)
        count += np.isfinite(d)
    shade_tiles, cov_tiles = [], []
    ramp = lambda c: np.stack([np.clip(255 - 60 * (c - 1), 60, 255), np.clip(60 * c, 0, 220), np.full_like(c, 60)], 1)
    for az in (0, 90, 180, 270):
        r = np.radians(az)
        rot = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
        p, n = pts @ rot.T, nrm @ rot.T
        light = np.clip(-n[:, 1] * 0.8 + n[:, 2] * 0.3, 0.08, 1.0)
        grey = (55 + 200 * light).astype(np.uint8)
        bounds = (np.min(p, 0), np.max(p, 0))
        shade_tiles.append(_label(_project(p, np.stack([grey] * 3, 1), (0, 2), 1, -1.0, size, bounds, splat=1),
                                  f"azimuth {az} deg"))
        cc = np.where(count[:, None] > 0, ramp(count.astype(float)), 0).astype(np.uint8)
        cov_tiles.append(_label(_project(p, cc, (0, 2), 1, -1.0, size, bounds, splat=1), f"azimuth {az} deg"))
    cv2.imwrite(str(out / "mesh_render.png"), np.concatenate(shade_tiles, axis=1))
    cv2.imwrite(str(out / "coverage.png"), np.concatenate(cov_tiles, axis=1))
    # coverage-coloured mesh vertices
    vt = np.asarray(mesh.vertices)
    vcount = np.zeros(len(vt), int)
    for t in trees:
        d, _ = t.query(vt, distance_upper_bound=coverage_radius_m)
        vcount += np.isfinite(d)
    cm = o3d.geometry.TriangleMesh(mesh)
    cm.vertex_colors = o3d.utility.Vector3dVector(np.where(vcount[:, None] > 0, ramp(vcount.astype(float)), 0) / 255.0)
    o3d.io.write_triangle_mesh(str(out / "mesh_coverage.ply"), cm)
    return {"mesh_render": "mesh_render.png", "coverage": "coverage.png",
            "coverage_stats": {"surface_samples": int(len(pts)), "seen_by_0_views": int((count == 0).sum()),
                               "seen_by_1_view": int((count == 1).sum()), "seen_by_3plus": int((count >= 3).sum())}}


def write_report(out: Path, summary: dict, images: dict) -> str:
    esc = lambda s: html.escape(str(s))
    checks = summary.get("pose", {}).get("checks", [])
    rows = "".join(f"<tr class='{esc(c['status'])}'><td>{esc(c['name'])}</td><td>{esc(c['status'])}</td>"
                   f"<td>{esc(_fmt(c['value']))}</td><td>{esc(_fmt(c['threshold']))}</td><td>{esc(c['message'])}</td></tr>"
                   for c in checks)
    edges = "".join(f"<tr class='{'pass' if e['ok'] else 'fail'}'><td>{e['i']}&rarr;{e['j']}</td><td>{esc(e['kind'])}</td>"
                    f"<td>{_fmt(e.get('theta_deg'))}</td><td>{_fmt(e.get('overlap'))}</td>"
                    f"<td>{_fmt(None if e.get('rms_m') is None else e['rms_m'] * 1000)}</td>"
                    f"<td>{esc(', '.join(e.get('weak_directions') or []))}</td><td>{esc('; '.join(e['reasons']))}</td></tr>"
                    for e in summary.get("pose", {}).get("edges", []))
    img_html = "".join(f"<figure><img src='{esc(v)}' alt='{esc(k)}'><figcaption>{esc(k)}</figcaption></figure>"
                       for k, v in images.items() if isinstance(v, str) and v.endswith(".png"))
    status = summary.get("status", "?")
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Reconstruction report</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#1b1f24;background:#fff}}
h1{{font-size:20px}} .banner{{padding:10px 14px;border-radius:8px;margin:12px 0;background:#fbf0d9}}
.status-pass{{background:#e3f4ea}} .status-fail{{background:#fdeceb}}
table{{border-collapse:collapse;margin:10px 0 24px;font-size:13px}} td,th{{border-bottom:1px solid #e2e5ea;padding:4px 8px;text-align:left}}
tr.fail td{{color:#b42318}} tr.warn td{{color:#8a5a00}} figure{{margin:0 0 18px}} img{{max-width:100%;border:1px solid #e2e5ea}}
code{{font-family:Consolas,monospace}}</style></head><body>
<h1>Self-rotation reconstruction: {esc(summary.get('source', {}).get('label', ''))}</h1>
<div class='banner status-{esc(status)}'><b>Status: {esc(status)}</b>. {esc(summary.get('status_note', ''))}</div>
<p>Run <code>{esc(summary.get('run_id'))}</code>, {esc(summary.get('n_views'))} views, pose algorithm
<code>{esc(summary.get('pose', {}).get('algorithm_version'))}</code>. Thresholds are proposed values checked on synthetic
data only. Nothing here is a validated measurement.</p>
<h2>Pose checks</h2><table><tr><th>check</th><th>status</th><th>value</th><th>limit</th><th>meaning</th></tr>{rows}</table>
<h2>Images</h2>{img_html}
<h2>Registration edges</h2><table><tr><th>edge</th><th>kind</th><th>turn (deg)</th><th>overlap</th><th>RMS (mm)</th><th>weak</th><th>rejected because</th></tr>{edges}</table>
<h2>Files</h2><p><code>fused_mesh.ply</code>, <code>mesh_coverage.ply</code>, <code>views_colored.ply</code> open in MeshLab,
CloudCompare, or <code>dt view-rgbd &lt;run dir&gt;</code>. Full details: <code>run_log.json</code>.</p>
</body></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")
    return "report.html"


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return v


def to_jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
