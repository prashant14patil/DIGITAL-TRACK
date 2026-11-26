"""Compare body meshes from different pipelines on the SAME recording (diagnostic; no accuracy claims).

    python tools/compare_body_meshes.py --new <tracked run> [--old-rgbd <selfrot run>] [--rgb-only <vhull run>]
                                        [--other <run dir with mesh.ply>] --out <dir>

For every mesh: topology and completeness (components, holes, boundary length, surface area, height extent).
For METRIC meshes (RGB-D pipelines), additionally the distance between surfaces after aligning each one to the new
mesh by a yaw grid search + rigid ICP. The alignment exists only to put the meshes side by side; it is logged and
never applied to any result. A difference between two unvalidated methods is not an error of either.
The RGB-only visual hull is height-normalised (no metric scale), so it is only compared by topology/coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from digitaltrack.rgbd import visualize as vis  # noqa: E402
from digitaltrack.rgbd.consistency import pixel_residuals  # noqa: E402
from digitaltrack.rgbd.tracked import _slice_image, mesh_quality  # noqa: E402


def _load(p: Path) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(str(p))
    m.remove_unreferenced_vertices()
    return m


def _align(src: o3d.geometry.TriangleMesh, dst: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, dict]:
    """src -> dst: z unchanged (both floor frames, z up), yaw grid every 5 deg with centroid matching, then ICP."""
    o3d.utility.random.seed(0)
    ps = src.sample_points_uniformly(40000)
    pd = dst.sample_points_uniformly(40000)
    pd.estimate_normals()
    a, b = np.asarray(ps.points), np.asarray(pd.points)
    best = (np.inf, None)
    tree = o3d.geometry.KDTreeFlann(pd)
    for yaw in np.radians(np.arange(0, 360, 5)):
        c, s = np.cos(yaw), np.sin(yaw)
        t = np.eye(4)
        t[:2, :2] = [[c, -s], [s, c]]
        t[:2, 3] = b[:, :2].mean(0) - t[:2, :2] @ a[:, :2].mean(0)
        q = a[::20] @ t[:3, :3].T + t[:3, 3]
        d = [np.sqrt(tree.search_knn_vector_3d(x, 1)[2][0]) for x in q]
        score = float(np.median(d))
        if score < best[0]:
            best = (score, t)
    icp = o3d.pipelines.registration.registration_icp(
        ps, pd, 0.05, best[1], o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp2 = o3d.pipelines.registration.registration_icp(
        ps, pd, 0.02, icp.transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane())
    t = np.asarray(icp2.transformation)
    return t, {"coarse_yaw_median_m": best[0], "icp_fitness": icp2.fitness, "icp_rmse_m": icp2.inlier_rmse,
               "transform": t.tolist(), "note": "comparison only; never applied to any result"}


def _distances(a: o3d.geometry.TriangleMesh, b: o3d.geometry.TriangleMesh, n: int = 100000) -> dict:
    o3d.utility.random.seed(1)
    pa = np.asarray(a.sample_points_uniformly(n).points)
    pb = np.asarray(b.sample_points_uniformly(n).points)
    dab, dba = pixel_residuals(b, pa), pixel_residuals(a, pb)
    f = lambda d: {"p50_mm": float(np.median(d) * 1000), "p90_mm": float(np.percentile(d, 90) * 1000),  # noqa: E731
                   "p95_mm": float(np.percentile(d, 95) * 1000)}
    return {"other_to_new": f(dab), "new_to_other": f(dba)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old-rgbd")
    ap.add_argument("--rgb-only")
    ap.add_argument("--other")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    new = _load(Path(a.new) / "fused_mesh.ply")
    res: dict = {"interpretation": "differences between unvalidated methods; not errors of either",
                 "meshes": {"new_rgbd_tracked": {"run": a.new, **mesh_quality(new)}}}
    renders = {"new_rgbd_tracked": new}
    metric = {}
    if a.old_rgbd:
        m = _load(Path(a.old_rgbd) / "fused_mesh.ply")
        log = json.loads((Path(a.old_rgbd) / "run_log.json").read_text(encoding="utf-8"))
        res["meshes"]["old_rgbd_selfrot"] = {"run": a.old_rgbd, "run_status": log.get("status"),
                                             "pose_checks": [{k: c[k] for k in ("name", "status", "value")}
                                                             for c in log.get("pose", {}).get("checks", [])],
                                             **mesh_quality(m)}
        renders["old_rgbd_selfrot"], metric["old_rgbd_selfrot"] = m, m
    if a.other:
        m = _load(Path(a.other) / "mesh.ply")
        res["meshes"]["other_rgbd"] = {"run": a.other, **mesh_quality(m)}
        renders["other_rgbd"], metric["other_rgbd"] = m, m
    if a.rgb_only:
        m = _load(Path(a.rgb_only) / "mesh_height_normalised.ply")
        rep = json.loads((Path(a.rgb_only) / "report.json").read_text(encoding="utf-8"))
        res["meshes"]["rgb_only_visual_hull"] = {"run": a.rgb_only, "status": rep.get("status"),
                                                 "status_reasons": rep.get("status_reasons"),
                                                 "units": "height-normalised (standing height = 1.0); not metric",
                                                 **mesh_quality(m)}
        renders["rgb_only_visual_hull"] = m
    for name, m in metric.items():
        t, info = _align(m, new)
        al = o3d.geometry.TriangleMesh(m).transform(t)
        res["meshes"][name]["alignment_to_new"] = info
        res["meshes"][name]["surface_distance_after_alignment"] = _distances(al, new)
        metric[name] = al
    # images: renders side by side, cross-sections overlaid
    tiles = []
    for name, m in renders.items():
        sub = out / f"render_{name}"
        sub.mkdir(exist_ok=True)
        vis.write_mesh_renders(m, [np.asarray(m.vertices)], sub, size=260)
        img = cv2.imread(str(sub / "mesh_render.png"))
        cv2.putText(img, name, (8, img.shape[0] - 10), 0, 0.6, (0, 0, 200), 2)
        tiles.append(img)
    w = max(t.shape[1] for t in tiles)
    cv2.imwrite(str(out / "renders.png"), np.vstack([cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT,
                                                                        value=(255, 255, 255)) for t in tiles]))
    colours = [(0, 0, 220), (220, 80, 0)]
    _slice_image([(m, colours[k % 2]) for k, m in enumerate(metric.values())] + [(new, (0, 0, 0))],
                 (0.6, 0.95, 1.1, 1.3, 1.45), out / "cross_sections.png")
    res["cross_sections_legend"] = {"black": "new_rgbd_tracked",
                                    **{str(colours[k % 2]): n for k, n in enumerate(metric)}}
    (out / "comparison.json").write_text(json.dumps(vis.to_jsonable(res), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({n: {k: v for k, v in d.items() if k in ("components", "holes", "boundary_length_m",
                                                                "surface_area_m2", "surface_distance_after_alignment")}
                      for n, d in res["meshes"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
