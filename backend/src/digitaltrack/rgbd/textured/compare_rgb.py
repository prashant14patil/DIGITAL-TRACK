"""Compare an RGB-D mesh with the EXISTING RGB-only pipeline (experimental visual hull) on the same recording.

The RGB-only result has no metric scale, so it is scaled to the RGB-D mesh's vertical extent. That scale is
BORROWED from depth: height can therefore never be compared, and nothing here says which one is right. The
numbers describe how far apart the two shapes are after the best turn about the vertical axis.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ...ingest.video import find_tool
from .evaluate import distance_stats, distances_to_mesh, mesh_topology
from .sensors import gravity, read_export


def upright_filter(up_camera: np.ndarray) -> str:
    """ffmpeg filter that turns stored frames upright, from the gravity direction in the camera frame."""
    x, y = float(up_camera[0]), float(up_camera[1])
    if abs(x) > abs(y):
        return "transpose=1" if x < 0 else "transpose=2"  # image-left is up -> clockwise, else counter-clockwise
    return "null" if y < 0 else "hflip,vflip"


def run_rgb_only_baseline(recording_dir: str | Path, work_dir: str | Path, *, frame_step: int = 6,
                          progress=None) -> Path:
    """Run the existing RGB-only pipeline on a DERIVED upright, half-size copy of rgb.mp4. -> its run directory.

    Settings follow the web app (about 10 fps, vote fraction 0.8, 720 x 960 working images).
    """
    from ...experimental.rgb_visual_hull import run_visual_hull
    from ...ingest.metadata import validate_capture_metadata
    from ...ingest.session import ingest_video

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=False)
    exp, _ = read_export(recording_dir)
    up, _ = gravity(exp)
    rot = upright_filter(up)
    portrait = rot.startswith("transpose")
    w, h = exp.rgb_size[::-1] if portrait else exp.rgb_size
    video = work / "upright_derived.mp4"
    subprocess.run([find_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(exp.root / "rgb.mp4"), "-vf",
                    f"{rot},scale={w // 2}:{h // 2}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an",
                    "-map_metadata", "-1", str(video)], check=True)
    md = validate_capture_metadata({"subject_id": "unspecified", "setup_id": "stray-rgb-derived",
                                    "protocol_version": "web-demo-unvalidated", "capture_type": "rgb_video"})
    ingest_video(video, work / "session", md, frame_step=frame_step)
    run_dir, _ = run_visual_hull(work / "session", downscale=1, voxel_px=3.0, vote_fraction=0.8, progress=progress)
    return run_dir


def _sample(mesh: o3d.geometry.TriangleMesh, n: int) -> np.ndarray:
    o3d.utility.random.seed(0)
    return np.asarray(mesh.sample_points_uniformly(n).points)


def _normalised(mesh: o3d.geometry.TriangleMesh, scale: float = 1.0) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(mesh)
    m.scale(scale, center=np.zeros(3))
    pts = _sample(m, 20000)
    # surface centroid (rotates with the shape, unlike a per-axis median)
    m.translate(-np.array([pts[:, 0].mean(), pts[:, 1].mean(), m.get_min_bound()[2]]))
    return m


def _rz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])


def best_yaw(moving: np.ndarray, fixed: np.ndarray) -> tuple[float, float]:
    """Turn about z (degrees) with the smallest median nearest-neighbour distance."""
    tree = cKDTree(fixed)
    cost = lambda a: float(np.median(tree.query(moving @ _rz(a).T)[0]))
    coarse = min(range(0, 360, 4), key=cost)
    fine = min(np.arange(coarse - 4, coarse + 4.01, 0.5), key=cost)
    return float(fine % 360), cost(fine)


def compare_meshes(rgbd: o3d.geometry.TriangleMesh, rgb_only: o3d.geometry.TriangleMesh,
                   fractions=(0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.92)) -> tuple[dict, o3d.geometry.TriangleMesh,
                                                                                o3d.geometry.TriangleMesh]:
    a = _normalised(rgbd)
    height = float(a.get_max_bound()[2])
    source_height = float(rgb_only.get_max_bound()[2] - rgb_only.get_min_bound()[2])
    b = _normalised(rgb_only, height / source_height)
    pa, pb = _sample(a, 30000), _sample(b, 30000)
    yaw, cost = best_yaw(pb[::4], pa)
    mirrored = pb * [-1, 1, 1]
    _, mirror_cost = best_yaw(mirrored[::4], pa)
    transform = np.eye(4)
    transform[:3, :3] = _rz(yaw)
    b.transform(transform)
    pb = _sample(b, 30000)
    slices = []
    for f in fractions:
        z = f * height
        sa, sb = pa[np.abs(pa[:, 2] - z) < 0.01][:, :2], pb[np.abs(pb[:, 2] - z) < 0.01][:, :2]
        if len(sa) < 20 or len(sb) < 20:
            slices.append({"height_fraction": f, "rgbd_extent_mm": None, "rgb_only_extent_mm": None})
            continue
        axes = np.linalg.svd(sa - sa.mean(0), full_matrices=False)[2]
        ext = lambda s: [float(v) * 1000 for v in np.ptp(s @ axes.T, axis=0)]
        slices.append({"height_fraction": f, "rgbd_extent_mm": ext(sa), "rgb_only_extent_mm": ext(sb)})
    report = {
        "scale_of_rgb_only": "borrowed from the RGB-D mesh vertical extent (RGB-only has no metric scale)",
        "yaw_alignment_deg": yaw, "median_alignment_distance_mm": cost * 1000,
        "mirrored_fits_better": bool(mirror_cost < 0.9 * cost),
        "rgb_only_to_rgbd_surface": distance_stats(distances_to_mesh(a, pb)),
        "rgbd_to_rgb_only_surface": distance_stats(distances_to_mesh(b, pa)),
        "slice_outline_extents": {"note": "major and minor extent of everything at that height (arms included); "
                                          "not a body measurement", "rows": slices},
        "rgbd_topology": mesh_topology(rgbd), "rgb_only_topology": mesh_topology(rgb_only),
        "interpretation": "difference between two unvalidated methods, not an error of either",
    }
    return report, a, b


def _render(mesh: o3d.geometry.TriangleMesh, azimuth: float, height: float, size=(300, 600)) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    a = np.radians(azimuth)
    forward = np.array([np.cos(a), np.sin(a), 0.0])
    right = np.array([np.sin(a), -np.cos(a), 0.0])
    half = height * 0.3
    u, v = np.meshgrid(np.linspace(-half, half, size[0]), np.linspace(height * 1.05, -0.05 * height, size[1]))
    origin = u[..., None] * right + v[..., None] * np.array([0.0, 0.0, 1.0]) - 3.0 * forward
    rays = np.concatenate([origin, np.broadcast_to(forward, origin.shape)], -1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    shade = 60 + 180 * np.abs(hit["primitive_normals"].numpy() @ forward)
    return np.where(np.isfinite(hit["t_hit"].numpy()), shade, 255).astype(np.uint8)


MIN_SILHOUETTE_COVERAGE = 0.9  # proposed: typical RGB-only silhouette height / person height seen by depth


def silhouette_coverage(rgbd: o3d.geometry.TriangleMesh, quality: dict, rgb_report: dict, mask_height_px: int) -> float:
    """Typical RGB-only silhouette height divided by the height the RGB-D mesh projects to in the same image.

    Well below 1 means the RGB-only segmentation lost part of the person (e.g. legs against a similar floor).
    """
    to_floor = np.asarray(quality["views"][0]["floor_from_body"])
    to_camera = np.linalg.inv(np.asarray(quality["floor_from_camera"])) @ to_floor
    v = np.asarray(rgbd.vertices) @ to_camera[:3, :3].T + to_camera[:3, 3]
    along_up = v @ np.asarray(quality["gravity"]["up_camera"]) / v[:, 2]
    audit = quality["inputs"]["audit"]
    fx = float(np.mean(audit["intrinsics"]["fx_range"])) * mask_height_px / max(audit["rgb"]["size"])
    return float(rgb_report["carving"]["height_ref_px"] / (fx * np.ptp(along_up)))


def compare_turn_angles(quality: dict, rgb_report: dict) -> list[dict] | None:
    """The RGB-only turn anchors against the RGB-D headings at the same moments (neither is ground truth)."""
    angles = rgb_report.get("angles") or {}
    views, step = quality.get("views") or [], (rgb_report.get("frame_step") or 6)
    if not angles.get("anchors") or len(views) < 2:
        return None
    frames = np.array([v["frame"] for v in views], float)
    heading = np.degrees(np.unwrap(np.radians([v["turn_deg"] for v in views])))
    source = np.array([(a["frame_id"] - 1) * step for a in angles["anchors"]], float)
    at = np.interp(source, frames, heading)
    return [{"video_frame": int(f), "rgb_only_deg": float(a["angle_deg"] - angles["anchors"][0]["angle_deg"]),
             "rgbd_deg": float(abs(h - at[0]))} for f, a, h in zip(source, angles["anchors"], at)]


def compare_with_rgb_only(rgbd_run_dir: str | Path, rgb_only_run_dir: str | Path) -> dict:
    """Write comparison_rgb_only.json / .png into the RGB-D run directory."""
    run, other = Path(rgbd_run_dir), Path(rgb_only_run_dir)
    rgbd = o3d.io.read_triangle_mesh(str(run / "mesh.ply"))
    rgb_only = o3d.io.read_triangle_mesh(str(other / "mesh_height_normalised.ply"))
    if not len(rgbd.triangles) or not len(rgb_only.triangles):
        raise ValueError("both runs need a mesh")
    src = json.loads((other / "report.json").read_text(encoding="utf-8")) if (other / "report.json").is_file() else {}
    quality = json.loads((run / "quality_report.json").read_text(encoding="utf-8"))         if (run / "quality_report.json").is_file() else {}
    coverage = None
    masks = sorted((other / "silhouettes").glob("*.png"))
    if masks and src.get("carving") and quality.get("views"):
        coverage = silhouette_coverage(rgbd, quality, src, cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE).shape[0])
    report, a, b = compare_meshes(rgbd, rgb_only)
    if coverage is not None and coverage < MIN_SILHOUETTE_COVERAGE:
        # an incomplete body stretched to the full height would give meaningless distances: withhold them
        for key in ("scale_of_rgb_only", "yaw_alignment_deg", "median_alignment_distance_mm", "mirrored_fits_better",
                    "rgb_only_to_rgbd_surface", "rgbd_to_rgb_only_surface", "slice_outline_extents"):
            report[key] = None
        report.update(comparable=False, reason=f"RGB-only silhouettes cover only {coverage:.0%} of the person's height "
                      "(segmentation lost part of the body), so its shape cannot be scaled or compared")
        b = _normalised(rgb_only, float(a.get_max_bound()[2]) / float(np.ptp(np.asarray(rgb_only.vertices)[:, 2])))
    else:
        report["comparable"] = True
    report["rgb_only_silhouette_coverage"] = coverage
    report["turn_angles"] = compare_turn_angles(quality, src)
    report["rgb_only_run"] = {k: src.get(k) for k in ("algorithm", "status", "status_reasons", "parameters", "runtime_s")}
    report["rgb_only_run"]["angle_anchors"] = (src.get("angles") or {}).get("anchors")
    height = float(a.get_max_bound()[2])
    rows = []
    other_label = ("RGB-only visual hull (scale borrowed)" if report["comparable"]
                   else "RGB-only visual hull: INCOMPLETE, not to scale")
    for name, m in (("RGB-D (textured)", a), (other_label, b)):
        row = np.hstack([_render(m, az, height) for az in (0, 45, 90, 135, 180, 270)]).copy()
        cv2.putText(row, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(str(run / "comparison_rgb_only.png"), np.vstack(rows))
    (run / "comparison_rgb_only.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
