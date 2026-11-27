"""EXPERIMENTAL: prepare a LiDAR depth mesh (self-rotation run) for the Body avatar page.

The avatar's region heuristic (body_regions.py) expects: Z up, feet at the bottom, the subject facing -Y at the
start, +X = the subject's LEFT. A depth run's body frame is the floor frame of stop 0: Z up and floor at z = 0,
but its horizontal axes depend on how the iPad was held (portrait recordings differ). So a DISPLAY copy is turned
about the vertical axis only:
  forward = horizontal component of the camera's viewing direction at stop 0 (camera -> subject)
  the subject faced the camera at stop 0, so it faced -forward; the subject's left = the camera's right
  turn so that forward -> +Y (subject faces -Y) and camera-right -> +X
and its floor-plane centre is moved to x = y = 0. This is a rigid motion: sizes are unchanged and nothing is
measured from it. The display copy is then cleaned shape-preservingly (display_mesh.py: floor spill, loose
debris, Taubin smoothing, decimation) with its deviation from the source checked and recorded. The metric mesh
(fused_mesh.ply) is never modified. Every transform and cleanup step is written to avatar_display.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

ALGORITHM = "depth-avatar-display-0.4 (EXPERIMENTAL)"  # 0.3: closed display surface, estimated areas, landmarks
DISPLAY_MESH = "mesh_avatar_display.ply"


def display_transform(floor_from_camera: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """4x4 rigid transform body frame -> avatar display frame (rotation about z + horizontal centring)."""
    f_fc = np.asarray(floor_from_camera, float)
    forward = f_fc[:3, :3] @ np.array([0.0, 0.0, 1.0])  # camera viewing direction in the body frame
    forward[2] = 0.0
    if np.linalg.norm(forward) < 0.2:
        raise ValueError("camera looks almost straight down: the start direction is not defined")
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 0.0, -1.0], forward)  # camera right (x = down x forward in OpenCV terms)
    yaw = np.arctan2(forward[0], forward[1])  # rotate forward onto +Y
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t = np.eye(4)
    t[:3, :3] = rz
    centre = np.median(vertices[:, :2], axis=0)
    t[:2, 3] = -(rz[:2, :2] @ centre)
    info = {"yaw_deg": float(np.degrees(yaw)), "centre_xy_m": centre.tolist(),
            "forward_after": (rz @ forward).tolist(), "camera_right_after": (rz @ right).tolist(),
            "camera_position_body_m": f_fc[:3, 3].tolist()}  # camera centre (camera origin) in the body frame
    return t, info


def _write_template_regions(run: Path, mesh, labels, template_only, report) -> dict:
    """mesh_regions.glb (region id in red, template-only = green 255) + regions.json, as body_regions writes them."""
    import trimesh

    from .body_regions import COLOUR_STEP, ID

    from .template_fit import REGIONS

    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    ids = np.array([ID[REGIONS[k]] for k in labels], np.uint8)
    rgba = np.zeros((len(v), 4), np.uint8)
    rgba[:, 0] = ids * COLOUR_STEP
    rgba[:, 1] = np.where(template_only, 255, 0)
    band = np.asarray(report.pop("pelvis_band", np.zeros(len(v), bool)), bool)
    rgba[:, 2] = np.where(band, 255, 0)  # blue 255 = pelvis display band (drawn a deeper tone, like the reference)
    rgba[:, 3] = 255
    (run / "mesh_regions.glb").write_bytes(
        trimesh.Trimesh(vertices=v, faces=f, vertex_colors=rgba, process=False).export(file_type="glb"))
    anchors = {}
    for name, rid in ID.items():
        pts = v[ids == rid]
        if not len(pts):
            continue
        if name.endswith("_arm"):
            pts = pts[pts[:, 2] >= np.quantile(pts[:, 2], 0.6)]
        elif name.endswith("_leg"):
            pts = pts[(pts[:, 2] >= np.quantile(pts[:, 2], 0.75)) & (pts[:, 2] <= np.quantile(pts[:, 2], 0.95))]  # thigh
        anchors[name] = [float(c) for c in pts.mean(axis=0)]
    info = {"algorithm": report.get("algorithm"), "status": "template fitted to the scan (display only)",
            "regions": [{"id": rid, "name": n, "vertex_count": int((ids == rid).sum()), "anchor": anchors.get(n)}
                        for n, rid in ID.items()],
            "colour_encoding": f"vertex red channel = id * {COLOUR_STEP} / 255; green 255 = template detail, not measured",
            "estimated_vertex_fraction": float(np.mean(template_only)),
            "left_right": "+X is the subject's left (subject faced the iPad at the start)",
            "note": "display only; no measurement is derived from the avatar"}
    (run / "regions.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def prepare_depth_avatar(run_dir: str | Path) -> dict:
    """Write mesh_avatar_display.ply, mesh_regions.glb, regions.json and avatar_display.json into `run_dir`."""
    from .body_regions import write_region_files
    from .display_mesh import build_display_surface, clean_display_mesh

    run = Path(run_dir)
    mesh_p, log_p = run / "fused_mesh.ply", run / "run_log.json"
    if not mesh_p.is_file():
        raise FileNotFoundError("this run has no mesh")
    log = json.loads(log_p.read_text(encoding="utf-8")) if log_p.is_file() else {}
    if "floor_from_camera" not in log:
        raise ValueError("run_log.json has no floor_from_camera: cannot tell which way the subject faced")
    mesh = o3d.io.read_triangle_mesh(str(mesh_p))
    v = np.asarray(mesh.vertices)
    if len(v) < 100:
        raise ValueError("mesh too small for the avatar")
    t, info = display_transform(np.array(log["floor_from_camera"]), v)
    disp = o3d.geometry.TriangleMesh(mesh)
    disp.transform(t)  # rigid (rotation about z + horizontal shift): floor stays at z = 0
    refined_p = run / "refined_mesh.ply"
    if refined_p.is_file():  # the pipeline's gated refined mesh: same rigid transform, estimated = > 5 mm from data
        from .display_mesh import _distances

        surf = o3d.io.read_triangle_mesh(str(refined_p))
        surf.transform(t)
        surf.compute_vertex_normals()
        estimated = _distances(disp, np.asarray(surf.vertices)) > 0.005
        rq = run / "refined_quality.json"
        surface_report = {"source": "refined_mesh.ply (pipeline, gated)",
                          "gate": json.loads(rq.read_text(encoding="utf-8")).get("status") if rq.is_file() else None,
                          "estimated_vertex_fraction": float(estimated.mean())}
    else:
        surf, estimated, surface_report = build_display_surface(disp)
    if surf is not None:
        disp, cleanup = surf, {"used": "closed display surface", **surface_report}
    else:
        disp, cleanup = clean_display_mesh(disp)
        cleanup["closed_surface_attempt"] = surface_report
        estimated = None
    # Template body fitted to the scan surface (user decision 2026-11-27): reference look, nothing measured from it.
    template_report = None
    try:
        from .template_fit import fit_template

        fitted, labels, template_only, template_report = fit_template(disp)
    except (ValueError, FileNotFoundError) as e:
        fitted, template_report = None, {"status": "refused", "reason": str(e)}
    levels = None
    bm = run / "body_measurements.json"
    if bm.is_file():
        levels = json.loads(bm.read_text(encoding="utf-8")).get("landmarks")
    if fitted is not None:
        o3d.io.write_triangle_mesh(str(run / DISPLAY_MESH), fitted)
        regions = _write_template_regions(run, fitted, labels, template_only, template_report)
    else:
        o3d.io.write_triangle_mesh(str(run / DISPLAY_MESH), disp)
        # landmark heights from the measuring mesh (display transform is a rotation about z + horizontal shift)
        regions = write_region_files(run, mesh_name=DISPLAY_MESH, levels_z=levels, estimated=estimated)
    record = {
        "algorithm": ALGORITHM, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_mesh": "fused_mesh.ply (unchanged)", "display_mesh": DISPLAY_MESH,
        "transform_body_to_display": t.tolist(), **info,
        "display_cleanup": cleanup,
        "template_fit": template_report,
        "avatar_surface": "template fitted to the scan (face/hands/feet are template detail, not measured)"
                          if fitted is not None else "scan surface (template fit refused or unavailable)",
        "region_levels_from": "body_measurements.json landmarks" if levels else "region heuristic",
        "run_status": log.get("status"), "run_status_note": log.get("status_note"),
        "units": "metres (rigid motion + shape-preserving cleanup of the depth mesh; nothing is measured from the display copy)",
        "assumption": "the subject faced the iPad at the first stop",
    }
    (run / "avatar_display.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"display": record, "regions": regions}
