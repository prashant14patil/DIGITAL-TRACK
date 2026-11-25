"""EXPERIMENTAL: prepare a LiDAR depth mesh (self-rotation run) for the Body avatar page.

The avatar's region heuristic (body_regions.py) expects: Z up, feet at the bottom, the subject facing -Y at the
start, +X = the subject's LEFT. A depth run's body frame is the floor frame of stop 0: Z up and floor at z = 0,
but its horizontal axes depend on how the iPad was held (portrait recordings differ). So a DISPLAY copy is turned
about the vertical axis only:
  forward = horizontal component of the camera's viewing direction at stop 0 (camera -> subject)
  the subject faced the camera at stop 0, so it faced -forward; the subject's left = the camera's right
  turn so that forward -> +Y (subject faces -Y) and camera-right -> +X
and its floor-plane centre is moved to x = y = 0. This is a rigid motion: sizes are unchanged and nothing is
measured from it. The metric mesh (fused_mesh.ply) is never modified. Every transform is written to
avatar_display.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

ALGORITHM = "depth-avatar-display-0.1 (EXPERIMENTAL)"
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


def prepare_depth_avatar(run_dir: str | Path) -> dict:
    """Write mesh_avatar_display.ply, mesh_regions.glb, regions.json and avatar_display.json into `run_dir`."""
    from .body_regions import write_region_files

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
    disp.transform(t)
    o3d.io.write_triangle_mesh(str(run / DISPLAY_MESH), disp)
    regions = write_region_files(run, mesh_name=DISPLAY_MESH)
    record = {
        "algorithm": ALGORITHM, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_mesh": "fused_mesh.ply (unchanged)", "display_mesh": DISPLAY_MESH,
        "transform_body_to_display": t.tolist(), **info,
        "run_status": log.get("status"), "run_status_note": log.get("status_note"),
        "units": "metres (rigid motion of the depth mesh; nothing is measured from the display copy)",
        "assumption": "the subject faced the iPad at the first stop",
    }
    (run / "avatar_display.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"display": record, "regions": regions}
