"""validation.capture_checks on SYNTHETIC depth frames of a figure with KNOWN pose (legs apart, arms away)."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from digitaltrack.dataset.loader import Intrinsics
from digitaltrack.rgbd.views import RGBDView
from digitaltrack.synthetic.body_figure import Figure, build
from digitaltrack.validation.capture_checks import _clusters_1d, body_capture_checks

K = Intrinsics(212.6, 212.6, 96.0, 128.0, 192, 256)  # iPad depth intrinsics, held portrait (as recorded)


def _render(mesh, dist, yaw_deg):
    """Fixed camera at height 0.95 m, `dist` m in front of the figure, looking horizontally; wall 1.5 m behind it,
    floor at z = 0. Returns depth (m)."""
    scene = o3d.t.geometry.RaycastingScene()
    m = o3d.geometry.TriangleMesh(mesh)
    r = np.radians(yaw_deg)
    m.rotate(np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]), center=(0, 0, 0))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    floor = o3d.geometry.TriangleMesh.create_box(6, 6, 0.001).translate((-3, -3, -0.001))
    wall = o3d.geometry.TriangleMesh.create_box(6, 0.01, 3).translate((-3, 1.5, 0))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wall))
    c = np.array([0.0, -dist, 0.95])
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # rows = camera x, y (down), z (forward) in the world
    v, u = np.mgrid[0:K.height, 0:K.width]
    d = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u, float)], -1) @ rot
    rays = np.concatenate([np.broadcast_to(c, d.shape), d], -1).reshape(-1, 6).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(K.height, K.width)
    return np.where(np.isfinite(hit), hit, 0).astype(np.float32), rot


def _views(dist, n=24):
    fig = build(Figure(leg_y=0.11, arm_gap=0.12))
    empty = o3d.geometry.TriangleMesh.create_box(0.001, 0.001, 0.001).translate((0, 5, 50))
    up_cam = np.array([0.0, -1.0, 0.0])  # camera upright: world up = -y in the image
    bg = [RGBDView(i, _render(empty, dist, 0)[0], K, np.full((K.height, K.width), 2, np.uint8), up_camera=up_cam)
          for i in range(5)]
    vs = [RGBDView(i, _render(fig, dist, 360 * i / n)[0], K, np.full((K.height, K.width), 2, np.uint8),
                   timestamp_s=i * 1.5, up_camera=up_cam) for i in range(n)]
    return vs, bg


def test_clusters_1d():
    t = np.r_[np.linspace(0, 0.1, 50), np.linspace(0.2, 0.3, 50)]
    assert len(_clusters_1d(t, 0.05)) == 2 and len(_clusters_1d(t, 0.2)) == 1


def test_good_pose_passes_and_far_camera_warns():
    vs, bg = _views(1.6)
    checks, det = body_capture_checks(vs, bg, {"duration_s": 36.0})
    st = {c["id"]: c["status"] for c in checks}
    assert st["legs_apart"] == "ok" and st["arms_away"] == "ok" and st["arms_still"] == "ok", checks
    assert st["resolution"] == "ok" and st["turn_duration"] == "ok"
    vs, bg = _views(2.6)
    st = {c["id"]: c["status"] for c in body_capture_checks(vs, bg, {"duration_s": 112.0})[0]}
    assert st["resolution"] == "warn" and st["turn_duration"] == "warn"
