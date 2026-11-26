"""Unit tests for the RGB-D tracked pipeline components (rgbd.track, rgbd.depthfilter, rgbd.consistency, stray
helpers). Everything here is SYNTHETIC with known answers; no real scan data is used."""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import pytest

from digitaltrack.dataset.loader import Intrinsics
from digitaltrack.rgbd import track as T
from digitaltrack.rgbd.consistency import ConsistencyParams, cull_mesh, free_space_votes
from digitaltrack.rgbd.depthfilter import DepthFilterParams, fusion_mask
from digitaltrack.rgbd.stray import StrayFormatError, imu_gravity_up_camera, iter_rgb_frames
from digitaltrack.rgbd.views import RGBDView


def _pose(yaw_deg: float, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = T._rz(np.radians(yaw_deg))
    m[:3, 3] = t
    return m


def _body_features(n: int = 600, seed: int = 0):
    """Textured 'body': points on an elliptic cylinder (r 0.15 x 0.11 m, 0.2-1.7 m high) with unique descriptors."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0.2, 1.7, n)
    pts = np.stack([0.15 * np.cos(a), 0.11 * np.sin(a), z], 1)
    des = rng.normal(size=(n, 128)).astype(np.float32)
    return pts, des


def _observe(pts, des, body_from_floor, cam_dir=np.array([1.0, 0.0]), noise=0.002, seed=0):
    """Features visible from a fixed camera on the +x side of the floor frame, for a body at `body_from_floor`."""
    rng = np.random.default_rng(seed)
    floor_from_body = np.linalg.inv(body_from_floor)
    p = pts @ floor_from_body[:3, :3].T + floor_from_body[:3, 3]
    normal = p[:, :2] - floor_from_body[:2, 3]
    vis = normal @ cam_dir > 0.03
    return T.ViewFeatures(0, p[vis] + rng.normal(0, noise, (vis.sum(), 3)), des[vis], int(vis.sum()))


def test_register_features_recovers_yaw_and_shift():
    pts, des = _body_features()
    a = _observe(pts, des, _pose(0))
    b = _observe(pts, des, _pose(-12.0, (0.01, -0.02, 0.0)), seed=1)
    e = T.register_features(a, b, 0, 1, "sequential", T.TrackParams())
    assert e.ok, e.reasons
    truth = np.linalg.inv(_pose(-12.0, (0.01, -0.02, 0.0))) @ _pose(0)  # X_b = inv(pose_b) pose_a X_a
    assert abs(np.degrees(T._yaw(e.transform) - T._yaw(truth))) < 0.5
    assert np.linalg.norm(e.transform[:3, 3] - truth[:3, 3]) < 0.005


def test_register_features_rejects_unrelated_views():
    pts, des = _body_features(seed=0)
    pts2, des2 = _body_features(seed=5)
    e = T.register_features(_observe(pts, des, _pose(0)), _observe(pts2, des2, _pose(0)), 0, 1, "loop", T.TrackParams())
    assert not e.ok


def test_full_turn_pose_graph_with_loop_closure():
    """60 views over a full 354 deg turn with sway; poses must close the loop and match the truth."""
    pts, des = _body_features(900)
    rng = np.random.default_rng(3)
    yaws = np.linspace(0, 354, 60)
    truth = [_pose(-y, (*rng.normal(0, 0.01, 2), 0.0)) for y in yaws]
    truth = [np.linalg.inv(truth[0]) @ t for t in truth]  # view 0 = body frame
    feats = [_observe(pts, des, t, seed=k) for k, t in enumerate(truth)]
    res = T.estimate_poses_tracked(feats, T.TrackParams(offsets=(1, 2, 3), loop_min_separation=20, loop_stride=1))
    assert len(res.body_from_floor) == 60
    err = [abs((T._yaw(res.body_from_floor[k]) - T._yaw(truth[k]) + np.pi) % (2 * np.pi) - np.pi) for k in range(60)]
    assert np.degrees(max(err)) < 1.0
    assert res.log["edges_by_kind"]["loop"]["accepted"] > 0
    assert abs(res.log["turn_range_deg"] - 354) < 2


def test_huber_graph_spreads_drift_and_flags_outlier():
    """A gross outlier edge must be flagged; the remaining chain must not be broken to absorb it."""
    rng = np.random.default_rng(1)
    n = 40
    truth = {k: _pose(9.0 * k, rng.normal(0, 0.03, 3)) for k in range(n)}
    truth[0] = np.eye(4)
    edges = []
    for i in range(n):
        for off in (1, 2, 5):
            j = i + off
            if j < n:
                t = np.linalg.inv(truth[j]) @ truth[i]
                t[:3, 3] += rng.normal(0, 0.001, 3)
                edges.append(T.PairResult(i, j, "skip", True, t, np.eye(6), 100, 100, 0.003, 0, 0.15))
    bad = edges[10]
    bad.transform = bad.transform @ _pose(20, (0.05, 0, 0))
    est = T._optimise_yaw(list(range(n)), edges, T.TrackParams())
    err = [abs((T._yaw(est[k]) - T._yaw(truth[k]) + np.pi) % (2 * np.pi) - np.pi) for k in range(n)]
    assert np.degrees(max(err)) < 0.1
    assert not bad.ok and bad.reasons[-1].startswith("outlier")


def _plane_view(depth_far=2.5):
    k = Intrinsics(200.0, 200.0, 64.0, 48.0, 128, 96)
    d = np.full((96, 128), depth_far, np.float32)
    d[20:76, 40:88] = 2.0  # subject block 0.5 m in front of the wall
    conf = np.full(d.shape, 2, np.uint8)
    subject = d < 2.2
    return RGBDView(0, d, k, conf), subject


def test_fusion_mask_drops_low_confidence_edges_and_flying_pixels():
    v, subject = _plane_view()
    v.confidence[30, 50] = 1
    v.depth_m[50, 60] = 2.3  # flying pixel inside the subject
    m, stats = fusion_mask(v, subject, DepthFilterParams())
    assert not m[30, 50]  # low confidence
    assert not m[50, 60] and not m[50, 61]  # discontinuity removes the pixel and its neighbours
    assert not m[20, 50] and m[21 + 1, 50 + 1]  # outline eroded by one pixel
    assert stats["subject"] > stats["after_view_angle"] > 0


def test_free_space_vote_removes_ghost_keeps_body():
    """Body sphere seen by 12 views; a ghost blob exists in the mesh but every view sees through it."""
    body = o3d.geometry.TriangleMesh.create_sphere(0.15, 30)
    body.translate((0, 0, 1.0))
    ghost = o3d.geometry.TriangleMesh.create_sphere(0.04, 10)
    ghost.translate((0.0, 0.32, 1.0))
    k = Intrinsics(200.0, 200.0, 64.0, 48.0, 128, 96)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(body))
    views, masks, poses = [], [], []
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        c = np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 1.0])
        zc = -c / np.linalg.norm(c)
        zc[2] = 0
        zc /= np.linalg.norm(zc)
        xc = np.cross([0, 0, -1.0], zc)
        yc = np.cross(zc, xc)
        r = np.stack([xc, yc, zc])
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = r, -r @ c
        vv, uu = np.mgrid[0:96, 0:128]
        dirs = np.stack([(uu - k.cx) / k.fx, (vv - k.cy) / k.fy, np.ones_like(uu, float)], -1) @ r
        rays = np.concatenate([np.broadcast_to(c, dirs.shape), dirs], -1).reshape(-1, 6).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(96, 128)
        depth = np.where(np.isfinite(hit), hit, 3.0).astype(np.float32)  # background wall at 3 m
        views.append(RGBDView(0, depth, k, np.full(depth.shape, 2, np.uint8)))
        masks.append(np.isfinite(hit))
        poses.append(t)
    mesh = body + ghost
    sup, vio = free_space_votes(np.asarray(mesh.vertices), views, masks, poses, ConsistencyParams(view_stride=1))
    out, log = cull_mesh(mesh, sup, vio, ConsistencyParams(), min_cluster_fraction=0.0)
    nb = len(body.vertices)
    assert (vio[nb:] > 2 * sup[nb:] + 3).all()  # every ghost vertex is contradicted
    assert (vio[:nb] <= 2 * sup[:nb] + 3).all()  # no body vertex is removed
    assert len(out.vertices) == nb


def test_imu_gravity_mapping(tmp_path):
    # device upright in portrait, rear camera image is landscape with the device top on the image LEFT
    rows = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
    rows += [f"{i * 0.01}, 0.0, -1.0, -0.05, 0, 0, 0" for i in range(50)]
    (tmp_path / "imu.csv").write_text("\n".join(rows) + "\n")
    up, info = imu_gravity_up_camera(tmp_path)
    assert up[0] < -0.99  # up = image left = -x
    assert up[2] < 0  # a_z = -0.05 g -> up has +0.05 along device z (screen) = -0.05 along camera z
    assert abs(up[2] + 0.05) < 0.002
    assert info["rows"] == 50


def test_imu_rejects_wrong_units(tmp_path):
    rows = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"] + [f"{i}, 0, -9.81, 0, 0, 0, 0" for i in range(20)]
    (tmp_path / "imu.csv").write_text("\n".join(rows) + "\n")
    with pytest.raises(StrayFormatError):
        imu_gravity_up_camera(tmp_path)


def test_iter_rgb_frames_sequential(tmp_path):
    w = cv2.VideoWriter(str(tmp_path / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    got = list(iter_rgb_frames(tmp_path, [3, 7, 15]))
    assert [f for f, _ in got] == [3, 7, 15]
    assert abs(int(got[1][1].mean()) - 70) <= 6  # frame 7 decoded in order (lossy codec tolerance)
