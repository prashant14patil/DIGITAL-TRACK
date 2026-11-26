"""Components of the texture-guided RGB-D pipeline, on SYNTHETIC inputs with known answers."""

import json

import cv2
import numpy as np
import open3d as o3d
import pytest

from digitaltrack.dataset.loader import Intrinsics
from digitaltrack.rgbd.pose import se2, se2_params
from digitaltrack.rgbd.stray import StrayFormatError
from digitaltrack.rgbd.textured.compare_rgb import (best_yaw, compare_meshes, compare_with_rgb_only,
                                                 upright_filter)
from digitaltrack.rgbd.textured.depth import DepthFilterParams, filter_depth, temporal_median
from digitaltrack.rgbd.textured.pipeline import export_mesh
from digitaltrack.rgbd.textured.sensors import gravity, iter_frames, read_export
from digitaltrack.rgbd.textured.tracking import (Features, PairEstimate, TrackingParams, estimate_pair,
                                                 solve_pose_graph)
from digitaltrack.synthetic.stray_scene import StraySceneConfig, make_stray_recording


@pytest.fixture(scope="module")
def tiny_recording(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny") / "rec"
    truth = make_stray_recording(folder, StraySceneConfig(n_stops=3, pause_frames=3, move_frames=2))
    return folder, truth


def test_flying_pixel_is_removed_but_both_edges_are_kept():
    raw = np.full((9, 9), 2000, np.uint16)
    raw[:, 5:] = 3000  # background
    raw[:, 4] = 2500  # mixed foreground/background pixels
    depth, stats = filter_depth(raw, np.full(raw.shape, 2, np.uint8), DepthFilterParams())
    assert np.all(depth[:, 4] == 0) and stats["flying"] == 9
    assert np.allclose(depth[:, 3], 2.0) and np.allclose(depth[:, 5], 3.0)  # the contour itself is untouched


def test_confidence_filter_and_no_smoothing():
    rng = np.random.default_rng(0)
    raw = (2000 + rng.integers(-3, 4, (20, 20))).astype(np.uint16)
    conf = np.full(raw.shape, 2, np.uint8)
    conf[0] = 1
    depth, stats = filter_depth(raw, conf, DepthFilterParams(min_confidence=2))
    assert stats["low_confidence"] == 20 and np.all(depth[0] == 0)
    assert np.array_equal(depth[1:], raw[1:].astype(np.float32) * np.float32(0.001))  # values pass through unchanged


def test_temporal_median_needs_half_of_the_samples():
    a = np.array([[2.0, 2.0, 0.0]], np.float32)
    b = np.array([[2.2, 0.0, 0.0]], np.float32)
    c = np.array([[2.1, 0.0, 1.0]], np.float32)
    med, _ = temporal_median([a, b, c], DepthFilterParams())
    assert med[0, 0] == pytest.approx(2.1) and med[0, 1] == 0 and med[0, 2] == 0


def test_export_audit_gravity_and_imu(tiny_recording):
    folder, truth = tiny_recording
    exp, audit = read_export(folder)
    assert len(exp) == truth["n_frames"] == audit["rgb"]["packets"]
    assert exp.intrinsics_depth(0).fx == pytest.approx(850.0 / 4)
    up, info = gravity(exp)
    assert np.allclose(up, truth["up_camera"], atol=1e-6)  # portrait: up = image left
    assert info["imu"]["disagreement_deg"] < 0.01
    assert upright_filter(up) == "transpose=1"


def test_export_audit_refuses_incomplete_export(tiny_recording, tmp_path):
    import shutil

    folder = tmp_path / "copy"
    shutil.copytree(tiny_recording[0], folder)
    (folder / "confidence" / "000002.png").unlink()
    with pytest.raises(StrayFormatError, match="lack"):
        read_export(folder)


def test_frame_selection_returns_the_requested_frames(tmp_path):
    path = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(240):
        out.write(np.full((48, 64, 3), i, np.uint8))
    out.release()
    want = list(range(3, 150)) + list(range(151, 240, 7))  # long runs and many singles: nested expression
    got = {i: float(img.mean()) for i, img in iter_frames(path, want, (64, 48))}
    assert list(got) == want
    assert max(abs(v - i) for i, v in got.items()) < 5


def _features(points_floor, cam: Intrinsics, camera_from_floor, desc):
    pc = points_floor @ camera_from_floor[:3, :3].T + camera_from_floor[:3, 3]
    uv = np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], 1)
    return Features(points_floor, uv, desc.astype(np.float32), cam, camera_from_floor)


def test_pair_estimate_recovers_turn_and_shift_despite_outliers():
    rng = np.random.default_rng(1)
    cam = Intrinsics(800, 800, 512, 384, 1024, 768)
    camera_from_floor = np.array([[1, 0, 0, 0], [0, 0, -1, 0.9], [0, 1, 0, 2.0], [0, 0, 0, 1.0]])
    a = rng.uniform([-0.2, -0.15, 0.4], [0.2, 0.15, 1.4], (200, 3))
    true = se2(np.radians(17.0), 0.03, -0.02)
    b = a.copy()
    b[:, :2] = a[:, :2] @ true[:2, :2].T + true[:2, 2]
    b[:40] += rng.uniform(-0.3, 0.3, (40, 3))  # wrong matches
    b[40:] += rng.normal(0, 0.002, (160, 3))
    desc = rng.uniform(0, 255, (200, 128))
    e = estimate_pair(_features(a, cam, camera_from_floor, desc), _features(b, cam, camera_from_floor, desc),
                      0, 1, "sequential", TrackingParams())
    th, x, y = se2_params(e.transform)
    assert np.degrees(th) == pytest.approx(17.0, abs=0.3)
    assert (x, y) == pytest.approx((0.03, -0.02), abs=0.003)
    assert 140 <= e.inliers <= 162
    assert np.all(np.linalg.eigvalsh(e.information) > 0)


def test_pose_graph_spreads_drift_and_leaves_unlinked_nodes_out():
    n = 12
    truth = [se2(np.radians(30 * k), 0, 0) for k in range(n)]

    def edge(i, j, kind, bias_deg):
        t = se2(np.radians(bias_deg), 0, 0) @ truth[j] @ np.linalg.inv(truth[i])
        return PairEstimate(i, j, kind, t, np.diag([3000.0, 1e4, 1e4]), 100, 120, 0.005, 1.0)

    edges = [edge(k, k + 1, "sequential", -2.0) for k in range(n - 1)]  # every step under-rotates
    edges.append(edge(n - 1, 0, "loop", 0.0))
    poses, _ = solve_pose_graph(n + 1, edges)  # node 12 has no edge
    assert n not in poses and len(poses) == n
    err = [abs(np.degrees(se2_params(poses[k] @ np.linalg.inv(truth[k]))[0])) for k in range(n)]
    chain_end_error = 2.0 * (n - 1)
    assert max(err) < chain_end_error / 2


def test_exports_round_trip(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_sphere(0.3, 40)
    result = export_mesh(mesh, tmp_path)
    assert all(result[k]["verified"] for k in ("ply", "obj", "glb")), result
    assert (tmp_path / "mesh.glb").read_bytes()[:4] == b"glTF"


def test_shape_comparison_finds_the_turn_between_two_meshes():
    box = o3d.geometry.TriangleMesh.create_box(0.4, 0.2, 1.5)
    bar = o3d.geometry.TriangleMesh.create_box(0.2, 0.05, 0.1).translate((0.4, 0.0, 1.0))
    a = box + bar
    b = o3d.geometry.TriangleMesh(a)
    b.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(-70))), center=(0, 0, 0))
    b.scale(1 / 1.5, center=(0, 0, 0))  # height-normalised, like the RGB-only output
    report, _, _ = compare_meshes(a, b)
    assert report["yaw_alignment_deg"] == pytest.approx(70.0, abs=2.0)
    assert report["rgb_only_to_rgbd_surface"]["p95_mm"] < 15
    assert "borrowed" in report["scale_of_rgb_only"]
    assert best_yaw(np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))[0] == pytest.approx(90.0, abs=0.6)


@pytest.mark.parametrize("silhouette_px, comparable", [(660, True), (380, False)])
def test_comparison_withholds_distances_when_rgb_only_lost_part_of_the_body(tmp_path, silhouette_px, comparable):
    run, other = tmp_path / "rgbd", tmp_path / "rgb"
    (other / "silhouettes").mkdir(parents=True)
    run.mkdir()
    box = o3d.geometry.TriangleMesh.create_box(0.4, 0.2, 1.7).translate((-0.2, -0.1, 0.0))
    o3d.io.write_triangle_mesh(str(run / "mesh.ply"), box)
    o3d.io.write_triangle_mesh(str(other / "mesh_height_normalised.ply"), box)
    # level portrait camera 2.2 m away: a 1.7 m body at fx 800 (960 px working height) spans about 650 px
    floor_from_camera = [[0, 0, 1, -2.2], [0, 1, 0, 0], [-1, 0, 0, 0.9], [0, 0, 0, 1]]
    quality = {"floor_from_camera": floor_from_camera, "gravity": {"up_camera": [-1, 0, 0]},
               "inputs": {"audit": {"intrinsics": {"fx_range": [1600, 1600]}, "rgb": {"size": [1920, 1440]}}},
               "views": [{"frame": 0, "turn_deg": 0.0, "floor_from_body": np.eye(4).tolist()},
                         {"frame": 600, "turn_deg": -100.0, "floor_from_body": np.eye(4).tolist()}]}
    (run / "quality_report.json").write_text(json.dumps(quality), encoding="utf-8")
    rgb = {"carving": {"height_ref_px": silhouette_px},
           "angles": {"anchors": [{"frame_id": 1, "angle_deg": 0.0}, {"frame_id": 101, "angle_deg": 90.0}]}}
    (other / "report.json").write_text(json.dumps(rgb), encoding="utf-8")
    cv2.imwrite(str(other / "silhouettes" / "mask_000001.png"), np.zeros((960, 720), np.uint8))
    report = compare_with_rgb_only(run, other)
    assert report["comparable"] is comparable
    assert (report["rgb_only_to_rgbd_surface"] is not None) is comparable
    assert report["turn_angles"][1] == {"video_frame": 600, "rgb_only_deg": 90.0, "rgbd_deg": pytest.approx(100.0)}
    assert (run / "comparison_rgb_only.png").is_file()
