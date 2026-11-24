"""Self-rotation RGB-D pipeline on SYNTHETIC scenes (fixed camera, box turned AND shifted by hand on a floor).

Bounds below are regression bounds observed on the virtual sensor, NOT accuracy claims for real iPad depth.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from digitaltrack.geometry.transforms import invert, transform_points
from digitaltrack.reconstruction.tsdf import TSDFParams
from digitaltrack.rgbd.fuse import fuse_views
from digitaltrack.rgbd.pipeline import PipelineParams, run_self_rotation
from digitaltrack.rgbd.pose import PoseParams, estimate_poses, register_pair, se2_params
from digitaltrack.rgbd.segment import (SegmentationError, SegmentParams, background_plate, estimate_floor,
                                       make_floor_frame, segment_subject)
from digitaltrack.rgbd.stops import StopParams, detect_stops, median_view
from digitaltrack.rgbd.views import RGBDView, ViewError, backproject
from digitaltrack.synthetic.rgbd_scene import SceneConfig, make_sequence


def _pose_errors(seq, floor_from_cam, est):
    """Per-view (rotation deg, translation m) error of estimated floor_from_body vs synthetic truth."""
    g = floor_from_cam @ seq.camera_from_floor  # truth floor frame -> estimated floor frame
    out = []
    for k, e in enumerate(est):
        truth = g @ seq.floor_from_body[k] @ invert(seq.floor_from_body[0]) @ invert(g)
        d = invert(truth) @ e
        out.append((np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))), np.linalg.norm(d[:3, 3])))
    return np.array(out)


def _prepare(seq, p=SegmentParams()):
    floor = estimate_floor(seq.background_views, p, source="background_views")
    f = make_floor_frame(floor, np.array([0.0, 0.0, 1.4]))
    plate = background_plate(seq.background_views, p)
    segs = [segment_subject(v, f, p, plate) for v in seq.views]
    return floor, f, segs


@pytest.fixture(scope="module")
def short_seq():
    return make_sequence(SceneConfig(n_stops=4, noise_sigma_m=0.002))


# ---- scene, views ------------------------------------------------------------------------------

def test_synthetic_depth_is_metric(short_seq):
    v = short_seq.views[0]
    v.validate()
    pts = transform_points(invert(short_seq.camera_from_floor), backproject(v, v.depth_m > 0))
    floor = pts[np.abs(pts[:, 2]) < 0.01]
    assert len(floor) > 1000 and abs(np.median(floor[:, 2])) < 0.002  # floor at z = 0 within noise


def test_view_validation_rejects_wrong_units_and_sizes(short_seq):
    v = short_seq.views[0]
    with pytest.raises(ViewError):
        RGBDView(0, (v.depth_m * 1000).astype(np.float32), v.intrinsics).validate()  # millimetres
    with pytest.raises(ViewError):
        RGBDView(0, v.depth_m[:-1], v.intrinsics).validate()
    with pytest.raises(ViewError):
        RGBDView(0, v.depth_m.astype(np.float64), v.intrinsics).validate()


# ---- floor and segmentation --------------------------------------------------------------------

def test_floor_plane_recovered(short_seq):
    floor, _, _ = _prepare(short_seq)
    truth = invert(short_seq.camera_from_floor)[:3, :3].T @ np.array([0, 0, 1.0])
    assert np.degrees(np.arccos(np.clip(np.dot(floor.plane.normal, truth), -1, 1))) < 0.2


def test_wall_is_not_accepted_as_floor(short_seq):
    v = short_seq.views[0]
    wall_only = RGBDView(0, np.where(v.depth_m > 2.0, v.depth_m, 0).astype(np.float32), v.intrinsics, v.confidence)
    with pytest.raises(SegmentationError):
        estimate_floor([wall_only], SegmentParams(), source="test")


def test_segmentation_keeps_only_the_object(short_seq):
    _, f, segs = _prepare(short_seq)
    g = f @ short_seq.camera_from_floor
    for k, s in enumerate(segs):
        body = transform_points(invert(short_seq.floor_from_body[k]) @ invert(g), s.points_floor)
        # object = box [-0.2, 0.2] x [-0.15, 0.15] x [0, 0.5] plus block up to x = 0.32
        assert np.all(body[:, 0] > -0.215) and np.all(body[:, 0] < 0.335)
        assert np.all(np.abs(body[:, 1]) < 0.165) and np.all(body[:, 2] < 0.515)
        assert s.stats["subject_points"] > 2000


# ---- stops -------------------------------------------------------------------------------------

def test_stop_detection_with_transitions():
    seq = make_sequence(SceneConfig(n_stops=5, frames_per_stop=4, transition_frames=3, noise_sigma_m=0.003))
    stops, _ = detect_stops(seq.views, StopParams())
    assert len(stops) == 5
    for s in stops:
        assert {seq.stop_of_view[i] for i in range(s.start, s.end + 1)} == {seq.stop_of_view[s.keyframe]}
        assert None not in {seq.stop_of_view[i] for i in range(s.start, s.end + 1)}
    mv = median_view(seq.views, stops[0])
    assert mv.derived and mv.source["method"] == "temporal_median"
    assert np.all(seq.views[stops[0].start].depth_m == seq.views[stops[0].start].depth_m)  # raw untouched


def test_no_stops_when_always_moving():
    seq = make_sequence(SceneConfig(n_stops=4, frames_per_stop=1, transition_frames=3))
    stops, _ = detect_stops(seq.views, StopParams(min_frames=3))
    assert stops == []


# ---- pairwise registration and pose graph ------------------------------------------------------

def test_pairwise_registration_recovers_turn_and_shift(short_seq):
    _, f, segs = _prepare(short_seq)
    g = f @ short_seq.camera_from_floor
    e = register_pair(segs[0].points_floor, segs[1].points_floor, 0, 1, "sequential", PoseParams())
    assert e.ok, e.reasons
    truth = g @ short_seq.floor_from_body[1] @ invert(short_seq.floor_from_body[0]) @ invert(g)
    th, tx, ty = se2_params(e.transform)
    assert abs(np.degrees(th) - np.degrees(np.arctan2(truth[1, 0], truth[0, 0]))) < 0.5
    assert np.hypot(tx - truth[0, 3], ty - truth[1, 3]) < 0.015
    assert e.information is not None and e.std is not None


def test_symmetric_cylinder_is_never_a_confident_pass():
    seq = make_sequence(SceneConfig(n_stops=6, step_deg=30, object_kind="cylinder", noise_sigma_m=0.002))
    _, _, segs = _prepare(seq)
    r = estimate_poses([s.points_floor for s in segs])
    assert r.status == "fail" and r.floor_from_body == []  # no poses fabricated
    assert any("ambiguous" in reason for e in r.edges for reason in e.reasons)


def test_unrelated_views_cannot_be_placed(short_seq):
    _, _, segs = _prepare(short_seq)
    rng = np.random.default_rng(0)
    junk = np.column_stack([rng.uniform(1, 2, 3000), rng.uniform(1, 2, 3000), rng.uniform(0.02, 0.3, 3000)])
    r = estimate_poses([segs[0].points_floor, junk])
    assert r.status == "fail" and r.floor_from_body == []


def test_partial_turn_has_no_loop_closure(short_seq):
    _, f, segs = _prepare(short_seq)
    r = estimate_poses([s.points_floor for s in segs])
    names = {c["name"]: c for c in r.checks}
    assert names["turn_coverage_deg"]["status"] == "warn"
    assert all(e.kind != "closure" for e in r.edges)
    assert len(r.floor_from_body) == 4
    err = _pose_errors(short_seq, f, r.floor_from_body)
    assert err[:, 0].max() < 0.6 and err[:, 1].max() < 0.02


# ---- fusion ------------------------------------------------------------------------------------

def test_fusion_with_true_poses_reproduces_the_object(short_seq):
    _, f, segs = _prepare(short_seq)
    g = f @ short_seq.camera_from_floor
    cam_from_body = [invert(f) @ g @ short_seq.floor_from_body[k] @ invert(short_seq.floor_from_body[0]) @ invert(g)
                     for k in range(len(segs))]
    mesh, per_view, log = fuse_views(short_seq.views, [s.mask for s in segs], cam_from_body,
                                     TSDFParams(voxel_length_m=0.005, depth_trunc_m=4.0))
    assert len(mesh.triangles) > 1000 and log["integrated"].startswith("subject pixels only")
    z = np.asarray(mesh.vertices)[:, 2]
    assert 0.49 < z.max() < 0.515  # box height 0.5 m (TSDF surface within ~1 voxel)


# ---- end to end --------------------------------------------------------------------------------

@pytest.mark.slow
def test_full_turn_pipeline_end_to_end(tmp_path):
    seq = make_sequence(SceneConfig(n_stops=24, noise_sigma_m=0.003, dropout=0.02, frames_per_stop=3,
                                    transition_frames=4))
    r = run_self_rotation(seq.views, tmp_path, background_views=seq.background_views, run_id="t1",
                          source={"synthetic": True, "label": "test"})
    assert r.status in ("pass", "warn"), r.log.get("status_note")
    assert r.log["stops"]["n"] == 24
    loop = r.log["pose"]["loop_closure"]
    assert loop is not None and loop["rotation_deg"] < 3 and loop["translation_m"] < 0.025
    err = _pose_errors(seq, np.array(r.log["floor_from_camera"]), r.pose.floor_from_body)
    assert err[:, 0].max() < 1.0 and err[:, 1].max() < 0.015  # regression bounds (virtual sensor)
    assert 0.49 < r.log["mesh"]["bounds_m"][1][2] < 0.515

    names = {p.name for p in r.run_dir.iterdir()}
    for f in ("fused_mesh.ply", "mesh_coverage.ply", "views_colored.ply", "views_front.png", "views_top.png",
              "slices.png", "poses.png", "mesh_render.png", "coverage.png", "report.html", "run_log.json", "poses.json"):
        assert f in names, f
    for png in ("slices.png", "poses.png", "mesh_render.png"):
        assert cv2.imread(str(r.run_dir / png)) is not None
    log = json.loads((r.run_dir / "run_log.json").read_text(encoding="utf-8"))
    assert "synthetic data only" in log["note"] and log["outputs"]["fused_mesh.ply"]
    assert "Status:" in (r.run_dir / "report.html").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):  # runs are never overwritten
        run_self_rotation(seq.views[:3], tmp_path, run_id="t1")


def test_failed_run_writes_report_without_mesh(tmp_path):
    seq = make_sequence(SceneConfig(n_stops=4, step_deg=30, object_kind="cylinder"))
    r = run_self_rotation(seq.views, tmp_path, background_views=seq.background_views, run_id="f1",
                          params=PipelineParams(detect_stops=False))
    assert r.status == "fail" and r.mesh is None
    names = {p.name for p in r.run_dir.iterdir()}
    assert "fused_mesh.ply" not in names and {"report.html", "run_log.json"} <= names
