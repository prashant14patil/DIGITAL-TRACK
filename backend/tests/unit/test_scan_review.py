import json

import cv2
import numpy as np
import open3d as o3d

from digitaltrack.experimental.scan_review import quality_checks, rough_slices, write_review_files


def _run(tmp_path, boxes, times=(0, 5, 10, 15, 20)):
    """A run folder with rectangular silhouettes (200x100 image); boxes = (top, bottom, left, right) per frame."""
    (tmp_path / "silhouettes").mkdir()
    for i, (top, bottom, left, right) in enumerate(boxes):
        m = np.zeros((200, 100), np.uint8)
        m[top:bottom, left:right] = 255
        cv2.imencode(".png", m)[1].tofile(tmp_path / "silhouettes" / f"mask_{i:06d}.png")
    n = len(boxes) - 1
    anchors = [{"frame_id": round(n * k / 4), "time_s": s, "angle_deg": 90 * k} for k, s in enumerate(times)]
    (tmp_path / "report.json").write_text(json.dumps({"angles": {"anchors": anchors}}))
    return {c["id"]: c["status"] for c in quality_checks(tmp_path)["checks"]}


def test_good_recording_has_no_warnings(tmp_path):
    assert set(_run(tmp_path, [(40, 180, 30, 70)] * 80).values()) == {"ok"}


def test_each_problem_is_flagged(tmp_path):
    boxes = [(10, 200, 2, 60)] * 20  # cut off at top, bottom and side; few frames
    boxes[-1] = (10, 200, 2, 20)  # last pose narrower (but a rectangle stays a rectangle after alignment)
    s = _run(tmp_path, boxes, times=(0, 2, 10, 12, 14))
    for cid in ("head_room", "feet_room", "side_room", "frame_count", "turn_speed"):
        assert s[cid] == "warn", cid


def test_slices_of_a_cylinder(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=1.0, resolution=200, split=4)
    path = tmp_path / "mesh_height_normalised.ply"
    o3d.io.write_triangle_mesh(str(path), mesh)
    items = rough_slices(path)["items"]
    assert all(i["status"] == "unreliable" for i in items)
    assert abs(items[0]["value"] - 100 * 2 * np.pi * 0.1) < 0.2


def test_no_silhouettes_means_no_quality_file(tmp_path):
    write_review_files(tmp_path)
    assert not (tmp_path / "quality.json").exists() and not (tmp_path / "slices.json").exists()
