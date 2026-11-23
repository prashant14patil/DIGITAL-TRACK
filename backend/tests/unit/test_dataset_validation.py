import json
import os
import stat

import numpy as np
import pytest

from digitaltrack import DATASET_SCHEMA_VERSION as SCHEMA_VERSION
from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.validate import validate_session
from digitaltrack.dataset.writer import FrameData, write_session
from digitaltrack.synthetic.renderer import RenderConfig, render_session
from digitaltrack.synthetic.shapes import cylinder


def _edit_json(path, fn):
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


def test_clean_session_validates_with_no_issues(small_cylinder_session):
    rep = validate_session(small_cylinder_session)
    assert rep.ok and rep.issues == [], [str(i) for i in rep.issues]


def test_missing_directory():
    assert "missing_directory" in validate_session("does/not/exist").codes()


def test_missing_depth_file(mutable_session):
    (mutable_session / "frames/frame_000002/depth_raw.bin").unlink()
    rep = validate_session(mutable_session)
    assert not rep.ok and "missing_file" in rep.codes()


def test_corrupted_depth_detected_by_checksum(mutable_session):
    p = mutable_session / "frames/frame_000003/depth_raw.bin"
    data = bytearray(p.read_bytes())
    data[100] ^= 0xFF
    p.write_bytes(bytes(data))
    rep = validate_session(mutable_session)
    assert not rep.ok and "checksum_mismatch" in rep.codes()
    # same corruption is invisible to structural checks, which is why checksums exist
    assert "checksum_mismatch" not in validate_session(mutable_session, check_checksums=False).codes()


def test_truncated_depth_file(mutable_session):
    p = mutable_session / "frames/frame_000001/depth_raw.bin"
    p.write_bytes(p.read_bytes()[:-4])
    assert "bad_file_size" in validate_session(mutable_session).codes()


def test_schema_violation_in_session_json(mutable_session):
    _edit_json(mutable_session / "session.json", lambda d: d.pop("calibration_id"))
    assert "schema_violation" in validate_session(mutable_session).codes()


def test_calibration_id_mismatch(mutable_session):
    _edit_json(mutable_session / "calibration/camera_calibration.json",
               lambda d: d.__setitem__("calibration_id", "other"))
    assert "calibration_id_mismatch" in validate_session(mutable_session).codes()


def test_depth_resolution_must_match_intrinsics(mutable_session):
    _edit_json(mutable_session / "calibration/camera_calibration.json",
               lambda d: d.__setitem__("image_width", d["image_width"] + 1))
    assert "resolution_mismatch" in validate_session(mutable_session).codes()


def test_turntable_rotation_must_be_proper(mutable_session):
    def scale(d):
        d["camera_from_turntable"][0][0] *= 2.0
    _edit_json(mutable_session / "calibration/turntable_calibration.json", scale)
    assert "bad_rotation" in validate_session(mutable_session).codes()


def test_unlisted_frame_directory_is_a_warning(mutable_session):
    (mutable_session / "frames/frame_000099").mkdir()
    rep = validate_session(mutable_session)
    assert rep.ok and "unlisted_frame_dir" in rep.codes()


def test_negative_depth_is_an_error(mutable_session):
    p = mutable_session / "frames/frame_000001/depth_raw.bin"
    arr = np.fromfile(p, dtype="<f4")
    arr[5] = -1.0
    arr.tofile(p)
    rep = validate_session(mutable_session, check_checksums=False)
    assert "negative_depth" in rep.codes()


def test_frame_angle_must_agree_between_manifest_and_frame_metadata(mutable_session):
    _edit_json(mutable_session / "frames/frame_000002/frame_metadata.json",
               lambda d: d.__setitem__("turntable_angle_deg", 1234.0))
    assert "angle_mismatch" in validate_session(mutable_session).codes()


# ---- raw data integrity / immutability -------------------------------------------------------

def test_writer_refuses_to_overwrite_existing_session(small_cylinder_session):
    with pytest.raises(FileExistsError):
        render_session(cylinder(), small_cylinder_session, RenderConfig(n_views=2))


def test_raw_files_are_marked_read_only(small_cylinder_session):
    depth = small_cylinder_session / "frames/frame_000001/depth_raw.bin"
    assert not os.access(depth, os.W_OK)
    assert not (depth.stat().st_mode & stat.S_IWRITE)
    with pytest.raises(PermissionError):
        open(depth, "ab")


def test_depth_roundtrip_is_bit_exact_float32(tmp_path):
    rng = np.random.default_rng(1)
    depth = (rng.random((8, 6)) * 3).astype(np.float32)
    depth[0, 0] = np.float32(1.2345678)
    depth[1, 1] = 0.0
    conf = rng.integers(0, 3, (8, 6)).astype(np.uint8)
    cam = {"schema_version": SCHEMA_VERSION, "calibration_id": "c", "camera_frame": "opencv_x_right_y_down_z_forward",
           "pixel_convention": "index_center", "image_source": "depth", "image_width": 6, "image_height": 8,
           "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 2.5, "cy": 3.5}}
    tt = {"schema_version": SCHEMA_VERSION, "calibration_id": "c", "method": "test", "turntable_frame": "t",
          "camera_from_turntable": np.eye(4).tolist(), "angle_direction_sign": 1}
    meta = {"session_id": "rt", "created_utc": "now", "synthetic": True, "input_type": "synthetic_depth",
            "device": {"model": "m", "os_version": "o", "app_version": "a"},
            "capture": {"camera_fixed": True, "subject_on_turntable": True, "depth_source": "x",
                        "depth_unit": "meters", "depth_dtype": "float32", "rotation_source": "manual",
                        "rotation_axis_frame": "t"},
            "calibration_id": "c"}
    root = write_session(tmp_path / "rt", session_meta=meta, camera_calibration=cam, turntable_calibration=tt,
                         frames=[FrameData(depth, conf, 0.0, 12.5, "manual")])
    assert validate_session(root).ok
    s = load_session(root)
    got = s.frames[0].load_depth()
    assert got.dtype == np.float32 and np.array_equal(got, depth)
    assert np.array_equal(s.frames[0].load_confidence(), conf)
    assert (root / "frames/frame_000001/depth_raw.bin").stat().st_size == 8 * 6 * 4


def test_writer_rejects_wrong_frame_shape(tmp_path, monkeypatch):
    from digitaltrack.synthetic import renderer
    monkeypatch.setattr(renderer, "render_frame",
                        lambda *a, **k: (np.zeros((3, 3), np.float32), np.zeros((3, 3), np.uint8)))
    with pytest.raises(ValueError):
        render_session(cylinder(), tmp_path / "bad", RenderConfig(n_views=1))
