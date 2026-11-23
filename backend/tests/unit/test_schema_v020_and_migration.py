"""Dataset schema 0.2.0: RGB-only sessions, source video, and backward compatibility with 0.1.0.

The 0.1.0 reference is a FROZEN session written by the unmodified 0.1.0 writer (tests/fixtures/README.md).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

from digitaltrack import DATASET_SCHEMA_VERSION
from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.migrate import MigrationError, migrate_session_copy, upgrade_camera_010
from digitaltrack.dataset.schemas import schema_errors
from digitaltrack.dataset.validate import validate_session
from digitaltrack.dataset.writer import FrameData, encode_png, sha256_file, write_session


FIXTURE_V010 = Path(__file__).resolve().parents[1] / "fixtures" / "session_v0_1_0_cylinder"


def make_writable(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _edit_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fn(obj)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture()
def v010_copy(tmp_path) -> Path:
    dst = tmp_path / "v010"
    shutil.copytree(FIXTURE_V010, dst)
    make_writable(dst)
    return dst


# ---- 0.1.0 backward compatibility -------------------------------------------------------------

def test_frozen_v010_fixture_is_really_v010():
    s = json.loads((FIXTURE_V010 / "session.json").read_text(encoding="utf-8"))
    assert s["schema_version"] == "0.1.0" and "input_type" not in s
    # it does NOT satisfy the current schema, so passing validation below proves version dispatch works
    assert schema_errors(s, "session")


def test_v010_session_still_validates_cleanly():
    before = tree_hashes(FIXTURE_V010)
    rep = validate_session(FIXTURE_V010)
    assert rep.ok and rep.issues == [], [str(i) for i in rep.issues]
    assert rep.schema_version == "0.1.0"
    assert tree_hashes(FIXTURE_V010) == before  # validation never modifies data


def test_v010_session_loads_upgraded_in_memory_without_touching_disk():
    before = tree_hashes(FIXTURE_V010)
    s = load_session(FIXTURE_V010)
    assert s.schema_version_on_disk == "0.1.0"
    assert s.meta["schema_version"] == DATASET_SCHEMA_VERSION and s.input_type == "synthetic_depth"
    assert s.camera["image_source"] == "depth" and s.has_depth and s.angles_complete
    raw = np.fromfile(FIXTURE_V010 / "frames/frame_000001/depth_raw.bin", dtype="<f4").reshape(64, 48)
    assert np.array_equal(s.frames[0].load_depth(), raw)
    assert tree_hashes(FIXTURE_V010) == before


def test_v010_integrity_checks_still_enforced(v010_copy):
    p = v010_copy / "frames/frame_000004/depth_raw.bin"
    b = bytearray(p.read_bytes())
    b[40] ^= 0xFF
    p.write_bytes(bytes(b))
    assert "checksum_mismatch" in validate_session(v010_copy).codes()


def test_v010_schema_violation_still_detected(v010_copy):
    _edit_json(v010_copy / "frames/frame_000002/frame_metadata.json", lambda d: d.pop("depth"))
    assert "schema_violation" in validate_session(v010_copy).codes()


def test_mixed_schema_versions_rejected(v010_copy):
    _edit_json(v010_copy / "frames/frame_000003/frame_metadata.json",
               lambda d: d.__setitem__("schema_version", DATASET_SCHEMA_VERSION))
    assert "mixed_schema_versions" in validate_session(v010_copy).codes()


@pytest.mark.parametrize("version", ["0.0.9", "9.0.0", None])
def test_unsupported_schema_version_rejected(v010_copy, version):
    _edit_json(v010_copy / "session.json", lambda d: d.__setitem__("schema_version", version))
    rep = validate_session(v010_copy)
    assert not rep.ok and "unsupported_schema_version" in rep.codes()


# ---- migration (copy, never in place) ---------------------------------------------------------

def test_migrated_copy_is_valid_lossless_and_source_untouched(tmp_path):
    before = tree_hashes(FIXTURE_V010)
    dst = migrate_session_copy(FIXTURE_V010, tmp_path / "migrated")
    assert tree_hashes(FIXTURE_V010) == before

    rep = validate_session(dst)
    assert rep.ok and rep.issues == [] and rep.schema_version == DATASET_SCHEMA_VERSION, [str(i) for i in rep.issues]
    new = json.loads((dst / "session.json").read_text(encoding="utf-8"))
    assert new["migrated_from"]["schema_version"] == "0.1.0"
    assert new["migrated_from"]["session_json_sha256"] == sha256_file(FIXTURE_V010 / "session.json")

    # every raw binary (and the synthetic ground truth) is byte-identical
    new_hashes = tree_hashes(dst)
    for rel, h in before.items():
        if rel.endswith(".bin") or rel == "ground_truth.json":
            assert new_hashes[rel] == h, rel

    a, b = load_session(FIXTURE_V010), load_session(dst)
    assert a.intrinsics == b.intrinsics
    assert np.array_equal(a.camera_from_turntable, b.camera_from_turntable)
    assert a.angle_direction_sign == b.angle_direction_sign
    assert [f.angle_deg for f in a.frames] == [f.angle_deg for f in b.frames]
    assert [f.timestamp_s for f in a.frames] == [f.timestamp_s for f in b.frames]
    for fa, fb in zip(a.frames, b.frames):
        assert np.array_equal(fa.load_depth(), fb.load_depth())
        assert np.array_equal(fa.load_confidence(), fb.load_confidence())


def test_migration_refuses_non_empty_destination(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "x.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        migrate_session_copy(FIXTURE_V010, tmp_path / "d")
    assert (tmp_path / "d" / "x.txt").read_text() == "keep"


def test_migration_refuses_current_version(tmp_path):
    dst = migrate_session_copy(FIXTURE_V010, tmp_path / "m1")
    with pytest.raises(MigrationError):
        migrate_session_copy(dst, tmp_path / "m2")


def test_migration_refuses_invalid_source(v010_copy, tmp_path):
    (v010_copy / "frames/frame_000001/confidence.bin").unlink()
    with pytest.raises(MigrationError):
        migrate_session_copy(v010_copy, tmp_path / "m")
    assert not (tmp_path / "m").exists()


def test_migration_never_guesses_unknown_distortion():
    cam = json.loads((FIXTURE_V010 / "calibration/camera_calibration.json").read_text(encoding="utf-8"))
    cam["distortion"] = {"k1": 0.1}
    with pytest.raises(MigrationError):
        upgrade_camera_010(cam)
    cam["distortion"] = {}
    assert "distortion" not in upgrade_camera_010(cam)


# ---- 0.2.0 RGB-only sessions ------------------------------------------------------------------

def _rgb_meta(input_type="synthetic_rgb", **extra):
    return {"session_id": "rgb1", "created_utc": "now", "synthetic": input_type != "video_rgb", "input_type": input_type,
            "capture": {"camera_fixed": True, "subject_on_turntable": True, "rotation_source": "none",
                        "rotation_axis_frame": "turntable"},
            "calibration_id": "cal1", **extra}


def _rgb_frames(n=3, h=10, w=8, dtype=np.uint8, seed=0):
    rng = np.random.default_rng(seed)
    hi = 255 if dtype == np.uint8 else 65535
    return [FrameData(None, None, 0.1 * i, None, "none", rgb=rng.integers(0, hi + 1, (h, w, 3)).astype(dtype),
                      video_frame_index=3 * i, video_pts_s=0.1 * i) for i in range(n)]


def _rgb_cam(w=8, h=10):
    return {"schema_version": DATASET_SCHEMA_VERSION, "calibration_id": "cal1",
            "camera_frame": "opencv_x_right_y_down_z_forward", "pixel_convention": "index_center",
            "image_source": "rgb", "image_width": w, "image_height": h,
            "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 3.5, "cy": 4.5},
            "distortion": {"model": "opencv_brown_conrady", "coefficients": [0.01, -0.02, 0.0, 0.0, 0.0]}}


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_rgb_only_session_roundtrip_is_bit_exact(tmp_path, dtype):
    frames = _rgb_frames(dtype=dtype)
    root = write_session(tmp_path / "s", session_meta=_rgb_meta(), camera_calibration=None,
                         turntable_calibration=None, frames=frames)
    rep = validate_session(root)
    assert rep.ok, [str(i) for i in rep.errors]
    assert {"calibration_missing", "angles_missing"} <= rep.codes()
    s = load_session(root)
    assert not s.has_depth and not s.angles_complete and s.schema_version_on_disk == DATASET_SCHEMA_VERSION
    for fr, src in zip(s.frames, frames):
        got = fr.load_rgb()
        assert got.dtype == dtype and np.array_equal(got, src.rgb)
        assert fr.meta["video"]["source_frame_index"] == src.video_frame_index
    with pytest.raises(ValueError):
        s.intrinsics  # no calibration: fail loudly, never default


def test_png_bytes_are_deterministic():
    img = _rgb_frames(1)[0].rgb
    assert encode_png(img) == encode_png(img.copy())


def test_tsdf_refuses_rgb_only_session(tmp_path):
    from digitaltrack.reconstruction.tsdf import reconstruct_tsdf

    root = write_session(tmp_path / "s", session_meta=_rgb_meta(), camera_calibration=_rgb_cam(),
                         turntable_calibration=None, frames=_rgb_frames())
    with pytest.raises(ValueError, match="depth"):
        reconstruct_tsdf(load_session(root), write=False)


def test_rgb_resolution_must_match_rgb_intrinsics(tmp_path):
    root = write_session(tmp_path / "s", session_meta=_rgb_meta(), camera_calibration=_rgb_cam(),
                         turntable_calibration=None, frames=_rgb_frames(), readonly=False)
    assert validate_session(root).ok
    _edit_json(root / "calibration/camera_calibration.json", lambda d: d.__setitem__("image_width", 9))
    assert "resolution_mismatch" in validate_session(root).codes()


def test_rgb_header_and_checksum_checks(tmp_path):
    root = write_session(tmp_path / "s", session_meta=_rgb_meta(), camera_calibration=None,
                         turntable_calibration=None, frames=_rgb_frames(), readonly=False)
    (root / "frames/frame_000002/rgb.png").write_bytes(encode_png(np.zeros((4, 4, 3), np.uint8)))
    codes = validate_session(root).codes()
    assert {"image_header_mismatch", "checksum_mismatch"} <= codes
    (root / "frames/frame_000003/rgb.png").write_bytes(b"not a png at all, just bytes......")
    assert "bad_image_format" in validate_session(root).codes()


def test_angle_source_none_with_angle_is_inconsistent(tmp_path):
    frames = _rgb_frames(1)
    frames[0].turntable_angle_deg = 10.0
    root = write_session(tmp_path / "s", session_meta=_rgb_meta(), camera_calibration=None,
                         turntable_calibration=None, frames=frames)
    assert "angle_source_inconsistent" in validate_session(root).codes()


def test_depth_input_type_requires_depth_frames(tmp_path):
    meta = _rgb_meta("synthetic_depth")
    meta["capture"].update(depth_source="x", depth_unit="meters", depth_dtype="float32")
    root = write_session(tmp_path / "s", session_meta=meta, camera_calibration=_rgb_cam(),
                         turntable_calibration={"schema_version": DATASET_SCHEMA_VERSION, "calibration_id": "cal1",
                                                "method": "t", "turntable_frame": "t",
                                                "camera_from_turntable": np.eye(4).tolist(), "angle_direction_sign": 1},
                         frames=_rgb_frames())
    assert "missing_depth" in validate_session(root).codes()


# ---- 0.2.0 video sessions ---------------------------------------------------------------------

def _video_meta():
    return _rgb_meta("video_rgb", frame_extraction={"method": "test", "parameters": {}, "software": {}},
                     source_video={"probe": {"codec": "test"}})


def test_video_session_requires_source_video_and_extraction(tmp_path):
    meta = _rgb_meta("video_rgb")
    with pytest.raises(ValueError, match="source_video|frame_extraction"):
        write_session(tmp_path / "s", session_meta=meta, camera_calibration=None, turntable_calibration=None,
                      frames=_rgb_frames())


def test_video_source_is_copied_verbatim_and_protected(tmp_path):
    src = tmp_path / "IMG_0001.MOV"
    src.write_bytes(bytes(range(256)) * 50)
    root = write_session(tmp_path / "s", session_meta=_video_meta(), camera_calibration=None,
                         turntable_calibration=None, frames=_rgb_frames(), source_video=src, readonly=False)
    s = json.loads((root / "session.json").read_text(encoding="utf-8"))
    sv = s["source_video"]
    assert sv["file"] == "source/source_video.mov" and sv["original_filename"] == "IMG_0001.MOV"
    assert sv["probe"] == {"codec": "test"}  # caller-provided keys kept
    assert (root / sv["file"]).read_bytes() == src.read_bytes() and sv["sha256"] == sha256_file(src)
    assert validate_session(root).ok

    p = root / sv["file"]
    b = bytearray(p.read_bytes())
    b[7] ^= 1
    p.write_bytes(bytes(b))
    assert "checksum_mismatch" in validate_session(root).codes()


def test_video_frames_must_reference_source_frame(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * 100)
    frames = _rgb_frames()
    frames[1].video_frame_index = None
    root = write_session(tmp_path / "s", session_meta=_video_meta(), camera_calibration=None,
                         turntable_calibration=None, frames=frames, source_video=src)
    assert "missing_video_reference" in validate_session(root).codes()
