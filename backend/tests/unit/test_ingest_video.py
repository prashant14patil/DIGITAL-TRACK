"""Camera-app video ingestion on synthetic videos made with ffmpeg (no real iPad recording involved)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.validate import validate_session
from digitaltrack.ingest import session as ingest_session
from digitaltrack.ingest.errors import IngestError
from digitaltrack.ingest.metadata import validate_capture_metadata
from digitaltrack.ingest.session import ingest_video, ingest_video_from_upload
from digitaltrack.ingest.video import check_source_file, frame_timestamps, probe_video

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not (FFMPEG and shutil.which("ffprobe")),
                                reason="ffmpeg/ffprobe not on PATH (needed to build synthetic test videos)")

MD = validate_capture_metadata({"subject_id": "S001", "setup_id": "lab-A", "protocol_version": "capture-0.3-draft",
                                "capture_type": "rgb_video", "recording_app": "Camera"})


def _frames(n=6, h=48, w=64, dtype=np.uint8, seed=0):
    rng = np.random.default_rng(seed)
    hi = 256 if dtype == np.uint8 else 65536
    return [rng.integers(0, hi, (h, w, 3), dtype=dtype) for _ in range(n)]


def make_video(path: Path, frames, *, fps=10, codec="png", pre_input=(), post=()):
    h, w = frames[0].shape[:2]
    pix = "rgb24" if frames[0].dtype == np.uint8 else "rgb48le"
    data = b"".join(np.ascontiguousarray(f.astype("<u2") if pix == "rgb48le" else f).tobytes() for f in frames)
    cmd = [FFMPEG, "-v", "error", *pre_input, "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0", "-c:v", codec, *post, str(path)]
    subprocess.run(cmd, input=data, check=True, capture_output=True)
    return path


def read_png_rgb(p: Path):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)


# ---- pre-checks ----------------------------------------------------------------------------------

def test_source_file_checks(tmp_path):
    with pytest.raises(IngestError) as e:
        check_source_file(tmp_path / "nope.mov")
    assert e.value.code == "source_not_found"
    with pytest.raises(IngestError) as e:
        check_source_file(tmp_path)
    assert e.value.code == "source_not_a_file"
    (tmp_path / "empty.mov").write_bytes(b"")
    with pytest.raises(IngestError) as e:
        check_source_file(tmp_path / "empty.mov")
    assert e.value.code == "source_empty"
    (tmp_path / "clip.avi").write_bytes(b"x" * 20)
    with pytest.raises(IngestError) as e:
        check_source_file(tmp_path / "clip.avi")
    assert e.value.code == "unsupported_extension"
    (tmp_path / "fake.mov").write_bytes(b"\x00" * 4 + b"JUNK" + b"\x00" * 50)
    with pytest.raises(IngestError) as e:
        check_source_file(tmp_path / "fake.mov")
    assert e.value.code == "not_quicktime_mp4"


def test_checksum_and_size_expectations(tmp_path):
    v = make_video(tmp_path / "c.mov", _frames(2))
    good = hashlib.sha256(v.read_bytes()).hexdigest()
    assert check_source_file(v, good, v.stat().st_size).sha256 == good
    with pytest.raises(IngestError) as e:
        check_source_file(v, "0" * 64)
    assert e.value.code == "checksum_mismatch"
    with pytest.raises(IngestError) as e:
        check_source_file(v, expected_size=1)
    assert e.value.code == "size_mismatch"


# ---- probe ---------------------------------------------------------------------------------------

def test_probe_reports_stream_facts_and_flags_without_leaking_location(tmp_path):
    plain = make_video(tmp_path / "p.mov", _frames(4), post=("-metadata", "location=+12.3456+045.6789/"))
    rot = tmp_path / "r.mov"
    subprocess.run([FFMPEG, "-v", "error", "-display_rotation", "90", "-i", str(plain), "-c", "copy", str(rot)],
                   check=True, capture_output=True)
    p = probe_video(plain)
    assert (p["codec"], p["width"], p["height"], p["bit_depth"], p["r_frame_rate"]) == ("png", 64, 48, 8, "10/1")
    assert p["has_location_metadata"] and "location_metadata_present_not_copied" in p["flags"]
    assert "12.3456" not in json.dumps(p)
    r = probe_video(rot)
    assert r["display_rotation_deg"] == 90.0 and "display_rotation_not_applied" in r["flags"]
    assert frame_timestamps(plain, p["video_stream_index"]) == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_probe_rejects_files_without_video(tmp_path):
    a = tmp_path / "audio.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.5", "-c:a", "aac", str(a)],
                   check=True, capture_output=True)
    with pytest.raises(IngestError) as e:
        probe_video(a)
    assert e.value.code == "no_video_stream"


# ---- ingest --------------------------------------------------------------------------------------

def test_lossless_ingest_is_pixel_exact_and_valid(tmp_path):
    frames = _frames(6)
    src = make_video(tmp_path / "clip.mov", frames)
    res = ingest_video(src, tmp_path / "ses", MD, synthetic=True)
    assert (res.n_frames, res.n_decoded) == (6, 6)
    assert res.validation_codes == ["angles_missing", "calibration_missing"]  # expected: open items
    rep = validate_session(res.session_dir)
    assert rep.ok
    s = load_session(res.session_dir)
    assert s.input_type == "video_rgb" and not s.angles_complete
    for fr, ref in zip(s.frames, frames):
        assert np.array_equal(fr.load_rgb(), ref)
        assert fr.meta["angle_source"] == "none" and "turntable_angle_deg" not in fr.meta
    assert [f.timestamp_s for f in s.frames] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert [f.meta["video"]["source_frame_index"] for f in s.frames] == list(range(6))
    sv = s.meta["source_video"]
    assert sv["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest() and sv["original_filename"] == "clip.mov"
    stored = res.session_dir / sv["file"]
    assert stored.read_bytes() == src.read_bytes() and not os.access(stored, os.W_OK)
    fx = s.meta["frame_extraction"]
    assert fx["parameters"]["noautorotate"] is True and fx["parameters"]["frames_in_stream"] == 6
    assert "ffmpeg" in fx["software"]["ffmpeg"].lower()
    assert s.meta["capture"]["subject_on_turntable"] is False and s.meta["capture"]["rotation_source"] == "none"


def test_sixteen_bit_source_gives_sixteen_bit_frames(tmp_path):
    frames = _frames(3, dtype=np.uint16)
    src = make_video(tmp_path / "v16.mov", frames, post=("-pix_fmt", "rgb48be"))
    res = ingest_video(src, tmp_path / "ses", MD)
    s = load_session(res.session_dir)
    assert "high_bit_depth" in res.warnings
    assert s.meta["frame_extraction"]["parameters"]["png_bit_depth"] == 16
    for fr, ref in zip(s.frames, frames):
        assert fr.meta["rgb"]["bit_depth"] == 16 and np.array_equal(fr.load_rgb(), ref)


def test_frame_step_keeps_source_indices_and_timestamps(tmp_path):
    src = make_video(tmp_path / "clip.mov", _frames(7))
    res = ingest_video(src, tmp_path / "ses", MD, frame_step=3)
    s = load_session(res.session_dir)
    assert [f.meta["video"]["source_frame_index"] for f in s.frames] == [0, 3, 6]
    assert [f.timestamp_s for f in s.frames] == pytest.approx([0.0, 0.3, 0.6])
    assert (res.n_frames, res.n_decoded) == (3, 7)


def test_lossy_h264_ingest_accounts_for_every_frame(tmp_path):
    src = make_video(tmp_path / "clip.mp4", _frames(12, h=64, w=96), codec="libx264", post=("-pix_fmt", "yuv420p"))
    res = ingest_video(src, tmp_path / "ses", MD)
    assert (res.n_frames, res.n_decoded) == (12, 12) and validate_session(res.session_dir).ok
    ts = [f.timestamp_s for f in load_session(res.session_dir).frames]
    assert ts == sorted(ts) and len(set(ts)) == 12


def test_guards_and_refusals(tmp_path):
    src = make_video(tmp_path / "clip.mov", _frames(5))
    with pytest.raises(IngestError) as e:
        ingest_video(src, tmp_path / "s1", MD, max_frames=4)
    assert e.value.code == "too_many_frames"
    with pytest.raises(IngestError) as e:
        ingest_video(src, tmp_path / "s2", MD, frame_step=0)
    assert e.value.code == "invalid_parameter"
    rgbd = validate_capture_metadata({**{k: getattr(MD, k) for k in ("subject_id", "setup_id", "protocol_version")},
                                      "capture_type": "rgbd"})
    with pytest.raises(IngestError) as e:
        ingest_video(src, tmp_path / "s3", rgbd)
    assert e.value.code == "wrong_capture_type"  # no RGB-D adapter exists
    (tmp_path / "exists").mkdir()
    with pytest.raises(IngestError) as e:
        ingest_video(src, tmp_path / "exists", MD)
    assert e.value.code == "destination_exists"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(("s1", "s2", "s3"))]


def test_failed_ingest_leaves_nothing_behind(tmp_path, monkeypatch):
    src = make_video(tmp_path / "clip.mov", _frames(5))
    real = ingest_session.decode_frames

    def short(*a, **k):  # decoder silently drops the last frame
        frames = list(real(*a, **k))
        yield from frames[:-1]
    monkeypatch.setattr(ingest_session, "decode_frames", short)
    with pytest.raises(IngestError) as e:
        ingest_video(src, tmp_path / "ses", MD)
    assert e.value.code == "frame_count_mismatch"
    assert not (tmp_path / "ses").exists() and not list(tmp_path.glob("ses.partial-*"))


# ---- ingest from a finalized raw upload (api/storage.py layout, read-only) ------------------------

def _fake_upload(root: Path, video: Path, rel: str, *, finalize=True, tamper=False) -> Path:
    up = root / "up_20261123T000000Z_abcdef"
    dst = up / "export" / Path(*rel.split("/"))
    dst.parent.mkdir(parents=True)
    shutil.copyfile(video, dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    if tamper:
        dst.write_bytes(dst.read_bytes()[:-1] + b"\x00")
    if finalize:
        (up / "manifest.json").write_text(json.dumps(
            {"upload_id": up.name, "files": [{"path": rel, "size": video.stat().st_size, "sha256": sha}]}))
    for p in up.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)
    return up


def test_ingest_from_finalized_upload_verifies_manifest(tmp_path):
    v = make_video(tmp_path / "IMG_0001.MOV", _frames(3))
    up = _fake_upload(tmp_path / "u", v, "export_folder/IMG_0001.MOV")
    before = {p: p.read_bytes() for p in up.rglob("*") if p.is_file()}
    res = ingest_video_from_upload(up, "export_folder/IMG_0001.MOV", tmp_path / "ses", MD)
    assert validate_session(res.session_dir).ok
    assert {p: p.read_bytes() for p in up.rglob("*") if p.is_file()} == before  # upload untouched


def test_upload_problems_are_explicit(tmp_path):
    v = make_video(tmp_path / "IMG.MOV", _frames(2))
    up = _fake_upload(tmp_path / "a", v, "IMG.MOV", finalize=False)
    with pytest.raises(IngestError) as e:
        ingest_video_from_upload(up, "IMG.MOV", tmp_path / "s1", MD)
    assert e.value.code == "upload_not_finalized"
    up = _fake_upload(tmp_path / "b", v, "IMG.MOV")
    with pytest.raises(IngestError) as e:
        ingest_video_from_upload(up, "other.MOV", tmp_path / "s2", MD)
    assert e.value.code == "upload_file_not_listed"
    up = _fake_upload(tmp_path / "c", v, "IMG.MOV", tamper=True)
    with pytest.raises(IngestError) as e:
        ingest_video_from_upload(up, "IMG.MOV", tmp_path / "s3", MD)
    assert e.value.code == "checksum_mismatch"
