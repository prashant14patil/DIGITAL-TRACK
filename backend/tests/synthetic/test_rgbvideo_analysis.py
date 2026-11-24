"""RGB-only video analysis on a SYNTHETIC video: a figure in front of a wall and carpet that changes its width in
steps with pauses (like a person turning in place), encoded losslessly with ffmpeg. Skipped without ffmpeg."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from digitaltrack.rgbvideo.analyze import VideoAnalysisParams, analyze_video, display_rotate

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")

W, H, FPS = 240, 432, 30  # stored landscape-free portrait frame
WIDTHS = [90, 60, 30, 60, 90]  # figure width per stop (front, 3/4, side, 3/4, back)
STOP_FRAMES, MOVE_FRAMES = 20, 10


def _scene(width: int, rng) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    floor = int(H * 0.72)
    y = np.linspace(0, 1, floor)[:, None]
    img[:floor] = (np.array([228, 226, 222]) - 18 * y[..., None]).astype(np.uint8)  # wall with a gradient
    tiles = rng.integers(0, 2, ((H - floor) // 12 + 1, W // 12 + 1))  # blue carpet, fixed pattern
    carpet = np.where(np.kron(tiles, np.ones((12, 12)))[: H - floor, :W, None] > 0, [60, 70, 140], [80, 90, 160])
    img[floor:] = carpet.astype(np.uint8)
    cx, top, bottom = W // 2, int(H * 0.25), int(H * 0.80)
    cv2.rectangle(img, (cx - width // 2, top + 30), (cx + width // 2, bottom), (90, 110, 70), -1)  # body (olive)
    cv2.circle(img, (cx, top + 15), 15, (150, 110, 90), -1)  # head
    return img


def _figure_mask(width: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cx, top, bottom = W // 2, int(H * 0.25), int(H * 0.80)
    cv2.rectangle(m, (cx - width // 2, top + 30), (cx + width // 2, bottom), 1, -1)
    cv2.circle(m, (cx, top + 15), 15, 1, -1)
    return m.astype(bool)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    rng = np.random.default_rng(3)
    frames = []
    for k, w in enumerate(WIDTHS):
        if k:
            for j in range(1, MOVE_FRAMES + 1):
                frames.append(_scene(int(WIDTHS[k - 1] + (w - WIDTHS[k - 1]) * j / (MOVE_FRAMES + 1)), np.random.default_rng(3)))
        frames += [_scene(w, np.random.default_rng(3))] * STOP_FRAMES
    path = tmp_path_factory.mktemp("vid") / "synthetic.mkv"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@pytest.fixture(scope="module")
def result(video, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis")
    params = VideoAnalysisParams(analysis_width=120, segment_height=432, background_sample_every=5, min_stop_s=0.3)
    return analyze_video(video, out, params=params, label="synthetic", run_id="r1"), out / "r1", video


def test_display_rotation():
    img = np.arange(6).reshape(2, 3)
    assert display_rotate(img, -90).shape == (3, 2)
    assert display_rotate(img, -90)[0, 0] == img[1, 0]  # 90 deg clockwise
    assert np.array_equal(display_rotate(img, 0), img)


def test_stops_found_with_keyframes_inside(result):
    r, _, _ = result
    assert len(r["stops"]) == len(WIDTHS)
    for k, s in enumerate(r["stops"]):
        start = k * (STOP_FRAMES + MOVE_FRAMES)
        assert start - 1 <= s["start_frame"] and s["end_frame"] <= start + STOP_FRAMES
        assert s["start_frame"] <= s["keyframe"] <= s["end_frame"]


def test_silhouettes_match_the_figure_above_the_floor_junction(result):
    """Above the wall/floor junction the silhouette must match. Below it, without an empty-scene recording, the
    floor behind a subject that never leaves its spot is unknown; the module must FLAG that, not hide it."""
    r, run, _ = result
    junction = int(r["background_model"]["floor_row_fraction"] * H) - int(0.03 * H) - 2
    for k, s in enumerate(r["stops"]):
        m = cv2.imdecode(np.fromfile(run / s["silhouette_png"], np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        truth = _figure_mask(WIDTHS[k])
        a, b = m[:junction], truth[:junction]
        iou = (a & b).sum() / (a | b).sum()
        assert iou > 0.9, (k, iou)
        assert "feet_floor_contact_unreliable" in s["silhouette"]["flags"]


def test_outputs_and_source_untouched(result):
    r, run, video = result
    assert r["source"]["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    for f in ("analysis.json", "report.html", "timeline.png", "silhouettes_overlay.png"):
        assert (run / f).is_file(), f
    kf = run / r["stops"][0]["keyframe_png"]
    assert hashlib.sha256(kf.read_bytes()).hexdigest() == r["stops"][0]["keyframe_sha256"]
    data = json.loads((run / "analysis.json").read_text(encoding="utf-8"))
    assert data["depth_present"] is False and data["missing_for_3d"]
    assert "not metric" in data["note"] or "nothing here is metric" in data["note"]
    assert "No 3-D reconstruction" in (run / "report.html").read_text(encoding="utf-8")


def test_refuses_to_overwrite_a_run(result):
    _, run, video = result
    with pytest.raises(FileExistsError):
        analyze_video(video, run.parent, run_id="r1")
