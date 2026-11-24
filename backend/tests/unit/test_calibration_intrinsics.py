"""Intrinsics calibration against synthetic chessboard views with exactly known intrinsics.

Pinhole renders without lens distortion: this validates the code path, not real iPad optics.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from digitaltrack import DATASET_SCHEMA_VERSION
from digitaltrack.calibration.intrinsics import (CalibrationError, ChessboardSpec, attach_camera_calibration,
                                                 calibrate_intrinsics, camera_calibration_document, detect_chessboard)
from digitaltrack.dataset.schemas import schema_errors
from digitaltrack.dataset.validate import validate_session
from digitaltrack.dataset.writer import FrameData, write_session
from digitaltrack.synthetic.board import random_poses, render_view

SPEC = ChessboardSpec(9, 6, 0.025)
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
SIZE = (640, 480)


@pytest.fixture(scope="module")
def views():
    return [render_view(SPEC, K, r, t, SIZE, noise_sigma=2, seed=i)
            for i, (r, t) in enumerate(random_poses(SPEC, 25, seed=2))]


def test_detects_every_synthetic_view(views):
    assert all(detect_chessboard(v, SPEC) is not None for v in views)


def test_recovers_known_intrinsics_without_distortion_model(views):
    res = calibrate_intrinsics(views, SPEC, estimate_distortion=False)
    k = res.intrinsics
    assert k["fx"] == pytest.approx(600, rel=0.005) and k["fy"] == pytest.approx(600, rel=0.005)
    assert abs(k["cx"] - 320) < 2 and abs(k["cy"] - 240) < 2
    assert res.rms_reprojection_px < 0.5 and len(res.views_used) == 25 and not res.views_rejected
    assert np.allclose(res.dist_coeffs, 0)


def test_recovers_known_intrinsics_with_distortion_estimated(views):
    res = calibrate_intrinsics(views, SPEC)
    k = res.intrinsics
    assert k["fx"] == pytest.approx(600, rel=0.01) and abs(k["cx"] - 320) < 3 and abs(k["cy"] - 240) < 3
    assert len(res.per_view_rms_px) == 25 and max(res.per_view_rms_px) < 1.0


def test_too_few_views_and_unusable_images_are_reported(views):
    blank = np.full((480, 640, 3), 128, np.uint8)
    with pytest.raises(CalibrationError, match="at least 8"):
        calibrate_intrinsics([*views[:3], blank], SPEC)
    res = calibrate_intrinsics([blank, *views[:10]], SPEC, estimate_distortion=False)
    assert res.views_rejected == [0] and res.views_used == list(range(1, 11))


def test_mixed_resolutions_are_refused(views):
    with pytest.raises(CalibrationError, match="one resolution"):
        calibrate_intrinsics([views[0], views[1][:240, :320]], SPEC)
    with pytest.raises(CalibrationError):
        calibrate_intrinsics([], SPEC)


def test_document_is_schema_valid(views):
    res = calibrate_intrinsics(views, SPEC)
    doc = camera_calibration_document(res, "labA_4k30_v1", video_mode={"width": 640, "height": 480,
                                                                       "frame_rate": 30, "lens": "1x"})
    assert schema_errors(doc, "camera_calibration") == []
    assert doc["schema_version"] == DATASET_SCHEMA_VERSION and doc["image_source"] == "rgb"
    assert doc["distortion"]["model"] == "opencv_brown_conrady" and len(doc["distortion"]["coefficients"]) == 5
    assert doc["quality"]["views_used"] == 25
    with pytest.raises(CalibrationError):
        camera_calibration_document(res, "")  # empty id violates the schema


# ---- attaching a calibration to an RGB session ----------------------------------------------------

def _rgb_session(path, calibration_id="labA_4k30_v1", shape=(480, 640)):
    rgb = np.zeros((*shape, 3), np.uint8)
    meta = {"session_id": "ses_test", "created_utc": "2026-11-23T00:00:00Z", "synthetic": True,
            "input_type": "synthetic_rgb", "calibration_id": calibration_id,
            "capture": {"camera_fixed": True, "subject_on_turntable": False, "rotation_source": "none",
                        "rotation_axis_frame": "none"}}
    return write_session(path, session_meta=meta, camera_calibration=None, turntable_calibration=None,
                         frames=[FrameData(None, None, 0.0, None, "none", rgb=rgb)])


def test_attach_calibration_to_session(tmp_path, views):
    ses = _rgb_session(tmp_path / "ses")
    assert "calibration_missing" in validate_session(ses).codes()
    doc = camera_calibration_document(calibrate_intrinsics(views, SPEC), "labA_4k30_v1")
    p = attach_camera_calibration(ses, doc)
    assert json.loads(p.read_text()) == doc and not os.access(p, os.W_OK)
    rep = validate_session(ses)
    assert rep.ok and "resolution_mismatch" not in rep.codes() and "calibration_id_mismatch" not in rep.codes()
    with pytest.raises(FileExistsError):  # calibrations are never overwritten
        attach_camera_calibration(ses, doc)


def test_attach_refuses_mismatched_id_or_resolution(tmp_path, views):
    doc = camera_calibration_document(calibrate_intrinsics(views, SPEC), "labA_4k30_v1")
    with pytest.raises(CalibrationError, match="calibration_id"):
        attach_camera_calibration(_rgb_session(tmp_path / "a", calibration_id="other_cal"), doc)
    with pytest.raises(CalibrationError, match="is 320x240"):
        attach_camera_calibration(_rgb_session(tmp_path / "b", shape=(240, 320)), doc)
    assert not (tmp_path / "a" / "calibration" / "camera_calibration.json").exists()
