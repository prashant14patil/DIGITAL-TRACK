import pytest

from digitaltrack.dataset.schemas import load_schema
from digitaltrack.ingest.errors import IngestError
from digitaltrack.ingest.metadata import NO_AXIS, validate_capture_metadata

GOOD = {"subject_id": "S001", "setup_id": "lab-A", "protocol_version": "capture-0.3-draft",
        "capture_type": "rgb_video"}


def test_minimal_valid_metadata_and_defaults():
    md = validate_capture_metadata(GOOD)
    assert md.calibration_id == "pending" and md.device_model is None
    f = md.session_fields()
    cap = f["capture"]
    assert cap["subject_on_turntable"] is False  # manual self-rotation: no turntable
    assert cap["rotation_source"] == "none"  # angles/poses not determined (open item)
    assert cap["rotation_axis_frame"] == NO_AXIS and cap["camera_fixed"] is True
    assert f["subject"] == {"subject_id": "S001"} and f["rig"] == {"rig_id": "lab-A"}
    assert "device" not in f  # nothing invented


def test_emitted_fields_exist_in_schema_v020():
    md = validate_capture_metadata({**GOOD, "clothing": "tight shirt", "posture_notes": "A-pose",
                                    "camera_distance_m": 1.5, "camera_height_m": 1.0,
                                    "device_model": "iPad (unverified)", "recording_app": "Camera"})
    props = load_schema("session")["properties"]
    f = md.session_fields()
    for top, sub in f.items():
        assert top in props, top
        if isinstance(sub, dict):
            assert set(sub) <= set(props[top]["properties"]), (top, set(sub) - set(props[top]["properties"]))
    for v in load_schema("session")["properties"]["capture"]["required"]:
        assert v in f["capture"]


@pytest.mark.parametrize("bad_id", ["jane.doe@example.com", "Jane Doe", "ab", "", "-start", "x" * 65, None, 42])
def test_identifying_or_malformed_subject_ids_are_rejected(bad_id):
    with pytest.raises(IngestError) as e:
        validate_capture_metadata({**GOOD, "subject_id": bad_id})
    assert e.value.code == "invalid_metadata" and any("subject_id" in p for p in e.value.problems)


def test_all_problems_are_reported_together():
    with pytest.raises(IngestError) as e:
        validate_capture_metadata({"subject_id": "a b", "setup_id": "ok_setup", "protocol_version": "",
                                   "capture_type": "lidar", "camera_distance_m": -1, "colour": "red",
                                   "clothing": "line1\x00"})
    p = " | ".join(e.value.problems)
    for needle in ("subject_id", "protocol_version", "capture_type", "camera_distance_m", "unknown fields",
                   "control characters"):
        assert needle in p, needle
    assert len(e.value.problems) >= 6
    assert e.value.to_dict()["code"] == "invalid_metadata"


def test_non_numeric_and_out_of_range_distances():
    for v in ("far", 0, 11):
        with pytest.raises(IngestError):
            validate_capture_metadata({**GOOD, "camera_height_m": v})


def test_calibration_id_must_be_an_id_when_given():
    assert validate_capture_metadata({**GOOD, "calibration_id": "labA_4k30_v1"}).calibration_id == "labA_4k30_v1"
    with pytest.raises(IngestError):
        validate_capture_metadata({**GOOD, "calibration_id": "has space"})


def test_rgbd_capture_type_is_accepted_as_metadata_only():
    # the metadata can describe an RGB-D capture; no adapter consumes it yet (see test_ingest_video)
    assert validate_capture_metadata({**GOOD, "capture_type": "rgbd"}).capture_type == "rgbd"


def test_metadata_must_be_an_object():
    with pytest.raises(IngestError):
        validate_capture_metadata(["not", "a", "dict"])
