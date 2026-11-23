import copy
import json

import pytest
from jsonschema import Draft202012Validator

from digitaltrack.dataset.schemas import SCHEMA_DIR, SCHEMA_NAMES, load_schema, schema_errors


def test_schema_dir_exists():
    assert SCHEMA_DIR.is_dir(), SCHEMA_DIR


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_is_a_valid_draft_2020_12_schema(name):
    Draft202012Validator.check_schema(load_schema(name))


def test_unknown_schema_name_rejected():
    with pytest.raises(KeyError):
        load_schema("nope")


def test_session_schema_accepts_rendered_session_and_rejects_damage(small_cylinder_session):
    doc = json.loads((small_cylinder_session / "session.json").read_text())
    assert schema_errors(doc, "session") == []

    for mutate, expect in [
        (lambda d: d.pop("calibration_id"), "calibration_id"),
        (lambda d: d["capture"].__setitem__("depth_dtype", "uint16"), "depth_dtype"),
        (lambda d: d["capture"].__setitem__("depth_unit", "millimeters"), "depth_unit"),
        (lambda d: d["frames"][0].__setitem__("depth_sha256", "xyz"), "depth_sha256"),
        (lambda d: d.__setitem__("frames", []), "frames"),
        (lambda d: d.pop("synthetic"), "synthetic"),
    ]:
        bad = copy.deepcopy(doc)
        mutate(bad)
        errs = schema_errors(bad, "session")
        assert errs and any(expect in e for e in errs), (expect, errs)


def test_frame_metadata_requires_float32_little_endian(small_cylinder_session):
    doc = json.loads((small_cylinder_session / "frames/frame_000001/frame_metadata.json").read_text())
    assert schema_errors(doc, "frame_metadata") == []
    doc["depth"]["dtype"] = "float16"
    assert schema_errors(doc, "frame_metadata")


def test_calibration_schemas(small_cylinder_session):
    cam = json.loads((small_cylinder_session / "calibration/camera_calibration.json").read_text())
    tt = json.loads((small_cylinder_session / "calibration/turntable_calibration.json").read_text())
    assert schema_errors(cam, "camera_calibration") == []
    assert schema_errors(tt, "turntable_calibration") == []
    cam["intrinsics"]["fx"] = 0
    assert schema_errors(cam, "camera_calibration")
    tt["angle_direction_sign"] = 2
    assert schema_errors(tt, "turntable_calibration")
    tt["angle_direction_sign"] = 1
    tt["camera_from_turntable"] = tt["camera_from_turntable"][:3]
    assert schema_errors(tt, "turntable_calibration")


def test_measurements_and_quality_schemas_reject_bad_enums():
    meas = {
        "schema_version": "0.1.0", "session_id": "s", "run_id": "r", "definition_version": "d",
        "algorithm_version": "a",
        "measurements": [{"name": "x", "value": 1.0, "unit": "m", "source": "measured",
                          "quality_status": "pass", "definition_version": "d", "algorithm_version": "a"}],
    }
    assert schema_errors(meas, "measurements") == []
    meas["measurements"][0]["source"] = "guessed"
    assert schema_errors(meas, "measurements")
    meas["measurements"][0]["source"] = "model_estimated"
    meas["measurements"][0]["unit"] = "cm"
    assert schema_errors(meas, "measurements")

    qr = {"schema_version": "0.1.0", "session_id": "s", "run_id": "r", "algorithm_version": "a",
          "generated_utc": "t", "passed": True,
          "checks": [{"name": "c", "status": "pass", "value": 1, "threshold": 1, "message": "m"}], "metrics": {}}
    assert schema_errors(qr, "quality_report") == []
    qr["checks"][0]["status"] = "ok"
    assert schema_errors(qr, "quality_report")
