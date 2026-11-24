"""Capture-session metadata: what the operator enters about a capture, validated before anything is written.

Only fields that already exist in dataset schema 0.2.0 are emitted (session.subject, session.rig,
session.device, session.capture). Manual self-rotation (no turntable) uses the existing, schema-valid values
`capture.subject_on_turntable = false`, `capture.rotation_source = "none"` and a free-text
`capture.rotation_axis_frame` saying that no rotation axis is defined. No new schema fields are introduced.

Pseudonymity: subject IDs must match a restrictive pattern (letters, digits, `_`, `-`; no spaces, no `@`).
This blocks e-mail addresses and "First Last" names but cannot prove an ID is not identifying; choosing
pseudonyms remains the operator's responsibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import IngestError

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
CAPTURE_TYPES = ("rgbd", "rgb_video")  # rgbd = primary (LiDAR app export; adapter not built), rgb_video = fallback
MAX_TEXT = 500
NO_AXIS = "none: manual self-rotation; no fixed rotation axis defined (pose tracking and pose model are open items)"


@dataclass(frozen=True)
class CaptureMetadata:
    subject_id: str  # pseudonymous only
    setup_id: str  # physical setup / mount; also used as `rig.rig_id`
    protocol_version: str  # e.g. "capture-0.3-draft"
    capture_type: str  # "rgbd" | "rgb_video"
    calibration_id: str = "pending"  # id of the camera calibration this session will be paired with
    clothing: str = ""
    posture_notes: str = ""
    camera_distance_m: float | None = None
    camera_height_m: float | None = None
    device_model: str | None = None  # as stated by the operator; NOT verified
    recording_app: str | None = None  # e.g. "Camera" (built-in) for the fallback path

    # ---- schema-0.2.0 fragments -----------------------------------------------------------------

    def session_fields(self) -> dict[str, Any]:
        """Fragments of session.json (all existing v0.2.0 fields) describing this capture."""
        subject: dict[str, Any] = {"subject_id": self.subject_id}
        if self.clothing:
            subject["clothing"] = self.clothing
        if self.posture_notes:
            subject["posture_notes"] = self.posture_notes
        rig: dict[str, Any] = {"rig_id": self.setup_id}
        if self.camera_distance_m is not None:
            rig["camera_distance_m"] = self.camera_distance_m
        if self.camera_height_m is not None:
            rig["camera_height_m"] = self.camera_height_m
        device: dict[str, Any] = {}
        if self.device_model:
            device["model"] = self.device_model
        if self.recording_app:
            device["recording_app"] = self.recording_app
        out: dict[str, Any] = {
            "calibration_id": self.calibration_id,
            "subject": subject,
            "rig": rig,
            "capture": {
                "camera_fixed": True,
                "subject_on_turntable": False,
                "rotation_source": "none",
                "rotation_axis_frame": NO_AXIS,
                "protocol_version": self.protocol_version,
            },
        }
        if device:
            out["device"] = device
        return out


def _text_problems(name: str, value: Any, problems: list[str], required: bool = False) -> str:
    if value is None:
        if required:
            problems.append(f"{name} is required")
        return ""
    if not isinstance(value, str):
        problems.append(f"{name} must be text")
        return ""
    if required and not value.strip():
        problems.append(f"{name} must not be empty")
    if len(value) > MAX_TEXT:
        problems.append(f"{name} is longer than {MAX_TEXT} characters")
    if any(ord(c) < 32 and c not in "\t" for c in value):
        problems.append(f"{name} contains control characters")
    return value.strip()


def _id_problems(name: str, value: Any, problems: list[str]) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        problems.append(f"{name} must be 3-64 characters of letters, digits, '_' or '-', starting with a letter "
                        f"or digit (got {value!r}); use a pseudonym, never a name or e-mail")
        return str(value)
    return value


def _distance(name: str, value: Any, problems: list[str]) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} must be a number in metres")
        return None
    if not (0.0 < v <= 10.0):
        problems.append(f"{name} must be in (0, 10] metres (got {v})")
    return v


def validate_capture_metadata(raw: dict[str, Any]) -> CaptureMetadata:
    """Validate operator-entered metadata. Raises IngestError('invalid_metadata') listing every problem."""
    if not isinstance(raw, dict):
        raise IngestError("invalid_metadata", "metadata must be an object")
    problems: list[str] = []
    known = {"subject_id", "setup_id", "protocol_version", "capture_type", "calibration_id", "clothing",
             "posture_notes", "camera_distance_m", "camera_height_m", "device_model", "recording_app"}
    unknown = sorted(set(raw) - known)
    if unknown:
        problems.append(f"unknown fields (not stored, check spelling): {unknown}")
    subject_id = _id_problems("subject_id", raw.get("subject_id"), problems)
    setup_id = _id_problems("setup_id", raw.get("setup_id"), problems)
    calibration_id = raw.get("calibration_id", "pending")
    if calibration_id != "pending":
        calibration_id = _id_problems("calibration_id", calibration_id, problems)
    protocol_version = _text_problems("protocol_version", raw.get("protocol_version"), problems, required=True)
    capture_type = raw.get("capture_type")
    if capture_type not in CAPTURE_TYPES:
        problems.append(f"capture_type must be one of {CAPTURE_TYPES} (got {capture_type!r})")
    md = CaptureMetadata(
        subject_id=subject_id, setup_id=setup_id, protocol_version=protocol_version,
        capture_type=str(capture_type), calibration_id=str(calibration_id),
        clothing=_text_problems("clothing", raw.get("clothing"), problems),
        posture_notes=_text_problems("posture_notes", raw.get("posture_notes"), problems),
        camera_distance_m=_distance("camera_distance_m", raw.get("camera_distance_m"), problems),
        camera_height_m=_distance("camera_height_m", raw.get("camera_height_m"), problems),
        device_model=_text_problems("device_model", raw.get("device_model"), problems) or None,
        recording_app=_text_problems("recording_app", raw.get("recording_app"), problems) or None,
    )
    if problems:
        raise IngestError("invalid_metadata", f"{len(problems)} metadata problem(s)", problems=problems)
    return md
