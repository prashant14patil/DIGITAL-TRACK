"""Validate a session directory against the schemas and the on-disk contract.

Never modifies data. Errors are blocking; warnings are flagged but allowed. Each session is validated against
the schema version it declares (current, or a frozen legacy version in schemas/v<version>/).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import SUPPORTED_DATASET_SCHEMA_VERSIONS
from .schemas import schema_errors
from .writer import sha256_file

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RGB_ONLY_INPUTS = {"video_rgb", "synthetic_rgb"}
DEPTH_INPUTS = {"rgbd_depth", "synthetic_depth"}


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    path: str = ""

    def __str__(self) -> str:
        loc = f" [{self.path}]" if self.path else ""
        return f"{self.level.upper()} {self.code}{loc}: {self.message}"


@dataclass
class ValidationReport:
    root: str
    issues: list[Issue] = field(default_factory=list)
    schema_version: str | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def add(self, level: str, code: str, message: str, path: str = "") -> None:
        self.issues.append(Issue(level, code, message, path))


def _load_json(path: Path, report: ValidationReport, rel: str, required: bool = True):
    if not path.is_file():
        if required:
            report.add("error", "missing_file", "required file not found", rel)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        report.add("error", "invalid_json", str(e), rel)
        return None


def _schema(report: ValidationReport, obj, schema: str, rel: str, version: str) -> bool:
    if isinstance(obj, dict) and obj.get("schema_version") != version:
        report.add("error", "mixed_schema_versions",
                   f"schema_version {obj.get('schema_version')!r} != session {version!r}", rel)
        return False
    errs = schema_errors(obj, schema, version)
    for e in errs:
        report.add("error", "schema_violation", e, rel)
    return not errs


def png_header(path: Path) -> tuple[int, int, int, int] | None:
    """(width, height, bit_depth, colour_type) from the PNG IHDR chunk, or None if not a PNG."""
    with open(path, "rb") as f:
        head = f.read(33)
    if len(head) < 33 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        return None
    w, h, bit_depth, colour_type = struct.unpack(">IIBB", head[16:26])
    return w, h, bit_depth, colour_type


def _check_hash(report, path: Path, stored: str, listed: str | None, label: str, tag: str, rel: str):
    if listed is not None and stored != listed:
        report.add("error", "checksum_inconsistent", f"{label} sha256 differs between frame metadata and session.json", tag)
    if sha256_file(path) != stored:
        report.add("error", "checksum_mismatch",
                   f"{label} file content does not match recorded sha256 (raw data modified?)", rel)


def validate_session(
    root: str | Path, *, check_checksums: bool = True, check_depth_values: bool = True
) -> ValidationReport:
    root = Path(root)
    report = ValidationReport(root=str(root))
    if not root.is_dir():
        report.add("error", "missing_directory", "session directory not found", str(root))
        return report

    session = _load_json(root / "session.json", report, "session.json")
    if session is None:
        return report
    version = session.get("schema_version") if isinstance(session, dict) else None
    if version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
        report.add("error", "unsupported_schema_version",
                   f"schema_version {version!r} not in supported {SUPPORTED_DATASET_SCHEMA_VERSIONS}", "session.json")
        return report
    report.schema_version = version
    legacy = version == "0.1.0"
    input_type = "synthetic_depth" if legacy and session.get("synthetic") else \
        "rgbd_depth" if legacy else session.get("input_type")
    calibration_optional = input_type in RGB_ONLY_INPUTS

    cam = _load_json(root / "calibration" / "camera_calibration.json", report,
                     "calibration/camera_calibration.json", required=not calibration_optional)
    tt = _load_json(root / "calibration" / "turntable_calibration.json", report,
                    "calibration/turntable_calibration.json", required=not calibration_optional)
    session_ok = _schema(report, session, "session", "session.json", version)
    cam_ok = cam is not None and _schema(report, cam, "camera_calibration",
                                         "calibration/camera_calibration.json", version)
    tt_ok = tt is not None and _schema(report, tt, "turntable_calibration",
                                       "calibration/turntable_calibration.json", version)
    if calibration_optional and (cam is None or tt is None):
        missing = [n for n, o in (("camera", cam), ("turntable", tt)) if o is None]
        report.add("warning", "calibration_missing",
                   f"{' and '.join(missing)} calibration not present yet; session cannot be reconstructed",
                   "calibration/")
    cam_image_source = ("depth" if legacy else cam.get("image_source")) if cam_ok else None

    if session_ok and cam_ok and cam["calibration_id"] != session["calibration_id"]:
        report.add("error", "calibration_id_mismatch",
                   f"camera calibration {cam['calibration_id']!r} != session {session['calibration_id']!r}")
    if session_ok and tt_ok and tt["calibration_id"] != session["calibration_id"]:
        report.add("error", "calibration_id_mismatch",
                   f"turntable calibration {tt['calibration_id']!r} != session {session['calibration_id']!r}")
    if tt_ok:
        m = np.asarray(tt["camera_from_turntable"], dtype=float)
        r = m[:3, :3]
        if not (np.allclose(r @ r.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(r), 1.0, atol=1e-6)):
            report.add("error", "bad_rotation", "camera_from_turntable rotation is not a proper rotation",
                       "calibration/turntable_calibration.json")
        if not np.allclose(m[3], [0, 0, 0, 1]):
            report.add("error", "bad_transform", "camera_from_turntable last row must be [0,0,0,1]",
                       "calibration/turntable_calibration.json")

    if not session_ok:
        return report

    sv = session.get("source_video")
    if sv:
        p = root / sv["file"]
        if not p.is_file():
            report.add("error", "missing_file", "source video not found", sv["file"])
        elif p.stat().st_size != sv["size_bytes"]:
            report.add("error", "bad_file_size", f"source video is {p.stat().st_size} bytes, expected {sv['size_bytes']}",
                       sv["file"])
        elif check_checksums and sha256_file(p) != sv["sha256"]:
            report.add("error", "checksum_mismatch", "source video does not match recorded sha256 (modified?)", sv["file"])

    on_disk = {p.name for p in (root / "frames").glob("frame_*") if p.is_dir()} if (root / "frames").is_dir() else set()
    listed = {Path(f["dir"]).name for f in session["frames"]}
    for missing in sorted(listed - on_disk):
        report.add("error", "missing_frame_dir", "frame listed in session.json is absent", f"frames/{missing}")
    for extra in sorted(on_disk - listed):
        report.add("warning", "unlisted_frame_dir", "frame directory not listed in session.json", f"frames/{extra}")

    ids = [f["frame_id"] for f in session["frames"]]
    if len(set(ids)) != len(ids):
        report.add("error", "duplicate_frame_id", "frame_id values are not unique", "session.json")
    times = [f["timestamp_s"] for f in session["frames"]]
    if any(b < a for a, b in zip(times, times[1:])):
        report.add("warning", "non_monotonic_timestamps", "frame timestamps are not non-decreasing", "session.json")
    n_no_angle = sum("turntable_angle_deg" not in f for f in session["frames"])
    if n_no_angle:
        report.add("warning", "angles_missing",
                   f"{n_no_angle} frame(s) have no turntable angle yet; session cannot be reconstructed", "session.json")

    for entry in session["frames"]:
        rel_dir = entry["dir"]
        fdir = root / rel_dir
        if not fdir.is_dir():
            continue
        fm = _load_json(fdir / "frame_metadata.json", report, f"{rel_dir}/frame_metadata.json")
        if fm is None or not _schema(report, fm, "frame_metadata", f"{rel_dir}/frame_metadata.json", version):
            continue
        tag = rel_dir
        if fm["frame_id"] != entry["frame_id"]:
            report.add("error", "frame_id_mismatch", "frame_metadata frame_id != session.json", tag)
        if fm["session_id"] != session["session_id"]:
            report.add("error", "session_id_mismatch", "frame_metadata session_id != session.json", tag)
        if fm.get("turntable_angle_deg") != entry.get("turntable_angle_deg"):
            report.add("error", "angle_mismatch", "recorded angle differs between session.json and frame metadata", tag)
        if fm["angle_source"] == "none" and "turntable_angle_deg" in fm:
            report.add("error", "angle_source_inconsistent", "angle present but angle_source is 'none'", tag)
        if input_type in DEPTH_INPUTS and "depth" not in fm:
            report.add("error", "missing_depth", f"{input_type} session frame has no depth", tag)
        if input_type in RGB_ONLY_INPUTS and "rgb" not in fm:
            report.add("error", "missing_rgb", f"{input_type} session frame has no rgb image", tag)
        if input_type == "video_rgb" and "video" not in fm:
            report.add("error", "missing_video_reference", "video frame does not record its source frame index", tag)

        if "rgb" in fm:
            _check_rgb(report, fm["rgb"], entry, fdir, rel_dir, tag, cam_image_source, cam, check_checksums)
        if "depth" in fm:
            _check_depth(report, fm, entry, fdir, rel_dir, tag, cam_image_source, cam, check_checksums,
                         check_depth_values)
    return report


def _check_rgb(report, r, entry, fdir, rel_dir, tag, cam_image_source, cam, check_checksums):
    p = fdir / r["file"]
    rel = f"{rel_dir}/{r['file']}"
    if not p.is_file():
        report.add("error", "missing_file", "rgb file not found", rel)
        return
    hdr = png_header(p)
    if hdr is None:
        report.add("error", "bad_image_format", "rgb file is not a PNG", rel)
        return
    w, h, bit_depth, colour_type = hdr
    if (w, h) != (r["width"], r["height"]) or bit_depth != r["bit_depth"] or colour_type != 2:
        report.add("error", "image_header_mismatch",
                   f"PNG is {w}x{h} {bit_depth}-bit colour type {colour_type}; metadata says "
                   f"{r['width']}x{r['height']} {r['bit_depth']}-bit RGB", rel)
    if cam_image_source == "rgb" and (r["width"], r["height"]) != (cam["image_width"], cam["image_height"]):
        report.add("error", "resolution_mismatch",
                   f"rgb {r['width']}x{r['height']} != camera calibration "
                   f"{cam['image_width']}x{cam['image_height']} (intrinsics would be wrong)", tag)
    if check_checksums:
        _check_hash(report, p, r["sha256"], entry.get("rgb_sha256"), "rgb", tag, rel)


def _check_depth(report, fm, entry, fdir, rel_dir, tag, cam_image_source, cam, check_checksums, check_depth_values):
    d, c = fm["depth"], fm.get("confidence")
    if c is None:
        report.add("error", "missing_confidence", "depth frame has no confidence map", tag)
        return
    if cam_image_source == "depth" and (d["width"], d["height"]) != (cam["image_width"], cam["image_height"]):
        report.add("error", "resolution_mismatch",
                   f"depth {d['width']}x{d['height']} != camera calibration "
                   f"{cam['image_width']}x{cam['image_height']} (intrinsics would be wrong)", tag)
    if (d["width"], d["height"]) != (c["width"], c["height"]):
        report.add("error", "confidence_size_mismatch", "confidence map size differs from depth map", tag)

    dpath, cpath = fdir / d["file"], fdir / c["file"]
    depth_ok = False
    for p, expected_bytes, label in ((dpath, d["width"] * d["height"] * 4, "depth"),
                                     (cpath, c["width"] * c["height"], "confidence")):
        if not p.is_file():
            report.add("error", "missing_file", f"{label} file not found", f"{rel_dir}/{p.name}")
        elif p.stat().st_size != expected_bytes:
            report.add("error", "bad_file_size",
                       f"{label} file is {p.stat().st_size} bytes, expected {expected_bytes}", f"{rel_dir}/{p.name}")
        elif label == "depth":
            depth_ok = True

    if check_checksums:
        for p, stored, listed_sha, label in ((dpath, d["sha256"], entry.get("depth_sha256"), "depth"),
                                             (cpath, c["sha256"], entry.get("confidence_sha256"), "confidence")):
            if p.is_file():
                _check_hash(report, p, stored, listed_sha, label, tag, f"{rel_dir}/{p.name}")

    if check_depth_values and depth_ok:
        arr = np.fromfile(dpath, dtype="<f4")
        if not np.all(np.isfinite(arr)):
            report.add("warning", "non_finite_depth", f"{int((~np.isfinite(arr)).sum())} non-finite depth values", tag)
        if np.any(arr[np.isfinite(arr)] < 0):
            report.add("error", "negative_depth", "negative depth values present", tag)
        if not np.any(arr > 0):
            report.add("error", "empty_depth", "frame contains no valid (positive) depth", tag)
