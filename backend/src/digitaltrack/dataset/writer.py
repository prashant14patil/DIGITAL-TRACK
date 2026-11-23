"""Write a scan session in the on-disk contract (docs/dataset_spec.md), dataset schema 0.2.0.

Raw files are written once, never overwritten, and marked read-only:
* source video: copied byte-for-byte into source/ (video_rgb sessions),
* RGB frames: lossless PNG (8- or 16-bit, RGB channel order inside the PNG),
* depth (optional branch): little-endian float32 metres; confidence uint8.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .. import DATASET_SCHEMA_VERSION
from .schemas import schema_errors

PNG_COMPRESSION = 3  # lossless at any level; fixed so identical pixels give identical bytes/checksums


@dataclass
class FrameData:
    depth: np.ndarray | None  # (H, W) float32 metres, 0 = no measurement; None for RGB-only frames
    confidence: np.ndarray | None  # (H, W) uint8 (0 low, 1 medium, 2 high); required iff depth is given
    timestamp_s: float
    turntable_angle_deg: float | None  # RECORDED angle; None = not yet determined
    angle_source: str  # manual | marker | encoder | synthetic | none
    camera_transform: list[list[float]] | None = None  # world_from_camera from a depth device, if any
    rgb: np.ndarray | None = None  # (H, W, 3) uint8/uint16 in RGB channel order
    video_frame_index: int | None = None  # 0-based index of this frame in the source video
    video_pts_s: float | None = None  # presentation time in the source video


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _check(obj: Any, schema: str) -> None:
    errs = schema_errors(obj, schema)
    if errs:
        raise ValueError(f"{schema} violates schema: {errs}")


def make_readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def encode_png(rgb: np.ndarray) -> bytes:
    """Lossless PNG bytes for an (H, W, 3) RGB uint8/uint16 image. Uses imencode (safe for non-ASCII paths)."""
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"rgb must be (H, W, 3) uint8/uint16, got {rgb.shape} {rgb.dtype}")
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]),
                           [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def write_session(
    session_dir: str | Path,
    *,
    session_meta: dict[str, Any],
    camera_calibration: dict[str, Any] | None,
    turntable_calibration: dict[str, Any] | None,
    frames: Iterable[FrameData],
    extra_json: dict[str, Any] | None = None,
    source_video: str | Path | None = None,
    readonly: bool = True,
) -> Path:
    """Create a new session directory. Refuses to touch an existing non-empty directory.

    `session_meta` holds everything in session.json except `frames` and `schema_version` (and, when
    `source_video` is given, except the file/sha256/size fields of `source_video`, which are computed here;
    any other `source_video` keys such as `probe` are kept). Calibrations may be None for RGB input whose
    calibration is not known yet; such a session validates with warnings and cannot be reconstructed.
    `extra_json` maps relative file name -> JSON object (e.g. synthetic ground_truth.json).
    """
    root = Path(session_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty directory (raw data is immutable): {root}")
    (root / "calibration").mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir(exist_ok=True)

    if camera_calibration is not None:
        _check(camera_calibration, "camera_calibration")
    if turntable_calibration is not None:
        _check(turntable_calibration, "turntable_calibration")
    session_id = session_meta["session_id"]
    meta_out = dict(session_meta)
    written: list[Path] = []

    if source_video is not None:
        src = Path(source_video)
        (root / "source").mkdir()
        dst = root / "source" / f"source_video{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        written.append(dst)
        sv = dict(meta_out.get("source_video", {}))
        sv.update({"file": f"source/{dst.name}", "original_filename": src.name,
                   "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size})
        if sv["sha256"] != sha256_file(src):
            raise IOError("source video copy differs from the original")
        meta_out["source_video"] = sv

    cal_src = camera_calibration["image_source"] if camera_calibration else None
    cal_size = (camera_calibration["image_height"], camera_calibration["image_width"]) if camera_calibration else None

    frame_entries: list[dict[str, Any]] = []
    for i, fr in enumerate(frames, start=1):
        if fr.depth is None and fr.rgb is None:
            raise ValueError(f"frame {i}: needs rgb and/or depth")
        if (fr.depth is None) != (fr.confidence is None):
            raise ValueError(f"frame {i}: depth and confidence must be given together")
        fdir = root / "frames" / f"frame_{i:06d}"
        fdir.mkdir()
        meta: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "session_id": session_id,
            "frame_id": i,
            "timestamp_s": float(fr.timestamp_s),
            "angle_source": fr.angle_source,
        }
        entry: dict[str, Any] = {"frame_id": i, "dir": f"frames/frame_{i:06d}", "timestamp_s": float(fr.timestamp_s)}
        if fr.turntable_angle_deg is not None:
            meta["turntable_angle_deg"] = entry["turntable_angle_deg"] = float(fr.turntable_angle_deg)

        if fr.rgb is not None:
            h, w = fr.rgb.shape[:2]
            if cal_src == "rgb" and (h, w) != cal_size:
                raise ValueError(f"frame {i}: rgb shape must be {cal_size}")
            data = encode_png(fr.rgb)
            (fdir / "rgb.png").write_bytes(data)
            sha = sha256_bytes(data)
            meta["rgb"] = {"file": "rgb.png", "format": "png", "width": int(w), "height": int(h), "channels": 3,
                           "bit_depth": 8 if fr.rgb.dtype == np.uint8 else 16, "sha256": sha}
            entry["rgb_sha256"] = sha
            written.append(fdir / "rgb.png")

        if fr.depth is not None:
            depth = np.ascontiguousarray(fr.depth, dtype="<f4")
            conf = np.ascontiguousarray(fr.confidence, dtype=np.uint8)
            if depth.shape != conf.shape:
                raise ValueError(f"frame {i}: depth/confidence shapes differ")
            if cal_src == "depth" and depth.shape != cal_size:
                raise ValueError(f"frame {i}: depth/confidence shape must be {cal_size}")
            h, w = depth.shape
            depth_bytes, conf_bytes = depth.tobytes(), conf.tobytes()
            (fdir / "depth_raw.bin").write_bytes(depth_bytes)
            (fdir / "confidence.bin").write_bytes(conf_bytes)
            d_sha, c_sha = sha256_bytes(depth_bytes), sha256_bytes(conf_bytes)
            meta["depth"] = {"file": "depth_raw.bin", "dtype": "float32", "byte_order": "little",
                             "width": w, "height": h, "unit": "meters", "sha256": d_sha}
            meta["confidence"] = {"file": "confidence.bin", "dtype": "uint8", "width": w, "height": h, "sha256": c_sha}
            entry["depth_sha256"], entry["confidence_sha256"] = d_sha, c_sha
            written += [fdir / "depth_raw.bin", fdir / "confidence.bin"]

        if fr.video_frame_index is not None:
            meta["video"] = {"source_frame_index": int(fr.video_frame_index),
                             "pts_s": float(fr.video_pts_s if fr.video_pts_s is not None else fr.timestamp_s)}
        if fr.camera_transform is not None:
            meta["camera_transform"] = {"convention": "world_from_camera", "matrix": fr.camera_transform}
        if camera_calibration is not None:
            meta["intrinsics_reference"] = "calibration/camera_calibration.json"
        _check(meta, "frame_metadata")
        _write_json(fdir / "frame_metadata.json", meta)
        written.append(fdir / "frame_metadata.json")
        frame_entries.append(entry)

    session = {"schema_version": DATASET_SCHEMA_VERSION, **meta_out, "frames": frame_entries}
    _check(session, "session")
    _write_json(root / "session.json", session)
    written.append(root / "session.json")
    for obj, rel in ((camera_calibration, "calibration/camera_calibration.json"),
                     (turntable_calibration, "calibration/turntable_calibration.json")):
        if obj is not None:
            _write_json(root / rel, obj)
            written.append(root / rel)
    for name, obj in (extra_json or {}).items():
        _write_json(root / name, obj)
        written.append(root / name)

    if readonly:
        for p in written:
            make_readonly(p)
    return root
