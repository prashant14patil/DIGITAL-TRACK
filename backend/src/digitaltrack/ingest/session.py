"""Camera-app video (RGB-only fallback) -> dataset-schema-0.2.0 `video_rgb` session.

Guarantees:
* The source video is checked (type, size, SHA-256) before use and copied byte-for-byte; the copy is re-hashed.
* Every decoded frame is accounted for: the decoder's frame count must equal ffprobe's timestamp count.
* The session is written into a temporary sibling folder and renamed into place only after the whole
  session validates. A failed ingest never leaves a partial session behind.
* Angles are NOT determined here (`angle_source: "none"`): pose tracking for manual self-rotation is an open
  item, so the resulting session validates with `angles_missing` / `calibration_missing` warnings and cannot
  be reconstructed yet. That is intended.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .. import __version__
from ..dataset.validate import validate_session
from ..dataset.writer import FrameData, utc_now, write_session
from .errors import IngestError
from .metadata import CaptureMetadata
from .video import (check_source_file, decode_frames, extraction_pix_fmt, find_tool, frame_timestamps, probe_video,
                    tool_version)

DEFAULT_MAX_FRAMES = 2000  # guard against filling the disk with 4K PNGs; raise explicitly when exceeded


@dataclass
class IngestResult:
    session_dir: Path
    session_id: str
    n_frames: int
    n_decoded: int
    warnings: list[str] = field(default_factory=list)
    validation_codes: list[str] = field(default_factory=list)


def new_session_id() -> str:
    return f"ses_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"


def _rmtree_readonly(path: Path) -> None:
    def onerror(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


def ingest_video(source: str | Path, session_dir: str | Path, metadata: CaptureMetadata, *,
                 session_id: str | None = None, frame_step: int = 1, max_frames: int = DEFAULT_MAX_FRAMES,
                 expected_sha256: str | None = None, expected_size: int | None = None,
                 synthetic: bool = False) -> IngestResult:
    """Create a `video_rgb` session from a Camera-app video. See module docstring for guarantees."""
    if metadata.capture_type != "rgb_video":
        raise IngestError("wrong_capture_type",
                          f"ingest_video handles capture_type 'rgb_video' only (got {metadata.capture_type!r}); "
                          "no RGB-D export adapter exists until a real export has been inspected")
    if not isinstance(frame_step, int) or frame_step < 1:
        raise IngestError("invalid_parameter", f"frame_step must be a positive integer (got {frame_step!r})")
    dest = Path(session_dir)
    if dest.exists():
        raise IngestError("destination_exists", "session directory already exists (raw data is never overwritten)",
                          path=str(dest))

    src = check_source_file(source, expected_sha256, expected_size)
    ffprobe, ffmpeg = find_tool("ffprobe"), find_tool("ffmpeg")
    probe = probe_video(src.path, ffprobe=ffprobe)
    stamps = frame_timestamps(src.path, probe["video_stream_index"], ffprobe=ffprobe)
    selected = list(range(0, len(stamps), frame_step))
    if len(selected) > max_frames:
        raise IngestError("too_many_frames",
                          f"{len(selected)} frames would be extracted (limit {max_frames}); use a larger frame_step "
                          f"or raise max_frames deliberately", path=str(src.path))
    pix_fmt, _, png_bits = extraction_pix_fmt(probe)
    session_id = session_id or new_session_id()

    counter = {"decoded": 0}

    def frames() -> Iterator[FrameData]:
        sel = set(selected)
        for i, rgb in enumerate(decode_frames(src.path, probe, ffmpeg=ffmpeg)):
            counter["decoded"] = i + 1
            if i >= len(stamps):
                raise IngestError("frame_count_mismatch", f"decoder produced more frames than ffprobe listed "
                                  f"({len(stamps)})", path=str(src.path))
            if i in sel:
                yield FrameData(depth=None, confidence=None, timestamp_s=stamps[i], turntable_angle_deg=None,
                                angle_source="none", rgb=rgb, video_frame_index=i, video_pts_s=stamps[i])
        if counter["decoded"] != len(stamps):
            raise IngestError("frame_count_mismatch", f"decoder produced {counter['decoded']} frames, ffprobe listed "
                              f"{len(stamps)}", path=str(src.path))

    session_meta: dict[str, Any] = {
        "session_id": session_id,
        "created_utc": utc_now(),
        "synthetic": synthetic,
        "input_type": "video_rgb",
        **metadata.session_fields(),
        "source_video": {"probe": probe},
        "frame_extraction": {
            "method": "ffmpeg_decode_every_frame_then_select_by_step",
            "parameters": {"frame_step": frame_step, "pix_fmt": pix_fmt, "png_bit_depth": png_bits,
                           "noautorotate": True, "fps_mode": "passthrough",
                           "video_stream_index": probe["video_stream_index"],
                           "frames_in_stream": len(stamps), "frames_selected": len(selected),
                           "selection": "every frame_step-th decoded frame, starting at frame 0; "
                                        "no still-frame/pause detection yet"},
            "software": {"digitaltrack": __version__, "ffmpeg": tool_version(ffmpeg), "ffprobe": tool_version(ffprobe)},
        },
    }

    partial = dest.with_name(f"{dest.name}.partial-{uuid.uuid4().hex[:8]}")
    try:
        write_session(partial, session_meta=session_meta, camera_calibration=None, turntable_calibration=None,
                      frames=frames(), source_video=src.path)
        if json.loads((partial / "session.json").read_text(encoding="utf-8"))["source_video"]["sha256"] != src.sha256:
            raise IngestError("checksum_mismatch", "stored source video differs from the checked original",
                              path=str(src.path))
        report = validate_session(partial)
        if not report.ok:
            raise IngestError("validation_failed", "ingested session failed validation",
                              problems=[str(i) for i in report.errors])
        os.replace(partial, dest)
    except BaseException:
        if partial.exists():
            _rmtree_readonly(partial)
        raise
    return IngestResult(session_dir=dest, session_id=session_id, n_frames=len(selected), n_decoded=counter["decoded"],
                        warnings=list(probe["flags"]), validation_codes=sorted(report.codes()))


def ingest_video_from_upload(upload_dir: str | Path, export_relpath: str, session_dir: str | Path,
                             metadata: CaptureMetadata, **kwargs: Any) -> IngestResult:
    """Ingest one video file from a FINALIZED raw upload (api/storage.py layout).

    The file's size and SHA-256 must match the upload's manifest.json, so what is ingested is provably what was
    uploaded. Only the named file is read; nothing else in the export is interpreted.
    """
    up = Path(upload_dir)
    manifest_p = up / "manifest.json"
    if not manifest_p.is_file():
        raise IngestError("upload_not_finalized", "upload has no manifest.json (not finalized)", path=str(up))
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    rel = export_relpath.replace("\\", "/")
    entry = next((f for f in manifest.get("files", []) if f.get("path") == rel), None)
    if entry is None:
        raise IngestError("upload_file_not_listed", f"{rel!r} is not in the upload manifest", path=str(manifest_p))
    src = up / "export" / Path(*rel.split("/"))
    return ingest_video(src, session_dir, metadata, expected_sha256=entry["sha256"], expected_size=int(entry["size"]),
                        **kwargs)
