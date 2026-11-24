"""Camera-app video: pre-checks, probe (ffprobe) and lossless frame extraction (ffmpeg).

Design decisions (recorded in every session's `frame_extraction`):
* Frames are decoded with `-noautorotate`: images keep the stored (sensor) orientation. Any display-rotation
  metadata is reported in the probe, not applied, so intrinsics calibrated from the same extraction rule stay
  valid. Never rotate an image without rotating its intrinsics.
* `-fps_mode passthrough`: no frames are duplicated or dropped; every decoded frame keeps its own timestamp.
* Pixel format: 8-bit sources -> rgb24 (8-bit PNG); >8-bit sources -> rgb48le (16-bit PNG) so decoding does
  not add a quantisation step. HDR transfer functions are NOT linearised; HDR input is flagged.
* Location metadata (GPS) is never copied: the probe only records that it was present.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..dataset.writer import sha256_file
from .errors import IngestError

VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v")
# Top-level atoms a QuickTime / ISO-BMFF file may start with.
LEADING_ATOMS = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
LOCATION_TAG_HINTS = ("location", "iso6709", "gps")


# ---- external tools ------------------------------------------------------------------------------

def find_tool(name: str) -> str:
    """Path to ffmpeg/ffprobe: env DIGITALTRACK_FFMPEG / DIGITALTRACK_FFPROBE, else PATH."""
    override = os.environ.get(f"DIGITALTRACK_{name.upper()}")
    path = override or shutil.which(name)
    if not path or not Path(path).exists():
        raise IngestError("tool_missing", f"{name} not found on PATH (install FFmpeg, e.g. `winget install "
                          f"Gyan.FFmpeg`, or set DIGITALTRACK_{name.upper()})")
    return path


def tool_version(path: str) -> str:
    out = subprocess.run([path, "-hide_banner", "-version"], capture_output=True, text=True, check=False)
    return (out.stdout.splitlines() or ["unknown"])[0].strip()


# ---- source file checks --------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceFile:
    path: Path
    size_bytes: int
    sha256: str
    leading_atom: str


def check_source_file(path: str | Path, expected_sha256: str | None = None,
                      expected_size: int | None = None) -> SourceFile:
    """Integrity and type checks before anything is copied or decoded."""
    p = Path(path)
    if not p.exists():
        raise IngestError("source_not_found", "source video does not exist", path=str(p))
    if not p.is_file():
        raise IngestError("source_not_a_file", "source is not a regular file", path=str(p))
    size = p.stat().st_size
    if size == 0:
        raise IngestError("source_empty", "source video is empty (0 bytes)", path=str(p))
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        raise IngestError("unsupported_extension",
                          f"expected one of {VIDEO_EXTENSIONS} (Camera-app QuickTime/MP4), got {p.suffix!r}", path=str(p))
    with open(p, "rb") as f:
        head = f.read(12)
    atom = head[4:8] if len(head) >= 8 else b""
    if atom not in LEADING_ATOMS:
        raise IngestError("not_quicktime_mp4",
                          f"file does not start with a QuickTime/ISO-BMFF atom (found {atom!r}); wrong or corrupt file",
                          path=str(p))
    if expected_size is not None and size != expected_size:
        raise IngestError("size_mismatch", f"file is {size} bytes, expected {expected_size}", path=str(p))
    sha = sha256_file(p)
    if expected_sha256 is not None and sha != expected_sha256:
        raise IngestError("checksum_mismatch", "file content does not match the expected sha256", path=str(p))
    return SourceFile(p, size, sha, atom.decode("latin-1"))


# ---- probe ---------------------------------------------------------------------------------------

def _frac(s: Any) -> Fraction | None:
    try:
        f = Fraction(str(s))
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _rotation(stream: dict[str, Any]) -> float:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return float(sd["rotation"])
    rot = (stream.get("tags") or {}).get("rotate")
    return float(rot) if rot is not None else 0.0


def _bit_depth(pix_fmt: str, bits_per_raw_sample: Any) -> int:
    """Bits per colour component, from ffprobe's raw sample size or the pixel-format name."""
    try:
        raw = int(bits_per_raw_sample or 0)
    except ValueError:
        raw = 0
    if raw:
        return raw
    pf = pix_fmt.lower()
    if pf.startswith(("rgb48", "bgr48", "rgba64", "bgra64", "gray16")) or "p16" in pf or "16le" in pf or "16be" in pf:
        return 16
    if "p12" in pf or "12le" in pf or "12be" in pf:
        return 12
    if "p10" in pf or "10le" in pf or "10be" in pf or pf.startswith("x2rgb10"):
        return 10
    return 8


def probe_video(path: str | Path, *, ffprobe: str | None = None) -> dict[str, Any]:
    """Container/stream facts needed for ingestion. Never contains tag values (e.g. GPS)."""
    exe = ffprobe or find_tool("ffprobe")
    out = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise IngestError("probe_failed", f"ffprobe could not read the file: {out.stderr.strip()[:300]}",
                          path=str(path))
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not (s.get("disposition") or {}).get("attached_pic")]
    if not videos:
        raise IngestError("no_video_stream", "file contains no video stream", path=str(path))
    v = videos[0]
    fmt = info.get("format", {})
    tag_keys = [k.lower() for k in (fmt.get("tags") or {})] + [k.lower() for s in streams for k in (s.get("tags") or {})]
    r_rate, avg_rate = _frac(v.get("r_frame_rate")), _frac(v.get("avg_frame_rate"))
    pix_fmt = v.get("pix_fmt", "")
    bits = _bit_depth(pix_fmt, v.get("bits_per_raw_sample"))
    sar = v.get("sample_aspect_ratio", "1:1")
    probe = {
        "container": fmt.get("format_name"),
        "duration_s": float(fmt["duration"]) if fmt.get("duration") else None,
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
        "video_stream_index": int(v["index"]),
        "codec": v.get("codec_name"),
        "profile": v.get("profile"),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "pix_fmt": pix_fmt,
        "bit_depth": bits,
        "sample_aspect_ratio": sar,
        "r_frame_rate": v.get("r_frame_rate"),
        "avg_frame_rate": v.get("avg_frame_rate"),
        "nb_frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
        "color_transfer": v.get("color_transfer"),
        "color_primaries": v.get("color_primaries"),
        "display_rotation_deg": _rotation(v),
        "other_streams": [{"index": int(s["index"]), "codec_type": s.get("codec_type"),
                           "codec_name": s.get("codec_name"), "codec_tag_string": s.get("codec_tag_string")}
                          for s in streams if s is not v],
        "has_location_metadata": any(h in k for k in tag_keys for h in LOCATION_TAG_HINTS),
    }
    flags = []
    if r_rate and avg_rate and r_rate != avg_rate:
        flags.append("variable_frame_rate_suspected")
    if (probe["color_transfer"] or "") in HDR_TRANSFERS:
        flags.append("hdr_transfer_function")
    if bits > 8:
        flags.append("high_bit_depth")
    if probe["display_rotation_deg"]:
        flags.append("display_rotation_not_applied")
    if sar not in ("1:1", "0:1", None):
        flags.append("non_square_pixels")
    if probe["has_location_metadata"]:
        flags.append("location_metadata_present_not_copied")
    probe["flags"] = flags
    return probe


def frame_timestamps(path: str | Path, stream_index: int, *, ffprobe: str | None = None) -> list[float]:
    """Presentation timestamps (s) of every decoded frame of one stream, in output order."""
    exe = ffprobe or find_tool("ffprobe")
    out = subprocess.run([exe, "-v", "error", "-select_streams", str(stream_index), "-show_entries",
                          "frame=best_effort_timestamp_time", "-of", "json", str(path)],
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise IngestError("probe_failed", f"ffprobe could not list frames: {out.stderr.strip()[:300]}", path=str(path))
    frames = json.loads(out.stdout or "{}").get("frames", [])
    ts = []
    for i, fr in enumerate(frames):
        t = fr.get("best_effort_timestamp_time")
        if t is None:
            raise IngestError("missing_timestamp", f"decoded frame {i} has no timestamp", path=str(path))
        ts.append(float(t))
    if not ts:
        raise IngestError("no_frames", "video stream decodes to zero frames", path=str(path))
    return ts


# ---- extraction ----------------------------------------------------------------------------------

def extraction_pix_fmt(probe: dict[str, Any]) -> tuple[str, np.dtype, int]:
    if probe["bit_depth"] > 8:
        return "rgb48le", np.dtype("<u2"), 16
    return "rgb24", np.dtype(np.uint8), 8


def decode_frames(path: str | Path, probe: dict[str, Any], *, ffmpeg: str | None = None) -> Iterator[np.ndarray]:
    """Yield every decoded frame of the probed video stream as (H, W, 3) RGB, in presentation order."""
    exe = ffmpeg or find_tool("ffmpeg")
    pix_fmt, dtype, _ = extraction_pix_fmt(probe)
    w, h = probe["width"], probe["height"]
    frame_bytes = w * h * 3 * dtype.itemsize
    cmd = [exe, "-v", "error", "-nostdin", "-noautorotate", "-i", str(path), "-map", f"0:{probe['video_stream_index']}",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finished = False
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                raise IngestError("extraction_failed", f"truncated frame from decoder ({len(buf)} of {frame_bytes} bytes)",
                                  path=str(path))
            yield np.frombuffer(buf, dtype=dtype).reshape(h, w, 3).astype(dtype.newbyteorder("="), copy=False)
        finished = True
    finally:
        if not finished:  # consumer stopped early or an error occurred: stop the decoder, report nothing here
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
    if rc != 0:
        raise IngestError("extraction_failed", f"ffmpeg exited with {rc}: {err.strip()[:300]}", path=str(path))
