"""Read a FINALIZED capture upload (api.storage layout) without modifying it, and describe what is inside.

* `open_finalized_upload` verifies manifest.json against the files on disk (size + SHA-256). It only reads.
* `inventory` describes an export generically (folders, extensions, numbering, safe header probes of .npy/.png,
  first lines of text files). It does NOT interpret the format; it is what we look at before writing an adapter.
* `find_adapter` returns a registered export adapter that recognises the upload. NO adapter is registered yet:
  no real export has been inspected, and formats must not be guessed (CLAUDE.md). Real uploads therefore stop
  with `NoAdapterError`, which carries the inventory.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .views import RGBDView


class UploadNotReady(ValueError):
    pass


class UploadIntegrityError(ValueError):
    pass


class NoAdapterError(RuntimeError):
    def __init__(self, message: str, inventory: dict[str, Any]):
        super().__init__(message)
        self.inventory = inventory


@dataclass(frozen=True)
class FinalizedUpload:
    root: Path
    manifest: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def upload_id(self) -> str:
        return self.manifest["upload_id"]

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    def file(self, relpath: str) -> Path:
        return self.export_dir.joinpath(*relpath.split("/"))

    @property
    def files(self) -> list[dict[str, Any]]:
        return self.manifest["files"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_finalized_upload(upload_dir: str | Path, *, verify_hashes: bool = True) -> FinalizedUpload:
    root = Path(upload_dir)
    man_p = root / "manifest.json"
    if not man_p.is_file():
        raise UploadNotReady(f"{root} has no manifest.json: the upload is not finalized (incomplete transfer?)")
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    meta_p = root / "metadata.json"
    metadata = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    up = FinalizedUpload(root, manifest, metadata)
    problems = []
    for f in up.files:
        p = up.file(f["path"])
        if not p.is_file():
            problems.append(f"missing: {f['path']}")
        elif p.stat().st_size != f["size"]:
            problems.append(f"size differs: {f['path']}")
        elif verify_hashes and _sha256(p) != f["sha256"]:
            problems.append(f"sha256 differs: {f['path']}")
    listed = {f["path"] for f in up.files}
    extra = [p.relative_to(up.export_dir).as_posix() for p in up.export_dir.rglob("*") if p.is_file()]
    problems += [f"not in manifest: {p}" for p in extra if p not in listed]
    if problems:
        raise UploadIntegrityError(f"upload {root.name} does not match its manifest: " + "; ".join(problems[:20]))
    return up


# ---- generic inventory ------------------------------------------------------------------------

def _probe_npy(p: Path) -> dict[str, Any]:
    with open(p, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return {"kind": "npy", "shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran)}


def _probe_png(p: Path) -> dict[str, Any]:
    with open(p, "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return {"kind": "png", "error": "not a PNG header"}
    w, h, bit_depth, colour_type = struct.unpack(">IIBB", head[16:26])
    return {"kind": "png", "width": w, "height": h, "bit_depth": bit_depth,
            "colour_type": {0: "gray", 2: "rgb", 3: "palette", 4: "gray+alpha", 6: "rgba"}.get(colour_type, colour_type)}


def _probe_text(p: Path, n_lines: int = 4) -> dict[str, Any]:
    lines = []
    with open(p, "rb") as f:
        for _ in range(n_lines):
            line = f.readline(400)
            if not line:
                break
            lines.append(line.decode("utf-8", errors="replace").rstrip("\r\n")[:200])
    with open(p, "rb") as f:
        n_total = sum(1 for _ in f)
    return {"kind": "text", "first_lines": lines, "line_count": n_total}


def _probe(p: Path) -> dict[str, Any]:
    ext = p.suffix.lower()
    try:
        if ext == ".npy":
            return _probe_npy(p)
        if ext == ".png":
            return _probe_png(p)
        if ext in (".csv", ".txt", ".json", ".yaml", ".yml", ".xml"):
            return _probe_text(p)
        if ext in (".mp4", ".mov", ".m4v"):
            try:
                from ..ingest.video import probe_video  # other workstream's ffprobe wrapper (read-only use)
                return {"kind": "video", "probe": probe_video(p)}
            except Exception as e:  # ffprobe missing or failed: report, do not guess
                return {"kind": "video", "probe_error": str(e)[:300]}
    except Exception as e:
        return {"kind": "unreadable", "error": str(e)[:300]}
    return {"kind": "other"}


def inventory(up: FinalizedUpload, *, samples_per_group: int = 3) -> dict[str, Any]:
    """Describe the export without interpreting it. Groups files by (folder, extension)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for f in up.files:
        folder, _, name = f["path"].rpartition("/")
        groups[(folder, Path(name).suffix.lower())].append({**f, "name": name})
    out_groups = []
    for (folder, ext), files in sorted(groups.items()):
        files.sort(key=lambda f: f["name"])
        nums = [int(m.group(1)) for f in files if (m := re.fullmatch(r"\D*?(\d+)\.[^.]+", f["name"]))]
        sample_idx = sorted({0, len(files) // 2, len(files) - 1})[:samples_per_group]
        samples = [{"name": files[i]["name"], "size": files[i]["size"],
                    **_probe(up.file(files[i]["path"]))} for i in sample_idx]
        out_groups.append({
            "folder": folder or ".", "extension": ext or "(none)", "count": len(files),
            "total_bytes": sum(f["size"] for f in files),
            "numbered": len(nums) == len(files) and len(files) > 1,
            "number_range": [min(nums), max(nums)] if nums else None,
            "number_gaps": (max(nums) - min(nums) + 1 - len(set(nums))) if nums else None,
            "samples": samples,
        })
    return {
        "upload_id": up.upload_id,
        "recording_type": up.metadata.get("recording_type"),
        "file_count": len(up.files),
        "total_bytes": sum(f["size"] for f in up.files),
        "extensions": dict(Counter(Path(f["path"]).suffix.lower() or "(none)" for f in up.files)),
        "groups": out_groups,
        "note": "Generic description only. No export format has been assumed.",
    }


# ---- adapter registry -------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportAdapter:
    name: str
    detect: Callable[[FinalizedUpload], bool]
    load_views: Callable[[FinalizedUpload], list[RGBDView]]
    verified_against: str  # description of the real export this adapter was written from


ADAPTERS: list[ExportAdapter] = []  # intentionally empty until a real export has been inspected


def find_adapter(up: FinalizedUpload) -> ExportAdapter:
    matches = [a for a in ADAPTERS if a.detect(up)]
    if len(matches) == 1:
        return matches[0]
    inv = inventory(up)
    if not matches:
        raise NoAdapterError(
            f"BLOCKED: no export adapter recognises upload {up.upload_id}. No real RGB-D export has been inspected "
            "yet, so no file format, depth unit, intrinsics or timing is assumed. Share this inventory to have the "
            "adapter written against the actual files.", inv)
    raise NoAdapterError(f"ambiguous: several adapters match ({[a.name for a in matches]})", inv)
