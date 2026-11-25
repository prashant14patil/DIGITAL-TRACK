"""Unpack uploaded .zip exports into a DERIVED copy and describe what is inside.

The zip under export/ stays byte-for-byte as uploaded. The copy goes to <upload>/extracted/<zip name>/ and can be
deleted and regenerated at any time. No export format is interpreted here.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..rgbd.upload import FinalizedUpload, inventory, open_finalized_upload
from .storage import UploadError, safe_relpath

MAX_MEMBERS = 200_000
MAX_UNPACKED_BYTES = 40 * 1024**3


def _wanted(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return "__MACOSX" not in parts and parts[-1] not in ("", ".DS_Store")


DONE_MARKER = ".unpacked_ok"


def _unpack(zip_path: Path, dest: Path) -> None:
    """Extract into `dest`, then write a marker; a folder without the marker is incomplete and is redone.
    (No folder rename: on Windows, antivirus scanning freshly written files makes renames fail with WinError 5.)"""
    if dest.exists():
        shutil.rmtree(dest)
    tmp = dest
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = [m for m in z.infolist() if not m.is_dir() and _wanted(m.filename)]
            if not members:
                raise UploadError(f"{zip_path.name} contains no files")
            if len(members) > MAX_MEMBERS or sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
                raise UploadError(f"{zip_path.name} is too large to unpack")
            for m in members:
                rel = safe_relpath(m.filename)  # rejects absolute paths and '..'
                out = tmp.joinpath(*rel.split("/"))
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(out, "xb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        (dest / DONE_MARKER).write_text("complete\n", encoding="utf-8")
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError, UploadError) as e:
        shutil.rmtree(dest, ignore_errors=True)
        if isinstance(e, UploadError):
            raise
        raise UploadError(f"cannot unpack {zip_path.name}: {e}") from e


@dataclass(frozen=True)
class _Extracted(FinalizedUpload):
    @property
    def export_dir(self) -> Path:
        return self.root / "extracted"


def inspect_archives(upload_dir: Path) -> dict[str, Any]:
    """Unpack every zip of a finalized upload (once) and return a generic description of the contents."""
    up = open_finalized_upload(upload_dir, verify_hashes=False)  # hashes were taken while receiving
    zips = [f["path"] for f in up.files if f["path"].lower().endswith(".zip")]
    if not zips:
        raise UploadError("this upload has no .zip file")
    root = up.root / "extracted"
    root.mkdir(exist_ok=True)
    for rel in zips:
        dest = root / rel.replace("/", "__")[:-4]
        if not (dest / DONE_MARKER).is_file():
            _unpack(up.file(rel), dest)
    files = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != DONE_MARKER and ".part-" not in p.as_posix()]
    result = inventory(_Extracted(up.root, {**up.manifest, "files": files}, up.metadata))
    result["zips"] = zips
    return result
