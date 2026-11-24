"""Raw capture-upload storage.

An upload is a set of files exactly as exported by the capture app (folder structure preserved). They are
written byte-for-byte, hashed (SHA-256) while streaming, never overwritten, and made read-only when the upload
is finalized. The export format is NOT interpreted here: no adapter exists until a real export has been
inspected (see CLAUDE.md / the capture inspection report).

Layout:  <data_dir>/<upload_id>/
            metadata.json     what the uploader entered (recording type, object, device, protocol)
            export/...        the uploaded files, relative paths preserved
            manifest.json     written on finalize: every file with size + sha256
         <data_dir>/uploads.sqlite   index (status, metadata, files)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
FORBIDDEN_CHARS = set('<>:"|?*')
MAX_PATH_LEN = 1024


class UploadError(ValueError):
    """Client error: bad path, wrong state, duplicate file, verification failure."""


class UploadNotFound(KeyError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_relpath(path: str) -> str:
    """Validate a client-supplied relative path; return it with '/' separators. Never trusts the client."""
    p = path.replace("\\", "/")
    if not p or len(p) > MAX_PATH_LEN:
        raise UploadError("empty or too long path")
    if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        raise UploadError(f"absolute path not allowed: {path!r}")
    parts = p.split("/")
    for s in parts:
        if s in ("", ".", ".."):
            raise UploadError(f"invalid path segment in {path!r}")
        if any(ord(c) < 32 for c in s) or FORBIDDEN_CHARS & set(s):
            raise UploadError(f"invalid character in {path!r}")
        if s != s.rstrip(" ."):
            raise UploadError(f"path segment ends with space or dot: {path!r}")
        if s.split(".")[0].upper() in WINDOWS_RESERVED:
            raise UploadError(f"reserved file name: {path!r}")
    return "/".join(parts)


@dataclass
class UploadStore:
    data_dir: Path

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, finalized_utc TEXT,
                    status TEXT NOT NULL, recording_type TEXT, label TEXT, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS files (
                    upload_id TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
                    received_utc TEXT NOT NULL, PRIMARY KEY (upload_id, path));
            """)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection: commit on success, roll back on error, always close."""
        db = sqlite3.connect(self.data_dir / "uploads.sqlite")
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _dir(self, upload_id: str) -> Path:
        if not re.fullmatch(r"up_\d{8}T\d{6}Z_[0-9a-f]{6}", upload_id):
            raise UploadNotFound(upload_id)
        return self.data_dir / upload_id

    def _row(self, db: sqlite3.Connection, upload_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        if row is None:
            raise UploadNotFound(upload_id)
        return row

    # ---- lifecycle ------------------------------------------------------------------------------

    def create(self, metadata: dict[str, Any]) -> str:
        upload_id = f"up_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        d = self._dir(upload_id)
        (d / "export").mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        with self._lock, self._db() as db:
            db.execute("INSERT INTO uploads VALUES (?, ?, NULL, 'receiving', ?, ?, ?)",
                       (upload_id, utc_now(), metadata.get("recording_type"), metadata.get("label"),
                        json.dumps(metadata)))
        return upload_id

    async def write_file(self, upload_id: str, path: str, chunks: AsyncIterator[bytes],
                         max_bytes: int) -> dict[str, Any]:
        rel = safe_relpath(path)
        with self._db() as db:
            if self._row(db, upload_id)["status"] != "receiving":
                raise UploadError("upload is finalized; files can no longer be added")
        export = (self._dir(upload_id) / "export").resolve()
        dest = export.joinpath(*rel.split("/"))
        if export not in dest.resolve().parents:
            raise UploadError(f"path escapes the upload folder: {path!r}")
        if dest.exists():
            raise UploadError(f"file already uploaded (raw files are never overwritten): {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + f".part-{uuid.uuid4().hex[:8]}")
        h, size = hashlib.sha256(), 0
        try:
            with open(part, "xb") as f:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadError(f"file exceeds the {max_bytes} byte limit: {rel}")
                    h.update(chunk)
                    f.write(chunk)
            with self._lock:
                if dest.exists():
                    raise UploadError(f"file already uploaded (raw files are never overwritten): {rel}")
                os.replace(part, dest)
                with self._db() as db:
                    db.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?)",
                               (upload_id, rel, size, h.hexdigest(), utc_now()))
        finally:
            if part.exists():
                part.unlink()
        return {"path": rel, "size": size, "sha256": h.hexdigest()}

    def finalize(self, upload_id: str, expected: list[dict[str, Any]]) -> dict[str, Any]:
        """Check that exactly the expected files arrived with the expected sizes, then freeze the upload."""
        exp = {safe_relpath(e["path"]): int(e["size"]) for e in expected}
        with self._lock, self._db() as db:
            row = self._row(db, upload_id)
            if row["status"] != "receiving":
                raise UploadError(f"upload is already {row['status']}")
            got = {r["path"]: r for r in db.execute("SELECT * FROM files WHERE upload_id = ?", (upload_id,))}
            problems = [f"missing: {p}" for p in sorted(exp.keys() - got.keys())]
            problems += [f"not expected: {p}" for p in sorted(got.keys() - exp.keys())]
            problems += [f"size differs: {p} (expected {exp[p]}, received {got[p]['size']})"
                         for p in sorted(exp.keys() & got.keys()) if got[p]["size"] != exp[p]]
            if not got:
                problems.append("no files received")
            d = self._dir(upload_id)
            for p, r in got.items():  # re-verify what is on disk before freezing
                f = d / "export" / Path(*p.split("/"))
                if not f.is_file() or f.stat().st_size != r["size"]:
                    problems.append(f"on-disk file differs from what was received: {p}")
            if problems:
                raise UploadError("; ".join(problems))
            manifest = {
                "upload_id": upload_id, "finalized_utc": utc_now(),
                "note": "Files under export/ are byte-for-byte as uploaded. Nothing has been converted.",
                "file_count": len(got), "total_bytes": sum(r["size"] for r in got.values()),
                "files": [{"path": p, "size": got[p]["size"], "sha256": got[p]["sha256"]} for p in sorted(got)],
            }
            (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            for f in [d / "manifest.json", d / "metadata.json", *(p for p in (d / "export").rglob("*") if p.is_file())]:
                os.chmod(f, stat.S_IREAD)
            db.execute("UPDATE uploads SET status = 'complete', finalized_utc = ? WHERE id = ?",
                       (manifest["finalized_utc"], upload_id))
        return manifest

    def discard(self, upload_id: str) -> None:
        """Delete an upload that was never finalized (a failed or abandoned transfer). Finalized uploads are kept."""
        import shutil

        with self._lock, self._db() as db:
            if self._row(db, upload_id)["status"] != "receiving":
                raise UploadError("finalized uploads are raw data and cannot be deleted from the web page")
            db.execute("DELETE FROM files WHERE upload_id = ?", (upload_id,))
            db.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
        shutil.rmtree(self._dir(upload_id), ignore_errors=True)

    # ---- queries --------------------------------------------------------------------------------

    def get(self, upload_id: str) -> dict[str, Any]:
        with self._db() as db:
            row = self._row(db, upload_id)
            files = [dict(r) for r in db.execute(
                "SELECT path, size, sha256, received_utc FROM files WHERE upload_id = ? ORDER BY path", (upload_id,))]
        return {"id": row["id"], "created_utc": row["created_utc"], "finalized_utc": row["finalized_utc"],
                "status": row["status"], "metadata": json.loads(row["metadata"]),
                "file_count": len(files), "total_bytes": sum(f["size"] for f in files), "files": files}

    def list(self) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("""
                SELECT u.id, u.created_utc, u.finalized_utc, u.status, u.recording_type, u.label,
                       COUNT(f.path) AS file_count, COALESCE(SUM(f.size), 0) AS total_bytes
                FROM uploads u LEFT JOIN files f ON f.upload_id = u.id
                GROUP BY u.id ORDER BY u.created_utc DESC""").fetchall()
        return [dict(r) for r in rows]
