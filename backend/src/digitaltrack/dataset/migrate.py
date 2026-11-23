"""Dataset schema migration (0.1.0 -> 0.2.0).

Two uses:
* `upgrade_documents` upgrades JSON documents IN MEMORY. The loader uses it so the pipeline sees one shape.
  Nothing on disk changes.
* `migrate_session_copy` writes an upgraded COPY to a new directory. The source session is never modified
  (raw data is immutable). Raw binary files are copied byte-for-byte and re-verified against their recorded
  SHA-256; lineage is recorded in `migrated_from`.

0.1.0 -> 0.2.0 is purely additive for depth sessions: no raw byte and no numeric value changes.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

from .. import DATASET_SCHEMA_VERSION, __version__
from .schemas import schema_errors
from .writer import _write_json, make_readonly, sha256_file, utc_now

Doc = dict[str, Any]


class MigrationError(ValueError):
    """The source cannot be migrated without guessing; nothing was written for it."""


def _bump(doc: Doc, expect: str) -> Doc:
    if doc.get("schema_version") != expect:
        raise MigrationError(f"expected schema_version {expect!r}, got {doc.get('schema_version')!r}")
    out = copy.deepcopy(doc)
    out["schema_version"] = "0.2.0"
    return out


def upgrade_session_010(doc: Doc) -> Doc:
    out = _bump(doc, "0.1.0")
    # 0.1.0 only knew depth input; `synthetic` tells which adapter wrote it.
    out["input_type"] = "synthetic_depth" if doc.get("synthetic") else "rgbd_depth"
    return out


def upgrade_frame_010(doc: Doc) -> Doc:
    return _bump(doc, "0.1.0")


def upgrade_camera_010(doc: Doc) -> Doc:
    out = _bump(doc, "0.1.0")
    out["image_source"] = "depth"  # 0.1.0 intrinsics were defined for the depth image
    dist = out.get("distortion")
    if dist is not None and "model" not in dist:
        if dist:
            raise MigrationError("0.1.0 distortion block has no model; refusing to guess its meaning")
        out.pop("distortion")
    return out


def upgrade_turntable_010(doc: Doc) -> Doc:
    return _bump(doc, "0.1.0")


def upgrade_documents(session: Doc, frames: list[Doc], camera: Doc | None,
                      turntable: Doc | None) -> tuple[Doc, list[Doc], Doc | None, Doc | None]:
    """Return current-version copies of a session's JSON documents (inputs untouched)."""
    v = session.get("schema_version")
    if v == DATASET_SCHEMA_VERSION:
        return session, frames, camera, turntable
    if v != "0.1.0":
        raise MigrationError(f"no migration path from schema_version {v!r}")
    return (upgrade_session_010(session), [upgrade_frame_010(f) for f in frames],
            None if camera is None else upgrade_camera_010(camera),
            None if turntable is None else upgrade_turntable_010(turntable))


def migrate_session_copy(src: str | Path, dst: str | Path, *, readonly: bool = True) -> Path:
    """Write an upgraded copy of `src` to `dst` (which must not exist or be empty)."""
    from .validate import validate_session  # local import: validate imports writer too

    src, dst = Path(src), Path(dst)
    rep = validate_session(src)
    if not rep.ok:
        raise MigrationError("source session is invalid; fix or re-import it first: "
                             + "; ".join(str(i) for i in rep.errors))
    from .loader import _read_json
    session = _read_json(src / "session.json")
    if session["schema_version"] == DATASET_SCHEMA_VERSION:
        raise MigrationError("source is already at the current schema version")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty directory: {dst}")

    frame_docs = [_read_json(src / e["dir"] / "frame_metadata.json") for e in session["frames"]]
    cam_p, tt_p = src / "calibration/camera_calibration.json", src / "calibration/turntable_calibration.json"
    cam = _read_json(cam_p) if cam_p.is_file() else None
    tt = _read_json(tt_p) if tt_p.is_file() else None
    new_session, new_frames, new_cam, new_tt = upgrade_documents(session, frame_docs, cam, tt)
    new_session["migrated_from"] = {
        "schema_version": session["schema_version"],
        "session_json_sha256": sha256_file(src / "session.json"),
        "migrated_utc": utc_now(),
        "tool": f"digitaltrack.dataset.migrate {__version__}",
    }
    for doc, name in [(new_session, "session"), *[(f, "frame_metadata") for f in new_frames],
                      *([(new_cam, "camera_calibration")] if new_cam else []),
                      *([(new_tt, "turntable_calibration")] if new_tt else [])]:
        errs = schema_errors(doc, name)
        if errs:
            raise MigrationError(f"migrated {name} violates current schema: {errs}")

    written: list[Path] = []
    (dst / "calibration").mkdir(parents=True, exist_ok=True)
    for entry, fdoc in zip(session["frames"], new_frames):
        fdir = dst / entry["dir"]
        fdir.mkdir(parents=True)
        for key in ("depth", "confidence", "rgb", "depth_smoothed"):
            block = fdoc.get(key)
            if not block or "file" not in block:
                continue
            s, d = src / entry["dir"] / block["file"], fdir / block["file"]
            shutil.copyfile(s, d)
            if "sha256" in block and sha256_file(d) != block["sha256"]:
                raise MigrationError(f"copied {d} does not match its recorded sha256")
            written.append(d)
        _write_json(fdir / "frame_metadata.json", fdoc)
        written.append(fdir / "frame_metadata.json")
    _write_json(dst / "session.json", new_session)
    written.append(dst / "session.json")
    for doc, p in ((new_cam, "calibration/camera_calibration.json"), (new_tt, "calibration/turntable_calibration.json")):
        if doc is not None:
            _write_json(dst / p, doc)
            written.append(dst / p)
    # Top-level extra files (e.g. synthetic ground_truth.json) are copied unchanged. processing/ is derived
    # data tied to the old version and is NOT copied; re-run processing on the migrated session.
    for p in src.iterdir():
        if p.is_file() and p.name != "session.json":
            shutil.copyfile(p, dst / p.name)
            written.append(dst / p.name)
    if readonly:
        for p in written:
            make_readonly(p)
    out = validate_session(dst)
    if not out.ok:
        raise MigrationError("migrated copy failed validation: " + "; ".join(str(i) for i in out.errors))
    return dst
