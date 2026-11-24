"""Reading finalized uploads for RGB-D processing: integrity, generic inventory, and the explicit BLOCKED state.

File names below are GENERIC test files. They are not a guess at any capture app's export format.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import stat

import cv2
import numpy as np
import pytest

from digitaltrack.api.storage import UploadStore
from digitaltrack.cli import main as cli_main
from digitaltrack.rgbd.upload import (NoAdapterError, UploadIntegrityError, UploadNotReady, find_adapter, inventory,
                                      open_finalized_upload)


def _npy(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


FILES = {
    "rec/a/000000.npy": _npy(np.zeros((4, 5), np.uint16)),
    "rec/a/000001.npy": _npy(np.ones((4, 5), np.uint16)),
    "rec/b/000000.png": cv2.imencode(".png", np.zeros((4, 5), np.uint8))[1].tobytes(),
    "rec/table.csv": b"h1,h2\n1,2\n3,4\n",
    "rec/notes.bin": os.urandom(64),
}


async def _stream(data: bytes):
    yield data


def _make_upload(root, files=FILES, finalize=True):
    store = UploadStore(root)
    uid = store.create({"recording_type": "R2", "label": "test"})
    for p, d in files.items():
        asyncio.run(store.write_file(uid, p, _stream(d), 10**9))
    if finalize:
        store.finalize(uid, [{"path": p, "size": len(d)} for p, d in files.items()])
    return root / uid


def _tree(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_open_verifies_manifest_and_never_writes(tmp_path):
    d = _make_upload(tmp_path)
    before = _tree(d)
    up = open_finalized_upload(d)
    assert up.upload_id == d.name and len(up.files) == len(FILES)
    assert _tree(d) == before


def test_incomplete_upload_is_refused(tmp_path):
    with pytest.raises(UploadNotReady):
        open_finalized_upload(_make_upload(tmp_path, finalize=False))


def test_tampered_or_extra_files_are_detected(tmp_path):
    d = _make_upload(tmp_path)
    f = d / "export/rec/table.csv"
    os.chmod(f, stat.S_IWRITE | stat.S_IREAD)
    f.write_bytes(b"h1,h2\n9,9\n3,4\n")  # same size, different content
    with pytest.raises(UploadIntegrityError, match="sha256"):
        open_finalized_upload(d)
    d2 = _make_upload(tmp_path / "second")
    (d2 / "export/rec/sneaky.txt").write_text("x")
    with pytest.raises(UploadIntegrityError, match="not in manifest"):
        open_finalized_upload(d2)


def test_inventory_is_generic_and_safe(tmp_path):
    inv = inventory(open_finalized_upload(_make_upload(tmp_path)))
    groups = {(g["folder"], g["extension"]): g for g in inv["groups"]}
    npy = groups[("rec/a", ".npy")]
    assert npy["count"] == 2 and npy["numbered"] and npy["number_range"] == [0, 1]
    assert npy["samples"][0]["shape"] == [4, 5] and npy["samples"][0]["dtype"] == "uint16"
    png = groups[("rec/b", ".png")]["samples"][0]
    assert (png["width"], png["height"], png["bit_depth"], png["colour_type"]) == (5, 4, 8, "gray")
    assert groups[("rec", ".csv")]["samples"][0]["first_lines"][0] == "h1,h2"
    assert "No export format has been assumed" in inv["note"]


def test_pickled_npy_is_not_loaded(tmp_path):
    files = {"x/obj.npy": _npy(np.array([{"a": 1}], dtype=object))}
    inv = inventory(open_finalized_upload(_make_upload(tmp_path, files)))
    s = inv["groups"][0]["samples"][0]
    assert s["kind"] == "npy" and s["dtype"] == "object"  # header read only; the payload is never unpickled


def test_no_adapter_means_blocked_with_inventory(tmp_path):
    up = open_finalized_upload(_make_upload(tmp_path))
    with pytest.raises(NoAdapterError, match="BLOCKED") as ei:
        find_adapter(up)
    assert ei.value.inventory["file_count"] == len(FILES)


def test_cli_reconstruct_rgbd_reports_blocked_and_leaves_upload_untouched(tmp_path, capsys):
    d = _make_upload(tmp_path / "uploads")
    before = _tree(d)
    out = tmp_path / "processing"
    assert cli_main(["reconstruct-rgbd", str(d), "--out", str(out)]) == 2
    assert "BLOCKED" in capsys.readouterr().out
    assert json.loads((out / d.name / "inventory.json").read_text(encoding="utf-8"))["upload_id"] == d.name
    assert _tree(d) == before


def test_cli_inspect_upload(tmp_path, capsys):
    d = _make_upload(tmp_path)
    assert cli_main(["inspect-upload", str(d)]) == 0
    assert json.loads(capsys.readouterr().out)["file_count"] == len(FILES)
