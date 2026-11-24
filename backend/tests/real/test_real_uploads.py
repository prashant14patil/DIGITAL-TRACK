"""REAL-recording validation. Kept separate from synthetic tests and skipped unless real data is configured.

Set DIGITALTRACK_REAL_UPLOADS to a folder of finalized uploads (default layout data/real/uploads/<upload_id>/).
Nothing here modifies the uploads. Reconstruction is only attempted once an export adapter exists; until then
the test records the explicit BLOCKED state and the inventory to inspect.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from digitaltrack.rgbd.upload import NoAdapterError, find_adapter, inventory, open_finalized_upload

pytestmark = pytest.mark.real

ROOT = os.environ.get("DIGITALTRACK_REAL_UPLOADS")
UPLOADS = sorted(p for p in Path(ROOT).iterdir() if (p / "manifest.json").is_file()) if ROOT and Path(ROOT).is_dir() else []


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.skipif(not UPLOADS, reason="no real finalized uploads configured (DIGITALTRACK_REAL_UPLOADS)")
@pytest.mark.parametrize("upload_dir", UPLOADS, ids=[p.name for p in UPLOADS])
def test_real_upload_integrity_and_adapter_state(upload_dir, tmp_path):
    before = _tree(upload_dir)
    up = open_finalized_upload(upload_dir)  # verifies every SHA-256 in the manifest
    inv = inventory(up)
    assert inv["file_count"] == len(up.files)
    try:
        adapter = find_adapter(up)
    except NoAdapterError:
        pytest.xfail("BLOCKED: no export adapter yet; inspect `dt inspect-upload` output to write one")
    views = adapter.load_views(up)
    for v in views:
        v.validate()
    assert _tree(upload_dir) == before
