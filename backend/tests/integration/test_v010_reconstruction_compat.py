"""Backward compatibility end to end: a frozen 0.1.0 session and its 0.2.0 migrated copy must reconstruct to
identical geometry and measurements (the migration changes metadata only, never data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from digitaltrack.cli import main as cli_main
from digitaltrack.dataset.loader import load_session
from digitaltrack.dataset.migrate import migrate_session_copy
from digitaltrack.dataset.validate import validate_session
from digitaltrack.reconstruction.tsdf import reconstruct_tsdf

FIXTURE_V010 = Path(__file__).resolve().parents[1] / "fixtures" / "session_v0_1_0_cylinder"


def _canonical_vertices(mesh) -> np.ndarray:
    v = np.asarray(mesh.vertices)
    return v[np.lexsort(v.T[::-1])]


def _canonical_triangles(mesh) -> np.ndarray:
    """Each triangle as its 3 vertex coordinates (sorted within the triangle), rows sorted."""
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]  # (n, 3, 3)
    tri = np.sort(tri.view([("x", float), ("y", float), ("z", float)]).reshape(tri.shape[:2]), axis=1)
    flat = tri.view(float).reshape(len(tri), 9)
    return flat[np.lexsort(flat.T[::-1])]


@pytest.mark.slow
def test_v010_and_migrated_copy_reconstruct_identically(tmp_path):
    migrated = migrate_session_copy(FIXTURE_V010, tmp_path / "migrated")
    a = reconstruct_tsdf(load_session(FIXTURE_V010), run_id="compat", write=False)
    b = reconstruct_tsdf(load_session(migrated), run_id="compat", write=False)
    assert len(a.mesh.triangles) > 0
    # Open3D's hashed voxel grid emits vertices in a run-dependent ORDER even for identical input (checked:
    # two runs on the same session differ in order, not in content), so compare order-independent geometry.
    assert np.array_equal(_canonical_vertices(a.mesh), _canonical_vertices(b.mesh))
    assert np.array_equal(_canonical_triangles(a.mesh), _canonical_triangles(b.mesh))
    # The ICP diagnostic sums in mesh-vertex order, so floats can differ in the last bits (~1e-16 relative,
    # also between two runs on the SAME session). Statuses and pass/fail must match exactly.
    _assert_same(a.quality, b.quality, "quality", skip={"generated_utc"})


def _assert_same(x, y, path: str, skip=frozenset()) -> None:
    if isinstance(x, dict):
        assert isinstance(y, dict) and set(x) == set(y), path
        for k in x:
            if k not in skip:
                _assert_same(x[k], y[k], f"{path}.{k}", skip)
    elif isinstance(x, list):
        assert isinstance(y, list) and len(x) == len(y), path
        for i, (u, v) in enumerate(zip(x, y)):
            _assert_same(u, v, f"{path}[{i}]", skip)
    elif isinstance(x, float) and not isinstance(x, bool):
        assert np.isclose(x, y, rtol=1e-9, atol=1e-12), (path, x, y)
    else:
        assert x == y, (path, x, y)


def test_cli_migrate_and_validate(tmp_path, capsys):
    dst = tmp_path / "m"
    assert cli_main(["migrate", str(FIXTURE_V010), str(dst)]) == 0
    assert validate_session(dst).ok
    assert cli_main(["migrate", str(FIXTURE_V010), str(dst)]) == 1  # non-empty destination refused
    assert "migration failed" in capsys.readouterr().out
