"""Shared fixtures. Sessions are rendered once per test run into pytest temp dirs.

Everything here is SYNTHETIC; no real human scan data exists in or is needed by the tests.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest

from digitaltrack.synthetic.renderer import RenderConfig, render_session
from digitaltrack.synthetic.shapes import cylinder


def make_writable(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


@pytest.fixture(scope="session")
def small_cylinder_session(tmp_path_factory) -> Path:
    """6 views of a clean cylinder: cheap session for validator/writer tests."""
    return render_session(cylinder(0.15, 0.6), tmp_path_factory.mktemp("small") / "s", RenderConfig(n_views=6))


@pytest.fixture()
def mutable_session(small_cylinder_session, tmp_path) -> Path:
    """A writable copy of the small session that a test may corrupt."""
    dst = tmp_path / "copy"
    shutil.copytree(small_cylinder_session, dst)
    make_writable(dst)
    return dst


@pytest.fixture(scope="session")
def cylinder_body():
    return cylinder(0.15, 0.6)


@pytest.fixture(scope="session")
def full_cylinder_session(tmp_path_factory, cylinder_body) -> Path:
    """36-view noise-free cylinder (the reference reconstruction case)."""
    return render_session(cylinder_body, tmp_path_factory.mktemp("full") / "s", RenderConfig(n_views=36))
