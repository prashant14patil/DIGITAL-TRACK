"""Mesh-vs-tape comparison on SYNTHETIC meshes with exactly known dimensions."""

from __future__ import annotations

import json

import numpy as np
import open3d as o3d
import pytest

from digitaltrack.validation.compare import COMPARE_VERSION, compare_run


def _run(tmp_path, mesh, status="pass"):
    d = tmp_path / "run"
    d.mkdir()
    o3d.io.write_triangle_mesh(str(d / "fused_mesh.ply"), mesh)
    (d / "run_log.json").write_text(json.dumps({"run_id": "r", "status": status}), encoding="utf-8")
    return d


def _box(l=0.40, w=0.30, h=0.50, yaw_deg=20.0):
    m = o3d.geometry.TriangleMesh.create_box(l, w, h).translate((-l / 2, -w / 2, 0))
    m.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(yaw_deg))), center=(0, 0, 0))
    return m.subdivide_midpoint(3)


def _cylinder(r=0.15, h=1.7):
    return o3d.geometry.TriangleMesh.create_cylinder(r, h, resolution=360, split=40).translate((0, 0, h / 2))


def M(name, mm, level=None):
    d = {"name": name, "readings_mm": [mm, mm + 1, mm - 1], "instrument": "tape"}
    if level is not None:
        d["level_mm"] = level
    return d


def test_box_dimensions_match_known_box(tmp_path):
    run = _run(tmp_path, _box())
    meta = {"object_kind": "box", "measurements": [M("length", 400), M("width", 300), M("height", 500)]}
    out = compare_run(run, meta)
    rows = {r["name"]: r for r in out["rows"]}
    for name in ("length", "width", "height"):
        assert rows[name]["status"] == "measured", rows[name]
        assert abs(rows[name]["error_mm"]) < 3, rows[name]  # rotated box, percentile height
    assert out["compare_version"] == COMPARE_VERSION and (run / "comparison.json").is_file()
    assert rows["length"]["tape"]["sd_mm"] == pytest.approx(1.0)


def test_circumference_at_tape_level(tmp_path):
    run = _run(tmp_path, _cylinder(0.15, 1.7))
    truth = 2 * np.pi * 150
    meta = {"object_kind": "none", "contains_person": True,
            "measurements": [M("waist", truth, level=1000), M("thigh", truth, level=600), M("chest", truth),
                             M("height", 1700), M("shoe size", 42)]}
    rows = {r["name"]: r for r in compare_run(run, meta)["rows"]}
    assert rows["waist"]["status"] == "measured" and abs(rows["waist"]["error_mm"]) < 2
    assert rows["thigh"]["status"] == "estimated"  # limb: horizontal cut, not perpendicular
    assert rows["chest"]["status"] == "unavailable" and "level" in rows["chest"]["why"]  # no level: no guess
    assert abs(rows["height"]["error_mm"]) < 3
    assert rows["shoe size"]["status"] == "unavailable"


def test_warn_run_is_unreliable_not_measured(tmp_path):
    run = _run(tmp_path, _box(), status="warn")
    rows = compare_run(run, {"object_kind": "box", "measurements": [M("height", 500)]})["rows"]
    assert rows[0]["status"] == "unreliable" and rows[0]["mesh_mm"] is not None


def test_no_mesh_means_unavailable(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "run_log.json").write_text(json.dumps({"status": "fail"}), encoding="utf-8")
    out = compare_run(d, {"object_kind": "box", "measurements": [M("height", 500)]})
    assert out["rows"][0]["status"] == "unavailable" and out["rows"][0]["mesh_mm"] is None
