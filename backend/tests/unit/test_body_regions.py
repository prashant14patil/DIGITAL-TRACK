"""Display-region segmentation on a synthetic A-pose figure built from primitives with known parts."""

import json

import numpy as np
import open3d as o3d
import pytest
import trimesh

from digitaltrack.experimental.body_regions import COLOUR_STEP, ID, segment_regions, write_region_files


def _cyl(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = trimesh.creation.cylinder(radius=r, segment=[p0, p1], sections=24)
    return trimesh.remesh.subdivide_to_size(c.vertices, c.faces, max_edge=0.01)


def figure(legs_apart=True):
    parts = {}
    lx = 0.08 if legs_apart else 0.045
    parts["left_leg"] = _cyl([lx, 0, 0.0], [lx, 0, 0.47], 0.05)
    parts["right_leg"] = _cyl([-lx, 0, 0.0], [-lx, 0, 0.47], 0.05)
    parts["trunk"] = _cyl([0, 0, 0.46], [0, 0, 0.82], 0.13)
    parts["head"] = _cyl([0, 0, 0.87], [0, 0, 1.0], 0.07)
    neck = _cyl([0, 0, 0.81], [0, 0, 0.88], 0.035)
    parts["left_arm"] = _cyl([0.16, 0, 0.78], [0.38, 0, 0.50], 0.03)
    parts["right_arm"] = _cyl([-0.16, 0, 0.78], [-0.38, 0, 0.50], 0.03)
    verts = np.vstack([v for v, _ in parts.values()] + [neck[0]])
    truth = np.concatenate([np.full(len(v), ID[n]) for n, (v, _) in parts.items()] + [np.full(len(neck[0]), -1)])
    return verts, truth


@pytest.mark.parametrize("legs_apart", [True, False])
def test_parts_are_labelled(legs_apart):
    v, truth = figure(legs_apart)
    labels, info = segment_regions(v)
    for name, rid in ID.items():
        sel = truth == rid
        if name == "trunk":  # trunk ends overlap legs/neck: require a clear majority
            assert (labels[sel] == rid).mean() > 0.8, name
        elif name == "head":
            assert (labels[sel & (v[:, 2] > 0.9)] == rid).mean() > 0.99
        else:
            # leg core away from the centre line (merged-leg cylinders overlap there: ambiguous ground truth)
            core = sel & (v[:, 2] < 0.75) & (v[:, 2] > 0.05) if "arm" in name else \
                sel & (v[:, 2] < 0.42) & (np.abs(v[:, 0]) > 0.01)
            assert (labels[core] == rid).mean() > 0.95, (name, (labels[core] == rid).mean())
    lv = info["levels_height_fraction"]
    assert 0.80 <= lv["neck"] <= 0.9  # true neck cylinder spans 0.81-0.88
    if legs_apart:
        assert info["crotch_method"] == "detected leg gap" and 0.4 <= lv["crotch"] <= 0.5
    else:
        assert info["crotch_method"].startswith("anthropometric default")
    anchors = {r["name"]: r["anchor"] for r in info["regions"]}
    assert anchors["left_arm"][0] > 0.15 and anchors["right_arm"][0] < -0.15
    assert anchors["left_leg"][0] > 0 > anchors["right_leg"][0]


def test_tiny_or_flat_input_is_refused():
    with pytest.raises(ValueError):
        segment_regions(np.zeros((50, 3)))


def test_region_files_round_trip(tmp_path):
    tm = trimesh.util.concatenate([trimesh.Trimesh(*_cyl([0.08, 0, 0], [0.08, 0, 0.47], 0.05)),
                                   trimesh.Trimesh(*_cyl([-0.08, 0, 0], [-0.08, 0, 0.47], 0.05)),
                                   trimesh.Trimesh(*_cyl([0, 0, 0.46], [0, 0, 0.82], 0.13)),
                                   trimesh.Trimesh(*_cyl([0, 0, 0.87], [0, 0, 1.0], 0.07))])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices), o3d.utility.Vector3iVector(tm.faces))
    o3d.io.write_triangle_mesh(str(tmp_path / "mesh_height_normalised.ply"), mesh)
    info = write_region_files(tmp_path)
    g = trimesh.load(tmp_path / "mesh_regions.glb", force="mesh")
    ids = np.round(g.visual.vertex_colors[:, 0] / COLOUR_STEP).astype(int)
    assert len(g.vertices) == len(tm.vertices)
    assert np.bincount(ids, minlength=6).tolist() == [r["vertex_count"] for r in info["regions"]]
    assert json.loads((tmp_path / "regions.json").read_text())["status"] == "provisional"
