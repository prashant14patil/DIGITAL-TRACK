"""experimental.template_fit: the CC0 template fitted to a SYNTHETIC body (display only)."""

from __future__ import annotations

import numpy as np
import pytest

from digitaltrack.experimental.template_fit import REGIONS, TEMPLATE, fit_template, load_template
from digitaltrack.synthetic.body_figure import Figure, build

pytestmark = pytest.mark.skipif(not TEMPLATE.is_file(), reason="template asset missing")


def test_template_loads_in_display_frame():
    t = load_template()
    x = t["vertices"]
    assert 1.5 < x[:, 2].max() < 1.9 and abs(x[:, 2].min()) < 0.01  # metres, feet on the floor
    assert t["joints"]["l-shoulder"][0] > 0 > t["joints"]["r-shoulder"][0]  # +X = the body's left
    assert t["joints"]["l-eye"][1] < t["joints"]["head"][1]  # faces -Y
    assert "CC0" in t["license"]


def _scaled_template(h=1.08, w=1.10):
    import open3d as o3d

    t = load_template()
    x = t["vertices"].copy()
    c = np.median(x[:, :2], 0)
    x[:, :2] = c + (x[:, :2] - c) * w
    x[:, 2] *= h
    body = ~t["is_eye"]
    keep = np.all(body[t["triangles"]], axis=1)
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x), o3d.utility.Vector3iVector(t["triangles"][keep]))


def test_fit_follows_a_known_human_shape_and_keeps_template_detail():
    target = _scaled_template()  # the template itself, 8 % taller and 10 % wider: a known human shape
    top = np.asarray(target.vertices)[:, 2].max()
    fitted, labels, template_only, rep = fit_template(target)
    x = np.asarray(fitted.vertices)
    assert rep["status"] == "fitted"
    assert abs(x[:, 2].max() - top) < 0.03  # height follows the target
    assert set(np.unique(labels)) == set(range(len(REGIONS)))  # every region present
    assert template_only[labels == 0].mean() > 0.9  # the head is template detail, shown pale
    assert rep["deviation_full_weight_m"]["rms"] < 0.01


def test_fit_is_refused_when_the_target_is_not_a_person():
    """A plain pillar has no legs, arms or neck: the fit is refused (caller shows the scan), never shown as fitted."""
    import open3d as o3d

    pillar = o3d.geometry.TriangleMesh.create_box(0.6, 0.6, 1.8).translate((-0.3, -0.3, 0.0)).subdivide_midpoint(3)
    with pytest.raises(ValueError):
        fit_template(pillar)
