"""rgbd.rgbmask on a SYNTHETIC scene: floor leak and wall halo are removed, the body (even light trousers) is kept."""

from __future__ import annotations

import numpy as np

from digitaltrack.dataset.loader import Intrinsics
from digitaltrack.rgbd.rgbmask import RGBMaskParams, build_background, refine_mask
from digitaltrack.rgbd.views import RGBDView

W, H = 128, 96  # depth size; RGB 4x
K = Intrinsics(100.0, 100.0, 64.0, 48.0, W, H)


def _floor_frame():
    """Camera at 1 m height looking horizontally: floor = rows below the horizon. floor_from_cam maps camera
    (x right, y down, z forward) to floor (z up)."""
    t = np.eye(4)
    t[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # floor x = cam x, floor y = cam z, floor z = -cam y
    t[2, 3] = 1.0
    return t


def _scene(rng):
    v, u = np.mgrid[0:H, 0:W]
    ray_y = (v - K.cy) / K.fy
    depth = np.full((H, W), 3.0, np.float32)  # wall at 3 m
    floor_rows = ray_y > 1.0 / 3.0  # the floor (1 m below the camera) is nearer than the wall there
    depth[floor_rows] = (1.0 / ray_y[floor_rows]).astype(np.float32)
    rgb = np.zeros((H * 4, W * 4, 3), np.uint8)
    rgb[...] = (205, 205, 205)  # white wall (BGR)
    fr = np.repeat(np.repeat(floor_rows, 4, 0), 4, 1)
    rgb[fr] = (120, 60, 30)  # dark blue carpet
    rgb = np.clip(rgb.astype(int) + rng.normal(0, 3, rgb.shape), 0, 255).astype(np.uint8)
    return depth, rgb


def test_floor_leak_and_halo_removed_body_kept():
    rng = np.random.default_rng(0)
    plate_depth, plate_rgb = _scene(rng)
    fv = RGBDView(0, plate_depth, K, None)
    bg = build_background([plate_rgb], plate_depth, fv, _floor_frame(), RGBMaskParams(work_size=(W * 4, H * 4)))
    depth, rgb = _scene(np.random.default_rng(1))
    body = np.zeros((H, W), bool)
    body[20:80, 56:72] = True  # standing person, feet reach the floor rows
    depth[body] = 2.0
    rgb4 = np.repeat(np.repeat(body, 4, 0), 4, 1)
    rgb[rgb4] = (185, 190, 195)  # light trousers, close to the wall colour on purpose
    shirt = rgb4 & (np.arange(H * 4)[:, None] < 120)
    rgb[shirt] = (90, 40, 160)  # shirt on the upper body
    mask = body.copy()
    mask[80:92, 44:84] = True  # floor leak around the feet
    mask[30:50, 72] = True  # 1-pixel halo on the wall beside the body
    out, st = refine_mask(mask, RGBDView(0, depth, K, None), rgb, _floor_frame(), bg, RGBMaskParams(work_size=(W * 4, H * 4)))
    leak = mask & ~body
    assert out[body].mean() > 0.97, st  # the body, including light trousers, survives
    assert out[80:92, 44:84][~body[80:92, 44:84]].mean() < 0.1, st  # floor leak removed
    assert out[30:50, 72].mean() <= 0.1, st  # wall halo removed (threshold = wall P95, so a stray pixel may stay)
    assert st["pixels_out"] < st["pixels_in"] and leak.sum() > 0
