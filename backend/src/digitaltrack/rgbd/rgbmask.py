"""Sharpen the depth-based subject mask with the RGB video (fixed camera + empty-scene recording).

Why (first real recording): ARKit scene depth is smoothed, so depth from the body "bleeds" into neighbouring pixels.
Overlaying the depth mask on the RGB frame showed a large loop on the carpet around the feet (floor counted as body)
and a ~1 pixel (~1 cm) halo around arms, head and between the legs. The RGB image is 7.5x sharper than the depth.
The mask is only ever REDUCED, never grown, and interior trousers/shirt pixels are protected by the rules below.

Two vetoes, both learned from the empty-scene recording (same rig):
  * floor: the colours of floor pixels (empty-scene depth within 2 cm of the floor plane) form a colour histogram;
    a depth pixel lower than `floor_veto_height_m` whose RGB footprint is mostly floor-coloured is removed. This needs
    no pixel alignment between the clips (the carpet's fine pattern did NOT line up: the iPad shifted slightly).
  * wall: above `wall_min_height_m`, a depth pixel whose RGB footprint mostly matches the empty-scene image (wall area
    only, where differencing is reliable) is removed. The match threshold is the 95th percentile of the frame's own
    wall difference away from the person (exposure gain corrected), so it adapts to video noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..geometry.transforms import transform_points
from .views import RGBDView, backproject


@dataclass(frozen=True)
class RGBMaskParams:
    work_size: tuple[int, int] = (960, 720)
    plate_frames: int = 30
    floor_band_m: float = 0.02  # empty-scene pixels this close to the floor plane are floor samples
    floor_hist_bins: int = 32
    floor_hist_mass: float = 0.995  # colours covering this share of floor samples count as floor-coloured
    floor_veto_height_m: float = 0.25
    floor_veto_fraction: float = 0.5
    wall_min_height_m: float = 0.10
    wall_match_percentile: float = 95.0
    wall_veto_fraction: float = 0.8


@dataclass
class BackgroundRGB:
    plate_lab: np.ndarray  # (h, w, 3) float32 Lab of the empty scene
    floor_hist: np.ndarray  # (b, b, b) normalised
    floor_thr: float
    wall_px: np.ndarray  # (h, w) bool, empty-scene pixels on the wall (above wall_min_height_m)
    info: dict


def _heights(depth: np.ndarray, view: RGBDView, floor_from_cam: np.ndarray) -> np.ndarray:
    m = np.isfinite(depth) & (depth > 0)
    h = np.full(depth.shape, np.nan)
    vv, uu = np.nonzero(m)
    v = RGBDView(view.index, depth.astype(np.float32), view.intrinsics, None)
    h[vv, uu] = transform_points(floor_from_cam, backproject(v, m))[:, 2]
    return h


def build_background(rgb_frames: list[np.ndarray], plate_depth: np.ndarray, any_view: RGBDView,
                     floor_from_cam: np.ndarray, p: RGBMaskParams | None = None) -> BackgroundRGB:
    p = p or RGBMaskParams()
    plate = np.median(np.stack([cv2.resize(f, p.work_size, interpolation=cv2.INTER_AREA) for f in rgb_frames]), 0)
    plate = plate.astype(np.uint8)
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
    h = _heights(plate_depth, any_view, floor_from_cam)
    up = lambda a: cv2.resize(a.astype(np.uint8), p.work_size, interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    floor = cv2.erode(up(np.abs(h) < p.floor_band_m).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    q = (lab[floor].astype(int) * p.floor_hist_bins) // 256
    hist = np.zeros((p.floor_hist_bins,) * 3)
    np.add.at(hist, (q[:, 0], q[:, 1], q[:, 2]), 1)
    hist /= max(hist.sum(), 1)
    srt = np.sort(hist.ravel())[::-1]
    thr = float(srt[min(np.searchsorted(np.cumsum(srt), p.floor_hist_mass), srt.size - 1)])
    wall = up(np.isfinite(h) & (h > p.wall_min_height_m))
    info = {"plate_frames": len(rgb_frames), "floor_samples": int(floor.sum()), "wall_pixels": int(wall.sum()),
            "floor_colour_threshold": thr, "params": asdict(p)}
    return BackgroundRGB(lab.astype(np.float32), hist, thr, wall, info)


def refine_mask(mask: np.ndarray, view: RGBDView, rgb: np.ndarray, floor_from_cam: np.ndarray, bg: BackgroundRGB,
                p: RGBMaskParams | None = None) -> tuple[np.ndarray, dict]:
    """Return (reduced mask, stats). `rgb` is the frame covering the depth image's field of view."""
    p = p or RGBMaskParams()
    dh, dw = mask.shape
    img = cv2.resize(rgb, p.work_size, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h = _heights(view.depth_m, view, floor_from_cam)
    down = lambda a: cv2.resize(a.astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA)  # noqa: E731

    q = (lab.astype(int) * p.floor_hist_bins) // 256
    floorlike = bg.floor_hist[q[..., 0], q[..., 1], q[..., 2]] >= bg.floor_thr
    floor_veto = mask & (h < p.floor_veto_height_m) & (down(floorlike) > p.floor_veto_fraction)

    labf = lab.astype(np.float32)
    near = cv2.resize(cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)), p.work_size,
                      interpolation=cv2.INTER_NEAREST) > 0
    away = bg.wall_px & ~near
    wall_veto = np.zeros_like(mask)
    stats = {"pixels_in": int(mask.sum()), "floor_veto": int(floor_veto.sum())}
    if away.sum() > 1000:
        gain = float(np.median(labf[..., 0][away]) / max(np.median(bg.plate_lab[..., 0][away]), 1.0))
        diff = np.sqrt((labf[..., 0] - gain * bg.plate_lab[..., 0]) ** 2
                       + (2 * np.linalg.norm(labf[..., 1:] - bg.plate_lab[..., 1:], axis=2)) ** 2)
        thr = float(np.percentile(diff[away], p.wall_match_percentile))
        walllike = (diff < thr) & bg.wall_px
        wall_veto = mask & ~floor_veto & (h >= p.wall_min_height_m) & (down(walllike) > p.wall_veto_fraction)
        stats.update(exposure_gain=gain, wall_threshold=thr)
    out = mask & ~floor_veto & ~wall_veto
    stats.update(wall_veto=int(wall_veto.sum()), pixels_out=int(out.sum()))
    return out, stats


def outline_overlay(rgb: np.ndarray, before: np.ndarray, after: np.ndarray, path, size=(960, 720)) -> str:
    """Red = depth-only mask outline, green = RGB-sharpened outline (for the report)."""
    o = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    for m, col in ((before, (0, 0, 255)), (after, (0, 220, 0))):
        up = cv2.resize(m.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
        o[cv2.dilate(cv2.Canny(up, 50, 100), np.ones((2, 2), np.uint8)) > 0] = col
    if size[0] > size[1]:
        o = cv2.rotate(o, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(path), o)
    return str(path)
