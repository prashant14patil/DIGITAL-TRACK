"""Per-pixel depth reliability for FUSION (never modifies depth; returns masks).

LiDAR depth on the iPad is 256x192 and is least reliable at silhouettes: pixels straddling body and background get
an in-between depth ("flying pixels"), and surfaces seen at grazing angles are smeared. With a 360 deg turn, every
surface point is also seen frontally in some other view, so these pixels can be dropped without losing coverage,
and dropping them removes the outward bias projective TSDF shows at silhouettes (see docs/architecture.md).

Filters (each logged as a pixel count):
  * confidence >= min_confidence (Stray: 0 low, 1 medium, 2 high)
  * silhouette erosion by `erode_px` (mixed pixels at the subject outline)
  * depth discontinuity: |depth - any valid 4-neighbour| > jump_m * depth  (flying pixels inside the mask)
  * grazing angle: angle between the depth-derived surface normal and the viewing ray > max_view_angle_deg
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .views import RGBDView


@dataclass(frozen=True)
class DepthFilterParams:
    min_confidence: int = 2
    erode_px: int = 1
    jump_rel: float = 0.01  # 1 % of depth (~2 cm at 2 m)
    max_view_angle_deg: float = 70.0


def fusion_mask(view: RGBDView, subject: np.ndarray, p: DepthFilterParams) -> tuple[np.ndarray, dict]:
    d = view.depth_m
    m = subject & np.isfinite(d) & (d > 0)
    stats = {"subject": int(m.sum())}
    if view.confidence is not None:
        m &= view.confidence >= p.min_confidence
    stats["after_confidence"] = int(m.sum())
    if p.erode_px > 0:
        k = np.ones((2 * p.erode_px + 1, 2 * p.erode_px + 1), np.uint8)
        m &= cv2.erode(subject.astype(np.uint8), k, borderValue=0).astype(bool)
    stats["after_erosion"] = int(m.sum())
    dd = np.where(subject, d, np.nan)
    pad = np.pad(dd, 1, constant_values=np.nan)
    jump = np.zeros_like(m)
    for dv, du in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nb = pad[dv:dv + d.shape[0], du:du + d.shape[1]]
        with np.errstate(invalid="ignore"):
            jump |= np.abs(nb - d) > p.jump_rel * d
    m &= ~jump
    stats["after_discontinuity"] = int(m.sum())
    # normals from depth (central differences of back-projected points)
    k = view.intrinsics
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    pts = np.stack([(u - k.cx) / k.fx * d, (v - k.cy) / k.fy * d, d], -1)
    gx = np.zeros_like(pts)
    gy = np.zeros_like(pts)
    gx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    gy[1:-1] = pts[2:] - pts[:-2]
    n = np.cross(gx, gy)
    nn = np.linalg.norm(n, axis=-1)
    ray = pts / np.maximum(np.linalg.norm(pts, axis=-1, keepdims=True), 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.abs(np.sum(n * ray, -1)) / nn
    m &= np.nan_to_num(cosang, nan=0.0) >= np.cos(np.radians(p.max_view_angle_deg))
    stats["after_view_angle"] = int(m.sum())
    return m, stats


def params_dict(p: DepthFilterParams) -> dict:
    return asdict(p)
