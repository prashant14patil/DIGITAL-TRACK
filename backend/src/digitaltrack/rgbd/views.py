"""In-memory RGB-D views: the contract between export adapters and the pipeline.

A view holds depth in METRES (float32; 0 or non-finite = no measurement) together with the intrinsics of the
DEPTH image (never RGB intrinsics unless an adapter has verified they apply to the depth image). Adapters must
convert units explicitly and record what they did in `source`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..dataset.loader import Intrinsics


class ViewError(ValueError):
    pass


@dataclass
class RGBDView:
    index: int  # position in the recorded sequence (0-based)
    depth_m: np.ndarray  # (H, W) float32 metres
    intrinsics: Intrinsics  # for the depth image (width/height must equal depth shape)
    confidence: np.ndarray | None = None  # (H, W) uint8, higher = more confident; None if not provided
    timestamp_s: float | None = None
    rgb: np.ndarray | None = None  # optional; NOT assumed pixel-aligned with depth
    source: dict[str, Any] = field(default_factory=dict)  # provenance: files, conversions, adapter name
    derived: bool = False  # True if computed from several raw frames (e.g. temporal median over a stop)

    def validate(self) -> None:
        d = self.depth_m
        if d.ndim != 2 or d.dtype != np.float32:
            raise ViewError(f"view {self.index}: depth must be 2-D float32 metres, got {d.shape} {d.dtype}")
        k = self.intrinsics
        if d.shape != (k.height, k.width):
            raise ViewError(f"view {self.index}: depth {d.shape} != intrinsics {(k.height, k.width)}")
        if not (k.fx > 0 and k.fy > 0):
            raise ViewError(f"view {self.index}: focal lengths must be positive")
        if self.confidence is not None and self.confidence.shape != d.shape:
            raise ViewError(f"view {self.index}: confidence {self.confidence.shape} != depth {d.shape}")
        finite = d[np.isfinite(d)]
        if finite.size and finite.min() < 0:
            raise ViewError(f"view {self.index}: negative depth")
        if finite.size and np.nanmax(finite) > 100:
            raise ViewError(f"view {self.index}: depth up to {np.nanmax(finite):.1f} m; units are probably not metres")


def valid_mask(view: RGBDView, *, min_confidence: int | None, depth_min_m: float, depth_max_m: float) -> np.ndarray:
    d = view.depth_m
    m = np.isfinite(d) & (d > depth_min_m) & (d < depth_max_m)
    if min_confidence is not None and view.confidence is not None:
        m &= view.confidence >= min_confidence
    return m


def backproject(view: RGBDView, mask: np.ndarray) -> np.ndarray:
    """Camera-frame points (OpenCV: x right, y down, z forward) of the masked depth pixels."""
    k = view.intrinsics
    v, u = np.nonzero(mask)
    z = view.depth_m[v, u].astype(np.float64)
    return np.stack([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z], axis=1)
