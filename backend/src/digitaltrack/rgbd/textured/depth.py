"""Confidence-aware depth filtering. No spatial smoothing: a surviving pixel keeps a measured (or temporal-median)
depth, so body contours are not rounded.

1. confidence >= min_confidence (ARKit levels 0..2)
2. flying pixels: a value strictly between two opposite neighbours and far from BOTH is a foreground/background
   mixture, not a surface; the foreground and background pixels next to it are kept
3. over a pause: per-pixel median of the surviving samples, only where at least `min_valid_fraction` survive
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthFilterParams:
    min_confidence: int = 2
    depth_min_m: float = 0.2
    depth_max_m: float = 4.0
    flying_jump_m: float = 0.04  # proposed
    min_valid_fraction: float = 0.5
    median_max_frames: int = 30  # per pause


def flying_pixels(z: np.ndarray, jump_m: float) -> np.ndarray:
    """True where a pixel lies between its two opposite neighbours and differs from both by more than jump_m."""
    pad = np.pad(z, 1, mode="edge")
    h, w = z.shape
    out = np.zeros(z.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = pad[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        b = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        d1, d2 = z - a, b - z
        out |= (a > 0) & (b > 0) & (z > 0) & (d1 * d2 > 0) & (np.abs(d1) > jump_m) & (np.abs(d2) > jump_m)
    return out


def filter_depth(raw_mm: np.ndarray, confidence: np.ndarray, p: DepthFilterParams) -> tuple[np.ndarray, dict]:
    """-> (float32 metres, 0 = rejected; counts per rejection reason)."""
    z = raw_mm.astype(np.float32) * 0.001
    in_range = (z > p.depth_min_m) & (z < p.depth_max_m)
    low_conf = in_range & (confidence < p.min_confidence)
    flying = flying_pixels(z, p.flying_jump_m) & in_range & ~low_conf
    keep = in_range & ~low_conf & ~flying
    return np.where(keep, z, 0.0).astype(np.float32), {
        "pixels": int(z.size), "out_of_range": int((~in_range).sum()), "low_confidence": int(low_conf.sum()),
        "flying": int(flying.sum()), "kept": int(keep.sum())}


def temporal_median(depths: list[np.ndarray], p: DepthFilterParams) -> tuple[np.ndarray, dict]:
    """Median of filtered depth maps of one pause. DERIVED data; raw files are never touched."""
    stack = np.stack([np.where(d > 0, d, np.nan) for d in depths])
    valid = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - med), axis=0)
    keep = (valid > 0) & (valid >= p.min_valid_fraction * len(depths))
    out = np.where(keep, med, 0.0).astype(np.float32)
    return out, {"frames": len(depths), "kept": int(keep.sum()),
                 "p90_abs_deviation_mm": round(float(np.percentile(mad[keep], 90)) * 1000, 3) if keep.any() else None}
