"""Stop detection for step-and-pause capture: find runs of frames where the scene does not move.

Motion score between consecutive frames = fraction of pixels (valid in both) whose depth changed by more than
`change_m`. The static background dominates any average, so an average or median would miss a moving subject;
the changed-pixel fraction does not. A stop is a run of at least `min_frames` consecutive frames whose scores are
all below `max_changed_fraction`. The keyframe is
the middle frame of the run. Optionally a temporal median over the run gives a less noisy DERIVED view; raw
frames are never modified.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from .views import RGBDView


@dataclass(frozen=True)
class StopParams:
    change_m: float = 0.02  # a pixel "moved" if its depth changed more than this (well above sensor noise)
    max_changed_fraction: float = 0.002  # a frame pair is still if fewer pixels than this fraction moved
    min_frames: int = 3
    depth_min_m: float = 0.2
    depth_max_m: float = 4.0
    # Real LiDAR flickers at silhouette edges far above the synthetic noise floor (first real iPad recording:
    # quietest frames ~0.7% changed, turning ~5-8%). Adaptive: threshold = P10 + adaptive_fraction*(P90 - P10),
    # clipped to [max_changed_fraction, adaptive_cap]. Proposed values from ONE real recording; unvalidated.
    adaptive: bool = False
    adaptive_fraction: float = 0.3
    adaptive_cap: float = 0.03


def stop_threshold(scores: np.ndarray, p: StopParams) -> float:
    if not p.adaptive or len(scores) < 10:
        return p.max_changed_fraction
    lo, hi = np.percentile(scores[1:], [10, 90])
    return float(np.clip(lo + p.adaptive_fraction * (hi - lo), p.max_changed_fraction, p.adaptive_cap))


@dataclass
class Stop:
    start: int  # index into the view list (inclusive)
    end: int  # inclusive
    keyframe: int
    median_changed_fraction: float

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


def motion_scores(views: list[RGBDView], p: StopParams) -> np.ndarray:
    """scores[i] = changed-pixel fraction between view i-1 and view i (scores[0] = 0)."""
    out = np.zeros(len(views))
    for i in range(1, len(views)):
        a, b = views[i - 1].depth_m, views[i].depth_m
        m = (a > p.depth_min_m) & (a < p.depth_max_m) & (b > p.depth_min_m) & (b < p.depth_max_m)
        m &= np.isfinite(a) & np.isfinite(b)
        out[i] = float(np.mean(np.abs(a[m] - b[m]) > p.change_m)) if m.any() else 1.0
    return out


def detect_stops(views: list[RGBDView], p: StopParams | None = None) -> tuple[list[Stop], np.ndarray]:
    p = p or StopParams()
    s = motion_scores(views, p)
    still = s < stop_threshold(s, p)
    # frame i belongs to a still run if it matches the previous frame (i) or the next frame (i+1)
    member = still.copy()
    member[:-1] |= still[1:]
    stops, i = [], 0
    while i < len(views):
        if not member[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(views) and still[j + 1]:
            j += 1
        if j - i + 1 >= p.min_frames:
            inner = s[i + 1:j + 1]
            stops.append(Stop(i, j, (i + j) // 2, float(np.median(inner)) if inner.size else 0.0))
        i = j + 1
    return stops, s


def median_view(views: list[RGBDView], stop: Stop, p: StopParams | None = None) -> RGBDView:
    """Temporal median depth over a stop (pixel valid if valid in at least half the frames). DERIVED data."""
    p = p or StopParams()
    frames = views[stop.start:stop.end + 1]
    stack = np.stack([np.where((f.depth_m > p.depth_min_m) & (f.depth_m < p.depth_max_m), f.depth_m, np.nan)
                      for f in frames])
    n = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
    depth = np.where(n * 2 >= len(frames), med, 0.0).astype(np.float32)
    key = views[stop.keyframe]
    conf = None
    if all(f.confidence is not None for f in frames):
        conf = np.min(np.stack([f.confidence for f in frames]), axis=0)  # conservative
    return RGBDView(key.index, depth, key.intrinsics, conf, key.timestamp_s, key.rgb,
                    {"derived_from": [f.index for f in frames], "method": "temporal_median", **key.source},
                    derived=True, up_camera=key.up_camera)


def params_dict(p: StopParams) -> dict:
    return asdict(p)
