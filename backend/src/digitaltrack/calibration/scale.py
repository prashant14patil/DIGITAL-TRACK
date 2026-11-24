"""Scale from measured vs known reference lengths (e.g. marker edges, a calibrated rod, a reference object).

Library only: schema 0.2.0 has no field for a scale estimate, so results are returned, not persisted.
The warn/fail tolerances below are EXPERIMENTAL defaults, not validated requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class ScaleError(ValueError):
    pass


@dataclass(frozen=True)
class ScaleEstimate:
    scale: float  # multiply measured lengths by this to get metres (least squares through the origin)
    n: int
    residuals_m: tuple[float, ...]  # scale*measured - reference
    rms_residual_m: float
    max_relative_residual: float  # max |residual| / reference

    def status(self, warn_rel: float = 0.005, fail_rel: float = 0.01) -> str:
        """pass | warn | fail by the largest relative residual. Experimental thresholds."""
        r = self.max_relative_residual
        return "fail" if r > fail_rel else "warn" if r > warn_rel else "pass"


def scale_from_known_lengths(measured: Sequence[float], reference_m: Sequence[float]) -> ScaleEstimate:
    m = np.asarray(measured, dtype=float)
    r = np.asarray(reference_m, dtype=float)
    if m.shape != r.shape or m.ndim != 1:
        raise ScaleError("measured and reference must be 1-D sequences of equal length")
    if m.size < 1:
        raise ScaleError("at least one length pair is required")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(r))):
        raise ScaleError("lengths must be finite")
    if np.any(m <= 0) or np.any(r <= 0):
        raise ScaleError("lengths must be positive")
    s = float(np.dot(m, r) / np.dot(m, m))
    res = s * m - r
    return ScaleEstimate(scale=s, n=int(m.size), residuals_m=tuple(float(x) for x in res),
                         rms_residual_m=float(np.sqrt(np.mean(res ** 2))),
                         max_relative_residual=float(np.max(np.abs(res) / r)))
