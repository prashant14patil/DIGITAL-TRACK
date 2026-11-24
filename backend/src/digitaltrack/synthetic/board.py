"""Synthetic chessboard views with exactly known intrinsics and poses (for calibration tests).

Pinhole projection without lens distortion. Results validate the calibration code path against analytic
truth; they say nothing about real iPad optics.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..calibration.intrinsics import ChessboardSpec


def board_texture(spec: ChessboardSpec, px_per_square: int = 40, margin_squares: int = 1) -> np.ndarray:
    """White-margined chessboard image (uint8 gray) with (inner_cols+1) x (inner_rows+1) squares."""
    cols, rows = spec.inner_cols + 1, spec.inner_rows + 1
    m = margin_squares * px_per_square
    tex = np.full((rows * px_per_square + 2 * m, cols * px_per_square + 2 * m), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                tex[m + r * px_per_square:m + (r + 1) * px_per_square,
                    m + c * px_per_square:m + (c + 1) * px_per_square] = 0
    return tex


def render_view(spec: ChessboardSpec, camera_matrix: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                image_size: tuple[int, int], *, px_per_square: int = 40, noise_sigma: float = 0.0,
                seed: int = 0) -> np.ndarray:
    """RGB uint8 image of the board at pose (rvec, tvec): board point X (metres, z=0) -> camera R X + t."""
    tex = board_texture(spec, px_per_square)
    s = spec.square_size_m
    # texture pixel -> board metres; inner corner k lies at (k+1) squares from the board edge
    off = -(px_per_square + px_per_square) * s / px_per_square  # margin (1 square) + first square
    a = np.array([[s / px_per_square, 0, off], [0, s / px_per_square, off], [0, 0, 1]], float)
    r, _ = cv2.Rodrigues(np.asarray(rvec, float))
    hb = np.asarray(camera_matrix, float) @ np.column_stack([r[:, 0], r[:, 1], np.asarray(tvec, float).reshape(3)])
    h = hb @ a
    img = cv2.warpPerspective(tex, h, image_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=200)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(float) + rng.normal(0, noise_sigma, img.shape), 0, 255).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def random_poses(spec: ChessboardSpec, n: int, *, distance_m: tuple[float, float] = (0.5, 0.8),
                 max_tilt_deg: float = 35.0, lateral_m: float = 0.04, seed: int = 0):
    """n varied board poses (rvec, tvec) with the board centre roughly on the optical axis."""
    rng = np.random.default_rng(seed)
    centre = np.array([(spec.inner_cols - 1) * spec.square_size_m / 2, (spec.inner_rows - 1) * spec.square_size_m / 2, 0])
    out = []
    for _ in range(n):
        ang = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg, 3) * np.array([1, 1, 0.3]))
        r, _ = cv2.Rodrigues(ang)
        t = np.array([rng.uniform(-lateral_m, lateral_m), rng.uniform(-lateral_m, lateral_m),
                      rng.uniform(*distance_m)]) - r @ centre
        out.append((ang.reshape(3, 1), t.reshape(3, 1)))
    return out
