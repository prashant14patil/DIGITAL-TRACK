"""Rigid-transform helpers and the single place where the turntable convention lives.

Convention (see docs/dataset_spec.md): the subject is fixed in the *body/turntable frame*
(origin = rotation centre, +Z up). At recorded angle A the subject is rotated by
sign*A about +Z, so a body point appears in the camera at

    p_cam = camera_from_turntable @ Rz(sign * A) @ p_body

Hence camera_from_body(A) = camera_from_turntable @ Rz(sign*A). This 4x4 is exactly the
"extrinsic" (world->camera) that TSDF integration needs, with world == body frame.
"""

from __future__ import annotations

import numpy as np


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    m = np.eye(4)
    m[1:3, 1:3] = [[c, -s], [s, c]]
    return m


def camera_from_body(camera_from_turntable: np.ndarray, angle_deg: float, sign: int = 1) -> np.ndarray:
    return camera_from_turntable @ rot_z(sign * angle_deg)


def transform_points(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ t[:3, :3].T + t[:3, 3]


def invert(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out
