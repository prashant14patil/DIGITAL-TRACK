"""Analytic synthetic bodies with exactly known dimensions.

Every body is a stack of elliptical cross-sections along +Z (body/turntable frame, Z up,
axis of rotation through the origin):

    inside(x, y, z)  <=>  z_min <= z <= z_max  and  (x/a(z))^2 + (y/b(z))^2 <= 1

so the true circumference at any height, the true height and the true volume are analytic.
a(z), b(z) are stored as dense tables (np.interp) so the renderer and the ground truth use
the identical function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ellipe


def ellipse_perimeter(a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """Exact ellipse perimeter 4*A*E(m), with A=max(a,b), m=1-(B/A)^2."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    big, small = np.maximum(a, b), np.minimum(a, b)
    return 4.0 * big * ellipe(1.0 - (small / big) ** 2)


@dataclass(frozen=True)
class ProfileBody:
    name: str
    z_min: float
    z_max: float
    z_grid: np.ndarray
    a_grid: np.ndarray  # semi-axis along x (body frame)
    b_grid: np.ndarray  # semi-axis along y (body frame)
    center_xy: tuple[float, float] = (0.0, 0.0)  # offset from the rotation axis (makes sign/axis errors visible)

    def semi_axes(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.interp(z, self.z_grid, self.a_grid), np.interp(z, self.z_grid, self.b_grid)

    def inside(self, p: np.ndarray) -> np.ndarray:
        """p: (N,3) body-frame points -> bool mask."""
        a, b = self.semi_axes(p[:, 2])
        in_z = (p[:, 2] >= self.z_min) & (p[:, 2] <= self.z_max)
        return in_z & (((p[:, 0] - self.center_xy[0]) / a) ** 2 + ((p[:, 1] - self.center_xy[1]) / b) ** 2 <= 1.0)

    def with_center(self, cx: float, cy: float) -> "ProfileBody":
        return ProfileBody(self.name, self.z_min, self.z_max, self.z_grid, self.a_grid, self.b_grid, (cx, cy))

    @property
    def max_radius(self) -> float:
        return float(max(self.a_grid.max(), self.b_grid.max()))

    @property
    def bounding_radius(self) -> float:
        """Radius of a circle about the rotation axis that contains the whole body."""
        return self.max_radius + float(np.hypot(*self.center_xy))

    @property
    def height(self) -> float:
        return self.z_max - self.z_min

    def circumference(self, z: float | np.ndarray) -> np.ndarray:
        a, b = self.semi_axes(np.asarray(z, dtype=float))
        return ellipse_perimeter(a, b)

    def volume(self) -> float:
        return float(np.trapezoid(np.pi * self.a_grid * self.b_grid, self.z_grid))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "center_xy": list(self.center_xy), "z_min": self.z_min, "z_max": self.z_max,
            "height_m": self.height, "volume_m3": self.volume(),
            "z_grid": self.z_grid.tolist(), "a_grid": self.a_grid.tolist(), "b_grid": self.b_grid.tolist(),
        }


def _from_profile(name: str, z_ctrl, a_ctrl, b_ctrl, n: int = 2001) -> ProfileBody:
    z_ctrl = np.asarray(z_ctrl, dtype=float)
    z = np.linspace(z_ctrl[0], z_ctrl[-1], n)
    a = PchipInterpolator(z_ctrl, a_ctrl)(z)
    b = PchipInterpolator(z_ctrl, b_ctrl)(z)
    return ProfileBody(name, float(z[0]), float(z[-1]), z, a, b)


def cylinder(radius: float = 0.15, height: float = 0.6) -> ProfileBody:
    return elliptic_cylinder(radius, radius, height, name=f"cylinder_r{radius}_h{height}")


def elliptic_cylinder(a: float = 0.18, b: float = 0.11, height: float = 0.6, name: str | None = None) -> ProfileBody:
    """a = semi-axis along x, b = along y (body frame)."""
    return _from_profile(name or f"ellipse_a{a}_b{b}_h{height}", [0.0, height], [a, a], [b, b])


def procedural_torso() -> ProfileBody:
    """Smooth hip -> waist -> chest -> shoulder profile (elliptical sections), 0.60 m tall.

    NOT anatomically validated; only a smooth, concave-and-convex test shape with an
    analytic circumference at every height (waist narrower than hips and chest).
    """
    z = [0.00, 0.10, 0.25, 0.40, 0.52, 0.60]
    a = [0.170, 0.175, 0.140, 0.160, 0.190, 0.170]
    b = [0.110, 0.115, 0.100, 0.110, 0.100, 0.090]
    return _from_profile("procedural_torso", z, a, b)


SHAPES = {
    "cylinder": cylinder,
    "ellipse": elliptic_cylinder,
    "torso": procedural_torso,
}
