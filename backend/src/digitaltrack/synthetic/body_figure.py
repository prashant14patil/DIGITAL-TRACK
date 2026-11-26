"""SYNTHETIC standing figure with KNOWN dimensions, for testing body measurement and the validation statistics.

Built from closed primitives (no booleans): two legs, a torso of three stacked elliptic cylinders (hip, waist,
chest), hanging arms separated from the torso, shoulders, neck, head. Body frame: floor z = 0, +z up, lateral = y.
`scan_copy` imitates a repeat scan: surface noise along the normals, a global scale error, a random turn and a
different arm gap. It checks the MEASUREMENT ENGINE and the statistics only, never the capture or reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d

from .shapes import ellipse_perimeter


@dataclass(frozen=True)
class Figure:
    hip: tuple[float, float] = (0.17, 0.12)  # (lateral, depth) semi-axes, z 0.805-0.95
    waist: tuple[float, float] = (0.145, 0.10)  # z 0.95-1.15
    chest: tuple[float, float] = (0.165, 0.115)  # z 1.15-1.35
    leg_r: float = 0.07
    leg_y: float = 0.10
    arm_r: float = 0.04
    arm_gap: float = 0.10  # gap between torso (chest width) and arm surface
    head_top: float = 1.74

    def truth_mm(self) -> dict:
        p = lambda a: float(ellipse_perimeter(*a)) * 1000  # noqa: E731
        return {"chest": p(self.chest), "waist": p(self.waist), "hip": p(self.hip),
                "thigh": 2 * np.pi * self.leg_r * 1000, "height": self.head_top * 1000}

    def levels_mm(self) -> dict:  # tape levels inside each torso section
        return {"chest": 1250.0, "waist": 1050.0, "hip": 880.0}


def _cyl(r, z0, z1, x=0.0, y=0.0, sx=1.0, sy=1.0, res=180):
    m = o3d.geometry.TriangleMesh.create_cylinder(1.0, z1 - z0, res, 8)
    m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) * [r * sx, r * sy, 1.0] + [x, y, (z0 + z1) / 2])
    return m


def _box(x, y, z0, z1):
    m = o3d.geometry.TriangleMesh.create_box(2 * x, 2 * y, z1 - z0)
    m.translate((-x, -y, z0))
    return m


def build(f: Figure) -> o3d.geometry.TriangleMesh:
    arm_y = f.chest[0] + f.arm_gap + f.arm_r
    head_r = 0.10
    head = o3d.geometry.TriangleMesh.create_sphere(head_r, 60)
    head.translate((0, 0, f.head_top - head_r))
    parts = [
        _cyl(f.leg_r, 0.02, 0.80, y=-f.leg_y), _cyl(f.leg_r, 0.02, 0.80, y=f.leg_y),
        _cyl(1.0, 0.805, 0.95, sx=f.hip[1], sy=f.hip[0]),
        _cyl(1.0, 0.95, 1.15, sx=f.waist[1], sy=f.waist[0]),
        _cyl(1.0, 1.15, 1.35, sx=f.chest[1], sy=f.chest[0]),
        _cyl(f.arm_r, 0.85, 1.35, y=-arm_y), _cyl(f.arm_r, 0.85, 1.35, y=arm_y),
        _box(0.10, arm_y + f.arm_r, 1.35, 1.45), _cyl(0.055, 1.45, f.head_top - 2 * head_r + 0.005), head,
    ]
    m = parts[0]
    for q in parts[1:]:
        m += q
    m.compute_vertex_normals()
    return m


def scan_copy(f: Figure, rng: np.random.Generator, *, noise_m: float = 0.002, scale_sd: float = 0.003,
              arm_gap_sd: float = 0.02) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """One simulated repeat scan of figure `f` (arm gap varies; scale error and noise applied to the surface)."""
    g = Figure(**{**asdict(f), "arm_gap": float(max(0.05, f.arm_gap + rng.normal(0, arm_gap_sd)))})
    m = build(g).subdivide_midpoint(1)
    m.compute_vertex_normals()
    v = np.asarray(m.vertices)
    n = np.asarray(m.vertex_normals)
    s = 1.0 + rng.normal(0, scale_sd)
    v = (v + n * rng.normal(0, noise_m, (len(v), 1))) * s
    yaw = rng.uniform(0, 2 * np.pi)
    c, si = np.cos(yaw), np.sin(yaw)
    v[:, :2] = v[:, :2] @ np.array([[c, si], [-si, c]])
    m.vertices = o3d.utility.Vector3dVector(v)
    return m, {"scale": s, "yaw_deg": float(np.degrees(yaw)), "arm_gap_m": g.arm_gap, "noise_m": noise_m}
