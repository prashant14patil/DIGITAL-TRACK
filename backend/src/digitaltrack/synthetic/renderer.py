"""Virtual turntable: ray-casts a synthetic body from a fixed camera while it rotates.

Frames and sign convention (identical to geometry/transforms.py and docs/dataset_spec.md):
  * Camera frame: OpenCV (x right, y down, z forward). Depth = z-coordinate (metres).
  * Turntable frame: origin on the rotation centre, +Z up. Camera sits at (0, -D, h) looking
    along +Y (optionally pitched down).
  * At true angle A the body is rotated by +A about +Z:   p_cam = T_cam_from_table @ Rz(A) @ p_body

Impairments (all configurable, all seeded): depth noise, missing depth, recorded-angle bias
and noise, and a *calibration* error (what the backend is told about the rig differs from
the truth). The truth lives in ground_truth.json, separate from raw data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import DATASET_SCHEMA_VERSION
from ..dataset.writer import FrameData, utc_now, write_session
from ..geometry.transforms import rot_x, rot_z
from .shapes import ProfileBody


@dataclass
class RenderConfig:
    n_views: int = 36
    total_rotation_deg: float = 360.0
    image_width: int = 192
    image_height: int = 256
    focal_px: float = 230.0
    camera_distance_m: float = 1.2
    camera_height_m: float = 0.3
    camera_pitch_deg: float = 0.0  # camera tilted down by this angle
    noise_sigma_m: float = 0.0  # depth noise, constant part
    noise_sigma_rel: float = 0.0  # depth noise proportional to depth
    dropout_prob: float = 0.0  # fraction of valid pixels randomly returned as 0 (no data)
    angle_bias_deg: float = 0.0  # recorded = true + bias + noise
    angle_noise_deg: float = 0.0
    calibration_axis_tilt_deg: float = 0.0  # error in the calibration reported to the backend
    # direction of that tilt: 0 = about turntable X (axis leans toward/away from the camera),
    # 90 = about turntable Y (axis leans sideways in the image)
    calibration_axis_tilt_azimuth_deg: float = 0.0
    calibration_center_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    march_step_m: float = 0.003
    seed: int = 0


def _intrinsics(cfg: RenderConfig) -> dict:
    return {"fx": cfg.focal_px, "fy": cfg.focal_px,
            "cx": (cfg.image_width - 1) / 2.0, "cy": (cfg.image_height - 1) / 2.0}


def camera_from_turntable_true(cfg: RenderConfig) -> np.ndarray:
    """True 4x4 transform: turntable-frame point -> camera-frame point (angle 0)."""
    c = np.array([0.0, -cfg.camera_distance_m, cfg.camera_height_m])
    # columns = camera axes expressed in table frame: x_cam=+X, y_cam(down)=-Z, z_cam(forward)=+Y
    r_table_from_cam = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)
    r_table_from_cam = r_table_from_cam @ rot_x(-cfg.camera_pitch_deg)[:3, :3]
    t = np.eye(4)
    t[:3, :3] = r_table_from_cam.T
    t[:3, 3] = -r_table_from_cam.T @ c
    return t


def _raycast(body: ProfileBody, o_b: np.ndarray, d_b: np.ndarray, step: float,
             n_bisect: int = 18) -> np.ndarray:
    """First hit parameter t per ray (NaN = miss). Rays: o_b + t*d_b, d_b unnormalised.

    Each ray is marched only through the body's bounding circle (in XY), then the first
    inside sample is refined by bisection.
    """
    n = d_b.shape[0]
    hit_t = np.full(n, np.nan)
    rmax = body.bounding_radius + 1e-6
    dxy, oxy = d_b[:, :2], o_b[:2]
    a = (dxy**2).sum(1)
    b = 2.0 * dxy @ oxy
    c = float(oxy @ oxy) - rmax**2
    disc = b * b - 4.0 * a * c
    active = np.nonzero((disc > 0) & (a > 0))[0]
    sq = np.sqrt(disc[active])
    t_cur = (-b[active] - sq) / (2.0 * a[active])
    t_end = (-b[active] + sq) / (2.0 * a[active])
    while active.size:
        ins = body.inside(o_b + t_cur[:, None] * d_b[active])
        if ins.any():
            idx = active[ins]
            d = d_b[idx]
            hi = t_cur[ins].copy()
            lo = hi - step
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                m_in = body.inside(o_b + mid[:, None] * d)
                hi = np.where(m_in, mid, hi)
                lo = np.where(m_in, lo, mid)
            hit_t[idx] = hi
        t_cur = t_cur + step
        keep = ~ins & (t_cur - step < t_end)
        active, t_cur, t_end = active[keep], t_cur[keep], t_end[keep]
    return hit_t


def render_frame(body: ProfileBody, cfg: RenderConfig, true_angle_deg: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth float32 HxW metres, confidence uint8 HxW)."""
    w, h = cfg.image_width, cfg.image_height
    k = _intrinsics(cfg)
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    d_cam = np.stack([(u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], np.ones_like(u)], -1).reshape(-1, 3)

    t_true = camera_from_turntable_true(cfg)
    r_tfc = t_true[:3, :3].T
    cam_pos = -(t_true[:3, :3].T @ t_true[:3, 3])  # camera position in the turntable frame
    rz_inv = rot_z(-true_angle_deg)[:3, :3]
    o_b = rz_inv @ cam_pos
    d_b = (rz_inv @ (r_tfc @ d_cam.T)).T

    hit_t = _raycast(body, o_b, d_b, cfg.march_step_m)

    hit = np.isfinite(hit_t)
    depth = np.where(hit, hit_t, 0.0)  # z_cam == t because d_cam has z-component 1

    # incidence angle from the analytic normal (xy part only) -> synthetic confidence
    conf = np.zeros(depth.shape, dtype=np.uint8)
    if hit.any():
        p = o_b + hit_t[hit, None] * d_b[hit]
        a, b = body.semi_axes(p[:, 2])
        n = np.stack([(p[:, 0] - body.center_xy[0]) / a**2, (p[:, 1] - body.center_xy[1]) / b**2,
                      np.zeros(p.shape[0])], -1)
        nn = np.linalg.norm(n, axis=1)
        dn = d_b[hit] / np.linalg.norm(d_b[hit], axis=1, keepdims=True)
        cosang = np.abs(np.einsum("ij,ij->i", n, dn)) / np.maximum(nn, 1e-12)
        conf[hit] = np.where(cosang >= 0.35, 2, 1).astype(np.uint8)

    if cfg.noise_sigma_m > 0 or cfg.noise_sigma_rel > 0:
        sigma = cfg.noise_sigma_m + cfg.noise_sigma_rel * depth
        depth = np.where(hit, depth + rng.normal(size=depth.shape) * sigma, 0.0)
    if cfg.dropout_prob > 0:
        drop = rng.random(depth.shape) < cfg.dropout_prob
        depth = np.where(drop, 0.0, depth)
        conf = np.where(drop, 0, conf).astype(np.uint8)
    depth = np.clip(depth, 0.0, None)
    return depth.reshape(h, w).astype(np.float32), conf.reshape(h, w)


def _recorded_turntable_calibration(cfg: RenderConfig, calibration_id: str) -> dict:
    t = camera_from_turntable_true(cfg)
    injected = bool(cfg.calibration_axis_tilt_deg or any(cfg.calibration_center_offset_m))
    if injected:
        az = cfg.calibration_axis_tilt_azimuth_deg
        p = rot_z(az) @ rot_x(cfg.calibration_axis_tilt_deg) @ rot_z(-az)
        p[:3, 3] = np.asarray(cfg.calibration_center_offset_m, dtype=float)
        t = t @ p  # error expressed in the turntable frame
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "calibration_id": calibration_id,
        "method": "synthetic_truth_with_injected_error" if injected else "synthetic_truth",
        "turntable_frame": "origin=rotation centre on platform, +Z=rotation axis (up); subject rotated by sign*angle about +Z",
        "camera_from_turntable": t.tolist(),
        "angle_direction_sign": 1,
        "uncertainty": {"axis_tilt_deg": abs(cfg.calibration_axis_tilt_deg),
                        "center_m": float(np.linalg.norm(cfg.calibration_center_offset_m)),
                        "angle_deg": abs(cfg.angle_bias_deg) + cfg.angle_noise_deg},
    }


def render_session(body: ProfileBody, out_dir: str | Path, cfg: RenderConfig | None = None,
                   session_id: str | None = None, readonly: bool = True) -> Path:
    """Render a synthetic scan session to `out_dir` (must not already exist or must be empty)."""
    cfg = cfg or RenderConfig()
    rng = np.random.default_rng(cfg.seed)
    session_id = session_id or f"synthetic_{body.name}"
    calibration_id = f"{session_id}_cal"
    true_angles = np.arange(cfg.n_views) * (cfg.total_rotation_deg / cfg.n_views)
    recorded = true_angles + cfg.angle_bias_deg + rng.normal(0.0, cfg.angle_noise_deg, cfg.n_views) \
        if cfg.angle_noise_deg else true_angles + cfg.angle_bias_deg

    def frames() -> Iterator[FrameData]:
        for i, a_true in enumerate(true_angles):
            depth, conf = render_frame(body, cfg, float(a_true), rng)
            yield FrameData(depth, conf, timestamp_s=0.5 * i, turntable_angle_deg=float(recorded[i]),
                            angle_source="synthetic")

    session_meta = {
        "session_id": session_id,
        "created_utc": utc_now(),
        "synthetic": True,
        "input_type": "synthetic_depth",
        "device": {"model": "SYNTHETIC virtual turntable", "os_version": "n/a", "app_version": "digitaltrack.synthetic"},
        "capture": {"camera_fixed": True, "subject_on_turntable": True, "depth_source": "synthetic ray-cast",
                    "depth_unit": "meters", "depth_dtype": "float32", "rotation_source": "synthetic",
                    "rotation_axis_frame": "turntable"},
        "calibration_id": calibration_id,
        "rig": {"camera_distance_m": cfg.camera_distance_m, "camera_height_m": cfg.camera_height_m},
    }
    camera_cal = {
        "schema_version": DATASET_SCHEMA_VERSION, "calibration_id": calibration_id,
        "camera_frame": "opencv_x_right_y_down_z_forward", "pixel_convention": "index_center",
        "image_source": "depth",
        "image_width": cfg.image_width, "image_height": cfg.image_height,
        "intrinsics": _intrinsics(cfg), "method": "synthetic_truth",
    }
    ground_truth = {
        "note": "SYNTHETIC ground truth. Never used by reconstruction; only by tests and error reporting.",
        "body": body.to_dict(),
        "true_angles_deg": true_angles.tolist(),
        "render_config": asdict(cfg),
        "true_camera_from_turntable": camera_from_turntable_true(cfg).tolist(),
    }
    return write_session(
        out_dir, session_meta=session_meta, camera_calibration=camera_cal,
        turntable_calibration=_recorded_turntable_calibration(cfg, calibration_id),
        frames=frames(), extra_json={"ground_truth.json": ground_truth}, readonly=readonly,
    )
