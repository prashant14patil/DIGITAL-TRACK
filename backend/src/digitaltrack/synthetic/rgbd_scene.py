"""Synthetic RGB-D scenes of a rigid object that is turned AND shifted by hand on a floor, seen by a fixed camera.

Purpose: test the self-rotation pipeline against exactly known poses. Everything is SYNTHETIC; results say
nothing about real iPad depth. Depth is ray-cast with Open3D's CPU RaycastingScene.

World = floor frame: z up, floor at z = 0. A static back wall stands behind the object. The object (an
asymmetric box with a side block, so no 90/180 degree symmetry) rests on the floor. At stop k its pose is
floor_from_body_k = Translate(x_k, y_k, 0) @ Rz(theta_k); the body frame is the object frame at stop 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from ..dataset.loader import Intrinsics
from ..geometry.transforms import invert, rot_x, rot_z
from ..rgbd.views import RGBDView


@dataclass(frozen=True)
class SceneConfig:
    n_stops: int = 24
    step_deg: float = 15.0
    step_jitter_deg: float = 2.0  # actual step = step_deg + U(-j, j)
    shift_sigma_m: float = 0.02  # random walk of the object position per stop (person shuffling)
    width: int = 256
    height: int = 192
    fx: float = 212.0
    fy: float = 212.0
    camera_distance_m: float = 1.4  # horizontal distance camera -> object start position
    camera_height_m: float = 0.9
    camera_pitch_deg: float = 25.0  # tilted down, so the floor around the object is visible
    noise_sigma_m: float = 0.0  # Gaussian depth noise
    dropout: float = 0.0  # fraction of valid pixels returned as 0
    wall_distance_m: float = 1.2  # wall behind the object start position
    box_size_m: tuple[float, float, float] = (0.40, 0.30, 0.50)  # x, y, z
    block_size_m: tuple[float, float, float] = (0.12, 0.10, 0.20)  # attached to the +x face, off-centre
    object_kind: str = "box"  # "box" (asymmetric box + side block) or "cylinder" (rotationally symmetric)
    cylinder_radius_m: float = 0.15
    cylinder_height_m: float = 0.5
    frames_per_stop: int = 1  # >1 adds repeated (noisy) frames per stop
    transition_frames: int = 0  # frames rendered while the object moves between stops
    seed: int = 0


@dataclass
class SyntheticSequence:
    views: list[RGBDView]
    background_views: list[RGBDView]
    floor_from_body: list[np.ndarray]  # truth per stop (4x4)
    stop_of_view: list[int | None]  # stop index per view (None during transitions)
    camera_from_floor: np.ndarray
    config: SceneConfig
    truth: dict = field(default_factory=dict)


def _box(size, center) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(np.asarray(center, float) - np.asarray(size, float) / 2)
    return m


def object_mesh(cfg: SceneConfig) -> o3d.geometry.TriangleMesh:
    if cfg.object_kind == "cylinder":
        return o3d.geometry.TriangleMesh.create_cylinder(cfg.cylinder_radius_m, cfg.cylinder_height_m, resolution=120,
                                                         split=8).translate((0, 0, cfg.cylinder_height_m / 2))
    bx, by, bz = cfg.box_size_m
    kx, ky, kz = cfg.block_size_m
    m = _box(cfg.box_size_m, (0, 0, bz / 2))
    m += _box(cfg.block_size_m, (bx / 2 + kx / 2, by / 4, kz / 2))  # off-centre in y: no mirror symmetry
    return m


def camera_from_floor(cfg: SceneConfig) -> np.ndarray:
    """Camera looks along +y_floor toward the object, pitched down. OpenCV camera axes."""
    # floor axes in camera coords before pitch: x_cam = x_floor, y_cam = -z_floor, z_cam = y_floor
    base = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    pitch = rot_x(cfg.camera_pitch_deg)  # positive pitch tilts the optical axis toward -z_floor (down)
    floor_from_cam = np.eye(4)
    floor_from_cam[:3, :3] = (pitch[:3, :3] @ base[:3, :3]).T
    floor_from_cam[:3, 3] = [0.0, -cfg.camera_distance_m, cfg.camera_height_m]
    return invert(floor_from_cam)


def _intrinsics(cfg: SceneConfig) -> Intrinsics:
    return Intrinsics(cfg.fx, cfg.fy, (cfg.width - 1) / 2, (cfg.height - 1) / 2, cfg.width, cfg.height)


def _render(meshes: list[o3d.geometry.TriangleMesh], cam_from_floor: np.ndarray, cfg: SceneConfig,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    scene = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    k = _intrinsics(cfg)
    u, v = np.meshgrid(np.arange(cfg.width), np.arange(cfg.height))
    dirs_cam = np.stack([(u - k.cx) / k.fx, (v - k.cy) / k.fy, np.ones_like(u, float)], axis=-1).reshape(-1, 3)
    floor_from_cam = invert(cam_from_floor)
    dirs = dirs_cam @ floor_from_cam[:3, :3].T  # unnormalised: hit = origin + t * dir, and t == camera z
    origin = np.broadcast_to(floor_from_cam[:3, 3], dirs.shape)
    rays = o3d.core.Tensor(np.concatenate([origin, dirs], axis=1).astype(np.float32))
    t = scene.cast_rays(rays)["t_hit"].numpy().reshape(cfg.height, cfg.width).astype(np.float64)
    depth = np.where(np.isfinite(t), t, 0.0)
    conf = np.where(depth > 0, 2, 0).astype(np.uint8)
    if cfg.noise_sigma_m:
        depth = np.where(depth > 0, depth + rng.normal(0, cfg.noise_sigma_m, depth.shape), 0.0)
    if cfg.dropout:
        drop = (rng.random(depth.shape) < cfg.dropout) & (depth > 0)
        depth[drop] = 0.0
        conf[drop] = 0
    return depth.astype(np.float32), conf


def _static_meshes(cfg: SceneConfig) -> list[o3d.geometry.TriangleMesh]:
    floor = _box((6.0, 6.0, 0.01), (0, 0, -0.005))
    wall = _box((6.0, 0.05, 3.0), (0, cfg.wall_distance_m + 0.025, 1.5))
    return [floor, wall]


def make_sequence(cfg: SceneConfig | None = None) -> SyntheticSequence:
    cfg = cfg or SceneConfig()
    rng = np.random.default_rng(cfg.seed)
    cam = camera_from_floor(cfg)
    k = _intrinsics(cfg)
    obj = object_mesh(cfg)
    static = _static_meshes(cfg)

    thetas, xys = [0.0], [np.zeros(2)]
    for _ in range(cfg.n_stops - 1):
        thetas.append(thetas[-1] + cfg.step_deg + rng.uniform(-cfg.step_jitter_deg, cfg.step_jitter_deg))
        xys.append(xys[-1] + rng.normal(0, cfg.shift_sigma_m, 2))
    poses = []
    for th, xy in zip(thetas, xys):
        p = rot_z(th)
        p[:2, 3] = xy
        poses.append(p)

    views, stop_of_view = [], []
    t_now = 0.0

    def add(pose: np.ndarray, stop: int | None) -> None:
        nonlocal t_now
        m = o3d.geometry.TriangleMesh(obj)
        m.transform(pose)
        depth, conf = _render(static + [m], cam, cfg, rng)
        views.append(RGBDView(len(views), depth, k, conf, timestamp_s=round(t_now, 6),
                              source={"synthetic": True, "stop": stop}))
        stop_of_view.append(stop)
        t_now += 1 / 30

    for s, pose in enumerate(poses):
        if s > 0:
            for j in range(1, cfg.transition_frames + 1):  # object moving: interpolate pose
                a = j / (cfg.transition_frames + 1)
                th = thetas[s - 1] + a * (thetas[s] - thetas[s - 1])
                p = rot_z(th)
                p[:2, 3] = (1 - a) * xys[s - 1] + a * xys[s]
                add(p, None)
        for _ in range(cfg.frames_per_stop):
            add(pose, s)

    bg = []
    for i in range(3):
        depth, conf = _render(static, cam, cfg, rng)
        bg.append(RGBDView(i, depth, k, conf, source={"synthetic": True, "background": True}))

    truth = {"note": "SYNTHETIC ground truth; never used by the pipeline",
             "theta_deg": thetas, "xy_m": [x.tolist() for x in xys],
             "box_size_m": list(cfg.box_size_m), "block_size_m": list(cfg.block_size_m)}
    return SyntheticSequence(views, bg, poses, stop_of_view, cam, cfg, truth)
