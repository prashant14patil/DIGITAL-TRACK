"""SYNTHETIC Stray-Scanner-format recording of a TEXTURED object turning in steps, with exact truth.

Same file layout and conventions as the real export (see rgbd.textured.sensors): portrait camera by default,
odometry quaternion in a +Y-up world, accelerometer in device axes, depth in mm. The object is an elliptic
cylinder (smooth: its shape barely constrains the turn angle) with two unequal side bars; its surface carries a
random pattern fixed to the body, which turns with it. For tests only; says nothing about real iPad data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from ..rgbd.textured.sensors import IMU_TO_CAMERA


@dataclass(frozen=True)
class StraySceneConfig:
    n_stops: int = 12
    pause_frames: int = 10
    move_frames: int = 5
    fps: float = 8.0
    shift_sigma_m: float = 0.015
    depth_noise_m: float = 0.0015
    semi_x_m: float = 0.20
    semi_y_m: float = 0.13
    height_m: float = 1.5
    camera_distance_m: float = 2.0
    camera_height_m: float = 0.9
    portrait: bool = True
    rgb_size: tuple[int, int] = (1024, 768)
    depth_size: tuple[int, int] = (256, 192)
    fx_rgb: float = 850.0
    seed: int = 0


def body_mesh(cfg: StraySceneConfig) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh.create_cylinder(1.0, cfg.height_m, resolution=90, split=20)
    v = np.asarray(m.vertices) * [cfg.semi_x_m, cfg.semi_y_m, 1.0] + [0, 0, cfg.height_m / 2]
    m.vertices = o3d.utility.Vector3dVector(v)
    for size, corner in (((0.18, 0.06, 0.08), (cfg.semi_x_m - 0.02, -0.03, 1.05)),
                         ((0.09, 0.06, 0.30), (-cfg.semi_x_m - 0.07, -0.03, 0.80))):
        box = o3d.geometry.TriangleMesh.create_box(*size)
        box.translate(corner)
        m += box
    return m


def camera_pose(cfg: StraySceneConfig) -> np.ndarray:
    """floor_from_camera: level camera looking along +y_floor."""
    forward = np.array([0.0, 1.0, 0.0])
    x = np.array([0.0, 0.0, -1.0]) if cfg.portrait else np.array([1.0, 0.0, 0.0])  # portrait: image left is up
    t = np.eye(4)
    t[:3, :3] = np.stack([x, np.cross(forward, x), forward], axis=1)
    t[:3, 3] = [0.0, -cfg.camera_distance_m, cfg.camera_height_m]
    return t


def _texture(points: np.ndarray, noise: np.ndarray, cell_m: float) -> np.ndarray:
    return map_coordinates(noise, (points / cell_m).T, order=1, mode="wrap")


def make_stray_recording(folder: str | Path, cfg: StraySceneConfig | None = None) -> dict:
    """Write the recording. Returns truth: floor_from_body per stop and the key frame of each pause."""
    cfg = cfg or StraySceneConfig()
    folder = Path(folder)
    (folder / "depth").mkdir(parents=True)
    (folder / "confidence").mkdir()
    rng = np.random.default_rng(cfg.seed)
    noise = rng.uniform(20, 235, (64, 64, 64))
    floor_from_cam = camera_pose(cfg)
    step = 360.0 / cfg.n_stops
    stops = [(0.0, 0.0, 0.0)]
    for k in range(1, cfg.n_stops):
        dx, dy = rng.normal(0, cfg.shift_sigma_m, 2)
        stops.append((np.radians(k * step), stops[-1][1] + dx, stops[-1][2] + dy))

    frames, keys = [], []
    for k, pose in enumerate(stops):
        keys.append(len(frames) + cfg.pause_frames // 2)
        frames += [pose] * cfg.pause_frames
        if k + 1 < len(stops):
            nxt = stops[k + 1]
            frames += [tuple(a + (b - a) * (i + 1) / (cfg.move_frames + 1) for a, b in zip(pose, nxt))
                       for i in range(cfg.move_frames)]

    w, h = cfg.rgb_size
    dw, dh = cfg.depth_size
    fx = cfg.fx_rgb
    cx, cy = (w - 1) / 2, (h - 1) / 2

    def rays(width, height, scale):
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        d = np.stack([(u - cx * scale) / (fx * scale), (v - cy * scale) / (fx * scale), np.ones_like(u, float)], -1)
        return d @ floor_from_cam[:3, :3].T  # camera z == ray parameter

    dirs_rgb, dirs_depth = rays(w, h, 1.0), rays(dw, dh, dw / w)
    origin = floor_from_cam[:3, 3]
    body = body_mesh(cfg)
    floor = o3d.geometry.TriangleMesh.create_box(4.0, 3.5, 0.02).translate((-2.0, -2.0, -0.02))
    wall = o3d.geometry.TriangleMesh.create_box(4.0, 0.02, 2.8).translate((-2.0, 1.2, 0.0))

    def cast(dirs, placed):
        scene = o3d.t.geometry.RaycastingScene()
        ids = [scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m)) for m in (placed, floor, wall)]
        r = np.concatenate([np.broadcast_to(origin, dirs.shape), dirs], -1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(r))
        return ans["t_hit"].numpy(), ans["geometry_ids"].numpy(), ids[0]

    video = cv2.VideoWriter(str(folder / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (w, h))
    video.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    world_from_floor = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # odometry world is +Y up
    quat = Rotation.from_matrix(world_from_floor @ floor_from_cam[:3, :3]).as_quat()
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
    cache: dict[tuple, tuple] = {}
    for i, pose in enumerate(frames):
        if pose not in cache:
            th, x, y = pose
            c, s = np.cos(th), np.sin(th)
            t = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])
            placed = o3d.geometry.TriangleMesh(body)
            placed.transform(t)
            hit, gid, body_id = cast(dirs_rgb, placed)
            pts = origin + dirs_rgb * np.where(np.isfinite(hit), hit, 0.0)[..., None]
            in_body = (pts - t[:3, 3]) @ t[:3, :3]  # body-frame coordinates: the pattern turns with the object
            tex_xyz = np.where((gid == body_id)[..., None], in_body, pts)
            img = _texture(tex_xyz.reshape(-1, 3), noise, 0.018).reshape(h, w)
            img = np.where(np.isfinite(hit), img, 0).astype(np.uint8)
            zd = cast(dirs_depth, placed)[0]
            cache[pose] = (img, np.where(np.isfinite(zd), zd, 0.0))
        img, z = cache[pose]
        video.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        noisy = np.where(z > 0, z + rng.normal(0, cfg.depth_noise_m, z.shape), 0.0)
        cv2.imwrite(str(folder / "depth" / f"{i:06d}.png"), np.round(noisy * 1000).astype(np.uint16))
        cv2.imwrite(str(folder / "confidence" / f"{i:06d}.png"), np.full(z.shape, 2, np.uint8))
        rows.append(f"{100 + i / cfg.fps:.6f}, {i:06d}, 0, 0, 0, " + ", ".join(f"{q:.9f}" for q in quat)
                    + f", {fx}, {fx}, {cx}, {cy}")
    video.release()
    (folder / "odometry.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    up_cam = floor_from_cam[:3, :3].T @ [0, 0, 1.0]
    acc = -(IMU_TO_CAMERA.T @ up_cam)
    imu = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
    imu += [f"{100 + i / 100:.6f}, {acc[0]}, {acc[1]}, {acc[2]}, 0, 0, 0" for i in range(int(len(frames) / cfg.fps * 100))]
    (folder / "imu.csv").write_text("\n".join(imu) + "\n", encoding="utf-8")

    truth = []
    for th, x, y in stops:
        c, s = np.cos(th), np.sin(th)
        truth.append(np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
    return {"floor_from_body": truth, "key_frames": keys, "floor_from_camera": floor_from_cam, "n_frames": len(frames),
            "up_camera": up_cam, "config": cfg}
