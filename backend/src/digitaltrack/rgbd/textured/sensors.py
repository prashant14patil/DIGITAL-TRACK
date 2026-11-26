"""Stray Scanner export: audit, frame synchronisation, RGB decoding, gravity (odometry + IMU), camera motion.

Conventions below were checked on the real export of 2026-11-25 and are re-checked on every run where possible:

* Frame association: depth/NNNNNN.png, confidence/NNNNNN.png, odometry row `frame` and the N-th packet of rgb.mp4
  belong together. rgb.mp4 starts with a packet at negative PTS hidden by an MP4 edit list, so a normal decode
  drops frame 0 and pairs every RGB frame with the previous depth frame. Decoding therefore uses
  `-ignore_editlist 1`. MP4 timestamps are nominal (exact 1/60 s grid); odometry timestamps are the clock.
* Odometry quaternion: camera-to-world rotation of a camera with OpenCV axes (x right, y down, z forward) in a
  gravity-aligned world with +Y up, so gravity "up" in the camera frame is R.T @ (0, 1, 0). Verified against the
  floor/wall planes of the depth maps (about 1 deg) and the accelerometer.
  NOTE: `rgbd.stray._up_camera` applies an extra diag(1, -1, -1), which disagrees with those checks by about 5 deg.
* imu.csv: accelerometer in g, device axes; camera "up" = IMU_TO_CAMERA @ (-a/|a|). Gyro units are not verifiable
  from a static device and the gyro is not used for poses.
* Intrinsics: odometry fx, fy, cx, cy refer to the rgb.mp4 size and are scaled to the depth size.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ...dataset.loader import Intrinsics
from ...ingest.video import find_tool, probe_video
from ..stray import StrayFormatError

IMU_TO_CAMERA = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
MM_TO_M = 0.001


@dataclass
class Export:
    root: Path
    frames: np.ndarray  # frame numbers (contiguous from odometry)
    t: np.ndarray  # odometry timestamps, s
    position: np.ndarray  # (N, 3) odometry camera position, world, m
    quat: np.ndarray  # (N, 4) camera-to-world, xyzw
    k_rgb: np.ndarray  # (N, 4) fx fy cx cy at rgb size
    imu: np.ndarray | None  # (M, 7) t, accel (g), gyro
    rgb_size: tuple[int, int]
    depth_size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.frames)

    def depth_raw(self, i: int) -> np.ndarray:
        p = self.root / "depth" / f"{int(self.frames[i]):06d}.png"
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None or a.dtype != np.uint16 or a.shape != self.depth_size[::-1]:
            raise StrayFormatError(f"{p.name}: expected {self.depth_size} uint16 depth in mm")
        return a

    def confidence(self, i: int) -> np.ndarray:
        p = self.root / "confidence" / f"{int(self.frames[i]):06d}.png"
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None or a.shape != self.depth_size[::-1] or a.max() > 2:
            raise StrayFormatError(f"confidence {p.name}: missing, wrong size or values outside 0..2")
        return a.astype(np.uint8)

    def intrinsics_rgb(self, i: int) -> Intrinsics:
        fx, fy, cx, cy = self.k_rgb[i]
        return Intrinsics(float(fx), float(fy), float(cx), float(cy), *self.rgb_size)

    def intrinsics_depth(self, i: int) -> Intrinsics:
        sx = self.depth_size[0] / self.rgb_size[0]
        sy = self.depth_size[1] / self.rgb_size[1]
        fx, fy, cx, cy = self.k_rgb[i]
        return Intrinsics(float(fx * sx), float(fy * sy), float(cx * sx), float(cy * sy), *self.depth_size)

    def world_from_camera(self, i: int) -> np.ndarray:
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat(self.quat[i]).as_matrix()
        t[:3, 3] = self.position[i]
        return t


def _rgb_packet_pts(mp4: Path) -> np.ndarray:
    out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True, check=True)
    return np.sort(np.array([float(x.strip().strip(",")) for x in out.stdout.splitlines() if x.strip().strip(",")]))


def read_export(root: str | Path) -> tuple[Export, dict]:
    """Load the tables of an export and audit them. Raises StrayFormatError on anything that would need guessing."""
    root = Path(root)
    odo = root / "odometry.csv"
    for need in (odo, root / "rgb.mp4", root / "depth", root / "confidence"):
        if not need.exists():
            raise StrayFormatError(f"export lacks {need.name}")
    with open(odo, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    cols = ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy")
    if len(rows) < 2 or set(cols) - set(rows[0]):
        raise StrayFormatError(f"odometry.csv needs at least 2 rows and columns {cols}")
    arr = np.array([[float(r[c]) for c in cols] for r in rows])
    t, frames = arr[:, 0], arr[:, 1].astype(int)
    if np.any(np.diff(t) <= 0) or np.any(np.diff(frames) <= 0):
        raise StrayFormatError("odometry timestamps / frame numbers are not strictly increasing")
    depth_nums = {int(p.stem) for p in (root / "depth").glob("*.png") if p.stem.isdigit()}
    conf_nums = {int(p.stem) for p in (root / "confidence").glob("*.png") if p.stem.isdigit()}
    missing_d, missing_c = set(frames) - depth_nums, set(frames) - conf_nums
    if missing_d or missing_c:
        raise StrayFormatError(f"{len(missing_d)} frames lack depth and {len(missing_c)} lack confidence "
                               "(incomplete or partially unpacked export)")
    first = cv2.imread(str(root / "depth" / f"{frames[0]:06d}.png"), cv2.IMREAD_UNCHANGED)
    if first is None or first.ndim != 2 or first.dtype != np.uint16:
        raise StrayFormatError("depth is not 16-bit single-channel PNG (mm)")
    probe = probe_video(root / "rgb.mp4")
    rgb_size = (int(probe["width"]), int(probe["height"]))
    depth_size = (first.shape[1], first.shape[0])
    if abs(rgb_size[0] / rgb_size[1] - depth_size[0] / depth_size[1]) > 0.01:
        raise StrayFormatError(f"rgb {rgb_size} and depth {depth_size} aspect ratios differ")
    cx, cy = arr[:, 11], arr[:, 12]
    if not (np.all(np.abs(cx / rgb_size[0] - 0.5) < 0.15) and np.all(np.abs(cy / rgb_size[1] - 0.5) < 0.15)):
        raise StrayFormatError("principal point is not near the image centre: intrinsics refer to another size?")
    pts = _rgb_packet_pts(root / "rgb.mp4")
    if len(pts) != len(rows) or not np.array_equal(frames, np.arange(len(rows))):
        raise StrayFormatError(f"rgb.mp4 has {len(pts)} frames but odometry has {len(rows)} rows (frames "
                               f"{frames[0]}..{frames[-1]}): frames cannot be matched by index without guessing")
    imu = None
    if (root / "imu.csv").is_file():
        imu = np.loadtxt(root / "imu.csv", delimiter=",", skiprows=1, ndmin=2)
        if imu.shape[1] != 7:
            raise StrayFormatError(f"imu.csv has {imu.shape[1]} columns, expected 7")
    exp = Export(root, frames, t, arr[:, 2:5], arr[:, 5:9], arr[:, 9:13], imu, rgb_size, depth_size)

    dt = np.diff(t)
    nominal = float(np.median(dt))
    drift = (t - t[0]) - (pts - pts[0])
    audit = {
        "frames": len(rows), "duration_s": round(float(t[-1] - t[0]), 3),
        "fps_from_odometry": round(1.0 / nominal, 3),
        "timestamp_gaps_over_1_5x": int((dt > 1.5 * nominal).sum()),
        "rgb": {"size": list(rgb_size), "codec": probe.get("codec"), "packets": int(len(pts)),
                "first_packet_pts_s": float(pts[0]), "pts_minus_odometry_max_ms": round(float(np.abs(drift).max()) * 1e3, 2),
                "association": "by frame index (MP4 timestamps are nominal), edit list ignored"},
        "depth": {"size": list(depth_size), "dtype": "uint16", "units": "mm (x0.001 -> m, per app documentation)"},
        "confidence_values": "0, 1, 2",
        "intrinsics": {"source": "odometry.csv per frame, scaled to depth size", "fx_range": [float(arr[:, 9].min()), float(arr[:, 9].max())],
                       "camera_matrix_csv": (root / "camera_matrix.csv").is_file()},
        "world_frame": "odometry: gravity aligned, +Y up, camera-to-world quaternion for OpenCV camera axes",
        "imu_samples": None if imu is None else int(len(imu)),
    }
    return exp, audit


def _select_expression(indices: list[int]) -> str:
    """ffmpeg expression true for exactly these (sorted) frame numbers. Runs become between(); the sum is nested
    as a balanced tree because ffmpeg's parser rejects long flat sums."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    terms = [f"eq(n,{a})" if a == b else f"between(n,{a},{b})" for a, b in runs]
    while len(terms) > 1:
        terms = [f"({terms[k]}+{terms[k + 1]})" if k + 1 < len(terms) else terms[k] for k in range(0, len(terms), 2)]
    return terms[0]


def iter_frames(mp4: Path, indices: list[int], size: tuple[int, int], *, gray: bool = True,
                scale_to: tuple[int, int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, image) for the given frame indices, in stored orientation, counting the hidden first frame."""
    want = sorted(set(int(i) for i in indices))
    if not want:
        return
    w, h = scale_to or size
    ch, fmt = (1, "gray") if gray else (3, "bgr24")
    sel = f"select='{_select_expression(want)}'"
    if scale_to:
        sel += f",scale={w}:{h}"
    proc = subprocess.Popen(
        [find_tool("ffmpeg"), "-v", "error", "-nostdin", "-ignore_editlist", "1", "-noautorotate", "-i", str(mp4),
         "-vf", sel, "-frames:v", str(len(want)), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", fmt,
         "pipe:1"], stdout=subprocess.PIPE)
    try:
        for i in want:
            buf = proc.stdout.read(w * h * ch)
            if len(buf) != w * h * ch:
                raise StrayFormatError(f"rgb.mp4 ended before frame {i}")
            img = np.frombuffer(buf, np.uint8)
            yield i, img.reshape(h, w) if gray else img.reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def check_sync(exp: Export, *, window: int = 600, max_lag: int = 3) -> dict:
    """Correlate frame-to-frame motion in RGB and depth at several index lags on the busiest window.

    A confirmation only: the lag is never changed. Returns best_lag None when there is too little motion.
    """
    n = len(exp)
    window = min(window, n - 2 * max_lag - 1)
    if window < 30:
        return {"best_lag": None, "reason": "recording too short"}
    step = max(1, n // 100)
    coarse = [exp.depth_raw(i).astype(np.int32) for i in range(0, n, step)]
    motion = np.array([np.mean(np.abs(b - a) > 60) for a, b in zip(coarse, coarse[1:])])
    span = max(1, window // step)
    csum = np.convolve(motion, np.ones(min(span, len(motion))), mode="valid")
    start = int(np.clip(int(np.argmax(csum)) * step, max_lag, n - window - max_lag - 1))
    w, h = exp.depth_size
    rgb = [img.astype(np.float32) for _, img in iter_frames(exp.root / "rgb.mp4", range(start, start + window),
                                                            exp.rgb_size, scale_to=(w, h))]
    depth = [exp.depth_raw(i).astype(np.int32) for i in range(start - max_lag, start + window + max_lag)]
    dm = [cv2.GaussianBlur((np.abs(b - a) > 60).astype(np.float32), (3, 3), 0).ravel() for a, b in zip(depth, depth[1:])]
    rm = [cv2.GaussianBlur(np.abs(b - a), (3, 3), 0).ravel() for a, b in zip(rgb, rgb[1:])]
    scores = {}
    for lag in range(-max_lag, max_lag + 1):
        vals = []
        for i, r in enumerate(rm):
            d = dm[i + max_lag + lag]
            if r.std() > 0 and d.std() > 0:
                vals.append(float(np.corrcoef(r, d)[0, 1]))
        scores[lag] = float(np.mean(vals)) if len(vals) >= 20 else float("nan")
    arr = np.array(list(scores.values()))
    if not np.isfinite(arr).all() or arr.max() < 0.05:
        return {"best_lag": None, "reason": "not enough motion to test", "window_start": start}
    return {"best_lag": int(max(scores, key=scores.get)), "scores": {str(k): round(v, 4) for k, v in scores.items()},
            "window_start": start, "window_frames": window,
            "meaning": "lag 0 = RGB frame i matches depth frame i"}


def gravity(exp: Export) -> tuple[np.ndarray, dict]:
    """Unit "up" vector in the camera frame: odometry (primary), cross-checked with the accelerometer."""
    ups = np.stack([Rotation.from_quat(q).as_matrix().T @ np.array([0.0, 1.0, 0.0]) for q in exp.quat[::10]])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    info = {"source": "odometry quaternion", "up_camera": up.round(5).tolist(),
            "spread_deg": round(float(np.degrees(np.arccos(np.clip(ups @ up, -1, 1))).max()), 3), "imu": None}
    if exp.imu is not None and len(exp.imu):
        acc = exp.imu[:, 1:4]
        norm = np.linalg.norm(acc, axis=1)
        still = np.abs(norm - np.median(norm)) < 0.02
        a = IMU_TO_CAMERA @ (-acc[still].mean(0))
        a /= np.linalg.norm(a)
        info["imu"] = {"up_camera": a.round(5).tolist(), "disagreement_deg": round(float(np.degrees(np.arccos(np.clip(a @ up, -1, 1)))), 3),
                       "accel_norm_g": round(float(np.median(norm)), 4), "accel_norm_std_g": round(float(norm.std()), 4),
                       "gyro_mean": exp.imu[:, 4:7].mean(0).round(5).tolist(),
                       "gyro_std": exp.imu[:, 4:7].std(0).round(5).tolist(),
                       "gyro_used": False, "gyro_note": "camera is fixed; gyro cannot observe the subject's rotation",
                       "clock": {"start_offset_s": round(float(exp.imu[0, 0] - exp.t[0]), 4),
                                 "end_offset_s": round(float(exp.imu[-1, 0] - exp.t[-1]), 4)}}
    return up, info


def camera_motion(exp: Export, indices: list[int], subject_masks: dict[int, np.ndarray], *,
                  static_px: float = 3.0, agree_px: float = 3.0) -> tuple[dict[int, np.ndarray], dict]:
    """Decide whether the camera moved and return ref_camera_from_camera_i for each index (ref = indices[0]).

    Evidence is the RGB background (pixels outside the subject): its feature shift against the reference frame.
    * shift <= static_px everywhere: the camera is treated as static (identity); odometry drift is not injected.
    * larger shift that odometry predicts (within agree_px): odometry relative poses are applied.
    * larger shift that odometry does not explain: status "fail".
    """
    ref = indices[0]
    sift = cv2.SIFT_create(nfeatures=3000)
    kernel = np.ones((9, 9), np.uint8)

    def background_features(i: int, img: np.ndarray):
        m = subject_masks.get(i)
        mask = np.full(img.shape, 255, np.uint8)
        if m is not None:
            big = cv2.resize(m.astype(np.uint8), exp.rgb_size, interpolation=cv2.INTER_NEAREST)
            mask[cv2.dilate(big, kernel, iterations=6) > 0] = 0
        return sift.detectAndCompute(img, mask)

    images = dict(iter_frames(exp.root / "rgb.mp4", indices, exp.rgb_size))
    k0, d0 = background_features(ref, images[ref])
    depth_ref = exp.depth_raw(ref).astype(np.float32) * MM_TO_M
    kd, kr = exp.intrinsics_depth(ref), exp.intrinsics_rgb(ref)
    scale_x = exp.depth_size[0] / exp.rgb_size[0]
    t_w0 = exp.world_from_camera(ref)
    rows, rel = [], {}
    for i in indices:
        t_rel = np.linalg.inv(t_w0) @ exp.world_from_camera(i)
        rel[i] = t_rel
        if i == ref:
            continue
        k1, d1 = background_features(i, images[i])
        row = {"frame": int(exp.frames[i]), "matches": 0, "measured_px": None, "odometry_predicted_px": None,
               "odometry_residual_px": None}
        if d0 is not None and d1 is not None and len(k0) >= 8 and len(k1) >= 8:
            pairs = [m[0] for m in cv2.BFMatcher().knnMatch(d0, d1, k=2) if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
            p0 = np.array([k0[m.queryIdx].pt for m in pairs]).reshape(-1, 2)
            p1 = np.array([k1[m.trainIdx].pt for m in pairs]).reshape(-1, 2)
            near = np.linalg.norm(p1 - p0, axis=1) < 100  # repeated floor texture gives far false matches
            p0, p1 = p0[near], p1[near]
            if len(p0) >= 8:
                z = depth_ref[np.clip((p0[:, 1] * scale_x).round().astype(int), 0, kd.height - 1),
                              np.clip((p0[:, 0] * scale_x).round().astype(int), 0, kd.width - 1)]
                p3 = np.stack([(p0[:, 0] - kr.cx) / kr.fx * z, (p0[:, 1] - kr.cy) / kr.fy * z, z], 1)
                inv = np.linalg.inv(t_rel)
                q = p3 @ inv[:3, :3].T + inv[:3, 3]
                pred = np.stack([kr.fx * q[:, 0] / q[:, 2] + kr.cx, kr.fy * q[:, 1] / q[:, 2] + kr.cy], 1)
                row.update(matches=int(len(p0)),
                           measured_px=round(float(np.linalg.norm(np.median(p1 - p0, axis=0))), 2),
                           odometry_predicted_px=round(float(np.linalg.norm(np.median(pred - p0, axis=0))), 2),
                           odometry_residual_px=round(float(np.linalg.norm(np.median(p1 - pred, axis=0))), 2))
        rows.append(row)
    measured = [r["measured_px"] for r in rows if r["measured_px"] is not None]
    info = {"reference_frame": int(exp.frames[ref]), "per_frame": rows,
            "odometry_max_displacement_m": round(float(np.linalg.norm(exp.position - exp.position[ref], axis=1).max()), 4),
            "static_limit_px": static_px, "max_measured_px": max(measured) if measured else None}
    identity = {i: np.eye(4) for i in indices}
    if len(measured) < max(1, len(rows) // 2):
        info.update(decision="unverified", status="warn", note="too little background texture to check the camera; "
                    "treated as static (fixed-mount protocol)")
        return identity, info
    if max(measured) <= static_px:
        info.update(decision="static", status="pass", note="background shift within the static limit; odometry "
                    "differences are not applied")
        return identity, info
    resid = [r["odometry_residual_px"] for r in rows if r["odometry_residual_px"] is not None]
    if max(resid) <= agree_px:
        info.update(decision="odometry_applied", status="warn", note="camera moved and odometry explains the motion")
        return rel, info
    info.update(decision="unexplained_motion", status="fail", note="camera moved and odometry does not explain it")
    return identity, info
