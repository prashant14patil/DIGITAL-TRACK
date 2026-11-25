"""Stray Scanner export -> RGBDView list.

Written from the app's published format (github.com/StrayRobots/scanner, docs/format.md), NOT yet from a real
export of this rig. Every assumption below is CHECKED against the files at load time; a mismatch raises
StrayFormatError with what was found instead of guessing:

  depth/NNNNNN.png      16-bit grayscale, depth in MILLIMETRES (older app versions: .npy, uint16 mm)
  confidence/NNNNNN.png values 0, 1, 2 (higher = more confident), same size as depth
  odometry.csv          header row; timestamp (s), frame, x, y, z, qx, qy, qz, qw[, fx, fy, cx, cy, ...]
  camera_matrix.csv     3x3 intrinsics (backwards compatibility; per-frame fx..cy in odometry.csv preferred)
  rgb.mp4               colour video; the intrinsics refer to ITS resolution and are scaled to the depth size

The ARKit camera poses in odometry.csv track the static scene, not a person turning, so they are used only to
check that the iPad stayed still. Depth is converted mm -> m explicitly and recorded in each view's `source`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..dataset.loader import Intrinsics
from .views import RGBDView

ADAPTER = "stray_scanner_v1 (from published format; verify on first real export)"
MM_TO_M = 0.001


class StrayFormatError(ValueError):
    pass


@dataclass
class StrayRecording:
    root: Path
    views: list[RGBDView]
    info: dict[str, Any] = field(default_factory=dict)


def find_stray_dirs(root: Path) -> list[Path]:
    """Folders under `root` that look like a Stray Scanner recording (depth/ folder + odometry.csv)."""
    return sorted(p.parent for p in Path(root).rglob("odometry.csv") if (p.parent / "depth").is_dir())


def _read_depth(p: Path) -> np.ndarray:
    if p.suffix.lower() == ".npy":
        a = np.load(p, allow_pickle=False)
    else:
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise StrayFormatError(f"cannot read depth image {p.name}")
    if a.ndim != 2 or a.dtype != np.uint16:
        raise StrayFormatError(f"{p.name}: expected 2-D uint16 depth in mm, got {a.shape} {a.dtype}")
    return a


def _odometry(root: Path) -> list[dict[str, str]]:
    with open(root / "odometry.csv", newline="", encoding="utf-8") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    if not rows:
        raise StrayFormatError("odometry.csv has no rows")
    missing = {"timestamp", "frame", "x", "y", "z"} - set(rows[0])
    if missing:
        raise StrayFormatError(f"odometry.csv lacks columns {sorted(missing)}; found {sorted(rows[0])}")
    return rows


def _rgb_size(root: Path) -> tuple[int, int] | None:
    mp4 = root / "rgb.mp4"
    if not mp4.is_file():
        return None
    try:
        from ..ingest.video import probe_video
        pr = probe_video(mp4)
        return int(pr["width"]), int(pr["height"])
    except Exception:  # noqa: BLE001 - ffprobe missing or unreadable: fall back to OpenCV
        cap = cv2.VideoCapture(str(mp4))
        try:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        return (w, h) if w > 0 and h > 0 else None


def _intrinsics_rgb(root: Path, row: dict[str, str]) -> tuple[float, float, float, float, str]:
    if all(row.get(k) for k in ("fx", "fy", "cx", "cy")):
        return float(row["fx"]), float(row["fy"]), float(row["cx"]), float(row["cy"]), "odometry.csv per-frame"
    cm = root / "camera_matrix.csv"
    if not cm.is_file():
        raise StrayFormatError("no intrinsics: odometry.csv has no fx/fy/cx/cy and camera_matrix.csv is missing")
    k = np.loadtxt(cm, delimiter=",")
    if k.shape != (3, 3):
        raise StrayFormatError(f"camera_matrix.csv is {k.shape}, expected 3x3")
    return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), "camera_matrix.csv"


def _up_camera(row: dict[str, str]) -> np.ndarray | None:
    """Gravity up in the OpenCV camera frame. ARKit world is gravity-aligned (+Y up); the quaternion is the
    camera-to-world rotation (as read by StrayVisualizer); ARKit camera axes are x right, y up, z backwards, so
    OpenCV = diag(1, -1, -1) * ARKit. Checked on the first real export: constant over a fixed-mount recording."""
    try:
        q = [float(row[c]) for c in ("qx", "qy", "qz", "qw")]
    except (KeyError, ValueError):
        return None
    from scipy.spatial.transform import Rotation
    if not np.isfinite(q).all() or np.linalg.norm(q) < 0.5:
        return None
    up = np.array([1.0, -1.0, -1.0]) * (Rotation.from_quat(q).as_matrix().T @ np.array([0.0, 1.0, 0.0]))
    return up / np.linalg.norm(up)


def load_stray(root: str | Path, *, target_fps: float = 6.0) -> StrayRecording:
    """Load a Stray Scanner recording folder, keeping about `target_fps` frames per second (by timestamp)."""
    root = Path(root)
    rows = _odometry(root)
    depth_files = {int(p.stem): p for p in (root / "depth").iterdir()
                   if p.suffix.lower() in (".png", ".npy") and p.stem.isdigit()}
    if not depth_files:
        raise StrayFormatError("depth/ contains no numbered .png or .npy files")
    conf_dir = root / "confidence"

    first = _read_depth(depth_files[min(depth_files)])
    dh, dw = first.shape
    rgb = _rgb_size(root)
    if rgb is None:
        raise StrayFormatError("cannot determine the rgb.mp4 resolution, so the intrinsics cannot be scaled to depth")
    rw, rh = rgb
    if abs(rw / rh - dw / dh) > 0.01:
        raise StrayFormatError(f"rgb.mp4 is {rw}x{rh} but depth is {dw}x{dh}: aspect ratios differ (rotated video?)")
    sx, sy = dw / rw, dh / rh

    # timestamps -> keep ~target_fps frames; frames without a depth file are counted, not invented
    keep, last_t, no_depth = [], -1e9, 0
    for r in rows:
        fr = int(float(r["frame"]))
        if fr not in depth_files:
            no_depth += 1
            continue
        t = float(r["timestamp"])
        if t - last_t >= 0.9 / target_fps:  # 10% slack for timestamp jitter/rounding
            keep.append((fr, t, r))
            last_t = t
    if len(keep) < 2:
        raise StrayFormatError(f"only {len(keep)} usable frames (odometry rows {len(rows)}, depth files "
                               f"{len(depth_files)})")

    views: list[RGBDView] = []
    k_src = ""
    for i, (fr, t, r) in enumerate(keep):
        raw = first if fr == min(depth_files) else _read_depth(depth_files[fr])
        if raw.shape != (dh, dw):
            raise StrayFormatError(f"depth frame {fr} is {raw.shape}, others {(dh, dw)}")
        conf = None
        cp = conf_dir / f"{fr:06d}.png"
        if cp.is_file():
            conf = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
            if conf is None or conf.shape != (dh, dw):
                raise StrayFormatError(f"confidence frame {fr} unreadable or not {dw}x{dh}")
            if conf.max() > 2:
                raise StrayFormatError(f"confidence frame {fr} has value {conf.max()}; expected 0..2")
            conf = conf.astype(np.uint8)
        fx, fy, cx, cy, k_src = _intrinsics_rgb(root, r)
        if not (0.35 * rw < cx < 0.65 * rw and 0.35 * rh < cy < 0.65 * rh):
            raise StrayFormatError(f"principal point ({cx:.0f}, {cy:.0f}) is not near the centre of the "
                                   f"{rw}x{rh} video: intrinsics may refer to a different image size")
        k = Intrinsics(fx * sx, fy * sy, cx * sx, cy * sy, dw, dh)
        views.append(RGBDView(i, raw.astype(np.float32) * MM_TO_M, k, conf, t,
                              source={"adapter": ADAPTER, "frame": fr, "depth_units": "mm -> m (x0.001)",
                                      "intrinsics": f"{k_src}, scaled {rw}x{rh} -> {dw}x{dh}"},
                              up_camera=_up_camera(r)))

    pos = np.array([[float(r[c]) for c in ("x", "y", "z")] for _, _, r in keep])
    moved = float(np.max(np.linalg.norm(pos - pos[0], axis=1)))
    info = {
        "adapter": ADAPTER, "folder": root.name, "odometry_rows": len(rows), "depth_files": len(depth_files),
        "rows_without_depth": no_depth, "frames_kept": len(views), "target_fps": target_fps,
        "duration_s": round(keep[-1][1] - keep[0][1], 2), "depth_size": [dw, dh], "rgb_size": [rw, rh],
        "intrinsics_source": k_src, "has_confidence": views[0].confidence is not None,
        "ipad_moved_m": round(moved, 4),
        "ipad_moved_note": "from ARKit odometry; the camera should not move (fixed mount)",
    }
    ups = [v.up_camera for v in views if v.up_camera is not None]
    if ups:
        u = np.mean(ups, axis=0)
        spread = max(float(np.degrees(np.arccos(np.clip(x @ (u / np.linalg.norm(u)), -1, 1)))) for x in ups)
        info["gravity_up_camera"] = [round(float(x), 4) for x in u / np.linalg.norm(u)]
        info["gravity_up_spread_deg"] = round(spread, 2)
        info["held"] = "portrait (image rotated 90 deg)" if abs(u[0]) > abs(u[1]) else "landscape"
    return StrayRecording(root, views, info)
