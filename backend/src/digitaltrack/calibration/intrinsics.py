"""Camera intrinsics from views of a planar chessboard (OpenCV), written as `camera_calibration.json`.

The result describes ONE image source at ONE resolution (schema 0.2.0 `image_source` + `image_width/height`).
Calibrate from frames extracted exactly like the session frames (same video mode, same extraction rule, e.g.
`-noautorotate`), otherwise the intrinsics do not apply.

Chessboard only for now. The capture protocol mentions ChArUco boards; ChArUco support is a later addition
(OpenCV 5 provides `cv2.aruco.CharucoDetector`) and does not change the output document.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .. import DATASET_SCHEMA_VERSION
from ..dataset.schemas import schema_errors


class CalibrationError(ValueError):
    """Calibration could not produce a trustworthy result (too few views, inconsistent images, ...)."""


@dataclass(frozen=True)
class ChessboardSpec:
    inner_cols: int  # inner corners along the board's x direction
    inner_rows: int  # inner corners along y
    square_size_m: float  # measured edge length of one square, metres

    def object_points(self) -> np.ndarray:
        """(N, 3) float32 board-frame corner coordinates in metres, row-major, z = 0."""
        g = np.mgrid[0:self.inner_cols, 0:self.inner_rows].T.reshape(-1, 2)
        pts = np.zeros((g.shape[0], 3), np.float32)
        pts[:, :2] = g * self.square_size_m
        return pts


@dataclass
class IntrinsicsResult:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # OpenCV order k1, k2, p1, p2, k3
    rms_reprojection_px: float
    per_view_rms_px: list[float]
    views_used: list[int]  # indices of input images where the board was found
    views_rejected: list[int] = field(default_factory=list)

    @property
    def intrinsics(self) -> dict[str, float]:
        k = self.camera_matrix
        return {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])}


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise CalibrationError(f"unsupported image shape {img.shape}")
    if g.dtype == np.uint16:
        g = (g >> 8).astype(np.uint8)
    if g.dtype != np.uint8:
        raise CalibrationError(f"unsupported image dtype {g.dtype}")
    return g


def detect_chessboard(image: np.ndarray, spec: ChessboardSpec) -> np.ndarray | None:
    """(N, 1, 2) float32 sub-pixel corners in row-major board order, or None if the full board is not found."""
    g = _gray(image)
    found, corners = cv2.findChessboardCorners(
        g, (spec.inner_cols, spec.inner_rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    return cv2.cornerSubPix(g, corners.astype(np.float32), (5, 5), (-1, -1), crit)


def calibrate_intrinsics(images: Iterable[np.ndarray], spec: ChessboardSpec, *, min_views: int = 8,
                         fix_k3: bool = True, estimate_distortion: bool = True) -> IntrinsicsResult:
    """Calibrate from chessboard images (RGB or gray, uint8/uint16). All images must have the same size.

    `estimate_distortion=False` fixes all distortion terms at zero (only for optics known to be distortion-free,
    e.g. synthetic pinhole renders). Real lenses: keep the default.
    """
    obj_pts, img_pts, used, rejected = [], [], [], []
    size: tuple[int, int] | None = None
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            raise CalibrationError(f"image {i} is {w}x{h}, expected {size[0]}x{size[1]}: one calibration covers "
                                   "exactly one resolution")
        corners = detect_chessboard(img, spec)
        if corners is None:
            rejected.append(i)
            continue
        obj_pts.append(spec.object_points())
        img_pts.append(corners)
        used.append(i)
    if size is None:
        raise CalibrationError("no images given")
    if len(used) < min_views:
        raise CalibrationError(f"board found in {len(used)} image(s); at least {min_views} are required "
                               f"(rejected: {rejected})")
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    if not estimate_distortion:
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, np.zeros(5), flags=flags)
    per_view = []
    for o, ip, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, k, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))))
    return IntrinsicsResult(size[0], size[1], k, dist.reshape(-1)[:5], float(rms), per_view, used, rejected)


def camera_calibration_document(result: IntrinsicsResult, calibration_id: str, *, image_source: str = "rgb",
                                video_mode: dict[str, Any] | None = None, method: str | None = None) -> dict[str, Any]:
    """Schema-0.2.0 `camera_calibration.json` content. Raises if it would not validate."""
    doc: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "calibration_id": calibration_id,
        "camera_frame": "opencv_x_right_y_down_z_forward",
        "pixel_convention": "index_center",
        "image_source": image_source,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "intrinsics": result.intrinsics,
        "distortion": {"model": "opencv_brown_conrady", "coefficients": [float(c) for c in result.dist_coeffs]},
        "quality": {"reprojection_rms_px": result.rms_reprojection_px, "views_used": len(result.views_used)},
        "method": method or "opencv_calibrateCamera_chessboard",
    }
    if video_mode:
        doc["video_mode"] = video_mode
    errs = schema_errors(doc, "camera_calibration")
    if errs:
        raise CalibrationError(f"camera calibration document violates schema: {errs}")
    return doc


def attach_camera_calibration(session_dir: str | Path, doc: dict[str, Any]) -> Path:
    """Add `calibration/camera_calibration.json` to a session that has none yet.

    Refuses when a camera calibration already exists (never overwritten), when `calibration_id` differs from
    the session's, or when the calibrated resolution does not match the session's frames of that image source.
    """
    root = Path(session_dir)
    session = json.loads((root / "session.json").read_text(encoding="utf-8"))
    errs = schema_errors(doc, "camera_calibration")
    if errs:
        raise CalibrationError(f"camera calibration document violates schema: {errs}")
    if session.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise CalibrationError(f"session is schema {session.get('schema_version')!r}; migrate it to "
                               f"{DATASET_SCHEMA_VERSION} first (dt migrate)")
    if doc["calibration_id"] != session["calibration_id"]:
        raise CalibrationError(f"calibration_id {doc['calibration_id']!r} != session calibration_id "
                               f"{session['calibration_id']!r}")
    key = doc["image_source"]
    for entry in session["frames"]:
        fm = json.loads((root / entry["dir"] / "frame_metadata.json").read_text(encoding="utf-8"))
        if key in fm and (fm[key]["width"], fm[key]["height"]) != (doc["image_width"], doc["image_height"]):
            raise CalibrationError(f"{entry['dir']}: {key} is {fm[key]['width']}x{fm[key]['height']}, calibration is "
                                   f"{doc['image_width']}x{doc['image_height']}")
    dest = root / "calibration" / "camera_calibration.json"
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "x", encoding="utf-8") as f:  # 'x': never overwrite an existing calibration
        f.write(json.dumps(doc, indent=2) + "\n")
    os.chmod(dest, stat.S_IREAD)
    return dest
