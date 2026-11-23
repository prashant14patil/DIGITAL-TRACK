"""Load a validated session. Raw arrays are read-only views of what is on disk.

Sessions written under an older dataset schema are upgraded IN MEMORY (dataset.migrate) so downstream code
sees the current shape; the files on disk are never changed. `Session.schema_version_on_disk` keeps the
original version for logging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameRecord:
    root: Path
    frame_id: int
    dir: str
    timestamp_s: float
    angle_deg: float | None  # recorded angle; None = not yet determined
    meta: dict[str, Any]

    @property
    def has_depth(self) -> bool:
        return "depth" in self.meta

    @property
    def has_rgb(self) -> bool:
        return "rgb" in self.meta

    def load_depth(self) -> np.ndarray:
        if not self.has_depth:
            raise ValueError(f"frame {self.frame_id} has no depth (RGB-only session)")
        d = self.meta["depth"]
        arr = np.fromfile(self.root / self.dir / d["file"], dtype="<f4")
        return arr.reshape(d["height"], d["width"]).astype(np.float32, copy=False)

    def load_confidence(self) -> np.ndarray:
        if "confidence" not in self.meta:
            raise ValueError(f"frame {self.frame_id} has no confidence map")
        c = self.meta["confidence"]
        arr = np.fromfile(self.root / self.dir / c["file"], dtype=np.uint8)
        return arr.reshape(c["height"], c["width"])

    def load_rgb(self) -> np.ndarray:
        """(H, W, 3) uint8/uint16 in RGB channel order, exactly as stored (lossless)."""
        import cv2

        if not self.has_rgb:
            raise ValueError(f"frame {self.frame_id} has no rgb image")
        r = self.meta["rgb"]
        buf = np.fromfile(self.root / self.dir / r["file"], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape != (r["height"], r["width"], 3):
            raise ValueError(f"frame {self.frame_id}: rgb decode failed or size differs from metadata")
        return img[..., ::-1].copy()


@dataclass(frozen=True)
class Session:
    root: Path
    meta: dict[str, Any]
    camera: dict[str, Any] | None
    turntable: dict[str, Any] | None
    frames: list[FrameRecord]
    schema_version_on_disk: str

    @property
    def session_id(self) -> str:
        return self.meta["session_id"]

    @property
    def calibration_id(self) -> str:
        return self.meta["calibration_id"]

    @property
    def input_type(self) -> str:
        return self.meta["input_type"]

    @property
    def has_depth(self) -> bool:
        return bool(self.frames) and all(f.has_depth for f in self.frames)

    @property
    def angles_complete(self) -> bool:
        return all(f.angle_deg is not None for f in self.frames)

    def _require(self, what: str, obj):
        if obj is None:
            raise ValueError(f"session {self.session_id} has no {what} (not calibrated yet)")
        return obj

    @property
    def intrinsics(self) -> Intrinsics:
        cam = self._require("camera calibration", self.camera)
        k = cam["intrinsics"]
        return Intrinsics(k["fx"], k["fy"], k["cx"], k["cy"], cam["image_width"], cam["image_height"])

    @property
    def camera_from_turntable(self) -> np.ndarray:
        return np.asarray(self._require("turntable calibration", self.turntable)["camera_from_turntable"],
                          dtype=np.float64)

    @property
    def angle_direction_sign(self) -> int:
        return int(self._require("turntable calibration", self.turntable)["angle_direction_sign"])


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def load_session(root: str | Path) -> Session:
    """Load without re-validating; call dataset.validate.validate_session first."""
    from .migrate import upgrade_documents

    root = Path(root)
    meta = _read_json(root / "session.json")
    cam_p, tt_p = root / "calibration" / "camera_calibration.json", root / "calibration" / "turntable_calibration.json"
    camera = _read_json(cam_p) if cam_p.is_file() else None
    turntable = _read_json(tt_p) if tt_p.is_file() else None
    frame_docs = [_read_json(root / e["dir"] / "frame_metadata.json") for e in meta["frames"]]
    on_disk = meta["schema_version"]
    meta, frame_docs, camera, turntable = upgrade_documents(meta, frame_docs, camera, turntable)
    frames = [FrameRecord(root, e["frame_id"], e["dir"], e["timestamp_s"], e.get("turntable_angle_deg"), fm)
              for e, fm in zip(meta["frames"], frame_docs)]
    return Session(root, meta, camera, turntable, frames, on_disk)
