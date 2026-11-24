# Dataset specification (dataset schema_version 0.2.0)

The contract between input adapters (video ingest, synthetic, optional future depth) and the processing pipeline. The JSON
Schemas in [`schemas/`](../schemas) are normative; this page explains the conventions they
cannot express. Synthetic and real data use the same format; synthetic sessions are marked
`"synthetic": true` and live under `data/synthetic/`, real scans under `data/real/` (both git-ignored).

## Layout

```text
scan_<session_id>/
  session.json                      # manifest: input_type, source video, extraction, per-frame checksums
  ground_truth.json                 # SYNTHETIC ONLY, never read by reconstruction
  source/
    source_video.<ext>              # video_rgb only: the uploaded file, byte-for-byte, read-only, SHA-256
  calibration/                      # optional for RGB input until calibrated (validator warns)
    camera_calibration.json         # intrinsics for ONE image source (rgb|depth) at ONE resolution
    turntable_calibration.json      # camera_from_turntable, angle sign
  frames/frame_000001/
    rgb.png                         # lossless PNG, 8/16-bit, RGB channel order (RGB input)
    depth_raw.bin                   # optional depth branch: float32 LE metres, row-major HxW, 0 = no data
    confidence.bin                  # optional depth branch: uint8 HxW (0 low, 1 medium, 2 high)
    frame_metadata.json             # timestamp, angle (if known), video frame index + pts, file checksums
  processing/<run_id>/              # derived data only; created by the backend, never edits raw files
    fused_mesh.ply  pointcloud_filtered.ply
    run_log.json  quality_report.json  # written by every reconstruction (`dt reconstruct`)
    measurements.json               # Phase 0: written only by tools/synthetic_demo.py (provisional slices)
```

## Input types (`session.input_type`)

| input_type | Frames carry | Notes |
|---|---|---|
| `video_rgb` | `rgb` + `video` reference | **Primary.** Requires `source_video` and `frame_extraction` in session.json. |
| `synthetic_rgb` | `rgb` | Synthetic RGB/silhouette renders for tests. |
| `rgbd_depth` | `depth` + `confidence` (+ optional `rgb`) | Optional future depth branch. |
| `synthetic_depth` | `depth` + `confidence` | Virtual depth turntable (Phase 0). |

A frame's `turntable_angle_deg` is absent until the angle is determined (`angle_source: "none"`); such a
session validates with an `angles_missing` warning and cannot be reconstructed. The angle-capture mechanism
is not decided yet.

## Raw data rules

* Raw files are written once, marked read-only, and listed with SHA-256 in `session.json` and
  `frame_metadata.json`. `dt validate` re-hashes them. The writer refuses to write into a non-empty directory.
* The uploaded video is the raw artifact for video input. Extracted frames are lossless PNG; extraction
  parameters and software versions are recorded in `frame_extraction`, so frames can be regenerated.
* PNG bytes are deterministic (fixed compression level), so identical pixels give identical checksums.
* Depth is never stored in a lossy or integer-quantised format. Smoothed depth, if ever used, is a
  separate file and never replaces raw depth.
* The angle in the frame metadata is the **recorded** angle (what the rig reported). Synthetic truth is
  kept in `ground_truth.json` only.

## Coordinate conventions (single source: `digitaltrack/geometry/transforms.py`)

* **Camera frame:** OpenCV: x right, y down, z forward. Depth is the z coordinate in metres.
* **Pixel convention:** integer pixel index is the pixel centre: `x = (u-cx)/fx * z`, `y = (v-cy)/fy * z`.
  Intrinsics are valid only for the `image_source` and image size recorded in `camera_calibration.json`.
  Distortion (`opencv_brown_conrady`, OpenCV coefficient order) applies to that image source only.
* **Turntable/body frame:** origin on the rotation centre at the platform surface, +Z up (rotation axis).
  The subject is fixed in this frame.
* **Angle:** at recorded angle `A` the subject is rotated by `sign * A` about +Z (right-handed):
  `p_cam = camera_from_turntable @ Rz(sign*A) @ p_body`. `angle_direction_sign` (+1/-1) lives in the
  turntable calibration and must be verified with a calibration object on real hardware.
* `camera_transform` in frame metadata is `world_from_camera` as reported by a depth device, if any. It is
  constant for a fixed mount and is **not** the turntable angle.

## Processing log

Every reconstruction writes `run_log.json` (schema `processing_log`): all parameters, software versions,
the input `session.json` hash, the per-frame `camera_from_body` transform actually used, each step's
counts (pixels before/after filtering, triangles before/after cleanup) and SHA-256 of every output.

## Versioning and migration

* Dataset files (session, frame_metadata, camera/turntable calibration) are at **0.2.0**. Processing outputs
  (measurements, quality_report, processing_log) have their own version, still **0.1.0**.
* Every file in a session must declare the same dataset `schema_version` (`mixed_schema_versions` error).
* **0.1.0 sessions remain readable.** They are validated against the frozen schemas in `schemas/v0.1.0/` and
  upgraded in memory by the loader (`input_type` = `synthetic_depth`/`rgbd_depth`, `image_source` =
  `depth`). Files on disk are never changed. `run_log.json` records `dataset_schema_version_on_disk`.
* `dt migrate <old> <new>` writes an upgraded **copy** (raw files byte-identical and re-verified,
  `migrated_from` records the old version and the old `session.json` SHA-256). `processing/` is not copied.
* `tests/fixtures/session_v0_1_0_cylinder/` is a frozen 0.1.0 session; tests prove it still validates, loads,
  migrates losslessly and reconstructs to the same geometry as its migrated copy.
* A format change bumps the version, keeps the previous schemas under `schemas/v<old>/`, adds a migration
  and needs explicit approval.
