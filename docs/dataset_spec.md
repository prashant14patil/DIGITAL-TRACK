# Dataset specification (schema_version 0.1.0)

The contract between input adapters (video ingest, synthetic, optional future depth) and the processing pipeline. The JSON
Schemas in [`schemas/`](../schemas) are normative; this page explains the conventions they
cannot express. Synthetic and real data use the same format; synthetic sessions are marked
`"synthetic": true` and live under `data/synthetic/`, real scans under `data/real/` (both git-ignored).

## Layout

```text
scan_<session_id>/
  session.json                      # manifest + per-frame checksums
  ground_truth.json                 # SYNTHETIC ONLY, never read by reconstruction
  calibration/
    camera_calibration.json         # intrinsics for the DEPTH image resolution
    turntable_calibration.json      # camera_from_turntable, angle sign
  frames/frame_000001/
    depth_raw.bin                   # float32, little-endian, metres, row-major HxW, 0 = no measurement
    confidence.bin                  # uint8 HxW (depth branch only; levels 0 low, 1 medium, 2 high)
    frame_metadata.json
  processing/<run_id>/              # derived data only; created by the backend, never edits raw files
    fused_mesh.ply  pointcloud_filtered.ply
    run_log.json  quality_report.json  # written by every reconstruction (`dt reconstruct`)
    measurements.json               # Phase 0: written only by tools/synthetic_demo.py (provisional slices)
```

## Raw data rules

* Raw files are written once, marked read-only, and listed with SHA-256 in `session.json` and
  `frame_metadata.json`. `dt validate` re-hashes them. The writer refuses to write into a non-empty directory.
* Depth is never stored in a lossy or integer-quantised format. Smoothed depth, if ever used, is a
  separate file and never replaces raw depth.
* The angle in the frame metadata is the **recorded** angle (what the rig reported). Synthetic truth is
  kept in `ground_truth.json` only.

## Coordinate conventions (single source: `digitaltrack/geometry/transforms.py`)

* **Camera frame:** OpenCV: x right, y down, z forward. Depth is the z coordinate in metres.
* **Pixel convention:** integer pixel index is the pixel centre: `x = (u-cx)/fx * z`, `y = (v-cy)/fy * z`.
  Intrinsics are valid only for the depth image size recorded in `camera_calibration.json`.
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

## Versioning

`schema_version` is checked as a constant; a format change bumps it and needs explicit approval.
