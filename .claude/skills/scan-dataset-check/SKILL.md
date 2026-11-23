---
name: scan-dataset-check
description: Validate a scan session folder (scan_<session_id>) for completeness and data integrity before reconstruction. Use when importing or debugging a video-derived (or synthetic/depth) scan dataset.
---

Given a scan session path, check and report pass/fail per item (do not modify data). Run `dt validate <path>`
first and include its result; then check what it does not cover.

1. Layout: `session.json`, `calibration/camera_calibration.json`, `calibration/turntable_calibration.json`,
   `frames/frame_*/frame_metadata.json`. Video sessions: `source/` holds the original video, unmodified,
   read-only, SHA-256 matches `session.json`. Frames carry lossless `rgb.png`; `depth_raw.bin` +
   `confidence.bin` only for the depth branch.
2. Video provenance: codec, resolution, frame rate, variable-frame-rate flag, rotation metadata, frame count,
   extraction parameters (selection rule, decoder, versions) recorded.
3. Per frame: video timestamp, frame index in the video, `turntable_angle_deg` with `angle_source`
   (marker/encoder/manual/synthetic). Manual or constant-speed angles are experiment-only: warn.
4. Calibration: intrinsics match the frame image size and video mode; reprojection error recorded; scale
   source and residual recorded.
5. Angular coverage: sorted angles span ~360°, max gap, frame count, monotonic direction.
6. Session metadata: pseudonymous subject ID only (no names), clothing/posture notes, rig/calibration ID,
   capture protocol version.
7. Depth branch only: lossless float32 metres, confidence size matches depth, zero/NaN/out-of-range fraction,
   smoothed depth stored separately.
8. Privacy: no location metadata in derived files; nothing under git.

Output a short table plus a list of blocking problems vs. warnings. Never "fix" by guessing missing values;
flag them.
