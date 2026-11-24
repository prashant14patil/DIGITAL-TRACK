# Ingest and calibration (Phase 1, safe components)

What is implemented, how it behaves, and what it deliberately does not do. Everything here was tested on
**synthetic** data only (generated videos and rendered chessboards). Nothing here says anything about real iPad
recordings or real measurement accuracy.

## Capture metadata (`digitaltrack.ingest.metadata`)

`validate_capture_metadata(dict) -> CaptureMetadata` checks operator input and reports **all** problems at once
(`IngestError` code `invalid_metadata`, one entry per problem):

| Field | Rule |
|---|---|
| `subject_id`, `setup_id`, optional `calibration_id` | 3-64 chars of letters, digits, `_`, `-`; starts with a letter/digit. Blocks e-mails and "First Last" names; cannot prove an ID is non-identifying. |
| `protocol_version` | required text |
| `capture_type` | `rgbd` (primary; no adapter yet) or `rgb_video` (fallback) |
| `camera_distance_m`, `camera_height_m` | optional, metres, in (0, 10] |
| `clothing`, `posture_notes`, `device_model`, `recording_app` | optional text, at most 500 chars, no control characters. `device_model` is stored as stated, unverified. |
| anything else | rejected as "unknown field" (nothing is silently dropped) |

`CaptureMetadata.session_fields()` emits **only existing schema-0.2.0 fields** (`subject`, `rig`, `device`,
`capture`, `calibration_id`). Manual self-rotation is expressed with existing values:
`capture.subject_on_turntable: false`, `capture.rotation_source: "none"`, and a free-text
`capture.rotation_axis_frame` stating that no rotation axis is defined. No schema change.

## Camera-app video ingest (`digitaltrack.ingest.video`, `digitaltrack.ingest.session`)

`ingest_video(source, session_dir, metadata, *, frame_step=1, max_frames=2000, expected_sha256=None, ...)`
creates a schema-0.2.0 `video_rgb` session. `ingest_video_from_upload(upload_dir, export_relpath, ...)` does the
same for one file of a **finalized** raw upload (`api/storage.py` layout); the file's size and SHA-256 must match
the upload's `manifest.json`, and the upload is only read, never written.

Steps and checks (each failure raises `IngestError` with a stable code):

1. **Source checks:** exists (`source_not_found`), is a file (`source_not_a_file`), non-empty (`source_empty`),
   extension `.mov/.mp4/.m4v` (`unsupported_extension`), starts with a QuickTime/ISO-BMFF atom
   (`not_quicktime_mp4`), optional expected size/SHA-256 (`size_mismatch`, `checksum_mismatch`).
2. **Probe (ffprobe):** first non-cover-art video stream; codec, size, pixel format, bit depth, frame rates,
   colour transfer, display rotation, other streams. Flags: `variable_frame_rate_suspected`,
   `hdr_transfer_function`, `high_bit_depth`, `display_rotation_not_applied`, `non_square_pixels`,
   `location_metadata_present_not_copied`. **Tag values (e.g. GPS) are never stored**, only the flag.
   No video stream: `no_video_stream`. The probe (with flags) is kept in `session.source_video.probe`.
3. **Timestamps:** ffprobe lists every decoded frame's presentation time.
4. **Decode (ffmpeg):** `-noautorotate` (stored orientation kept; rotation reported, not applied),
   `-fps_mode passthrough` (no duplicated/dropped frames), `rgb24` for 8-bit sources and `rgb48le` for >8-bit
   sources (16-bit PNG, no extra quantisation). HDR transfer functions are not linearised.
5. **Selection:** every `frame_step`-th decoded frame from frame 0. Exceeding `max_frames` stops the ingest
   (`too_many_frames`). No still-frame / pause detection yet.
6. **Write:** via `dataset.writer` into a temporary sibling folder: source video copied byte-for-byte and
   re-hashed, lossless PNG frames, per-frame `video.source_frame_index` + `pts_s`, `angle_source: "none"`.
   The decoded frame count must equal the timestamp count (`frame_count_mismatch`). The session must pass
   `validate_session` (`validation_failed`) before it is renamed into place. **On any failure nothing is left
   behind.** An existing destination is never overwritten (`destination_exists`).

Expected validation result of an ingested session: valid, with warnings `angles_missing` and
`calibration_missing`. It cannot be reconstructed: per-view pose tracking for manual self-rotation is an open
item.

`ffmpeg`/`ffprobe` must be on PATH (or set `DIGITALTRACK_FFMPEG` / `DIGITALTRACK_FFPROBE`); otherwise
`tool_missing`. The video tests are skipped, with a reason, when FFmpeg is absent.

**Verified on synthetic videos:** pixel-exact 8-bit and 16-bit frames (lossless PNG-in-MOV sources), exact
timestamps, frame-step indices, rotation flag without applying it, location flag without leaking the value,
H.264 (lossy) frame accounting, all refusal paths, and no partial output after a failure.

## Camera intrinsics (`digitaltrack.calibration.intrinsics`)

* `calibrate_intrinsics(images, ChessboardSpec(inner_cols, inner_rows, square_size_m), min_views=8)`:
  OpenCV chessboard detection + sub-pixel refinement + `calibrateCamera` (Brown-Conrady k1,k2,p1,p2; k3 fixed
  by default). Reports RMS and per-view reprojection error, used and rejected views. All images must share one
  resolution. `estimate_distortion=False` only for optics known to be distortion-free (synthetic renders).
* `camera_calibration_document(result, calibration_id, image_source="rgb", video_mode=...)`: schema-valid
  `camera_calibration.json` content (validated before returning).
* `attach_camera_calibration(session_dir, doc)`: adds `calibration/camera_calibration.json` to a session that has
  none. Refuses a different `calibration_id`, a resolution that does not match the session's frames, an
  existing calibration (never overwritten), or a non-0.2.0 session. Written read-only.

Calibrate from frames extracted exactly like the session frames (same video mode and extraction rule,
including `-noautorotate`); otherwise the intrinsics do not apply.

**Synthetic results (pinhole renders, 640x480, true fx = fy = 600, cx = 320, cy = 240, 2-level noise):**
25 views recover fx/fy within 0.07 % and cx/cy within about 1.2 px; 15 views gave about 0.35 % focal error,
so view count and pose variety matter. Lens distortion was not simulated. ChArUco boards (named in the capture
protocol) are not supported yet; the output document would be the same.

## Scale and floor (`digitaltrack.calibration.scale`, `digitaltrack.calibration.floor`)

* `scale_from_known_lengths(measured, reference_m)`: least-squares scale through the origin, residuals, RMS and
  largest relative residual; `status()` with **experimental** thresholds (warn > 0.5 %, fail > 1 %).
* `fit_floor_plane(points, distance_threshold_m=0.01, ...)`: seeded RANSAC + SVD refit; normal oriented towards
  `up_hint` (default `-y`, up in the OpenCV camera frame). Refuses degenerate input and planes holding less than
  `min_inlier_fraction` of the points. `floor_from_camera(plane, origin_hint)`: camera -> floor transform with
  origin on the floor and +Z up; its horizontal axes are **provisional** (the body frame depends on the open pose
  model).

Both are **library only**. Schema 0.2.0 has no field for a scale estimate or a floor plane, so nothing is
persisted. Persisting them would be a schema change (new optional calibration fields or a new calibration file,
a version bump with a migration) and needs approval first.

## Not implemented (by design)

* RGB-D (LiDAR) export adapter: no real export from our iPad has been inspected; no format is assumed.
* Per-view angle/pose estimation for manual self-rotation, and any ICP-based pose correction.
* Still-frame (pause) detection, segmentation, reconstruction from real captures.
* CLI commands for these functions (to avoid concurrent edits of `cli.py`); use the Python API for now.
