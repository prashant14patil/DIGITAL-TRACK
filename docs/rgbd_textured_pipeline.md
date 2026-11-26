# Texture-guided RGB-D reconstruction (Stray Scanner)

Package `digitaltrack.rgbd.textured`. A second pipeline next to `rgbd.pipeline` (unchanged, still used by the web
app). **Not validated against any physical reference**: it produces a mesh in metre units and a quality report,
and computes no body measurement.

```powershell
cd <repo>
backend\.venv\Scripts\dt reconstruct-stray <unpacked recording folder> --out data\real\processing\<id> --run-rgb-only
```

`--run-rgb-only` also runs the existing RGB-only pipeline (slow: 34 minutes for the 112 s recording below).
`--rgb-only-run <dir>` compares with an existing run instead. Needs FFmpeg 6 or newer on PATH.

## Export as inspected (Stray Scanner, recording of 2026-11-25)

| Item | Found |
|---|---|
| `rgb.mp4` | HEVC 1920x1440, 60 fps, stored sideways (device held portrait), no rotation tag |
| `depth/NNNNNN.png` | 256x192 uint16, millimetres (per app documentation; no reference length checked) |
| `confidence/NNNNNN.png` | 0, 1, 2 |
| `odometry.csv` | timestamp (s), frame, position, quaternion, fx fy cx cy per frame (at RGB size); distortion columns empty |
| `imu.csv` | about 100 Hz, acceleration in g (device axes) and angular rate; same clock as odometry |
| `camera_matrix.csv` | 3x3; may be absent, per-frame intrinsics are used |

Conventions that matter:

- **First frame is hidden.** `rgb.mp4` starts with a packet at negative PTS behind an edit list. A normal decode
  returns one frame fewer and pairs each RGB frame with the previous depth frame. Decode with
  `-ignore_editlist 1`. MP4 timestamps are a nominal 1/60 s grid; match frames by index.
- **Gravity.** "Up" in the camera frame is `R(q).T @ (0, 1, 0)`. `rgbd.stray._up_camera` additionally applies
  `diag(1, -1, -1)`, which disagrees with the floor and wall planes and the accelerometer by about 5 degrees.
  It is left unchanged here; the existing pipeline takes its vertical axis from the depth floor fit, so the effect
  there is on the floor-candidate gate and the reported tilt.
- **Accelerometer to camera axes:** `[[0,-1,0],[-1,0,0],[0,0,-1]] @ (-a/|a|)`.
- The pixel-centre convention of the scaled depth intrinsics (up to half a depth pixel, a few mm of lateral offset
  at 2-3 m) has not been verified.

## Stages

1. Audit of files and frame counts; RGB/depth lag test by motion correlation.
2. Gravity from odometry, cross-checked with the accelerometer. The gyroscope is not used: the camera is fixed and
   the gyroscope cannot observe the person turning.
3. Pauses (existing `detect_stops`); per pause, confidence-2 pixels, flying-pixel removal, temporal median.
   **No spatial smoothing.**
4. Floor height from depth; the vertical axis is gravity. Floor removed 2 cm above the fitted depth plane.
5. Subject segmentation (existing), plus a lower-body radius limit against background bridged between the legs.
6. Camera motion: odometry is applied only if the image background really moved.
7. Subject pose per node (every 0.5 s and every pause): SIFT matches on the body lifted with LiDAR depth, solved as
   an SE(2) pose graph with neighbour, pause-to-pause and loop pairs. ICP is diagnostic only.
8. Open3D TSDF (5 mm voxels, 20 mm truncation) of the pause views; topology-only cleanup (no vertex moved, no hole
   filled).
9. `mesh.ply`, `mesh.obj`, `mesh.glb` (metric, unsmoothed, read back to verify), `mesh_display.glb` (Taubin,
   display only), `pointcloud_registered.ply`.
10. `quality_report.json` and `report.html`, including the previous depth-shape pose method scored on the same views.

Coordinates are metres, **Z up**, floor at z = 0 (not glTF's Y-up).

## Result on the real recording of 2026-11-25 (112 s, 22 pauses)

Internal consistency only; no reference measurement exists for this scan.

| | texture poses (this pipeline) | depth-shape poses (`rgbd.pose`, same views) |
|---|---|---|
| Own status | warn | fail |
| Net turn, first to last pause | 335 deg | 295 deg |
| Loop error before optimisation | 40 deg (median over loop pairs) | 43 deg |
| Cross-view distance, median / p90 | 5.0 / 11.5 mm | 4.9 / 11.2 mm |
| View points to mesh, RMS / p95 | 12.8 / 24.8 mm | 14.3 / 25.0 mm |

The surface metrics are a tie. The difference is the net turn: in the video the last pause faces nearly the same way
as the first, and a direct image match puts them about 22 deg apart, which agrees with 335 and not with 295.
Pose sensitivity to the tracking interval is up to 3.9 deg and 14 mm (turn limit 3 deg: warn).

RGB-only (`experimental.rgb_visual_hull`) on the same recording: segmentation lost the legs (silhouettes cover 61 %
of the person's height), so no shape comparison is reported. Its estimated turn angles differ from the RGB-D
headings by 13 to 29 deg.

## Limits

- Needs textured clothing. Plain, tight clothing may give too few matches; the run then fails rather than guess.
- The person must turn in steps with pauses. Arms and hands that move between pauses reconstruct badly.
- Bottom 2 cm of the shoes is trimmed; the mesh is open (not watertight), so volume is unavailable.
- Internal consistency (cross-view distance, loop error) is not accuracy. A wrong but self-consistent result can
  score well; cross-view distance in particular barely changes when the turn angle is under-estimated.
- All pass/warn limits are proposed values.
- Next evidence needed: a known-size object recorded the same way, repeated scans of one person, tape references.
