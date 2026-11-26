# Architecture

Direction and rules: [`CLAUDE.md`](../CLAUDE.md). This page tracks what exists.

```text
iPad Camera app video (.MOV/.MP4)        [future: web upload -> API -> worker]
        │
        ▼
 input adapters ──► scan_<id>/ dataset (docs/dataset_spec.md, schemas/)
   video_rgb   (primary; Phase 1, in progress)
   rgbd_depth  (optional future depth branch)
   synthetic   (virtual turntable; implemented for depth)
        │
        ▼
 dataset.validate ─► calibration (intrinsics; turntable angle/axis/scale)  [Phase 1]
                 ─► segmentation (subject masks)                           [Phase 2]
                 ─► reconstruction backends
                       visual_hull (primary RGB)                            [Phase 2]
                       mvs (COLMAP/OpenMVS, optional, may need CUDA)        [later]
                       tsdf (Open3D VoxelBlockGrid; depth maps)             implemented
                 ─► quality ─► measurement (PROVISIONAL slices)
                 ─► processing/<run_id>/ {run_log, quality_report, measurements}
```

The primary workflow must run on CPU only (no NVIDIA GPU assumed). COLMAP dense MVS is an optional benchmark.

## Planned web layer (approved stack, not started; Phase 3)

| Part | Responsibility |
|---|---|
| Frontend (React + Vite + three.js) | Upload video + session metadata, job status per stage, 3D viewer of a display-only mesh coloured by result status, measurement table, quality report, downloads. Never computes measurements. |
| API (FastAPI) | Create sessions, accept uploads, enqueue jobs, serve status/results/display assets. Thin: calls `digitaltrack`. |
| Worker | Runs the pipeline outside the request cycle. |
| SQLite | Sessions, jobs, result index. Raw/derived files stay in `scan_<id>/`. |

## Implementation status

| Package | Status |
|---|---|
| `dataset` (schemas, writer, loader, validate) | implemented, tested (v0.1.0 depth sessions) |
| `synthetic` (shapes, virtual turntable depth renderer, truth evaluation) | implemented, tested |
| `geometry` (transforms, depth→points) | implemented, tested |
| `reconstruction.tsdf` | implemented, tested against analytic truth (depth branch) |
| `quality.basic`, `quality.registration` | implemented; blind to line-of-sight calibration error (xfail test) |
| `measurement.basic` | provisional horizontal-slice perimeter only |
| `rgbd.textured` (`dt reconstruct-stray`) | texture-guided Stray Scanner RGB-D pipeline with GLB/PLY/OBJ export, quality report and RGB-only comparison; CLI only, not wired into the web app; tested on synthetic data and run on one real recording, **accuracy not validated**. See [rgbd_textured_pipeline.md](rgbd_textured_pipeline.md) |
| `ingest` (capture metadata, Camera-app video probe + lossless frame extraction -> `video_rgb` session, ingest from a finalized upload) | implemented, tested on synthetic videos. **Stray Scanner zip adapter** (`rgbd/stray.py`) written from the app's published format (depth PNG mm, confidence 0-2, `odometry.csv` intrinsics scaled from `rgb.mp4` size); every assumption is checked at load time; tested only on synthetic data written in that format, **not yet on a real export**. See [ingest_and_calibration.md](ingest_and_calibration.md) |
| `calibration` (chessboard intrinsics -> `camera_calibration.json`; scale from known lengths; floor-plane fit) | implemented, tested on synthetic data; scale and floor results not persisted (no schema field). No ChArUco, no pose/angle estimation. See [ingest_and_calibration.md](ingest_and_calibration.md) |
| segmentation, visual hull, synthetic RGB | Phase 2, not built |
| `longitudinal`, `reporting`, `experimental` | empty placeholders |
| `experimental` (RGB visual-hull demo; `body_regions` head/trunk/arms/legs split) | experimental, all output `unreliable`/`provisional`, display only. Region split is a height/width heuristic (crotch falls back to a fixed 0.47 when legs touch); never used for measurement. `scan_review` adds recording warnings (untested thresholds) and rough slice lengths as % of body height (`unreliable`, `unavailable` when the outline has a gap). |
| web frontend / API / job runner | upload -> ingest -> visual-hull mesh job -> viewer works end to end (experimental, non-metric). **Stray Scanner .zip** -> unpacked derived copy (`<upload>/extracted/`, zip untouched) -> `rgbd` self-rotation TSDF job (optional empty-scene upload as background; without one, a 0.75 m subject ROI around the image centre; if no pauses are found, 48 evenly spaced frames) -> metric mesh in the viewer, labelled not validated, no measurements. **Body avatar** page: own scan mesh (display copy cleaned shape-preservingly by `experimental/display_mesh.py`: floor spill, loose debris, Taubin smoothing, decimation; deviation from the source checked, falls back to the untouched mesh if RMS > 2 mm or P95 > 4 mm; no hole filling) coloured per region, with **DEMO** body-composition readings (`frontend/src/demoData.js`); no real BCM values, no standard ranges, nothing derived from geometry. Auth, SQLite, longitudinal history not built |

## RGB-D tracked pipeline (`rgbd/tracked.py`, CLI `dt reconstruct-stray-tracked`), first real export 2026-11-26

Input: an unpacked Stray Scanner recording of the subject turning in front of a fixed iPad + an empty-scene
recording from the same rig. Not wired into the web job runner yet (a parallel texture-guided pipeline exists in
`rgbd/textured/`; which one the job uses is an open decision). Findings on the real export that shaped it:

| Finding (measured) | Consequence |
|---|---|
| iPad static: ARKit odometry moved 1.5 cm / 0.24 deg in 112 s | odometry/IMU give no subject pose; poses come from the body |
| rgb.mp4 HEVC CFR 60 fps, 6707 packets = 6707 odometry rows = 6707 depth PNGs | frame index = sync; OpenCV cannot seek, so RGB is decoded sequentially |
| depth 256x192 uint16 mm, confidence 0/1/2, depth outline matches RGB outline | depth is registered to RGB; intrinsics scaled 1920x1440 -> 256x192 |
| IMU accelerometer up agrees with the floor plane (1.5 deg) and the person's rotation axis (0.5-0.7 deg); up from the ARKit quaternion (`_up_camera`) is ~5 deg off | vertical = IMU gravity; ARKit-quaternion up only as a hint |
| shape-only registration (`rgbd.pose`) never closed the loop (48 deg) | RGB SIFT features lifted by confidence-2 depth, 4-DOF (yaw about vertical + xyz) RANSAC, multi-offset + loop-closure pose graph, convex Huber IRLS |
| Cauchy loss absorbed loop drift by breaking the chain at one weak view (22 deg jump) | Huber; outliers removed after the solve, never disconnecting a view |
| side views: texture is mostly the swinging arm -> yaw scatter ~+-5 deg | views with fast turns or contradicted poses are not fused; free-space vote with 5 cm margin |
| only subject pixels are fused, so moved arms/static strays leave ghosts | two-pass TSDF with free-space violation culling (removes geometry only; nothing filled or moved) |
| arms hung against the torso in some views | the largest closed contour at waist/hip height includes the arms: perimeters are NOT anatomical |

Outputs: `fused_mesh.ply/.obj/.glb` (metres, floor z = 0, +z up; GLB/OBJ carry display-only vertex colours),
`fused_points.ply`, `poses.json`, `quality_report.json`, `report.html` with previews, `run_log.json`. The report
includes a split-half check (interleaved halves fused separately). All measurements are `unreliable` until the
validation stages in CLAUDE.md are done on this rig.

## Open3D note

The legacy `ScalableTSDFVolume` returned an **empty mesh for every input** (even a flat plane) in
Open3D 0.20.0 on Windows / Python 3.11. The pipeline uses the sparse tensor API
`open3d.t.geometry.VoxelBlockGrid` instead. The backend and version are recorded in every `run_log.json`.
`UniformTSDFVolume` also worked in a quick check but needs a dense cube, which is expensive at body scale.

## Known TSDF characteristic (synthetic)

On noise-free cylinders the fused circumference is biased **outward** (about +0.8% at 4 mm voxel / 20 mm
truncation), growing with truncation distance and reduced by rejecting grazing-angle (low-confidence)
samples. It is a property of projective TSDF at silhouettes, reproduced on a virtual sensor. Behaviour on real
depth data must be measured on real objects before any correction is considered.
