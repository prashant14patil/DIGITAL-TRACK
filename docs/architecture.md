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
| `video` (probe, frame extraction) | Phase 1, not built |
| `calibration` (intrinsics, markers) | Phase 1, empty placeholder |
| segmentation, visual hull, synthetic RGB | Phase 2, not built |
| `longitudinal`, `reporting`, `experimental` | empty placeholders |
| web frontend / API / worker | Phase 3, not started |

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
