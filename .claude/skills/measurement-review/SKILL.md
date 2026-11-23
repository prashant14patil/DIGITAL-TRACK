---
name: measurement-review
description: Review reconstruction or anthropometric measurement code against this project's geometry-first rules. Use after writing or changing fusion, calibration, or measurement code.
---

Review the changed code (or the path given) and flag violations of:

- Raw source video preserved byte-for-byte; extracted frames lossless; raw depth/confidence (depth branch) preserved; smoothing never overwrites raw.
- Intrinsics match the image size and video mode actually processed; no silent alignment assumptions between image sources.
- Metric scale comes from a known-size reference; scale residual/uncertainty propagated to every measurement.
- Visual-hull (silhouette carving) results: concavity-affected regions/measurements flagged `estimated`, never `measured`.
- Display-only assets (decimated meshes, preview videos) never used for measurement.
- Primary path runs on CPU only; CUDA-only steps (e.g. COLMAP dense) stay optional.
- Turntable transform uses calibrated axis/center and measured angle; rotation sign verified; ICP only for small residuals with a rejection threshold.
- Filtering/smoothing/hole-filling quantified and logged; filled geometry marked as estimated.
- Circumference: horizontal/perpendicular plane-mesh slices, closed-contour perimeter, handling of multiple contours defined; no global max/min over the body; no convex-hull shortcut without justification; full circumference-vs-height profile stored.
- Height uses a ground/platform plane, not raw mesh extent.
- Volume only for sufficiently complete, documented-closure meshes.
- Each output has units, definition version, algorithm version, quality status, measured/estimated flag.
- Quality checks and failure flags exist; poor scans rejected.
- Synthetic-shape unit tests with known dimensions cover the change.
- No body-fat/BCM/internal-composition inference.

Report findings by severity with file:line references. Don't rewrite working code; propose minimal fixes.
