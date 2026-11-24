"""Calibration scaffolding.

Implemented:
* `intrinsics`: camera intrinsics from chessboard images -> schema-0.2.0 `camera_calibration.json`.
* `scale`: scale factor and residuals from measured vs known reference lengths (library only).
* `floor`: floor-plane fit (RANSAC + least squares) and a floor-aligned frame (library only).

`scale` and `floor` results are NOT persisted: dataset schema 0.2.0 has no field for them. Persisting them is a
later, approved schema change. Not implemented: turntable/self-rotation pose estimation (open item).
"""
