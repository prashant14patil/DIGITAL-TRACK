"""RGB-D pipeline for a fixed camera and a subject that turns in place (manual self-rotation).

Format-independent: everything here works on `RGBDView` objects (depth in metres + the intrinsics of the DEPTH
image). Producing views from a real capture export is the job of an export adapter (`rgbd.upload`); none is
registered until a real export has been inspected, so real uploads stop with a clear "blocked" error.

Pipeline: [stop detection] -> floor plane -> subject segmentation -> pairwise 3-DOF registration (FPFH + RANSAC,
no ICP) -> SE(2) pose graph with loop closure -> pose QC -> TSDF fusion with the estimated poses -> visual outputs.
"""
