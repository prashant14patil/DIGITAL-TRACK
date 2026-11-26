"""Texture-guided RGB-D reconstruction of a self-rotating subject from a Stray Scanner recording.

A second, independent pipeline next to `rgbd.pipeline` (which is unchanged): subject poses come from RGB texture
correspondences lifted with LiDAR depth, instead of depth shape alone. Not validated against any physical
reference; it produces a metric-unit mesh and a quality report, no body measurements.
"""
