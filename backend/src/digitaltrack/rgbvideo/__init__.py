"""RGB-only fallback route: analysis of an ordinary Camera-app video of a person turning in place.

Implemented: probe + checksum, motion timeline, stop detection, sharpest keyframe per stop (saved losslessly),
person silhouette per keyframe (classical GrabCut, quality-flagged), HTML report.
NOT implemented (and not faked): per-view turn angles, metric scale, camera calibration, 3-D reconstruction.
Those need markers / a known-size reference / a calibration recording, or a depth recording (primary route).
"""
