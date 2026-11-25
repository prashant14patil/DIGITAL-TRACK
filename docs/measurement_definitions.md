# Measurement definitions (versioned)

Current version: `provisional-1` (algorithm `basic-slice-0.2`).

`provisional-0` / `basic-slice-0.1` could report a small closed fragment as `pass` when the main outline
had a gap (the open outline was silently dropped). Do not compare results across these versions.

Only one provisional definition exists, used to check reconstruction against synthetic shapes:

* **circumference_at_z**: perimeter of the largest closed contour (by area) of the horizontal section of
  the mesh at body-frame height `z`. Multiple contours or open polylines downgrade status to `warn`;
  no closed contour, or an open outline at least as long as the largest closed one, yields `unavailable`
  (no number is invented).

Not defined yet (Phase 3): ground plane, body height, chest/waist/abdomen/hip levels and landmarks, limb
circumferences, volume. These need explicit anatomical definitions reviewed before implementation.
Every output carries units (metres), definition version, algorithm version, source (`measured`),
and quality status.
