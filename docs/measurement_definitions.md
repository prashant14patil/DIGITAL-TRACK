# Measurement definitions (versioned)

Current version: `provisional-0` (algorithm `basic-slice-0.1`).

Only one provisional definition exists, used to check reconstruction against synthetic shapes:

* **circumference_at_z**: perimeter of the largest closed contour (by area) of the horizontal section of
  the mesh at body-frame height `z`. Multiple contours or open polylines downgrade status to `warn`;
  no contour yields `unavailable` (no number is invented).

Not defined yet (Phase 3): ground plane, body height, chest/waist/abdomen/hip levels and landmarks, limb
circumferences, volume. These need explicit anatomical definitions reviewed before implementation.
Every output carries units (metres), definition version, algorithm version, source (`measured`),
and quality status.
