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

## `body-provisional-2` (algorithm `body-slices-0.2`, `measurement/body.py`) — PROPOSED, awaiting review

Changes from `body-provisional-1` (do not compare across versions): chest window is 2-10 cm (was 0-8 cm) below
arm separation (the top slices clipped the shoulders: synthetic chest repeatability Sw 13.7 -> 3.5 mm); perimeters
are measured on a smoothed outline (below); levels are chosen on a 5-slice running median of the profile.

For metric body meshes in the body frame (floor plane z = 0, +z up), e.g. `rgbd.tracked` output. Every value is
`unreliable` until validated against tape on this rig; `unavailable` when the geometry does not allow it.

* **Slice parts**: horizontal slices every 1 cm; the slice's surface points are rasterised (5 mm) and split into
  connected parts (parts closer than ~1 cm merge; parts with < 3 % of the slice's points are ignored). The
  **torso part** is the part nearest the body axis (median x, y of 30-70 % of the height).
* **Part circumference**: the largest closed polygon of the exact mesh section belonging to the part, resampled every
  2 mm and low-passed along the outline (Gaussian, sigma 10 mm, circular), perimeter with a first-order shrink
  correction exp(sigma^2 / 2R^2), R = perimeter / 2 pi. Reason: a tape follows a smooth path, while surface noise makes
  the raw outline zig-zag (synthetic: +19 % at 1 mm, +56 % at 2 mm noise). Synthetic error after smoothing: < 0.5 mm
  noise-free, 2-4 mm at 2 mm noise. `raw_outline_m` is kept as a diagnostic. If the polygon covers < 60 % of the
  part's filled area, the outline is open (a hole): `unavailable`.
  `convex_hull_m` (tape-like length bridging concavities) is reported alongside as a diagnostic.
* **head_top_z**: 99.95th percentile of surface height within 0.25 m of the body axis (robust to spikes; includes
  hair). **height_floor_to_head_top** = head_top_z.
* **neck_z**: smallest lateral width of the torso part between 78 % and 95 % of the top height.
* **arm_separation_z**: highest slice below the neck where separate parts exist on BOTH lateral sides of the torso.
  With arms hanging close this is below the true axilla.
* **crotch_z**: lowest height at which two parts separated laterally by > 8 cm (the legs) become one for at least
  3 consecutive slices. Flagged when outside 40-52 % of the top height (thighs touching in the mesh bias it low).
* **chest_circumference**: largest torso-part circumference 2-10 cm below arm_separation_z.
* **waist_circumference**: smallest torso-part circumference within 35-75 % of crotch_z .. arm_separation_z.
* **hip_circumference**: largest torso-part circumference within 0-35 % of crotch_z .. arm_separation_z.
* **thigh_circumference_side_neg / _pos**: each leg part 3 cm below crotch_z (left/right not assigned).
* **upper_arm_circumference**: `unavailable` (needs slices perpendicular to the arm axis).
* Floor rim (`trim_floor_rim`): geometry below 8 cm outside the feet footprint (8-14 cm band, +3 cm) is removed
  before measuring; logged.

Synthetic study (tools/synthetic_validation_study.py; 6 figures x 3 simulated scans, 2 mm noise, 0.3 % scale SD):
chest/waist/hip MAE 3.1-3.8 mm, within-subject SD 3.4-4.0 mm. Measurement engine on clean geometry only: not iPad
accuracy.
Synthetic check (tests/unit/test_body_measurements.py): elliptic torso with hanging arms 10 cm away, torso values
within 0.5 % of the analytic ellipse perimeter, thighs within 4 mm, landmarks within 1-2 cm.
First real scan (2026-11-25, loose shirt, arms partly against the body): crotch flagged (0.38 of height), thighs
unavailable (holes), torso values ~1.03-1.08 m, unvalidated.
