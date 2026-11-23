# Validation protocol

Authoritative plan: "Validation plan" section of the project context in `CLAUDE.md` (Stages 1-4).

## What Phase 0 has and has not validated

* Validated (virtual sensor only): coordinate/sign conventions, depth back-projection, TSDF fusion and
  slice perimeter against analytic cylinders/ellipses/a procedural torso, and sensitivity to noise,
  dropout, angle error and calibration error. Run `tools/synthetic_demo.py`.
* Quality gate (QC v0.2) detects lateral turntable-centre errors, sideways axis tilt, larger tilt toward the
  camera, and a wrong rotation direction. It is **blind** to calibration errors along the camera line of sight,
  which change measured size without making views disagree. Closing that needs a known-dimension reference
  object scanned on the rig (Stage 2). Demo output before/after the QC fix: `docs/evidence/`.
* **Not validated:** anything about real iPad video, camera calibration, the real turntable, RGB
  reconstruction, mannequins or humans. No accuracy claim exists for the real system.

Synthetic thresholds in the tests are regression bounds, not product targets.
