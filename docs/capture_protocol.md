# Capture protocol (draft, protocol_version `capture-0.1-draft`)

Recording with the iPad's built-in Camera app. Not yet validated; settings below are starting points to be
checked on the real rig. Record every deviation in the session metadata.

## Rig

- iPad on a rigid fixed mount. It must not move between the calibration clip and the subject clip.
- Camera height and camera-to-platform distance fixed and written down (cm) per rig ID.
- Platform level; rotation axis vertical. Plain, static, non-reflective background; even, constant lighting.
- Turntable angle mechanism: **to be decided** (pending: motorised or manually turned). Planned: printed
  ArUco markers of known size on the platform rim for angle, axis and scale.

## Camera app settings (one fixed video mode per calibration)

- Same lens (1x main camera) and no zoom changes.
- Same resolution and frame rate every time (for example 4K at 30 fps); record which.
- HDR video off (Settings > Camera > Record Video). Note the format setting (High Efficiency/HEVC or Most
  Compatible/H.264).
- Lock focus and exposure (press and hold until "AE/AF LOCK") on the subject's position before recording.
- Note whether any stabilization setting is enabled; it may change image geometry between frames.
- Location tagging off if possible (video files can carry GPS metadata).

## Per-session clips

1. **Calibration clip** (when the rig or video mode changed, or per calibration policy): ChArUco board moved
   slowly through the field of view at the subject distance, many orientations, covering the image corners.
2. **Empty-background clip**: 3-5 s of the empty platform (background plate for segmentation).
3. **Subject clip**: one full turn plus a small overlap, steady slow rotation.

## Subject

- Centred, feet placed consistently, arms slightly away from the torso, legs slightly apart, neutral head.
- Close-fitting, non-reflective clothing. Measurements describe the clothed surface, not skin.
- Hold still for the whole turn; no talking. Breathing normally; note anything unusual.

## Transfer

Copy the original file from the iPad without re-encoding (for example AirDrop, Files, or USB "Keep
Originals"). Do not export through apps that compress or edit the video.
