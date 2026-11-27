# RGB-D recording checklist (first real depth recordings)

Goal: the first real recordings for the depth pipeline (`dt reconstruct-rgbd`). **Objects first, no person.**
Stray Scanner is the provisional app; nothing about its export format is assumed. These recordings exist so the
format can be inspected and the importer written against the real files.

Print sheets: `backend\.venv\Scripts\dt.exe print-sheets` writes PDFs to `data\print_sheets\`.

## Before you start

- [ ] Install the depth app (Stray Scanner). Screenshot **every** settings screen you use. Do not change settings
      between recordings.
- [ ] iPad on a rigid mount (tripod plus clamp), held in a fixed orientation. Do not touch it during a recording.
- [ ] Note the camera height and the distance to where the object will stand, in cm.
- [ ] Even lighting, nothing moving in view (people, doors, screens).
- [ ] Print `box_texture_*.pdf` at **Actual size / 100 %**, then cover every face of a cardboard box. Measure the box
      **3 times** per edge (length, width, height) with a tape or ruler, in mm.
- [ ] Optional: print `marker_disc.pdf` at Actual size. Measure the 100 mm bar and one marker's black square. Glue the
      6 markers flat and evenly spaced around the rim of a rigid disc (plywood or stiff board, at least 50 cm across).
- [ ] Keep the floor around the object visible in the frame (the pipeline measures the floor plane).

## Recordings (upload each one separately)

| # | What | How | Why |
|---|---|---|---|
| R0 | Empty scene | Nothing in view where the object will stand, 10 s, iPad still | Background and floor plane |
| R1 | Flat wall or floor | Point at a flat surface at scan distance, 10 s still | Depth noise and bias |
| R2 | Box, still | Box on the floor at the scan spot, 10 s, no turning | Single-view size check |
| R4 | Box on the marker disc, turned | Turn by hand about **15 deg** per step, **pause 3 s** at each stop, full turn plus one extra step. Remove your hand from view during each pause. | Pose method compared against marker reference |
| R3 | Box on the floor, turned | Same as R4 but without the disc (only if there is no disc) | Pose method without equipment |

- Repeat R2 and R4 (or R3) **3 times**, putting the box back each time.
- A person comes only after the box works. Use the person protocol below.

## Person recording, protocol v2 (2026-11-26)

Why v2: the first person scan (2026-11-25, 26 stops) failed the pose check. The estimated steps added up to only
**289 deg instead of 360 deg** (loop-closure error 48 deg), and the turn angle was the weakest-determined value on
almost every step. A body is close to a cylinder, so its turn is fixed mainly by the arms, the feet and the head.
If those move between stops, or a step is too small to measure, the angle drifts. Earlier attempts also had no
detectable pauses, and some had the floor seen as a wall (iPad tilted or held in the wrong orientation).

**iPad**
- [ ] Landscape, on a tripod, lens at about **100-120 cm** height, **2.0-2.5 m** from the standing spot, tilted
      slightly down. The whole body, **head to feet plus about 50 cm of floor** in front, is in frame at every stop.
- [ ] Same orientation for every recording. Do not touch the iPad between the empty-scene clip and the person clip.

**Room**
- [ ] Record **R0 empty scene** first (10 s, nobody in view). It is used as the background, and without it the
      subject is guessed from the image centre.
- [ ] Mark the standing spot on the floor with tape. Mark the **foot positions** too (feet about 20 cm apart).
- [ ] Nothing else in view within about 1 m of the person: no chair, bag or wall close behind.

**Person**
- [ ] Close-fitting, non-shiny clothing; hair tied up; no loose sleeves.
- [ ] **A-pose: arms straight, about 30 deg away from the body, palms facing the legs.** The arms must not touch the
      body and **must stay in exactly the same pose for the whole turn**. They are what fixes the turn angle.
- [ ] Legs apart (the tape marks), head neutral, look straight ahead, breathe calmly, no talking.

**Turning**
- [ ] **24 stops of about 15 deg** each (a full turn of 360 deg). Turn with small shuffles of the feet and keep the
      upper body and arms frozen. Never turn in steps larger than about 20 deg.
- [ ] **Hold perfectly still for 3 s at each stop** (count "one-thousand-one ... three").
- [ ] **Start facing the iPad** (left and right limbs are named from this), and finish facing the iPad exactly as you started, and hold still there for 3 s too (this closes the loop).
- [ ] Someone else can call out "turn" / "hold"; they stay out of the frame.
- [ ] Record **3 repeat scans**, stepping off and back on the marks between them.

**Tape measurements of the person** (the reference for the validation report; take them right after the scans, same
clothing). For each: measure **3 times**, write all readings in mm, and the **height above the floor** of the tape:
- [ ] Height (stand against a wall, flat object on the head) — floor to top of head, hair pressed.
- [ ] Chest — tape horizontal at the fullest part of the chest, under the arms; note its height above the floor.
- [ ] Waist — horizontal at the narrowest part between ribs and hips; note its height.
- [ ] Hip — horizontal at the widest part of the buttocks; note its height.
- [ ] Thigh — horizontal, about 3 cm below the crotch, one leg; note its height and which leg.
Enter these on the upload page (name, readings, tape height). The comparison uses the tape height when given
(torso only, arms excluded at that height) and the landmark definitions otherwise.

**Before leaving the room**: upload the person clip and the empty-scene clip, then press **Check recording** with the
empty scene selected (seconds, no mesh). Besides pauses / floor / full turn it now checks what decides whether the
body can be MEASURED: distance (mm per depth pixel), whole body in the picture, thighs apart, arms clear of the body,
arms held still, and turn duration. Re-record while you are still there if any of these warns.

**After uploading**, check the result page: status must be `pass` or `warn`, every view placed, loop closure within
the limits. If it fails, note which check failed and change one thing at a time.

## Transfer and upload

- [ ] Copy each recording's export folder from the iPad **unchanged**. Do not open, rename or re-save files inside it.
- [ ] Start the site (`backend\.venv\Scripts\dt.exe serve`), open the site (e.g. http://127.0.0.1:8765) and upload each folder with
      **Choose folder…**. Pick the matching recording type (R0 to R5) and enter the measured box sizes, the camera
      distance and height, the step angle, the number of stops and the pause length.

## What happens next

1. `dt inspect-upload data\real\uploads\<id>` describes the files without assuming a format.
2. The importer is written against those real files (depth units, resolution, intrinsics, timing are checked, not
   assumed).
3. The web job (or `dt reconstruct-stray-tracked <recording> --background <empty scene>`) builds the metric mesh,
   the body landmarks/measurements and the quality report.
4. Box dimensions and body measurements are compared with your tape values on the result page.
5. With several people and 3 repeat scans each: `dt validation-report <manifest.json> --out <folder>` gives MAE,
   bias, Bland-Altman limits, repeatability and the smallest detectable change. Until then, no accuracy is claimed.
