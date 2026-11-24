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
- A person comes only after the box works. Same protocol, A-pose, 15 deg steps, 3 s pauses.

## Transfer and upload

- [ ] Copy each recording's export folder from the iPad **unchanged**. Do not open, rename or re-save files inside it.
- [ ] Start the site (`backend\.venv\Scripts\dt.exe serve`), open http://127.0.0.1:8000 and upload each folder with
      **Choose folder…**. Pick the matching recording type (R0 to R5) and enter the measured box sizes, the camera
      distance and height, the step angle, the number of stops and the pause length.

## What happens next

1. `dt inspect-upload data\real\uploads\<id>` describes the files without assuming a format.
2. The importer is written against those real files (depth units, resolution, intrinsics, timing are checked, not
   assumed).
3. `dt reconstruct-rgbd <R4 or R3 upload> --background <R0 upload>` builds the mesh and the inspection report.
4. Box dimensions from the mesh are compared with your tape measurements. That is the first real accuracy evidence.
   Until then, no accuracy is claimed.
