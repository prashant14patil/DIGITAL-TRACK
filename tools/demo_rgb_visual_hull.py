"""EXPERIMENTAL DEMO (CLI wrapper): approximate, NON-METRIC body mesh from one Camera-app RGB video session.

The algorithm lives in `digitaltrack.experimental.rgb_visual_hull` (shared with the web app). Options and
outputs are unchanged. Everything written is labelled `unreliable`; no measurements are computed.

Usage (repo root):
    backend\\.venv\\Scripts\\python.exe tools\\demo_rgb_visual_hull.py data\\real\\sessions\\ses_demo_vid_0003
"""

from __future__ import annotations

import argparse
import json
import sys

from digitaltrack.experimental.rgb_visual_hull import run_visual_hull


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--voxel-px", type=float, default=3.0, help="voxel edge in (downscaled) pixels")
    ap.add_argument("--vote-fraction", type=float, default=0.9)
    ap.add_argument("--reuse-masks", default=None,
                    help="processing/<run_id> of an earlier run: reuse its silhouettes instead of re-segmenting")
    ap.add_argument("--assume-height-m", type=float, default=None,
                    help="also write a mesh scaled to this ASSUMED standing height (not a measurement)")
    args = ap.parse_args()

    last = {"stage": None}

    def progress(stage: str, frac: float) -> None:
        if stage != last["stage"] or frac >= 1.0 or int(frac * 100) % 20 == 0:
            print(f"  {stage} {frac:.0%}", flush=True)
            last["stage"] = stage

    try:
        out, report = run_visual_hull(args.session, downscale=args.downscale, voxel_px=args.voxel_px,
                                      vote_fraction=args.vote_fraction, reuse_masks=args.reuse_masks,
                                      assume_height_m=args.assume_height_m, progress=progress)
    except (ValueError, RuntimeError) as e:
        print(f"failed: {e}")
        return 1
    print(json.dumps({k: report[k] for k in ("run_id", "status", "mesh", "runtime_s")}, indent=1))
    print("angle anchors:", [(a["time_s"], a["angle_deg"]) for a in report["angles"]["anchors"]],
          "sign", report["angles"]["direction_sign"])
    print("output:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
