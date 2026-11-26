"""Command line: dt validate | synth | reconstruct | migrate | serve | inspect-upload | reconstruct-rgbd |
demo-selfrotation | view-rgbd | reconstruct-stray"""

from __future__ import annotations

import argparse
import json
import sys

from .dataset.loader import load_session
from .dataset.migrate import MigrationError, migrate_session_copy
from .dataset.validate import validate_session
from .reconstruction.tsdf import TSDFParams, reconstruct_tsdf
from .synthetic.renderer import RenderConfig, render_session
from .synthetic.shapes import SHAPES


def _cmd_validate(args) -> int:
    rep = validate_session(args.session, check_checksums=not args.no_checksums)
    for issue in rep.issues:
        print(issue)
    print(f"{'OK' if rep.ok else 'INVALID'}: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    return 0 if rep.ok else 1


def _cmd_synth(args) -> int:
    body = SHAPES[args.shape]()
    cfg = RenderConfig(n_views=args.views, noise_sigma_m=args.noise_mm / 1000.0, dropout_prob=args.dropout,
                       angle_bias_deg=args.angle_bias, angle_noise_deg=args.angle_noise, seed=args.seed)
    root = render_session(body, args.out, cfg)
    print(f"wrote synthetic session: {root}")
    return 0


def _cmd_reconstruct(args) -> int:
    rep = validate_session(args.session)
    if not rep.ok:
        print("session failed validation; refusing to reconstruct:")
        for i in rep.errors:
            print(" ", i)
        return 1
    try:
        res = reconstruct_tsdf(load_session(args.session), TSDFParams(voxel_length_m=args.voxel_mm / 1000.0))
    except ValueError as e:
        print(f"cannot reconstruct: {e}")
        return 1
    print(f"run {res.run_id}: {len(res.mesh.triangles)} triangles -> {res.run_dir}")
    print(json.dumps({"quality_passed": res.quality["passed"]}))
    return 0


def _cmd_migrate(args) -> int:
    try:
        dst = migrate_session_copy(args.source, args.dest)
    except (MigrationError, FileExistsError) as e:
        print(f"migration failed: {e}")
        return 1
    print(f"wrote migrated copy: {dst} (source unchanged)")
    return 0


def _cmd_serve(args) -> int:
    import ipaddress

    import uvicorn

    from .api.app import create_app

    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback:
        print(f"WARNING: binding to {args.host} exposes body-scan uploads to the network (no authentication).")
    uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
    return 0


def _cmd_inspect_upload(args) -> int:
    from .rgbd.upload import UploadIntegrityError, UploadNotReady, inventory, open_finalized_upload

    try:
        up = open_finalized_upload(args.upload)
    except (UploadNotReady, UploadIntegrityError) as e:
        print(f"cannot inspect: {e}")
        return 1
    inv = inventory(up)
    text = json.dumps(inv, indent=2)
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        print(text)
    return 0


def _cmd_reconstruct_rgbd(args) -> int:
    from pathlib import Path

    from .rgbd.pipeline import run_self_rotation
    from .rgbd.upload import (NoAdapterError, UploadIntegrityError, UploadNotReady, find_adapter,
                              open_finalized_upload)

    try:
        up = open_finalized_upload(args.upload)
        bg_up = open_finalized_upload(args.background) if args.background else None
    except (UploadNotReady, UploadIntegrityError) as e:
        print(f"cannot use upload: {e}")
        return 1
    out = Path(args.out) / up.upload_id
    try:
        adapter = find_adapter(up)
        bg_adapter = find_adapter(bg_up) if bg_up else None
    except NoAdapterError as e:
        out.mkdir(parents=True, exist_ok=True)
        (out / "inventory.json").write_text(json.dumps(e.inventory, indent=2) + "\n", encoding="utf-8")
        print(str(e))
        print(f"inventory written to {out / 'inventory.json'} (original upload untouched)")
        return 2
    views = adapter.load_views(up)
    bg = bg_adapter.load_views(bg_up) if bg_adapter else None
    res = run_self_rotation(views, out, background_views=bg,
                            source={"upload_id": up.upload_id, "adapter": adapter.name,
                                    "background_upload_id": bg_up.upload_id if bg_up else None,
                                    "label": up.metadata.get("label", "")})
    print(f"status {res.status}: {res.log.get('status_note')}\nreport: {res.run_dir / 'report.html'}")
    return 0 if res.status != "fail" else 3


def _cmd_demo_selfrotation(args) -> int:
    import numpy as np

    from .geometry.transforms import invert
    from .rgbd.pipeline import run_self_rotation
    from .synthetic.rgbd_scene import SceneConfig, make_sequence

    cfg = SceneConfig(n_stops=args.stops, step_deg=360.0 / args.stops, noise_sigma_m=args.noise_mm / 1000,
                      dropout=args.dropout, frames_per_stop=3, transition_frames=4, seed=args.seed)
    seq = make_sequence(cfg)
    res = run_self_rotation(seq.views, args.out, background_views=seq.background_views,
                            source={"synthetic": True, "label": f"SYNTHETIC box, {args.stops} stops, "
                                                               f"{args.noise_mm} mm noise"})
    print(f"status {res.status}: {res.log.get('status_note')}")
    if res.pose is not None and res.pose.floor_from_body:
        g = np.array(res.log["floor_from_camera"]) @ seq.camera_from_floor
        errs = []
        for k, est in enumerate(res.pose.floor_from_body):
            truth = g @ seq.floor_from_body[k] @ invert(seq.floor_from_body[0]) @ invert(g)
            d = invert(truth) @ est
            errs.append((np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))), np.linalg.norm(d[:3, 3])))
        e = np.array(errs)
        print(f"pose error vs SYNTHETIC truth: rotation max {e[:, 0].max():.2f} deg, "
              f"position median {np.median(e[:, 1]) * 1000:.1f} mm, max {e[:, 1].max() * 1000:.1f} mm")
    print(f"report: {res.run_dir / 'report.html'}")
    return 0


def _cmd_analyze_video(args) -> int:
    from pathlib import Path

    from .rgbvideo.analyze import analyze_video

    src = Path(args.video)
    if src.is_dir():  # a finalized upload: use its single video file, verified against the manifest
        from .rgbd.upload import open_finalized_upload
        up = open_finalized_upload(src)
        vids = [f["path"] for f in up.files if Path(f["path"]).suffix.lower() in (".mov", ".mp4", ".m4v")]
        if len(vids) != 1:
            print(f"expected exactly one video in the upload, found {len(vids)}")
            return 1
        src = up.file(vids[0])
    out = Path(args.out) / src.stem
    r = analyze_video(src, out, label=args.label or src.name)
    print(f"{r['frames']} frames, {len(r['stops'])} stops; silhouettes flagged: "
          f"{sum(bool(s['silhouette']['flags']) for s in r['stops'])}")
    print("no 3-D reconstruction from this video; missing: " + "; ".join(r["missing_for_3d"]))
    print(f"report: {out / r['run_id'] / 'report.html'}")
    return 0


def _cmd_print_sheets(args) -> int:
    from .printsheets import write_sheets

    for f in write_sheets(args.out):
        print(f)
    print("Print the PDFs at 'Actual size' / 100 %, then measure the 100 mm bar and one marker with a ruler.")
    return 0


def _cmd_validation_report(args) -> int:
    from pathlib import Path

    from .validation.study import run_study

    summary = run_study(json.loads(Path(args.manifest).read_text(encoding="utf-8")), args.out)
    print(Path(args.out) / "validation_report.md")
    return 0 if summary["measurements"] else 1


def _cmd_reconstruct_stray_tracked(args) -> int:
    from dataclasses import replace

    from .rgbd.stray import StrayFormatError
    from .rgbd.tracked import TrackedParams, run_tracked

    p = replace(TrackedParams(), frame_step=args.frame_step, split_check=not args.no_split_check)
    try:
        res = run_tracked(args.recording, args.background, args.out, params=p,
                          source={"recording": str(args.recording), "background": str(args.background)},
                          progress=lambda s, f: print(f"  {s} {f:4.0%}", flush=True))
    except (StrayFormatError, FileExistsError, RuntimeError) as e:
        print(f"cannot reconstruct: {e}")
        return 1
    q = res.quality
    print(f"status {res.status}; mesh {q['mesh']['vertices']} vertices, {q['mesh']['holes']} holes; "
          f"depth residual p50 {q['depth_residual_m']['p50'] * 1000:.1f} mm")
    print(f"report: {res.run_dir / 'report.html'}")
    return 0 if res.status != "fail" else 3


def _cmd_reconstruct_stray(args) -> int:
    from pathlib import Path

    from .rgbd.stray import StrayFormatError
    from .rgbd.textured.compare_rgb import compare_with_rgb_only, run_rgb_only_baseline
    from .rgbd.textured.pipeline import reconstruct

    try:
        res = reconstruct(args.recording, args.out, progress=lambda s: print(f"  {s}", flush=True))
    except (StrayFormatError, FileExistsError) as e:
        print(f"cannot reconstruct: {e}")
        return 1
    print(f"status {res.status}: {res.report.get('status_note')}\nreport: {res.run_dir / 'report.html'}")
    other = args.rgb_only_run
    if res.mesh is not None and args.run_rgb_only:
        print("running the existing RGB-only pipeline (slow)")
        try:
            other = run_rgb_only_baseline(args.recording, Path(res.run_dir) / "rgb_only_baseline")
        except (ValueError, RuntimeError) as e:
            print(f"RGB-only pipeline failed on this recording: {e}")
    if res.mesh is not None and other:
        compare_with_rgb_only(res.run_dir, other)
        print(f"comparison: {res.run_dir / 'comparison_rgb_only.json'}")
    return 0 if res.status != "fail" else 3


def _cmd_view_rgbd(args) -> int:
    from pathlib import Path

    import open3d as o3d

    run = Path(args.run_dir)
    geoms = []
    mesh_p = run / ("mesh_coverage.ply" if args.coverage else "fused_mesh.ply")
    if mesh_p.is_file():
        m = o3d.io.read_triangle_mesh(str(mesh_p))
        m.compute_vertex_normals()
        geoms.append(m)
    if args.points and (run / "views_colored.ply").is_file():
        geoms.append(o3d.io.read_point_cloud(str(run / "views_colored.ply")))
    if not geoms:
        print(f"nothing to show in {run}")
        return 1
    o3d.visualization.draw_geometries(geoms, window_name=f"DIGITAL TRACK - {run.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dt")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate a scan session directory")
    v.add_argument("session")
    v.add_argument("--no-checksums", action="store_true")
    v.set_defaults(fn=_cmd_validate)

    s = sub.add_parser("synth", help="generate a synthetic session (data/synthetic/...)")
    s.add_argument("--shape", choices=sorted(SHAPES), default="cylinder")
    s.add_argument("--out", required=True)
    s.add_argument("--views", type=int, default=36)
    s.add_argument("--noise-mm", type=float, default=0.0)
    s.add_argument("--dropout", type=float, default=0.0)
    s.add_argument("--angle-bias", type=float, default=0.0)
    s.add_argument("--angle-noise", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_synth)

    r = sub.add_parser("reconstruct", help="TSDF-reconstruct a session into <session>/processing/")
    r.add_argument("session")
    r.add_argument("--voxel-mm", type=float, default=4.0)
    r.set_defaults(fn=_cmd_reconstruct)

    m = sub.add_parser("migrate", help="write an upgraded copy of an older-schema session (source untouched)")
    m.add_argument("source")
    m.add_argument("dest")
    m.set_defaults(fn=_cmd_migrate)

    w = sub.add_parser("serve", help="run the local upload web app (http://127.0.0.1:8000)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--data-dir", default=None, help="upload storage (default data/real/uploads, git-ignored)")
    w.set_defaults(fn=_cmd_serve)

    iu = sub.add_parser("inspect-upload", help="describe a finalized upload's files (generic; no format assumed)")
    iu.add_argument("upload", help="data/real/uploads/<upload_id>")
    iu.add_argument("--json", help="write the inventory to this file instead of printing it")
    iu.set_defaults(fn=_cmd_inspect_upload)

    rr = sub.add_parser("reconstruct-rgbd", help="self-rotation RGB-D reconstruction of a finalized upload")
    rr.add_argument("upload", help="data/real/uploads/<upload_id>")
    rr.add_argument("--background", help="finalized upload of the empty scene (recommended)")
    rr.add_argument("--out", default="data/real/processing", help="output root (git-ignored)")
    rr.set_defaults(fn=_cmd_reconstruct_rgbd)

    ds = sub.add_parser("demo-selfrotation", help="SYNTHETIC end-to-end self-rotation reconstruction with report")
    ds.add_argument("--out", default="data/synthetic/selfrot_demo")
    ds.add_argument("--stops", type=int, default=24)
    ds.add_argument("--noise-mm", type=float, default=3.0)
    ds.add_argument("--dropout", type=float, default=0.02)
    ds.add_argument("--seed", type=int, default=0)
    ds.set_defaults(fn=_cmd_demo_selfrotation)

    av = sub.add_parser("analyze-video", help="RGB-only video: stops, keyframes, silhouettes, report (no 3-D)")
    av.add_argument("video", help="a video file or a finalized upload folder (read-only)")
    av.add_argument("--out", default="data/real/processing", help="output root (git-ignored)")
    av.add_argument("--label", default="")
    av.set_defaults(fn=_cmd_analyze_video)

    ps = sub.add_parser("print-sheets", help="A4 marker-disc and box-texture sheets (PDF + PNG) for recording")
    ps.add_argument("--out", default="data/print_sheets")
    ps.set_defaults(fn=_cmd_print_sheets)

    vr = sub.add_parser("view-rgbd", help="open a reconstruction run in an interactive 3-D window")
    vr.add_argument("run_dir")
    vr.add_argument("--points", action="store_true", help="also show every view's points, one colour per view")
    vr.add_argument("--coverage", action="store_true", help="show the mesh coloured by observation count")
    vr.set_defaults(fn=_cmd_view_rgbd)

    st = sub.add_parser("reconstruct-stray", help="texture-guided RGB-D reconstruction of an unpacked Stray Scanner "
                                                  "recording, with quality report")
    st.add_argument("recording", help="folder with rgb.mp4, depth/, confidence/, odometry.csv")
    st.add_argument("--out", default="data/real/processing", help="output root (git-ignored)")
    st.add_argument("--run-rgb-only", action="store_true", help="also run the RGB-only pipeline and compare")
    st.add_argument("--rgb-only-run", help="compare with an existing RGB-only run directory instead")
    st.set_defaults(fn=_cmd_reconstruct_stray)

    sk = sub.add_parser("reconstruct-stray-tracked",
                        help="RGB-D body mesh from a Stray Scanner recording + empty-scene recording (fixed iPad, "
                             "subject turning): feature pose graph, filtered two-pass TSDF, PLY/OBJ/GLB, quality report")
    sk.add_argument("recording", help="unpacked Stray Scanner folder of the subject (rgb.mp4, depth/, confidence/, ...)")
    sk.add_argument("--background", required=True, help="unpacked Stray Scanner folder of the empty scene, same rig")
    sk.add_argument("--out", default="data/real/processing", help="output root (git-ignored)")
    sk.add_argument("--frame-step", type=int, default=6, help="use every n-th recorded frame (default 6: 60 -> 10 fps)")
    sk.add_argument("--no-split-check", action="store_true", help="skip the split-half repeatability check")
    sk.set_defaults(fn=_cmd_reconstruct_stray_tracked)

    vr2 = sub.add_parser("validation-report", help="accuracy / repeatability report from several scans + tape values")
    vr2.add_argument("manifest", help="JSON manifest (see digitaltrack/validation/study.py)")
    vr2.add_argument("--out", required=True)
    vr2.set_defaults(fn=_cmd_validation_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
