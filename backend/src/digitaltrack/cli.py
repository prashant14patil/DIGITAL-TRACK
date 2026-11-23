"""Command line: dt validate | synth | reconstruct | migrate"""

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

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
