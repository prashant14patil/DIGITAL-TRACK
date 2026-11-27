# Template body for the Body avatar (display only)

`makehuman_base.obj`: the MakeHuman base mesh, downloaded 2026-11-27 from
https://github.com/makehumancommunity/makehuman (makehuman/data/3dobjs/base.obj), unchanged.

License: **CC0 1.0 Universal** (public domain dedication). The file header states it was explicitly released as CC0 in
September 2020; the repository's LICENSE.md lists the assets as CC0 1.0 Universal.

Used by `experimental/template_fit.py` to show a clean body fitted to a person's scan. Nothing is measured from it.

Shape targets (same source and license, CC0, `makehuman/data/targets/macrodetails/`): `african-male-young.target`,
`caucasian-male-young.target`, `asian-male-young.target` (averaged: a young male), and
`universal-male-young-maxmuscle-averageweight.target` (applied at 50 %: athletic build). Format: `vertex dx dy dz`
offsets in the base mesh's units (decimetres).
