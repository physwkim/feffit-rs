#!/usr/bin/env python
"""Reference values for the diffkk port.

Runs larch's diffkk on the Cu foil (z=29, K edge), then `.kk()` with the default
scalar (sequential-sum) Maclaurin-series KK kernel. Dumps the tabulated Chantler
f1/f2 (xraydb, computed internally by larch) at full resolution so the Rust port
can be fed the identical arrays — isolating the diffKK algorithm from the
atomic-data table source for bit-exact verification.

Writes crates/xasproc/tests/data/ref_diffkk.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/xasproc/scripts/ref_diffkk.py
"""
import os
import numpy as np
from larch.xafs.diffkk import diffkk

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = os.path.join(here, "tests", "data")


def load_xmu(path):
    e, m = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            e.append(float(p[0]))
            m.append(float(p[1]))
    return np.array(e), np.array(m)


energy, mu = load_xmu(os.path.join(data, "cu.xmu"))

dkk = diffkk(energy, mu, z=29, edge="K")
dkk.kk(z=29, edge="K")  # default how='scalar' -> kkmclr_sca

out = os.path.join(data, "ref_diffkk.txt")
with open(out, "w") as f:
    f.write("# diffkk reference (larch, z=29 K edge, how='scalar')\n")
    f.write(f"npts {len(energy)!r}\n")
    f.write(f"e0 {dkk.e0!r}\n")
    f.write(f"grid_npts {len(dkk.grid)!r}\n")

    # full-resolution f1 / f2 — fed identically into the Rust port
    for i in range(len(energy)):
        f.write(f"f2 {i} {dkk.f2[i]!r}\n")
    for i in range(len(energy)):
        f.write(f"f1 {i} {dkk.f1[i]!r}\n")

    # outputs (sparse)
    for tag, arr in (("fpp", dkk.fpp), ("fp", dkk.fp)):
        for i in range(0, len(arr), 10):
            f.write(f"{tag} {i} {arr[i]!r}\n")

print(f"wrote {out} (grid_npts={len(dkk.grid)})")
