#!/usr/bin/env python
"""Reference values for the mback / mback_norm port.

Runs larch's mback (return_f1=True) and mback_norm on the Cu foil (z=29, K edge).
Dumps the tabulated Chantler f2/f1 (xraydb, computed internally by larch) at full
resolution so the Rust port can be fed the *identical* arrays — this isolates the
MBACK algorithm from the atomic-data table source and lets the port be verified
bit-exact regardless of which library supplies f1/f2.

Writes crates/xasproc/tests/data/ref_mback.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/xasproc/scripts/ref_mback.py
"""
import os
import numpy as np
from larch import Group
from larch.xafs.mback import mback, mback_norm

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

# --- mback (default fit_erfc=False), with f1 returned ---
g = Group(energy=energy.copy(), mu=mu.copy())
mback(g, z=29, edge="K", return_f1=True)
mb = g.mback_details.params

# --- mback_norm (default) on a fresh group ---
g2 = Group(energy=energy.copy(), mu=mu.copy())
mback_norm(g2, z=29, edge="K")
mn = g2.mback_params

out = os.path.join(data, "ref_mback.txt")
with open(out, "w") as f:
    f.write("# mback / mback_norm reference (larch, z=29 K edge)\n")
    f.write(f"npts {len(energy)!r}\n")

    # full-resolution f2 / f1 — fed identically into the Rust port
    for i in range(len(energy)):
        f.write(f"f2 {i} {g.f2[i]!r}\n")
    for i in range(len(energy)):
        f.write(f"f1 {i} {g.f1[i]!r}\n")

    # mback scalars + fit params
    f.write(f"mb_e0 {g.e0!r}\n")
    f.write(f"mb_edge_step {g.edge_step!r}\n")
    f.write(f"mb_s {mb['s']!r}\n")
    for i in range(0, 4):
        f.write(f"mb_c{i} {mb['c%d' % i]!r}\n")

    # mback arrays (sparse)
    for tag, arr in (("mb_fpp", g.fpp),
                     ("mb_norm", g.norm),
                     ("mb_normfunc", g.mback_details.norm_function)):
        for i in range(0, len(arr), 10):
            f.write(f"{tag} {i} {arr[i]!r}\n")

    # mback_norm scalars + fit params
    f.write(f"mn_edge_step {g2.edge_step!r}\n")
    f.write(f"mn_edge_step_poly {g2.edge_step_poly!r}\n")
    f.write(f"mn_slope {mn.fit_params['slope']!r}\n")
    f.write(f"mn_offset {mn.fit_params['offset']!r}\n")
    f.write(f"mn_scale {mn.fit_params['scale']!r}\n")

    # mback_norm arrays (sparse)
    for tag, arr in (("mn_norm", g2.norm),
                     ("mn_mback_mu", g2.mback_mu),
                     ("mn_model", mn.model),
                     ("mn_weights", mn.fit_weights)):
        for i in range(0, len(arr), 10):
            f.write(f"{tag} {i} {arr[i]!r}\n")

print(f"wrote {out}")
