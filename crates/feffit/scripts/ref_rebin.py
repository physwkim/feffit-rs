#!/usr/bin/env python
"""Reference values for the rebin_xafs / sort_xafs port.

Runs larch.xafs.rebin_xafs on the Cu foil mu(E) for the boxcar and centroid
methods (e0 = 8979) and dumps the full rebinned energy / mu / delta_mu arrays,
plus a sort_xafs round-trip checksum.

Writes crates/feffit/tests/data/ref_rebin.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/feffit/scripts/ref_rebin.py
"""
import os
import numpy as np
from larch import Group
from larch.xafs.rebin_xafs import rebin_xafs, sort_xafs

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
E0 = 8979.0

out = os.path.join(data, "ref_rebin.txt")
with open(out, "w") as f:
    f.write("# rebin_xafs reference (larch)\n")
    f.write(f"e0 {E0!r}\n")
    for method in ("boxcar", "centroid", "spline"):
        g = Group(energy=energy.copy(), mu=mu.copy())
        rebin_xafs(g, e0=E0, method=method)
        rb = g.rebinned
        f.write(f"# method {method}\n")
        f.write(f"{method}_n {len(rb.energy)!r}\n")
        for i in range(len(rb.energy)):
            dm = rb.delta_mu[i]
            f.write(f"{method} {i} {rb.energy[i]!r} {rb.mu[i]!r} {dm!r}\n")
    # sort_xafs round-trip on a shuffled copy
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(energy))
    g2 = Group(energy=energy[idx].copy(), mu=mu[idx].copy())
    sort_xafs(g2, overwrite=True)
    f.write(f"sort_n {len(g2.energy)!r}\n")
    f.write(f"sort_e_sum {float(np.sum(g2.energy))!r}\n")
    f.write(f"sort_mu_sum {float(np.sum(g2.mu))!r}\n")
    f.write(f"sort_e0 {g2.energy[0]!r}\n")
    f.write(f"sort_elast {g2.energy[-1]!r}\n")
print(f"wrote {out}")
