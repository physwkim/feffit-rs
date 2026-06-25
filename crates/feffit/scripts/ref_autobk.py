#!/usr/bin/env python
"""Reference values for the AUTOBK background-removal port.

Runs larch.xafs.autobk on the Cu foil mu(E) with default parameters (it runs
pre_edge internally) and dumps scalars (ek0, edge_step, rbkg, nspl, irbkg, grid
indices, kmax) plus sampled arrays (k, chi, bkg, chie, and the pre-fit init_bkg
/ init_chi which are deterministic).

Writes crates/feffit/tests/data/ref_autobk.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/feffit/scripts/ref_autobk.py
"""
import os
import numpy as np
from larch import Group
from larch.xafs.autobk import autobk

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
g = Group(energy=energy, mu=mu)
autobk(g)  # default parameters; runs pre_edge internally

d = g.autobk_details
out = os.path.join(data, "ref_autobk.txt")
with open(out, "w") as f:
    f.write("# autobk reference (larch, default params)\n")
    f.write(f"ek0 {g.ek0!r}\n")
    f.write(f"edge_step {g.edge_step!r}\n")
    f.write(f"rbkg {g.rbkg!r}\n")
    f.write(f"nspl {d.nspl!r}\n")
    f.write(f"irbkg {d.irbkg!r}\n")
    f.write(f"iek0 {d.iek0!r}\n")
    f.write(f"iemax {d.iemax!r}\n")
    f.write(f"kmax {d.kmax!r}\n")
    f.write(f"nk {len(g.k)!r}\n")
    f.write(f"nbkg {len(g.bkg)!r}\n")
    # sampled arrays: name index value
    for name, arr in (("k", g.k), ("chi", g.chi), ("bkg", g.bkg),
                      ("chie", g.chie), ("initbkg", d.init_bkg),
                      ("initchi", d.init_chi)):
        for i in range(0, len(arr), 20):
            f.write(f"{name} {i} {arr[i]!r}\n")
print(f"wrote {out}: ek0={g.ek0}, edge_step={g.edge_step}, "
      f"nspl={d.nspl}, irbkg={d.irbkg}, nk={len(g.k)}")
