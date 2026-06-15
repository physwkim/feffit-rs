#!/usr/bin/env python
"""Reference values for the pre_edge / normalization port.

Runs larch.xafs.pre_edge on the Cu foil mu(E) with default parameters and dumps
scalars (e0, edge_step, ranges, coefs) plus sampled arrays (norm, flat,
pre_edge, post_edge, dmude).

Writes crates/xasproc/tests/data/ref_preedge.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/xasproc/scripts/ref_preedge.py
"""
import os
import numpy as np
from larch import Group
from larch.xafs.pre_edge import pre_edge

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
pre_edge(g)  # default parameters

det = g.pre_edge_details
out = os.path.join(data, "ref_preedge.txt")
with open(out, "w") as f:
    f.write("# pre_edge reference (larch, default params)\n")
    f.write(f"npts {len(g.energy)!r}\n")
    f.write(f"e0 {g.e0!r}\n")
    f.write(f"edge_step {g.edge_step!r}\n")
    f.write(f"pre1 {det.pre1!r}\n")
    f.write(f"pre2 {det.pre2!r}\n")
    f.write(f"norm1 {det.norm1!r}\n")
    f.write(f"norm2 {det.norm2!r}\n")
    f.write(f"nnorm {det.nnorm!r}\n")
    f.write(f"nvict {det.nvict!r}\n")
    f.write(f"pre_offset {det.pre_offset!r}\n")
    f.write(f"pre_slope {det.pre_slope!r}\n")
    # sampled arrays: name index value
    for name, arr in (("norm", g.norm), ("flat", g.flat),
                      ("preedge", g.pre_edge), ("postedge", g.post_edge),
                      ("dmude", g.dmude)):
        for i in range(0, len(arr), 50):
            f.write(f"{name} {i} {arr[i]!r}\n")
print(f"wrote {out}: e0={g.e0}, edge_step={g.edge_step}")
