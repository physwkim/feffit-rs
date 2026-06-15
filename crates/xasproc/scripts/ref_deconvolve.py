#!/usr/bin/env python
"""Reference values for the xas_deconvolve / xas_convolve port.

Runs pre_edge to get norm(E), then xas_deconvolve (lorentzian + gaussian, with
and without SG smoothing) and xas_convolve (lorentzian + gaussian) on the Cu
foil. Dumps the full deconv / conv arrays.

Writes crates/xasproc/tests/data/ref_deconvolve.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/xasproc/scripts/ref_deconvolve.py
"""
import os
import numpy as np
from larch import Group
from larch.xafs.pre_edge import pre_edge
from larch.xafs.deconvolve import xas_deconvolve, xas_convolve

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
pre_edge(g)
norm = g.norm

out = os.path.join(data, "ref_deconvolve.txt")
with open(out, "w") as f:
    f.write("# xas_deconvolve / xas_convolve reference (larch)\n")
    f.write(f"npts {len(energy)!r}\n")

    cases = [
        ("dec_lor_smooth", lambda gg: xas_deconvolve(gg, form="lorentzian"), "deconv"),
        ("dec_lor_raw",
         lambda gg: xas_deconvolve(gg, form="lorentzian", smooth=False), "deconv"),
        ("dec_gau_smooth", lambda gg: xas_deconvolve(gg, form="gaussian"), "deconv"),
        ("con_lor", lambda gg: xas_convolve(gg, form="lorentzian"), "conv"),
        ("con_gau", lambda gg: xas_convolve(gg, form="gaussian"), "conv"),
    ]
    for tag, fn, attr in cases:
        gg = Group(energy=energy.copy(), norm=norm.copy())
        fn(gg)
        arr = getattr(gg, attr)
        f.write(f"{tag}_n {len(arr)!r}\n")
        for i in range(0, len(arr), 10):
            f.write(f"{tag} {i} {arr[i]!r}\n")
print(f"wrote {out}")
