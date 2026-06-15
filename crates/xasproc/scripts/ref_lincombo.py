#!/usr/bin/env python
"""Reference values for the lincombo_fit (XANES LCF) port.

There is no multi-standard dataset locally, so this synthesizes one from the Cu
foil: pre_edge -> norm, then three components (norm, a 5-pt boxcar-broadened
norm, and a +5 eV energy-shifted norm) plus a target that is a known linear
combination (0.5/0.3/0.2) perturbed by a deterministic out-of-span shape so the
constrained fit has a genuine optimum distinct from the lstsq seed. All arrays
share one energy grid, so groups2matrix's cubic interpolation is the identity
and the reference isolates the fit (lstsq seed + sum_to_one-constrained leastsq).

Writes crates/xasproc/tests/data/ref_lincombo.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/xasproc/scripts/ref_lincombo.py
"""
import os
import numpy as np
from larch import Group
from larch.xafs.pre_edge import pre_edge
from larch.math.lincombo_fitting import lincombo_fit

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
g = Group(energy=energy.copy(), mu=mu.copy())
pre_edge(g)
en, norm = g.energy, g.norm

comp0 = norm.copy()
comp1 = np.convolve(norm, np.ones(5) / 5.0, mode="same")
comp2 = np.interp(en, en + 5.0, norm)
extra = np.sin((en - en[0]) / (en[-1] - en[0]) * 7.0)
target = 0.5 * comp0 + 0.3 * comp1 + 0.2 * comp2 + 0.03 * extra


def mkg(name, arr):
    gg = Group(energy=en.copy(), norm=arr.copy())
    gg.filename = name
    return gg


comps = [mkg("comp0", comp0), mkg("comp1", comp1), mkg("comp2", comp2)]
res = lincombo_fit(mkg("target", target), comps)

out = os.path.join(data, "ref_lincombo.txt")
with open(out, "w") as f:
    f.write("# lincombo_fit (XANES LCF) reference (larch, synthetic Cu-derived)\n")
    f.write(f"npts {len(en)!r}\n")
    f.write(f"ncomps {len(comps)!r}\n")
    f.write(f"nvarys {res.result.nvarys!r}\n")

    # full arrays fed identically into the Rust port
    for tag, arr in (("ydat", target), ("comp0", comp0),
                     ("comp1", comp1), ("comp2", comp2)):
        for i in range(len(en)):
            f.write(f"{tag} {i} {arr[i]!r}\n")

    # scalar fit results
    for i, c in enumerate(("comp0", "comp1", "comp2")):
        f.write(f"w{i} {res.weights[c]!r}\n")
        f.write(f"ls{i} {res.weights_lstsq[c]!r}\n")
    f.write(f"chisqr {res.chisqr!r}\n")
    f.write(f"redchi {res.redchi!r}\n")
    f.write(f"rfactor {res.rfactor!r}\n")

    # yfit (sparse)
    for i in range(0, len(res.yfit), 10):
        f.write(f"yfit {i} {res.yfit[i]!r}\n")

print(f"wrote {out}")
