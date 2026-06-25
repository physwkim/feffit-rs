#!/usr/bin/env python
"""Reference values for the E0 / energy-step port and the math primitives.

Loads the Cu foil mu(E) (crates/feffit/tests/data/cu.xmu) and dumps:
  - find_energy_step(energy)
  - find_e0(energy, mu)
  - smooth(...) and polyfit(...) on fixed inputs (direct primitive parity)

Writes crates/feffit/tests/data/ref_e0.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/feffit/scripts/ref_e0.py
"""
import os
import numpy as np
from larch.xafs.pre_edge import find_e0, find_energy_step
from larch.math.utils import smooth, polyfit

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = os.path.join(here, "tests", "data")


def load_xmu(path):
    e, m = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            e.append(float(parts[0]))
            m.append(float(parts[1]))
    return np.array(e), np.array(m)


energy, mu = load_xmu(os.path.join(data, "cu.xmu"))

estep = find_energy_step(energy)
e0 = find_e0(energy, mu)

# smooth on the raw derivative, with a fixed step/sigma (lorentzian, npad=5)
dmu = np.gradient(mu) / np.gradient(energy)
sm = smooth(energy, dmu, xstep=0.5, sigma=0.5, form='lorentzian')

# polyfit: a noisy quadratic over an energy-scale domain
xpf = energy[100:140]
ypf = mu[100:140]
pcoef = polyfit(xpf, ypf, 2)  # low->high order

out = os.path.join(data, "ref_e0.txt")
with open(out, "w") as f:
    f.write("# E0 / primitives reference (larch)\n")
    f.write(f"npts {len(energy)!r}\n")
    f.write(f"find_energy_step {estep!r}\n")
    f.write(f"find_e0 {e0!r}\n")
    f.write(f"polyfit2 {pcoef[0]!r} {pcoef[1]!r} {pcoef[2]!r}\n")
    # smooth: dump a handful of sampled points (index value) to keep file small
    f.write("# smooth samples: index value\n")
    for i in range(0, len(sm), 60):
        f.write(f"smooth {i} {sm[i]!r}\n")
print(f"wrote {out}: estep={estep}, e0={e0}")
