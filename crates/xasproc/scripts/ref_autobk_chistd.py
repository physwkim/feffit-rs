#!/usr/bin/env python
"""Reference values for the AUTOBK chi_std/k_std background-constraint port.

Runs larch.xafs.autobk on the Cu foil mu(E) with an analytic standard chi(k)
(a damped first-shell-like sinusoid on a 0.1-spaced k grid, deliberately
different from the 0.05 output grid so the np.interp onto kout is exercised).
The standard constrains the spline so the residual minimizes the low-R content
of (chi - chi_std); the reported group.chi stays the true (mu-bkg)/edge_step.

Dumps the exact standard arrays (k_std, chi_std) so the Rust test feeds a
bit-identical input, plus scalars and the constrained outputs (k, chi, bkg,
chie) and the pre-fit init arrays (which larch computes without the standard).

Writes crates/xasproc/tests/data/ref_autobk_chistd.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/xasproc/scripts/ref_autobk_chistd.py
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

# analytic standard chi(k): damped sinusoid at Cu-Cu first-shell R ~ 2.55 Ang,
# on a 0.1-spaced grid (output grid is 0.05) so np.interp(kout, k_std, chi_std)
# is genuinely exercised. Amplitude ~0.15 (raw mu units) so it materially shifts
# the fitted background — a bug that ignored chi_std would change the outputs.
k_std = np.arange(0.0, 20.0 + 1e-9, 0.1)
R = 2.55
chi_std = 0.15 * np.sin(2 * R * k_std + 0.4) * np.exp(-2 * 0.005 * k_std**2)

g = Group(energy=energy, mu=mu)
autobk(g, k_std=k_std, chi_std=chi_std)  # default params otherwise

d = g.autobk_details
out = os.path.join(data, "ref_autobk_chistd.txt")
with open(out, "w") as f:
    f.write("# autobk chi_std reference (larch, default params + analytic standard)\n")
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
    # exact standard input arrays (every point) so the Rust test is bit-identical
    for i in range(len(k_std)):
        f.write(f"kstd {i} {k_std[i]!r}\n")
    for i in range(len(chi_std)):
        f.write(f"chistd {i} {chi_std[i]!r}\n")
    # constrained outputs + pre-fit init arrays, sampled
    for name, arr in (("k", g.k), ("chi", g.chi), ("bkg", g.bkg),
                      ("chie", g.chie), ("initbkg", d.init_bkg),
                      ("initchi", d.init_chi)):
        for i in range(0, len(arr), 20):
            f.write(f"{name} {i} {arr[i]!r}\n")
print(f"wrote {out}: ek0={g.ek0}, edge_step={g.edge_step}, "
      f"nspl={d.nspl}, irbkg={d.irbkg}, nk={len(g.k)}, "
      f"nkstd={len(k_std)}")
