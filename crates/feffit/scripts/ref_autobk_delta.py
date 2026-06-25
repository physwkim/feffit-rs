#!/usr/bin/env python
"""Reference values for autobk_delta_chi (AUTOBK uncertainty bands).

Runs larch.xafs.autobk on the Cu foil with default parameters and
calc_uncertainties=True, err_sigma=1, then dumps the uncertainty arrays
(delta_chi, delta_bkg) plus the fit details they derive from (covar, coefs_std,
redchi) and the scalar intermediates (prob, degrees of freedom, the two Student-t
percent points). Also dumps standalone spot tables for the special functions the
port needs (erf and scipy.stats.t.ppf == Cephes stdtri) so they can be unit
tested in isolation.

Writes crates/feffit/tests/data/ref_autobk_delta.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/feffit/scripts/ref_autobk_delta.py
"""
import os
import numpy as np
from scipy.special import erf
from scipy.stats import t
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
err_sigma = 1
autobk(g, calc_uncertainties=True, err_sigma=err_sigma)

d = g.autobk_details
nspl = d.nspl
nchi = len(g.k)
nmue = d.iemax - d.iek0 + 1
prob = 0.5 * (1.0 + erf(err_sigma / np.sqrt(2.0)))
df_chi = nchi - nspl
df_bkg = nmue - nspl

out = os.path.join(data, "ref_autobk_delta.txt")
with open(out, "w") as f:
    f.write("# autobk_delta_chi reference (larch, default params, err_sigma=1)\n")
    f.write(f"err_sigma {err_sigma!r}\n")
    f.write(f"nspl {nspl!r}\n")
    f.write(f"iek0 {d.iek0!r}\n")
    f.write(f"iemax {d.iemax!r}\n")
    f.write(f"nchi {nchi!r}\n")
    f.write(f"nmue {nmue!r}\n")
    f.write(f"redchi {d.redchi!r}\n")
    f.write(f"chisqr {d.chisqr!r}\n")
    f.write(f"prob {prob!r}\n")
    f.write(f"df_chi {df_chi!r}\n")
    f.write(f"df_bkg {df_bkg!r}\n")
    f.write(f"tppf_chi {float(t.ppf(prob, df_chi))!r}\n")
    f.write(f"tppf_bkg {float(t.ppf(prob, df_bkg))!r}\n")
    f.write(f"ndchi {len(g.delta_chi)!r}\n")
    f.write(f"ndbkg {len(g.delta_bkg)!r}\n")

    # covariance (nspl x nspl) and per-coef std the bands derive from
    for i in range(nspl):
        f.write(f"coefs_std {i} {d.coefs_std[i]!r}\n")
        for j in range(nspl):
            f.write(f"covar {i} {j} {d.covar[i, j]!r}\n")

    # uncertainty bands (sampled)
    for i in range(0, len(g.delta_chi), 5):
        f.write(f"dchi {i} {g.delta_chi[i]!r}\n")
    for i in range(0, len(g.delta_bkg), 20):
        f.write(f"dbkg {i} {g.delta_bkg[i]!r}\n")

    # --- standalone special-function spot tables ---
    erf_xs = [0.0, 0.1, 0.5, 0.7071067811865476, 1.0, 1.5, 2.0, 3.0,
              -0.5, -1.3, 5.0, 0.25, 4.2]
    for x in erf_xs:
        f.write(f"sf_erf {x!r} {float(erf(x))!r}\n")

    tppf_cases = [
        (0.8413447460685429, 500), (0.8413447460685429, 50),
        (0.8413447460685429, 7), (0.8413447460685429, 3),
        (0.9772498680518208, 300), (0.9772498680518208, 12),
        (0.5, 25), (0.6, 25), (0.75, 9), (0.95, 100),
        (0.99, 4), (0.025, 30), (0.1, 1000),
    ]
    for (p, df) in tppf_cases:
        f.write(f"sf_tppf {p!r} {df!r} {float(t.ppf(p, df))!r}\n")

print(f"wrote {out}")
print(f"nspl={nspl} nchi={nchi} nmue={nmue} df_chi={df_chi} df_bkg={df_bkg}")
print(f"tppf_chi={float(t.ppf(prob, df_chi))} tppf_bkg={float(t.ppf(prob, df_bkg))}")
print(f"delta_chi[:3]={g.delta_chi[:3]}")
