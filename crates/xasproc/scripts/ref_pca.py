#!/usr/bin/env python
"""Reference values for the PCA port (pca_train + pca_fit), default numpy path.

No multi-standard set is available locally, so this synthesizes one from the Cu
foil: pre_edge -> norm, three well-separated basis shapes (norm, a 7-pt boxcar
broadening, a +6 eV shift), and six training spectra that are distinct linear
blends plus a tiny per-spectrum deterministic perturbation (so all six
eigenvalues are distinct and the components are comparable up to sign). A
seventh "unknown" spectrum is fit against the trained model with pca_fit.

All spectra share pca_model.x, so groups2matrix's cubic interpolation is the
identity and the reference isolates the algebra: eigh (pca_train) and
lstsq + a lower-bounded `scale` leastsq (pca_fit). We dump pca_model.ydat (the
exact matrix the algorithm consumed) so the Rust port sees identical inputs.

Writes crates/xasproc/tests/data/ref_pca.txt.

Run from the repo root with the project venv:
    .venv/bin/python crates/xasproc/scripts/ref_pca.py
"""
import os
import numpy as np
from larch import Group
from larch.xafs.pre_edge import pre_edge
from larch.math.pca import pca_train, pca_fit

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

b0 = norm.copy()
b1 = np.convolve(norm, np.ones(7) / 7.0, mode="same")
b2 = np.interp(en, en + 6.0, norm)
span = en[-1] - en[0]


def pert(k):
    return 0.002 * np.sin((en - en[0]) / span * k)


blends = [
    (1.00, 0.00, 0.00, 3.0),
    (0.70, 0.30, 0.00, 5.0),
    (0.40, 0.00, 0.60, 7.0),
    (0.50, 0.25, 0.25, 9.0),
    (0.20, 0.70, 0.10, 11.0),
    (0.33, 0.33, 0.34, 13.0),
]
specs = [a * b0 + b * b1 + c * b2 + pert(k) for (a, b, c, k) in blends]


def mkg(name, arr):
    gg = Group(energy=en.copy(), norm=arr.copy())
    gg.filename = name
    return gg


groups = [mkg(f"s{i}", s) for i, s in enumerate(specs)]
model = pca_train(groups)

# unknown spectrum to fit (distinct blend, on the same grid)
unknown = 0.45 * b0 + 0.35 * b1 + 0.20 * b2 + pert(8.0)
ug = mkg("unknown", unknown)
ncomps = 3
pca_fit(ug, model, ncomps=ncomps, rescale=True)
res = ug.pca_result

narr, nfreq = model.ydat.shape

out = os.path.join(data, "ref_pca.txt")
with open(out, "w") as f:
    f.write("# PCA reference (larch, synthetic Cu-derived)\n")
    f.write(f"narr {narr!r}\n")
    f.write(f"nfreq {nfreq!r}\n")
    f.write(f"ncomps {ncomps!r}\n")
    f.write(f"nsig {int(model.nsig)!r}\n")

    # exact training matrix the algorithm consumed (narr x nfreq), row-major
    for i in range(narr):
        for j in range(nfreq):
            f.write(f"ydat {i} {j} {model.ydat[i, j]!r}\n")

    # mean (nfreq), eigenvalues / variances / ind (narr)
    for j in range(nfreq):
        f.write(f"mean {j} {model.mean[j]!r}\n")
    for i in range(narr):
        f.write(f"eigval {i} {model.eigenvalues[i]!r}\n")
        f.write(f"variance {i} {model.variances[i]!r}\n")
        f.write(f"ind {i} {model.ind[i]!r}\n")

    # components (narr x nfreq) — compared up to per-row sign
    for i in range(narr):
        for j in range(nfreq):
            f.write(f"comp {i} {j} {model.components[i, j]!r}\n")

    # pca_fit results
    f.write(f"fit_scale {res.data_scale!r}\n")
    f.write(f"fit_chisq {res.chi_square!r}\n")
    for i in range(ncomps):
        f.write(f"fit_weight {i} {res.weights[i]!r}\n")
    for j in range(nfreq):
        f.write(f"unknown {j} {unknown[j]!r}\n")
    for j in range(0, nfreq, 10):
        f.write(f"fit_yfit {j} {res.yfit[j]!r}\n")

    # also record an unbounded-scale fit value, to confirm the bounded
    # transform actually changes the answer (or not) at our tolerance
    comps = model.components[:ncomps].transpose()

    def unbounded_scale():
        from scipy.optimize import leastsq

        def resid(p):
            sc = p[0]
            w, *_ = np.linalg.lstsq(comps, unknown * sc - model.mean, rcond=None)
            yfit = (w * comps).sum(axis=1) + model.mean
            return sc * unknown - yfit

        sol, _ = leastsq(resid, [1.0], ftol=1e-5, xtol=1e-5, gtol=1e-5, epsfcn=1e-5)
        return sol[0]

    f.write(f"fit_scale_unbounded {unbounded_scale()!r}\n")

print(f"wrote {out}")
print(f"narr={narr} nfreq={nfreq} nsig={model.nsig}")
print("eigenvalues:", model.eigenvalues)
print("variances:", model.variances)
print("scale:", res.data_scale, "chisq:", res.chi_square, "weights:", res.weights)
