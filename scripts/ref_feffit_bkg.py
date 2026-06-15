#!/usr/bin/env python3
"""Reference generator for a background-refinement `feffit()` fit (`refine_bkg`).

larch's `refine_bkg=True` adds a cubic B-spline background as extra fit
variables `bkg00..bkgNN` (`nspline = 1 + round(2*rbkg*(kmax-kmin)/π)` of them),
subtracted from χ(k) before the model. `prepare_fit` also mutates the transform
(`rbkg = max(rbkg, rmin)`, `rmin = rstep`) and the independent-point count
(`n_idp = 1 + 2*rmax*(kmax-kmin)/π`), so the low-R region enters the residual.

Same two-path Cu fit as `ref_feffit_fit.py`, with `rbkg=1.0` and
`refine_bkg=True`. The data is synthesized from pure paths (no background), so
the refined background should sit near zero; the point is end-to-end parity of
the background machinery (nspline, knots, the spline subtraction, the modified
n_idp/rmin, the extra fit variables and their uncertainties).

Run from the repo root with the project venv (xraylarch installed):
    .venv/bin/python scripts/ref_feffit_bkg.py
"""
import numpy as np
from lmfit import Parameters
from larch import Group
from larch.xafs import feffpath, feffit_transform, feffit_dataset, feffit
from larch.xafs.feffdat import ff2chi

DATADIR = "crates/feffit/tests/data"
PATHFILES = ["feff0001.dat", "feff0002.dat"]

TRUE_PARS = [
    dict(s02=0.90, e0=2.0, deltar=0.005, sigma2=0.0035),
    dict(s02=0.90, e0=2.0, deltar=0.012, sigma2=0.0055),
]

EPSILON_K = 0.001
KSTEP = 0.05
NPTS = 401
RBKG = 1.0

VAR_INIT = dict(amp=0.80, del_e0=0.0, alpha=0.0, sig2_1=0.003, sig2_2=0.003)
DERIVED = dict(alpha_x10="alpha*10")
PATH_WIRING = [
    dict(s02="amp", e0="del_e0", deltar="alpha*reff", sigma2="sig2_1"),
    dict(s02="amp", e0="del_e0", deltar="alpha*reff", sigma2="sig2_2"),
]
TRANSFORM = dict(
    kmin=3.0, kmax=15.0, kweight=2, dk=4.0, window="kaiser",
    rmin=1.4, rmax=3.0, dr=0.0, rwindow="hanning",
    nfft=2048, kstep=KSTEP, fitspace="r", rbkg=RBKG,
)


def synth_data():
    k = KSTEP * np.arange(NPTS)
    paths = [feffpath(f"{DATADIR}/{f}", **p) for f, p in zip(PATHFILES, TRUE_PARS)]
    g = ff2chi(paths, params=Parameters(), k=k)
    return k, g.chi


def run_fit(k, chi):
    trans = feffit_transform(**TRANSFORM)
    data = Group(k=k, chi=chi)
    paths = [feffpath(f"{DATADIR}/{f}", **w) for f, w in zip(PATHFILES, PATH_WIRING)]
    ds = feffit_dataset(data=data, paths=paths, transform=trans,
                        epsilon_k=EPSILON_K, refine_bkg=True)
    params = Parameters()
    for name, val in VAR_INIT.items():
        params.add(name, value=val, vary=True)
    for name, expr in DERIVED.items():
        params.add(name, expr=expr)
    return feffit(params, ds)


def fmt(v):
    return repr(float(v))


def bkg_params(result):
    """The bkg00..bkgNN parameters, in index order (names carry a hashkey)."""
    items = []
    for name, par in result.params.items():
        if name.startswith("bkg"):
            idx = int(name[3:5])
            items.append((idx, par))
    items.sort(key=lambda t: t[0])
    return items


def write_ref(k, chi, result):
    ds0 = result.datasets[0]
    var_names = list(VAR_INIT.keys())
    lines = [f"#fitspace {TRANSFORM['fitspace']}"]
    for key in ("kmin", "kmax", "kweight", "dk", "window",
                "rmin", "rmax", "dr", "rwindow", "nfft", "kstep", "rbkg"):
        lines.append(f"#transform {key} {TRANSFORM[key]}")
    lines.append(f"#epsilon_k {fmt(EPSILON_K)}")
    lines.append(f"#nspline {ds0.bkg_spline['nspline']}")
    for name, val in VAR_INIT.items():
        lines.append(f"#var {name} {fmt(val)}")
    lines.append(f"#nfev {int(result.nfev)}")
    lines.append(f"#nvarys {int(result.nvarys)}")
    lines.append(f"#nfree {int(result.nfree)}")
    lines.append(f"#ndata {int(result.ndata)}")
    lines.append(f"#n_idp {fmt(result.n_independent)}")
    lines.append(f"#chi_square {fmt(result.chi_square)}")
    lines.append(f"#chi2_reduced {fmt(result.chi2_reduced)}")
    lines.append(f"#rfactor {fmt(result.rfactor)}")
    lines.append(f"#aic {fmt(result.aic)}")
    lines.append(f"#bic {fmt(result.bic)}")
    for name in var_names:
        par = result.params[name]
        std = par.stderr if par.stderr is not None else float("nan")
        lines.append(f"#best {name} {fmt(par.value)} {fmt(std)}")
    for name in DERIVED:
        par = result.params[name]
        std = par.stderr if par.stderr is not None else float("nan")
        lines.append(f"#derived {name} {fmt(par.value)} {fmt(std)}")
    for idx, par in bkg_params(result):
        std = par.stderr if par.stderr is not None else float("nan")
        lines.append(f"#bkg {idx} {fmt(par.value)} {fmt(std)}")
    # the spline knot vector (so the Rust closed-form knots can be checked here too)
    lines.append("#begin knots")
    lines.extend(fmt(v) for v in ds0.bkg_spline['knots'])
    lines.append("#end")
    lines.append("#begin data_k")
    lines.extend(fmt(v) for v in k)
    lines.append("#end")
    lines.append("#begin data_chi")
    lines.extend(fmt(v) for v in chi)
    lines.append("#end")

    with open(f"{DATADIR}/ref_feffit_bkg.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote ref_feffit_bkg.txt")
    print(f"  nspline={ds0.bkg_spline['nspline']} nfev={result.nfev} "
          f"nvarys={result.nvarys} ndata={result.ndata} n_idp={result.n_independent:.4f}")
    for name in var_names:
        par = result.params[name]
        print(f"  {name:8s} = {par.value:.6g} +/- "
              f"{(par.stderr if par.stderr is not None else float('nan')):.4g}")
    bp = bkg_params(result)
    print(f"  bkg coefs ({len(bp)}): "
          + ", ".join(f"{p.value:.2e}" for _, p in bp))


def main():
    k, chi = synth_data()
    write_ref(k, chi, run_fit(k, chi))


if __name__ == "__main__":
    main()
