#!/usr/bin/env python3
"""Reference generator for a *bounded*-variable `feffit()` fit (Rust `feffit`).

Identical two-path Cu fit to `ref_feffit_fit.py`, but three of the five global
variables carry min/max bounds (`amp`, `sig2_1`, `sig2_2`), while `del_e0` and
`alpha` stay unbounded. This exercises the Rust port's lmfit-style Minuit
internal<->external bound transform per variable:

- lmfit/larch optimise in *internal* (unbounded) coordinates and map back to the
  bounded value before each residual evaluation, so the fit is still a plain
  MINPACK leastsq — but on the transformed coordinates. The Rust `feffit` does
  the same (`Parameters::internal_x0`/`set_var_internal`), so `nfev` and the
  whole trajectory must match larch exactly (not just the endpoint).
- The covariance is transformed back with the MINUIT gradient scaling
  (`cov_ext = grad ⊗ grad * cov_int`), per variable, so the reported
  uncertainties match too.

The bounds are deliberately *inactive* (the best fit is interior), so the
solution equals the unbounded fit while the transform/gradient-scaling is fully
exercised — a clean parity check that isolates the bound machinery.

Run from the repo root with the project venv (xraylarch installed):
    .venv/bin/python scripts/ref_feffit_bounds.py
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

# (value, min, max); None bound = unbounded on that side
VAR_INIT = dict(amp=0.80, del_e0=0.0, alpha=0.0, sig2_1=0.003, sig2_2=0.003)
VAR_BOUNDS = dict(
    amp=(0.0, 1.0),
    sig2_1=(0.0, 0.01),
    sig2_2=(0.0, 0.01),
    # del_e0, alpha: unbounded
)

DERIVED = dict(alpha_x10="alpha*10")

PATH_WIRING = [
    dict(s02="amp", e0="del_e0", deltar="alpha*reff", sigma2="sig2_1"),
    dict(s02="amp", e0="del_e0", deltar="alpha*reff", sigma2="sig2_2"),
]

TRANSFORM = dict(
    kmin=3.0, kmax=15.0, kweight=2, dk=4.0, window="kaiser",
    rmin=1.4, rmax=3.0, dr=0.0, rwindow="hanning",
    nfft=2048, kstep=KSTEP, fitspace="r",
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
    ds = feffit_dataset(data=data, paths=paths, transform=trans, epsilon_k=EPSILON_K)

    params = Parameters()
    for name, val in VAR_INIT.items():
        lo, hi = VAR_BOUNDS.get(name, (-np.inf, np.inf))
        params.add(name, value=val, vary=True, min=lo, max=hi)
    for name, expr in DERIVED.items():
        params.add(name, expr=expr)

    return feffit(params, ds)


def fmt(v):
    return repr(float(v))


def write_ref(k, chi, result):
    var_names = list(VAR_INIT.keys())
    lines = [f"#fitspace {TRANSFORM['fitspace']}"]
    for key in ("kmin", "kmax", "kweight", "dk", "window",
                "rmin", "rmax", "dr", "rwindow", "nfft", "kstep"):
        lines.append(f"#transform {key} {TRANSFORM[key]}")
    lines.append(f"#epsilon_k {fmt(EPSILON_K)}")
    for name, val in VAR_INIT.items():
        lines.append(f"#var {name} {fmt(val)}")
    for name, (lo, hi) in VAR_BOUNDS.items():
        lines.append(f"#bound {name} {fmt(lo)} {fmt(hi)}")
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
    for di, dsi in enumerate(result.datasets):
        for pi, (label, path) in enumerate(dsi.paths.items()):
            path.store_feffdat()
            for pname in ("degen", "s02", "e0", "ei",
                          "deltar", "sigma2", "third", "fourth"):
                obj = path.params[path.pathpar_name(pname)]
                std = obj.stderr if obj.stderr is not None else float("nan")
                lines.append(f"#pathparam {di} {pi} {pname} {fmt(obj.value)} {fmt(std)}")
    lines.append("#begin data_k")
    lines.extend(fmt(v) for v in k)
    lines.append("#end")
    lines.append("#begin data_chi")
    lines.extend(fmt(v) for v in chi)
    lines.append("#end")

    with open(f"{DATADIR}/ref_feffit_bounds.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote ref_feffit_bounds.txt")
    print(f"  nfev={result.nfev} nvarys={result.nvarys} ndata={result.ndata} "
          f"n_idp={result.n_independent:.4f}")
    for name in var_names:
        par = result.params[name]
        lo, hi = VAR_BOUNDS.get(name, (None, None))
        b = f" in [{lo},{hi}]" if lo is not None else " (free)"
        print(f"  {name:8s} = {par.value:.6g} +/- "
              f"{(par.stderr if par.stderr is not None else float('nan')):.4g}{b}")


def main():
    k, chi = synth_data()
    write_ref(k, chi, run_fit(k, chi))


if __name__ == "__main__":
    main()
