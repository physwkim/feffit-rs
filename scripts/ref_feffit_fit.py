#!/usr/bin/env python3
"""Reference generator for the end-to-end `feffit()` fit (Rust `feffit` crate).

Drives the real `larch.xafs.feffit` minimiser (gold standard) on a two-path Cu
fit with five unbounded global variables, two of which feed path parameters
through expressions (`deltar = 'alpha*reff'` exercises the per-path `reff`
symbol). Emits the synthesized data, the best-fit values + uncertainties, and
the rescaled fit statistics (chi_square, reduced, rfactor, aic, bic) consumed by
`crates/feffit/tests/parity.rs`.

Unbounded variables are used deliberately so lmfit's internal<->external
parameter transform is the identity and the fit reduces to a plain
scipy/MINPACK leastsq on the variables — matching the Rust `lm` port directly.

Run from the repo root with the project venv (xraylarch installed):
    .venv/bin/python scripts/ref_feffit_fit.py
"""
import numpy as np
from lmfit import Parameters
from larch import Group
from larch.xafs import feffpath, feffit_transform, feffit_dataset, feffit
from larch.xafs.feffdat import ff2chi

DATADIR = "crates/feffit/tests/data"
PATHFILES = ["feff0001.dat", "feff0002.dat"]

# "true" path parameters used to synthesize the data
TRUE_PARS = [
    dict(s02=0.90, e0=2.0, deltar=0.005, sigma2=0.0035),
    dict(s02=0.90, e0=2.0, deltar=0.012, sigma2=0.0055),
]

EPSILON_K = 0.001
KSTEP = 0.05
NPTS = 401

# starting values for the five global fit variables (all unbounded)
VAR_INIT = dict(amp=0.80, del_e0=0.0, alpha=0.0, sig2_1=0.003, sig2_2=0.003)

# global constraint (expression) parameters — not used by any path, present only
# to exercise uncertainty propagation onto derived parameters
DERIVED = dict(alpha_x10="alpha*10")

# how each path's parameters map to the global variables / expressions
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
        params.add(name, value=val, vary=True)
    for name, expr in DERIVED.items():
        params.add(name, expr=expr)

    result = feffit(params, ds)
    return result


def fmt(v):
    return repr(float(v))


def write_ref(k, chi, result):
    var_names = list(VAR_INIT.keys())
    lines = []
    lines.append(f"#fitspace {TRANSFORM['fitspace']}")
    for key in ("kmin", "kmax", "kweight", "dk", "window",
                "rmin", "rmax", "dr", "rwindow", "nfft", "kstep"):
        lines.append(f"#transform {key} {TRANSFORM[key]}")
    lines.append(f"#epsilon_k {fmt(EPSILON_K)}")
    for name, val in VAR_INIT.items():
        lines.append(f"#var {name} {fmt(val)}")
    # statistics
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
    # best-fit values + uncertainties (in declaration order)
    for name in var_names:
        par = result.params[name]
        std = par.stderr if par.stderr is not None else float("nan")
        lines.append(f"#best {name} {fmt(par.value)} {fmt(std)}")
    # derived (global constraint) parameters: value + propagated stderr
    for name in DERIVED:
        par = result.params[name]
        std = par.stderr if par.stderr is not None else float("nan")
        lines.append(f"#derived {name} {fmt(par.value)} {fmt(std)}")
    # path parameters: value + propagated stderr, per dataset/path. store_feffdat()
    # restores each path's own `reff` before reading `.value` (larch leaves the
    # shared symbol's `reff` at the last path's value otherwise).
    for di, dsi in enumerate(result.datasets):
        for pi, (label, path) in enumerate(dsi.paths.items()):
            path.store_feffdat()
            for pname in ("degen", "s02", "e0", "ei",
                          "deltar", "sigma2", "third", "fourth"):
                obj = path.params[path.pathpar_name(pname)]
                std = obj.stderr if obj.stderr is not None else float("nan")
                lines.append(
                    f"#pathparam {di} {pi} {pname} {fmt(obj.value)} {fmt(std)}"
                )
    # synthesized data so the Rust fit consumes byte-identical input
    lines.append("#begin data_k")
    lines.extend(fmt(v) for v in k)
    lines.append("#end")
    lines.append("#begin data_chi")
    lines.extend(fmt(v) for v in chi)
    lines.append("#end")

    with open(f"{DATADIR}/ref_feffit_fit.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote ref_feffit_fit.txt")
    print(f"  nfev={result.nfev} nvarys={result.nvarys} ndata={result.ndata} "
          f"n_idp={result.n_independent:.4f}")
    print(f"  chi_square={result.chi_square:.6g} reduced={result.chi2_reduced:.6g} "
          f"rfactor={result.rfactor:.6g} aic={result.aic:.4g} bic={result.bic:.4g}")
    for name in var_names:
        par = result.params[name]
        print(f"  {name:8s} = {par.value:.6g} +/- "
              f"{(par.stderr if par.stderr is not None else float('nan')):.4g}")


def main():
    k, chi = synth_data()
    result = run_fit(k, chi)
    write_ref(k, chi, result)


if __name__ == "__main__":
    main()
