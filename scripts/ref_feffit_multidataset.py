#!/usr/bin/env python3
"""Reference generator for a *multi-dataset* simultaneous `feffit()` fit.

larch's `feffit(params, [ds0, ds1, ...])` fits a shared global parameter set
across several datasets at once: the residual is the per-dataset residuals
concatenated, `n_idp` is the sum of the per-dataset independent-point counts,
and the global variables couple every dataset. The Rust `feffit` already accepts
`&mut [FitDataSet]` and aggregates the same way, but the single-dataset
references never exercised more than one dataset — this closes that gap.

Layout (orthogonal to `ref_feffit_fit.py`, which is two paths in *one* dataset):
two datasets, **one path each**, sharing the same transform:

- dataset 0: `feff0001.dat` (first Cu shell), data synthesized with σ²=0.0035
- dataset 1: `feff0002.dat` (second Cu shell), data synthesized with σ²=0.0055

Five unbounded global variables: `amp`, `del_e0`, `alpha` are shared by both
datasets (`deltar = alpha*reff` in each), while `sig2_1` feeds only dataset 0's
path and `sig2_2` only dataset 1's — a genuine simultaneous fit (the shared
variables are constrained by both datasets, the σ² variables by one each).

Run from the repo root with the project venv (xraylarch installed):
    .venv/bin/python scripts/ref_feffit_multidataset.py
"""
import numpy as np
from lmfit import Parameters
from larch import Group
from larch.xafs import feffpath, feffit_transform, feffit_dataset, feffit
from larch.xafs.feffdat import ff2chi

DATADIR = "crates/feffit/tests/data"

# one path per dataset; (datafile, true σ² used to synthesize that dataset)
DATASETS = [
    dict(pathfile="feff0001.dat", true_sigma2=0.0035, sigma2_var="sig2_1"),
    dict(pathfile="feff0002.dat", true_sigma2=0.0055, sigma2_var="sig2_2"),
]

# shared "true" path parameters for the synthesized data (σ² is per-dataset)
TRUE_SHARED = dict(s02=0.90, e0=2.0, deltar=0.008)

EPSILON_K = 0.001
KSTEP = 0.05
NPTS = 401

# starting values for the five global fit variables (all unbounded)
VAR_INIT = dict(amp=0.80, del_e0=0.0, alpha=0.0, sig2_1=0.003, sig2_2=0.003)

# global constraint (expression) parameter — exercises derived-parameter
# uncertainty propagation in the multi-dataset context
DERIVED = dict(alpha_x10="alpha*10")

TRANSFORM = dict(
    kmin=3.0, kmax=15.0, kweight=2, dk=4.0, window="kaiser",
    rmin=1.4, rmax=3.0, dr=0.0, rwindow="hanning",
    nfft=2048, kstep=KSTEP, fitspace="r",
)


def synth_one(pathfile, true_sigma2):
    k = KSTEP * np.arange(NPTS)
    p = feffpath(f"{DATADIR}/{pathfile}", sigma2=true_sigma2, **TRUE_SHARED)
    g = ff2chi([p], params=Parameters(), k=k)
    return k, g.chi


def build_dataset(spec):
    k, chi = synth_one(spec["pathfile"], spec["true_sigma2"])
    trans = feffit_transform(**TRANSFORM)
    data = Group(k=k, chi=chi)
    path = feffpath(
        f"{DATADIR}/{spec['pathfile']}",
        s02="amp", e0="del_e0", deltar="alpha*reff", sigma2=spec["sigma2_var"],
    )
    ds = feffit_dataset(data=data, paths=[path], transform=trans,
                        epsilon_k=EPSILON_K)
    return (k, chi), ds


def run_fit():
    synths, datasets = [], []
    for spec in DATASETS:
        synth, ds = build_dataset(spec)
        synths.append(synth)
        datasets.append(ds)

    params = Parameters()
    for name, val in VAR_INIT.items():
        params.add(name, value=val, vary=True)
    for name, expr in DERIVED.items():
        params.add(name, expr=expr)

    result = feffit(params, datasets)
    return synths, result


def fmt(v):
    return repr(float(v))


def write_ref(synths, result):
    var_names = list(VAR_INIT.keys())
    lines = [f"#fitspace {TRANSFORM['fitspace']}"]
    for key in ("kmin", "kmax", "kweight", "dk", "window",
                "rmin", "rmax", "dr", "rwindow", "nfft", "kstep"):
        lines.append(f"#transform {key} {TRANSFORM[key]}")
    lines.append(f"#ndatasets {len(DATASETS)}")
    lines.append(f"#epsilon_k {fmt(EPSILON_K)}")
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
    for di, dsi in enumerate(result.datasets):
        for pi, (label, path) in enumerate(dsi.paths.items()):
            path.store_feffdat()
            for pname in ("degen", "s02", "e0", "ei",
                          "deltar", "sigma2", "third", "fourth"):
                obj = path.params[path.pathpar_name(pname)]
                std = obj.stderr if obj.stderr is not None else float("nan")
                lines.append(f"#pathparam {di} {pi} {pname} {fmt(obj.value)} {fmt(std)}")
    # per-dataset synthesized data so the Rust fit consumes byte-identical input
    for di, (k, chi) in enumerate(synths):
        lines.append(f"#begin data_k_{di}")
        lines.extend(fmt(v) for v in k)
        lines.append("#end")
        lines.append(f"#begin data_chi_{di}")
        lines.extend(fmt(v) for v in chi)
        lines.append("#end")

    with open(f"{DATADIR}/ref_feffit_multidataset.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote ref_feffit_multidataset.txt")
    print(f"  ndatasets={len(DATASETS)} nfev={result.nfev} nvarys={result.nvarys} "
          f"ndata={result.ndata} n_idp={result.n_independent:.4f}")
    for name in var_names:
        par = result.params[name]
        print(f"  {name:8s} = {par.value:.6g} +/- "
              f"{(par.stderr if par.stderr is not None else float('nan')):.4g}")


def main():
    synths, result = run_fit()
    write_ref(synths, result)


if __name__ == "__main__":
    main()
