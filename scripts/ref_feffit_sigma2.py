#!/usr/bin/env python3
"""Reference for an end-to-end `feffit()` fit whose σ² is a `sigma2_eins(...)`
path expression — exercising the Rust port's path-bound function context
(`FuncCtx`) and the numerical uncertainty propagation through an opaque function.

A single Cu path is fit with `sigma2 = sigma2_eins(temp, theta)` (temp fixed,
theta varied) plus `amp` (s02) and `del_e0` (e0). Gaussian noise is added to the
synthesized χ(k) (seeded, and emitted verbatim) so the covariance — and hence
the propagated σ² uncertainty — is non-zero.

`sigma2_eins` needs no Feff DLL, so larch runs this fit directly; the companion
`sigma2_debye` model can only be checked standalone on arm64 (its C library is
x86_64-only), so it is verified in the Rust tests by an independent finite
difference rather than here.

Run from the repo root with the project venv:
    .venv/bin/python scripts/ref_feffit_sigma2.py
"""
import numpy as np
from lmfit import Parameters
from larch import Group
from larch.xafs import feffpath, feffit_transform, feffit_dataset, feffit
from larch.xafs.feffdat import ff2chi
from larch.xafs.sigma2_models import sigma2_eins

DATADIR = "crates/feffit/tests/data"
PATHFILE = "feff0001.dat"

TEMP = 300.0       # fixed sample temperature (K)
THETA_TRUE = 320.0  # true Einstein temperature (K)
KSTEP = 0.05
NPTS = 401
NOISE = 2.0e-4
SEED = 11

VAR_INIT = dict(amp=0.85, del_e0=0.0, theta=280.0)

TRANSFORM = dict(
    kmin=3.0, kmax=15.0, kweight=2, dk=4.0, window="kaiser",
    rmin=1.4, rmax=3.0, dr=0.0, rwindow="hanning",
    nfft=2048, kstep=KSTEP, fitspace="r",
)


def synth_data():
    k = KSTEP * np.arange(NPTS)
    p = feffpath(f"{DATADIR}/{PATHFILE}")
    p.store_feffdat()
    sig2 = sigma2_eins(TEMP, THETA_TRUE, p)
    path = feffpath(f"{DATADIR}/{PATHFILE}", s02=0.9, e0=2.0, deltar=0.0, sigma2=sig2)
    chi = ff2chi([path], params=Parameters(), k=k).chi
    chi = chi + np.random.RandomState(SEED).normal(scale=NOISE, size=chi.shape)
    return k, chi


def run_fit(k, chi):
    trans = feffit_transform(**TRANSFORM)
    path = feffpath(f"{DATADIR}/{PATHFILE}", s02="amp", e0="del_e0",
                    sigma2="sigma2_eins(temp, theta)")
    ds = feffit_dataset(data=Group(k=k, chi=chi), paths=[path], transform=trans,
                        epsilon_k=NOISE)
    params = Parameters()
    params.add("temp", value=TEMP, vary=False)
    for name, val in VAR_INIT.items():
        params.add(name, value=val, vary=True)
    return feffit(params, ds)


def fmt(v):
    return repr(float(v))


def write_ref(k, chi, result):
    lines = [f"#fitspace {TRANSFORM['fitspace']}"]
    for key in ("kmin", "kmax", "kweight", "dk", "window",
                "rmin", "rmax", "dr", "rwindow", "nfft", "kstep"):
        lines.append(f"#transform {key} {TRANSFORM[key]}")
    lines.append(f"#epsilon_k {fmt(NOISE)}")
    lines.append(f"#temp {fmt(TEMP)}")
    for name, val in VAR_INIT.items():
        lines.append(f"#var {name} {fmt(val)}")
    lines.append(f"#nfev {int(result.nfev)}")
    lines.append(f"#nvarys {int(result.nvarys)}")
    lines.append(f"#nfree {int(result.nfree)}")
    lines.append(f"#ndata {int(result.ndata)}")
    lines.append(f"#n_idp {fmt(result.n_independent)}")
    lines.append(f"#chi_square {fmt(result.chi_square)}")
    lines.append(f"#chi2_reduced {fmt(result.chi2_reduced)}")
    for name in VAR_INIT:
        par = result.params[name]
        std = par.stderr if par.stderr is not None else float("nan")
        lines.append(f"#best {name} {fmt(par.value)} {fmt(std)}")
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

    with open(f"{DATADIR}/ref_feffit_sigma2.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote ref_feffit_sigma2.txt")
    print(f"  nfev={result.nfev} nvarys={result.nvarys} "
          f"chi_square={result.chi_square:.6g}")
    for name in VAR_INIT:
        par = result.params[name]
        print(f"  {name:8s} = {par.value:.6g} +/- "
              f"{(par.stderr if par.stderr is not None else float('nan')):.4g}")


def main():
    k, chi = synth_data()
    write_ref(k, chi, run_fit(k, chi))


if __name__ == "__main__":
    main()
