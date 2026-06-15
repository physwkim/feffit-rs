#!/usr/bin/env python3
"""Reference generator for the `feffit` Rust port (residual core).

Drives the real `larch.xafs.feffit` machinery (gold standard) to produce the
fit residual for fixed numeric path parameters in k/R/q space, plus the
epsilon_k/epsilon_r noise estimates. Emits labeled-block files consumed by
`crates/feffit/tests/parity.rs`.

Run from the repo root with the project venv (which has xraylarch installed):
    .venv/bin/python scripts/ref_feffit.py
"""
import numpy as np
from lmfit import Parameters
from larch import Group
from larch.xafs import feffpath, feffit_transform, feffit_dataset
from larch.xafs.feffdat import ff2chi

DATADIR = "crates/feffit/tests/data"

# the two Cu paths used for both the synthetic data and the fit model
PATHFILES = ["feff0001.dat", "feff0002.dat"]

# "true" path parameters used to synthesize the data (s02,e0,deltar,sigma2,third,fourth,ei)
TRUE_PARS = [
    dict(s02=0.90, e0=1.5, deltar=0.004, sigma2=0.0035, third=0.0, fourth=0.0, ei=0.0),
    dict(s02=0.90, e0=1.5, deltar=0.010, sigma2=0.0055, third=0.0, fourth=0.0, ei=0.0),
]
# "model" path parameters evaluated in the residual (deliberately different)
MODEL_PARS = [
    dict(s02=1.00, e0=0.0, deltar=0.000, sigma2=0.0030, third=0.0, fourth=0.0, ei=0.0),
    dict(s02=1.00, e0=0.0, deltar=0.000, sigma2=0.0050, third=0.0, fourth=0.0, ei=0.0),
]

EPSILON_K = 0.001
KSTEP = 0.05
NPTS = 401  # k: 0 .. 20.0


def make_paths(parlist):
    paths = []
    for fname, pars in zip(PATHFILES, parlist):
        paths.append(feffpath(f"{DATADIR}/{fname}", **pars))
    return paths


def synth_data():
    k = KSTEP * np.arange(NPTS)
    paths = make_paths(TRUE_PARS)
    g = ff2chi(paths, params=Parameters(), k=k)
    return k, g.chi


def write_blocks(path, params, paths_spec, blocks):
    with open(path, "w") as fh:
        for key, val in params.items():
            fh.write(f"#param {key} {val}\n")
        for fname, pars in paths_spec:
            vals = " ".join(
                repr(float(pars[p]))
                for p in ("s02", "e0", "deltar", "sigma2", "third", "fourth", "ei")
            )
            fh.write(f"#path {fname} {vals}\n")
        for name, arr in blocks.items():
            fh.write(f"#begin {name}\n")
            for v in arr:
                fh.write(f"{float(v)!r}\n")
            fh.write("#end\n")


def common_transform(fitspace):
    return dict(
        kmin=3.0, kmax=15.0, kweight=2, dk=4.0, window="kaiser",
        rmin=1.4, rmax=3.0, dr=0.0, rwindow="hanning",
        nfft=2048, kstep=KSTEP, fitspace=fitspace,
    )


def wavelet_transform():
    # The 'w' residual is realimag(cwt).ravel() over the [rmin,rmax)x[kmin,kmax)
    # mask, so its length grows as (n_rrows * n_kcols * 2). Deliberately compact
    # k/R ranges keep the reference fixture small while exercising the full cwt
    # code path (FFTs, Cauchy filter, mask slice, realimag ordering).
    return dict(
        kmin=3.0, kmax=6.0, kweight=2, dk=4.0, window="kaiser",
        rmin=1.2, rmax=2.0, dr=0.0, rwindow="hanning",
        nfft=2048, kstep=KSTEP, fitspace="w",
    )


def run_case(fitspace, k, chi, tpars=None):
    if tpars is None:
        tpars = common_transform(fitspace)
    trans = feffit_transform(**tpars)
    data = Group(k=k, chi=chi)
    paths = make_paths(MODEL_PARS)
    ds = feffit_dataset(data=data, paths=paths, transform=trans, epsilon_k=EPSILON_K)
    params = Parameters()
    ds.prepare_fit(params)
    resid = ds._residual(params)

    out_params = dict(tpars)
    out_params["epsilon_k"] = EPSILON_K
    out_params["epsilon_r"] = repr(float(ds.epsilon_r))
    out_params["n_idp"] = repr(float(ds.n_idp))
    paths_spec = list(zip(PATHFILES, MODEL_PARS))
    blocks = dict(
        data_k=k, data_chi=chi,
        model_chi=ds.model.chi,
        residual=resid,
    )
    write_blocks(f"{DATADIR}/ref_feffit_{fitspace}.txt", out_params, paths_spec, blocks)
    print(f"wrote ref_feffit_{fitspace}.txt  (resid len={len(resid)}, eps_r={ds.epsilon_r:.6g})")


def run_estimate_noise(k, chi):
    """Separate reference for estimate_noise (no autobk involved)."""
    tpars = common_transform("r")
    trans = feffit_transform(**tpars)
    data = Group(k=k, chi=chi)
    paths = make_paths(MODEL_PARS)
    ds = feffit_dataset(data=data, paths=paths, transform=trans, epsilon_k=EPSILON_K)
    params = Parameters()
    ds.prepare_fit(params)  # sets ds._chi
    # call estimate_noise directly on the interpolated data chi
    ds.estimate_noise(chi=ds._chi, rmin=15.0, rmax=30.0)
    out_params = dict(tpars)
    out_params["epsilon_k"] = EPSILON_K
    out_params["rmin_noise"] = 15.0
    out_params["rmax_noise"] = 30.0
    out_params["est_epsilon_k"] = repr(float(ds.epsilon_k))
    out_params["est_epsilon_r"] = repr(float(ds.epsilon_r))
    paths_spec = list(zip(PATHFILES, MODEL_PARS))
    blocks = dict(data_k=k, data_chi=chi)
    write_blocks(f"{DATADIR}/ref_feffit_noise.txt", out_params, paths_spec, blocks)
    print(f"wrote ref_feffit_noise.txt  (est_eps_k={ds.epsilon_k:.6g}, est_eps_r={ds.epsilon_r:.6g})")


def main():
    k, chi = synth_data()
    for fitspace in ("r", "k", "q"):
        run_case(fitspace, k, chi)
    run_case("w", k, chi, wavelet_transform())
    run_estimate_noise(k, chi)


if __name__ == "__main__":
    main()
