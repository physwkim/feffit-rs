#!/usr/bin/env python3
"""Reference generator for `feffit()` *output arrays* (`save_outputs`/`_xafsft`).

After a fit, larch forward-transforms the data and model χ(k) (and each path's
χ(k)) into χ(R) on a uniform R-grid out to `rmax_out`, and back-transforms χ(R)
into χ(q) on a uniform q-grid out to `kmax+2`. `feffit()` calls `save_outputs`
automatically, attaching `chir_*`/`chiq_*` arrays to each dataset's `.data`,
`.model`, and path groups.

This drives the same two-path Cu fit as `ref_feffit_fit.py`, then dumps those
output arrays for the Rust `DataSet::save_outputs` parity test. The data χ(R) is
independent of the fit (a fixed FFT of the synthesized data), so it matches to
FFT round-off; the model/path χ(R) carry the best-fit ULP drift.

Run from the repo root with the project venv (xraylarch installed):
    .venv/bin/python scripts/ref_feffit_outputs.py
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
RMAX_OUT = 10.0

VAR_INIT = dict(amp=0.80, del_e0=0.0, alpha=0.0, sig2_1=0.003, sig2_2=0.003)
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
        params.add(name, value=val, vary=True)
    for name, expr in DERIVED.items():
        params.add(name, expr=expr)
    return feffit(params, ds, rmax_out=RMAX_OUT, path_outputs=True)


def fmt(v):
    return repr(float(v))


def block(lines, name, arr):
    lines.append(f"#begin {name}")
    lines.extend(fmt(v) for v in arr)
    lines.append("#end")


def dump_xafs(lines, prefix, grp):
    for suffix in ("chir_re", "chir_im", "chir_mag", "chir_pha",
                   "chiq_re", "chiq_im", "chiq_mag", "chiq_pha"):
        block(lines, f"{prefix}_{suffix}", getattr(grp, suffix))


def write_ref(k, chi, result):
    ds = result.datasets[0]
    lines = [f"#fitspace {TRANSFORM['fitspace']}"]
    for key in ("kmin", "kmax", "kweight", "dk", "window",
                "rmin", "rmax", "dr", "rwindow", "nfft", "kstep"):
        lines.append(f"#transform {key} {TRANSFORM[key]}")
    lines.append(f"#epsilon_k {fmt(EPSILON_K)}")
    lines.append(f"#rmax_out {fmt(RMAX_OUT)}")
    for name, val in VAR_INIT.items():
        lines.append(f"#var {name} {fmt(val)}")

    # synthesized data, so the Rust test rebuilds the identical fit
    block(lines, "data_k", k)
    block(lines, "data_chi", chi)

    # shared output grids (same rstep/irmax and q-grid for data/model/paths)
    block(lines, "out_r", ds.data.r)
    block(lines, "out_q", ds.data.q)

    # output arrays: data, model, and each path
    dump_xafs(lines, "data", ds.data)
    dump_xafs(lines, "model", ds.model)
    for pi, (label, path) in enumerate(ds.paths.items()):
        dump_xafs(lines, f"path{pi}", path)

    with open(f"{DATADIR}/ref_feffit_outputs.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote ref_feffit_outputs.txt")
    print(f"  rmax_out={RMAX_OUT} len(r)={len(ds.data.r)} len(q)={len(ds.data.q)} "
          f"npaths={len(ds.paths)}")
    print(f"  data chir_mag peak={np.max(ds.data.chir_mag):.6g} "
          f"model chir_mag peak={np.max(ds.model.chir_mag):.6g}")


def main():
    k, chi = synth_data()
    write_ref(k, chi, run_fit(k, chi))


if __name__ == "__main__":
    main()
