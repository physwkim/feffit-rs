#!/usr/bin/env python3
"""Reference generator for the `xafsft` Rust port.

Replicates `larch/xafs/xafsft.py` (ftwindow, xftf_fast, xftr_fast, xftf_prep,
xftf, xftr) using the exact libraries larch uses: `scipy.fftpack` for the FFTs
and `scipy.special.i0` for the Kaiser-Bessel window. Emits labeled-block files
consumed by `crates/xafsft/tests/parity.rs`.

Run from the repo root with the project venv:
    .venv/bin/python scripts/ref_xftf.py
"""
import numpy as np
from numpy import pi, arange, zeros, ones, sin, cos, exp, sqrt, where, linspace
from scipy.fftpack import fft, ifft
from scipy.special import i0 as bessel_i0

sqrtpi = sqrt(pi)
OUT = "crates/xafsft/tests/data"


def ftwindow(x, xmin=None, xmax=None, dx=1, dx2=None, window="hanning"):
    nam = window.strip().lower()[:3]
    dx1 = dx
    if dx2 is None:
        dx2 = dx1
    if xmin is None:
        xmin = min(x)
    if xmax is None:
        xmax = max(x)
    xstep = (x[-1] - x[0]) / (len(x) - 1)
    if xstep < 0 or np.isnan(xstep):
        xstep = 1.0e-3
    xeps = 1.0e-4 * xstep
    x1 = max(min(x), xmin - dx1 / 2.0)
    x2 = xmin + dx1 / 2.0 + xeps
    x3 = xmax - dx2 / 2.0 - xeps
    x4 = min(max(x), xmax + dx2 / 2.0)
    if nam == "fha":
        if dx1 < 0:
            dx1 = 0
        if dx2 > 1:
            dx2 = 1
        x2 = x1 + xeps + dx1 * (xmax - xmin) / 2.0
        x3 = x4 - xeps - dx2 * (xmax - xmin) / 2.0
    elif nam == "gau":
        dx1 = max(dx1, xeps)

    def asint(val):
        return int((val + xeps) / xstep)

    i1, i2, i3, i4 = asint(x1), asint(x2), asint(x3), asint(x4)
    i1, i2 = max(0, i1), max(0, i2)
    i3, i4 = min(len(x) - 1, i3), min(len(x) - 1, i4)
    if i2 == i1:
        i1 = max(0, i2 - 1)
    if i4 == i3:
        i3 = max(i2, i4 - 1)
    x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
    if x1 == x2:
        x2 = x2 + xeps
    if x3 == x4:
        x4 = x4 + xeps
    fwin = zeros(len(x))
    if i3 > i2:
        fwin[i2:i3] = ones(i3 - i2)
    if nam in ("han", "fha"):
        fwin[i1 : i2 + 1] = sin((pi / 2) * (x[i1 : i2 + 1] - x1) / (x2 - x1)) ** 2
        fwin[i3 : i4 + 1] = cos((pi / 2) * (x[i3 : i4 + 1] - x3) / (x4 - x3)) ** 2
    elif nam == "par":
        fwin[i1 : i2 + 1] = (x[i1 : i2 + 1] - x1) / (x2 - x1)
        fwin[i3 : i4 + 1] = 1 - (x[i3 : i4 + 1] - x3) / (x4 - x3)
    elif nam == "wel":
        fwin[i1 : i2 + 1] = 1 - ((x[i1 : i2 + 1] - x2) / (x2 - x1)) ** 2
        fwin[i3 : i4 + 1] = 1 - ((x[i3 : i4 + 1] - x3) / (x4 - x3)) ** 2
    elif nam in ("kai", "bes"):
        cen = (x4 + x1) / 2
        wid = (x4 - x1) / 2
        arg = 1 - (x - cen) ** 2 / (wid**2)
        arg[where(arg < 0)] = 0
        if nam == "bes":
            fwin = bessel_i0(dx * sqrt(arg)) / bessel_i0(dx)
            fwin[where(x <= x1)] = 0
            fwin[where(x >= x4)] = 0
        else:
            scale = max(1.0e-10, bessel_i0(dx) - 1)
            fwin = (bessel_i0(dx * sqrt(arg)) - 1) / scale
    elif nam == "sin":
        fwin[i1 : i4 + 1] = sin(pi * (x4 - x[i1 : i4 + 1]) / (x4 - x1))
    elif nam == "gau":
        cen = (x4 + x1) / 2
        fwin = exp(-(((x - cen) ** 2) / (2 * dx1 * dx1)))
    return fwin


def xftf_fast(chi, nfft=2048, kstep=0.05):
    cchi = zeros(nfft, dtype="complex128")
    cchi[0 : len(chi)] = chi
    return (kstep / sqrtpi) * fft(cchi)[: int(nfft / 2)]


def xftr_fast(chir, nfft=2048, kstep=0.05):
    cchi = zeros(nfft, dtype="complex128")
    cchi[0 : len(chir)] = chir
    return (4 * sqrtpi / kstep) * ifft(cchi)[: int(nfft / 2)]


def xftf_prep(k, chi, kmin=0, kmax=20, kweight=2, dk=1, dk2=None, window="kaiser", nfft=2048, kstep=0.05):
    if dk2 is None:
        dk2 = dk
    kweight = int(kweight)
    npts = int(1.01 + max(k) / kstep)
    k_max = max(max(k), kmax + dk2)
    k_ = kstep * np.arange(int(1.01 + k_max / kstep), dtype="float64")
    chi_ = np.interp(k_, k, chi)
    win = ftwindow(k_, xmin=kmin, xmax=kmax, dx=dk, dx2=dk2, window=window)
    return ((chi_[:npts] * k_[:npts] ** kweight), win[:npts])


def xftf(k, chi, kmin=0, kmax=20, kweight=2, dk=1, dk2=None, window="kaiser", rmax_out=10, nfft=2048, kstep=0.05):
    cchi, win = xftf_prep(k, chi, kmin, kmax, kweight, dk, dk2, window, nfft, kstep)
    out = xftf_fast(cchi * win, kstep=kstep, nfft=nfft)
    rstep = pi / (kstep * nfft)
    irmax = int(min(nfft / 2, 1.01 + rmax_out / rstep))
    r = rstep * arange(irmax)
    return dict(kwin=win[: len(chi)], r=r, chir=out[:irmax])


def xftr(r, chir, rmin=0, rmax=20, dr=1, dr2=None, rw=0, window="kaiser", qmax_out=30, nfft=2048):
    rstep = r[1] - r[0]
    kstep = pi / (rstep * nfft)
    scale = 1.0
    cchir = zeros(nfft, dtype="complex128")
    r_ = rstep * arange(nfft, dtype="float64")
    cchir[0 : len(chir)] = chir
    if chir.dtype == np.dtype("complex128"):
        scale = 0.5
    win = ftwindow(r_, xmin=rmin, xmax=rmax, dx=dr, dx2=dr2, window=window)
    out = scale * xftr_fast(cchir * win * r_**rw, kstep=kstep, nfft=nfft)
    q = linspace(0, qmax_out, int(1.05 + qmax_out / kstep))
    nkpts = len(q)
    return dict(rwin=win[: len(chir)], q=q, chiq=out[:nkpts])


def write_blocks(path, params, blocks):
    with open(path, "w") as fh:
        for key, val in params.items():
            fh.write(f"#param {key} {val}\n")
        for name, arr in blocks.items():
            fh.write(f"#begin {name}\n")
            for v in arr:
                fh.write(f"{float(v)!r}\n")
            fh.write("#end\n")


def main():
    kstep = 0.05
    npts = 361  # k: 0 .. 18.0
    k = kstep * np.arange(npts)
    chi = (
        0.8 * np.sin(2 * 2.55 * k + 0.3) * np.exp(-2 * 0.003 * k**2)
        + 0.5 * np.sin(2 * 3.61 * k - 0.7) * np.exp(-2 * 0.006 * k**2)
    )

    # forward, kaiser
    fk = dict(kmin=3.0, kmax=17.0, kweight=2, dk=4.0, window="kaiser", rmax_out=10.0, nfft=2048, kstep=kstep)
    ok = xftf(k, chi, **fk)
    write_blocks(
        f"{OUT}/ref_xftf_kaiser.txt", fk,
        dict(k=k, chi=chi, kwin=ok["kwin"], r=ok["r"], chir_re=ok["chir"].real, chir_im=ok["chir"].imag),
    )

    # forward, hanning
    fh = dict(kmin=3.0, kmax=17.0, kweight=2, dk=1.0, window="hanning", rmax_out=10.0, nfft=2048, kstep=kstep)
    oh = xftf(k, chi, **fh)
    write_blocks(
        f"{OUT}/ref_xftf_hanning.txt", fh,
        dict(k=k, chi=chi, kwin=oh["kwin"], r=oh["r"], chir_re=oh["chir"].real, chir_im=oh["chir"].imag),
    )

    # reverse, from the kaiser forward output (complex chir)
    rp = dict(rmin=1.0, rmax=3.0, dr=1.0, rw=0, window="kaiser", qmax_out=20.0, nfft=2048)
    orv = xftr(ok["r"], ok["chir"], **rp)
    write_blocks(
        f"{OUT}/ref_xftr_kaiser.txt", rp,
        dict(
            r=ok["r"], chir_re=ok["chir"].real, chir_im=ok["chir"].imag,
            rwin=orv["rwin"], q=orv["q"], chiq_re=orv["chiq"].real, chiq_im=orv["chiq"].imag,
        ),
    )
    print("wrote xftf_kaiser, xftf_hanning, xftr_kaiser references")


if __name__ == "__main__":
    main()
