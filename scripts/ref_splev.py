#!/usr/bin/env python3
"""Reference for the FITPACK cubic B-spline evaluation `splev` (Rust `feffit`
background-refinement port).

larch's `refine_bkg` builds a cubic B-spline background whose knot vector comes
from `splrep(linspace(kmin,kmax,nspline), dummy, k=3)` (only the KNOTS are kept;
the coefficients are the `bkg00..bkgNN` fit variables) and evaluates it every
iteration with `splev(model.k, [knots, coefs, 3])`. The Rust port reproduces the
knot vector in closed form and ports `splev`; this dumps a parity reference for
`splev` alone — arbitrary coefficients on the real knot vector, evaluated on the
model k-grid (which extends below `kmin` and above `kmax`, exercising the
FITPACK extrapolation that larch relies on for k < kmin).

Run from the repo root with the project venv (scipy installed):
    .venv/bin/python scripts/ref_splev.py
"""
import numpy as np
from scipy.interpolate import splrep, splev

DATADIR = "crates/feffit/tests/data"
KMIN, KMAX, KSTEP = 3.0, 15.0, 0.05


def fmt(v):
    return repr(float(v))


def block(lines, name, arr):
    lines.append(f"#begin {name}")
    lines.extend(fmt(v) for v in arr)
    lines.append("#end")


def main():
    lines = []
    for nspline in (5, 9, 13):
        kk = np.linspace(KMIN, KMAX, nspline)
        ky = np.linspace(-1e-4, 1e-4, nspline)
        knots, _coefs, order = splrep(kk, ky, k=3)
        assert order == 3
        # arbitrary, non-trivial coefficients (length nspline = the bkg vars)
        j = np.arange(nspline)
        coefs = list(0.3 * np.sin(0.7 * j + 0.4) - 0.1 * np.cos(1.3 * j)) \
            + [0.0, 0.0, 0.0, 0.0]   # FITPACK pads coefs to len(knots)
        coefs = np.array(coefs[:len(knots)])
        # model.k-style grid: 0 .. 20, so includes x<kmin (extrapolation) and x>kmax
        x = KSTEP * np.arange(int(1.01 + 20.0 / KSTEP))
        y = splev(x, [knots, coefs, order])
        lines.append(f"#case nspline {nspline}")
        block(lines, f"knots_{nspline}", knots)
        block(lines, f"coefs_{nspline}", coefs)
        block(lines, f"x_{nspline}", x)
        block(lines, f"y_{nspline}", y)
    with open(f"{DATADIR}/ref_splev.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote ref_splev.txt")


if __name__ == "__main__":
    main()
