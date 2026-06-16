#!/usr/bin/env python
"""Reference generator for `xasproc::xanes` parity.

Covers the larch/lmfit-comparable XANES primitives:
  * peak / valley  — larch resolves the region with `index_of(lo)`/`index_of(hi)`,
    then numpy argmax/argmin over that inclusive window.
  * arctan_step    — lmfit `StepModel(form='arctan')` (the shape larch's
    `pre_edge_baseline` uses for the edge-step component).
  * centroid       — larch `pre_edge_baseline`'s `(edat*peaks).sum()/peaks.sum()`.

(`x_at_y` is XAFSView's own level-crossing cursor with no larch counterpart, so it
is covered by unit tests only.) larch is imported from source with stubbed package
__init__ (see scripts/ref_groups2matrix.py); lmfit is imported directly. Run:

    XRAYLARCH=/Users/stevek/codes/xraylarch PYTHONPATH=$XRAYLARCH python scripts/ref_xanes.py
"""

import os
import sys
import types

import numpy as np

LARCH_SRC = os.environ.get("XRAYLARCH", "/Users/stevek/codes/xraylarch")

_larch = types.ModuleType("larch")
_larch.__path__ = [os.path.join(LARCH_SRC, "larch")]
_larch.Group = type("Group", (), {})
_larch.isgroup = lambda *a, **k: False
sys.modules["larch"] = _larch
_lmath = types.ModuleType("larch.math")
_lmath.__path__ = [os.path.join(LARCH_SRC, "larch", "math")]
sys.modules["larch.math"] = _lmath

from larch.math.utils import index_of  # noqa: E402
from lmfit.models import StepModel  # noqa: E402


def fmt(a):
    return " ".join("%.17g" % v for v in np.asarray(a).ravel())


def main():
    grid = np.linspace(7000.0, 7200.0, 81)
    # An edge (logistic) plus a white-line bump — a XANES-shaped curve.
    y = 1.0 / (1.0 + np.exp(-(grid - 7110.0) / 3.0)) + 0.4 * np.exp(
        -(((grid - 7115.0) / 4.0) ** 2)
    )

    lo, hi = 7105.0, 7130.0
    a = index_of(grid, lo)
    b = index_of(grid, hi)
    win = slice(a, b + 1)
    imax = a + int(np.argmax(y[win]))
    imin = a + int(np.argmin(y[win]))
    peak = (grid[imax], y[imax])
    valley = (grid[imin], y[imin])

    amp, center, sigma = 1.2, 7112.0, 2.5
    sm = StepModel(form="arctan")
    pars = sm.make_params(amplitude=amp, center=center, sigma=sigma)
    astep = sm.eval(pars, x=grid)

    # centroid weights: the white-line peak component (larch's `peaks`).
    cw = 0.4 * np.exp(-(((grid - 7115.0) / 4.0) ** 2))
    centroid = float((grid * cw).sum() / cw.sum())

    out = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "crates",
            "xasproc",
            "tests",
            "data",
            "ref_xanes.txt",
        )
    )
    with open(out, "w") as fh:
        fh.write("# xanes parity reference (larch index_of + lmfit StepModel arctan)\n")
        fh.write("grid %s\n" % fmt(grid))
        fh.write("y %s\n" % fmt(y))
        fh.write("region %s\n" % fmt([lo, hi]))
        fh.write("peak %s\n" % fmt(peak))
        fh.write("valley %s\n" % fmt(valley))
        fh.write("astep_params %s\n" % fmt([amp, center, sigma]))
        fh.write("astep %s\n" % fmt(astep))
        fh.write("cweights %s\n" % fmt(cw))
        fh.write("centroid %s\n" % fmt([centroid]))
    print("wrote", out)
    print("peak", peak, "valley", valley, "centroid", centroid)


if __name__ == "__main__":
    main()
