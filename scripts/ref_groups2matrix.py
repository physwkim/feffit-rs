#!/usr/bin/env python
"""Reference generator for `xasproc::lincombo::groups2matrix` parity.

Runs larch's `larch.math.lincombo_fitting.groups2matrix` (the LCF/PCA regridding
step, `interp_kind='cubic'`) on three standards sampled on three *different*
energy grids, so the cubic interpolation of the non-reference curves is actually
exercised. Emits inputs (so the Rust test feeds bit-identical arrays) and the
expected (xdat, matrix) to `crates/feffit/tests/data/ref_groups2matrix.txt`.

larch is imported from its source tree (`$XRAYLARCH` or the default path) with a
stubbed `larch` / `larch.math` package so the heavy package `__init__` chain
(pyshortcuts, wx, …) is bypassed — `groups2matrix` only needs
`larch.math.utils.{interp, index_of}`. Run with an env that has numpy/scipy/lmfit:

    XRAYLARCH=/Users/stevek/codes/xraylarch \
    PYTHONPATH=$XRAYLARCH python scripts/ref_groups2matrix.py
"""

import os
import sys
import types

import numpy as np

LARCH_SRC = os.environ.get("XRAYLARCH", "/Users/stevek/codes/xraylarch")

# Bypass larch's heavy package __init__: register synthetic namespace packages
# for `larch` and `larch.math`, then import the lincombo_fitting module directly.
_larch = types.ModuleType("larch")
_larch.__path__ = [os.path.join(LARCH_SRC, "larch")]
_larch.Group = type("Group", (), {})          # used only by an unrelated import
_larch.isgroup = lambda *a, **k: False
sys.modules["larch"] = _larch
_lmath = types.ModuleType("larch.math")
_lmath.__path__ = [os.path.join(LARCH_SRC, "larch", "math")]
sys.modules["larch.math"] = _lmath

from larch.math.lincombo_fitting import groups2matrix  # noqa: E402


class G:
    """Minimal stand-in for a larch Group (groups2matrix reads .energy/.norm)."""

    def __init__(self, energy, norm):
        self.energy = energy
        self.norm = norm


def main():
    xmin, xmax = 7110.0, 7190.0
    # Reference grid A plus two off-grid standards (B wider+coarser, C narrower+finer).
    eA = np.linspace(7100.0, 7200.0, 60)
    eB = np.linspace(7090.0, 7210.0, 45)
    eC = np.linspace(7105.0, 7195.0, 80)
    nA = np.sin(0.05 * (eA - 7100.0)) + 0.3 * np.cos(0.02 * (eA - 7100.0))
    nB = np.exp(-(((eB - 7150.0) / 30.0) ** 2))
    nC = np.tanh(0.03 * (eC - 7150.0)) + 1.0

    xdat, ydat = groups2matrix(
        [G(eA, nA), G(eB, nB), G(eC, nC)],
        yname="norm",
        xname="energy",
        xmin=xmin,
        xmax=xmax,
    )

    def fmt(a):
        return " ".join("%.17g" % v for v in np.asarray(a).ravel())

    out = os.path.join(
        os.path.dirname(__file__),
        "..",
        "crates",
        "xasproc",
        "tests",
        "data",
        "ref_groups2matrix.txt",
    )
    with open(os.path.abspath(out), "w") as fh:
        fh.write("# groups2matrix parity reference (larch source)\n")
        fh.write("xmin %.17g\n" % xmin)
        fh.write("xmax %.17g\n" % xmax)
        fh.write("eA %s\n" % fmt(eA))
        fh.write("nA %s\n" % fmt(nA))
        fh.write("eB %s\n" % fmt(eB))
        fh.write("nB %s\n" % fmt(nB))
        fh.write("eC %s\n" % fmt(eC))
        fh.write("nC %s\n" % fmt(nC))
        fh.write("grid %s\n" % fmt(xdat))
        fh.write("row0 %s\n" % fmt(ydat[0]))
        fh.write("row1 %s\n" % fmt(ydat[1]))
        fh.write("row2 %s\n" % fmt(ydat[2]))
    print("wrote", os.path.abspath(out))
    print("grid n =", len(xdat), " matrix =", ydat.shape)


if __name__ == "__main__":
    main()
