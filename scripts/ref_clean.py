#!/usr/bin/env python
"""Reference generator for `xasproc::clean` parity.

The deglitch masks resolve their cut points through larch's
`larch.math.utils.{index_of, index_nearest}` ('at or below' / 'nearest'); trim is
a pure inclusive-window comparison. This dumps larch's index_of/index_nearest for
a set of probe energies on a realistic grid, so the Rust parity test asserts both
the resolver parity and that the deglitch masks drop exactly the larch-resolved
indices. larch is imported from its source tree with stubbed package __init__
(see scripts/ref_groups2matrix.py). Run:

    XRAYLARCH=/Users/stevek/codes/xraylarch PYTHONPATH=$XRAYLARCH python scripts/ref_clean.py
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

from larch.math.utils import index_nearest, index_of  # noqa: E402


def fmt(a):
    return " ".join("%.17g" % v for v in np.asarray(a).ravel())


def main():
    grid = np.linspace(7000.0, 7200.0, 81)  # 2.5 eV spacing
    probes = np.array(
        [6990.0, 7000.0, 7001.0, 7001.5, 7099.9, 7100.0, 7123.75, 7200.0, 7210.0]
    )
    idx_of = [index_of(grid, p) for p in probes]
    idx_nr = [index_nearest(grid, p) for p in probes]

    out = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "crates",
            "xasproc",
            "tests",
            "data",
            "ref_clean.txt",
        )
    )
    with open(out, "w") as fh:
        fh.write("# clean parity reference (larch index_of/index_nearest)\n")
        fh.write("grid %s\n" % fmt(grid))
        fh.write("probes %s\n" % fmt(probes))
        fh.write("index_of %s\n" % fmt(idx_of))
        fh.write("index_nearest %s\n" % fmt(idx_nr))
    print("wrote", out, "(", len(grid), "grid,", len(probes), "probes )")


if __name__ == "__main__":
    main()
