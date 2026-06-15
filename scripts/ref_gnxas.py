#!/usr/bin/env python
"""Reference values for the gnxas port: larch's module-level `gnxas`.

larch's asteval-injected `gnxas` (the form a feffit path expression would call)
is broken in current larch — its debug `print('> ', reff, ...)` references an
undefined name and raises, so a `gnxas(...)` path expression evaluates to None.
The module-level `gnxas(r0, sigma, beta, path)` is the documented, working
implementation and is numerically identical, so it is the reference here.

Writes `crates/feffdat/tests/data/ref_gnxas.txt`.

Run from the repo root with the project venv:
    .venv/bin/python scripts/ref_gnxas.py
"""
import os
from larch.xafs.sigma2_models import gnxas


class _FeffDat:
    def __init__(self, reff):
        self.reff = reff


class _Path:
    def __init__(self, reff):
        self._feffdat = _FeffDat(reff)


# (r0, sigma, beta, reff): physically sensible, alpha > 0 (finite amplitude).
CASES = [
    (2.5, 0.05, 0.30, 2.55),   # x > 0
    (2.5, 0.08, 0.25, 2.55),
    (2.4, 0.06, 0.40, 2.70),
    (1.5, 0.10, 0.20, 2.00),
    (2.6, 0.05, 0.30, 2.55),   # x < 0, alpha still > 0
    (2.55, 0.07, 0.35, 2.55),  # reff == r0  → x = 0, alpha = q
    (3.0, 0.09, 0.50, 3.20),
]

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(here, "crates", "feffdat", "tests", "data", "ref_gnxas.txt")
with open(out, "w") as f:
    f.write("# r0 sigma beta reff gnxas  (larch module-level gnxas)\n")
    for r0, sigma, beta, reff in CASES:
        v = gnxas(r0, sigma, beta, _Path(reff))
        f.write(f"{r0!r} {sigma!r} {beta!r} {reff!r} {v!r}\n")
print(f"wrote {len(CASES)} rows to {out}")
