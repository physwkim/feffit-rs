#!/usr/bin/env python
"""Reference values for the Cephes gamma port: scipy.special.gamma at sample x.

Covers every branch of Cephes `Gamma`: the (2,3) rational interval, the
recurrence for x<2 and x>=3, the near-zero `small` branch, Stirling for x>33,
and the reflection formula for large negative x. Avoids the negative-integer
poles. Writes `crates/feffit/tests/data/ref_gamma.txt`.

Run from the repo root with the project venv:
    .venv/bin/python scripts/ref_gamma.py
"""
import os
from scipy.special import gamma

XS = [
    0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5, 2.0, 2.5, 2.9, 3.0, 3.7, 4.0, 5.5,
    10.25, 20.0, 33.0, 33.5,
    4.0 / 0.3**2,   # the gnxas q for beta = 0.3
    4.0 / 0.25**2,  # the gnxas q for beta = 0.25
    100.0, 150.0, 170.0, 171.0,
    -0.5, -0.9, -1.5, -2.5, -10.3, -20.7, -33.7, -40.2, -100.9,
]

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(here, "crates", "feffdat", "tests", "data", "ref_gamma.txt")
with open(out, "w") as f:
    f.write("# x gamma(x)  (scipy.special.gamma)\n")
    for x in XS:
        f.write(f"{x!r} {gamma(x)!r}\n")
print(f"wrote {len(XS)} rows to {out}")
