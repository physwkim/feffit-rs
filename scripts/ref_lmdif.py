#!/usr/bin/env python3
"""Reference generator for the `lm` Rust port (MINPACK lmdif).

Emits one block per least-squares problem with the solution, residuals, fnorm,
function-evaluation count, termination code, and unscaled covariance that
`scipy.optimize.leastsq` (which wraps the same MINPACK `lmdif`) produces with
its default controls. The Rust test reconstructs each residual function with
byte-identical data/grouping and asserts the same outputs.

Run from the repo root with the project venv:
    .venv/bin/python scripts/ref_lmdif.py
"""
import math

import numpy as np
from scipy.optimize import leastsq

OUT = "crates/lm/tests/data"

# ---- problems ---------------------------------------------------------------
# Residual grouping is mirrored EXACTLY in the Rust test so the LM iteration
# path (hence nfev/info) is bit-identical, not merely close.

# linear:  r_i = a*t_i + b - y_i
T_LIN = [0.0, 1.0, 2.0, 3.0, 4.0]
Y_LIN = [1.0, 2.9, 5.1, 6.8, 9.2]


def f_linear(p):
    a, b = p
    return [a * ti + b - yi for ti, yi in zip(T_LIN, Y_LIN)]


# rosenbrock: r = [10*(x1 - x0*x0), 1 - x0]
def f_rosenbrock(p):
    x0, x1 = p
    return [10.0 * (x1 - x0 * x0), 1.0 - x0]


# rational: r_i = p0/(p1 + t_i) - y_i
T_RAT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
Y_RAT = [3.8, 2.1, 1.4, 1.1, 0.9, 0.75]


def f_rational(p):
    p0, p1 = p
    return [p0 / (p1 + ti) - yi for ti, yi in zip(T_RAT, Y_RAT)]


# Powell singular function (J^T J is singular at the solution -> exercises the
# rank-deficient / "no covariance" path).
S5 = math.sqrt(5.0)
S10 = math.sqrt(10.0)


def f_powell(p):
    x0, x1, x2, x3 = p
    return [
        x0 + 10.0 * x1,
        S5 * (x2 - x3),
        (x1 - 2.0 * x2) * (x1 - 2.0 * x2),
        S10 * ((x0 - x3) * (x0 - x3)),
    ]


# exponential decay: r_i = p0*exp(-p1*t_i) - y_i  (transcendental)
T_EXP = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
Y_EXP = [2.5, 2.0, 1.6, 1.3, 1.05, 0.85, 0.7]


def f_expdecay(p):
    p0, p1 = p
    return [p0 * math.exp(-p1 * ti) - yi for ti, yi in zip(T_EXP, Y_EXP)]


CASES = [
    ("linear", [0.0, 0.0], f_linear),
    ("rosenbrock", [-1.2, 1.0], f_rosenbrock),
    ("rational", [1.0, 0.5], f_rational),
    ("powell", [3.0, -1.0, 0.0, 1.0], f_powell),
    ("expdecay", [2.0, 0.3], f_expdecay),
]


def fmt(v):
    return repr(float(v))


def write_ref():
    lines = []
    for name, x0, fcn in CASES:
        x, cov, info, mesg, ier = leastsq(fcn, x0, full_output=True)
        x = np.atleast_1d(x)
        fvec = np.asarray(info["fvec"], dtype=float)
        nfev = int(info["nfev"])
        fnorm = float(np.sqrt(np.sum(fvec * fvec)))
        n = len(x)

        lines.append(f"#case {name}")
        lines.append("#x " + " ".join(fmt(v) for v in x))
        lines.append("#fvec " + " ".join(fmt(v) for v in fvec))
        lines.append("#fnorm " + fmt(fnorm))
        lines.append(f"#nfev {nfev}")
        lines.append(f"#info {int(ier)}")
        if cov is None:
            lines.append("#cov none")
        else:
            flat = np.asarray(cov, dtype=float).reshape(n * n)
            lines.append("#cov " + " ".join(fmt(v) for v in flat))
        lines.append("#end")
        print(f"  {name}: ier={ier} nfev={nfev} x={[round(v, 6) for v in x]} "
              f"cov={'none' if cov is None else 'present'}")

    with open(f"{OUT}/ref_lmdif.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote ref_lmdif.txt ({len(CASES)} cases)")


if __name__ == "__main__":
    write_ref()
