#!/usr/bin/env python3
"""Reference generator for the Debye-Waller σ² models (Rust `feffdat::sigma2`).

Emits, for each Cu path, larch's reduced mass (`feffpath.rmass`), `sigma2_eins`
(larch's closed form, no DLL), and `sigma2_debye` over a grid of temperatures /
characteristic temperatures.

For `sigma2_debye` the FEFF6 C library (`libfeff6`) is x86_64-only and will not
load on arm64, so this calls the pure-Python `sigma2_correldebye_py` (a port of
Feff6 `sigms.f`) directly, with the float-conversion the larch build omits
(`feffpath.geom` coordinates are stored as strings). The Rust port targets this
pure-Python reference, not larch's runtime C-DLL path.

Run from the repo root with the project venv:
    .venv/bin/python scripts/ref_sigma2.py
"""
from larch.xafs import feffpath
from larch.xafs.sigma2_models import sigma2_eins, sigma2_correldebye_py

DATADIR = "crates/feffdat/tests/data"
PATHFILES = ["feff0001.dat", "feff0002.dat"]

# (t, theta) grid — spans low/high T and the t==theta crossover
EINS_GRID = [(10.0, 335.0), (150.0, 335.0), (300.0, 335.0),
             (335.0, 335.0), (500.0, 200.0), (1.0e-6, 335.0)]
DEBYE_GRID = [(10.0, 315.0), (150.0, 315.0), (300.0, 315.0),
              (315.0, 315.0), (500.0, 200.0), (1.0e-6, 315.0)]


def fmt(v):
    return repr(float(v))


def main():
    lines = []
    for fn in PATHFILES:
        p = feffpath(f"{DATADIR}/{fn}")
        p.store_feffdat()
        fp = p._feffdat
        natoms = len(fp.geom)
        ax = [float(g[4]) for g in fp.geom]
        ay = [float(g[5]) for g in fp.geom]
        az = [float(g[6]) for g in fp.geom]
        am = [float(g[3]) for g in fp.geom]

        lines.append(f"#path {fn}")
        lines.append(f"#rmass {fmt(fp.rmass)}")
        lines.append(f"#rnorman {fmt(fp.rnorman)}")
        for t, th in EINS_GRID:
            lines.append(f"#eins {fmt(t)} {fmt(th)} {fmt(sigma2_eins(t, th, p))}")
        for t, th in DEBYE_GRID:
            tk = max(float(t), 1.0e-5)
            thd = max(float(th), 1.0e-5)
            val = sigma2_correldebye_py(natoms, tk, thd, fp.rnorman, ax, ay, az, am)
            lines.append(f"#debye {fmt(t)} {fmt(th)} {fmt(val)}")

    with open(f"{DATADIR}/ref_sigma2.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote ref_sigma2.txt")
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__":
    main()
