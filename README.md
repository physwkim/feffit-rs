# feffit-rs

A Rust port of [xraylarch](https://github.com/xraypy/xraylarch)'s EXAFS path-fitting
core (`feffit` / `feffdat`).

## Scope and the FFI boundary

xraylarch is **not** pure Python: the EXAFS fitting layer (`feffdat.py`,
`feffit.py`) is Python on top of numpy/scipy/lmfit, but the FEFF path
*generator* (FEFF6 / FEFF8l) is original Fortran shipped as per-OS prebuilt
shared libraries (`libfeff6`, `libfeff8lpath`, `libpotph`, …) and standalone
executables, loaded via `ctypes` / subprocess.

This port keeps that boundary: the EXAFS math (parse `feffNNNN.dat` → χ(k) →
Fourier transform → fit) is ported to Rust, and FEFF path generation will be
reached over **FFI** to the existing Fortran libraries rather than reimplemented.

```
structure → feff.inp ──▶ [FEFF6/8l Fortran]  ── FFI ──▶ feffNNNN.dat   (path generation; not ported)
feffNNNN.dat ─▶ FeffDatFile ─▶ path2chi/ff2chi ─▶ xafsft ─▶ feffit       (ported to Rust)
```

## Status

| Component | State | Verification |
|-----------|-------|--------------|
| `feffdat` parser (`FeffDatFile`) | done | values transcribed by hand from `feff0001.dat` |
| EXAFS equation + linear interp (`path2chi`/`ff2chi`, `interp='lin'`) | done | bit-exact vs numpy reference (max\|Δχ\| ≈ 1e-16) |
| Cubic-spline interp (larch default, `interp='cubic'`) | done | vs scipy `UnivariateSpline(s=0)` reference (max\|Δχ\| ≈ 5e-14, incl. extrapolation) |
| `xafsft` (Fourier transforms: `xftf`/`xftr`/windows) | done | vs scipy.fftpack + scipy.special references (kwin ≈ 2e-16, χ(R) ≈ 3e-14, FFT round-off) |
| `feffit` residual core (`Transform`, `DataSet._residual` in k/R/q) | done | vs **larch** `FeffitDataSet._residual` (model χ ≈ 7e-16, residual ≈ 1e-13–3e-11) |
| `params` (lmfit-style parameters + constraint expressions) | done | vs **asteval** (expr eval, bit-exact) and **lmfit** `update_constraints` (max\|Δ\| < 1e-12) |
| `lm` Levenberg-Marquardt minimiser (MINPACK `lmdif` port) | done | vs **scipy** `optimize.leastsq`: `info`/`nfev` exact for converged cases; x/cov ≈ 1e-9–1e-7 (ULP drift vs scipy's FORTRAN MINPACK) |
| `feffit` end-to-end fit (minimiser + statistics) | not started | — |
| `feff-sys` (FFI to FEFF) | not started | — |

## Layout

```
crates/feffdat/        # parse feffNNNN.dat + compute chi(k)
  src/constants.rs     # KTOE/ETOK, bit-identical to larch xafsutils
  src/parser.rs        # FeffDatFile._read port
  src/interp.rs        # numpy.interp (exact) + not-a-knot cubic spline
  src/path.rs          # _calc_chi / path2chi / ff2chi
  tests/parity.rs      # parser + linear-chi parity tests
  tests/data/          # example .dat files + generated references
crates/xafsft/         # XAFS Fourier transforms (xftf/xftr) + FT windows
  src/bessel.rs        # Cephes I0 (parity with scipy.special.i0)
  src/window.rs        # ftwindow (hanning/kaiser/parzen/welch/…)
  src/transform.rs     # xftf/xftr/*_fast (rustfft)
crates/feffit/         # path-sum fitting core
  src/transform.rs     # TransformGroup: k/R windows, fftf/fftr
  src/dataset.rs       # FeffitDataSet: prepare_fit, residual, epsilon estimation
crates/params/         # lmfit-style parameters with constraint expressions
  src/expr.rs          # asteval-subset expression parser/evaluator
  src/parameters.rs    # Parameters: vary/fixed/expr, dependency-ordered resolve
crates/lm/             # Levenberg-Marquardt least squares (MINPACK lmdif port)
  src/lmdif.rs         # enorm/fdjac2/qrfac/qrsolv/lmpar/lmdif + covariance
scripts/ref_chi.py     # numpy-only reference generator (also emits cubic when scipy present)
scripts/ref_xftf.py    # scipy.fftpack/scipy.special reference for xafsft
scripts/ref_feffit.py  # larch.xafs.feffit reference for feffit (needs xraylarch)
scripts/ref_lmdif.py   # scipy.optimize.leastsq reference for the lm minimiser
```

## Build & test

```sh
cargo nextest run -p feffdat      # or: cargo test -p feffdat
cargo clippy -p feffdat --all-targets -- -D warnings
```

## Regenerating references

```sh
python3 scripts/ref_chi.py crates/feffdat/tests/data/feff0001.dat \
        crates/feffdat/tests/data/ref_cu_default.txt
```

When scipy is installed, `ref_chi.py` additionally emits a `chi_cubic` column
(used by the `chi_cubic_*` parity tests). A `.venv` with `numpy`+`scipy` is the
supported way to regenerate it.

## Provenance

Ported from xraylarch (`larch/xafs/feffdat.py`). Upstream is BSD-licensed.
