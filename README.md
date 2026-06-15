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
| `'w'` Cauchy-wavelet fit space (`TransformGroup.cwt` + `'w'` residual) | done | vs **larch** `FeffitDataSet._residual` in `'w'` space (bit-exact: residual max\|Δ\|/peak ≈ 2e-15, pure rustfft vs numpy.fft round-off) |
| `params` (lmfit-style parameters + constraint expressions) | done | vs **asteval** (expr eval, bit-exact) and **lmfit** `update_constraints` (max\|Δ\| < 1e-12) |
| `lm` Levenberg-Marquardt minimiser (MINPACK `lmdif` port) | done | vs **scipy** `optimize.leastsq`: `info`/`nfev` exact for converged cases; x/cov ≈ 1e-9–1e-7 (ULP drift vs scipy's FORTRAN MINPACK) |
| `feffit` end-to-end fit (`fit::feffit`: params → path exprs → residual → LM → statistics) | done | vs **larch** `feffit()` on a 2-path Cu fit: `nfev`/`nvarys`/`ndata` exact; best-fit values ≈ 1e-12–1e-7, uncertainties + chi²/reduced/R-factor/AIC/BIC ≈ 1e-6 |
| Uncertainty propagation onto constraint + path parameters (forward-mode AD, `stderr = sqrt(gᵀ C g)`) | done | AD gradients vs central finite differences (≈ 1e-10); propagated stderrs vs **larch** `eval_stderr`/`uncertainties` on the Cu fit (≈ 1e-4 rel, lmdif ULP drift) |
| Debye-Waller σ² models (`sigma2_eins`, `sigma2_debye`) + `rmass`/atomic masses, callable in path expressions | done | `rmass`/`sigma2_eins`/`sigma2_debye` vs **larch** (eins) and its pure-Python `sigms.f` port (debye, since the Feff6 C lib is x86_64-only) — bit-exact; end-to-end `sigma2_eins` fit + uncertainty vs **larch** (≈ 1e-9) |
| List-valued k-weights (`kweight=[1,2,3]`: per-k-weight residuals concatenated) | done | vs **larch** `feffit()` on the two-path Cu fit with `kweight=[1,2,3]`: `ndata` = 3× single (n_idp unchanged); best-fit values/uncertainties/statistics match to lmdif ULP drift (values ≈ 1e-7, stderr ≈ 1e-4) |
| Parameter bounds (min/max, lmfit Minuit internal↔external transform) | done | vs **larch** `feffit()` on the two-path Cu fit with `amp`/`sig2_1`/`sig2_2` bounded (interior solution): the fit runs on internal coords so `nfev` is exact (31), best-fit values match to lmdif ULP drift, and the gradient-scaled (`cov_ext = g⊗g·cov_int`) uncertainties match (≈ 1e-4 rel) |
| `feff-sys` (FFI to FEFF) | not started | — |

## Layout

```
crates/feffdat/        # parse feffNNNN.dat + compute chi(k)
  src/constants.rs     # KTOE/ETOK, bit-identical to larch xafsutils
  src/parser.rs        # FeffDatFile._read port (incl. path geometry + rmass)
  src/interp.rs        # numpy.interp (exact) + not-a-knot cubic spline
  src/path.rs          # _calc_chi / path2chi / ff2chi
  src/mass.rs          # atomic masses by Z (generated from xraydb)
  src/sigma2.rs        # sigma2_eins / sigma2_debye Debye-Waller models
  tests/parity.rs      # parser + linear-chi parity tests
  tests/sigma2_parity.rs  # rmass / sigma2_eins / sigma2_debye vs larch
  tests/data/          # example .dat files + generated references
crates/xafsft/         # XAFS Fourier transforms (xftf/xftr) + FT windows
  src/bessel.rs        # Cephes I0 (parity with scipy.special.i0)
  src/window.rs        # ftwindow (hanning/kaiser/parzen/welch/…)
  src/transform.rs     # xftf/xftr/*_fast (rustfft)
crates/feffit/         # path-sum fitting core
  src/transform.rs     # TransformGroup: k/R windows, fftf/fftr
  src/dataset.rs       # FeffitDataSet: prepare_fit, residual, epsilon estimation
  src/fit.rs           # feffit(): params + path exprs + LM + statistics
crates/params/         # lmfit-style parameters with constraint expressions
  src/expr.rs          # asteval-subset parser/evaluator (+ AD, FuncCtx hook)
  src/parameters.rs    # Parameters: vary/fixed/expr, dependency-ordered resolve
crates/lm/             # Levenberg-Marquardt least squares (MINPACK lmdif port)
  src/lmdif.rs         # enorm/fdjac2/qrfac/qrsolv/lmpar/lmdif + covariance
scripts/ref_chi.py     # numpy-only reference generator (also emits cubic when scipy present)
scripts/ref_xftf.py    # scipy.fftpack/scipy.special reference for xafsft
scripts/ref_feffit.py  # larch.xafs.feffit residual reference (needs xraylarch)
scripts/ref_feffit_fit.py  # larch.xafs.feffit end-to-end fit reference
scripts/ref_feffit_multikw.py # larch feffit reference for a kweight=[1,2,3] fit
scripts/ref_feffit_bounds.py  # larch feffit reference for a bounded-variable fit
scripts/ref_feffit_sigma2.py  # larch feffit reference for a sigma2_eins fit
scripts/ref_sigma2.py  # larch rmass / sigma2_eins / sigma2_debye reference
scripts/gen_atomic_mass.py # emit crates/feffdat/src/mass.rs from xraydb
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
