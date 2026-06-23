# feffit-rs

A Rust port of [xraylarch](https://github.com/xraypy/xraylarch)'s EXAFS
path-fitting core (`feffit` / `feffdat`) and `larch.xafs` data-reduction chain
(`xasproc` / `xasdata`), plus **XAFSView** (`xafsview`) — a desktop GUI that
re-implements the LabVIEW *XAFSView v1.6* (성낙언/POSTECH) EXAFS toolkit on top
of them.

## Scope and the FEFF boundary

xraylarch is **not** pure Python: the EXAFS fitting layer (`feffdat.py`,
`feffit.py`) is Python on top of numpy/scipy/lmfit, but the FEFF path
*generator* (FEFF6 / FEFF8l) is original Fortran shipped as per-OS prebuilt
shared libraries (`libfeff6`, `libfeff8lpath`, `libpotph`, …) and standalone
executables, loaded via `ctypes` / subprocess.

This port keeps that boundary: the EXAFS math (parse `feffNNNN.dat` → χ(k) →
Fourier transform → fit) is ported to Rust, and FEFF path generation stays the
original Fortran (FEFF8L), driven by the `feffrun` crate as a **subprocess**
pipeline of the `feff8l_*` executables rather than reimplemented. Subprocess
keeps the boundary at the `feffNNNN.dat` file interface `feffdat` already parses
bit-for-bit, and decouples the FEFF executable's architecture from this crate's
(an arm64 build drives arm64 `feff8l_*`). The `libfeff8lpath`/`libpotph` shared
libraries are an alternative per-path FFI route not currently used.

```
structure → feff.inp ──▶ [FEFF8L Fortran]  ── subprocess ──▶ feffNNNN.dat   (feffrun; FEFF not ported)
feffNNNN.dat ─▶ FeffDatFile ─▶ path2chi/ff2chi ─▶ xafsft ─▶ feffit             (ported to Rust)
```

The FEFF8L Fortran is built from the [`feff85exafs`](https://github.com/xraypy/feff85exafs)
project (native per-host architecture); `feffrun` finds the `feff8l_*`
executables via the `FEFF8L_DIR` environment variable or `PATH`.

## XAFSView GUI

The `xafsview` crate is a desktop application — [egui](https://github.com/emilk/egui)/eframe
with the `siplot` GPU plotter — that re-implements the LabVIEW **XAFSView v1.6**
EXAFS toolkit on the engines above. The data model lives in `xasdata`, the
reduction math in `xasproc`, and the fitting math in `feffit`/`feffdat`; this
binary is the shell that wires them to the UI. The original LabVIEW manual is the
porting spec, and the section numbers below follow it.

```sh
cargo run -p xafsview --release
```

Path fitting (the Feff tab) needs the FEFF8L `feff8l_*` executables on `PATH` (or
`FEFF8L_DIR`), as above; data reduction, plotting, and the calculators are
self-contained.

**Tabs** — Autobk (import + pre-edge/normalize + AUTOBK), Feffit (path fitting),
Feffit_txt (fit report), Atoms (crystal → `feff.inp`), Feff (edit `feff.inp` /
run FEFF8L), Folders (working directories), About.

**Menu bar**

- **File** — open a data file; quit.
- **Multiple_data** — *Plot Data* multi-group overlay window (stacking, average,
  5-point smoothing, NEXAFS normalize options, multiple-peak catching);
  *Multiple AUTOBK* (reduce every loaded group with one parameter set); *Make
  μ(E) from files* (batch column-import → numbered `.xmu`); *Feffit batch* (one
  fit config per group, with Save Items).
- **Smoothing** — *Edit μ(E)*: deglitch / trim / smooth.
- **Periodic table** — element + atom-data browser.
- **Tools** — wavelet transform |W(k,R)|; LCF; PCA; XANES tools (peak / cursors /
  arctangent subtraction); MBACK / NEXAFS normalization; ion-chamber / gas
  absorption; powder weight; k ↔ E conversion; Extract XAS measured time
  (time-resolved); Plot Sites (3D scattering-cluster viewer).
- **Change BG** — toggle dark / light theme.

Each pop-up is a real OS window (an egui immediate viewport), so it can be moved
independently of the main window.

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
| Multi-dataset simultaneous fit (`feffit(&mut [FitDataSet])`: residual concatenated, `n_idp` summed, shared globals couple datasets) | done | vs **larch** `feffit(params, [ds0, ds1])` on a two-dataset Cu fit (one path each, shared `amp`/`del_e0`/`alpha`, per-dataset σ²): `ndata` = 208 = 2×104, `n_idp` ≈ 2×13.223, `nfev` exact (31); best-fit values/uncertainties match to lmdif ULP drift |
| Fit output arrays (`DataSet::save_outputs`/`_xafsft`: data/model/per-path χ(R) + χ(q), `chir_re`/`im`/`mag`/`pha` + `chiq_*`) | done | vs **larch** `feffit(..., path_outputs=True)` on the two-path Cu fit: data χ(R)/χ(q) (fixed FFT of the data) to round-off (≈ 1e-15 rel), model + per-path arrays to ≈ 1e-12 |
| Background refinement (`refine_bkg`: cubic B-spline background as extra `bkg*` fit variables) + FITPACK `splev`/knot vector | done | `splev` vs **scipy** `interpolate.splev` (≈ 1e-16, incl. extrapolation); end-to-end vs **larch** `feffit(refine_bkg=True)` on the two-path Cu fit (`nspline`=12): `nfev` exact (91), `nvarys`=17, `ndata`=192, knots bit-exact, the 12 `bkg*` coefficients + all values/uncertainties match to lmdif ULP drift |
| FEFF path generation (`feffrun`: subprocess driver for the FEFF8L `feff8l_*` pipeline) | done | full pipeline (`rdinp→pot→xsph→pathfinder→genfmt→ff2x`) on a Cu `feff.inp` → 14 `feffNNNN.dat`, parsed by `feffdat` (first shell `reff` = 2.5527 Å, `nleg` = 2, `degen` = 12); FEFF8L built native arm64 from `feff85exafs` |

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
  src/outputs.rs       # save_outputs/_xafsft: data/model/path chi(R) + chi(q)
  src/bkg.rs           # refine_bkg cubic B-spline: FITPACK splev + knot vector
crates/params/         # lmfit-style parameters with constraint expressions
  src/expr.rs          # asteval-subset parser/evaluator (+ AD, FuncCtx hook)
  src/parameters.rs    # Parameters: vary/fixed/expr, dependency-ordered resolve
crates/lm/             # Levenberg-Marquardt least squares (MINPACK lmdif port)
  src/lmdif.rs         # enorm/fdjac2/qrfac/qrsolv/lmpar/lmdif + covariance
crates/feffrun/        # drive FEFF8L (feff8l_* subprocess) feff.inp -> feffNNNN.dat
  src/lib.rs           # Feff8l runner: pipeline, exe discovery (FEFF8L_DIR/PATH)
  tests/data/feff.inp  # real Cu feff.inp fixture (from feff85exafs)
crates/feffinp/        # crystal cell -> feff.inp + parse feff.inp (input side of the FEFF boundary)
  src/lib.rs           # Crystal::cluster (cell -> FEFF cluster) + feff.inp parser
crates/xasproc/        # larch.xafs reduction: pre_edge/normalize, AUTOBK, rebin, deconvolve (vs larch)
crates/xasdata/        # XAS session model: XasGroup (mu(E)->norm->AUTOBK->FT), Session/Folders, beamline I/O, batch drivers
crates/xafsview/       # the XAFSView desktop GUI (egui/eframe + siplot)
  src/app.rs           # app shell: tabs, menu bar, shared plot, window wiring
  src/reduce_ui.rs     # Autobk tab: import + pre-edge/normalize + AUTOBK
  src/feffit_ui.rs     # Feffit tab: path list, variables, fit + report
  src/feffit_batch.rs  # per-group batch fits + Save Items
  src/plot_data.rs     # Plot Data overlay window (stack/average/peak/normalize)
  src/atoms_ui.rs      # Atoms/Feff tabs + Plot Sites 3D cluster viewer
  src/calc_ui.rs       # Tools calculators: periodic table, gas absorption, powder, k<->E
  src/{analysis,xanes,mback,wavelet,timeres}_ui.rs  # Tools windows: LCF/PCA, XANES, MBACK, wavelet, time-resolved
scripts/ref_chi.py     # numpy-only reference generator (also emits cubic when scipy present)
scripts/ref_xftf.py    # scipy.fftpack/scipy.special reference for xafsft
scripts/ref_feffit.py  # larch.xafs.feffit residual reference (needs xraylarch)
scripts/ref_feffit_fit.py  # larch.xafs.feffit end-to-end fit reference
scripts/ref_feffit_multikw.py # larch feffit reference for a kweight=[1,2,3] fit
scripts/ref_feffit_bounds.py  # larch feffit reference for a bounded-variable fit
scripts/ref_feffit_multidataset.py # larch feffit reference for a 2-dataset simultaneous fit
scripts/ref_feffit_outputs.py # larch feffit save_outputs (chi(R)/chi(q)) reference
scripts/ref_feffit_bkg.py     # larch feffit reference for a refine_bkg fit
scripts/ref_splev.py          # scipy FITPACK splev reference for the bkg spline
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

The fitting + reduction engines are ported from xraylarch (`larch/xafs/*.py`,
notably `feffdat.py` / `feffit.py` / the `pre_edge`/`autobk` chain); upstream is
BSD-licensed. The `xafsview` GUI re-implements the feature set of the LabVIEW
*XAFSView v1.6* (성낙언/POSTECH) EXAFS toolkit, which serves as its porting spec.
