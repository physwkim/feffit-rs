# feffit-rs — remaining work

Status snapshot as of 2026-06-15. This tracks what is done + verified, what is
blocked, and the candidate next milestones (so the next session can pick up
without re-deriving scope).

## Repo / push state

- Branch `main` tracks `origin/main` (remote `origin` =
  `https://github.com/physwkim/feffit-rs.git`, added 2026-06-15). The first 17
  commits are pushed; later commits (push-state doc, `feffrun`) are **ahead of
  origin and unpushed** — pushing requires explicit user confirmation per the
  global rules (`git push` only when asked).

## Done + larch-verified (see README status table)

| Area | Verification |
|------|--------------|
| `feffdat` parser, EXAFS equation, linear + cubic-spline interp | bit-exact vs numpy/scipy |
| `xafsft` Fourier transforms (`xftf`/`xftr`/windows) + raw `fft_padded`/`ifft_padded` | vs scipy.fftpack / scipy.special |
| `feffit` residual core in k/R/q + **`'w'` Cauchy-wavelet** | vs larch `FeffitDataSet._residual` (w: bit-exact, rel ≈ 2e-15) |
| `params` (lmfit-style params + constraint exprs, forward-mode AD) | vs asteval / lmfit |
| `lm` Levenberg-Marquardt (`lmdif` port) | vs scipy `optimize.leastsq` |
| `feffit` end-to-end fit + statistics + uncertainty propagation | vs larch `feffit()` on a 2-path Cu fit |
| Debye-Waller σ² models (`sigma2_eins`, `sigma2_debye`) + `rmass`/atomic masses | vs larch (eins) and its pure-Python `sigms.f` port (debye) |
| **List-valued k-weights** (`kweight=[1,2,3]`) | vs larch `feffit()` (ndata 3×, n_idp unchanged) |
| **Parameter bounds** (min/max, lmfit Minuit internal↔external transform) | vs larch `feffit()` with `amp`/`sig2_1`/`sig2_2` bounded (interior): `nfev` exact (31), values + grad-scaled stderr match |
| **Multi-dataset simultaneous fit** (`feffit(&mut [FitDataSet])`) | vs larch `feffit(params, [ds0, ds1])`: 2 datasets, 1 path each, shared globals; `ndata`=208, `n_idp`≈2×13.223, `nfev` exact |
| **Fit output arrays** (`save_outputs`/`_xafsft`: data/model/path χ(R)+χ(q)) | vs larch `feffit(..., path_outputs=True)`: data χ(R)/χ(q) to round-off (≈1e-15), model+path ≈1e-12 |
| **Background refinement** (`refine_bkg`: cubic B-spline bkg as extra `bkg*` vars) + FITPACK `splev`/knots | `splev` vs scipy (≈1e-16); end-to-end vs larch `feffit(refine_bkg=True)` (nspline=12): `nfev` exact (91), knots bit-exact, 12 bkg coefs match |
| **FEFF path generation** (`feffrun`: subprocess driver for the FEFF8L `feff8l_*` pipeline) | full pipeline on a Cu `feff.inp` → 14 `feffNNNN.dat` parsed by `feffdat` (1st shell reff=2.5527, nleg=2, degen=12); FEFF8L built native arm64 from `feff85exafs` |

## Candidate next milestones (not yet ported)

### 1. GNXAS `gnxas` path-expression helper  — *niche, broken upstream*

`gnxas(r0, sigma, beta)` is a path-expression amplitude helper (registered in
asteval like `sigma2_eins`), not a separate solver. For a path of radius `reff`:
`q = 4/β²`; `alpha = q + 2·(reff−r0)/(β·σ)`; `out = max(0, 2·e^(−alpha)·alpha^(q−1)/(σ·|β|·Γ(q)))`.
Needs the gamma function `Γ(q)` (Cephes port for bit-exactness, or Lanczos).

- **Upstream is broken** in this larch: the asteval-injected `gnxas`
  (`sigma2_models.py` `_sigma2_funcs` string, ~line 379) has a debug
  `print('> ', reff, …)` referencing an undefined `reff` → `NameError`, so a
  feffit path expression using `gnxas(...)` evaluates to `None`. Confirmed:
  removing that one debug line makes it run and its value is **bit-identical**
  to the working module-level `gnxas(r0,sigma,beta,path)` (line 19). So there is
  **no stock-larch end-to-end feffit reference**; parity must be against the
  module-level `gnxas` (the documented-correct formula), with the upstream
  defect noted.
- **Effort:** small (one path helper + a gamma function), but niche.

*(Multi-dataset simultaneous fitting, fit output arrays (`save_outputs`), and
background refinement (`refine_bkg`) are now done + larch-verified — see the
table above. The `refine_bkg` port reproduces the FITPACK knot vector in closed
form (only the knots are needed; larch's coefficients are the fit variables) and
ports `splev`, so no full `splrep`/FITPACK port was required.)*

## Resolved blockers

- **FEFF path generation on arm64 — RESOLVED.** larch's *prebuilt* FEFF6/8l
  libs/executables are all **x86_64-only** (won't load/run native on this arm64
  Mac). Resolution: rebuilt FEFF8L native arm64 from the `feff85exafs` Fortran
  source (`~/codes/feff85exafs`) with Homebrew gfortran 15. Only fix needed was
  `src/GENFMT/Makefile`'s Darwin branch (drop hardcoded `-arch x86_64`; point
  `CLINKARGS` at `/opt/homebrew/lib/gcc/current` instead of the absent
  `/usr/local/gfortran/lib`); gfortran 15 compiled all 411 `.f` files with no
  legacy-flag changes. Products (`feff8l_*`, `feff6l`, `libfeff8lpath.dylib`,
  `libpotph.dylib`) are native arm64. The `feffrun` crate drives the
  `feff8l_*` pipeline as a subprocess (no Rosetta, no FFI). Run the integration
  test with `FEFF8L_DIR=~/codes/feff85exafs/local_install/bin`.
- Note: the `sigma2_debye` parity was earlier checked against larch's
  pure-Python `sigms.f` port (not the x86_64 C lib); that remains as-is.

## Conventions for the next session

- Reference generators live in `scripts/ref_*.py`; run from repo root with the
  project venv (`.venv/bin/python scripts/<name>.py`). larch + xraydb are
  installed there.
- Per-crate checks before commit: `cargo fmt --all`; `cargo clippy -p <crate>
  --all-targets -- -D warnings`; `cargo nextest run -p <crate>` (+ `cargo test
  --doc -p <crate>` when doctests change). Full-workspace variants are owed
  before `git push` / `cargo publish` / tag.
- MINPACK Fortran reference for the `lm` port: `~/codes/minpack`.
