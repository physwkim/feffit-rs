# feffit-rs — remaining work

Status snapshot as of 2026-06-15. This tracks what is done + verified, what is
blocked, and the candidate next milestones (so the next session can pick up
without re-deriving scope).

## Repo / push state

- Branch `main` has **17 commits, none pushed** — there is **no git remote
  configured** (`git remote -v` is empty). The push the user requested
  (`push 후 다음 계속`, choice: "Push main to origin") is **blocked on a remote
  URL**, which must be added by the user:
  ```sh
  ! git remote add origin <your-repo-url>
  git push -u origin main
  ```
  Do **not** fabricate a remote URL.

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

## Candidate next milestones (not yet ported)

### 1. GNXAS g(r) model  — *niche*

A separate modeling paradigm: χ from an asymmetric radial distribution `g(r)`
parametrised by `(N, R, σ, β)` rather than a FEFF path sum.

- **Blocker:** needs the gamma function `Γ(x)` (e.g. a Lanczos port) and a
  different path-sum than `ff2chi`.
- **Effort:** medium; mostly self-contained but a different code path, and used
  by fewer people than FEFF-path fitting.

*(Multi-dataset simultaneous fitting, fit output arrays (`save_outputs`), and
background refinement (`refine_bkg`) are now done + larch-verified — see the
table above. The `refine_bkg` port reproduces the FITPACK knot vector in closed
form (only the knots are needed; larch's coefficients are the fit variables) and
ports `splev`, so no full `splrep`/FITPACK port was required.)*

## Blocked

- **`feff-sys` (FFI to FEFF path generation).** larch's production FEFF6/8l path
  generator ships as **x86_64-only** prebuilt shared libs (`libfeff6.dylib`,
  …) which **will not load on this arm64 Mac**
  (`OSError: incompatible architecture`). The same blocker already forced the
  `sigma2_debye` parity to be checked against larch's pure-Python `sigms.f` port
  instead of the C lib. Unblocking requires rebuilding FEFF for arm64 (separate
  effort).

## Conventions for the next session

- Reference generators live in `scripts/ref_*.py`; run from repo root with the
  project venv (`.venv/bin/python scripts/<name>.py`). larch + xraydb are
  installed there.
- Per-crate checks before commit: `cargo fmt --all`; `cargo clippy -p <crate>
  --all-targets -- -D warnings`; `cargo nextest run -p <crate>` (+ `cargo test
  --doc -p <crate>` when doctests change). Full-workspace variants are owed
  before `git push` / `cargo publish` / tag.
- MINPACK Fortran reference for the `lm` port: `~/codes/minpack`.
