# feffit-rs — remaining work

Status snapshot as of 2026-06-15. This tracks what is done + verified, what is
blocked, and the candidate next milestones (so the next session can pick up
without re-deriving scope).

## Repo / push state

- Branch `main` has **10 commits, none pushed** — there is **no git remote
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

## Candidate next milestones (not yet ported)

Pick one — they are independent. Effort/value notes below.

### 1. Parameter bounds (min/max)  — *recommended, highest value*

Real fits bound variables (e.g. `s02 ∈ [0,1]`, `sigma2 ≥ 0`). The port
currently only supports **unbounded** variables (the references use unbounded
vars so lmfit's internal↔external transform is the identity and the fit reduces
to a plain `leastsq`).

- **What:** port lmfit's bounded-parameter transform. lmfit maps a bounded
  external value to an unbounded internal coordinate and minimises over the
  internal coords; at the solution the covariance is scaled by
  `d(external)/d(internal)`.
  - lmfit two-sided (min and max finite): `ext = min + (max-min)*(sin(int)+1)/2`.
  - one-sided (`min` only): `ext = min - 1 + sqrt(int² + 1)`; (`max` only):
    `ext = max + 1 - sqrt(int² + 1)`.
- **Where:** `crates/params` (add bounds to a variable), `crates/lm` (the fit
  runs on internal coords; `feffit` seeds `x0` as internal, maps back to
  external before evaluating constraints), `crates/feffit/src/fit.rs`
  (covariance/stderr rescaling by the transform Jacobian at the solution).
- **Verify:** a bounded fit vs lmfit/larch (`Parameters.add(..., min=, max=)`),
  best-fit values + stderr.
- **Effort:** medium; self-contained in params + lm + fit. No new numerics
  (just the algebraic transform + its derivative).

### 2. `refine_bkg` (background refinement)  — *large*

larch refines a cubic B-spline background as extra fit variables
(`feffit.py` `prepare_fit` lines ~421-435, residual lines ~526-535).

- **What:** `nspline = 1 + round(2*rbkg*(kmax-kmin)/π)` knots over `[kmin,kmax]`;
  create `bkg00..bkgNN` fit vars; residual subtracts
  `_bkg = splev(model.k, [knots, coefs, order])`.
- **Blocker:** needs a **FITPACK** cubic B-spline port — `splrep` (build the
  knot vector + coefficients) and `splev` (de Boor evaluation) — for bit-parity.
  This is a substantial, error-prone numeric port (comparable to the cubic-spline
  interp already in `feffdat/src/interp.rs`, but a different B-spline representation).
- **Also touches:** `feffit()`'s "unused variable" handling, dataset hashkeys,
  and n_idp accounting (bkg vars are excluded from some bookkeeping).
- **Effort:** large.

### 3. GNXAS g(r) model  — *niche*

A separate modeling paradigm: χ from an asymmetric radial distribution `g(r)`
parametrised by `(N, R, σ, β)` rather than a FEFF path sum.

- **Blocker:** needs the gamma function `Γ(x)` (e.g. a Lanczos port) and a
  different path-sum than `ff2chi`.
- **Effort:** medium; mostly self-contained but a different code path, and used
  by fewer people than FEFF-path fitting.

### 4. Fit outputs + multi-dataset verification  — *smaller completion*

- **`save_outputs`:** forward-FT the fitted model and data into
  `chir`/`chir_mag`/`chir_re`/`chir_im` (and `chiq*`) arrays for plotting/use of
  the fit result. Self-contained (just `xftf`/`xftr` on the model/data χ(k)).
- **Multi-dataset:** `feffit(&mut [FitDataSet])` already accepts multiple
  datasets (n_idp sums, residual concatenates), but it is only tested with **one**
  dataset. Add a 2-dataset simultaneous-fit parity test vs larch.
- **Effort:** small.

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
