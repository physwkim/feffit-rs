# feffit-rs — remaining work

Status snapshot as of 2026-06-15. This tracks what is done + verified, what is
blocked, and the candidate next milestones (so the next session can pick up
without re-deriving scope).

## Repo / push state

- Branch `main` tracks `origin/main` (remote `origin` =
  `https://github.com/physwkim/feffit-rs.git`, added 2026-06-15). The first 17
  commits are pushed; the next 3 (push-state doc, `feffrun`, `gnxas`/gamma) are
  **ahead of origin and unpushed** — pushing requires explicit user confirmation
  per the global rules (`git push` only when asked).

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
| **GNXAS `gnxas` path-amplitude helper** (`feffdat::gnxas`, wired into feffit path expressions; needs `feffdat::gamma`, a Cephes `Gamma` port) | gamma vs `scipy.special.gamma` and gnxas vs larch's module-level `gnxas` both **bit-exact** (max rel err 0e0). Upstream asteval-injected `gnxas` is broken (`NameError` on an undefined `reff`), so parity is against the numerically-identical module-level `gnxas` |

## Candidate next milestones (not yet ported)

*(Multi-dataset simultaneous fitting, fit output arrays (`save_outputs`),
background refinement (`refine_bkg`), and the GNXAS `gnxas` path helper are now
done + larch-verified — see the table above. The `refine_bkg` port reproduces
the FITPACK knot vector in closed form (only the knots are needed; larch's
coefficients are the fit variables) and ports `splev`, so no full
`splrep`/FITPACK port was required.)*

No further larch feffit/feffdat features are currently queued. Remaining
candidate work is integration polish (an end-to-end `feffrun` → `feffit`
capstone test that generates `feffNNNN.dat` and fits them in one flow) rather
than new ports.

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
