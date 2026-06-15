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
| Cubic-spline interp (larch default, `interp='cubic'`) | implemented | **NOT verified** — scipy unavailable; not-a-knot spline is a provisional match to `UnivariateSpline(s=0)` |
| `xafsft` (Fourier transforms) | not started | — |
| `feff-sys` (FFI to FEFF) | not started | — |
| `feffit` (lmfit-equivalent fit) | not started | — |

## Layout

```
crates/feffdat/        # parse feffNNNN.dat + compute chi(k)
  src/constants.rs     # KTOE/ETOK, bit-identical to larch xafsutils
  src/parser.rs        # FeffDatFile._read port
  src/interp.rs        # numpy.interp (exact) + not-a-knot cubic spline
  src/path.rs          # _calc_chi / path2chi / ff2chi
  tests/parity.rs      # parser + linear-chi parity tests
  tests/data/          # example .dat files + generated references
scripts/ref_chi.py     # numpy-only reference generator (also emits cubic when scipy present)
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

If scipy is installed, `ref_chi.py` additionally emits a `chi_cubic` column,
which will let the cubic-spline path be verified to parity (currently pending).

## Provenance

Ported from xraylarch (`larch/xafs/feffdat.py`). Upstream is BSD-licensed.
