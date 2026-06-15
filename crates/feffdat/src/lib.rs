//! `feffdat` — a pure-Rust port of `xraylarch`'s `larch/xafs/feffdat.py`.
//!
//! Reads `feffNNNN.dat` files (the per-path scattering amplitudes/phases that
//! Feff6/Feff8l emit) and evaluates the EXAFS equation to produce chi(k) for a
//! single path or a sum of paths.
//!
//! Status: parser + EXAFS equation + both interpolation modes are validated
//! against a numpy/scipy reference — linear (`numpy.interp`) to ~1e-16 and
//! cubic (larch's default, scipy `UnivariateSpline(s=0)`) to ~5e-14.

pub mod constants;
pub mod interp;
pub mod parser;
pub mod path;

pub use constants::{etok, ktoe, ETOK, KTOE, SMALL_ENERGY};
pub use interp::{interp_linear, CubicSpline, Interp};
pub use parser::{FeffDatFile, GeomAtom, Potential};
pub use path::{ff2chi, path2chi, FeffPath, KGrid, PathParams};
