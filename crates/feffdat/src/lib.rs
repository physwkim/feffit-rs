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
pub mod gamma;
pub mod gnxas;
pub mod interp;
pub mod mass;
pub mod parser;
pub mod path;
pub mod sigma2;

pub use constants::{ETOK, KTOE, SMALL_ENERGY, etok, ktoe};
pub use gamma::gamma;
pub use gnxas::gnxas;
pub use interp::{CubicSpline, Interp, interp_linear};
pub use mass::atomic_mass;
pub use parser::{FeffDatFile, GeomAtom, Potential};
pub use path::{FeffPath, KGrid, PathParams, ff2chi, path2chi};
pub use sigma2::{EINS_FACTOR, sigma2_debye, sigma2_eins};
