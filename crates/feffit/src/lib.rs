//! `feffit` — a pure-Rust port of `xraylarch`'s `larch/xafs/feffit.py`.
//!
//! This crate fits a sum of Feff paths to experimental EXAFS chi(k). It builds
//! on [`feffdat`] (the per-path EXAFS equation) and [`xafsft`] (the Fourier
//! transforms and FT windows).
//!
//! This milestone provides the **residual core**: a [`Transform`] (the k/R FT
//! configuration with cached windows) and a [`DataSet`] that produces the fit
//! residual for fixed numeric path parameters in k/R/q space with a scalar
//! k-weight. Verified against larch's `FeffitDataSet._residual` to FFT
//! round-off.
//!
//! It also provides the **end-to-end fit** ([`fit::feffit`]): the global
//! parameter/constraint system ([`params`]) drives per-path parameter
//! expressions through the residual core and the MINPACK Levenberg-Marquardt
//! minimiser ([`lm`]), then computes the fit statistics. Verified against
//! larch's `feffit()` on a two-path Cu fit.
//!
//! Not yet ported: list-valued k-weights, the `'w'` (Cauchy-wavelet) fit space,
//! background refinement (`refine_bkg`), uncertainty propagation onto path
//! parameters, and the `sigma2_debye`/`sigma2_eins` constraint helpers.

pub mod dataset;
pub mod fit;
pub mod transform;

pub use dataset::DataSet;
pub use fit::{feffit, Best, FeffitResult, FitDataSet, FitError, PathSpec, Spec};
pub use transform::{FitSpace, Transform};
