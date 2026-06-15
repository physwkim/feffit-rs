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
//! Not yet ported: list-valued k-weights, the `'w'` (Cauchy-wavelet) fit space,
//! background refinement (`refine_bkg`), the parameter/constraint expression
//! system, and the Levenberg-Marquardt minimiser.

pub mod dataset;
pub mod transform;

pub use dataset::DataSet;
pub use transform::{FitSpace, Transform};
