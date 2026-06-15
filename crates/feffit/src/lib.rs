//! `feffit` — a pure-Rust port of `xraylarch`'s `larch/xafs/feffit.py`.
//!
//! This crate fits a sum of Feff paths to experimental EXAFS chi(k). It builds
//! on [`feffdat`] (the per-path EXAFS equation) and [`xafsft`] (the Fourier
//! transforms and FT windows).
//!
//! This milestone provides the **residual core**: a [`Transform`] (the k/R FT
//! configuration with cached windows) and a [`DataSet`] that produces the fit
//! residual for fixed numeric path parameters in k/R/q space, for one or more
//! k-weights (a list-valued `kweight` concatenates the per-k-weight residuals).
//! Verified against larch's `FeffitDataSet._residual` to FFT round-off.
//!
//! It also provides the **end-to-end fit** ([`fit::feffit`]): the global
//! parameter/constraint system ([`params`]) drives per-path parameter
//! expressions through the residual core and the MINPACK Levenberg-Marquardt
//! minimiser ([`lm`]), then computes the fit statistics and propagates the
//! parameter uncertainties onto the constraint and path parameters by
//! forward-mode automatic differentiation (`stderr(f) = sqrt(gᵀ C g)`, the
//! first-order propagation larch performs with the `uncertainties` package).
//! Verified against larch's `feffit()` on a two-path Cu fit.
//!
//! The `sigma2_eins`/`sigma2_debye` Debye-Waller constraint helpers are
//! available in path expressions, bound to each path's geometry through a
//! [`params::FuncCtx`].
//!
//! Not yet ported: the `'w'` (Cauchy-wavelet) fit space, background refinement
//! (`refine_bkg`), and the GNXAS g(r) model.

pub mod dataset;
pub mod fit;
pub mod transform;

pub use dataset::DataSet;
pub use fit::{
    feffit, Best, FeffitResult, FitDataSet, FitError, PathParam, PathSpec, Spec, PATH_PNAMES,
};
pub use transform::{FitSpace, Transform};
