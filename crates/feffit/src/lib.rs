//! `feffit` — a pure-Rust port of `xraylarch`'s `larch/xafs/feffit.py`.
//!
//! This crate fits a sum of Feff paths to experimental EXAFS chi(k). It builds
//! on [`feffdat`] (the per-path EXAFS equation) and [`xafsft`](mod@xafsft) (the Fourier
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
//! [`crate::params::FuncCtx`].
//!
//! After a fit, [`DataSet::save_outputs`] forward/back-transforms the data and
//! model (and optionally each path) χ(k) into χ(R)/χ(q) output arrays (larch
//! `save_outputs`/`_xafsft`).
//!
//! Background refinement ([`DataSet::enable_refine_bkg`]) fits a cubic B-spline
//! background as extra variables (larch `refine_bkg`), with the FITPACK spline
//! pieces in [`bkg`].
//!
//! Not yet ported: the GNXAS g(r) model.

pub mod bkg;
pub mod dataset;
pub mod fit;
pub mod outputs;
pub mod transform;

// Absorbed sub-crates. These were sibling workspace crates; they now live here
// as modules so the whole engine ships as a single publishable crate. Their
// public APIs are unchanged — only the path (`feffit::feffdat::…` instead of
// `feffdat::…`) differs.
pub mod feffdat;
pub mod feffinp;
pub mod feffrun;
pub mod lm;
pub mod params;
pub mod textio;
pub mod xafsft;
pub mod xasdata;
pub mod xasproc;

pub use dataset::DataSet;
pub use fit::{
    Best, FeffitResult, FitDataSet, FitError, PATH_PNAMES, PathParam, PathSpec, Spec, feffit,
    feffit_eval,
};
pub use outputs::{DataSetOutput, XafsOutput, xafsft};
pub use transform::{FitSpace, Transform};
