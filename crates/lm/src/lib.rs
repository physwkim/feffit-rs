//! `lm` — a faithful Rust port of MINPACK's `lmdif` (Levenberg-Marquardt least
//! squares with a forward-difference Jacobian), ported from
//! `fortran-lang/minpack`. This is the minimiser `scipy.optimize.leastsq`
//! wraps; `feffit` uses it to fit EXAFS path parameters.
//!
//! ```
//! use lm::{lmdif, LmConfig};
//! // fit a*exp(b*t) to data with a single exponential
//! let t = [0.0, 1.0, 2.0, 3.0, 4.0];
//! let y = [1.0, 2.0, 4.0, 8.0, 16.0];
//! let res = lmdif(
//!     |p: &[f64]| t.iter().zip(y).map(|(&ti, yi)| p[0] * (p[1] * ti).exp() - yi).collect(),
//!     &[1.0, 0.5],
//!     &LmConfig::default(),
//! );
//! assert!(res.info >= 1 && res.info <= 4);
//! assert!((res.x[1] - 2.0_f64.ln()).abs() < 1e-6);
//! ```

mod lmdif;

pub use lmdif::{enorm, lmdif, LmConfig, LmResult};
