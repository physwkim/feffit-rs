//! `lmdif` on an improper problem (MINPACK `info = 0`: here an empty residual
//! vector): no factorization ran, so the covariance is undefined and `covar`
//! must say so instead of indexing the empty Jacobian columns.

use feffit::lm::{LmConfig, lmdif};

#[test]
fn covar_is_none_when_the_input_was_rejected() {
    let cfg = LmConfig {
        ftol: 1.0e-6,
        xtol: 1.0e-6,
        gtol: 1.0e-6,
        maxfev: 100,
        epsfcn: 1.0e-10,
        factor: 100.0,
    };
    let res = lmdif(|_x: &[f64]| Vec::new(), &[1.0, 2.0], &cfg);
    assert_eq!(res.info, 0);
    assert_eq!(res.x, vec![1.0, 2.0]);
    assert!(res.covar().is_none());
}
