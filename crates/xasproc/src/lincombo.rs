//! XANES linear-combination fitting — port of
//! `larch.math.lincombo_fitting.lincombo_fit` (the default configuration:
//! `sum_to_one=True`, `vary_e0=False`, no explicit per-component bounds).
//!
//! Given a target spectrum `ydat` and `ncomps` component spectra `ycomps[i]`
//! (all already on the target's energy grid — `groups2matrix`'s cubic
//! interpolation onto a common grid is the caller's job), find weights `c_i`
//! minimizing `|| sum_i c_i*ycomps[i] - ydat ||`. larch seeds the weights with
//! an unconstrained `np.linalg.lstsq`, then refines with `lmfit.minimize`
//! imposing the `sum_to_one` constraint (`c_{n-1} = 1 - sum(c_0..c_{n-2})`, a
//! derived parameter — so only `n-1` weights actually vary).
//!
//! Parity is *not* bit-exact: the lstsq seed uses `nalgebra`'s SVD rather than
//! LAPACK `gelsd`, so it differs at ~1e-11. The refined weights converge to the
//! same constrained optimum and agree to ~1e-9 (lmfit's default `ftol`/`xtol`
//! are `1.5e-8`).
//!
//! Only the default config is ported. Per-component bounds and `vary_e0` would
//! additionally need lmfit's MINUIT-style bounded-parameter transform and an
//! energy-shift cubic re-interpolation; those are intentionally omitted.

use lm::{LmConfig, lmdif};
use nalgebra::{DMatrix, DVector};

/// Tunable inputs to [`lincombo_fit`].
#[derive(Debug, Clone)]
pub struct LincomboParams {
    /// force the weights to sum to 1 via a derived last weight. Default true.
    pub sum_to_one: bool,
}

impl Default for LincomboParams {
    fn default() -> Self {
        LincomboParams { sum_to_one: true }
    }
}

/// Output of [`lincombo_fit`].
#[derive(Debug, Clone)]
pub struct Lincombo {
    /// fitted component weights `c_i`.
    pub weights: Vec<f64>,
    /// unconstrained least-squares seed weights (`np.linalg.lstsq`).
    pub weights_lstsq: Vec<f64>,
    /// sum of the fitted weights (1 when `sum_to_one`).
    pub total: f64,
    /// `sum(resid^2)` at the solution.
    pub chisqr: f64,
    /// reduced chi-square `chisqr / (npts - nvarys)`.
    pub redchi: f64,
    /// `sum((ydat-yfit)^2) / sum(ydat^2)`.
    pub rfactor: f64,
    /// the fitted spectrum `sum_i c_i*ycomps[i]`.
    pub yfit: Vec<f64>,
}

/// lmfit's `leastsq` defaults (no tolerance overrides in `lincombo_fit`):
/// `ftol=xtol=1.5e-8`, `gtol=0`, `epsfcn=1e-10`, `maxfev=2*max_nfev` with
/// `max_nfev` defaulting to 100000, `factor=100`.
fn lmfit_default_cfg() -> LmConfig {
    LmConfig {
        ftol: 1.5e-8,
        xtol: 1.5e-8,
        gtol: 0.0,
        maxfev: 200_000,
        epsfcn: 1.0e-10,
        factor: 100.0,
    }
}

/// `np.linalg.lstsq(A, b)` where `A`'s columns are `ycomps[i]`: the minimum-norm
/// least-squares solution via SVD (`nalgebra` in place of LAPACK `gelsd`).
fn lstsq(ycomps: &[Vec<f64>], ydat: &[f64]) -> Vec<f64> {
    let npts = ydat.len();
    let ncomps = ycomps.len();
    let a = DMatrix::from_fn(npts, ncomps, |r, c| ycomps[c][r]);
    let b = DVector::from_row_slice(ydat);
    let svd = a.svd(true, true);
    // keep all singular values well above round-off (components are independent)
    let x = svd.solve(&b, 1.0e-14).expect("lstsq SVD solve failed");
    x.iter().copied().collect()
}

/// `larch.math.lincombo_fitting.lincombo_fit` (default config).
///
/// `ycomps[i]` is component `i` sampled on the same grid as `ydat`. Requires
/// `ncomps >= 2` and every component the same length as `ydat`.
pub fn lincombo_fit(ydat: &[f64], ycomps: &[Vec<f64>], p: &LincomboParams) -> Lincombo {
    let npts = ydat.len();
    let ncomps = ycomps.len();
    assert!(ncomps >= 2, "need at least 2 components");
    for (i, c) in ycomps.iter().enumerate() {
        assert_eq!(c.len(), npts, "component {i} length mismatch");
    }

    let ls_vals = lstsq(ycomps, ydat);

    // varying weights: all of them, or all-but-last under sum_to_one
    let nvary = if p.sum_to_one { ncomps - 1 } else { ncomps };

    // reconstruct the full weight vector from the varying subset
    let full_weights = |v: &[f64]| -> Vec<f64> {
        let mut c = vec![0.0; ncomps];
        c[..nvary].copy_from_slice(&v[..nvary]);
        if p.sum_to_one {
            c[ncomps - 1] = 1.0 - v[..nvary].iter().sum::<f64>();
        }
        c
    };

    let model = |c: &[f64]| -> Vec<f64> {
        (0..npts)
            .map(|j| {
                let mut y = 0.0;
                for (i, comp) in ycomps.iter().enumerate() {
                    y += c[i] * comp[j];
                }
                y
            })
            .collect()
    };

    let resid = |v: &[f64]| -> Vec<f64> {
        let c = full_weights(v);
        let yfit = model(&c);
        (0..npts).map(|j| yfit[j] - ydat[j]).collect()
    };

    let seed: Vec<f64> = ls_vals[..nvary].to_vec();
    let result = lmdif(resid, &seed, &lmfit_default_cfg());

    let weights = full_weights(&result.x);
    let yfit = model(&weights);
    let total: f64 = weights.iter().sum();

    let chisqr: f64 = (0..npts).map(|j| (yfit[j] - ydat[j]).powi(2)).sum();
    let redchi = chisqr / (npts - nvary) as f64;
    let denom: f64 = ydat.iter().map(|&y| y * y).sum();
    let rfactor = (0..npts).map(|j| (ydat[j] - yfit[j]).powi(2)).sum::<f64>() / denom;

    Lincombo {
        weights,
        weights_lstsq: ls_vals,
        total,
        chisqr,
        redchi,
        rfactor,
        yfit,
    }
}
