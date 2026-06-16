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

use crate::mathutils::{index_of, interp_cubic};

/// `larch.math.lincombo_fitting.groups2matrix` for energy/`norm`-style arrays
/// (`interp_kind='cubic'`).
///
/// `curves[0]` is the reference whose native x-grid — sliced to `[xmin, xmax]`
/// by [`index_of`] — becomes the common grid; its y is taken **as-is** (not
/// interpolated, exactly as larch leaves the first group). Every other curve is
/// cubic-interpolated onto that grid via [`interp_cubic`] (larch
/// `interp(kind='cubic')`).
///
/// Returns `(xdat, rows)` where `rows[0]` is the reference y on `xdat` and
/// `rows[i]` is curve `i` interpolated onto `xdat`; `None` if `curves` is empty,
/// the reference is malformed, or the sliced grid has fewer than 2 points.
///
/// This is the regridding step shared by LCF ([`lincombo_fit`], reference =
/// the unknown) and PCA ([`pca_train`](crate::pca::pca_train), reference = the
/// first training standard). It replaces a uniform-grid + linear resample, which
/// diverged from larch on both the grid choice and the interpolation kind.
pub fn groups2matrix(
    curves: &[(&[f64], &[f64])],
    xmin: f64,
    xmax: f64,
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let (x0, y0) = *curves.first()?;
    if x0.len() < 2 || x0.len() != y0.len() {
        return None;
    }
    let imin = index_of(x0, xmin);
    let imax = index_of(x0, xmax) + 1;
    if imax <= imin {
        return None;
    }
    let xdat = x0[imin..imax].to_vec();
    if xdat.len() < 2 {
        return None;
    }
    let mut rows = Vec::with_capacity(curves.len());
    rows.push(y0[imin..imax].to_vec());
    for &(x, y) in &curves[1..] {
        rows.push(interp_cubic(x, y, &xdat));
    }
    Some((xdat, rows))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups2matrix_uses_reference_native_grid_sliced() {
        // The reference (curves[0]) keeps its own x-points within [xmin, xmax];
        // index_of is "at or below", so xmax=3.0 includes the node at 3.0.
        let x0 = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y0 = [10.0, 11.0, 12.0, 13.0, 14.0];
        let xc = [0.0, 1.0, 2.0, 3.0, 4.0];
        let yc = [0.0, 2.0, 4.0, 6.0, 8.0];
        let (grid, rows) =
            groups2matrix(&[(&x0[..], &y0[..]), (&xc[..], &yc[..])], 1.0, 3.0).expect("matrix");
        assert_eq!(grid, vec![1.0, 2.0, 3.0]);
        // reference row is the native slice, untouched
        assert_eq!(rows[0], vec![11.0, 12.0, 13.0]);
        // a component sampled on a coincident grid interpolates to itself
        for (got, want) in rows[1].iter().zip([2.0, 4.0, 6.0]) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn groups2matrix_cubic_interp_off_grid() {
        // Component on a finer/offset grid is cubic-interpolated onto the
        // reference grid; a quadratic is reproduced (cubic spline exact on a
        // cubic-or-lower polynomial sampled densely enough).
        let x0 = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y0 = [0.0; 5];
        let xc: Vec<f64> = (0..=40).map(|i| i as f64 * 0.1).collect();
        let yc: Vec<f64> = xc.iter().map(|&x| x * x).collect();
        let (grid, rows) =
            groups2matrix(&[(&x0[..], &y0[..]), (&xc[..], &yc[..])], 0.0, 4.0).expect("matrix");
        assert_eq!(grid, x0.to_vec());
        for (&g, r) in grid.iter().zip(&rows[1]) {
            assert!((r - g * g).abs() < 1e-6, "x={g}: got {r}, want {}", g * g);
        }
    }

    #[test]
    fn groups2matrix_rejects_empty_or_degenerate_range() {
        let x0 = [0.0, 1.0, 2.0, 3.0];
        let y0 = [0.0, 1.0, 2.0, 3.0];
        // no curves
        assert!(groups2matrix(&[], 0.0, 1.0).is_none());
        // xmin above the whole grid → < 2 points
        assert!(groups2matrix(&[(&x0[..], &y0[..])], 9.0, 10.0).is_none());
        // inverted range
        assert!(groups2matrix(&[(&x0[..], &y0[..])], 3.0, 0.0).is_none());
    }

    #[test]
    fn lincombo_on_coincident_grid_recovers_known_weights() {
        // Build a target as 0.3*A + 0.7*B on the same grid → fit recovers it.
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let a: Vec<f64> = x.iter().map(|&v| (v * 0.5).sin()).collect();
        let b: Vec<f64> = x.iter().map(|&v| (v * 0.3).cos()).collect();
        let target: Vec<f64> = (0..x.len()).map(|i| 0.3 * a[i] + 0.7 * b[i]).collect();
        let (_grid, rows) = groups2matrix(
            &[(&x[..], &target[..]), (&x[..], &a[..]), (&x[..], &b[..])],
            -f64::INFINITY,
            f64::INFINITY,
        )
        .expect("matrix");
        let comps = rows[1..].to_vec();
        let lc = lincombo_fit(&rows[0], &comps, &LincomboParams { sum_to_one: true });
        assert!((lc.weights[0] - 0.3).abs() < 1e-6, "w0={}", lc.weights[0]);
        assert!((lc.weights[1] - 0.7).abs() < 1e-6, "w1={}", lc.weights[1]);
    }
}
