//! Energy-grid rebinning and sorting — port of `larch.xafs.rebin_xafs`
//! (`sort_xafs` + `rebin_xafs`).
//!
//! `rebin_xafs` resamples `mu(E)` onto a standard 3-region grid (pre-edge in
//! E, XANES in E, EXAFS in k). Each output bin draws from the input points
//! assigned to its energy segment: a boxcar mean, a centroid, a not-a-knot
//! cubic spline (scipy `CubicSpline`), or — for short segments — a NaN-filling
//! linear interpolation.
//!
//! Boxcar / centroid / interp paths are bit-exact vs larch. The `'spline'`
//! method stands a Thomas tridiagonal solve in for scipy's LAPACK
//! `solve_banded`; for these well-conditioned interpolation systems the two
//! agree to round-off (~1e-16 on the Cu foil), see `cubic_spline_at`.

use crate::mathutils::{etok, index_of, ktoe, remove_dups, remove_nans2};

const TINY_ENERGY: f64 = 0.00050;

/// Per-bin reduction used by [`rebin_xafs`] when a segment has ≥ 3 points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RebinMethod {
    /// arithmetic mean of the segment (`larch` `'boxcar'`, the default).
    #[default]
    Boxcar,
    /// energy-weighted centroid `mean(mu*E)/mean(E)` (`larch` `'centroid'`).
    Centroid,
    /// not-a-knot cubic spline through the segment, evaluated at the bin
    /// energy (`larch` `'spline'`, scipy `CubicSpline`).
    Spline,
}

/// Tunable inputs to [`rebin_xafs`]; `None` fields reproduce larch's defaults.
#[derive(Debug, Clone)]
pub struct RebinParams {
    /// energy reference; all region bounds are relative to this.
    pub e0: f64,
    /// start of pre-edge region; `None` → `pre_step*trunc(emin/pre_step)`.
    pub pre1: Option<f64>,
    /// end of pre-edge / start of XANES. Default -30.
    pub pre2: f64,
    /// pre-edge step (eV). Default 2.
    pub pre_step: f64,
    /// XANES step (eV); `None` → `0.05*max(1, int(e0/1250))`.
    pub xanes_step: Option<f64>,
    /// end of XANES / start of EXAFS. Default 15.
    pub exafs1: f64,
    /// end of EXAFS region; `None` → `max(energy)-e0`.
    pub exafs2: Option<f64>,
    /// EXAFS k-step. Default 0.05.
    pub exafs_kstep: f64,
    /// per-bin reduction method. Default boxcar.
    pub method: RebinMethod,
}

impl RebinParams {
    /// larch's defaults around a given `e0`.
    pub fn new(e0: f64) -> Self {
        RebinParams {
            e0,
            pre1: None,
            pre2: -30.0,
            pre_step: 2.0,
            xanes_step: None,
            exafs1: 15.0,
            exafs2: None,
            exafs_kstep: 0.05,
            method: RebinMethod::Boxcar,
        }
    }
}

/// Output of [`rebin_xafs`].
#[derive(Debug, Clone)]
pub struct Rebinned {
    /// new energy grid.
    pub energy: Vec<f64>,
    /// rebinned `mu`.
    pub mu: Vec<f64>,
    /// per-bin standard deviation (`NaN` for empty bins).
    pub delta_mu: Vec<f64>,
    /// e0 used.
    pub e0: f64,
}

/// `larch.xafs.rebin_xafs.sort_xafs`: sort `(energy, mu)` by increasing energy,
/// optionally de-duplicating repeats and removing non-finite points.
pub fn sort_xafs(
    energy: &[f64],
    mu: &[f64],
    fix_repeats: bool,
    remove_nans: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..energy.len()).collect();
    order.sort_by(|&i, &j| {
        energy[i]
            .partial_cmp(&energy[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut new_e: Vec<f64> = order.iter().map(|&i| energy[i]).collect();
    let mut new_mu: Vec<f64> = order.iter().map(|&i| mu[i]).collect();

    if fix_repeats {
        new_e = remove_dups(&new_e, TINY_ENERGY);
    }
    if remove_nans
        && (new_e.iter().any(|v| !v.is_finite()) || new_mu.iter().any(|v| !v.is_finite()))
    {
        let (e, m) = remove_nans2(&new_e, &new_mu);
        new_e = e;
        new_mu = m;
    }
    (new_e, new_mu)
}

/// scipy `interp1d(kind='linear', bounds_error=False, fill_value=nan)` at a
/// single point: linear interpolation inside `[x[0], x[-1]]`, `NaN` outside.
/// `x` is assumed strictly increasing.
fn interp1d_point(x: &[f64], y: &[f64], xv: f64) -> f64 {
    let n = x.len();
    if n < 2 {
        return if n == 1 && xv == x[0] { y[0] } else { f64::NAN };
    }
    if xv < x[0] || xv > x[n - 1] {
        return f64::NAN;
    }
    // searchsorted-left: first index with x[idx] >= xv, clipped to [1, n-1]
    let mut idx = x.partition_point(|&v| v < xv);
    if idx == 0 {
        idx = 1;
    }
    if idx > n - 1 {
        idx = n - 1;
    }
    let lo = idx - 1;
    let slope = (y[idx] - y[lo]) / (x[idx] - x[lo]);
    slope * (xv - x[lo]) + y[lo]
}

fn mean(s: &[f64]) -> f64 {
    s.iter().sum::<f64>() / s.len() as f64
}

/// First derivatives at each node for scipy's not-a-knot `CubicSpline`. Builds
/// the same tridiagonal system scipy does and solves it with the Thomas
/// algorithm (scipy uses LAPACK `solve_banded`). The not-a-knot boundary rows
/// are not diagonally dominant, so Thomas (no pivoting) could in principle
/// differ from LAPACK's pivoted banded LU, but for these well-conditioned
/// interpolation systems the two agree to round-off. `x` strictly increasing,
/// `n >= 3`.
fn cubic_spline_derivs(x: &[f64], dx: &[f64], slope: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 3 {
        // scipy's parabola special case: a 3x3 dense solve. Closed form, since
        // rows 0 and 2 give s0 = 2*slope0 - s1 and s2 = 2*slope1 - s1.
        let s1 = (dx[0] * slope[1] + dx[1] * slope[0]) / (dx[0] + dx[1]);
        return vec![2.0 * slope[0] - s1, s1, 2.0 * slope[1] - s1];
    }

    let mut sub = vec![0.0; n]; // sub[i] multiplies s[i-1]
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n]; // sup[i] multiplies s[i+1]
    let mut rhs = vec![0.0; n];
    for i in 1..n - 1 {
        sub[i] = dx[i];
        diag[i] = 2.0 * (dx[i - 1] + dx[i]);
        sup[i] = dx[i - 1];
        rhs[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
    }
    // not-a-knot start
    let d0 = x[2] - x[0];
    diag[0] = dx[1];
    sup[0] = d0;
    rhs[0] = ((dx[0] + 2.0 * d0) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / d0;
    // not-a-knot end (mirror of the start row: diag uses dx[-2] = dx[n-3])
    let dn = x[n - 1] - x[n - 3];
    diag[n - 1] = dx[n - 3];
    sub[n - 1] = dn;
    rhs[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3]
        + (2.0 * dn + dx[n - 2]) * dx[n - 3] * slope[n - 2])
        / dn;

    // Thomas algorithm
    let mut cprime = vec![0.0; n];
    let mut dprime = vec![0.0; n];
    cprime[0] = sup[0] / diag[0];
    dprime[0] = rhs[0] / diag[0];
    for i in 1..n {
        let m = diag[i] - sub[i] * cprime[i - 1];
        cprime[i] = sup[i] / m;
        dprime[i] = (rhs[i] - sub[i] * dprime[i - 1]) / m;
    }
    let mut s = vec![0.0; n];
    s[n - 1] = dprime[n - 1];
    for i in (0..n - 1).rev() {
        s[i] = dprime[i] - cprime[i] * s[i + 1];
    }
    s
}

/// scipy `CubicSpline(x, y)` (default not-a-knot, `extrapolate=True`) evaluated
/// at a single point `xv`. Solves for the node first derivatives, forms the
/// cubic-Hermite coefficients, and evaluates by Horner — matching scipy's
/// algorithm except for the banded solve (Thomas vs LAPACK; agrees to round-off
/// here, see [`cubic_spline_derivs`]). `x` strictly increasing with `n >= 3`.
fn cubic_spline_at(x: &[f64], y: &[f64], xv: f64) -> f64 {
    let n = x.len();
    let dx: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    let slope: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / dx[i]).collect();
    let s = cubic_spline_derivs(x, &dx, &slope);

    // interval containing xv (extrapolate by clamping to [0, n-2])
    let i = x.partition_point(|&v| v <= xv).saturating_sub(1).min(n - 2);
    let z = xv - x[i];
    let t = (s[i] + s[i + 1] - 2.0 * slope[i]) / dx[i];
    let c0 = t / dx[i];
    let c1 = (slope[i] - s[i]) / dx[i] - t;
    let c2 = s[i];
    let c3 = y[i];
    ((c0 * z + c1) * z + c2) * z + c3
}

/// `numpy.ndarray.std` with `ddof=0` (population standard deviation).
fn std_pop(s: &[f64]) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    let m = mean(s);
    let var = s.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / s.len() as f64;
    var.sqrt()
}

/// `larch.xafs.rebin_xafs.rebin_xafs`: rebin `mu(E)` onto a standard 3-region
/// (pre-edge / XANES / EXAFS) grid relative to `e0`.
pub fn rebin_xafs(energy: &[f64], mu: &[f64], p: &RebinParams) -> Rebinned {
    assert_eq!(energy.len(), mu.len(), "energy and mu length mismatch");
    let n = energy.len();
    assert!(n > 1, "need at least 2 data points");
    let e0 = p.e0;

    let emin = energy.iter().cloned().fold(f64::INFINITY, f64::min) - e0;
    let emax = energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - e0;

    let pre_step = p.pre_step;
    let exafs_kstep = p.exafs_kstep;

    let mut pre1 = p
        .pre1
        .unwrap_or_else(|| pre_step * (emin / pre_step).trunc());
    let mut pre2 = p.pre2;
    let mut exafs1 = p.exafs1;
    let mut exafs2 = p.exafs2.unwrap_or(emax);
    let xanes_step = p
        .xanes_step
        .unwrap_or_else(|| 0.05 * (1.0_f64).max((e0 / 1250.0).trunc()));

    // clip into data range
    pre1 = pre1.max(emin);
    pre2 = pre2.max(pre1 + pre_step.abs()).min(emax);
    exafs1 = exafs1.max(pre2 + xanes_step.abs()).min(emax);
    exafs2 = exafs2.max(exafs1 + exafs_kstep.abs() * 20.0).min(emax);

    // enforce monotonically increasing
    if pre2 <= pre1 {
        pre2 = (pre1 + pre_step.abs()).min(emax);
    }
    if exafs1 <= pre2 {
        exafs1 = (pre2 + xanes_step.abs()).min(emax);
    }
    if exafs2 <= exafs1 {
        exafs2 = (exafs1 + exafs_kstep.abs() * 20.0).min(emax);
    }

    // build the new (absolute) energy grid from the 3 segments
    let mut en: Vec<f64> = Vec::new();
    let segments = [
        (pre1, pre2, pre_step, false),
        (pre2, exafs1, xanes_step, false),
        (exafs1, exafs2, exafs_kstep, true),
    ];
    for &(mut start, mut stop, step, isk) in &segments {
        if start == stop {
            continue;
        }
        if isk {
            start = etok(start);
            stop = etok(stop);
        }
        let npts = 1 + (0.1 + (stop - start).abs() / step) as usize;
        if npts < 2 {
            continue;
        }
        let lin_step = (stop - start) / (npts as f64 - 1.0);
        // np.linspace(start, stop, npts)[:-1] -> i = 0..npts-1
        for i in 0..npts - 1 {
            let mut v = start + i as f64 * lin_step;
            if isk {
                v = ktoe(v);
            }
            en.push(e0 + v);
        }
    }

    // segment boundaries in the input energy array
    let bounds: Vec<usize> = en.iter().map(|&e| index_of(energy, e)).collect();
    let nen = en.len();
    let mut mu_out = Vec::with_capacity(nen);
    let mut err_out = Vec::with_capacity(nen);

    let mut j0: usize = 0;
    for i in 0..nen {
        let j1 = if i == nen - 1 {
            n - 1
        } else {
            // larch int((bounds[i]+bounds[i+1]+1)/2.0) == ceil-div of the sum by 2
            (bounds[i] + bounds[i + 1]).div_ceil(2)
        };
        if i == 0 && j0 == 0 {
            j0 = index_of(energy, en[0] - 5.0);
        }

        let val = if j1.saturating_sub(j0) < 3 {
            let mut jx = (j1 + 1).min(n);
            if jx.saturating_sub(j0) < 3 {
                jx = (jx + 1).min(n);
            }
            let mut v = interp1d_point(&energy[j0..jx], &mu[j0..jx], en[i]);
            if v.is_nan() {
                j0 = j0.saturating_sub(1);
                jx = (jx + 1).min(n);
                v = interp1d_point(&energy[j0..jx], &mu[j0..jx], en[i]);
            }
            v
        } else {
            match p.method {
                RebinMethod::Boxcar => mean(&mu[j0..j1]),
                RebinMethod::Centroid => {
                    let num = mean(
                        &mu[j0..j1]
                            .iter()
                            .zip(&energy[j0..j1])
                            .map(|(&m, &e)| m * e)
                            .collect::<Vec<_>>(),
                    );
                    num / mean(&energy[j0..j1])
                }
                RebinMethod::Spline => cubic_spline_at(&energy[j0..j1], &mu[j0..j1], en[i]),
            }
        };
        mu_out.push(val);
        if j0 == j1 {
            err_out.push(f64::NAN);
        } else {
            // j0 only ever decreases in the retry path, so j0 < j1 holds here
            err_out.push(std_pop(&mu[j0..j1]));
        }
        j0 = j1;
    }

    Rebinned {
        energy: en,
        mu: mu_out,
        delta_mu: err_out,
        e0,
    }
}
