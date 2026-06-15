//! Differential Kramers-Kronig — port of `larch.xafs.diffkk` (the `diffkk`
//! group's `kk` method, default `how='scalar'`).
//!
//! Steps: match `mu(E)` to the tabulated `f''(E)` with MBACK ([`mback`]),
//! interpolate the *difference* `f2 - fpp` onto an even ~1 eV grid with an even
//! number of points, run the reverse (`f'' -> f'`) Maclaurin-series KK transform
//! on it, interpolate the result back to the input grid, and add it to the
//! tabulated `f1` to form `fp`.
//!
//! As with [`mback`], the tabulated Chantler `f1`/`f2` are *inputs* here rather
//! than looked up internally — feed `xraydb.f1_chantler`/`f2_chantler(z,
//! energy)` (on the deduplicated grid) to reproduce larch bit-exactly.
//!
//! Only the scalar (sequential-sum) KK kernel `kkmclr_sca` is ported: it is the
//! default `how='scalar'` path. The vector form differs only in summation order
//! (numpy pairwise vs sequential) and omits the `TINY` denominator guard.

use std::f64::consts::PI;

use crate::mathutils::remove_dups;
use crate::mback::{MbackParams, mback};

/// smallest tolerated energy step, in eV (`larch` `TINY_ENERGY`).
const TINY_ENERGY: f64 = 0.00050;
/// `larch` `diffkk` `FOPI = 4/pi`.
const FOPI: f64 = 4.0 / PI;
/// `larch` `diffkk` `TINY` denominator floor.
const TINY: f64 = 1e-20;

/// `numpy.linspace(start, stop, num)` with `endpoint=True`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    v[num - 1] = stop;
    v
}

/// `scipy.interpolate.interp1d(kind='linear', bounds_error=False,
/// fill_value=fill)` — `larch.math.interp1d`. `x` must be strictly increasing.
/// Matches scipy's `_call_linear` arithmetic (searchsorted-left, clip to
/// `[1, n-1]`, `slope*(xnew - x_lo) + y_lo`) for bit-exact parity.
fn interp1d_linear(x: &[f64], y: &[f64], xnew: &[f64], fill: f64) -> Vec<f64> {
    let n = x.len();
    xnew.iter()
        .map(|&v| {
            if v < x[0] || v > x[n - 1] {
                return fill;
            }
            // searchsorted(x, v, side='left'), then clip to [1, n-1]
            let idx = x.partition_point(|&xi| xi < v).clamp(1, n - 1);
            let lo = idx - 1;
            let hi = idx;
            let slope = (y[hi] - y[lo]) / (x[hi] - x[lo]);
            slope * (v - x[lo]) + y[lo]
        })
        .collect()
}

/// `larch.xafs.diffkk.kkmclr_sca`: reverse (`f'' -> f'`) Maclaurin-series KK
/// transform. `e` must be on an even grid with an even number of points.
fn kkmclr_sca(e: &[f64], finp: &[f64]) -> Vec<f64> {
    let npts = e.len();
    assert!(npts >= 2, "array too short in kkmclr");
    assert!(
        npts.is_multiple_of(2),
        "array has an odd number of elements in kkmclr"
    );

    let factor = -FOPI * (e[npts - 1] - e[0]) / (npts - 1) as f64;
    let nptsk = npts / 2;
    let mut fout = vec![0.0f64; npts];
    for i in 0..npts {
        let ei2 = e[i] * e[i];
        let ioff = (i % 2) as isize - 1;
        let mut acc = 0.0f64;
        for k in 0..nptsk {
            // j = 2k + ioff; Python's j == -1 wraps to the last element.
            let mut j = 2 * k as isize + ioff;
            if j < 0 {
                j += npts as isize;
            }
            let j = j as usize;
            let mut de2 = e[j] * e[j] - ei2;
            if de2.abs() <= TINY {
                de2 = TINY;
            }
            acc += e[j] * finp[j] / de2;
        }
        fout[i] = acc * factor;
    }
    fout
}

/// Output of [`diffkk`], on the deduplicated energy grid.
#[derive(Debug, Clone)]
pub struct DiffKK {
    /// echoed tabulated `f1(E)`.
    pub f1: Vec<f64>,
    /// echoed tabulated `f2(E)`.
    pub f2: Vec<f64>,
    /// MBACK-matched `f''(E)` (`mback`'s `fpp`).
    pub fpp: Vec<f64>,
    /// KK-transformed `f'(E)`: `f1 + diffKK(f2 - fpp)`.
    pub fp: Vec<f64>,
    /// the even KK grid.
    pub grid: Vec<f64>,
    /// edge energy from the MBACK fit.
    pub e0: f64,
}

/// `larch.xafs.diffkk` + `diffKKGroup.kk` (default `how='scalar'`).
///
/// `f1`/`f2` are the tabulated `f'(E)`/`f''(E)` on the deduplicated `energy`
/// grid (`xraydb.f1_chantler`/`f2_chantler(z, energy)`). `mb` are the MBACK
/// parameters (larch's diffKK uses the defaults: `order=3`, `fit_erfc=False`).
pub fn diffkk(
    energy_in: &[f64],
    mu_in: &[f64],
    f1: &[f64],
    f2: &[f64],
    mb: &MbackParams,
) -> DiffKK {
    let energy = remove_dups(energy_in, TINY_ENERGY);
    let n = energy.len();
    assert_eq!(mu_in.len(), n, "energy and mu length mismatch");
    assert_eq!(f1.len(), n, "energy and f1 length mismatch");
    assert_eq!(f2.len(), n, "energy and f2 length mismatch");

    let matched = mback(&energy, mu_in, f2, Some(f1), mb);
    let fpp = matched.fpp;

    // even grid with an even number of points, ~1 eV spacing
    let span = (energy[n - 1] - energy[0]).trunc() as i64;
    let npts = (span + span % 2) as usize;
    let grid = linspace(energy[0], energy[n - 1], npts);

    // diffKK on (f2 - fpp): forward interp, reverse KK, back interp
    let diff: Vec<f64> = (0..n).map(|j| f2[j] - fpp[j]).collect();
    let fpp_grid = interp1d_linear(&energy, &diff, &grid, 0.0);
    let fp_grid = kkmclr_sca(&grid, &fpp_grid);
    let fp_back = interp1d_linear(&grid, &fp_grid, &energy, 0.0);
    let fp: Vec<f64> = (0..n).map(|j| f1[j] + fp_back[j]).collect();

    DiffKK {
        f1: f1.to_vec(),
        f2: f2.to_vec(),
        fpp,
        fp,
        grid,
        e0: matched.e0,
    }
}
