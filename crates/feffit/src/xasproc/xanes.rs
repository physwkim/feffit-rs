//! XANES analysis primitives — peak/valley search, level-crossing interpolation
//! ("x@y", half-step, half-max), arctangent-step baseline, and peak centroid.
//!
//! These back XAFSView's interactive XANES cursor tools. All operate on
//! co-indexed `(energy, y)` arrays (`y` is typically normalized or flattened
//! μ(E)); region limits are resolved through [`crate::xasproc::mathutils::index_of`] for
//! larch parity (`index of array at or below value`, clamped to 0 below the
//! minimum). The arctangent step matches lmfit's `StepModel(form='arctan')`,
//! which larch's `pre_edge_baseline` uses for the edge-step component, so an
//! interactive arctangent subtraction reproduces that baseline shape.

use crate::xasproc::mathutils::index_of;

/// `(energy, value)` of the **maximum** of `y` over the inclusive energy window
/// `[lo, hi]`, or `None` if the inputs are empty. Bounds may be given in either
/// order. The window endpoints are resolved with [`index_of`] (larch
/// convention): `index_of(energy, lo)` and `index_of(energy, hi)`.
pub fn peak(energy: &[f64], y: &[f64], lo: f64, hi: f64) -> Option<(f64, f64)> {
    extremum(energy, y, lo, hi, true)
}

/// `(energy, value)` of the **minimum** of `y` over `[lo, hi]` — the valley
/// analogue of [`peak`] (XAFSView's "min search").
pub fn valley(energy: &[f64], y: &[f64], lo: f64, hi: f64) -> Option<(f64, f64)> {
    extremum(energy, y, lo, hi, false)
}

fn extremum(energy: &[f64], y: &[f64], lo: f64, hi: f64, want_max: bool) -> Option<(f64, f64)> {
    let n = energy.len().min(y.len());
    if n == 0 {
        return None;
    }
    let slice = &energy[..n];
    let (mut a, mut b) = (index_of(slice, lo), index_of(slice, hi));
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    let mut best = a;
    for i in a..=b {
        let better = if want_max {
            y[i] > y[best]
        } else {
            y[i] < y[best]
        };
        if better {
            best = i;
        }
    }
    Some((energy[best], y[best]))
}

/// The first energy at which `y` crosses `target`, scanning low → high index,
/// linearly interpolated between the two bracketing samples. Returns `None` if
/// no segment of `y` reaches `target`.
///
/// This is XAFSView's "x@y" cursor. With `target` set to half the edge step (on
/// normalized μ this is `0.5`) it gives the **half-step** energy; applied to a
/// sub-slice of the rising edge with `target` at half a peak's height it gives
/// the **half-max** energy. Needs at least one segment (`len >= 2`).
pub fn x_at_y(energy: &[f64], y: &[f64], target: f64) -> Option<f64> {
    let n = energy.len().min(y.len());
    for i in 1..n {
        let (y0, y1) = (y[i - 1], y[i]);
        // [y0, y1] brackets target (rising or falling) when the shifted endpoints
        // have opposite signs (or one is exactly on target).
        if (y0 - target) * (y1 - target) <= 0.0 {
            let denom = y1 - y0;
            if denom.abs() < 1.0e-300 {
                // flat segment sitting on target: report its low edge.
                return Some(energy[i - 1]);
            }
            let frac = (target - y0) / denom;
            return Some(energy[i - 1] + frac * (energy[i] - energy[i - 1]));
        }
    }
    None
}

/// lmfit `StepModel(form='arctan')` sampled on `energy`:
/// `amplitude * (0.5 + atan(sign(sigma)·(E − center)/max(1e-30, |sigma|)) / π)`.
///
/// This is the edge-step shape larch's `pre_edge_baseline` fits; here it is built
/// directly from user-set `amplitude` / `center` / `sigma` for an interactive
/// arctangent subtraction. `sigma > 0` gives a rising step (value 0 well below
/// `center`, → `amplitude` well above, `amplitude/2` at `center`).
pub fn arctan_step(energy: &[f64], amplitude: f64, center: f64, sigma: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    // lmfit: arg = sign(sigma)*(x-center)/max(tiny*tiny, |sigma|), tiny = 1e-15.
    let denom = sigma.abs().max(1.0e-30);
    let sgn = if sigma > 0.0 {
        1.0
    } else if sigma < 0.0 {
        -1.0
    } else {
        0.0
    };
    energy
        .iter()
        .map(|&e| amplitude * (0.5 + (sgn * (e - center) / denom).atan() / PI))
        .collect()
}

/// Intensity-weighted centroid `Σ(energy·weights) / Σ(weights)`, matching larch's
/// `pre_edge_baseline` centroid (`(edat*peaks).sum()/peaks.sum()`). Returns
/// `None` if the arrays are empty/mismatched or the weights sum to zero.
pub fn centroid(energy: &[f64], weights: &[f64]) -> Option<f64> {
    let n = energy.len().min(weights.len());
    if n == 0 {
        return None;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += energy[i] * weights[i];
        den += weights[i];
    }
    if den.abs() < 1.0e-300 {
        return None;
    }
    Some(num / den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn ramp() -> (Vec<f64>, Vec<f64>) {
        // energy 0,1,..,10 ; y a unimodal bump peaking at E=6 (value 4), valley at E=2 (value -1)
        let e: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let y = vec![0.0, -0.5, -1.0, 0.0, 1.5, 3.0, 4.0, 3.2, 2.0, 1.0, 0.5];
        (e, y)
    }

    #[test]
    fn peak_finds_max_in_region_either_bound_order() {
        let (e, y) = ramp();
        assert_eq!(peak(&e, &y, 3.0, 9.0), Some((6.0, 4.0)));
        // bounds reversed give the same window
        assert_eq!(peak(&e, &y, 9.0, 3.0), Some((6.0, 4.0)));
        // restricting away from the peak picks the local max in range
        assert_eq!(peak(&e, &y, 7.0, 10.0), Some((7.0, 3.2)));
    }

    #[test]
    fn valley_finds_min_in_region() {
        let (e, y) = ramp();
        assert_eq!(valley(&e, &y, 0.0, 5.0), Some((2.0, -1.0)));
    }

    #[test]
    fn x_at_y_interpolates_rising_crossing() {
        // straight line y = 2*E over 0..5; crossing y=5 is at E=2.5
        let e: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let y: Vec<f64> = e.iter().map(|&x| 2.0 * x).collect();
        let x = x_at_y(&e, &y, 5.0).unwrap();
        assert!((x - 2.5).abs() < 1e-12, "got {x}");
    }

    #[test]
    fn x_at_y_handles_falling_and_unreached() {
        let e: Vec<f64> = (0..=4).map(|i| i as f64).collect();
        let y = vec![4.0, 3.0, 2.0, 1.0, 0.0]; // falling
        let x = x_at_y(&e, &y, 1.5).unwrap();
        assert!((x - 2.5).abs() < 1e-12, "got {x}");
        // target never reached
        assert_eq!(x_at_y(&e, &y, 9.0), None);
    }

    #[test]
    fn arctan_step_matches_lmfit_shape() {
        let amp = 2.0;
        let center = 10.0;
        let sigma = 1.0;
        let e = vec![center - sigma, center, center + sigma];
        let s = arctan_step(&e, amp, center, sigma);
        // at center-σ: 2*(0.5 + atan(-1)/π) = 0.5 ; at center: 1.0 ; at center+σ: 1.5
        assert!((s[0] - 0.5).abs() < 1e-12, "lo {}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-12, "mid {}", s[1]);
        assert!((s[2] - 1.5).abs() < 1e-12, "hi {}", s[2]);
        // far above approaches amplitude, far below approaches 0
        let far = arctan_step(&[center + 1e6, center - 1e6], amp, center, sigma);
        assert!((far[0] - amp).abs() < amp * 1e-6 / PI + 1e-9);
        assert!(far[1].abs() < amp * 1e-6 / PI + 1e-9);
    }

    #[test]
    fn centroid_of_symmetric_weights_is_center() {
        let e = vec![8.0, 9.0, 10.0, 11.0, 12.0];
        let w = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let c = centroid(&e, &w).unwrap();
        assert!((c - 10.0).abs() < 1e-12, "got {c}");
        // zero weight sum → None
        assert_eq!(centroid(&e, &[0.0; 5]), None);
    }
}
