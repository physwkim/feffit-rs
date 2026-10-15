//! Gaussian peak fit of a 1-D curve — the BL10C ScanWindow "Simple Fit"
//! (PyMca `ScanFit`: Specfit theory *Area Gaussians* on a *Linear*
//! background, `ForcePeakPresence`, unweighted): the peaks are estimated from
//! the data, then every parameter is refined by Levenberg–Marquardt
//! ([`feffit::lm::lmdif`]). Used by the In-situ window on the extracted
//! profile.
//!
//! The estimate differs from Specfit's Mariscotti second-derivative search:
//! peaks are the local maxima of the 5-point-smoothed, background-subtracted
//! curve that rise more than 3.5 × the smoothing residual (Specfit's default
//! sensitivity), the background seed is the line through the end-point means.
//! The refined fit does not depend on that seed beyond convergence.

use feffit::lm::{LmConfig, lmdif};

/// `FWHM / σ` of a Gaussian.
const FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949;
/// Specfit's default peak-search sensitivity (σ of the noise).
const SENSITIVITY: f64 = 3.5;

/// One fitted peak (area-normalised Gaussian) with 1-σ uncertainties.
#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    pub center: f64,
    pub fwhm: f64,
    pub area: f64,
    pub center_err: f64,
    pub fwhm_err: f64,
    pub area_err: f64,
}

/// The result of [`fit_area_gaussians`].
#[derive(Debug, Clone, PartialEq)]
pub struct PeakFit {
    /// Peaks in ascending `center` order.
    pub peaks: Vec<Peak>,
    /// Linear background `offset + slope · x`.
    pub offset: f64,
    pub slope: f64,
    /// The fitted curve on the input `x`.
    pub model: Vec<f64>,
    /// Σ(y − model)² / (points − parameters): the unweighted reduced χ² the
    /// uncertainties are scaled by.
    pub chi2_reduced: f64,
}

/// Fit `y(x)` with up to `max_peaks` Gaussians (area form) on a linear
/// background. `x` need not be ascending or evenly spaced. Errors: fewer than
/// five finite points, or more parameters than points.
pub fn fit_area_gaussians(x: &[f64], y: &[f64], max_peaks: usize) -> Result<PeakFit, String> {
    let n = x.len().min(y.len());
    if n < 5 {
        return Err("need at least 5 points".into());
    }
    let (x, y) = (&x[..n], &y[..n]);
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err("non-finite values in the curve".into());
    }
    let max_peaks = max_peaks.max(1);

    // Background seed: the line through the first and last three points' means.
    let m = 3.min(n / 2);
    let mean = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
    let (x0, y0) = (mean(&x[..m]), mean(&y[..m]));
    let (x1, y1) = (mean(&x[n - m..]), mean(&y[n - m..]));
    let slope0 = if x1 != x0 { (y1 - y0) / (x1 - x0) } else { 0.0 };
    let offset0 = y0 - slope0 * x0;
    let resid: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| yi - (offset0 + slope0 * xi))
        .collect();

    // Peak seeds from the smoothed residual.
    let smooth = smooth5(&resid);
    let span =
        x.iter().cloned().fold(f64::MIN, f64::max) - x.iter().cloned().fold(f64::MAX, f64::min);
    let min_sigma = (span / n as f64).abs().max(1e-12);
    let mut p = vec![offset0, slope0];
    for i in find_peaks(&resid, &smooth, max_peaks) {
        let (l, r) = half_max_span(&smooth, i);
        let fwhm = (x[r] - x[l]).abs().max(FWHM_PER_SIGMA * min_sigma);
        let sigma = fwhm / FWHM_PER_SIGMA;
        let area = smooth[i] * sigma * (2.0 * std::f64::consts::PI).sqrt();
        p.extend([area, x[i], sigma]);
    }
    if p.len() >= n {
        return Err(format!(
            "{} parameters for {n} points — too few points for {} peak(s)",
            p.len(),
            (p.len() - 2) / 3
        ));
    }

    let res = lmdif(
        |p| {
            let model = model(p, x, min_sigma);
            y.iter().zip(&model).map(|(yi, mi)| yi - mi).collect()
        },
        &p,
        &LmConfig::default(),
    );
    let covar = res.covar();
    let p = res.x;
    let dof = (n - p.len()) as f64;
    let chi2_reduced = res.fnorm * res.fnorm / dof;
    // Unweighted data: scale (JᵀJ)⁻¹ by the reduced χ² (scipy `curve_fit`
    // with `absolute_sigma=False`, which is what Specfit reports unweighted).
    let err = |k: usize| -> f64 {
        covar
            .as_ref()
            .map(|c| (c[k][k] * chi2_reduced).max(0.0).sqrt())
            .unwrap_or(f64::NAN)
    };
    let mut peaks: Vec<Peak> = (0..(p.len() - 2) / 3)
        .map(|j| {
            let k = 2 + 3 * j;
            let sigma = p[k + 2].abs().max(min_sigma);
            Peak {
                area: p[k],
                center: p[k + 1],
                fwhm: FWHM_PER_SIGMA * sigma,
                area_err: err(k),
                center_err: err(k + 1),
                fwhm_err: FWHM_PER_SIGMA * err(k + 2),
            }
        })
        .collect();
    peaks.sort_by(|a, b| a.center.total_cmp(&b.center));
    Ok(PeakFit {
        peaks,
        offset: p[0],
        slope: p[1],
        model: model(&p, x, min_sigma),
        chi2_reduced,
    })
}

/// `offset + slope·x + Σ area / (σ√2π) · exp(−(x − c)² / 2σ²)`, σ clamped to
/// `min_sigma` so the minimiser cannot collapse a peak onto one point.
fn model(p: &[f64], x: &[f64], min_sigma: f64) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let mut v = p[0] + p[1] * xi;
            for g in p[2..].as_chunks::<3>().0 {
                let sigma = g[2].abs().max(min_sigma);
                let z = (xi - g[1]) / sigma;
                v += g[0] / (sigma * (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * z * z).exp();
            }
            v
        })
        .collect()
}

/// 5-point moving average (the window shrinks at the ends).
fn smooth5(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(n);
            v[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        })
        .collect()
}

/// Indices of up to `max_peaks` local maxima of `smooth` that rise above
/// `SENSITIVITY` × the smoothing residual's RMS, strongest first, skipping a
/// maximum that lies inside an already-taken peak's half-max span (the same
/// peak's shoulder); ascending. Always at least one (the global maximum) —
/// Specfit's `ForcePeakPresence`.
fn find_peaks(resid: &[f64], smooth: &[f64], max_peaks: usize) -> Vec<usize> {
    let n = smooth.len();
    let noise = (resid
        .iter()
        .zip(smooth)
        .map(|(r, s)| (r - s) * (r - s))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let threshold = SENSITIVITY * noise;
    let mut cands: Vec<usize> = (1..n - 1)
        .filter(|&i| {
            smooth[i] >= smooth[i - 1] && smooth[i] > smooth[i + 1] && smooth[i] > threshold
        })
        .collect();
    cands.sort_by(|&a, &b| smooth[b].total_cmp(&smooth[a]));
    let mut taken: Vec<usize> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for i in cands {
        if taken.len() == max_peaks {
            break;
        }
        if spans.iter().all(|&(l, r)| i < l || i > r) {
            taken.push(i);
            spans.push(half_max_span(smooth, i));
        }
    }
    if taken.is_empty() {
        let argmax = (0..n)
            .max_by(|&a, &b| smooth[a].total_cmp(&smooth[b]))
            .unwrap_or(0);
        taken.push(argmax);
    }
    taken.sort_unstable();
    taken
}

/// The index span `(left, right)` around the maximum at `i` where `v` stays
/// above half of `v[i]` (clamped to the ends, at least one point each side).
fn half_max_span(v: &[f64], i: usize) -> (usize, usize) {
    let half = v[i] / 2.0;
    let mut l = i;
    while l > 0 && v[l - 1] > half {
        l -= 1;
    }
    let mut r = i;
    while r + 1 < v.len() && v[r + 1] > half {
        r += 1;
    }
    (l.min(i.saturating_sub(1)), r.max((i + 1).min(v.len() - 1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(x: f64, area: f64, c: f64, fwhm: f64) -> f64 {
        let s = fwhm / FWHM_PER_SIGMA;
        area / (s * (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * ((x - c) / s).powi(2)).exp()
    }

    /// Two separated Gaussians on a sloped line: the fit recovers every
    /// parameter, in ascending-center order, with a near-zero reduced χ².
    #[test]
    fn recovers_two_gaussians_on_a_linear_background() {
        let x: Vec<f64> = (0..400).map(|i| 6530.0 + i as f64 * 0.25).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&e| {
                0.2 + 0.001 * (e - 6530.0) + gauss(e, 3.0, 6552.0, 2.5) + gauss(e, 1.2, 6570.0, 4.0)
            })
            .collect();
        let fit = fit_area_gaussians(&x, &y, 5).expect("fit");
        assert_eq!(fit.peaks.len(), 2, "{:?}", fit.peaks);
        let (a, b) = (&fit.peaks[0], &fit.peaks[1]);
        assert!((a.center - 6552.0).abs() < 1e-3, "{a:?}");
        assert!((a.fwhm - 2.5).abs() < 1e-3, "{a:?}");
        assert!((a.area - 3.0).abs() < 1e-3, "{a:?}");
        assert!((b.center - 6570.0).abs() < 1e-3, "{b:?}");
        assert!((b.fwhm - 4.0).abs() < 1e-3, "{b:?}");
        assert!((b.area - 1.2).abs() < 1e-3, "{b:?}");
        assert!((fit.slope - 0.001).abs() < 1e-6);
        assert!((fit.offset - (0.2 - 0.001 * 6530.0)).abs() < 1e-3);
        assert!(fit.chi2_reduced < 1e-12, "{}", fit.chi2_reduced);
        assert_eq!(fit.model.len(), x.len());
    }

    /// Specfit's ForcePeakPresence: a curve with no peak above the noise still
    /// fits exactly one Gaussian, seeded at the maximum; the cap on the peak
    /// count holds when more maxima qualify.
    #[test]
    fn forces_one_peak_and_honours_the_cap() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let flat: Vec<f64> = x.iter().map(|v| 1.0 + 0.01 * (v * 0.7).sin()).collect();
        let fit = fit_area_gaussians(&x, &flat, 4).expect("fit");
        assert_eq!(fit.peaks.len(), 1);

        let three: Vec<f64> = x
            .iter()
            .map(|&v| {
                gauss(v, 5.0, 10.0, 3.0) + gauss(v, 5.0, 25.0, 3.0) + gauss(v, 5.0, 40.0, 3.0)
            })
            .collect();
        let fit = fit_area_gaussians(&x, &three, 2).expect("fit");
        assert_eq!(fit.peaks.len(), 2);
        let fit = fit_area_gaussians(&x, &three, 3).expect("fit");
        assert_eq!(fit.peaks.len(), 3);
        let centers: Vec<f64> = fit.peaks.iter().map(|p| p.center).collect();
        for (c, want) in centers.iter().zip([10.0, 25.0, 40.0]) {
            assert!((c - want).abs() < 1e-3, "{centers:?}");
        }
    }

    /// Too few points, and non-finite values, are refused.
    #[test]
    fn rejects_short_or_non_finite_curves() {
        assert!(fit_area_gaussians(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0], 1).is_err());
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut y = vec![0.0; 10];
        y[5] = f64::NAN;
        assert!(fit_area_gaussians(&x, &y, 1).is_err());
    }
}
