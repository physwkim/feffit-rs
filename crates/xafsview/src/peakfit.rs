//! Gaussian peak fit of a 1-D curve — the BL10C ScanWindow "Simple Fit"
//! (PyMca `ScanFit`: Specfit theory *Area Gaussians* on a *Linear*
//! background, `ForcePeakPresence`, unweighted), ported from silx's
//! `silx.math.fit` — the maintained Specfit lineage: `fitmanager`,
//! `fittheories`, `bgtheories`, `peaks.c`, `strip.c`, `smoothnd.c`,
//! `leastsq`. Used by the In-situ window on the extracted profile.
//!
//! The pipeline, as `FitManager.estimate` + `runfit` run it:
//!
//! 1. background estimate (`bgtheories.estimate_linear`): strip the peaks
//!    (width 2, 5000 iterations) and fit a line through what is left;
//! 2. peak estimate (`fittheories.estimate_agauss`): a Savitzky–Golay(5) +
//!    strip background; the Mariscotti search (`peaks.c` `seek`) with the
//!    FWHM guessed from the largest peak (`guess_fwhm`) and sensitivity 2.5,
//!    or the global maximum when nothing is found (`ForcePeakPresence`);
//!    four Gefit iterations of height/position/FWHM Gaussians on the
//!    background-subtracted data (height ≥ 0, position within ±FWHM/2,
//!    FWHM ≥ 0); heights converted to areas;
//! 3. the fit (`leastsq`, Gefit's Marquardt loop): line + area Gaussians on
//!    the raw data, unweighted, area and FWHM constrained positive.
//!
//! Deviations from silx, each deliberate:
//!
//! - `AutoScaling` is on (silx's default is off): the search's significance
//!   test assumes Poisson variance (counts), and the profiles here are
//!   normalised values ≲ 1, for which the unscaled test finds nothing. The
//!   scaling is silx's own `guess_yscaling`.
//! - the peak count is capped (`max_peaks`, highest Mariscotti relevance
//!   kept); silx has no cap.
//! - uncertainties are scaled by √(reduced χ²); Gefit/silx report the
//!   unit-variance ones when no σ is given.

/// `FWHM / σ` of a Gaussian, `2·√(2·ln 2)`.
const FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949;
/// `fittheories.DEFAULT_CONFIG`: the Mariscotti sensitivity (in noise σ),
/// the strip background's width and iteration count, the Savitzky–Golay
/// width.
const SENSITIVITY: f64 = 2.5;
const STRIP_WIDTH: usize = 2;
const STRIP_ITERATIONS: usize = 5000;
const SMOOTHING_WIDTH: usize = 5;
/// `guess_fwhm`'s floor on the search FWHM, in points.
const FWHM_GUESS_MIN: usize = 4;
/// Gefit iterations: the estimate's quick refinement and the fit proper.
const ESTIMATE_ITERATIONS: usize = 4;
const FIT_ITERATIONS: usize = 100;
/// `leastsq`'s stop: a χ² decrease below this percentage ends the fit.
const DELTACHI_PERCENT: f64 = 0.001;

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
/// five finite points, more parameters than points, or a singular normal
/// matrix (silx's `LinAlgError`).
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

    // 1. `estimate_linear`: a line through the strip background (the
    //    background theory's own config smooths nothing first).
    let (offset0, slope0) = line_through(x, &strip(y, STRIP_WIDTH, STRIP_ITERATIONS, 1.0));

    // 2. `estimate_agauss` → `estimate_height_position_fwhm`.
    let bg = strip(
        &savitzky_golay(y, SMOOTHING_WIDTH),
        STRIP_WIDTH,
        STRIP_ITERATIONS,
        1.0,
    );
    let search_fwhm = guess_fwhm(y);
    let mut found = peak_search(y, search_fwhm, SENSITIVITY, guess_yscaling(y));
    if found.is_empty() {
        // ForcePeakPresence: the first global maximum of y − bg.
        let i = (0..n).fold(0, |best, i| {
            if y[i] - bg[i] > y[best] - bg[best] {
                i
            } else {
                best
            }
        });
        found.push((i, 0.0));
    }
    if found.len() > max_peaks {
        found.sort_by(|a, b| b.1.total_cmp(&a.1));
        found.truncate(max_peaks);
        found.sort_by_key(|p| p.0);
    }
    // Seeds: height above the background, position, FWHM = 5 sampling steps;
    // the position may move ±FWHM/2 (in x, from the search FWHM in points).
    let fwhm_seed = 5.0 * (x[n - 1] - x[0]).abs() / n as f64;
    let (xmin, xmax) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let mut p = Vec::with_capacity(3 * found.len());
    let mut cons = Vec::with_capacity(3 * found.len());
    for &(i, _) in &found {
        // silx zeroes a sub-1e-16 position.
        let pos = if x[i].abs() < 1e-16 { 0.0 } else { x[i] };
        p.extend([y[i] - bg[i], pos, fwhm_seed]);
        let quoted = match x.get(search_fwhm) {
            Some(xf) => {
                let half = 0.5 * (xf - x[0]).abs();
                Constraint::Quoted(pos - half, pos + half)
            }
            None => Constraint::Quoted(xmin, xmax),
        };
        cons.extend([Constraint::Positive, quoted, Constraint::Positive]);
    }
    let y_minus_bg: Vec<f64> = y.iter().zip(&bg).map(|(a, b)| a - b).collect();
    let pre = leastsq(sum_gauss, x, &y_minus_bg, &p, &cons, ESTIMATE_ITERATIONS)?;

    // 3. `runfit`: line + area Gaussians on the data.
    let mut p = vec![offset0, slope0];
    let mut cons = vec![Constraint::Free, Constraint::Free];
    for g in pre.params.as_chunks::<3>().0 {
        let (height, pos, fwhm) = (g[0], g[1], g[2]);
        p.extend([
            (2.0 * std::f64::consts::PI).sqrt() * height * fwhm / FWHM_PER_SIGMA,
            pos,
            fwhm,
        ]);
        cons.extend([Constraint::Positive, Constraint::Free, Constraint::Positive]);
    }
    if p.len() >= n {
        return Err(format!(
            "{} parameters for {n} points — too few points for {} peak(s)",
            p.len(),
            (p.len() - 2) / 3
        ));
    }
    let fit = leastsq(line_plus_agauss, x, y, &p, &cons, FIT_ITERATIONS)?;
    let chi2_reduced = fit.chisq / (n - fit.n_free) as f64;
    let scale = chi2_reduced.sqrt();
    let p = fit.params;
    let mut peaks: Vec<Peak> = (0..(p.len() - 2) / 3)
        .map(|j| {
            let k = 2 + 3 * j;
            Peak {
                area: p[k],
                center: p[k + 1],
                fwhm: p[k + 2],
                area_err: fit.sigma[k] * scale,
                center_err: fit.sigma[k + 1] * scale,
                fwhm_err: fit.sigma[k + 2] * scale,
            }
        })
        .collect();
    peaks.sort_by(|a, b| a.center.total_cmp(&b.center));
    Ok(PeakFit {
        peaks,
        offset: p[0],
        slope: p[1],
        model: line_plus_agauss(x, &p),
        chi2_reduced,
    })
}

// ---------------------------------------------------------------------------
// Model functions (`functions/src/funs.c`).

/// `sum_gauss`: Σ height · exp(−½((x − centroid)/σ)²), σ from the FWHM.
fn sum_gauss(x: &[f64], p: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let mut v = 0.0;
            for g in p.as_chunks::<3>().0 {
                let sigma = g[2] / FWHM_PER_SIGMA;
                let d = (xi - g[1]) / sigma;
                if d <= 20.0 {
                    v += g[0] * (-0.5 * d * d).exp();
                }
            }
            v
        })
        .collect()
}

/// `sum_agauss`: like [`sum_gauss`] with the area in place of the height.
fn sum_agauss(x: &[f64], p: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let mut v = 0.0;
            for g in p.as_chunks::<3>().0 {
                let sigma = g[2] / FWHM_PER_SIGMA;
                let height = g[0] / (sigma * (2.0 * std::f64::consts::PI).sqrt());
                let d = (xi - g[1]) / sigma;
                if d <= 35.0 {
                    v += height * (-0.5 * d * d).exp();
                }
            }
            v
        })
        .collect()
}

/// `FitManager.fitfunction` for the Linear background: `p[0] + p[1]·x` plus
/// the area Gaussians in `p[2..]`.
fn line_plus_agauss(x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut v = sum_agauss(x, &p[2..]);
    for (vi, &xi) in v.iter_mut().zip(x) {
        *vi += p[0] + p[1] * xi;
    }
    v
}

// ---------------------------------------------------------------------------
// Estimation (`bgtheories`, `fittheories`, `filters`, `peaks`).

/// `strip` of `filters/src/strip.c` (no anchors): each iteration replaces
/// every point above `factor` × the mean of its neighbours `w` away by that
/// mean, until `niter` passes are done. Stops early once a pass changes
/// nothing, which every later pass would repeat.
fn strip(y: &[f64], w: usize, niter: usize, factor: f64) -> Vec<f64> {
    let n = y.len();
    let mut output = y.to_vec();
    if n < 2 * w + 1 {
        return output;
    }
    let mut input = y.to_vec();
    for _ in 0..niter {
        let mut changed = false;
        for i in w..n - w {
            let mean = 0.5 * (input[i - w] + input[i + w]);
            if input[i] > mean * factor {
                output[i] = mean;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        input.copy_from_slice(&output);
    }
    output
}

/// `SavitskyGolay` of `filters/src/smoothnd.c`: the quadratic Savitzky–Golay
/// smoother over `npoints` (made odd), applied where the full window fits;
/// as there, a window whose sum is not positive leaves the point as it was.
/// The ends are left alone: the C code's end smoothing calls `smooth1d` on
/// `npoints / 2` points, which does nothing below three.
fn savitzky_golay(y: &[f64], npoints: usize) -> Vec<f64> {
    let n = y.len();
    let mut out = y.to_vec();
    let npoints = npoints | 1;
    if !(3..=101).contains(&npoints) || n < npoints {
        return out;
    }
    let m = npoints / 2;
    let den = ((2 * m - 1) * (2 * m + 1) * (2 * m + 3)) as f64;
    let coeff: Vec<f64> = (0..npoints)
        .map(|j| {
            let i = (j as i64 - m as i64).unsigned_abs() as usize;
            (3 * (3 * m * m + 3 * m - 1)) as f64 - (15 * i * i) as f64
        })
        .collect();
    for i in m..n - m {
        let s: f64 = (0..npoints).map(|j| coeff[j] * y[i + j - m]).sum();
        if s > 0.0 {
            out[i] = s / den;
        }
    }
    out
}

/// `peaks.pyx` `guess_fwhm`: the half-maximum width, in points, of the
/// largest peak above a one-point strip background, at least four.
fn guess_fwhm(y: &[f64]) -> usize {
    let bg = strip(y, 1, 1000, 1.0);
    let yfit: Vec<f64> = y.iter().zip(&bg).map(|(a, b)| a - b).collect();
    let maximum = yfit.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // The last index holding the maximum.
    let pos = yfit.iter().rposition(|&v| v == maximum).unwrap_or(0);
    let half = 0.5 * yfit[pos];
    let mut imin = pos;
    while yfit[imin] > half && imin > 0 {
        imin -= 1;
    }
    let mut imax = pos;
    while yfit[imax] > half && imax + 1 < yfit.len() {
        imax += 1;
    }
    (imax as i64 - imin as i64 - 1).max(FWHM_GUESS_MIN as i64) as usize
}

/// `FitTheories.guess_yscaling`: the inverse of the mean squared deviation
/// from a three-point moving average, per unit |y| — a Poisson-style noise
/// scale that turns a curve of arbitrary units into "counts" for the search.
fn guess_yscaling(y: &[f64]) -> f64 {
    let n = y.len();
    let (mut sum, mut count) = (0.0, 0usize);
    for i in 0..n {
        if y[i].abs() <= 0.0 {
            continue;
        }
        // numpy.convolve(..., "same") with a 3-point box: zero outside.
        let mut s = y[i];
        if i > 0 {
            s += y[i - 1];
        }
        if i + 1 < n {
            s += y[i + 1];
        }
        let d = y[i] - s / 3.0;
        sum += d * d / y[i].abs();
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    let chisq = sum / count as f64;
    if chisq > 0.0 { 1.0 / chisq } else { 1.0 }
}

/// `FitTheories.peak_search`: pad `y` with `fwhm` copies of each end value,
/// scale, run [`seek`] and map the hits back; `(index, relevance)` pairs.
fn peak_search(y: &[f64], fwhm: usize, sensitivity: f64, scaling: f64) -> Vec<(usize, f64)> {
    let n = y.len();
    let mut ysearch = Vec::with_capacity(n + 2 * fwhm);
    ysearch.extend(std::iter::repeat_n(y[0] * scaling, fwhm));
    ysearch.extend(y.iter().map(|v| v * scaling));
    ysearch.extend(std::iter::repeat_n(y[n - 1] * scaling, fwhm));
    if ysearch.len() as f64 <= 1.5 * fwhm as f64 {
        return Vec::new();
    }
    seek(&ysearch, fwhm as f64, sensitivity)
        .into_iter()
        .filter(|&(i, _)| i >= fwhm && i - fwhm < n)
        .map(|(i, r)| (i - fwhm, r))
        .collect()
}

/// The Mariscotti peak search — `seek` of `peaks/src/peaks.c` over the whole
/// array. Each point is scored by its Gaussian-second-derivative smoothed
/// value over the Poisson estimate of that value's standard deviation; a
/// peak is a rising-then-falling run of scores above `sensitivity`, a
/// doublet a second rise more than 0.6 FWHM past the last top. Returns
/// `(index, relevance)` per peak, in index order.
fn seek(data: &[f64], fwhm: f64, sensitivity: f64) -> Vec<(usize, f64)> {
    let n = data.len();
    let mut data = data.to_vec();
    let sigma = fwhm / 2.35482;
    let sigma2 = sigma * sigma;
    let sigma4 = sigma2 * sigma2;
    let lowthreshold = 0.01 / sigma2;

    // Second-derivative-of-Gaussian factors, out to where they vanish.
    let max_cfac = ((n as i64 - 3) / 2 - 1).clamp(0, 100) as usize;
    let mut gfactor = Vec::with_capacity(max_cfac);
    for cfac in 0..max_cfac {
        let cfac2 = ((cfac + 1) * (cfac + 1)) as f64;
        let g = (sigma2 - cfac2) * (-cfac2 / (sigma2 * 2.0)).exp() / sigma4;
        gfactor.push(g);
        if g < lowthreshold && g > -lowthreshold {
            break;
        }
    }
    let nr_factor = gfactor.len();
    // (The C code's MCA lower-level discriminator, `lld`/`channel1`, is
    // computed there and never used.)

    // The smoothed, noise-normalised score at `cch`; `first` is the C code's
    // initial evaluation, which pairs factor `cfac` with offset `cfac` where
    // the loop proper pairs `gfactor[cfac - 1]` with offset `cfac`.
    let score = |data: &[f64], cch: usize, first: bool| -> f64 {
        let mut nom = data[cch] / sigma2;
        let mut den2 = data[cch] / sigma4;
        let (start, shift) = if first { (0, 0) } else { (1, 1) };
        for cfac in start..nr_factor {
            let s = data[(cch + cfac).min(n - 1)] + data[cch.saturating_sub(cfac)];
            let g = gfactor[cfac - shift];
            nom += g * s;
            den2 += g * g * s;
        }
        if den2 <= 0.0 { 0.0 } else { nom / den2.sqrt() }
    };

    let mut peaks: Vec<(usize, f64)> = Vec::new();
    let mut cch = 0usize;
    let mut prev = score(&data, cch, true);
    data[0] = data[1];
    let mut peakstarted = 0u8;
    while cch + 2 <= n {
        cch += 1;
        let cur = score(&data, cch, false);
        if cur > sensitivity {
            if peakstarted == 0 && cur > prev {
                // (The second test keeps a peak from outside the region from
                // being detected at the start of the search.)
                peakstarted = 1;
            }
            if peakstarted == 1 && cur < prev {
                // Just past the top.
                peaks.push((cch - 1, prev));
                peakstarted = 2;
            }
            if peakstarted == 2
                && let Some(&(last, _)) = peaks.last()
                && (cch - last) as f64 > 0.6 * fwhm
                && cur > prev
            {
                // Doublet: rising again past the last top.
                peakstarted = 1;
            }
        } else {
            peakstarted = 0;
        }
        prev = cur;
    }
    peaks
}

/// Ordinary least-squares line through `(x, y)`: `(constant, slope)`, both
/// zero when `x` is degenerate (`estimate_linear`'s regression).
fn line_through(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let deno = n * sxx - sx * sx;
    if deno != 0.0 {
        ((sxx * sy - sx * sxy) / deno, (n * sxy - sx * sy) / deno)
    } else {
        (0.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// Gefit's Levenberg–Marquardt (`leastsq.py`), unweighted, numerical
// central-difference derivatives, the three constraint kinds the pipeline
// uses.

/// How `leastsq` may move a parameter.
#[derive(Clone, Copy, Debug)]
enum Constraint {
    Free,
    /// Fitted through its absolute value (`CPOSITIVE`, the "abs method").
    Positive,
    /// Kept inside `[lo, hi]` through a sine transform (`CQUOTED`); a start
    /// value outside the range, or an empty range, pins the parameter.
    Quoted(f64, f64),
}

/// A `leastsq` result: the parameters, their 1-σ uncertainties for
/// unit-variance data (NaN where pinned or quoted — the estimate discards
/// those, as `estimate_height_position_fwhm` does), χ², free count.
struct LsqFit {
    params: Vec<f64>,
    sigma: Vec<f64>,
    chisq: f64,
    n_free: usize,
}

/// The normal equations at one parameter set (`chisq_alpha_beta`).
struct Normal {
    chisq: f64,
    alpha: Vec<Vec<f64>>,
    beta: Vec<f64>,
    free_index: Vec<usize>,
    fitparam: Vec<f64>,
}

/// `_get_parameters`: the values the model sees.
fn constrained(p: &[f64], cons: &[Constraint]) -> Vec<f64> {
    p.iter()
        .zip(cons)
        .map(|(&v, c)| {
            if matches!(c, Constraint::Positive) {
                v.abs()
            } else {
                v
            }
        })
        .collect()
}

/// `chisq_alpha_beta` for unit weights: χ², `α = JᵀJ`, `β = Jᵀ(y − f)` over
/// the free parameters, with the Jacobian by central differences at a
/// relative step of √ε (`(p + [p == 0])·√ε`) and the quoted parameters'
/// chain-rule factor `B·cos(asin((p − A)/B))`.
fn normal_equations<M>(
    model: M,
    p: &[f64],
    x: &[f64],
    y: &[f64],
    cons: &[Constraint],
    last_eval: Option<&[f64]>,
) -> Result<Normal, String>
where
    M: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    let sqrt_eps = f64::EPSILON.sqrt();
    let mut free_index = Vec::new();
    let mut fitparam = Vec::new();
    let mut derivfactor = Vec::new();
    for (i, c) in cons.iter().enumerate() {
        let (v, f) = match *c {
            Constraint::Free => (p[i], 1.0),
            Constraint::Positive => (p[i].abs(), 1.0),
            Constraint::Quoted(lo, hi) => {
                let (pmin, pmax) = (lo.min(hi), lo.max(hi));
                if !(pmax > pmin && p[i] >= pmin && p[i] <= pmax) {
                    continue;
                }
                let a = 0.5 * (pmax + pmin);
                let b = 0.5 * (pmax - pmin);
                (p[i], b * ((p[i] - a) / b).asin().cos())
            }
        };
        free_index.push(i);
        fitparam.push(v);
        derivfactor.push(f);
    }
    if free_index.is_empty() {
        return Err("no free parameters to fit".into());
    }
    let mut pwork = p.to_vec();
    for (k, &i) in free_index.iter().enumerate() {
        pwork[i] = fitparam[k];
    }
    let mut deriv: Vec<Vec<f64>> = Vec::with_capacity(free_index.len());
    for (k, &i) in free_index.iter().enumerate() {
        let delta = (fitparam[k] + if fitparam[k] == 0.0 { 1.0 } else { 0.0 }) * sqrt_eps;
        pwork[i] = fitparam[k] + delta;
        let f1 = model(x, &constrained(&pwork, cons));
        pwork[i] = fitparam[k] - delta;
        let f2 = model(x, &constrained(&pwork, cons));
        pwork[i] = fitparam[k];
        deriv.push(
            f1.iter()
                .zip(&f2)
                .map(|(a, b)| (a - b) / (2.0 * delta) * derivfactor[k])
                .collect(),
        );
    }
    let yfit = match last_eval {
        Some(v) => v.to_vec(),
        None => model(x, &constrained(&pwork, cons)),
    };
    let deltay: Vec<f64> = y.iter().zip(&yfit).map(|(a, b)| a - b).collect();
    let beta = deriv
        .iter()
        .map(|d| d.iter().zip(&deltay).map(|(a, b)| a * b).sum())
        .collect();
    let alpha = deriv
        .iter()
        .map(|di| {
            deriv
                .iter()
                .map(|dj| di.iter().zip(dj).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect();
    Ok(Normal {
        chisq: deltay.iter().map(|d| d * d).sum(),
        alpha,
        beta,
        free_index,
        fitparam,
    })
}

/// Gauss–Jordan inverse with partial pivoting; `None` on an exactly singular
/// (or non-finite) pivot, where numpy's `inv` raises `LinAlgError`.
fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for c in 0..n {
        let piv = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[piv][c] == 0.0 || !m[piv][c].is_finite() {
            return None;
        }
        m.swap(c, piv);
        let d = m[c][c];
        for v in m[c].iter_mut() {
            *v /= d;
        }
        let pivot_row = m[c].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == c || row[c] == 0.0 {
                continue;
            }
            let f = row[c];
            for (v, pv) in row.iter_mut().zip(&pivot_row) {
                *v -= f * pv;
            }
        }
    }
    Some(m.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// silx `leastsq` (Gefit's Marquardt loop) for an unweighted fit: λ starts
/// at 0.001, ×10 on a χ² increase (giving up past 1000), ÷10 on a decrease;
/// from the second iteration a decrease under [`DELTACHI_PERCENT`] % or
/// under √ε ends the fit. `max_iter` counts every trial step, as there.
fn leastsq<M>(
    model: M,
    x: &[f64],
    y: &[f64],
    p0: &[f64],
    cons: &[Constraint],
    max_iter: usize,
) -> Result<LsqFit, String>
where
    M: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    let sqrt_eps = f64::EPSILON.sqrt();
    let mut fitted = p0.to_vec();
    let mut flambda = 0.001;
    let mut iiter = max_iter as i64;
    let mut iteration = 0usize;
    let mut last_eval: Option<Vec<f64>> = None;
    let mut normal: Option<Normal> = None;
    while iiter > 0 {
        iteration += 1;
        let nm = normal_equations(&model, &fitted, x, y, cons, last_eval.as_deref())?;
        let n_free = nm.free_index.len();
        let mut chisq0 = nm.chisq;
        loop {
            let mut alpha = nm.alpha.clone();
            for (i, row) in alpha.iter_mut().enumerate() {
                row[i] *= 1.0 + flambda;
            }
            let inv = invert(&alpha).ok_or("singular normal matrix")?;
            let delta: Vec<f64> = (0..n_free)
                .map(|j| (0..n_free).map(|i| nm.beta[i] * inv[i][j]).sum())
                .collect();
            let mut newpar = p0.to_vec();
            for (k, &i) in nm.free_index.iter().enumerate() {
                newpar[i] = match cons[i] {
                    Constraint::Quoted(lo, hi) => {
                        let a = 0.5 * (hi.max(lo) + hi.min(lo));
                        let b = 0.5 * (hi.max(lo) - hi.min(lo));
                        a + b * (((nm.fitparam[k] - a) / b).asin() + delta[k]).sin()
                    }
                    _ => nm.fitparam[k] + delta[k],
                };
            }
            let newpar = constrained(&newpar, cons);
            let yfit = model(x, &newpar);
            let chisq: f64 = y.iter().zip(&yfit).map(|(a, b)| (a - b) * (a - b)).sum();
            let absdeltachi = chisq0 - chisq;
            let mut done = true;
            if absdeltachi < 0.0 {
                flambda *= 10.0;
                if flambda > 1000.0 {
                    iiter = 0;
                } else {
                    done = false;
                }
            } else {
                fitted = newpar;
                let lastdeltachi =
                    100.0 * absdeltachi / (chisq + if chisq == 0.0 { 1.0 } else { 0.0 });
                if iteration >= 2 && (lastdeltachi < DELTACHI_PERCENT || absdeltachi < sqrt_eps) {
                    iiter = 0;
                }
                chisq0 = chisq;
                flambda /= 10.0;
                last_eval = Some(yfit);
            }
            iiter -= 1;
            if done {
                break;
            }
        }
        normal = Some(Normal {
            chisq: chisq0,
            ..nm
        });
    }
    let nm = normal.expect("max_iter is at least one");
    // The covariance of the free parameters at the last linearisation.
    let cov = invert(&nm.alpha).ok_or("singular normal matrix")?;
    let mut sigma = vec![f64::NAN; p0.len()];
    for (k, &i) in nm.free_index.iter().enumerate() {
        if !matches!(cons[i], Constraint::Quoted(..)) {
            sigma[i] = cov[k][k].abs().sqrt();
        }
    }
    Ok(LsqFit {
        params: fitted,
        sigma,
        chisq: nm.chisq,
        n_free: nm.free_index.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(x: f64, area: f64, c: f64, fwhm: f64) -> f64 {
        let s = fwhm / FWHM_PER_SIGMA;
        area / (s * (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * ((x - c) / s).powi(2)).exp()
    }

    /// Two separated Gaussians on a sloped line: the fit recovers every
    /// parameter, in ascending-center order.
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
        assert!(fit.chi2_reduced < 1e-8, "{}", fit.chi2_reduced);
        assert_eq!(fit.model.len(), x.len());
    }

    /// Specfit's ForcePeakPresence: a curve with no peak above the noise still
    /// fits exactly one Gaussian; the cap on the peak count holds when more
    /// peaks are found.
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
        // Two of three equal peaks: the fit is under-modelled, so only the
        // count and the rough positions are asserted.
        let fit = fit_area_gaussians(&x, &three, 2).expect("fit");
        assert_eq!(fit.peaks.len(), 2);
        for p in &fit.peaks {
            assert!(
                [10.0, 25.0, 40.0]
                    .iter()
                    .any(|c| (p.center - c).abs() < 0.1),
                "{p:?}"
            );
        }
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

    /// The Mariscotti search on a count spectrum (no scaling): two peaks of
    /// eight-channel FWHM on a flat background are found at their channels,
    /// the stronger with the higher relevance; the flat part yields nothing.
    #[test]
    fn seek_finds_the_peaks_of_a_count_spectrum() {
        let y: Vec<f64> = (0..200)
            .map(|i| {
                let x = i as f64;
                50.0 + 2000.0 * gauss(x, 1.0, 60.0, 8.0) * 8.0
                    + 1200.0 * gauss(x, 1.0, 130.0, 8.0) * 8.0
            })
            .collect();
        let hits = seek(&y, 8.0, 2.5);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].0 as i64 - 60).abs() <= 1, "{hits:?}");
        assert!((hits[1].0 as i64 - 130).abs() <= 1, "{hits:?}");
        assert!(hits[0].1 > hits[1].1, "{hits:?}");
        assert!(seek(&vec![50.0; 200], 8.0, 2.5).is_empty());
    }

    /// strip: points on a line are never touched; a peak on that line is
    /// flattened back onto it, never below the data.
    #[test]
    fn strip_flattens_a_peak_onto_its_baseline() {
        let line: Vec<f64> = (0..100).map(|i| 10.0 + 0.1 * i as f64).collect();
        let y: Vec<f64> = line
            .iter()
            .enumerate()
            .map(|(i, b)| b + 20.0 * gauss(i as f64, 1.0, 50.0, 6.0) * 6.0)
            .collect();
        let bg = strip(&y, STRIP_WIDTH, STRIP_ITERATIONS, 1.0);
        for i in 0..100 {
            assert!(
                (bg[i] - line[i]).abs() < 1e-3,
                "{i}: {} vs {}",
                bg[i],
                line[i]
            );
            assert!(bg[i] <= y[i] + 1e-12);
        }
        let same = strip(&line, STRIP_WIDTH, STRIP_ITERATIONS, 1.0);
        assert!(same.iter().zip(&line).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    /// Savitzky–Golay(5) reproduces a quadratic exactly where the window
    /// fits and leaves the two points at each end as they were.
    #[test]
    fn savitzky_golay_reproduces_a_quadratic() {
        let y: Vec<f64> = (0..30)
            .map(|i| 1.0 + 0.5 * i as f64 + 0.02 * (i * i) as f64)
            .collect();
        let s = savitzky_golay(&y, 5);
        for i in 2..28 {
            assert!((s[i] - y[i]).abs() < 1e-9, "{i}: {} vs {}", s[i], y[i]);
        }
        assert_eq!(&s[..2], &y[..2]);
        assert_eq!(&s[28..], &y[28..]);
        assert_eq!(savitzky_golay(&y[..4], 5), &y[..4]);
    }

    /// guess_fwhm measures the largest peak's half-maximum width in points,
    /// never below four.
    #[test]
    fn guess_fwhm_measures_the_largest_peak() {
        let y: Vec<f64> = (0..200)
            .map(|i| gauss(i as f64, 1.0, 80.0, 12.0) + 3.0 * gauss(i as f64, 1.0, 150.0, 20.0))
            .collect();
        let w = guess_fwhm(&y);
        assert!((19..=21).contains(&w), "{w}");
        assert_eq!(guess_fwhm(&[1.0; 20]), FWHM_GUESS_MIN);
    }

    /// leastsq's constraints: a positive parameter whose free optimum is
    /// negative stays non-negative; a quoted one stays inside its range; a
    /// free fit of a line converges (to Gefit's χ²-decrement stop, not to
    /// machine precision).
    #[test]
    fn leastsq_honours_positive_and_quoted_constraints() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let line = |x: &[f64], p: &[f64]| x.iter().map(|v| p[0] + p[1] * v).collect::<Vec<_>>();
        let y: Vec<f64> = x.iter().map(|v| 2.0 - 1.0 * v).collect();
        let free = leastsq(line, &x, &y, &[0.0, 0.0], &[Constraint::Free; 2], 100).expect("fit");
        assert!(
            (free.params[0] - 2.0).abs() < 1e-4 && (free.params[1] + 1.0).abs() < 1e-4,
            "{:?}",
            free.params
        );
        assert_eq!(free.n_free, 2);
        let pos = leastsq(
            line,
            &x,
            &y,
            &[0.0, 0.5],
            &[Constraint::Free, Constraint::Positive],
            100,
        )
        .expect("fit");
        assert!(
            pos.params[1] >= 0.0 && pos.params[1] < 0.5,
            "{:?}",
            pos.params
        );
        assert!(pos.chisq > free.chisq);

        let constant = |x: &[f64], p: &[f64]| vec![p[0]; x.len()];
        let five = vec![5.0; 20];
        let quoted = leastsq(
            constant,
            &x,
            &five,
            &[1.0],
            &[Constraint::Quoted(0.0, 2.0)],
            100,
        )
        .expect("fit");
        assert!(
            quoted.params[0] > 1.9 && quoted.params[0] <= 2.0,
            "{:?}",
            quoted.params
        );
        assert!(quoted.sigma[0].is_nan());
    }
}
