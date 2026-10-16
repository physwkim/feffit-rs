//! Gaussian peak fit of a 1-D curve — the BL10C ScanWindow "Simple Fit"
//! (`ScanFit` → PyMca `Specfit`: theory *Area Gaussians*, background
//! *Linear*, `WeightFlag` 0, `ForcePeakPresence` 1, `McaMode` 0), ported from
//! PyMca5 `PyMcaMath/fitting/{Specfit.py, SpecfitFunctions.py, Gefit.py,
//! specfit/SpecfitFuns.c}`. Used by the In-situ window on the extracted
//! profile.
//!
//! The pipeline, as `Specfit.estimate` + `Specfit.startfit` run it:
//!
//! 1. background seed (`estimate_builtin_bkg`): a line through
//!    `subac(y, 1.0001, 1000)`;
//! 2. peak estimate (`SpecfitFunctions.estimate_gauss`): the Mariscotti
//!    search (`SpecfitFuns.seek`) over `y × Yscaling`, unpadded, with
//!    `FwhmPoints` 8 and `Sensitivity` 2.5 — or the maximum of
//!    `y − subac(y, 1, 10000)` when it finds nothing (`ForcePeakPresence`);
//!    four Gefit iterations of height/position/FWHM Gaussians on that
//!    background-subtracted curve (height ≥ 0, position within ±½ the
//!    search FWHM in x, FWHM ≥ 0); heights converted to areas
//!    (`estimate_agauss`);
//! 3. the fit (`Gefit.LeastSquaresFit`): line + area Gaussians on the raw
//!    data, unweighted, under the Specfit default restraints — area ≥ 0
//!    (`HeightAreaFlag`), position free (`PositionFlag` 0), FWHM ≥ 0
//!    (`PosFwhmFlag`) and, for every peak but the one estimated highest,
//!    FWHM = 1.0 × that peak's FWHM (`SameFwhmFlag`, a Gefit FACTOR
//!    constraint). The shared FWHM is what keeps a Gaussian from spreading
//!    over whatever baseline the line cannot follow.
//!
//! Deviations from PyMca's defaults, each deliberate:
//!
//! - `AutoScaling` is on (the ScanFit panel's "Auto Scaling" box, off by
//!   default): the search's significance test assumes Poisson variance,
//!   and unscaled it finds nothing on profiles of a few units — PyMca then
//!   fits one forced Gaussian. The scaling is PyMca's own
//!   `Specfit.guess_yscaling`.
//! - the peak count is capped (`max_peaks`, highest Mariscotti relevance
//!   kept); PyMca has no cap.
//! - uncertainties are scaled by √(reduced χ²); Gefit reports the
//!   unit-variance ones when the fit is unweighted.

/// `FWHM / σ` of a Gaussian, `2·√(2·ln 2)`.
const FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949;
/// `SpecfitFunctions` defaults: the search FWHM in points and the Mariscotti
/// sensitivity (in noise σ).
const FWHM_POINTS: usize = 8;
const SENSITIVITY: f64 = 2.5;
/// The two `subac` calls: the Linear background's seed and the estimate's
/// own background.
const BKG_STRIP_FACTOR: f64 = 1.0001;
const BKG_STRIP_ITERATIONS: usize = 1000;
const ESTIMATE_STRIP_ITERATIONS: usize = 10_000;
/// Gefit: the quick estimate's and the fit's iteration budgets, the relative
/// χ² decrease that ends a fit (`deltachi`), the relative derivative step.
const ESTIMATE_ITERATIONS: usize = 4;
const FIT_ITERATIONS: usize = 100;
const DELTACHI: f64 = 0.01;
const DERIVATIVE_STEP: f64 = 0.000_01;
/// `SpecfitFuns.seek` gives up past this many peaks.
const SEEK_MAX_PEAKS: usize = 150;

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
    /// Peaks in ascending `center` order. Every FWHM but the highest
    /// estimated peak's is tied to that peak's (and carries its
    /// uncertainty), as Specfit's `SameFwhmFlag` has it.
    pub peaks: Vec<Peak>,
    /// Linear background `offset + slope · x`.
    pub offset: f64,
    pub slope: f64,
    /// The fitted curve on the input `x`.
    pub model: Vec<f64>,
    /// Gefit's reduced χ²: Σ(y − model)² over (points − free parameters) at
    /// the last trial step; the uncertainties are scaled by its square root.
    pub chi2_reduced: f64,
}

/// Fit `y(x)` with up to `max_peaks` Gaussians (area form) on a linear
/// background. `x` need not be ascending or evenly spaced (`setdata` sorts
/// it). Errors: fewer than five finite points, more parameters than points,
/// or a singular normal matrix (numpy's `LinAlgError`).
pub fn fit_area_gaussians(x: &[f64], y: &[f64], max_peaks: usize) -> Result<PeakFit, String> {
    let n = x.len().min(y.len());
    if n < 5 {
        return Err("need at least 5 points".into());
    }
    let (x_in, y_in) = (&x[..n], &y[..n]);
    if x_in.iter().chain(y_in).any(|v| !v.is_finite()) {
        return Err("non-finite values in the curve".into());
    }
    let max_peaks = max_peaks.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x_in[a].total_cmp(&x_in[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x_in[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y_in[i]).collect();
    let (x, y) = (xs.as_slice(), ys.as_slice());

    // 1. `estimate_builtin_bkg`: a line through the stripped curve.
    let (offset0, slope0) = line_through(x, &subac(y, BKG_STRIP_FACTOR, BKG_STRIP_ITERATIONS));
    let yscaling = guess_yscaling(y);

    // 2. `estimate_gauss`.
    let zz = subac(y, 1.0, ESTIMATE_STRIP_ITERATIONS);
    let mut found: Vec<(usize, f64)> = if n as f64 > 1.5 * FWHM_POINTS as f64 {
        let ysearch: Vec<f64> = y.iter().map(|v| v * yscaling).collect();
        seek(&ysearch, FWHM_POINTS as f64, SENSITIVITY)?
            .into_iter()
            .filter(|&(i, _)| (1..n).contains(&i))
            .collect()
    } else {
        Vec::new()
    };
    if found.is_empty() {
        // ForcePeakPresence: the first maximum of y − zz.
        let i = (0..n).fold(0, |best, i| {
            if y[i] - zz[i] > y[best] - zz[best] {
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
    // the position may move ±½ the search FWHM (in x); the highest seed is
    // the FWHM master.
    let sig = 5.0 * (x[n - 1] - x[0]).abs() / n as f64;
    let fwhmx = x.get(FWHM_POINTS).map(|xf| (xf - x[0]).abs());
    let (xmin, xmax) = (x[0], x[n - 1]);
    let mut p = Vec::with_capacity(3 * found.len());
    let mut cons = Vec::with_capacity(3 * found.len());
    let mut largest_index = 0;
    for (j, &(i, _)) in found.iter().enumerate() {
        let height = y[i] - zz[i];
        // PyMca zeroes a sub-1e-16 position — of the first peak only.
        let pos = if j == 0 && x[i].abs() < 1e-16 {
            0.0
        } else {
            x[i]
        };
        if height
            > p.first()
                .map_or(f64::NEG_INFINITY, |_| p[3 * largest_index])
        {
            largest_index = j;
        }
        p.extend([height, pos, sig]);
        let quoted = match fwhmx {
            Some(f) => Constraint::Quoted(pos - 0.5 * f, pos + 0.5 * f),
            None => Constraint::Quoted(xmin, xmax),
        };
        cons.extend([Constraint::Positive, quoted, Constraint::Positive]);
    }
    let y_minus_zz: Vec<f64> = y.iter().zip(&zz).map(|(a, b)| a - b).collect();
    let pre = leastsq(sum_gauss, x, &y_minus_zz, &p, &cons, ESTIMATE_ITERATIONS)?;

    // 3. `startfit`: line + area Gaussians on the data, under the rebuilt
    //    restraints (`estimate_gauss`'s second block, indices shifted past
    //    the two background parameters by `Specfit.estimate`).
    let master = 2 + 3 * largest_index + 2;
    let mut p = vec![offset0, slope0];
    let mut cons = vec![Constraint::Free, Constraint::Free];
    for (j, g) in pre.params.as_chunks::<3>().0.iter().enumerate() {
        let (height, pos, fwhm) = (g[0], g[1], g[2]);
        p.extend([
            height * fwhm / FWHM_PER_SIGMA * (2.0 * std::f64::consts::PI).sqrt(),
            pos,
            fwhm,
        ]);
        let fwhm_cons = if j == largest_index {
            Constraint::Positive
        } else {
            Constraint::Factor(master, 1.0)
        };
        cons.extend([Constraint::Positive, Constraint::Free, fwhm_cons]);
    }
    if p.len() >= n {
        return Err(format!(
            "{} parameters for {n} points — too few points for {} peak(s)",
            p.len(),
            (p.len() - 2) / 3
        ));
    }
    let fit = leastsq(line_plus_agauss, x, y, &p, &cons, FIT_ITERATIONS)?;
    let chi2_reduced = fit.chisq_reduced;
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
        model: line_plus_agauss(x_in, &p),
        chi2_reduced,
    })
}

// ---------------------------------------------------------------------------
// Model functions (`SpecfitFuns.c`).

/// `SpecfitFuns.gauss`: Σ height · exp(−½((x − centroid)/σ)²), σ from the
/// FWHM; a term is skipped beyond 20 σ above its centroid, as there.
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

/// `SpecfitFuns.agauss`: like [`sum_gauss`] with the area in place of the
/// height (cutoff 35 σ).
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

/// `Specfit.fitfunction` for the Linear background: `p[0] + p[1]·x` plus the
/// area Gaussians in `p[2..]`.
fn line_plus_agauss(x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut v = sum_agauss(x, &p[2..]);
    for (vi, &xi) in v.iter_mut().zip(x) {
        *vi += p[0] + p[1] * xi;
    }
    v
}

// ---------------------------------------------------------------------------
// Estimation (`Specfit`, `SpecfitFunctions`, `SpecfitFuns.c`).

/// `SpecfitFuns.subac` (no anchors, `deltai` 1): each pass replaces every
/// point above `c` × the mean of its two neighbours by that mean, `niter`
/// times. Stops early once a pass changes nothing, which every later pass
/// would repeat.
fn subac(y: &[f64], c: f64, niter: usize) -> Vec<f64> {
    let n = y.len();
    let mut ret = y.to_vec();
    if n < 3 {
        return ret;
    }
    let mut data = y.to_vec();
    for _ in 0..niter {
        let mut changed = false;
        for j in 1..n - 1 {
            let t_mean = 0.5 * (data[j - 1] + data[j + 1]);
            if ret[j] > t_mean * c {
                ret[j] = t_mean;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        data.copy_from_slice(&ret);
    }
    ret
}

/// `Specfit.guess_yscaling`: the inverse of the mean squared deviation of
/// `y[1:-1]` from a three-point mean, per unit |y| — a Poisson-style noise
/// scale that turns a curve of arbitrary units into "counts" for the search.
/// PyMca takes the mean from numpy's full convolution, so `y[k + 1]` is
/// compared with the mean of `y[k − 2 ..= k]`; the port keeps that offset.
fn guess_yscaling(y: &[f64]) -> f64 {
    let n = y.len();
    let (mut sum, mut count) = (0.0, 0usize);
    for k in 0..n.saturating_sub(2) {
        let yy = y[k + 1];
        if yy.abs() <= 0.0 {
            continue;
        }
        let mut s = y[k];
        if k >= 1 {
            s += y[k - 1];
        }
        if k >= 2 {
            s += y[k - 2];
        }
        let d = yy - s / 3.0;
        sum += d * d / yy.abs();
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    let chisq = sum / count as f64;
    if chisq > 0.0 { 1.0 / chisq } else { 1.0 }
}

/// The Mariscotti peak search — `SpecfitFuns_seek2` over the whole array
/// (`BeginChannel` 0, `EndChannel` n). Each point is scored by its
/// Gaussian-second-derivative smoothed value over the Poisson estimate of
/// that value's standard deviation; a peak is a rising-then-falling run of
/// scores above `sensitivity`, a doublet a second rise more than 0.6 FWHM
/// past the last top. Returns `(index, relevance)` per peak, in index order;
/// an error past [`SEEK_MAX_PEAKS`], where the C code returns −2 and the
/// Python call raises.
fn seek(data: &[f64], fwhm: f64, sensitivity: f64) -> Result<Vec<(usize, f64)>, String> {
    let n = data.len();
    let mut yspec = data.to_vec();
    let sigma = fwhm / 2.35482;
    let sigma2 = sigma * sigma;
    let sigma4 = sigma2 * sigma2;
    let lowthreshold = 0.01 / sigma2;

    // Second-derivative-of-Gaussian factors up to the first below threshold
    // (which is computed but, like the C code, never used).
    let j = ((n as i64 - 2) / 2 - 1).min(100);
    let mut gfactor = Vec::new();
    let mut cfac = 1i64;
    while cfac < j + 1 {
        let cfac2 = (cfac * cfac) as f64;
        let g = (sigma2 - cfac2) * (-cfac2 / (sigma2 * 2.0)).exp() / sigma4;
        gfactor.push(g);
        if g < lowthreshold && g > -lowthreshold {
            break;
        }
        cfac += 1;
    }
    let nr_factor = cfac as usize;
    // (The C code's MCA lower-level discriminator, `lld`/`channel1`, is
    // computed there and never used.)

    let score = |yspec: &[f64], cch: usize| -> f64 {
        let mut nom = yspec[cch] / sigma2;
        let mut den2 = yspec[cch] / sigma4;
        for k in 1..nr_factor {
            let s = yspec[(cch + k).min(n - 1)] + yspec[cch.saturating_sub(k)];
            let g = gfactor[k - 1];
            nom += g * s;
            den2 += g * g * s;
        }
        if den2 <= 0.0 { 0.0 } else { nom / den2.sqrt() }
    };

    let mut peaks: Vec<(usize, f64)> = Vec::new();
    let mut cch = 0usize;
    let mut prev = score(&yspec, cch);
    yspec[0] = yspec[1];
    let mut peakstarted = 0u8;
    while cch + 2 <= n {
        cch += 1;
        let cur = score(&yspec, cch);
        if cur > sensitivity {
            if peakstarted == 0 && cur > prev {
                // (The second test keeps a peak from outside the region from
                // being detected at the start of the search.)
                peakstarted = 1;
            }
            if peakstarted == 1 && cur < prev {
                // Just past the top.
                if peaks.len() >= SEEK_MAX_PEAKS {
                    return Err("peak search found too many peaks".into());
                }
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
    Ok(peaks)
}

/// Ordinary least-squares line through `(x, y)`: `(constant, slope)`, both
/// zero when `x` is degenerate (`estimate_builtin_bkg`'s regression).
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
// Gefit's Levenberg–Marquardt (`RestreinedLeastSquaresFit`, the 2014 copy
// BL10C runs), unweighted, numerical central-difference derivatives, the
// four constraint kinds the pipeline uses.

/// How Gefit may move a parameter.
#[derive(Clone, Copy, Debug)]
enum Constraint {
    Free,
    /// Fitted through its absolute value (`CPOSITIVE`, the "abs method").
    Positive,
    /// Kept inside `[lo, hi]` through a sine transform (`CQUOTED`); a start
    /// value outside the range, or an empty range, pins the parameter.
    Quoted(f64, f64),
    /// Not fitted: `factor` × the parameter at the given index (`CFACTOR`).
    Factor(usize, f64),
}

/// A Gefit result: the parameters, their uncertainties for unit-variance
/// data (`getsigmaparameters`), the reduced χ² of the last trial step.
struct LsqFit {
    params: Vec<f64>,
    sigma: Vec<f64>,
    chisq_reduced: f64,
}

/// The normal equations at one parameter set (`ChisqAlphaBeta`).
struct Normal {
    chisq: f64,
    alpha: Vec<Vec<f64>>,
    beta: Vec<f64>,
    free_index: Vec<usize>,
    fitparam: Vec<f64>,
}

/// `getparameters`: the values the model sees — |p| for the positive ones,
/// then the factor-tied ones from their masters, in index order.
fn get_parameters(p: &[f64], cons: &[Constraint]) -> Vec<f64> {
    let mut out: Vec<f64> = p
        .iter()
        .zip(cons)
        .map(|(&v, c)| {
            if matches!(c, Constraint::Positive) {
                v.abs()
            } else {
                v
            }
        })
        .collect();
    for i in 0..out.len() {
        if let Constraint::Factor(master, factor) = cons[i] {
            out[i] = factor * out[master];
        }
    }
    out
}

/// `getsigmaparameters`: the free parameters' σ in order; a factor-tied
/// parameter gets `factor` × its master's; a quoted one inside its range
/// gets `|B·cos(p)·σ|` and otherwise its own value, as Gefit writes it.
fn get_sigma_parameters(p: &[f64], sigma0: &[f64], cons: &[Constraint]) -> Vec<f64> {
    let mut out = vec![0.0; p.len()];
    let mut k = 0;
    for i in 0..p.len() {
        match cons[i] {
            Constraint::Free | Constraint::Positive => {
                out[i] = sigma0[k];
                k += 1;
            }
            Constraint::Quoted(lo, hi) => {
                let (pmin, pmax) = (lo.min(hi), lo.max(hi));
                let b = 0.5 * (pmax - pmin);
                if b > 0.0 && p[i] < pmax && p[i] > pmin {
                    out[i] = (b * p[i].cos() * sigma0[k]).abs();
                    k += 1;
                } else {
                    out[i] = p[i];
                }
            }
            Constraint::Factor(..) => {}
        }
    }
    for i in 0..p.len() {
        if let Constraint::Factor(master, factor) = cons[i] {
            out[i] = factor * out[master];
        }
    }
    out
}

/// `ChisqAlphaBeta` for unit weights: χ², `α = JᵀJ`, `β = Jᵀ(y − f)` over the
/// free parameters, with the Jacobian by central differences at a relative
/// step of 1e-5 (`(p + [p == 0])·1e-5`) and the quoted parameters'
/// chain-rule factor `B·cos(asin((p − A)/B))`.
fn normal_equations<M>(
    model: M,
    p: &[f64],
    x: &[f64],
    y: &[f64],
    cons: &[Constraint],
) -> Result<Normal, String>
where
    M: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    let mut free_index = Vec::new();
    let mut fitparam = Vec::new();
    let mut derivfactor = Vec::new();
    for (i, c) in cons.iter().enumerate() {
        let (v, f) = match *c {
            Constraint::Free => (p[i], 1.0),
            Constraint::Positive => (p[i].abs(), 1.0),
            Constraint::Quoted(lo, hi) => {
                let (pmin, pmax) = (lo.min(hi), lo.max(hi));
                if !(pmax - pmin > 0.0 && p[i] <= pmax && p[i] >= pmin) {
                    continue;
                }
                let a = 0.5 * (pmax + pmin);
                let b = 0.5 * (pmax - pmin);
                (p[i], b * ((p[i] - a) / b).asin().cos())
            }
            Constraint::Factor(..) => continue,
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
        let delta = (fitparam[k] + if fitparam[k] == 0.0 { 1.0 } else { 0.0 }) * DERIVATIVE_STEP;
        pwork[i] = fitparam[k] + delta;
        let f1 = model(x, &get_parameters(&pwork, cons));
        pwork[i] = fitparam[k] - delta;
        let f2 = model(x, &get_parameters(&pwork, cons));
        pwork[i] = fitparam[k];
        deriv.push(
            f1.iter()
                .zip(&f2)
                .map(|(a, b)| (a - b) / (2.0 * delta) * derivfactor[k])
                .collect(),
        );
    }
    let yfit = model(x, &get_parameters(&pwork, cons));
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

/// Gefit's `RestreinedLeastSquaresFit` for an unweighted fit: λ starts at
/// 0.001, ×10 on a χ² increase (giving up past 1000), ÷10 on a decrease; a
/// decrease under [`DELTACHI`] of the previous χ² ends the fit. The first
/// iteration uses every second point when the points outnumber the
/// parameters three to one; `max_iter` counts every trial step. The
/// uncertainties come from the last linearisation, the reduced χ² from the
/// last trial step (accepted or not), as there.
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
    let n = x.len();
    let n_param = p0.len();
    let x_half: Vec<f64> = x.iter().step_by(2).copied().collect();
    let y_half: Vec<f64> = y.iter().step_by(2).copied().collect();
    let mut fittedpar = p0.to_vec();
    let mut flambda = 0.001;
    let mut iiter = max_iter as i64;
    let mut niter = 0usize;
    let mut alpha0: Vec<Vec<f64>> = Vec::new();
    // χ² and point count of the last trial evaluation.
    let mut last = (f64::NAN, 0usize);
    while iiter > 0 {
        niter += 1;
        let (xw, yw) = if niter < 2 && n_param * 3 < n {
            (x_half.as_slice(), y_half.as_slice())
        } else {
            (x, y)
        };
        let nm = normal_equations(&model, &fittedpar, xw, yw, cons)?;
        let n_free = nm.free_index.len();
        let chisq0 = nm.chisq;
        loop {
            let mut alpha = nm.alpha.clone();
            for (i, row) in alpha.iter_mut().enumerate() {
                row[i] *= 1.0 + flambda;
            }
            let inv = invert(&alpha).ok_or("singular normal matrix")?;
            let deltapar: Vec<f64> = (0..n_free)
                .map(|j| (0..n_free).map(|i| nm.beta[i] * inv[i][j]).sum())
                .collect();
            let mut newpar = p0.to_vec();
            for (k, &i) in nm.free_index.iter().enumerate() {
                newpar[i] = match cons[i] {
                    Constraint::Quoted(lo, hi) => {
                        let a = 0.5 * (hi.max(lo) + hi.min(lo));
                        let b = 0.5 * (hi.max(lo) - hi.min(lo));
                        a + b * (((nm.fitparam[k] - a) / b).asin() + deltapar[k]).sin()
                    }
                    _ => nm.fitparam[k] + deltapar[k],
                };
            }
            let newpar = get_parameters(&newpar, cons);
            let yfit = model(xw, &newpar);
            let chisq: f64 = yw.iter().zip(&yfit).map(|(a, b)| (a - b) * (a - b)).sum();
            last = (chisq, yfit.len());
            let mut done = true;
            if chisq > chisq0 {
                flambda *= 10.0;
                if flambda > 1000.0 {
                    iiter = 0;
                } else {
                    done = false;
                }
            } else {
                fittedpar = newpar;
                let lastdeltachi =
                    (chisq0 - chisq) / (chisq0 + if chisq0 == 0.0 { 1.0 } else { 0.0 });
                if lastdeltachi < DELTACHI {
                    iiter = 0;
                }
                flambda /= 10.0;
            }
            iiter -= 1;
            if done {
                break;
            }
        }
        alpha0 = nm.alpha;
    }
    let cov = invert(&alpha0).ok_or("singular normal matrix")?;
    let sigma0: Vec<f64> = (0..cov.len()).map(|k| cov[k][k].abs().sqrt()).collect();
    let sigma = get_sigma_parameters(&fittedpar, &sigma0, cons);
    Ok(LsqFit {
        params: fittedpar,
        sigma,
        chisq_reduced: last.0 / (last.1 as f64 - sigma0.len() as f64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(x: f64, area: f64, c: f64, fwhm: f64) -> f64 {
        let s = fwhm / FWHM_PER_SIGMA;
        area / (s * (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * ((x - c) / s).powi(2)).exp()
    }

    /// Two separated Gaussians of one width on a sloped line: the fit
    /// recovers every parameter, in ascending-center order.
    #[test]
    fn recovers_two_gaussians_on_a_linear_background() {
        let x: Vec<f64> = (0..400).map(|i| 6530.0 + i as f64 * 0.25).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&e| {
                0.2 + 0.001 * (e - 6530.0) + gauss(e, 3.0, 6552.0, 2.5) + gauss(e, 1.2, 6570.0, 2.5)
            })
            .collect();
        let fit = fit_area_gaussians(&x, &y, 5).expect("fit");
        assert_eq!(fit.peaks.len(), 2, "{:?}", fit.peaks);
        let (a, b) = (&fit.peaks[0], &fit.peaks[1]);
        assert!((a.center - 6552.0).abs() < 1e-3, "{a:?}");
        assert!((a.fwhm - 2.5).abs() < 1e-3, "{a:?}");
        assert!((a.area - 3.0).abs() < 1e-3, "{a:?}");
        assert!((b.center - 6570.0).abs() < 1e-3, "{b:?}");
        assert!((b.fwhm - 2.5).abs() < 1e-3, "{b:?}");
        assert!((b.area - 1.2).abs() < 1e-3, "{b:?}");
        assert!((fit.slope - 0.001).abs() < 1e-6);
        assert!((fit.offset - (0.2 - 0.001 * 6530.0)).abs() < 1e-3);
        assert!(fit.chi2_reduced < 1e-8, "{}", fit.chi2_reduced);
        assert_eq!(fit.model.len(), x.len());
    }

    /// Specfit's `SameFwhmFlag`: two Gaussians of different widths come out
    /// with one FWHM — the highest estimated peak's, which the other one is
    /// tied to, uncertainty included — and the baseline between them stays
    /// with the line rather than with a broadened Gaussian.
    #[test]
    fn ties_every_fwhm_to_the_highest_peak() {
        let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| 1.0 + 0.002 * v + gauss(v, 50.0, 60.0, 6.0) + gauss(v, 20.0, 140.0, 12.0))
            .collect();
        let fit = fit_area_gaussians(&x, &y, 5).expect("fit");
        assert_eq!(fit.peaks.len(), 2, "{:?}", fit.peaks);
        let (a, b) = (&fit.peaks[0], &fit.peaks[1]);
        assert_eq!(a.fwhm, b.fwhm, "{a:?} {b:?}");
        assert_eq!(a.fwhm_err, b.fwhm_err, "{a:?} {b:?}");
        assert!(a.fwhm > 6.0 && a.fwhm < 8.0, "{a:?}");
        assert!((a.center - 60.0).abs() < 0.1, "{a:?}");
        assert!((b.center - 140.0).abs() < 1.0, "{b:?}");
        assert!((fit.slope - 0.002).abs() < 1e-3, "{}", fit.slope);
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

        // Three equal peaks of the search FWHM (eight points) on a line,
        // which the auto-scaled search finds at relevance 8.6 each. (On an
        // exactly zero baseline the fitted slope collapses towards 1e-43,
        // its 1e-5 relative derivative step vanishes in the data's ulp and
        // the normal matrix is singular — in PyMca too.)
        let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let three: Vec<f64> = x
            .iter()
            .map(|&v| {
                1.0 + 0.002 * v
                    + gauss(v, 5.0, 40.0, 8.0)
                    + gauss(v, 5.0, 100.0, 8.0)
                    + gauss(v, 5.0, 160.0, 8.0)
            })
            .collect();
        // Two of three: the fit is under-modelled, so only the count and the
        // rough positions are asserted.
        let fit = fit_area_gaussians(&x, &three, 2).expect("fit");
        assert_eq!(fit.peaks.len(), 2);
        for p in &fit.peaks {
            assert!(
                [40.0, 100.0, 160.0]
                    .iter()
                    .any(|c| (p.center - c).abs() < 0.5),
                "{p:?}"
            );
        }
        let fit = fit_area_gaussians(&x, &three, 3).expect("fit");
        assert_eq!(fit.peaks.len(), 3);
        let centers: Vec<f64> = fit.peaks.iter().map(|p| p.center).collect();
        for (c, want) in centers.iter().zip([40.0, 100.0, 160.0]) {
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
        let hits = seek(&y, 8.0, 2.5).expect("seek");
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].0 as i64 - 60).abs() <= 1, "{hits:?}");
        assert!((hits[1].0 as i64 - 130).abs() <= 1, "{hits:?}");
        assert!(hits[0].1 > hits[1].1, "{hits:?}");
        assert!(seek(&vec![50.0; 200], 8.0, 2.5).expect("seek").is_empty());
    }

    /// subac: points on a line are never touched (at factor 1 or 1.0001); a
    /// peak on that line is flattened back onto it, never below the data.
    #[test]
    fn subac_flattens_a_peak_onto_its_baseline() {
        let line: Vec<f64> = (0..100).map(|i| 10.0 + 0.1 * i as f64).collect();
        let y: Vec<f64> = line
            .iter()
            .enumerate()
            .map(|(i, b)| b + 20.0 * gauss(i as f64, 1.0, 50.0, 6.0) * 6.0)
            .collect();
        let bg = subac(&y, 1.0, ESTIMATE_STRIP_ITERATIONS);
        for i in 0..100 {
            assert!(
                (bg[i] - line[i]).abs() < 1e-3,
                "{i}: {} vs {}",
                bg[i],
                line[i]
            );
            assert!(bg[i] <= y[i] + 1e-12);
        }
        for c in [1.0, BKG_STRIP_FACTOR] {
            let same = subac(&line, c, BKG_STRIP_ITERATIONS);
            assert!(same.iter().zip(&line).all(|(a, b)| (a - b).abs() < 1e-12));
        }
    }

    /// guess_yscaling reproduces PyMca's number for a short curve (worked by
    /// hand from `numpy.convolve(y, [1, 1, 1]) / 3` against `y[1:-1]`:
    /// χ² = 233/144), and is 1 when nothing can be compared.
    #[test]
    fn guess_yscaling_matches_pymca() {
        let s = guess_yscaling(&[1.0, 2.0, 4.0, 2.0, 1.0, 3.0]);
        assert!((s - 144.0 / 233.0).abs() < 1e-12, "{s}");
        assert_eq!(guess_yscaling(&[0.0; 6]), 1.0);
        assert_eq!(guess_yscaling(&[1.0, 2.0]), 1.0);
    }

    /// Gefit's constraints: a positive parameter whose free optimum is
    /// negative stays non-negative; a quoted one stays inside its range; a
    /// factor-tied one follows its master, uncertainty included; a free fit
    /// of a line converges.
    #[test]
    fn leastsq_honours_positive_quoted_and_factor_constraints() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let line = |x: &[f64], p: &[f64]| x.iter().map(|v| p[0] + p[1] * v).collect::<Vec<_>>();
        let y: Vec<f64> = x.iter().map(|v| 2.0 - 1.0 * v).collect();
        let free = leastsq(line, &x, &y, &[0.0, 0.0], &[Constraint::Free; 2], 100).expect("fit");
        assert!(
            (free.params[0] - 2.0).abs() < 1e-4 && (free.params[1] + 1.0).abs() < 1e-4,
            "{:?}",
            free.params
        );
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
        assert!(pos.chisq_reduced > free.chisq_reduced);

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

        let y: Vec<f64> = x.iter().map(|v| 3.0 + 6.0 * v).collect();
        let tied = leastsq(
            line,
            &x,
            &y,
            &[1.0, 0.0],
            &[Constraint::Free, Constraint::Factor(0, 2.0)],
            100,
        )
        .expect("fit");
        assert!((tied.params[0] - 3.0).abs() < 1e-6, "{:?}", tied.params);
        assert_eq!(tied.params[1], 2.0 * tied.params[0]);
        assert_eq!(tied.sigma[1], 2.0 * tied.sigma[0]);
        assert!(tied.sigma[0] > 0.0);
    }
}
