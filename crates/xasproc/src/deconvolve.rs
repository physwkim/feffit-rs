//! XANES spectral (de)convolution — port of `larch.xafs.deconvolve`
//! (`xas_deconvolve` + `xas_convolve`).
//!
//! `xas_deconvolve` sharpens a normalized `mu(E)` by dividing out a Lorentzian
//! or Gaussian peak shape (via `scipy.signal.deconvolve`, i.e. an `lfilter`
//! polynomial division), optionally Savitzky–Golay smoothed. `xas_convolve`
//! does the forward broadening with `np.convolve`. Both work on a uniform energy
//! grid built from the data step, using a cubic spline (FITPACK `splrep(s=0)`)
//! for interpolation with larch's endpoint extrapolation.

use crate::mathutils::{convolve_full, convolve_valid, interp_cubic, remove_dups, solve_linear};

const TINY_ENERGY: f64 = 0.00050;
/// lmfit `s2pi` = sqrt(2*pi).
const S2PI: f64 = 2.506_628_274_631_000_2;
/// lmfit `tiny` floor used in the line-shape denominators.
const LS_TINY: f64 = 1.0e-15;

/// Peak shape used by the (de)convolution kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeconvForm {
    /// `lmfit.lineshapes.lorentzian` (larch default).
    #[default]
    Lorentzian,
    /// `lmfit.lineshapes.gaussian`.
    Gaussian,
}

/// `lmfit.lineshapes.gaussian(x, amplitude=1, center, sigma)`.
fn gaussian(x: f64, center: f64, sigma: f64) -> f64 {
    (1.0 / (LS_TINY).max(S2PI * sigma))
        * (-(x - center) * (x - center) / (LS_TINY).max(2.0 * sigma * sigma)).exp()
}

/// `lmfit.lineshapes.lorentzian(x, amplitude=1, center, sigma)`.
fn lorentzian(x: f64, center: f64, sigma: f64) -> f64 {
    let t = (x - center) / (LS_TINY).max(sigma);
    (1.0 / (1.0 + t * t)) / (LS_TINY).max(std::f64::consts::PI * sigma)
}

fn kernel(form: DeconvForm, x: &[f64], sigma: f64) -> Vec<f64> {
    match form {
        DeconvForm::Lorentzian => x.iter().map(|&v| lorentzian(v, 0.0, sigma)).collect(),
        DeconvForm::Gaussian => x.iter().map(|&v| gaussian(v, 0.0, sigma)).collect(),
    }
}

/// `scipy.signal.lfilter(b, a, x)` via the direct-form-II-transposed recurrence
/// (matching scipy's `sigtools._linear_filter`).
fn lfilter(b_in: &[f64], a_in: &[f64], x: &[f64]) -> Vec<f64> {
    let a0 = a_in[0];
    let nfilt = a_in.len().max(b_in.len());
    let mut b = vec![0.0; nfilt];
    let mut a = vec![0.0; nfilt];
    for (bi, &v) in b.iter_mut().zip(b_in) {
        *bi = v / a0;
    }
    for (ai, &v) in a.iter_mut().zip(a_in) {
        *ai = v / a0;
    }
    let mut z = vec![0.0; nfilt.saturating_sub(1)];
    let mut y = vec![0.0; x.len()];
    for (ym, &xm) in y.iter_mut().zip(x) {
        let yn = if nfilt > 1 {
            b[0] * xm + z[0]
        } else {
            b[0] * xm
        };
        if nfilt > 1 {
            for i in 0..nfilt - 2 {
                z[i] = b[i + 1] * xm + z[i + 1] - a[i + 1] * yn;
            }
            z[nfilt - 2] = b[nfilt - 1] * xm - a[nfilt - 1] * yn;
        }
        *ym = yn;
    }
    y
}

/// `scipy.signal.deconvolve(signal, divisor)` quotient (the remainder, which
/// larch discards, is not computed).
fn deconvolve_quotient(signal: &[f64], divisor: &[f64]) -> Vec<f64> {
    let (nn, d) = (signal.len(), divisor.len());
    if d > nn {
        return Vec::new();
    }
    let mut input = vec![0.0; nn - d + 1];
    input[0] = 1.0;
    lfilter(signal, divisor, &input)
}

/// `larch.math.savitzky_golay(y, window, order, deriv=0)`: polynomial-window
/// smoothing with edge reflection. `pinv` of the Vandermonde is computed via the
/// normal equations (well-conditioned for the small window).
fn savitzky_golay(y: &[f64], window: usize, order: usize) -> Vec<f64> {
    let mut window = window;
    if window < order + 2 {
        window = order + 3;
    }
    if window % 2 != 1 {
        window += 1;
    }
    let half = (window - 1) / 2;
    let np1 = order + 1;

    // B^T B where B[k][i] = kv^i, kv in -half..=half
    let kvs: Vec<f64> = (0..window).map(|k| k as f64 - half as f64).collect();
    let mut btb = vec![vec![0.0; np1]; np1];
    for &kv in &kvs {
        let mut pows = vec![1.0; np1];
        for i in 1..np1 {
            pows[i] = pows[i - 1] * kv;
        }
        for i in 0..np1 {
            for j in 0..np1 {
                btb[i][j] += pows[i] * pows[j];
            }
        }
    }
    // m = row `deriv`(=0) of pinv(B) = (B^T B)^-1 B^T -> solve (B^T B) z = e_0
    let mut e0 = vec![0.0; np1];
    e0[0] = 1.0;
    let z = solve_linear(btb, e0);
    // m[k] = sum_j z[j] * kv_k^j
    let m: Vec<f64> = kvs
        .iter()
        .map(|&kv| {
            let mut p = 1.0;
            let mut acc = 0.0;
            for &zj in &z {
                acc += zj * p;
                p *= kv;
            }
            acc
        })
        .collect();

    // edge reflection: firstvals = y[0] - |y[1..=half] reversed - y[0]|
    let n = y.len();
    let mut padded = Vec::with_capacity(n + 2 * half);
    for k in 0..half {
        // reversed y[1..=half] -> y[half-k]
        padded.push(y[0] - (y[half - k] - y[0]).abs());
    }
    padded.extend_from_slice(y);
    for k in 0..half {
        // reversed y[n-1-half .. n-1] -> y[n-2-k]
        padded.push(y[n - 1] + (y[n - 2 - k] - y[n - 1]).abs());
    }
    convolve_valid(&m, &padded)
}

/// Tunable inputs to [`xas_deconvolve`].
#[derive(Debug, Clone)]
pub struct DeconvParams {
    /// peak shape. Default Lorentzian.
    pub form: DeconvForm,
    /// energy sigma (eV). Default 1.0.
    pub esigma: f64,
    /// energy shift (eV) applied to the result. Default 0.
    pub eshift: f64,
    /// Savitzky–Golay smoothing of the result. Default true.
    pub smooth: bool,
    /// SG window; `None` → `int(esigma/estep)`.
    pub sgwindow: Option<usize>,
    /// SG polynomial order. Default 3.
    pub sgorder: usize,
}

impl Default for DeconvParams {
    fn default() -> Self {
        DeconvParams {
            form: DeconvForm::Lorentzian,
            esigma: 1.0,
            eshift: 0.0,
            smooth: true,
            sgwindow: None,
            sgorder: 3,
        }
    }
}

/// `larch.xafs.deconvolve.xas_deconvolve`: sharpen `norm(E)` by deconvolving a
/// peak shape. Returns `deconv` on the input energy grid.
pub fn xas_deconvolve(energy: &[f64], norm: &[f64], p: &DeconvParams) -> Vec<f64> {
    assert_eq!(energy.len(), norm.len(), "energy and norm length mismatch");
    let esigma = p.esigma;
    let eshift = p.eshift + 0.5 * esigma;

    let en0 = remove_dups(energy, TINY_ENERGY);
    let estep1 = (0.1 * en0[0]) as i64 as f64 * 2.0e-5;
    let en: Vec<f64> = en0.iter().map(|&e| e - en0[0]).collect();
    let min_step = en
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let estep = estep1.max(0.01 * ((min_step * 100.0) as i64 as f64));
    let enmax = en.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut npts = 1 + (enmax / estep) as usize;
    let mut estep = estep;
    if npts > 25000 {
        npts = 25001;
        estep = enmax / 25000.0;
    }

    let x: Vec<f64> = (0..npts).map(|i| i as f64 * estep).collect();
    let y = interp_cubic(&en, norm, &x);

    // ramp-extended signal
    let ylast = y[y.len() - 1];
    let mut yext = y.clone();
    yext.extend((0..y.len()).map(|i| i as f64 * ylast));

    let kern = kernel(p.form, &x, esigma);
    let ret_full = deconvolve_quotient(&yext, &kern);
    let nret = x.len().min(ret_full.len());
    let scale = yext[nret - 1] / ret_full[nret - 1];
    let mut ret: Vec<f64> = ret_full[..nret].iter().map(|&v| v * scale).collect();

    if p.smooth {
        let mut sgwindow = p.sgwindow.unwrap_or((esigma / estep) as usize);
        if sgwindow < p.sgorder + 1 {
            sgwindow = p.sgorder + 2;
        }
        if sgwindow.is_multiple_of(2) {
            sgwindow += 1;
        }
        ret = savitzky_golay(&ret, sgwindow, p.sgorder);
    }

    let xshift: Vec<f64> = x.iter().map(|&v| v + eshift).collect();
    interp_cubic(&xshift, &ret, &en)
}

/// Tunable inputs to [`xas_convolve`].
#[derive(Debug, Clone)]
pub struct ConvParams {
    /// peak shape. Default Lorentzian.
    pub form: DeconvForm,
    /// energy sigma (eV). Default 1.0.
    pub esigma: f64,
    /// energy shift (eV). Default 0.
    pub eshift: f64,
}

impl Default for ConvParams {
    fn default() -> Self {
        ConvParams {
            form: DeconvForm::Lorentzian,
            esigma: 1.0,
            eshift: 0.0,
        }
    }
}

/// `larch.xafs.deconvolve.xas_convolve`: broaden `norm(E)` by convolving a peak
/// shape. Returns `conv` on the input energy grid.
pub fn xas_convolve(energy: &[f64], norm: &[f64], p: &ConvParams) -> Vec<f64> {
    assert_eq!(energy.len(), norm.len(), "energy and norm length mismatch");
    let esigma = p.esigma;
    let eshift = p.eshift + 0.5 * esigma;

    let en0 = remove_dups(energy, TINY_ENERGY);
    let en: Vec<f64> = en0.iter().map(|&e| e - en0[0]).collect();
    let min_step = en
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let estep = 0.001_f64.max(0.001 * ((min_step * 1000.0) as i64 as f64));

    let npad = 1 + ((estep * 2.01).max(50.0 * esigma) / estep) as usize;
    let enmax = en.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let npts = npad + (enmax / estep) as usize;

    let x: Vec<f64> = (0..npts).map(|i| i as f64 * estep).collect();
    let y = interp_cubic(&en, norm, &x);

    let k = kernel(p.form, &x, esigma);
    let ret = convolve_full(&y, &k);

    let xshift: Vec<f64> = x.iter().map(|&v| v - eshift).collect();
    let out = interp_cubic(&xshift, &ret[..x.len()], &en);
    let ksum: f64 = k.iter().sum();
    out.iter().map(|&v| v / ksum).collect()
}
