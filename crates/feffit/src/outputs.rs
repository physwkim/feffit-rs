//! Fit-output arrays, a port of `TransformGroup._xafsft` and
//! `FeffitDataSet.save_outputs`.
//!
//! After a fit, larch forward-transforms the data χ(k) and the model χ(k) (and
//! optionally each path's χ(k)) into χ(R) on a uniform R-grid out to `rmax_out`,
//! and back-transforms χ(R) into χ(q) on a uniform q-grid out to `kmax + 2`.
//! These are the arrays a caller plots or stores after `feffit()`; they are not
//! part of the residual/minimisation.

use num_complex::Complex64;

use crate::transform::Transform;

/// The forward/back-FT outputs for one χ(k) (larch `_xafsft`): χ(R) on the
/// R-grid `r` and χ(q) on the q-grid `q`. `chir_*`/`chiq_*` are the real,
/// imaginary, magnitude, and (larch-unwrapped) phase parts.
#[derive(Debug, Clone)]
pub struct XafsOutput {
    /// Uniform R-grid, `rstep * arange(len)` out to `rmax_out`.
    pub r: Vec<f64>,
    pub chir_re: Vec<f64>,
    pub chir_im: Vec<f64>,
    pub chir_mag: Vec<f64>,
    pub chir_pha: Vec<f64>,
    /// Uniform q-grid, `linspace(0, kmax+2, nq)` (larch `_xafsft`).
    pub q: Vec<f64>,
    pub chiq_re: Vec<f64>,
    pub chiq_im: Vec<f64>,
    pub chiq_mag: Vec<f64>,
    pub chiq_pha: Vec<f64>,
}

/// The output arrays for one dataset (larch `save_outputs`): the data and model
/// transforms, plus one per path when `path_outputs` is set.
#[derive(Debug, Clone)]
pub struct DataSetOutput {
    pub data: XafsOutput,
    pub model: XafsOutput,
    /// One [`XafsOutput`] per path, in dataset path order (empty when
    /// `path_outputs` is false).
    pub paths: Vec<XafsOutput>,
}

/// Round half to even (`numpy.round`/IEEE round-ties-to-even) for `x >= 0`.
/// Hand-rolled because `f64::round_ties_even` is newer than the crate MSRV
/// (1.74); `f64::round` rounds half *away* from zero, which differs from numpy
/// at exact halves.
fn round_half_even(x: f64) -> f64 {
    let lower = x.floor();
    let frac = x - lower;
    if frac < 0.5 {
        lower
    } else if frac > 0.5 {
        lower + 1.0
    } else if (lower as i64) % 2 == 0 {
        lower
    } else {
        lower + 1.0
    }
}

/// Phase modulo 2π jumps, larch's `complex_phase` (larch/math/utils.py): the
/// `arctan2` phase with cumulative π-jumps removed. `np.round` rounds half to
/// even, matched by [`round_half_even`].
fn complex_phase(z: &[Complex64]) -> Vec<f64> {
    let phase: Vec<f64> = z.iter().map(|c| c.im.atan2(c.re)).collect();
    let mut out = phase.clone();
    let mut cum = 0.0;
    for i in 1..phase.len() {
        let d = (phase[i] - phase[i - 1]) / std::f64::consts::PI;
        // np.round(abs(d)) * np.sign(d); round=0 unless |d|>0.5 (where d≠0 so
        // sign is well-defined), so the np.sign(0)=0 vs signum(0)=1 difference
        // never reaches the product.
        cum += round_half_even(d.abs()) * d.signum();
        out[i] -= std::f64::consts::PI * cum;
    }
    out
}

/// `numpy.linspace(start, stop, num)`: `num` evenly spaced points including both
/// endpoints (the last is set to `stop` exactly, as numpy does).
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

/// Forward/back FT of one χ(k) into output arrays (larch `_xafsft`).
///
/// `chi` is sampled on the transform's k-grid (`trans.k_grid()`); the forward FT
/// uses the first k-weight (larch `get_kweight`). `rmax_out` bounds the R-grid.
pub fn xafsft(trans: &Transform, chi: &[f64], rmax_out: f64) -> XafsOutput {
    let kw0 = trans.kweight[0];
    let outr = trans.fftf(chi, kw0);

    let rstep = trans.rstep();
    let nfft_half = trans.nfft as f64 / 2.0;
    // int(min(nfft/2, 1.01 + rmax_out/rstep)) — Python int() truncates toward 0.
    let irmax = nfft_half.min(1.01 + rmax_out / rstep) as usize;

    let r: Vec<f64> = (0..irmax).map(|i| rstep * i as f64).collect();
    let chir_re: Vec<f64> = outr[..irmax].iter().map(|c| c.re).collect();
    let chir_im: Vec<f64> = outr[..irmax].iter().map(|c| c.im).collect();
    let chir_mag: Vec<f64> = outr[..irmax].iter().map(|c| c.norm()).collect();
    let chir_pha = complex_phase(&outr[..irmax]);

    // χ(q): back-transform the *full* outr (larch passes the un-sliced outr to
    // fftr), then take the first nq points on the q-grid.
    let outq = trans.fftr(&outr);
    let qmax_out = trans.kmax + 2.0;
    let nq = (1.05 + qmax_out / trans.kstep) as usize;
    let q = linspace(0.0, qmax_out, nq);
    let chiq_re: Vec<f64> = outq[..nq].iter().map(|c| c.re).collect();
    let chiq_im: Vec<f64> = outq[..nq].iter().map(|c| c.im).collect();
    let chiq_mag: Vec<f64> = outq[..nq].iter().map(|c| c.norm()).collect();
    let chiq_pha = complex_phase(&outq[..nq]);

    XafsOutput {
        r,
        chir_re,
        chir_im,
        chir_mag,
        chir_pha,
        q,
        chiq_re,
        chiq_im,
        chiq_mag,
        chiq_pha,
    }
}
