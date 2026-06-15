//! XAFS Fourier transforms, a port of `larch.xafs.xafsft`:
//! `xftf_fast`, `xftr_fast`, `xftf_prep`, `xftf`, `xftr`.
//!
//! FFT parity note: larch uses `scipy.fftpack.fft/ifft`; this port uses
//! `rustfft`. Both compute the same unnormalized DFT (forward `exp(-2πi)`),
//! so results agree to FFT round-off (~1e-13), not bit-for-bit. scipy `ifft`
//! is normalized by `1/N`; rustfft's inverse is not, so we divide by `nfft`.

use std::f64::consts::PI;

use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::window::{Window, ftwindow};

/// Forward FT of an already-prepared (windowed, k-weighted) array.
/// Returns `(kstep/sqrt(pi)) * fft(zero_pad(chi))[..nfft/2]`.
pub fn xftf_fast(chi: &[Complex64], nfft: usize, kstep: f64) -> Vec<Complex64> {
    let mut buf = vec![Complex64::new(0.0, 0.0); nfft];
    let m = chi.len().min(nfft);
    buf[..m].copy_from_slice(&chi[..m]);

    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(nfft).process(&mut buf);

    let scale = kstep / PI.sqrt();
    buf[..nfft / 2].iter().map(|c| c * scale).collect()
}

/// Reverse FT of a complex chi(R). Returns
/// `(4*sqrt(pi)/kstep) * ifft(zero_pad(chir))[..nfft/2]`, where `ifft` is the
/// `1/N`-normalized inverse (matching `scipy.fftpack.ifft`).
pub fn xftr_fast(chir: &[Complex64], nfft: usize, kstep: f64) -> Vec<Complex64> {
    let mut buf = vec![Complex64::new(0.0, 0.0); nfft];
    let m = chir.len().min(nfft);
    buf[..m].copy_from_slice(&chir[..m]);

    let mut planner = FftPlanner::new();
    planner.plan_fft_inverse(nfft).process(&mut buf);

    // rustfft inverse is unnormalized -> divide by nfft for scipy.ifft parity
    let scale = (4.0 * PI.sqrt() / kstep) / nfft as f64;
    buf[..nfft / 2].iter().map(|c| c * scale).collect()
}

/// Forward DFT with numpy `fft(a, n)` semantics: zero-pad (or truncate) `input`
/// to length `n`, then the unnormalized forward transform (`exp(-2πi)`). Returns
/// all `n` bins; the caller slices what it needs. Used by the Cauchy-wavelet
/// transform, which works on raw (un-XAFS-scaled) FFTs of size `2*nfft`.
pub fn fft_padded(input: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    let m = input.len().min(n);
    buf[..m].copy_from_slice(&input[..m]);
    FftPlanner::new().plan_fft_forward(n).process(&mut buf);
    buf
}

/// Inverse DFT with numpy `ifft(a, n)` semantics: zero-pad (or truncate) `input`
/// to length `n`, the inverse transform (`exp(+2πi)`), normalized by `1/n`
/// (rustfft's inverse is unnormalized). Returns all `n` points.
pub fn ifft_padded(input: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut buf = vec![Complex64::new(0.0, 0.0); n];
    let m = input.len().min(n);
    buf[..m].copy_from_slice(&input[..m]);
    FftPlanner::new().plan_fft_inverse(n).process(&mut buf);
    let inv = 1.0 / n as f64;
    for c in &mut buf {
        *c *= inv;
    }
    buf
}

/// `numpy.interp` (linear, endpoint-clamped); `xp` ascending.
fn interp(xq: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if xq <= xp[0] {
        return fp[0];
    }
    if xq >= xp[n - 1] {
        return fp[n - 1];
    }
    let i = match xp.binary_search_by(|v| v.partial_cmp(&xq).unwrap()) {
        Ok(i) => return fp[i],
        Err(i) => i,
    };
    let (x0, x1) = (xp[i - 1], xp[i]);
    let (y0, y1) = (fp[i - 1], fp[i]);
    y0 + (y1 - y0) * (xq - x0) / (x1 - x0)
}

/// Build the weighted chi(k) on a uniform grid plus the FT window
/// (`xftf_prep`). Returns `(weighted_chi, win)`, each of length `npts`.
#[allow(clippy::too_many_arguments)]
pub fn xftf_prep(
    k: &[f64],
    chi: &[f64],
    kmin: f64,
    kmax: f64,
    kweight: i32,
    dk: f64,
    dk2: Option<f64>,
    window: Window,
    nfft: usize,
    kstep: f64,
) -> (Vec<f64>, Vec<f64>) {
    let _ = nfft; // nfft unused here, kept for signature parity with larch
    let dk2 = dk2.unwrap_or(dk);
    let kmaxv = max_of(k);
    let npts = (1.01 + kmaxv / kstep) as usize;
    let k_max = kmaxv.max(kmax + dk2);
    let nk = (1.01 + k_max / kstep) as usize;
    let k_: Vec<f64> = (0..nk).map(|i| kstep * i as f64).collect();
    let chi_: Vec<f64> = k_.iter().map(|&kq| interp(kq, k, chi)).collect();
    let win = ftwindow(&k_, Some(kmin), Some(kmax), dk, Some(dk2), window);

    let weighted: Vec<f64> = (0..npts).map(|i| chi_[i] * k_[i].powi(kweight)).collect();
    (weighted, win[..npts].to_vec())
}

/// Output of a forward transform `xftf`.
#[derive(Debug, Clone)]
pub struct XftfOut {
    pub kwin: Vec<f64>,
    pub r: Vec<f64>,
    pub chir: Vec<Complex64>,
    pub chir_mag: Vec<f64>,
    pub chir_re: Vec<f64>,
    pub chir_im: Vec<f64>,
}

/// Full forward XAFS Fourier transform chi(k) -> chi(R) (`xftf`).
#[allow(clippy::too_many_arguments)]
pub fn xftf(
    k: &[f64],
    chi: &[f64],
    kmin: f64,
    kmax: f64,
    kweight: i32,
    dk: f64,
    dk2: Option<f64>,
    window: Window,
    rmax_out: f64,
    nfft: usize,
    kstep: Option<f64>,
) -> XftfOut {
    let kstep = kstep.unwrap_or(k[1] - k[0]);
    let (cchi, win) = xftf_prep(k, chi, kmin, kmax, kweight, dk, dk2, window, nfft, kstep);
    let weighted: Vec<Complex64> = cchi
        .iter()
        .zip(&win)
        .map(|(&c, &w)| Complex64::new(c * w, 0.0))
        .collect();
    let out = xftf_fast(&weighted, nfft, kstep);

    let rstep = PI / (kstep * nfft as f64);
    let irmax = ((nfft as f64 / 2.0).min(1.01 + rmax_out / rstep)) as usize;
    let r: Vec<f64> = (0..irmax).map(|i| rstep * i as f64).collect();
    let chir: Vec<Complex64> = out[..irmax].to_vec();
    let chir_mag = chir.iter().map(|c| c.norm()).collect();
    let chir_re = chir.iter().map(|c| c.re).collect();
    let chir_im = chir.iter().map(|c| c.im).collect();
    let kwin_len = win.len().min(chi.len());
    XftfOut {
        kwin: win[..kwin_len].to_vec(),
        r,
        chir,
        chir_mag,
        chir_re,
        chir_im,
    }
}

/// Output of a reverse transform `xftr`.
#[derive(Debug, Clone)]
pub struct XftrOut {
    pub rwin: Vec<f64>,
    pub q: Vec<f64>,
    pub chiq: Vec<Complex64>,
    pub chiq_mag: Vec<f64>,
    pub chiq_re: Vec<f64>,
    pub chiq_im: Vec<f64>,
}

/// Full reverse XAFS Fourier transform chi(R) -> chi(q) (`xftr`).
///
/// `chir` is taken as complex (larch's `complex128` branch), so `scale = 0.5`.
#[allow(clippy::too_many_arguments)]
pub fn xftr(
    r: &[f64],
    chir: &[Complex64],
    rmin: f64,
    rmax: f64,
    dr: f64,
    dr2: Option<f64>,
    rw: i32,
    window: Window,
    qmax_out: f64,
    nfft: usize,
) -> XftrOut {
    let rstep = r[1] - r[0];
    let kstep = PI / (rstep * nfft as f64);
    let scale = 0.5; // complex chir branch

    let r_: Vec<f64> = (0..nfft).map(|i| rstep * i as f64).collect();
    let win = ftwindow(&r_, Some(rmin), Some(rmax), dr, dr2, window);

    let m = chir.len().min(nfft);
    let prepped: Vec<Complex64> = (0..nfft)
        .map(|i| {
            if i < m {
                chir[i] * win[i] * r_[i].powi(rw)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    let raw = xftr_fast(&prepped, nfft, kstep);
    let out: Vec<Complex64> = raw.iter().map(|c| c * scale).collect();

    let nq = (1.05 + qmax_out / kstep) as usize;
    let q: Vec<f64> = (0..nq)
        .map(|i| qmax_out * i as f64 / (nq as f64 - 1.0))
        .collect();
    let nkpts = q.len();

    let chiq = out[..nkpts.min(out.len())].to_vec();
    let chiq_mag = chiq.iter().map(|c| c.norm()).collect();
    let chiq_re = chiq.iter().map(|c| c.re).collect();
    let chiq_im = chiq.iter().map(|c| c.im).collect();
    let rwin_len = win.len().min(chir.len());
    XftrOut {
        rwin: win[..rwin_len].to_vec(),
        q,
        chiq,
        chiq_mag,
        chiq_re,
        chiq_im,
    }
}

fn max_of(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
