//! The feffit FT configuration, a port of `larch.xafs.feffit.TransformGroup`.
//!
//! Holds the k- and R-space transform parameters and the cached window arrays,
//! and provides the internal `fftf` (forward) / `fftr` (reverse) transforms used
//! to build the fit residual. The FFT kernels come from `xafsft`
//! (`xftf_fast` / `xftr_fast`), so this layer only owns the windowing and
//! k-weighting that larch's `TransformGroup.fftf`/`fftr` apply before the FFT.

use std::f64::consts::PI;

use num_complex::Complex64;
use xafsft::{fft_padded, ftwindow, ifft_padded, xftf_fast, xftr_fast, Window};

/// Which space the fit residual is evaluated in (larch `fitspace`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitSpace {
    /// fit in k-space (`'k'`).
    K,
    /// fit in R-space (`'r'`) — the larch default.
    R,
    /// fit in back-transformed q-space (`'q'`).
    Q,
    /// fit in the Cauchy-wavelet transform (`'w'`).
    W,
}

/// FT parameters plus cached k/R windows (`TransformGroup`).
#[derive(Debug, Clone)]
pub struct Transform {
    pub kmin: f64,
    pub kmax: f64,
    /// The k-weight(s) the fit is evaluated at. larch's `kweight` is either a
    /// scalar or a list; this port stores it uniformly as a non-empty list (a
    /// single-element list is the scalar case). When more than one is given the
    /// residual is the per-k-weight residuals concatenated, exactly larch's
    /// list-valued `kweight` (`FeffitDataSet._residual`).
    pub kweight: Vec<i32>,
    pub dk: f64,
    pub dk2: Option<f64>,
    pub window: Window,
    pub nfft: usize,
    pub kstep: f64,
    pub rmin: f64,
    pub rmax: f64,
    pub dr: f64,
    pub dr2: Option<f64>,
    pub rwindow: Window,
    pub rbkg: f64,
    pub fitspace: FitSpace,

    rstep: f64,
    /// `kstep * arange(nfft)` — the full FFT-grid k array.
    k_: Vec<f64>,
    kwin: Vec<f64>,
    rwin: Vec<f64>,
}

impl Transform {
    /// Construct a transform; the k/R windows are built eagerly (they depend
    /// only on the FT parameters, not on the data, exactly as larch caches them).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kmin: f64,
        kmax: f64,
        kweight: Vec<i32>,
        dk: f64,
        dk2: Option<f64>,
        window: Window,
        nfft: usize,
        kstep: f64,
        rmin: f64,
        rmax: f64,
        dr: f64,
        dr2: Option<f64>,
        rwindow: Window,
        rbkg: f64,
        fitspace: FitSpace,
    ) -> Self {
        assert!(
            !kweight.is_empty(),
            "Transform requires at least one k-weight"
        );
        let rstep = PI / (kstep * nfft as f64);
        let k_: Vec<f64> = (0..nfft).map(|i| kstep * i as f64).collect();
        let r_: Vec<f64> = (0..nfft).map(|i| rstep * i as f64).collect();

        // kwin: ftwindow over the full nfft k-grid (larch builds it on self.k_).
        let kwin = ftwindow(&k_, Some(kmin), Some(kmax), dk, dk2, window);
        // rwin: xmin = max(rbkg, rmin), over the full nfft R-grid.
        let rwin_xmin = rbkg.max(rmin);
        let rwin = ftwindow(&r_, Some(rwin_xmin), Some(rmax), dr, dr2, rwindow);

        Transform {
            kmin,
            kmax,
            kweight,
            dk,
            dk2,
            window,
            nfft,
            kstep,
            rmin,
            rmax,
            dr,
            dr2,
            rwindow,
            rbkg,
            fitspace,
            rstep,
            k_,
            kwin,
            rwin,
        }
    }

    /// larch's default transform: kmin=0, kmax=20, kweight=2, dk=4, kaiser k-win;
    /// rmin=0, rmax=10, dr=0, hanning R-win; nfft=2048, kstep=0.05, fit in R.
    pub fn defaults() -> Self {
        Transform::new(
            0.0,
            20.0,
            vec![2],
            4.0,
            None,
            Window::Kaiser,
            2048,
            0.05,
            0.0,
            10.0,
            0.0,
            None,
            Window::Hanning,
            0.0,
            FitSpace::R,
        )
    }

    /// `rstep = pi / (kstep * nfft)`.
    pub fn rstep(&self) -> f64 {
        self.rstep
    }

    /// The full FFT-grid k array (`kstep * arange(nfft)`).
    pub fn k_grid(&self) -> &[f64] {
        &self.k_
    }

    /// The cached k-window over the full FFT grid.
    pub fn kwin(&self) -> &[f64] {
        &self.kwin
    }

    /// The cached R-window over the full FFT grid.
    pub fn rwin(&self) -> &[f64] {
        &self.rwin
    }

    /// Forward FT of `chi` (on the `k_` grid), windowed and k-weighted:
    /// `xftf_fast(chi * kwin * k_**kweight)`. Port of `TransformGroup.fftf`.
    pub fn fftf(&self, chi: &[f64], kweight: i32) -> Vec<Complex64> {
        let m = chi.len();
        let cx: Vec<Complex64> = (0..m)
            .map(|i| Complex64::new(chi[i] * self.kwin[i] * self.k_[i].powi(kweight), 0.0))
            .collect();
        xftf_fast(&cx, self.nfft, self.kstep)
    }

    /// Reverse FT of `chir`, windowed by the R-window: `xftr_fast(chir * rwin)`.
    /// Port of `TransformGroup.fftr`.
    pub fn fftr(&self, chir: &[Complex64]) -> Vec<Complex64> {
        let m = chir.len();
        let cx: Vec<Complex64> = (0..m).map(|i| chir[i] * self.rwin[i]).collect();
        xftr_fast(&cx, self.nfft, self.kstep)
    }

    /// The Cauchy-wavelet transform of `chi`, restricted to the fit's
    /// (R, k) window. Port of `TransformGroup.cwt`.
    ///
    /// Returns the complex wavelet on the masked region as a flat row-major
    /// (R outer, k inner) buffer of length `nrows * ncols`, paired with `ncols`
    /// so the caller can recover the 2-D shape. Only the rows/cols inside the
    /// `[rmin, rmax) × [kmin, kmax)` mask are computed (larch zeroes the rest
    /// with `_cauchymask` and then slices to exactly this region).
    pub fn cwt(&self, chi: &[f64], kweight: i32) -> (Vec<Complex64>, usize) {
        let nkpts = chi.len();
        let nfft = self.nfft;
        let kstep = self.kstep;
        let rstep = self.rstep;

        // apply k-weighting + window (larch: chi*kwin*k**kweight when kweight!=0)
        let weighted: Vec<f64> = if kweight != 0 {
            (0..nkpts)
                .map(|i| chi[i] * self.kwin[i] * self.k_[i].powi(kweight))
                .collect()
        } else {
            chi.to_vec()
        };

        // chix: length nfft/2, zero-padded, then FFT to length 2*nfft, keep [:nfft]
        let half = nfft / 2;
        let m = nkpts.min(half);
        let chix: Vec<Complex64> = (0..half)
            .map(|i| {
                if i < m {
                    Complex64::new(weighted[i], 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        let ffchi_full = fft_padded(&chix, 2 * nfft);
        let ffchi = &ffchi_full[..nfft];

        // nrpts is the FULL R-grid count: it enters the Cauchy filter (the wavelet
        // order) and the normalisation, independent of which rows we keep.
        let nrpts = (self.rmax / rstep).round() as usize;
        let omega: Vec<f64> = (0..nfft)
            .map(|j| PI * j as f64 / (kstep * nfft as f64))
            .collect();
        // cauchy_sum = log(2π) - log(nrpts!) = log(2π) - Σ_{j=1..nrpts} log(j)
        let log_fact: f64 = (1..=nrpts).map(|j| (j as f64).ln()).sum();
        let cauchy_sum = (2.0 * PI).ln() - log_fact;

        // mask / slice bounds (larch make_cwt_arrays); int() truncates toward 0
        let nfft_half = nfft as f64 / 2.0;
        let ikmin = (0.0f64.max(0.01 + self.kmin / kstep)) as usize;
        let ikmax = ((nfft_half.min(0.01 + self.kmax / kstep)) as usize).min(nkpts);
        let irmin = (0.0f64.max(0.01 + self.rmin / rstep)) as usize;
        let irmax = (nfft_half.min(0.01 + self.rmax / rstep)) as usize;
        let ncols = ikmax.saturating_sub(ikmin);

        let mut out = Vec::with_capacity(irmax.saturating_sub(irmin) * ncols);
        for i in irmin..irmax {
            // r[0] is forced to 1e-19 in larch to avoid a divide-by-zero in alpha
            let r_i = if i == 0 { 1.0e-19 } else { rstep * i as f64 };
            let alpha = nrpts as f64 / (2.0 * r_i);
            let prod: Vec<Complex64> = (0..nfft)
                .map(|j| {
                    let aom = alpha * omega[j];
                    // omega[0]=0 -> ln(0)=-inf -> exp=0, matching numpy
                    let filt = cauchy_sum + nrpts as f64 * aom.ln() - aom;
                    ffchi[j] * filt.exp()
                })
                .collect();
            let row = ifft_padded(&prod, 2 * nfft);
            out.extend_from_slice(&row[ikmin..ikmax]);
        }
        (out, ncols)
    }
}
