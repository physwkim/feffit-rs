//! The feffit FT configuration, a port of `larch.xafs.feffit.TransformGroup`.
//!
//! Holds the k- and R-space transform parameters and the cached window arrays,
//! and provides the internal `fftf` (forward) / `fftr` (reverse) transforms used
//! to build the fit residual. The FFT kernels come from `xafsft`
//! (`xftf_fast` / `xftr_fast`), so this layer only owns the windowing and
//! k-weighting that larch's `TransformGroup.fftf`/`fftr` apply before the FFT.

use std::f64::consts::PI;

use num_complex::Complex64;
use xafsft::{ftwindow, xftf_fast, xftr_fast, Window};

/// Which space the fit residual is evaluated in (larch `fitspace`).
///
/// larch also supports `'w'` (Cauchy wavelet); that is intentionally not ported
/// here — see the crate docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitSpace {
    /// fit in k-space (`'k'`).
    K,
    /// fit in R-space (`'r'`) — the larch default.
    R,
    /// fit in back-transformed q-space (`'q'`).
    Q,
}

/// FT parameters plus cached k/R windows (`TransformGroup`).
#[derive(Debug, Clone)]
pub struct Transform {
    pub kmin: f64,
    pub kmax: f64,
    pub kweight: i32,
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
        kweight: i32,
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
            2,
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
}
