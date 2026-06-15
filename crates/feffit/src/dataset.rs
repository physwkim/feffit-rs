//! A feffit dataset, a port of `larch.xafs.feffit.FeffitDataSet`.
//!
//! Ties together experimental chi(k) (`data`), a list of Feff paths, and a
//! [`Transform`], and produces the fit residual that the minimiser drives to
//! zero. This milestone covers the residual path for fixed (numeric) path
//! parameters in k/R/q space with a scalar k-weight; the `'w'` (wavelet) space
//! and list-valued k-weights are not ported.

use std::f64::consts::PI;

use feffdat::{ff2chi, interp_linear, FeffPath, Interp, KGrid};

use crate::transform::{FitSpace, Transform};

/// Truncate toward zero into a `usize` (Python `int()` semantics for x >= 0).
#[inline]
fn itrunc(x: f64) -> usize {
    x.trunc() as usize
}

/// Interleave a complex slice into `[re0, im0, re1, im1, ...]` (larch `realimag`).
fn realimag(z: &[num_complex::Complex64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * z.len());
    for c in z {
        out.push(c.re);
        out.push(c.im);
    }
    out
}

/// A dataset: data arrays, paths, transform, and (after `prepare_fit`) the
/// model k-grid, interpolated data, and noise estimates.
#[derive(Debug, Clone)]
pub struct DataSet {
    pub data_k: Vec<f64>,
    pub data_chi: Vec<f64>,
    pub paths: Vec<FeffPath>,
    pub transform: Transform,
    /// Interpolation used to evaluate path chi(k); larch's feffit default is cubic.
    pub interp: Interp,

    // populated by prepare_fit:
    model_k: Vec<f64>,
    chi_interp: Vec<f64>,
    n_idp: f64,
    epsilon_k: f64,
    epsilon_r: f64,
    prepared: bool,
}

impl DataSet {
    /// Build a dataset (call [`DataSet::prepare_fit`] before [`DataSet::residual`]).
    pub fn new(
        data_k: Vec<f64>,
        data_chi: Vec<f64>,
        paths: Vec<FeffPath>,
        transform: Transform,
    ) -> Self {
        DataSet {
            data_k,
            data_chi,
            paths,
            transform,
            interp: Interp::Cubic,
            model_k: Vec::new(),
            chi_interp: Vec::new(),
            n_idp: 0.0,
            epsilon_k: 0.0,
            epsilon_r: 0.0,
            prepared: false,
        }
    }

    /// Number of independent points: `1 + 2*(rmax-rmin)*(kmax-kmin)/pi`.
    pub fn n_idp(&self) -> f64 {
        self.n_idp
    }
    /// Uncertainty in chi(k).
    pub fn epsilon_k(&self) -> f64 {
        self.epsilon_k
    }
    /// Uncertainty in chi(R).
    pub fn epsilon_r(&self) -> f64 {
        self.epsilon_r
    }
    /// The model k-grid (`trans.k_[:ikmax]`).
    pub fn model_k(&self) -> &[f64] {
        &self.model_k
    }
    /// The data chi interpolated onto the model k-grid (`_chi`).
    pub fn chi_interp(&self) -> &[f64] {
        &self.chi_interp
    }

    /// Prepare the dataset for fitting (port of `prepare_fit`).
    ///
    /// `epsilon_k`: if `Some`, the explicit uncertainty is used via
    /// `set_epsilon_k`; if `None`, the noise is estimated from the high-R
    /// region (`estimate_noise`, rmin=15, rmax=30). The autobk `delta_chi`
    /// augmentation that larch's `prepare_fit` adds when no epsilon is given is
    /// not ported (no autobk here).
    pub fn prepare_fit(&mut self, epsilon_k: Option<f64>) {
        let trans = &self.transform;
        let kstep = trans.kstep;
        let kmax_data = self
            .data_k
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let ikmax = itrunc(1.01 + kmax_data / kstep);

        self.model_k = trans.k_grid()[..ikmax].to_vec();
        // _chi = numpy.interp(model.k, data.k, data.chi)  (linear, endpoint-clamped)
        self.chi_interp = self
            .model_k
            .iter()
            .map(|&kq| interp_linear(kq, &self.data_k, &self.data_chi))
            .collect();

        self.n_idp = 1.0 + 2.0 * (trans.rmax - trans.rmin) * (trans.kmax - trans.kmin) / PI;

        match epsilon_k {
            Some(eps_k) => self.set_epsilon_k(eps_k),
            None => {
                let chi = self.chi_interp.clone();
                self.estimate_noise(&chi, 15.0, 30.0);
            }
        }
        self.prepared = true;
    }

    /// Set epsilon_k / epsilon_r from an explicit scalar (port of `set_epsilon_k`,
    /// scalar-kweight branch).
    pub fn set_epsilon_k(&mut self, eps_k: f64) {
        let trans = &self.transform;
        let w = 2 * trans.kweight + 1;
        let denom = trans.kstep * (trans.kmax.powi(w) - trans.kmin.powi(w));
        let scale = 2.0 * (PI * w as f64 / denom).sqrt();
        self.epsilon_k = eps_k;
        self.epsilon_r = eps_k / scale;
    }

    /// Estimate epsilon_k / epsilon_r from high-R noise (port of `estimate_noise`,
    /// scalar-kweight branch).
    pub fn estimate_noise(&mut self, chi: &[f64], rmin: f64, rmax: f64) {
        let trans = &self.transform;
        let rstep = trans.rstep();
        let chir = trans.fftf(chi, trans.kweight);

        let irmin = itrunc(0.01 + rmin / rstep);
        let irmax = itrunc((trans.nfft as f64 / 2.0).min(1.01 + rmax / rstep));
        let highr = realimag(&chir[irmin..irmax]);

        // kwin_ave: mean window value scaled into the (kmax-kmin) range
        let kwin_sum: f64 = trans.kwin().iter().sum();
        let kwin_ave = kwin_sum * trans.kstep / (trans.kmax - trans.kmin);

        let ss: f64 = highr.iter().map(|v| v * v).sum();
        let eps_r = (ss / highr.len() as f64).sqrt() / kwin_ave;

        // Parseval scaling r -> k (note: a different convention than set_epsilon_k)
        let w = 2 * trans.kweight + 1;
        let denom = trans.kstep * (trans.kmax.powi(w) - trans.kmin.powi(w));
        let scale = (2.0 * PI * w as f64 / denom).sqrt();
        self.epsilon_k = scale * eps_r;
        self.epsilon_r = eps_r;
    }

    /// Sum the path chi(k) on the model k-grid (`ff2chi` over the paths),
    /// the model that the residual subtracts from the data.
    pub fn model_chi_sum(&mut self) -> Vec<f64> {
        let grid = KGrid::Explicit(self.model_k.clone());
        let (_mk, model_chi) = ff2chi(&mut self.paths, &grid, self.interp);
        model_chi
    }

    /// Compute the fit residual (port of `_residual`).
    ///
    /// `data_only = true` skips subtracting the model (used to extract the
    /// transformed data for statistics).
    pub fn residual(&mut self, data_only: bool) -> Vec<f64> {
        if !self.prepared {
            // larch lazily prepares with no explicit epsilon
            let chi = std::mem::take(&mut self.chi_interp);
            self.prepare_fit(None);
            if !chi.is_empty() {
                self.chi_interp = chi;
            }
        }

        // model chi = ff2chi(paths) on the model k-grid
        let grid = KGrid::Explicit(self.model_k.clone());
        let (_mk, model_chi) = ff2chi(&mut self.paths, &grid, self.interp);

        // diff = data - bkg(0) - model  (no refine_bkg support here)
        let mut diff: Vec<f64> = self.chi_interp.clone();
        if !data_only {
            for (d, m) in diff.iter_mut().zip(&model_chi) {
                *d -= *m;
            }
        }

        let trans = &self.transform;
        let rstep = trans.rstep();
        let nfft_half = trans.nfft as f64 / 2.0;

        match trans.fitspace {
            FitSpace::K => {
                let iqmin = itrunc(0.0f64.max(0.01 + trans.kmin / trans.kstep));
                let iqmax = itrunc(nfft_half.min(0.01 + trans.kmax / trans.kstep));
                let k = trans.k_grid();
                (iqmin..iqmax)
                    .map(|i| (diff[i] / self.epsilon_k) * k[i].powi(trans.kweight))
                    .collect()
            }
            FitSpace::R => {
                let chir = trans.fftf(&diff, trans.kweight);
                let irmin = itrunc(0.0f64.max(0.01 + trans.rmin / rstep));
                let irmax = itrunc(nfft_half.min(0.01 + trans.rmax / rstep));
                let scaled: Vec<num_complex::Complex64> = chir[irmin..irmax]
                    .iter()
                    .map(|c| c / self.epsilon_r)
                    .collect();
                realimag(&scaled)
            }
            FitSpace::Q => {
                let chir = trans.fftf(&diff, trans.kweight);
                let chiq = trans.fftr(&chir);
                let iqmin = itrunc(0.0f64.max(0.01 + trans.kmin / trans.kstep));
                let iqmax = itrunc(nfft_half.min(0.01 + trans.kmax / trans.kstep));
                // larch: realimag(chiq[iqmin:iqmax] / eps_r)[::2] -> the real parts
                chiq[iqmin..iqmax]
                    .iter()
                    .map(|c| c.re / self.epsilon_r)
                    .collect()
            }
        }
    }
}
