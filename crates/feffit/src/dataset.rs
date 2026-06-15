//! A feffit dataset, a port of `larch.xafs.feffit.FeffitDataSet`.
//!
//! Ties together experimental chi(k) (`data`), a list of Feff paths, and a
//! [`Transform`], and produces the fit residual that the minimiser drives to
//! zero. This covers the residual path for fixed (numeric) path parameters in
//! k/R/q and `'w'` (Cauchy-wavelet) space, for one or more k-weights (the
//! residual is the per-k-weight residuals concatenated, matching larch's
//! list-valued `kweight`).

use std::f64::consts::PI;

use feffdat::{ff2chi, interp_linear, FeffPath, Interp, KGrid};

use crate::bkg::{self, splev};
use crate::outputs::{xafsft, DataSetOutput};
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
    /// One uncertainty per k-weight in `transform.kweight` (larch's
    /// per-k-weight `epsilon_k`/`epsilon_r` lists; a single entry for the
    /// scalar-k-weight case).
    epsilon_k: Vec<f64>,
    epsilon_r: Vec<f64>,
    prepared: bool,

    /// Refine a cubic B-spline background as extra fit variables (larch
    /// `refine_bkg`). When set, `prepare_fit` mutates the transform and builds
    /// the knot vector below; the residual subtracts the spline.
    refine_bkg: bool,
    bkg_knots: Vec<f64>,
    bkg_nspline: usize,
    /// Current background spline coefficients (the `bkg00..bkgNN` fit variables);
    /// set by the fit loop before each residual evaluation.
    bkg_coefs: Vec<f64>,
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
            epsilon_k: Vec::new(),
            epsilon_r: Vec::new(),
            prepared: false,
            refine_bkg: false,
            bkg_knots: Vec::new(),
            bkg_nspline: 0,
            bkg_coefs: Vec::new(),
        }
    }

    /// Enable background refinement for this dataset (larch `refine_bkg=True`).
    /// The knot vector and the transform/n_idp adjustments are applied in
    /// [`DataSet::prepare_fit`]; the `bkg00..bkgNN` coefficients are supplied by
    /// the fit loop via [`DataSet::set_bkg_coefs`].
    pub fn enable_refine_bkg(&mut self) {
        self.refine_bkg = true;
    }

    /// Whether background refinement is enabled.
    pub fn refine_bkg(&self) -> bool {
        self.refine_bkg
    }

    /// Number of background spline coefficients (`nspline`), valid after
    /// [`DataSet::prepare_fit`]; `0` when background refinement is off.
    pub fn bkg_nspline(&self) -> usize {
        self.bkg_nspline
    }

    /// The background spline knot vector, valid after [`DataSet::prepare_fit`].
    pub fn bkg_knots(&self) -> &[f64] {
        &self.bkg_knots
    }

    /// Set the current background spline coefficients (the `bkg00..bkgNN` fit
    /// variables) before a residual evaluation.
    pub fn set_bkg_coefs(&mut self, coefs: &[f64]) {
        self.bkg_coefs.clear();
        self.bkg_coefs.extend_from_slice(coefs);
    }

    /// Number of independent points: `1 + 2*(rmax-rmin)*(kmax-kmin)/pi`.
    pub fn n_idp(&self) -> f64 {
        self.n_idp
    }
    /// Uncertainty in chi(k), one entry per k-weight in `transform.kweight`.
    pub fn epsilon_k(&self) -> &[f64] {
        &self.epsilon_k
    }
    /// Uncertainty in chi(R), one entry per k-weight in `transform.kweight`.
    pub fn epsilon_r(&self) -> &[f64] {
        &self.epsilon_r
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

        // refine_bkg: mutate the transform (rbkg/rmin), overwrite n_idp with the
        // background-refinement formula, and build the spline knot vector
        // (larch `prepare_fit`). Done after epsilon (which is independent of
        // rmin/rbkg), matching larch's ordering.
        if self.refine_bkg {
            self.transform.enable_refine_bkg();
            let (rmax, kmin, kmax, rbkg) = (
                self.transform.rmax,
                self.transform.kmin,
                self.transform.kmax,
                self.transform.rbkg,
            );
            self.n_idp = 1.0 + 2.0 * rmax * (kmax - kmin) / PI;
            let ns = bkg::nspline(rbkg, kmin, kmax);
            self.bkg_knots = bkg::bkg_knots(kmin, kmax, ns);
            self.bkg_nspline = ns;
            self.bkg_coefs = vec![0.0; ns];
        }

        self.prepared = true;
    }

    /// Set epsilon_k / epsilon_r from an explicit scalar (port of
    /// `set_epsilon_k`). The same `eps_k` is used for every k-weight; `eps_r`
    /// differs per k-weight via the Parseval scale (which depends on `2*kw+1`),
    /// matching larch's list-valued branch (a single-element list for a scalar
    /// k-weight).
    pub fn set_epsilon_k(&mut self, eps_k: f64) {
        let kstep = self.transform.kstep;
        let kmin = self.transform.kmin;
        let kmax = self.transform.kmax;
        let kweights = self.transform.kweight.clone();

        let mut ek = Vec::with_capacity(kweights.len());
        let mut er = Vec::with_capacity(kweights.len());
        for &kw in &kweights {
            let w = 2 * kw + 1;
            let denom = kstep * (kmax.powi(w) - kmin.powi(w));
            let scale = 2.0 * (PI * w as f64 / denom).sqrt();
            ek.push(eps_k);
            er.push(eps_k / scale);
        }
        self.epsilon_k = ek;
        self.epsilon_r = er;
    }

    /// Estimate epsilon_k / epsilon_r from high-R noise (port of
    /// `estimate_noise`). One `eps_r` is estimated per k-weight (the high-R
    /// region of `fftf(chi, kw)`), and converted to `eps_k` by the Parseval
    /// scale, matching larch's `all_kweights` branch.
    pub fn estimate_noise(&mut self, chi: &[f64], rmin: f64, rmax: f64) {
        let rstep = self.transform.rstep();
        let nfft = self.transform.nfft;
        let kstep = self.transform.kstep;
        let kmin = self.transform.kmin;
        let kmax = self.transform.kmax;
        let kweights = self.transform.kweight.clone();

        let irmin = itrunc(0.01 + rmin / rstep);
        let irmax = itrunc((nfft as f64 / 2.0).min(1.01 + rmax / rstep));

        // kwin_ave: mean window value scaled into the (kmax-kmin) range. The
        // window is k-weight-independent, so this is shared across k-weights.
        let kwin_sum: f64 = self.transform.kwin().iter().sum();
        let kwin_ave = kwin_sum * kstep / (kmax - kmin);

        let mut ek = Vec::with_capacity(kweights.len());
        let mut er = Vec::with_capacity(kweights.len());
        for &kw in &kweights {
            let chir = self.transform.fftf(chi, kw);
            let highr = realimag(&chir[irmin..irmax]);
            let ss: f64 = highr.iter().map(|v| v * v).sum();
            let eps_r = (ss / highr.len() as f64).sqrt() / kwin_ave;

            // Parseval scaling r -> k (note: a different convention than set_epsilon_k)
            let w = 2 * kw + 1;
            let denom = kstep * (kmax.powi(w) - kmin.powi(w));
            let scale = (2.0 * PI * w as f64 / denom).sqrt();
            ek.push(scale * eps_r);
            er.push(eps_r);
        }
        self.epsilon_k = ek;
        self.epsilon_r = er;
    }

    /// Forward/back-FT the data and model (and optionally each path) χ(k) into
    /// output arrays (port of `save_outputs`). The data transform uses the
    /// original `data_chi`; the model transform uses the path sum at the current
    /// parameters (`model_chi_sum`), which also populates each path's χ(k) so the
    /// per-path outputs come from the same evaluation. Call after a fit (or it
    /// lazily prepares with estimated noise, like `residual`).
    pub fn save_outputs(&mut self, rmax_out: f64, path_outputs: bool) -> DataSetOutput {
        if !self.prepared {
            self.prepare_fit(None);
        }
        let data = xafsft(&self.transform, &self.data_chi, rmax_out);
        let model_chi = self.model_chi_sum();
        let model = xafsft(&self.transform, &model_chi, rmax_out);
        let paths = if path_outputs {
            self.paths
                .iter()
                .map(|p| xafsft(&self.transform, &p.chi, rmax_out))
                .collect()
        } else {
            Vec::new()
        };
        DataSetOutput { data, model, paths }
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

        // diff = data - bkg - model. The refined background (a cubic B-spline on
        // the model k-grid) is subtracted on both the data-only and full paths,
        // matching larch (`diff = _chi - _bkg; if not data_only: diff -= model`).
        let mut diff: Vec<f64> = self.chi_interp.clone();
        if self.refine_bkg {
            let bkg = splev(&self.bkg_knots, &self.bkg_coefs, 3, &self.model_k);
            for (d, b) in diff.iter_mut().zip(&bkg) {
                *d -= *b;
            }
        }
        if !data_only {
            for (d, m) in diff.iter_mut().zip(&model_chi) {
                *d -= *m;
            }
        }

        let trans = &self.transform;
        let rstep = trans.rstep();
        let nfft_half = trans.nfft as f64 / 2.0;

        // For >1 k-weight the residual is the per-k-weight blocks concatenated,
        // in `transform.kweight` order, exactly larch's `all_kweights` branch.
        let mut out = Vec::new();
        match trans.fitspace {
            FitSpace::K => {
                let iqmin = itrunc(0.0f64.max(0.01 + trans.kmin / trans.kstep));
                let iqmax = itrunc(nfft_half.min(0.01 + trans.kmax / trans.kstep));
                let k = trans.k_grid();
                for (i, &kw) in trans.kweight.iter().enumerate() {
                    let eps = self.epsilon_k[i];
                    for j in iqmin..iqmax {
                        out.push((diff[j] / eps) * k[j].powi(kw));
                    }
                }
            }
            FitSpace::R => {
                let irmin = itrunc(0.0f64.max(0.01 + trans.rmin / rstep));
                let irmax = itrunc(nfft_half.min(0.01 + trans.rmax / rstep));
                for (i, &kw) in trans.kweight.iter().enumerate() {
                    let chir = trans.fftf(&diff, kw);
                    let eps = self.epsilon_r[i];
                    for c in &chir[irmin..irmax] {
                        out.push(c.re / eps);
                        out.push(c.im / eps);
                    }
                }
            }
            FitSpace::Q => {
                let iqmin = itrunc(0.0f64.max(0.01 + trans.kmin / trans.kstep));
                let iqmax = itrunc(nfft_half.min(0.01 + trans.kmax / trans.kstep));
                for (i, &kw) in trans.kweight.iter().enumerate() {
                    let chir = trans.fftf(&diff, kw);
                    let chiq = trans.fftr(&chir);
                    let eps = self.epsilon_r[i];
                    // larch: realimag(chiq[iqmin:iqmax] / eps_r)[::2] -> the real parts
                    for c in &chiq[iqmin..iqmax] {
                        out.push(c.re / eps);
                    }
                }
            }
            FitSpace::W => {
                // larch applies eps before the wavelet (`cwt(diff/eps_k, kw)`),
                // then `realimag(cwt).ravel()`. For a 2-D array larch's
                // `realimag` emits, per R row, all real parts (over k) followed
                // by all imag parts — NOT the interleaved order it uses in 1-D.
                // (larch's `'w'` branch only supports a scalar `epsilon_k`; this
                // per-k-weight indexing matches it for one k-weight and stays
                // consistent with k/R/q for more.)
                for (i, &kw) in trans.kweight.iter().enumerate() {
                    let eps = self.epsilon_k[i];
                    let scaled: Vec<f64> = diff.iter().map(|d| d / eps).collect();
                    let (wav, ncols) = trans.cwt(&scaled, kw);
                    for row in wav.chunks(ncols) {
                        out.extend(row.iter().map(|c| c.re));
                        out.extend(row.iter().map(|c| c.im));
                    }
                }
            }
        }
        out
    }
}
