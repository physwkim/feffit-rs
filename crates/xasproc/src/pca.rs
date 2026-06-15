//! Principal Component Analysis of a set of XAS spectra — port of
//! `larch.math.pca.pca_train` and `pca_fit` (the default numpy/lmfit path, not
//! the optional scikit-learn `pca_train_sklearn`).
//!
//! [`pca_train`] takes a matrix of `narr` training spectra (each already sampled
//! on a common `nfreq`-point grid — `groups2matrix`'s cubic interpolation onto a
//! shared grid is the caller's job) and returns the principal components, their
//! eigenvalues/variances, the Malinowski IND statistic, and the suggested
//! significant-component count `nsig`. [`pca_fit`] fits one unknown spectrum
//! against a trained model, optionally re-scaling the unknown.
//!
//! Parity is *not* bit-exact: the eigendecomposition (`np.linalg.eigh`) and the
//! least-squares solves (`np.linalg.lstsq`) use `nalgebra` (`SymmetricEigen` /
//! SVD) in place of LAPACK, so results agree to ~1e-11 rather than round-off.
//! Two structural points the test must respect:
//!   * eigenvector **signs** are arbitrary (LAPACK vs nalgebra may differ), so
//!     `components` matches larch only up to a per-component sign. Everything
//!     eigenvalue-derived (`eigenvalues`, `variances`, `ind`, `nsig`) and the
//!     `mean` are sign-independent, as is `pca_fit`'s `yfit`/`chi_square`/`scale`
//!     (a component sign flip flips its weight, leaving the product invariant).
//!   * `pca_fit(rescale=true)` fits a `scale` parameter bounded `min=0`; lmfit's
//!     MINUIT lower-bound transform is reproduced exactly (see [`from_internal`]).

use lm::{LmConfig, lmdif};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// A trained PCA model — the output of [`pca_train`], consumed by [`pca_fit`].
#[derive(Debug, Clone)]
pub struct PcaModel {
    /// the training matrix actually decomposed, `narr` rows × `nfreq` columns.
    pub ydat: Vec<Vec<f64>>,
    /// per-frequency mean over the training spectra (`nfreq`).
    pub mean: Vec<f64>,
    /// principal components in descending-eigenvalue order, `narr` × `nfreq`.
    /// Each row is defined only up to an overall sign.
    pub components: Vec<Vec<f64>>,
    /// eigenvalues in descending order (`narr`).
    pub eigenvalues: Vec<f64>,
    /// `eigenvalues / eigenvalues.sum()` (`narr`).
    pub variances: Vec<f64>,
    /// Malinowski IND statistic (`narr`; larch duplicates the first value).
    pub ind: Vec<f64>,
    /// suggested number of significant components, `argmin(ind)`.
    pub nsig: usize,
}

/// The result of fitting one unknown spectrum with [`pca_fit`].
#[derive(Debug, Clone)]
pub struct PcaFit {
    /// component weights from the final least-squares solve (`ncomps`).
    pub weights: Vec<f64>,
    /// the fitted spectrum `components.T @ weights + mean` (`nfreq`).
    pub yfit: Vec<f64>,
    /// the unknown after re-scaling (`nfreq`).
    pub ydat: Vec<f64>,
    /// `|| components.T @ weights - (ydat - mean) ||^2`.
    pub chi_square: f64,
    /// the fitted `scale` (1.0 when `rescale = false`).
    pub data_scale: f64,
}

/// lmfit's MINUIT lower-bound (`min=0`, `max=inf`) inverse transform: maps the
/// unbounded internal variable the minimiser sees to the external `scale`.
fn from_internal(val: f64) -> f64 {
    // self.min - 1 + sqrt(val*val + 1), with self.min = 0
    -1.0 + (val * val + 1.0).sqrt()
}

/// lmfit's forward transform: external `scale` -> internal seed.
fn to_internal(scale: f64) -> f64 {
    // sqrt((self._val - self.min + 1)^2 - 1), with self.min = 0
    ((scale + 1.0).powi(2) - 1.0).sqrt()
}

/// `np.linalg.lstsq(a, b)` via SVD (`nalgebra` in place of LAPACK `gelsd`).
/// Returns the minimum-norm solution and `||a@x - b||^2` (numpy's `residuals`).
fn lstsq(a: &DMatrix<f64>, b: &DVector<f64>) -> (DVector<f64>, f64) {
    let svd = a.clone().svd(true, true);
    let x = svd.solve(b, 1.0e-14).expect("lstsq SVD solve failed");
    let resid = a * &x - b;
    (x, resid.norm_squared())
}

/// `larch.math.pca.pca_train` (default numpy path).
///
/// `ydat[i]` is training spectrum `i`, all `narr` of them sampled on the same
/// `nfreq`-point grid. Requires `narr >= 2` and every row the same length.
pub fn pca_train(ydat: &[Vec<f64>]) -> PcaModel {
    let narr = ydat.len();
    assert!(narr >= 2, "need at least 2 training spectra");
    let nfreq = ydat[0].len();
    for (i, row) in ydat.iter().enumerate() {
        assert_eq!(row.len(), nfreq, "spectrum {i} length mismatch");
    }
    let nf = nfreq as f64;
    let na = narr as f64;

    // ymean over spectra (axis=0); a = ydat - ymean
    let mut mean = vec![0.0; nfreq];
    for row in ydat {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v;
        }
    }
    for m in &mut mean {
        *m /= na;
    }
    let a: Vec<Vec<f64>> = ydat
        .iter()
        .map(|row| row.iter().zip(&mean).map(|(&v, &m)| v - m).collect())
        .collect();

    // standardize each spectrum (row of `a`) to zero mean / unit population std,
    // then transpose: z has shape (nfreq, narr), z[:, i] is spectrum i.
    // larch: ynorm = (ynorm.T - ynorm.mean(axis=1)) / ynorm.std(axis=1)
    let mut z = DMatrix::<f64>::zeros(nfreq, narr);
    for (i, row) in a.iter().enumerate() {
        let rmean = row.iter().sum::<f64>() / nf;
        let var = row.iter().map(|&v| (v - rmean).powi(2)).sum::<f64>() / nf;
        let rstd = var.sqrt();
        for (j, &v) in row.iter().enumerate() {
            z[(j, i)] = (v - rmean) / rstd;
        }
    }

    // cov = z.T @ z / narr  (narr x narr), then eigh (ascending eigenvalues)
    let cov = (z.transpose() * &z) / na;
    let (eval_asc, evec_asc) = eigh_ascending(&cov);

    // eigvec = (z @ -evec_ / narr).T  -> row m corresponds to ascending eval m
    let m = (&z * (-&evec_asc)) / na; // (nfreq, narr)
    // reverse to descending eigenvalue order
    let mut eigenvalues = vec![0.0; narr];
    let mut components = vec![vec![0.0; nfreq]; narr];
    for k in 0..narr {
        let src = narr - 1 - k;
        eigenvalues[k] = eval_asc[src];
        for j in 0..nfreq {
            components[k][j] = m[(j, src)];
        }
    }

    let esum: f64 = eigenvalues.iter().sum();
    let variances: Vec<f64> = eigenvalues.iter().map(|&e| e / esum).collect();

    // Malinowski IND statistic. larch's loop seeds `ind=[indval]` on the first
    // iteration and then unconditionally appends, so ind[0] is duplicated and
    // len(ind) == narr.
    let mut ind: Vec<f64> = Vec::new();
    for r in 0..narr - 1 {
        let nr = (narr - r - 1) as f64;
        let suffix: f64 = eigenvalues[r..].iter().sum();
        let indval = (nf * suffix / nr).sqrt() / nr.powi(2);
        if ind.is_empty() {
            ind.push(indval);
        }
        ind.push(indval);
    }

    // nsig = argmin(ind), first occurrence (matches np.argmin)
    let mut nsig = 0;
    let mut best = ind[0];
    for (i, &v) in ind.iter().enumerate().skip(1) {
        if v < best {
            best = v;
            nsig = i;
        }
    }

    PcaModel {
        ydat: ydat.to_vec(),
        mean,
        components,
        eigenvalues,
        variances,
        ind,
        nsig,
    }
}

/// `np.linalg.eigh` of a symmetric matrix: eigenvalues ascending, eigenvectors
/// as columns (`nalgebra` `SymmetricEigen` + an explicit ascending sort, since
/// nalgebra does not guarantee the order LAPACK's `syevd` returns).
fn eigh_ascending(cov: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let n = cov.nrows();
    let se = SymmetricEigen::new(cov.clone());
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| se.eigenvalues[i].partial_cmp(&se.eigenvalues[j]).unwrap());
    let evals: Vec<f64> = idx.iter().map(|&i| se.eigenvalues[i]).collect();
    let mut evecs = DMatrix::<f64>::zeros(n, n);
    for (col, &src) in idx.iter().enumerate() {
        evecs.set_column(col, &se.eigenvectors.column(src));
    }
    (evals, evecs)
}

/// `larch.math.pca.pca_fit`.
///
/// Fits `unknown` (already on `model`'s grid) against the first `ncomps`
/// components. With `rescale`, a non-negative `scale` on the unknown is fitted
/// by least-squares (lmfit `leastsq`, all tolerances `1e-5`) before the final
/// component weights are solved.
pub fn pca_fit(unknown: &[f64], model: &PcaModel, ncomps: usize, rescale: bool) -> PcaFit {
    let nfreq = model.mean.len();
    assert_eq!(unknown.len(), nfreq, "unknown length mismatch");
    let ncomps = ncomps.min(model.components.len());

    // comps = components[:ncomps].T  -> (nfreq, ncomps)
    let comps = DMatrix::from_fn(nfreq, ncomps, |r, c| model.components[c][r]);
    let mean = DVector::from_row_slice(&model.mean);
    let unk = DVector::from_row_slice(unknown);

    let scale = if rescale {
        // minimize over the bounded `scale`; the minimiser works on the internal
        // (unbounded) variable, the residual converts internal -> external.
        let resid = |p: &[f64]| -> Vec<f64> {
            let sc = from_internal(p[0]);
            let b = &unk * sc - &mean;
            let (w, _) = lstsq(&comps, &b);
            let yfit = &comps * &w + &mean;
            (&unk * sc - yfit).iter().copied().collect()
        };
        let cfg = LmConfig {
            ftol: 1.0e-5,
            xtol: 1.0e-5,
            gtol: 1.0e-5,
            // lmfit leastsq default max_nfev = 2000*(nvarys+1); nvarys = 1.
            maxfev: 2000 * 2,
            epsfcn: 1.0e-5,
            factor: 100.0,
        };
        let seed = [to_internal(1.0)];
        let result = lmdif(resid, &seed, &cfg);
        from_internal(result.x[0])
    } else {
        1.0
    };

    let ydat_scaled = &unk * scale;
    let b = &ydat_scaled - &mean;
    let (w, chi_square) = lstsq(&comps, &b);
    let yfit = &comps * &w + &mean;

    PcaFit {
        weights: w.iter().copied().collect(),
        yfit: yfit.iter().copied().collect(),
        ydat: ydat_scaled.iter().copied().collect(),
        chi_square,
        data_scale: scale,
    }
}
