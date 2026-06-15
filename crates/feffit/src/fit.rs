//! End-to-end `feffit()`: wire the global [`Parameters`], the per-path
//! parameter expressions, the [`DataSet`] residual core, and the MINPACK
//! Levenberg-Marquardt minimiser ([`lm`]) into a single fit, then compute the
//! fit statistics. Port of `larch.xafs.feffit.feffit()`.
//!
//! Each path parameter (`degen`, `s02`, `e0`, `ei`, `deltar`, `sigma2`,
//! `third`, `fourth`) is either a constant or an expression over the global
//! parameters augmented with path-local symbols (`reff`, `degen`, `nleg`,
//! `rnorman`, `gam_ch`, `rs_int`, `vint`, `vmu`, `vfermi`). `rmass` (and the
//! `sigma2_debye`/`sigma2_eins` helpers that use it) are not yet ported.

use std::collections::HashMap;

use feffdat::{FeffDatFile, PathParams};
use lm::{lmdif, LmConfig};
use params::{parse, Expr, ExprError, ParamError, Parameters};

use crate::dataset::DataSet;

/// A path-parameter value: a fixed number or an expression string.
#[derive(Debug, Clone)]
pub enum Spec {
    /// A fixed numeric value.
    Const(f64),
    /// An expression over the global parameters and path-local symbols.
    Expr(String),
}

/// The eight path-parameter specs for one path (larch `PATH_PARS`).
#[derive(Debug, Clone)]
pub struct PathSpec {
    pub degen: Spec,
    pub s02: Spec,
    pub e0: Spec,
    pub ei: Spec,
    pub deltar: Spec,
    pub sigma2: Spec,
    pub third: Spec,
    pub fourth: Spec,
}

impl PathSpec {
    /// larch defaults: `degen` from the file, `s02 = 1`, the rest `0`.
    pub fn defaults(file_degen: f64) -> Self {
        PathSpec {
            degen: Spec::Const(file_degen),
            s02: Spec::Const(1.0),
            e0: Spec::Const(0.0),
            ei: Spec::Const(0.0),
            deltar: Spec::Const(0.0),
            sigma2: Spec::Const(0.0),
            third: Spec::Const(0.0),
            fourth: Spec::Const(0.0),
        }
    }
}

/// A dataset plus the parameter specs for each of its paths and the dataset's
/// explicit `epsilon_k` (or `None` to estimate it from high-R noise).
pub struct FitDataSet {
    pub dataset: DataSet,
    /// `specs[i]` describes `dataset.paths[i]`.
    pub specs: Vec<PathSpec>,
    pub epsilon_k: Option<f64>,
}

/// One fit variable's best-fit value and (rescaled) 1-sigma uncertainty.
#[derive(Debug, Clone)]
pub struct Best {
    pub name: String,
    pub value: f64,
    pub stderr: f64,
}

/// Result of [`feffit`]: best-fit variables, rescaled covariance, and the fit
/// statistics (all rescaled to `n_idp`, matching larch).
#[derive(Debug, Clone)]
pub struct FeffitResult {
    /// Best-fit value + stderr for each free variable, in `var_names` order.
    pub best: Vec<Best>,
    /// Covariance among the free variables, rescaled to `n_idp` (`None` if
    /// the Jacobian was singular). `covar[i][j]` matches `best[i]`/`best[j]`.
    pub covar: Option<Vec<Vec<f64>>>,
    pub nvarys: usize,
    pub nfree: usize,
    pub ndata: usize,
    pub n_idp: f64,
    pub nfev: i32,
    /// MINPACK termination code (1-4 indicate success).
    pub info: i32,
    pub chi_square: f64,
    pub chi2_reduced: f64,
    pub rfactor: f64,
    pub aic: f64,
    pub bic: f64,
}

/// Errors raised while setting up or running a [`feffit`] fit.
#[derive(Debug)]
pub enum FitError {
    /// A global constraint failed to resolve.
    Param(ParamError),
    /// A path-parameter expression failed to parse or evaluate.
    Expr(ExprError),
    /// Mismatched path/spec counts or other structural problem.
    Shape(String),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::Param(e) => write!(f, "{e}"),
            FitError::Expr(e) => write!(f, "path parameter expression: {e}"),
            FitError::Shape(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<ParamError> for FitError {
    fn from(e: ParamError) -> Self {
        FitError::Param(e)
    }
}
impl From<ExprError> for FitError {
    fn from(e: ExprError) -> Self {
        FitError::Expr(e)
    }
}

/// A path spec with its expressions pre-parsed (so the residual loop re-parses
/// nothing).
enum CompiledSpec {
    Const(f64),
    Expr(Expr),
}

struct CompiledPathSpec {
    degen: CompiledSpec,
    s02: CompiledSpec,
    e0: CompiledSpec,
    ei: CompiledSpec,
    deltar: CompiledSpec,
    sigma2: CompiledSpec,
    third: CompiledSpec,
    fourth: CompiledSpec,
}

fn compile(spec: &Spec) -> Result<CompiledSpec, ExprError> {
    Ok(match spec {
        Spec::Const(v) => CompiledSpec::Const(*v),
        Spec::Expr(s) => CompiledSpec::Expr(parse(s)?),
    })
}

impl CompiledPathSpec {
    fn from(spec: &PathSpec) -> Result<Self, ExprError> {
        Ok(CompiledPathSpec {
            degen: compile(&spec.degen)?,
            s02: compile(&spec.s02)?,
            e0: compile(&spec.e0)?,
            ei: compile(&spec.ei)?,
            deltar: compile(&spec.deltar)?,
            sigma2: compile(&spec.sigma2)?,
            third: compile(&spec.third)?,
            fourth: compile(&spec.fourth)?,
        })
    }

    /// Evaluate the eight specs into numeric [`PathParams`], using `base` (the
    /// global parameter values) augmented with this path's local symbols.
    fn eval(
        &self,
        base: &HashMap<String, f64>,
        fdat: &FeffDatFile,
    ) -> Result<PathParams, ExprError> {
        let mut sym = base.clone();
        // path-local symbols (larch FEFFDAT_VALUES, minus the deferred `rmass`)
        sym.insert("reff".to_string(), fdat.reff);
        sym.insert("degen".to_string(), fdat.degen);
        sym.insert("nleg".to_string(), fdat.nleg as f64);
        sym.insert("rnorman".to_string(), fdat.rnorman);
        sym.insert("gam_ch".to_string(), fdat.gam_ch);
        sym.insert("rs_int".to_string(), fdat.rs_int);
        sym.insert("vint".to_string(), fdat.vint);
        sym.insert("vmu".to_string(), fdat.vmu);
        sym.insert("vfermi".to_string(), fdat.vfermi);

        let ev = |c: &CompiledSpec| -> Result<f64, ExprError> {
            match c {
                CompiledSpec::Const(v) => Ok(*v),
                CompiledSpec::Expr(e) => e.eval(&sym),
            }
        };
        Ok(PathParams {
            degen: ev(&self.degen)?,
            s02: ev(&self.s02)?,
            e0: ev(&self.e0)?,
            ei: ev(&self.ei)?,
            deltar: ev(&self.deltar)?,
            sigma2: ev(&self.sigma2)?,
            third: ev(&self.third)?,
            fourth: ev(&self.fourth)?,
        })
    }
}

/// Push a trial variable vector through the global constraints and per-path
/// expressions, writing each path's numeric `PathParams`.
fn apply_params(
    params: &mut Parameters,
    datasets: &mut [FitDataSet],
    compiled: &[Vec<CompiledPathSpec>],
) -> Result<(), FitError> {
    params.update_constraints()?;
    let base = params.symbols();
    for (fds, cds) in datasets.iter_mut().zip(compiled) {
        for (path, cspec) in fds.dataset.paths.iter_mut().zip(cds) {
            path.params = cspec.eval(&base, &path.feffdat)?;
        }
    }
    Ok(())
}

/// Fit a sum of Feff paths to one or more datasets (port of larch `feffit()`).
///
/// `params` holds the free variables and any global constraint parameters. On
/// return, `params` and each path hold their best-fit values. The lmfit/larch
/// tolerances are used (`ftol = xtol = gtol = 1e-6`, `epsfcn = 1e-10`).
pub fn feffit(
    params: &mut Parameters,
    datasets: &mut [FitDataSet],
) -> Result<FeffitResult, FitError> {
    // prepare datasets and pre-parse the path-parameter expressions
    let mut compiled: Vec<Vec<CompiledPathSpec>> = Vec::with_capacity(datasets.len());
    for fds in datasets.iter_mut() {
        if fds.dataset.paths.len() != fds.specs.len() {
            return Err(FitError::Shape(format!(
                "dataset has {} paths but {} specs",
                fds.dataset.paths.len(),
                fds.specs.len()
            )));
        }
        fds.dataset.prepare_fit(fds.epsilon_k);
        let cspecs = fds
            .specs
            .iter()
            .map(CompiledPathSpec::from)
            .collect::<Result<Vec<_>, _>>()?;
        compiled.push(cspecs);
    }

    let var_names = params.var_names();
    let nvarys = var_names.len();

    // validate constraints/expressions once at the start point (so the inner
    // residual loop can treat resolution as infallible)
    params.update_constraints()?;
    let x0: Vec<f64> = var_names.iter().map(|n| params.value(n).unwrap()).collect();
    apply_params(params, datasets, &compiled)?;

    let cfg = LmConfig {
        ftol: 1.0e-6,
        xtol: 1.0e-6,
        gtol: 1.0e-6,
        maxfev: 4000 * (nvarys as i32 + 1),
        epsfcn: 1.0e-10,
        factor: 100.0,
    };

    // run the fit; the closure mutably borrows params + datasets only for the
    // duration of lmdif, which returns owned data.
    let result = {
        let params = &mut *params;
        let datasets = &mut *datasets;
        let compiled = &compiled;
        let fcn = |vars: &[f64]| -> Vec<f64> {
            params.set_var_values(vars);
            apply_params(params, datasets, compiled)
                .expect("constraint/expression resolution failed mid-fit");
            let mut out = Vec::new();
            for fds in datasets.iter_mut() {
                out.extend(fds.dataset.residual(false));
            }
            out
        };
        lmdif(fcn, &x0, &cfg)
    };

    // pin params + paths at the best-fit point
    params.set_var_values(&result.x);
    apply_params(params, datasets, &compiled)?;

    // ---- statistics (larch feffit(): rescaled to n_idp) ----
    let ndata = result.fvec.len();
    let nfree = ndata.saturating_sub(nvarys);
    let chisqr = result.fnorm * result.fnorm; // sum(residual^2)
    let n_idp: f64 = datasets.iter().map(|d| d.dataset.n_idp()).sum();

    let chi_square = chisqr * n_idp / ndata as f64;
    let chi2_reduced = chi_square / (n_idp - nvarys as f64);

    // r-factor: sum(model residual^2) / sum(data-only residual^2)
    let mut dat_ss = 0.0;
    for fds in datasets.iter_mut() {
        for v in fds.dataset.residual(true) {
            dat_ss += v * v;
        }
    }
    let rfactor = chisqr / dat_ss;

    let neg2_loglikel = n_idp * (chi_square / n_idp).ln();
    let aic = neg2_loglikel + 2.0 * nvarys as f64;
    let bic = neg2_loglikel + n_idp.ln() * nvarys as f64;

    // uncertainties: rescale the unscaled covariance by err_scale =
    // redchi * nfree / (n_idp - nvarys) = chisqr / (n_idp - nvarys), since
    // larch runs lmfit with scale_covar=False then rescales to n_idp.
    let err_scale = chisqr / (n_idp - nvarys as f64);
    let cov_unscaled = result.covar();
    let covar = cov_unscaled.as_ref().map(|c| {
        c.iter()
            .map(|row| row.iter().map(|v| v * err_scale).collect())
            .collect()
    });
    let best: Vec<Best> = var_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let stderr = cov_unscaled
                .as_ref()
                .map(|c| (c[i][i] * err_scale).sqrt())
                .unwrap_or(f64::NAN);
            Best {
                name: name.clone(),
                value: result.x[i],
                stderr,
            }
        })
        .collect();

    Ok(FeffitResult {
        best,
        covar,
        nvarys,
        nfree,
        ndata,
        n_idp,
        nfev: result.nfev,
        info: result.info,
        chi_square,
        chi2_reduced,
        rfactor,
        aic,
        bic,
    })
}
