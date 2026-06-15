//! End-to-end `feffit()`: wire the global [`Parameters`], the per-path
//! parameter expressions, the [`DataSet`] residual core, and the MINPACK
//! Levenberg-Marquardt minimiser ([`lm`]) into a single fit, then compute the
//! fit statistics. Port of `larch.xafs.feffit.feffit()`.
//!
//! Each path parameter (`degen`, `s02`, `e0`, `ei`, `deltar`, `sigma2`,
//! `third`, `fourth`) is either a constant or an expression over the global
//! parameters augmented with path-local symbols (`reff`, `nleg`, `degen`,
//! `rmass`, `rnorman`, `gam_ch`, `rs_int`, `vint`, `vmu`, `vfermi`) and the
//! path-bound σ² helpers `sigma2_eins(t, theta)` / `sigma2_debye(t, theta)`
//! (bound to the path geometry through a [`params::FuncCtx`]).

use std::collections::HashMap;

use feffdat::{FeffDatFile, PathParams, gnxas, sigma2_debye, sigma2_eins};
use lm::{LmConfig, lmdif};
use params::{Expr, ExprError, FuncCtx, ParamError, Parameters, parse};

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

/// The eight path parameters, in larch's canonical order (`PATH_PARS`).
pub const PATH_PNAMES: [&str; 8] = [
    "degen", "s02", "e0", "ei", "deltar", "sigma2", "third", "fourth",
];

/// One path parameter's best-fit value and propagated 1-sigma uncertainty.
#[derive(Debug, Clone)]
pub struct PathParam {
    /// Index of the dataset this path belongs to.
    pub dataset: usize,
    /// Index of the path within its dataset.
    pub path: usize,
    /// Parameter name (one of [`PATH_PNAMES`]).
    pub name: String,
    pub value: f64,
    /// Propagated uncertainty (`0` for a constant spec).
    pub stderr: f64,
}

/// Result of [`feffit`]: best-fit variables, rescaled covariance, and the fit
/// statistics (all rescaled to `n_idp`, matching larch).
#[derive(Debug, Clone)]
pub struct FeffitResult {
    /// Best-fit value + stderr for each free variable, in `var_names` order.
    pub best: Vec<Best>,
    /// Best-fit value + propagated stderr for each global constraint
    /// (expression) parameter, in declaration order.
    pub derived: Vec<Best>,
    /// Best-fit value + propagated stderr for every path parameter, in
    /// dataset → path → [`PATH_PNAMES`] order.
    pub path_params: Vec<PathParam>,
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
    /// global parameter values) augmented with this path's local symbols and
    /// the path-bound σ² helpers.
    fn eval(
        &self,
        base: &HashMap<String, f64>,
        fdat: &FeffDatFile,
    ) -> Result<PathParams, ExprError> {
        let sym = path_symbols(base, fdat);
        let ctx = PathFuncCtx { fdat };
        let ev = |c: &CompiledSpec| -> Result<f64, ExprError> {
            match c {
                CompiledSpec::Const(v) => Ok(*v),
                CompiledSpec::Expr(e) => e.eval_ctx(&sym, &ctx),
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

    /// Value and gradient (w.r.t. the `nvar`-variable basis) of each of the
    /// eight specs, in [`PATH_PNAMES`] order, for uncertainty propagation.
    /// `base`/`grads` are the global parameter values/gradients; path-local
    /// symbols are injected as constants (zero gradient), matching larch
    /// treating `reff` and the other `FEFFDAT_VALUES` as fixed.
    fn eval_dual(
        &self,
        base: &HashMap<String, f64>,
        grads: &HashMap<String, Vec<f64>>,
        nvar: usize,
        fdat: &FeffDatFile,
    ) -> Result<[(f64, Vec<f64>); 8], ExprError> {
        let sym = path_symbols(base, fdat);
        let ctx = PathFuncCtx { fdat };
        let ev = |c: &CompiledSpec| -> Result<(f64, Vec<f64>), ExprError> {
            match c {
                CompiledSpec::Const(v) => Ok((*v, vec![0.0; nvar])),
                CompiledSpec::Expr(e) => e.eval_dual_ctx(&sym, grads, nvar, &ctx),
            }
        };
        Ok([
            ev(&self.degen)?,
            ev(&self.s02)?,
            ev(&self.e0)?,
            ev(&self.ei)?,
            ev(&self.deltar)?,
            ev(&self.sigma2)?,
            ev(&self.third)?,
            ev(&self.fourth)?,
        ])
    }
}

/// `base` (global parameters) augmented with this path's local symbols — the
/// larch `FEFFDAT_VALUES`. These are constants for the fit (zero gradient).
fn path_symbols(base: &HashMap<String, f64>, fdat: &FeffDatFile) -> HashMap<String, f64> {
    let mut sym = base.clone();
    sym.insert("reff".to_string(), fdat.reff);
    sym.insert("nleg".to_string(), fdat.nleg as f64);
    sym.insert("degen".to_string(), fdat.degen);
    sym.insert("rmass".to_string(), fdat.rmass());
    sym.insert("rnorman".to_string(), fdat.rnorman);
    sym.insert("gam_ch".to_string(), fdat.gam_ch);
    sym.insert("rs_int".to_string(), fdat.rs_int);
    sym.insert("vint".to_string(), fdat.vint);
    sym.insert("vmu".to_string(), fdat.vmu);
    sym.insert("vfermi".to_string(), fdat.vfermi);
    sym
}

/// Binds the EXAFS σ² helpers to one path's geometry so a path expression can
/// call `sigma2_eins(t, theta)` / `sigma2_debye(t, theta)`, mirroring larch's
/// `add_sigma2funcs` (which closes the asteval-injected helpers over the
/// current path's `feffpath`).
struct PathFuncCtx<'a> {
    fdat: &'a FeffDatFile,
}

impl FuncCtx for PathFuncCtx<'_> {
    fn call(&self, name: &str, args: &[f64]) -> Option<Result<f64, ExprError>> {
        let arity2 = |f: &dyn Fn(f64, f64) -> f64| {
            if args.len() == 2 {
                Ok(f(args[0], args[1]))
            } else {
                Err(ExprError::Arity(name.to_string()))
            }
        };
        let arity3 = |f: &dyn Fn(f64, f64, f64) -> f64| {
            if args.len() == 3 {
                Ok(f(args[0], args[1], args[2]))
            } else {
                Err(ExprError::Arity(name.to_string()))
            }
        };
        match name {
            "sigma2_eins" => Some(arity2(&|t, th| sigma2_eins(t, th, &self.fdat.geom))),
            "sigma2_debye" => Some(arity2(&|t, th| {
                sigma2_debye(t, th, self.fdat.rnorman, &self.fdat.geom)
            })),
            "gnxas" => Some(arity3(&|r0, sigma, beta| {
                gnxas(r0, sigma, beta, self.fdat.reff)
            })),
            _ => None,
        }
    }
}

/// Push a trial variable vector through the global constraints and per-path
/// expressions, writing each path's numeric `PathParams`.
fn apply_params(
    params: &mut Parameters,
    datasets: &mut [FitDataSet],
    compiled: &[Vec<CompiledPathSpec>],
    bkg_names: &[Vec<String>],
) -> Result<(), FitError> {
    params.update_constraints()?;
    let base = params.symbols();
    for (di, (fds, cds)) in datasets.iter_mut().zip(compiled).enumerate() {
        for (path, cspec) in fds.dataset.paths.iter_mut().zip(cds) {
            path.params = cspec.eval(&base, &path.feffdat)?;
        }
        // feed the current background spline coefficients (the `bkg*` variables)
        // into the dataset so its residual subtracts the refined background.
        if !bkg_names[di].is_empty() {
            let coefs: Vec<f64> = bkg_names[di].iter().map(|n| base[n]).collect();
            fds.dataset.set_bkg_coefs(&coefs);
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

    // refine_bkg: each background-refining dataset contributes `nspline` free
    // variables (the spline coefficients), added after the user's variables so
    // the variable order matches larch (user variables, then bkg coefficients).
    // `bkg_names[di]` is empty for datasets that do not refine a background.
    let mut bkg_names: Vec<Vec<String>> = Vec::with_capacity(datasets.len());
    for (di, fds) in datasets.iter().enumerate() {
        if fds.dataset.refine_bkg() {
            let names: Vec<String> = (0..fds.dataset.bkg_nspline())
                .map(|i| format!("bkg{i:02}_ds{di}"))
                .collect();
            for name in &names {
                params.add_var(name, 0.0);
            }
            bkg_names.push(names);
        } else {
            bkg_names.push(Vec::new());
        }
    }

    let var_names = params.var_names();
    let nvarys = var_names.len();

    // validate constraints/expressions once at the start point (so the inner
    // residual loop can treat resolution as infallible)
    params.update_constraints()?;
    // The minimiser optimises in internal (unbounded) coordinates; for an
    // unbounded variable this equals its value, for a bounded one it is the
    // Minuit-style transform (lmfit `setup_bounds`). The residual closure maps
    // back to external values before evaluating, so lmdif stays a plain
    // unconstrained least-squares (matching lmfit/larch with bounds).
    let x0 = params.internal_x0();
    apply_params(params, datasets, &compiled, &bkg_names)?;

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
        let bkg_names = &bkg_names;
        let fcn = |vars: &[f64]| -> Vec<f64> {
            params.set_var_internal(vars);
            apply_params(params, datasets, compiled, bkg_names)
                .expect("constraint/expression resolution failed mid-fit");
            let mut out = Vec::new();
            for fds in datasets.iter_mut() {
                out.extend(fds.dataset.residual(false));
            }
            out
        };
        lmdif(fcn, &x0, &cfg)
    };

    // pin params + paths at the best-fit point (result.x is internal; map back)
    params.set_var_internal(&result.x);
    apply_params(params, datasets, &compiled, &bkg_names)?;

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
    // lmdif optimises in internal (unbounded) coordinates, so its covariance is
    // in internal space. Transform it to external (bounded) space by the MINUIT
    // gradient scaling `cov_ext[i][j] = cov_int[i][j] * g[i] * g[j]` (lmfit
    // `_int2ext_cov_x`) before the n_idp rescale. For unbounded variables every
    // `g` is 1, so this is the identity and the unbounded fit is unchanged.
    let grad = params.var_scale_gradients(&result.x);
    let cov_ext: Option<Vec<Vec<f64>>> = result.covar().as_ref().map(|c| {
        (0..nvarys)
            .map(|i| (0..nvarys).map(|j| c[i][j] * grad[i] * grad[j]).collect())
            .collect()
    });
    let covar: Option<Vec<Vec<f64>>> = cov_ext.as_ref().map(|c| {
        c.iter()
            .map(|row| row.iter().map(|v| v * err_scale).collect())
            .collect()
    });
    let best: Vec<Best> = var_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let stderr = cov_ext
                .as_ref()
                .map(|c| (c[i][i] * err_scale).sqrt())
                .unwrap_or(f64::NAN);
            Best {
                name: name.clone(),
                // external best-fit value (params hold it after set_var_internal)
                value: params.value(name).unwrap(),
                stderr,
            }
        })
        .collect();

    // ---- propagate uncertainties onto constraint + path parameters ----
    // First-order propagation `stderr(f) = sqrt(gᵀ C g)` against the rescaled
    // covariance, exactly larch's `correlated_values` + `eval_stderr`.
    let propagate = |g: &[f64]| -> f64 {
        match &covar {
            Some(c) => {
                let mut s = 0.0;
                for i in 0..nvarys {
                    for j in 0..nvarys {
                        s += g[i] * c[i][j] * g[j];
                    }
                }
                s.max(0.0).sqrt()
            }
            None => f64::NAN,
        }
    };

    let value_grads = params.value_grads()?;
    let grads: HashMap<String, Vec<f64>> = value_grads
        .iter()
        .map(|(k, (_, g))| (k.clone(), g.clone()))
        .collect();

    let derived: Vec<Best> = params
        .expr_names()
        .into_iter()
        .map(|name| {
            let (value, g) = &value_grads[&name];
            Best {
                name,
                value: *value,
                stderr: propagate(g),
            }
        })
        .collect();

    let base = params.symbols();
    let mut path_params = Vec::new();
    for (di, (fds, cds)) in datasets.iter().zip(&compiled).enumerate() {
        for (pi, (path, cspec)) in fds.dataset.paths.iter().zip(cds).enumerate() {
            let vgs = cspec.eval_dual(&base, &grads, nvarys, &path.feffdat)?;
            for (k, (value, g)) in vgs.iter().enumerate() {
                path_params.push(PathParam {
                    dataset: di,
                    path: pi,
                    name: PATH_PNAMES[k].to_string(),
                    value: *value,
                    stderr: propagate(g),
                });
            }
        }
    }

    Ok(FeffitResult {
        best,
        derived,
        path_params,
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

#[cfg(test)]
mod gnxas_wiring_tests {
    use super::*;
    use std::path::PathBuf;

    fn test_fdat() -> FeffDatFile {
        FeffDatFile::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/feff0001.dat"),
        )
        .unwrap()
    }

    /// The `gnxas` path helper is routed through `PathFuncCtx` with the path's
    /// own `reff` bound as the fourth argument (the three expression arguments
    /// are `r0`, `sigma`, `beta`).
    #[test]
    fn gnxas_routed_with_path_reff() {
        let fdat = test_fdat();
        let ctx = PathFuncCtx { fdat: &fdat };
        let got = ctx.call("gnxas", &[2.5, 0.05, 0.30]).unwrap().unwrap();
        let want = gnxas(2.5, 0.05, 0.30, fdat.reff);
        assert_eq!(got, want);
    }

    /// `gnxas` takes exactly three expression arguments; any other count is an
    /// arity error, not a silently wrong result.
    #[test]
    fn gnxas_wrong_arity_is_error() {
        let fdat = test_fdat();
        let ctx = PathFuncCtx { fdat: &fdat };
        assert!(matches!(
            ctx.call("gnxas", &[2.5, 0.05]),
            Some(Err(ExprError::Arity(_)))
        ));
        assert!(matches!(
            ctx.call("gnxas", &[2.5, 0.05, 0.30, 2.55]),
            Some(Err(ExprError::Arity(_)))
        ));
    }
}
