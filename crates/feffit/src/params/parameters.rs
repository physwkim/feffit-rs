//! An `lmfit.Parameters`-style collection: named parameters that are either a
//! free variable (`vary`), a fixed value, or a constraint expression evaluated
//! against the other parameters. Mirrors how `lmfit` resolves constraints with
//! `asteval`, including dependency ordering so chained expressions
//! (`b = a*2`, `c = b+1`) evaluate correctly.

use std::collections::HashMap;

use crate::params::expr::{Expr, ExprError, parse};

/// A single parameter.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    /// Current value (updated by [`Parameters::update_constraints`] for exprs).
    pub value: f64,
    /// Free fit variable when `true` and `expr` is `None`.
    pub vary: bool,
    pub min: f64,
    pub max: f64,
    /// Constraint expression; when set, `value` is derived, not fit.
    pub expr: Option<String>,
}

/// lmfit's snap-to-zero threshold for the internal coordinate (`tiny` in
/// `setup_bounds`): an internal value with `|v| < TINY` is set to exactly 0.
const TINY: f64 = 1.0e-15;

impl Param {
    fn var(name: &str, value: f64) -> Param {
        Param {
            name: name.to_string(),
            value,
            vary: true,
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            expr: None,
        }
    }

    /// External (user) value → internal (unbounded) coordinate the minimiser
    /// optimises over (lmfit `Parameter.setup_bounds`). The value is first
    /// clamped to `[min, max]` (lmfit clamps on assignment). The four branches
    /// — both/upper-only/lower-only/no bound — match lmfit exactly.
    fn to_internal(&self) -> f64 {
        let (min, max) = (self.min, self.max);
        let v = self.value.clamp(min, max);
        let internal = if min == f64::NEG_INFINITY && max == f64::INFINITY {
            v
        } else if max == f64::INFINITY {
            ((v - min + 1.0).powi(2) - 1.0).sqrt()
        } else if min == f64::NEG_INFINITY {
            ((max - v + 1.0).powi(2) - 1.0).sqrt()
        } else {
            (2.0 * (v - min) / (max - min) - 1.0).asin()
        };
        if internal.abs() < TINY { 0.0 } else { internal }
    }

    /// Internal coordinate → external (bounded) value (lmfit
    /// `Parameter.from_internal`; named `to_external` here as the inverse of
    /// [`Param::to_internal`], since a `from_*` method taking `&self` reads as
    /// a constructor).
    fn to_external(&self, val: f64) -> f64 {
        let (min, max) = (self.min, self.max);
        if min == f64::NEG_INFINITY && max == f64::INFINITY {
            val
        } else if max == f64::INFINITY {
            min - 1.0 + (val * val + 1.0).sqrt()
        } else if min == f64::NEG_INFINITY {
            max + 1.0 - (val * val + 1.0).sqrt()
        } else {
            min + (val.sin() + 1.0) * (max - min) / 2.0
        }
    }

    /// d(external)/d(internal) at internal value `val` — the MINUIT gradient
    /// scaling lmfit applies to transform the covariance to external space
    /// (lmfit `Parameter.scale_gradient`). `1.0` for an unbounded variable.
    fn scale_gradient(&self, val: f64) -> f64 {
        let (min, max) = (self.min, self.max);
        if min == f64::NEG_INFINITY && max == f64::INFINITY {
            1.0
        } else if max == f64::INFINITY {
            val / (val * val + 1.0).sqrt()
        } else if min == f64::NEG_INFINITY {
            -val / (val * val + 1.0).sqrt()
        } else {
            val.cos() * (max - min) / 2.0
        }
    }
}

/// Errors specific to resolving a [`Parameters`] set.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// A constraint expression failed to parse or evaluate.
    Expr(String, ExprError),
    /// The constraint dependency graph contains a cycle.
    Cycle(Vec<String>),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::Expr(n, e) => write!(f, "parameter '{n}': {e}"),
            ParamError::Cycle(ns) => write!(f, "constraint cycle among: {}", ns.join(", ")),
        }
    }
}

impl std::error::Error for ParamError {}

/// An ordered collection of parameters plus injected constants (e.g. `reff`,
/// `pi`). Insertion order is preserved, matching lmfit's `Parameters`.
#[derive(Debug, Clone, Default)]
pub struct Parameters {
    order: Vec<String>,
    map: HashMap<String, Param>,
    /// Extra symbols available to expressions but not themselves parameters
    /// (the asteval symbol-table injections, e.g. per-path `reff`).
    consts: HashMap<String, f64>,
}

impl Parameters {
    /// An empty set.
    pub fn new() -> Self {
        Parameters::default()
    }

    /// Add (or replace) a free variable.
    pub fn add_var(&mut self, name: &str, value: f64) {
        self.insert(Param::var(name, value));
    }

    /// Add (or replace) a free variable with bounds.
    pub fn add_var_bounded(&mut self, name: &str, value: f64, min: f64, max: f64) {
        let mut p = Param::var(name, value);
        p.min = min;
        p.max = max;
        self.insert(p);
    }

    /// Add (or replace) a fixed parameter (`vary = false`, no expression).
    pub fn add_fixed(&mut self, name: &str, value: f64) {
        let mut p = Param::var(name, value);
        p.vary = false;
        self.insert(p);
    }

    /// Add (or replace) a constraint parameter with an expression.
    pub fn add_expr(&mut self, name: &str, expr: &str) {
        let mut p = Param::var(name, f64::NAN);
        p.vary = false;
        p.expr = Some(expr.to_string());
        self.insert(p);
    }

    /// Inject a constant symbol available to expressions (not a parameter).
    pub fn set_const(&mut self, name: &str, value: f64) {
        self.consts.insert(name.to_string(), value);
    }

    fn insert(&mut self, p: Param) {
        if !self.map.contains_key(&p.name) {
            self.order.push(p.name.clone());
        }
        self.map.insert(p.name.clone(), p);
    }

    /// Number of free (varying, non-expression) parameters.
    pub fn n_vary(&self) -> usize {
        self.order
            .iter()
            .filter(|n| {
                let p = &self.map[*n];
                p.vary && p.expr.is_none()
            })
            .count()
    }

    /// Names of the free variables, in insertion order.
    pub fn var_names(&self) -> Vec<String> {
        self.order
            .iter()
            .filter(|n| {
                let p = &self.map[*n];
                p.vary && p.expr.is_none()
            })
            .cloned()
            .collect()
    }

    /// Names of the constraint (expression) parameters, in insertion order.
    pub fn expr_names(&self) -> Vec<String> {
        self.order
            .iter()
            .filter(|n| self.map[*n].expr.is_some())
            .cloned()
            .collect()
    }

    /// Current value of a parameter.
    pub fn value(&self, name: &str) -> Option<f64> {
        self.map.get(name).map(|p| p.value)
    }

    /// Immutable access to a parameter.
    pub fn get(&self, name: &str) -> Option<&Param> {
        self.map.get(name)
    }

    /// The full symbol table an expression sees: every parameter's current
    /// value by name, plus the injected constants. Call after
    /// [`Parameters::update_constraints`] so expression parameters are current.
    /// Used by feffit to evaluate per-path parameter expressions against the
    /// global parameters (augmented with path-local symbols like `reff`).
    pub fn symbols(&self) -> HashMap<String, f64> {
        let mut m = self.consts.clone();
        for n in &self.order {
            m.insert(n.clone(), self.map[n].value);
        }
        m
    }

    /// Set the values of the free variables (in `var_names` order). Used by the
    /// minimiser to push a trial point before re-resolving constraints.
    pub fn set_var_values(&mut self, vals: &[f64]) {
        let names = self.var_names();
        for (n, &v) in names.iter().zip(vals) {
            if let Some(p) = self.map.get_mut(n) {
                p.value = v.clamp(p.min, p.max);
            }
        }
    }

    /// Internal (unbounded) coordinates of the free variables, in `var_names`
    /// order — the vector the minimiser starts from. For an unbounded variable
    /// this is just its value; for a bounded one it is the Minuit-style
    /// transform of the value (lmfit `setup_bounds`). Pair with
    /// [`Parameters::set_var_internal`] so the minimiser optimises in
    /// unbounded space while the residual sees bounded values.
    pub fn internal_x0(&self) -> Vec<f64> {
        self.var_names()
            .iter()
            .map(|n| self.map[n].to_internal())
            .collect()
    }

    /// Set the free variables from internal (unbounded) coordinates (the
    /// minimiser's working vector): each external value is `to_external` of
    /// the corresponding internal coordinate. No clamping is needed — the
    /// transform keeps a bounded value within `[min, max]` by construction.
    pub fn set_var_internal(&mut self, internal: &[f64]) {
        let names = self.var_names();
        for (n, &v) in names.iter().zip(internal) {
            if let Some(p) = self.map.get_mut(n) {
                p.value = p.to_external(v);
            }
        }
    }

    /// d(external)/d(internal) for each free variable at the given internal
    /// coordinates (`var_names` order) — the MINUIT gradient scaling that
    /// transforms the covariance from the minimiser's internal space to
    /// external (bounded) space: `cov_ext[i][j] = cov_int[i][j] * g[i] * g[j]`.
    /// Every entry is `1.0` when no variable is bounded.
    pub fn var_scale_gradients(&self, internal: &[f64]) -> Vec<f64> {
        self.var_names()
            .iter()
            .zip(internal)
            .map(|(n, &v)| self.map[n].scale_gradient(v))
            .collect()
    }

    /// Resolve all constraint expressions in dependency order, writing each
    /// expression parameter's `value`. Free/fixed parameters are clamped to
    /// their bounds (matching lmfit applying bounds to varying parameters).
    pub fn update_constraints(&mut self) -> Result<(), ParamError> {
        // clamp free vars to bounds first (expr params ignore bounds, like lmfit)
        for n in self.order.clone() {
            let p = self.map.get_mut(&n).unwrap();
            if p.expr.is_none() && p.vary {
                p.value = p.value.clamp(p.min, p.max);
            }
        }

        // parse expressions and build the dependency edges among parameters
        let mut asts: HashMap<String, Expr> = HashMap::new();
        for n in &self.order {
            if let Some(src) = self.map[n].expr.clone() {
                let ast = parse(&src).map_err(|e| ParamError::Expr(n.clone(), e))?;
                asts.insert(n.clone(), ast);
            }
        }

        let order = self.topo_order(&asts)?;

        // build the symbol table: all non-expr params + consts, then fill exprs
        let mut sym: HashMap<String, f64> = self.consts.clone();
        for n in &self.order {
            let p = &self.map[n];
            if p.expr.is_none() {
                sym.insert(n.clone(), p.value);
            }
        }
        for n in &order {
            let val = asts[n]
                .eval(&sym)
                .map_err(|e| ParamError::Expr(n.clone(), e))?;
            sym.insert(n.clone(), val);
            self.map.get_mut(n).unwrap().value = val;
        }
        Ok(())
    }

    /// For each parameter, its current value and gradient with respect to the
    /// free-variable basis ([`Parameters::var_names`], in that order). Free
    /// variables map to unit basis vectors; fixed parameters and injected
    /// constants to zero; expression parameters are differentiated through
    /// their constraint expressions by forward-mode AD in dependency order.
    ///
    /// Combined with the fit covariance `C`, this yields the first-order
    /// uncertainty `stderr(f) = sqrt(gᵀ C g)` larch propagates with the
    /// `uncertainties` package (`larch.fitting.eval_stderr`). Call after
    /// [`Parameters::update_constraints`] so expression values are current.
    pub fn value_grads(&self) -> Result<HashMap<String, (f64, Vec<f64>)>, ParamError> {
        let var_names = self.var_names();
        let nvar = var_names.len();
        let index: HashMap<&str, usize> = var_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        let sym = self.symbols();

        // seed gradients: each free variable is a unit basis vector; every
        // other non-expression parameter is constant (zero gradient). Injected
        // constants are absent from the map (eval_dual treats absent as zero).
        let mut grads: HashMap<String, Vec<f64>> = HashMap::new();
        for n in &self.order {
            let p = &self.map[n];
            if p.expr.is_none() {
                let mut g = vec![0.0; nvar];
                if let Some(&i) = index.get(n.as_str()) {
                    g[i] = 1.0;
                }
                grads.insert(n.clone(), g);
            }
        }

        // differentiate expression parameters in dependency order
        let mut asts: HashMap<String, Expr> = HashMap::new();
        for n in &self.order {
            if let Some(src) = self.map[n].expr.clone() {
                let ast = parse(&src).map_err(|e| ParamError::Expr(n.clone(), e))?;
                asts.insert(n.clone(), ast);
            }
        }
        for n in &self.topo_order(&asts)? {
            let (_v, g) = asts[n]
                .eval_dual(&sym, &grads, nvar)
                .map_err(|e| ParamError::Expr(n.clone(), e))?;
            grads.insert(n.clone(), g);
        }

        let mut out = HashMap::with_capacity(self.order.len());
        for n in &self.order {
            let g = grads.remove(n).unwrap_or_else(|| vec![0.0; nvar]);
            out.insert(n.clone(), (self.map[n].value, g));
        }
        Ok(out)
    }

    /// Topologically order the expression parameters so each is evaluated after
    /// every expression parameter it depends on. Non-expression dependencies are
    /// already-known leaves. Returns `Cycle` if no valid order exists.
    fn topo_order(&self, asts: &HashMap<String, Expr>) -> Result<Vec<String>, ParamError> {
        // edges: expr param -> the expr params it references
        let mut deps: HashMap<String, Vec<String>> = HashMap::new();
        for (n, ast) in asts {
            let mut vars = Vec::new();
            ast.vars(&mut vars);
            let edges: Vec<String> = vars
                .into_iter()
                .filter(|v| asts.contains_key(v)) // only expr-param deps matter for ordering
                .collect();
            deps.insert(n.clone(), edges);
        }

        // deterministic Kahn-style: process in insertion order when ties
        let expr_order: Vec<String> = self
            .order
            .iter()
            .filter(|n| asts.contains_key(*n))
            .cloned()
            .collect();

        let mut resolved: Vec<String> = Vec::new();
        let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
        // iterate until all resolved or no progress (cycle)
        while resolved.len() < expr_order.len() {
            let mut progressed = false;
            for n in &expr_order {
                if done.contains(n) {
                    continue;
                }
                if deps[n].iter().all(|d| done.contains(d)) {
                    resolved.push(n.clone());
                    done.insert(n.clone());
                    progressed = true;
                }
            }
            if !progressed {
                let unresolved: Vec<String> = expr_order
                    .iter()
                    .filter(|n| !done.contains(*n))
                    .cloned()
                    .collect();
                return Err(ParamError::Cycle(unresolved));
            }
        }
        Ok(resolved)
    }
}
