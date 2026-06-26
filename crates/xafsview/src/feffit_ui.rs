//! The FEFFIT tab: load Feff path files, define global fit variables and
//! per-path parameter expressions, run the [`feffit`](fn@feffit::feffit) fit against the current
//! group's `chi(k)`, and display the statistics and a data-vs-model curve.
//!
//! Following the rest of the GUI, [`FeffitUi::controls`] only *collects* the
//! user's intent into a [`FeffitAction`] for the actions that need app-owned
//! resources (a file dialog, the current group's data); list/table edits it
//! applies to itself directly. The actual fit assembly lives in
//! [`FeffitUi::run`], which the app calls with the group's `k`/`chi`.

use eframe::egui;
use feffit::feffdat::FeffPath;
use feffit::params::Parameters;
use feffit::xasdata::Window;
use feffit::{
    DataSet, FeffitResult, FitDataSet, FitSpace, PathSpec, Spec, Transform, XafsOutput, feffit,
    feffit_eval, xafsft,
};

/// What the FEFFIT controls need the app to do this frame.
pub enum FeffitAction {
    /// Open a file dialog and add the chosen Feff path file(s).
    AddPath,
    /// Run the fit against the current group's `chi(k)`.
    Run,
    /// Redraw the data-vs-model plot (the plot space/part changed).
    Replot,
    /// Open the Plot Data overlay window for the fit's group (the original's
    /// "Send to plot data").
    SendToPlotData,
    /// Save the current fit report to a file (the original's "Save result").
    SaveResult,
    /// Load a saved result/text file into the pop-up viewer ("Load result").
    LoadResult,
    /// Import a UWXAFS `feffit.inp` to populate the fit ("Load inp").
    LoadInp,
}

/// How a global variable enters the fit.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    /// A free fit variable starting at `value`.
    Vary,
    /// Held fixed at `value`.
    Fixed,
    /// A constraint expression over the other variables.
    Expr,
}

/// One global fit variable / constraint row.
#[derive(Clone)]
struct ParamRow {
    name: String,
    kind: ParamKind,
    value: f64,
    expr: String,
}

impl ParamRow {
    fn var(name: &str, value: f64) -> Self {
        Self {
            name: name.to_owned(),
            kind: ParamKind::Vary,
            value,
            expr: String::new(),
        }
    }
}

/// The four per-path local fit parameters (the original's per-path "Guess/Set"
/// block), in display order. Each, when enabled, registers an auto-named
/// variable `<base><pathindex>` (1-based): the coordination factor `N` (the
/// amplitude is `amp·N`), the Debye-Waller `sig` (σ²), and the `third`/`fourth`
/// cumulants. `(base, hint)`.
const PATH_LOCALS: [(&str, &str); 4] = [
    ("N", "coordination factor — amplitude is amp·N"),
    ("sig", "σ² Debye-Waller (Å²)"),
    ("third", "3rd cumulant C₃ (Å³)"),
    ("fourth", "4th cumulant C₄ (Å⁴)"),
];
const LOC_N: usize = 0;
const LOC_SIG: usize = 1;
const LOC_THIRD: usize = 2;
const LOC_FOURTH: usize = 3;

/// One per-path fit parameter: a refined (`Vary` = "guess") or held
/// (`Fixed` = "set") value, an `Expr` override (an expression over other
/// variables, used by `.inp` import), and an enable flag. When disabled the path
/// parameter is the constant `value` (so a disabled `N` gives amplitude `amp·1`).
#[derive(Clone)]
struct PathLocal {
    kind: ParamKind,
    value: f64,
    expr: String,
    enabled: bool,
}

impl PathLocal {
    fn guess(value: f64) -> Self {
        Self {
            kind: ParamKind::Vary,
            value,
            expr: String::new(),
            enabled: true,
        }
    }
    fn set(value: f64) -> Self {
        Self {
            kind: ParamKind::Fixed,
            value,
            expr: String::new(),
            enabled: true,
        }
    }
    /// A parameter present in the block but unchecked (off): a fixed constant.
    fn off(value: f64) -> Self {
        Self {
            kind: ParamKind::Vary,
            value,
            expr: String::new(),
            enabled: false,
        }
    }

    /// The auto-named variable this local registers, or `None` when it is
    /// disabled or an expression (those reference existing variables instead).
    fn var(&self, base: &str, idx: usize) -> Option<(String, ParamKind, f64)> {
        (self.enabled && self.kind != ParamKind::Expr)
            .then(|| (format!("{base}{}", idx + 1), self.kind, self.value))
    }

    /// The [`Spec`] for an additive path parameter (σ²/third/fourth): the
    /// auto-named variable when enabled, the typed expression for `Expr`, else
    /// the constant value.
    fn spec(&self, base: &str, idx: usize) -> Spec {
        if !self.enabled {
            return Spec::Const(self.value);
        }
        match self.kind {
            ParamKind::Expr => parse_spec(&self.expr),
            _ => Spec::Expr(format!("{base}{}", idx + 1)),
        }
    }

    /// The multiplicative factor for the amplitude `amp · factor` (the `N`
    /// local): the variable, the parenthesised expression, or the constant.
    fn factor(&self, base: &str, idx: usize) -> String {
        if !self.enabled {
            return format!("{}", self.value);
        }
        match self.kind {
            ParamKind::Expr => format!("({})", self.expr.trim()),
            _ => format!("{base}{}", idx + 1),
        }
    }
}

/// One loaded Feff path: the shared-parameter wiring (editable expressions over
/// the global variables / the `.dat` file) plus the per-path local fit
/// parameters ([`PATH_LOCALS`]).
#[derive(Clone)]
struct PathRow {
    label: String,
    reff: f64,
    nleg: usize,
    enabled: bool,
    path: FeffPath,
    /// Editable expressions for the shared path parameters: `degen` (from the
    /// file), `e0`/`deltar` wired to the global variables, and `ei`.
    degen: String,
    e0: String,
    ei: String,
    deltar: String,
    /// Per-path amplitude override (`.inp` import / advanced). `None` derives the
    /// amplitude `amp·N` from the shared `amp` global and the `N` local.
    s02_override: Option<String>,
    /// The per-path local fit parameters, in [`PATH_LOCALS`] order.
    locals: [PathLocal; 4],
}

impl PathRow {
    /// Seed a freshly loaded path with the standard first-shell wiring: `degen`
    /// from the file, `e0`/`Δr` bound to the shared globals, the amplitude
    /// `amp·N`, and a refined per-path σ² — ready to fit.
    fn new(label: String, path: FeffPath) -> Self {
        let reff = path.feffdat.reff;
        let nleg = path.feffdat.nleg;
        let degen = format!("{}", path.feffdat.degen);
        Self {
            label,
            reff,
            nleg,
            enabled: true,
            path,
            degen,
            e0: "del_e0".to_owned(),
            ei: "0".to_owned(),
            deltar: "alpha*reff".to_owned(),
            s02_override: None,
            locals: default_locals(),
        }
    }

    /// Reset this path's parameters to the standard first-shell starter wiring
    /// (the original "Init" button for the selected path).
    fn reset_specs(&mut self) {
        self.degen = format!("{}", self.path.feffdat.degen);
        self.e0 = "del_e0".to_owned();
        self.ei = "0".to_owned();
        self.deltar = "alpha*reff".to_owned();
        self.s02_override = None;
        self.locals = default_locals();
    }

    /// Assemble this path's [`PathSpec`] at fit position `idx` (0-based; the
    /// per-path variables are named with the 1-based index).
    fn to_pathspec(&self, idx: usize) -> PathSpec {
        let s02 = match &self.s02_override {
            Some(e) => parse_spec(e),
            None => Spec::Expr(format!("amp*{}", self.locals[LOC_N].factor("N", idx))),
        };
        PathSpec {
            degen: parse_spec(&self.degen),
            s02,
            e0: parse_spec(&self.e0),
            ei: parse_spec(&self.ei),
            deltar: parse_spec(&self.deltar),
            sigma2: self.locals[LOC_SIG].spec("sig", idx),
            third: self.locals[LOC_THIRD].spec("third", idx),
            fourth: self.locals[LOC_FOURTH].spec("fourth", idx),
        }
    }
}

/// A spec field is a constant when it parses as a number, else an expression.
fn parse_spec(s: &str) -> Spec {
    let t = s.trim();
    match t.parse::<f64>() {
        Ok(v) => Spec::Const(v),
        Err(_) => Spec::Expr(t.to_owned()),
    }
}

/// The standard first-shell per-path locals: `N` fixed at 1 (so amplitude is
/// `amp·1`), a refined σ², and the cumulants present but off.
fn default_locals() -> [PathLocal; 4] {
    [
        PathLocal::set(1.0),
        PathLocal::guess(0.003),
        PathLocal::off(0.0),
        PathLocal::off(0.0),
    ]
}

/// Standard global-variable names offered by the "Add ▾" menu, so the common fit
/// parameters can be inserted by name (with a sensible starting value) instead of
/// remembered and typed. Each entry is `(name, default value, hint)`.
///
/// The first three are the shared variables the default path wiring references:
/// `amp`→s02 amplitude, `del_e0`→e0, `alpha`→Δr (`alpha*reff`). `temp`/`debye_temp`
/// are the inputs to the Debye-Waller `sigma2_debye`/`sigma2_eins` helpers, used
/// via a `%set` user function. (σ² and N are per-path locals on each path, not
/// globals — see [`PATH_LOCALS`].)
const STANDARD_VARS: [(&str, f64, &str); 5] = [
    ("amp", 0.9, "S₀² amplitude — shared across paths"),
    (
        "del_e0",
        0.0,
        "ΔE₀ energy-origin shift — shared across paths",
    ),
    ("alpha", 0.0, "lattice expansion — shared (Δr = alpha·reff)"),
    (
        "temp",
        300.0,
        "temperature (K) for the sigma2_debye/eins helpers",
    ),
    (
        "debye_temp",
        300.0,
        "Debye temperature θ_D (K) for sigma2_debye",
    ),
];

/// Parse the "user defined functions" box: each `%set NAME = EXPR` line yields a
/// `(name, expr)` pair (blank lines and `%`-comments are ignored), the way the
/// original XAFSView's UDF block defines extra named fit constants/constraints.
fn parse_user_funcs(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        // Only `%set` definitions; other `%`-lines are comments/directives.
        if t.len() < 4 || !t[..4].eq_ignore_ascii_case("%set") {
            continue;
        }
        let rest = &t[4..];
        if !rest.starts_with(char::is_whitespace) {
            continue; // e.g. "%setx" is not a "%set" definition
        }
        let Some((name, expr)) = rest.split_once('=') else {
            continue;
        };
        let (name, expr) = (name.trim(), expr.trim());
        let valid_name = !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_');
        if valid_name && !expr.is_empty() {
            out.push((name.to_owned(), expr.to_owned()));
        }
    }
    out
}

/// One global/local variable from a `feffit.inp`: `guess`/`set NAME = VALUE`.
#[derive(Debug, Clone, PartialEq)]
pub struct InpVar {
    pub name: String,
    /// `true` for `guess` (a free variable), `false` for `set` (fixed).
    pub guess: bool,
    pub value: f64,
}

/// One path entry from a `feffit.inp`: the `feffNNNN.dat` file (as written, with
/// Windows separators) and any per-parameter expression overrides.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InpPath {
    pub number: usize,
    pub file: String,
    pub s02: Option<String>,
    pub e0: Option<String>,
    pub ei: Option<String>,
    pub deltar: Option<String>,
    pub sigma2: Option<String>,
    pub third: Option<String>,
    pub fourth: Option<String>,
}

/// A parsed UWXAFS `feffit.inp` (the subset XAFSView writes): FT-window
/// parameters, the fit/no-fit flag, user functions, variables, and path
/// entries. Window fields are `Option` so an absent key leaves the current
/// value intact on apply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeffitInp {
    pub kmin: Option<f64>,
    pub kmax: Option<f64>,
    pub dk: Option<f64>,
    pub rmin: Option<f64>,
    pub rmax: Option<f64>,
    pub dr: Option<f64>,
    pub kweight: Option<i32>,
    pub iwindo: Option<i32>,
    pub nofit: Option<bool>,
    pub user_funcs: String,
    pub vars: Vec<InpVar>,
    pub paths: Vec<InpPath>,
}

/// Parse a UWXAFS `feffit.inp` into a [`FeffitInp`]. A line whose first
/// non-blank character is `%` is disabled (commented out) — except the `%set`
/// user-function directive — and an inline trailing `% …` comment is stripped
/// from active lines. Lines are classified by their first token: `set`/`guess`
/// variables, `path`/`e0`/`delR`/`s02`/`sigma2`/`third`/`fourth`/`ei` path
/// entries, `Nofit`, and the `kmin = … rmax = …` window-parameter pairs.
pub fn parse_feffit_inp(text: &str) -> FeffitInp {
    use std::collections::BTreeMap;

    let mut inp = FeffitInp::default();
    let mut paths: BTreeMap<usize, InpPath> = BTreeMap::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // `%set NAME = EXPR` is an active user-function directive.
        if line.len() >= 4
            && line[..4].eq_ignore_ascii_case("%set")
            && line[4..].starts_with(char::is_whitespace)
        {
            inp.user_funcs.push_str(line);
            inp.user_funcs.push('\n');
            continue;
        }
        // Any other leading `%` (including `%%`) is a comment.
        if line.starts_with('%') {
            continue;
        }
        // Strip a trailing inline `% …` comment, then normalise `=` spacing.
        let active = line.split('%').next().unwrap_or(line).trim();
        if active.is_empty() {
            continue;
        }
        let spaced = active.replace('=', " = ");
        let tok: Vec<&str> = spaced.split_whitespace().collect();
        if tok.is_empty() {
            continue;
        }
        let key = tok[0].to_ascii_lowercase();
        match key.as_str() {
            "set" | "guess" if tok.len() >= 4 && tok[2] == "=" => {
                if let Ok(v) = tok[3].parse::<f64>() {
                    inp.vars.push(InpVar {
                        name: tok[1].to_owned(),
                        guess: key == "guess",
                        value: v,
                    });
                }
            }
            "nofit" if tok.len() >= 3 && tok[1] == "=" => {
                inp.nofit = Some(tok[2].eq_ignore_ascii_case("true"));
            }
            "path" if tok.len() >= 3 => {
                if let Ok(n) = tok[1].parse::<usize>() {
                    let p = paths.entry(n).or_default();
                    p.number = n;
                    p.file = tok[2..].join(" ");
                }
            }
            "e0" | "delr" | "s02" | "sigma2" | "third" | "fourth" | "ei" if tok.len() >= 3 => {
                if let Ok(n) = tok[1].parse::<usize>() {
                    let expr = tok[2..].join(" ");
                    let p = paths.entry(n).or_default();
                    p.number = n;
                    match key.as_str() {
                        "e0" => p.e0 = Some(expr),
                        "delr" => p.deltar = Some(expr),
                        "s02" => p.s02 = Some(expr),
                        "sigma2" => p.sigma2 = Some(expr),
                        "third" => p.third = Some(expr),
                        "fourth" => p.fourth = Some(expr),
                        "ei" => p.ei = Some(expr),
                        _ => unreachable!(),
                    }
                }
            }
            // Otherwise scan for `key = value` window-parameter pairs.
            _ => {
                let mut i = 0;
                while i + 2 < tok.len() {
                    if tok[i + 1] == "=" {
                        let k = tok[i].to_ascii_lowercase();
                        let val = tok[i + 2];
                        match k.as_str() {
                            "kmin" => inp.kmin = val.parse().ok(),
                            "kmax" => inp.kmax = val.parse().ok(),
                            "dk" => inp.dk = val.parse().ok(),
                            "rmin" => inp.rmin = val.parse().ok(),
                            "rmax" => inp.rmax = val.parse().ok(),
                            "dr" => inp.dr = val.parse().ok(),
                            "kweight" => inp.kweight = val.parse().ok(),
                            "iwindo" => inp.iwindo = val.parse().ok(),
                            _ => {}
                        }
                        i += 3;
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    inp.paths = paths.into_values().collect();
    inp
}

/// Fit-transform (k/R window) settings for the FEFFIT fit.
#[derive(Clone)]
struct FtSettings {
    kmin: f64,
    kmax: f64,
    kweight: i32,
    dk: f64,
    kwindow: Window,
    rmin: f64,
    rmax: f64,
    dr: f64,
    rwindow: Window,
    fitspace: FitSpace,
}

impl Default for FtSettings {
    fn default() -> Self {
        Self {
            kmin: 3.0,
            kmax: 14.0,
            kweight: 2,
            dk: 1.0,
            kwindow: Window::Hanning,
            rmin: 1.4,
            rmax: 3.0,
            dr: 0.0,
            rwindow: Window::Hanning,
            fitspace: FitSpace::R,
        }
    }
}

impl FtSettings {
    /// Build the [`Transform`] for the fit. `nfft`/`kstep` use larch's defaults;
    /// `rbkg = 0` so the R-window starts at `rmin` (the fit lower bound).
    fn to_transform(&self) -> Transform {
        Transform::new(
            self.kmin,
            self.kmax,
            vec![self.kweight],
            self.dk,
            None,
            self.kwindow,
            2048,
            0.05,
            self.rmin,
            self.rmax,
            self.dr,
            None,
            self.rwindow,
            0.0,
            self.fitspace,
        )
    }
}

/// Which space and part of `chi` to draw for data vs model.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlotSpace {
    /// `kʷ·χ(k)`.
    K,
    /// `χ(R)` (R-space).
    R,
    /// `χ(q)` (back-transformed k-space).
    Q,
    /// `kʷ·χ(k)` and `χ(q)` overlaid (both vs Å⁻¹) — the original's "K+Q".
    KQ,
}

/// For R/Q space, which component to draw.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlotPart {
    Mag,
    Re,
    Im,
    Pha,
}

/// The original "Fit" dropdown (manual §1.2.2): what the Run button does — the
/// UWXAFS feffit `fit / no fit / only FT` modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    /// Forward-evaluate the model at the current guess values and overlay it on
    /// the data — no least-squares optimisation, no statistics.
    NoFit,
    /// Fourier-transform the data only (no paths, no model). The default, matching
    /// the original XAFSView (its "Fit" dropdown opens on "Only FT").
    OnlyFt,
    /// Run the least-squares fit, then overlay the best-fit model.
    Fit,
}

impl FitMode {
    /// All modes in the dropdown's display order (the default, "Only FT", first).
    pub const ALL: [FitMode; 3] = [FitMode::OnlyFt, FitMode::NoFit, FitMode::Fit];

    /// Dropdown label (matches the original's "Fit" ring text).
    pub fn label(self) -> &'static str {
        match self {
            FitMode::NoFit => "No fit",
            FitMode::OnlyFt => "Only FT",
            FitMode::Fit => "Fit",
        }
    }
}

/// One of the original's "View …" result reports (the buttons under "Feffit out
/// data"), shown in a pop-up text window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportKind {
    /// Pairwise correlations between the free variables.
    Correlations,
    /// The free (varied) variables: value ± stderr.
    FitValues,
    /// The fixed global variables: name = value.
    FixValues,
    /// The full fit report (statistics, variables, path parameters).
    FeffitSumm,
    /// The path parameters, grouped by path.
    PathSumm,
    /// The fit statistics block on its own.
    ResultsSumm,
}

impl ReportKind {
    /// The pop-up window title.
    pub fn title(self) -> &'static str {
        match self {
            ReportKind::Correlations => "Correlations",
            ReportKind::FitValues => "Fit values",
            ReportKind::FixValues => "Fix values",
            ReportKind::FeffitSumm => "Feffit summary",
            ReportKind::PathSumm => "Path summary",
            ReportKind::ResultsSumm => "Results summary",
        }
    }
}

/// Data and model arrays from the last fit, ready to plot in any space.
pub struct FeffitPlot {
    pub data_k: Vec<f64>,
    pub data_chi: Vec<f64>,
    pub model_chi: Vec<f64>,
    pub data: XafsOutput,
    pub model: XafsOutput,
    /// k-weight the fit used (for the `kʷ·χ(k)` plot).
    pub kweight: i32,
    /// Whether `model`/`model_chi` are meaningful. `false` in "Only FT" mode
    /// (data transformed alone) so consumers skip the model curve and `.fit`
    /// output files.
    pub has_model: bool,
}

impl FeffitPlot {
    /// Build the `(x, data_y, model_y, x-label, y-label)` series for a given
    /// plot space and part. For k-space the part is ignored (`kʷ·χ(k)`); for
    /// R/Q the part selects magnitude / real / imag / phase. `KQ` falls back to
    /// the k-space series — the combined view is drawn via [`kq_series`](Self::kq_series).
    pub fn series(
        &self,
        space: PlotSpace,
        part: PlotPart,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, &'static str, &'static str) {
        match space {
            PlotSpace::K | PlotSpace::KQ => {
                let kw = self.kweight;
                let weight = |k: &[f64], chi: &[f64]| -> Vec<f64> {
                    k.iter().zip(chi).map(|(&k, &c)| c * k.powi(kw)).collect()
                };
                (
                    self.data_k.clone(),
                    weight(&self.data_k, &self.data_chi),
                    weight(&self.data_k, &self.model_chi),
                    "k (Å⁻¹)",
                    "kʷ·χ(k)",
                )
            }
            PlotSpace::R => {
                let (dy, my, yl) = pick_part(part, &self.data, &self.model, true);
                (self.data.r.clone(), dy, my, "R (Å)", yl)
            }
            PlotSpace::Q => {
                let (dy, my, yl) = pick_part(part, &self.data, &self.model, false);
                (self.data.q.clone(), dy, my, "q (Å⁻¹)", yl)
            }
        }
    }

    /// The "K+Q" combined view: the `kʷ·χ(k)` series and the `χ(q)` series (the
    /// chosen `part`), both on the Å⁻¹ axis. Returns
    /// `((k_x, k_data, k_model), (q_x, q_data, q_model))`.
    #[allow(clippy::type_complexity)]
    pub fn kq_series(
        &self,
        part: PlotPart,
    ) -> (
        (Vec<f64>, Vec<f64>, Vec<f64>),
        (Vec<f64>, Vec<f64>, Vec<f64>),
    ) {
        let (kx, kd, km, _, _) = self.series(PlotSpace::K, part);
        let (qx, qd, qm, _, _) = self.series(PlotSpace::Q, part);
        ((kx, kd, km), (qx, qd, qm))
    }

    /// The six `(filename, content)` transforms the original's Plot Data reads,
    /// named from `stem`: k/r/q-space data → `<stem>k.dat`/`r.dat`/`q.dat` and
    /// the model → `<stem>k.fit`/`r.fit`/`q.fit`. Single owner of the
    /// field→file mapping, shared by the single-fit on-disk writer and the
    /// batch `(name, content)` builder so both stay byte-identical.
    pub fn output_pairs(&self, stem: &str) -> Vec<(String, String)> {
        use crate::chi_io::{chik_string, complex4_string};
        let (d, m) = (&self.data, &self.model);
        vec![
            (
                format!("{stem}k.dat"),
                chik_string(stem, &self.data_k, &self.data_chi),
            ),
            (
                format!("{stem}k.fit"),
                chik_string(stem, &self.data_k, &self.model_chi),
            ),
            (
                format!("{stem}r.dat"),
                complex4_string(stem, "R", &d.r, &d.chir_mag, &d.chir_re, &d.chir_im),
            ),
            (
                format!("{stem}r.fit"),
                complex4_string(stem, "R", &m.r, &m.chir_mag, &m.chir_re, &m.chir_im),
            ),
            (
                format!("{stem}q.dat"),
                complex4_string(stem, "q", &d.q, &d.chiq_mag, &d.chiq_re, &d.chiq_im),
            ),
            (
                format!("{stem}q.fit"),
                complex4_string(stem, "q", &m.q, &m.chiq_mag, &m.chiq_re, &m.chiq_im),
            ),
        ]
    }
}

/// Pick the data/model component arrays (and a y-label) for the chosen part,
/// from R-space (`r_space = true`) or q-space outputs.
fn pick_part(
    part: PlotPart,
    data: &XafsOutput,
    model: &XafsOutput,
    r_space: bool,
) -> (Vec<f64>, Vec<f64>, &'static str) {
    let (dmag, dre, dim, dpha) = if r_space {
        (&data.chir_mag, &data.chir_re, &data.chir_im, &data.chir_pha)
    } else {
        (&data.chiq_mag, &data.chiq_re, &data.chiq_im, &data.chiq_pha)
    };
    let (mmag, mre, mim, mpha) = if r_space {
        (
            &model.chir_mag,
            &model.chir_re,
            &model.chir_im,
            &model.chir_pha,
        )
    } else {
        (
            &model.chiq_mag,
            &model.chiq_re,
            &model.chiq_im,
            &model.chiq_pha,
        )
    };
    match part {
        PlotPart::Mag => (dmag.clone(), mmag.clone(), "|χ|"),
        PlotPart::Re => (dre.clone(), mre.clone(), "Re χ"),
        PlotPart::Im => (dim.clone(), mim.clone(), "Im χ"),
        PlotPart::Pha => (dpha.clone(), mpha.clone(), "Phase χ"),
    }
}

/// One fitted path's data for the batch "Save Items": its feff path number
/// (parsed from the `feffNNNN.dat` label), its `reff`, and the path parameters
/// that were fitted (each as `value` + propagated stderr).
pub struct SavedPath {
    /// The feff path number (`feff0007.dat` → 7), or the 1-based position when
    /// the label carries no number.
    pub number: usize,
    /// Half path length from the feff file.
    pub reff: f64,
    /// `(parameter name, value, stderr)` for each fitted path parameter, names
    /// drawn from [`feffit::PATH_PNAMES`].
    params: Vec<(String, f64, f64)>,
}

impl SavedPath {
    /// `(value, stderr)` of the named path parameter, if it was fitted.
    fn param(&self, name: &str) -> Option<(f64, f64)> {
        self.params
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, v, e)| (*v, *e))
    }

    /// `(value, stderr)` for a Save-Items key: a [`feffit::PATH_PNAMES`] parameter name,
    /// or `""` for the computed `reff + Δr` (Δr's value offset by the constant
    /// `reff`, carrying Δr's stderr). Missing parameters report `(0, 0)`, the
    /// original's `n = 0` filler for paths a fit does not use.
    pub fn item(&self, key: &str) -> (f64, f64) {
        if key.is_empty() {
            let (dv, de) = self.param("deltar").unwrap_or((0.0, 0.0));
            (self.reff + dv, de)
        } else {
            self.param(key).unwrap_or((0.0, 0.0))
        }
    }
}

/// The feff path number embedded in a `feffNNNN.dat` label (`feff0007.dat` → 7),
/// if present.
fn path_number(label: &str) -> Option<usize> {
    let lower = label.to_ascii_lowercase();
    let start = lower.find("feff")? + 4;
    let digits: String = lower[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// FEFFIT tab state: the path list, the variables, the transform, the plot
/// selection, and the last fit result.
pub struct FeffitUi {
    paths: Vec<PathRow>,
    params: Vec<ParamRow>,
    ft: FtSettings,
    space: PlotSpace,
    part: PlotPart,
    /// Which path's parameter specs the path panel is showing (the original's
    /// "Path index" selector); clamped to the path list.
    selected_path: usize,
    /// The "user defined functions" box: extra `%set name = expr` definitions
    /// parsed into the fit's parameters before each run.
    user_funcs: String,
    /// The original's "Fit" dropdown: fit / no fit / only FT. Defaults to
    /// `NoFit` (forward model preview without optimising).
    fit_mode: FitMode,
    /// Which "View …" result report pop-up is open (transient UI state).
    report_view: Option<ReportKind>,
    /// An ad-hoc `(title, body)` text pop-up — a loaded result file or the log.
    text_view: Option<(String, String)>,
    /// Snapshot of the variable table taken before "Use fit as guess", so that
    /// the adopt can be reverted with "Undo".
    param_undo: Option<Vec<ParamRow>>,
    result: Option<FeffitResult>,
    plot: Option<FeffitPlot>,
}

impl Default for FeffitUi {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            // The shared global variables the seeded path wiring references:
            // S₀² amplitude, ΔE₀, and the lattice expansion. (σ² and N are
            // per-path locals on each path, not globals.)
            params: vec![
                ParamRow::var("amp", 0.9),
                ParamRow::var("del_e0", 0.0),
                ParamRow::var("alpha", 0.0),
            ],
            ft: FtSettings::default(),
            space: PlotSpace::R,
            part: PlotPart::Mag,
            selected_path: 0,
            user_funcs: String::new(),
            fit_mode: FitMode::OnlyFt,
            report_view: None,
            text_view: None,
            param_undo: None,
            result: None,
            plot: None,
        }
    }
}

impl FeffitUi {
    /// A fresh copy of just the fit *configuration* — paths, variables, the
    /// transform, and the plot selection — with no result or computed plot. Used
    /// to seed each per-group batch fit from the Feffit tab as a template that
    /// can then be edited independently per group.
    pub fn config_clone(&self) -> FeffitUi {
        FeffitUi {
            paths: self.paths.clone(),
            params: self.params.clone(),
            ft: self.ft.clone(),
            space: self.space,
            part: self.part,
            selected_path: 0,
            user_funcs: self.user_funcs.clone(),
            fit_mode: self.fit_mode,
            report_view: None,
            text_view: None,
            param_undo: None,
            result: None,
            plot: None,
        }
    }

    /// The last fit result, if a fit has been run (for the batch result table).
    pub fn result(&self) -> Option<&FeffitResult> {
        self.result.as_ref()
    }

    /// Per-path fitted data for the batch "Save Items", in the order the paths
    /// were fitted (the enabled rows). Empty until a fit has been run. The
    /// `path` index of each [`PathParam`](feffit::PathParam) matches the enabled
    /// path position, since [`run`](Self::run) builds the fit's path list from
    /// exactly those rows in order.
    pub fn saved_paths(&self) -> Vec<SavedPath> {
        let Some(res) = &self.result else {
            return Vec::new();
        };
        self.paths
            .iter()
            .filter(|p| p.enabled)
            .enumerate()
            .map(|(pi, row)| SavedPath {
                number: path_number(&row.label).unwrap_or(pi + 1),
                reff: row.reff,
                params: res
                    .path_params
                    .iter()
                    .filter(|pp| pp.path == pi)
                    .map(|pp| (pp.name.clone(), pp.value, pp.stderr))
                    .collect(),
            })
            .collect()
    }

    /// The last fit's plot arrays, if a fit has been run.
    pub fn plot(&self) -> Option<&FeffitPlot> {
        self.plot.as_ref()
    }

    /// The active plot space and part.
    pub fn plot_selection(&self) -> (PlotSpace, PlotPart) {
        (self.space, self.part)
    }

    /// A plain-text fit report (the Feffit_txt view): statistics, free
    /// variables, derived constraints, and every path parameter with its
    /// propagated uncertainty. Empty string when no fit has been run.
    pub fn report_text(&self) -> String {
        let Some(res) = &self.result else {
            return String::new();
        };
        let mut s = String::new();
        s.push_str("=== Fit statistics ===\n");
        s.push_str(&format!("  n_independent  = {:.3}\n", res.n_idp));
        s.push_str(&format!("  n_varys        = {}\n", res.nvarys));
        s.push_str(&format!("  n_data         = {}\n", res.ndata));
        s.push_str(&format!("  chi_square     = {:.6}\n", res.chi_square));
        s.push_str(&format!("  reduced chi^2  = {:.6}\n", res.chi2_reduced));
        s.push_str(&format!("  R-factor       = {:.6}\n", res.rfactor));
        s.push_str(&format!("  Akaike (AIC)   = {:.4}\n", res.aic));
        s.push_str(&format!("  Bayesian (BIC) = {:.4}\n", res.bic));
        s.push_str(&format!("  n_function_evals = {}\n", res.nfev));
        s.push_str(&format!("  termination info = {}\n", res.info));

        s.push_str("\n=== Variables ===\n");
        for b in &res.best {
            s.push_str(&format!(
                "  {:<14} = {:>12.6} +/- {:.6}\n",
                b.name, b.value, b.stderr
            ));
        }
        if !res.derived.is_empty() {
            s.push_str("\n=== Derived (constraints) ===\n");
            for d in &res.derived {
                s.push_str(&format!(
                    "  {:<14} = {:>12.6} +/- {:.6}\n",
                    d.name, d.value, d.stderr
                ));
            }
        }

        s.push_str("\n=== Path parameters ===\n");
        for pp in &res.path_params {
            s.push_str(&format!(
                "  ds{} path{} {:<8} = {:>12.6} +/- {:.6}\n",
                pp.dataset, pp.path, pp.name, pp.value, pp.stderr
            ));
        }
        s
    }

    /// Text body for one of the "View …" result reports. Reports that need a
    /// fit result return a short notice when none has been run; "Fix values"
    /// reads the current parameter table, so it works without a fit.
    pub fn report_for(&self, kind: ReportKind) -> String {
        // Fix values come from the parameter table, not the fit result.
        if kind == ReportKind::FixValues {
            let mut s = String::from("=== Fixed variables ===\n");
            let mut any = false;
            for row in self.params.iter().filter(|r| r.kind == ParamKind::Fixed) {
                s.push_str(&format!("  {:<14} = {:>12.6}\n", row.name, row.value));
                any = true;
            }
            if !any {
                s.push_str("  (none — every variable is varied or an expression)\n");
            }
            return s;
        }

        let Some(res) = &self.result else {
            return "Run a fit first — this report needs fit results.".to_owned();
        };
        match kind {
            ReportKind::FixValues => unreachable!("handled above"),
            ReportKind::FitValues => {
                let mut s = String::from("=== Free variables (value ± stderr) ===\n");
                for b in &res.best {
                    s.push_str(&format!(
                        "  {:<14} = {:>12.6} ± {:.6}\n",
                        b.name, b.value, b.stderr
                    ));
                }
                s
            }
            ReportKind::Correlations => {
                let mut s = String::from("=== Correlations between free variables ===\n");
                let Some(cov) = &res.covar else {
                    s.push_str("  (covariance unavailable — the Jacobian was singular)\n");
                    return s;
                };
                let n = res.best.len();
                // corr[i][j] = cov[i][j] / sqrt(cov[i][i]·cov[j][j]); list each
                // pair once, largest magnitude first.
                let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        let d = (cov[i][i] * cov[j][j]).sqrt();
                        let c = if d > 0.0 { cov[i][j] / d } else { 0.0 };
                        pairs.push((i, j, c));
                    }
                }
                pairs.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
                if pairs.is_empty() {
                    s.push_str("  (fewer than two free variables)\n");
                }
                for (i, j, c) in pairs {
                    s.push_str(&format!(
                        "  {:<14} {:<14} = {:>+7.4}\n",
                        res.best[i].name, res.best[j].name, c
                    ));
                }
                s
            }
            ReportKind::PathSumm => {
                let mut s = String::from("=== Path parameters ===\n");
                let mut last = usize::MAX;
                for pp in &res.path_params {
                    if pp.path != last {
                        s.push_str(&format!("\n  path {}:\n", pp.path));
                        last = pp.path;
                    }
                    s.push_str(&format!(
                        "    {:<8} = {:>12.6} ± {:.6}\n",
                        pp.name, pp.value, pp.stderr
                    ));
                }
                s
            }
            ReportKind::ResultsSumm => {
                let mut s = String::from("=== Fit statistics ===\n");
                s.push_str(&format!("  n_independent  = {:.3}\n", res.n_idp));
                s.push_str(&format!("  n_varys        = {}\n", res.nvarys));
                s.push_str(&format!("  n_free         = {}\n", res.nfree));
                s.push_str(&format!("  chi_square     = {:.6}\n", res.chi_square));
                s.push_str(&format!("  reduced chi^2  = {:.6}\n", res.chi2_reduced));
                s.push_str(&format!("  R-factor       = {:.6}\n", res.rfactor));
                s.push_str(&format!("  Akaike (AIC)   = {:.4}\n", res.aic));
                s.push_str(&format!("  Bayesian (BIC) = {:.4}\n", res.bic));
                s
            }
            ReportKind::FeffitSumm => self.report_text(),
        }
    }

    /// Open the ad-hoc text pop-up (a loaded result file, or the log) with the
    /// given title and body.
    pub fn show_text(&mut self, title: impl Into<String>, body: String) {
        self.text_view = Some((title.into(), body));
    }

    /// Apply a parsed [`FeffitInp`]'s non-path settings — FT windows, fit mode,
    /// user functions, and variables — replacing the current configuration. The
    /// path list is cleared; the app loads each `.dat` file and re-adds it with
    /// [`add_inp_path`](Self::add_inp_path). Clears any stale result/plot.
    pub fn apply_inp(&mut self, inp: &FeffitInp) {
        if let Some(v) = inp.kmin {
            self.ft.kmin = v;
        }
        if let Some(v) = inp.kmax {
            self.ft.kmax = v;
        }
        if let Some(v) = inp.dk {
            self.ft.dk = v;
        }
        if let Some(v) = inp.rmin {
            self.ft.rmin = v;
        }
        if let Some(v) = inp.rmax {
            self.ft.rmax = v;
        }
        if let Some(v) = inp.dr {
            self.ft.dr = v;
        }
        if let Some(v) = inp.kweight {
            self.ft.kweight = v;
        }
        // iwindo = 1 is the UWXAFS Hanning window (the only code XAFSView writes);
        // other codes are left unmapped (no reference table) so the window keeps
        // its current value.
        if inp.iwindo == Some(1) {
            self.ft.kwindow = Window::Hanning;
            self.ft.rwindow = Window::Hanning;
        }
        if let Some(nofit) = inp.nofit {
            self.fit_mode = if nofit { FitMode::NoFit } else { FitMode::Fit };
        }
        self.user_funcs = inp.user_funcs.clone();
        self.params = inp
            .vars
            .iter()
            .map(|v| ParamRow {
                name: v.name.clone(),
                kind: if v.guess {
                    ParamKind::Vary
                } else {
                    ParamKind::Fixed
                },
                value: v.value,
                expr: String::new(),
            })
            .collect();
        self.paths.clear();
        self.selected_path = 0;
        self.result = None;
        self.plot = None;
    }

    /// Add a path loaded for a `Load inp` import, applying the `.inp`'s
    /// per-parameter expression overrides over the default first-shell wiring
    /// (`degen` stays as read from the `.dat` file). The `.inp` carries its own
    /// per-path variables (`e1`/`delr1`/`sig1`/…, brought into the global table
    /// by [`apply_inp`](Self::apply_inp)), so the shared expressions and the
    /// σ²/cumulant locals are kept verbatim as `Expr` references to them; the
    /// `.inp`'s amplitude expression overrides the derived `amp·N`.
    pub fn add_inp_path(&mut self, label: String, path: FeffPath, inp: &InpPath) {
        let mut row = PathRow::new(label, path);
        if let Some(v) = &inp.e0 {
            row.e0 = v.clone();
        }
        if let Some(v) = &inp.ei {
            row.ei = v.clone();
        }
        if let Some(v) = &inp.deltar {
            row.deltar = v.clone();
        }
        if let Some(v) = &inp.s02 {
            // The `.inp` drives amplitude itself; disable the unused N local so it
            // does not register a stray variable.
            row.s02_override = Some(v.clone());
            row.locals[LOC_N].enabled = false;
        }
        for (li, ov) in [
            (LOC_SIG, &inp.sigma2),
            (LOC_THIRD, &inp.third),
            (LOC_FOURTH, &inp.fourth),
        ] {
            if let Some(v) = ov {
                row.locals[li] = PathLocal {
                    kind: ParamKind::Expr,
                    value: 0.0,
                    expr: v.clone(),
                    enabled: true,
                };
            }
        }
        self.paths.push(row);
    }

    /// Adopt the last fit's best-fit values as the new starting guesses for the
    /// matching `vary` variables (the EXAFS iterate-the-fit workflow, and how
    /// the path parameters — being expressions over these variables — pick up
    /// the new guesses too). Snapshots the previous values first so
    /// [`undo_guess`](Self::undo_guess) can revert. Returns how many variables
    /// were updated.
    pub fn adopt_fit_as_guess(&mut self) -> usize {
        let best: Vec<(String, f64)> = match &self.result {
            Some(res) => res.best.iter().map(|b| (b.name.clone(), b.value)).collect(),
            None => return 0,
        };
        let snapshot = self.params.clone();
        let mut updated = 0;
        for row in self.params.iter_mut().filter(|r| r.kind == ParamKind::Vary) {
            if let Some((_, v)) = best.iter().find(|(n, _)| *n == row.name) {
                row.value = *v;
                updated += 1;
            }
        }
        if updated > 0 {
            self.param_undo = Some(snapshot);
        }
        updated
    }

    /// Revert the most recent "Use fit as guess" snapshot. Returns `true` if a
    /// snapshot was restored.
    pub fn undo_guess(&mut self) -> bool {
        match self.param_undo.take() {
            Some(prev) => {
                self.params = prev;
                true
            }
            None => false,
        }
    }

    /// Add a path file the app picked from a dialog.
    pub fn add_path(&mut self, label: String, path: FeffPath) {
        self.paths.push(PathRow::new(label, path));
    }

    /// Whether any enabled path is loaded (the fit needs at least one).
    pub fn has_enabled_path(&self) -> bool {
        self.paths.iter().any(|p| p.enabled)
    }

    /// Run the active [`FitMode`] against `data_k`/`data_chi`, storing the result
    /// (when fitting) and the data-vs-model plot arrays. Returns a one-line status
    /// on success or an error message.
    ///
    /// - [`FitMode::OnlyFt`]: Fourier-transform the data alone (no paths needed).
    /// - [`FitMode::NoFit`]: forward-evaluate the model at the current guess
    ///   values and overlay it — no optimisation, no statistics.
    /// - [`FitMode::Fit`]: full least-squares fit, then overlay the best-fit model.
    pub fn run(&mut self, data_k: &[f64], data_chi: &[f64]) -> Result<String, String> {
        if data_k.is_empty() || data_chi.len() != data_k.len() {
            return Err("Current group has no chi(k) — run AUTOBK first.".to_owned());
        }
        let rmax_out = self.ft.rmax + 2.0;

        // Only FT: transform the data on its own — no paths, no model, no fit.
        if self.fit_mode == FitMode::OnlyFt {
            let data = xafsft(&self.ft.to_transform(), data_chi, rmax_out);
            self.plot = Some(FeffitPlot {
                data_k: data_k.to_vec(),
                data_chi: data_chi.to_vec(),
                model_chi: Vec::new(),
                // Unused when `has_model` is false; clone the data so every plot
                // array stays length-consistent without an empty-`XafsOutput`.
                model: data.clone(),
                data,
                kweight: self.ft.kweight,
                has_model: false,
            });
            self.result = None;
            return Ok("Only FT: transformed data χ(k) → χ(R)/χ(q).".to_owned());
        }

        if !self.has_enabled_path() {
            return Err("No enabled Feff paths to fit.".to_owned());
        }

        let mut params = Parameters::new();
        for row in &self.params {
            match row.kind {
                ParamKind::Vary => params.add_var(&row.name, row.value),
                ParamKind::Fixed => params.add_fixed(&row.name, row.value),
                ParamKind::Expr => params.add_expr(&row.name, row.expr.trim()),
            }
        }
        // User-defined functions: `%set name = expr` lines become extra fixed
        // (numeric RHS) or expression parameters. Order-independent — the
        // dependency resolve happens in `update_constraints` at fit time.
        for (name, expr) in parse_user_funcs(&self.user_funcs) {
            match expr.parse::<f64>() {
                Ok(v) => params.add_fixed(&name, v),
                Err(_) => params.add_expr(&name, &expr),
            }
        }

        let mut feff_paths = Vec::new();
        let mut specs = Vec::new();
        for (idx, row) in self.paths.iter().enumerate() {
            if !row.enabled {
                continue;
            }
            // Register this path's enabled local variables (auto-named by the
            // 1-based path index), then wire its spec to them.
            for (li, (base, _)) in PATH_LOCALS.iter().enumerate() {
                if let Some((name, kind, value)) = row.locals[li].var(base, idx) {
                    match kind {
                        ParamKind::Vary => params.add_var(&name, value),
                        ParamKind::Fixed => params.add_fixed(&name, value),
                        ParamKind::Expr => {}
                    }
                }
            }
            feff_paths.push(row.path.clone());
            specs.push(row.to_pathspec(idx));
        }

        let dataset = DataSet::new(
            data_k.to_vec(),
            data_chi.to_vec(),
            feff_paths,
            self.ft.to_transform(),
        );
        let mut fds = vec![FitDataSet {
            dataset,
            specs,
            epsilon_k: None,
        }];

        // No fit: forward-evaluate the model at the guess values, no optimisation.
        if self.fit_mode == FitMode::NoFit {
            feffit_eval(&mut params, &mut fds).map_err(|e| format!("model eval failed: {e}"))?;
            let model_chi = fds[0].dataset.model_chi_sum();
            let out = fds[0].dataset.save_outputs(rmax_out, false);
            self.plot = Some(FeffitPlot {
                data_k: data_k.to_vec(),
                data_chi: data_chi.to_vec(),
                model_chi,
                data: out.data,
                model: out.model,
                kweight: self.ft.kweight,
                has_model: true,
            });
            self.result = None;
            return Ok("No fit: forward model at the current guess values.".to_owned());
        }

        // Fit: full least-squares optimisation.
        let res = feffit(&mut params, &mut fds).map_err(|e| format!("fit failed: {e}"))?;

        // Model chi(k) on the data grid, and forward-FT of both data and model.
        let model_chi = fds[0].dataset.model_chi_sum();
        let out = fds[0].dataset.save_outputs(rmax_out, false);
        self.plot = Some(FeffitPlot {
            data_k: data_k.to_vec(),
            data_chi: data_chi.to_vec(),
            model_chi,
            data: out.data,
            model: out.model,
            kweight: self.ft.kweight,
            has_model: true,
        });

        let summary = format!(
            "Fit: χ²ᵣ = {:.4}, R = {:.5}, n_idp = {:.1}, nvarys = {}, info = {}",
            res.chi2_reduced, res.rfactor, res.n_idp, res.nvarys, res.info
        );
        self.result = Some(res);
        Ok(summary)
    }

    /// Render the control column. Returns a [`FeffitAction`] for app-owned work.
    pub fn controls(&mut self, ui: &mut egui::Ui) -> Option<FeffitAction> {
        let mut action = None;

        ui.heading("Feffit");

        // --- Head param. (k/R Fourier-transform window) -------------------
        // Original XAFSView "Head param." block: kmin/rmin, kmax/rmax, dk/dr,
        // kweight/fit-space, k-window/R-window — two transform columns per row.
        ui.group(|ui| {
            ui.strong("Head param.");
            egui::Grid::new("feffit_head")
                .num_columns(4)
                .spacing([6.0, 4.0])
                .show(ui, |ui| {
                    ui.label("kmin");
                    ui.add(
                        egui::DragValue::new(&mut self.ft.kmin)
                            .speed(0.1)
                            .range(0.0..=8.0),
                    );
                    ui.label("rmin");
                    ui.add(
                        egui::DragValue::new(&mut self.ft.rmin)
                            .speed(0.05)
                            .range(0.0..=4.0),
                    );
                    ui.end_row();
                    ui.label("kmax");
                    ui.add(
                        egui::DragValue::new(&mut self.ft.kmax)
                            .speed(0.1)
                            .range(6.0..=20.0),
                    );
                    ui.label("rmax");
                    ui.add(
                        egui::DragValue::new(&mut self.ft.rmax)
                            .speed(0.05)
                            .range(1.0..=8.0),
                    );
                    ui.end_row();
                    ui.label("dk");
                    ui.add(
                        egui::DragValue::new(&mut self.ft.dk)
                            .speed(0.1)
                            .range(0.0..=4.0),
                    );
                    ui.label("dr");
                    ui.add(
                        egui::DragValue::new(&mut self.ft.dr)
                            .speed(0.05)
                            .range(0.0..=2.0),
                    );
                    ui.end_row();
                    ui.label("kweight");
                    ui.add(egui::DragValue::new(&mut self.ft.kweight).range(0..=4));
                    ui.label("fit space");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.ft.fitspace, FitSpace::K, "k");
                        ui.selectable_value(&mut self.ft.fitspace, FitSpace::R, "R");
                        ui.selectable_value(&mut self.ft.fitspace, FitSpace::Q, "q");
                    });
                    ui.end_row();
                    ui.label("k window");
                    window_combo(ui, "feffit_kwin", &mut self.ft.kwindow);
                    ui.label("R window");
                    window_combo(ui, "feffit_rwin", &mut self.ft.rwindow);
                    ui.end_row();
                });
        });

        // --- Fit mode (the original's "Fit" dropdown: fit / no fit / only FT) --
        ui.horizontal(|ui| {
            ui.label("Fit")
                .on_hover_text("Run: No fit = forward model preview; Only FT = transform data; Fit = least-squares fit");
            egui::ComboBox::from_id_salt("feffit_fit_mode")
                .selected_text(self.fit_mode.label())
                .show_ui(ui, |ui| {
                    for m in FitMode::ALL {
                        ui.selectable_value(&mut self.fit_mode, m, m.label());
                    }
                });
        });

        // --- Paths (a "Path index" selector + the chosen path's specs) ----
        // The original shows one path's parameter block at a time, indexed by a
        // "Path index" spinner; `selected_path` drives that selection.
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("Paths");
                if ui.button("Add feff path…").clicked() {
                    action = Some(FeffitAction::AddPath);
                }
            });
            if self.paths.is_empty() {
                ui.weak("Add a feffNNNN.dat path file to fit.");
            } else {
                let n = self.paths.len();
                if self.selected_path >= n {
                    self.selected_path = n - 1;
                }
                let mut remove_path = None;
                ui.horizontal(|ui| {
                    ui.label("Path index");
                    ui.add(egui::DragValue::new(&mut self.selected_path).range(0..=n - 1));
                    let idx = self.selected_path;
                    ui.checkbox(&mut self.paths[idx].enabled, "enable");
                    if crate::widgets::delete_box(ui).clicked() {
                        remove_path = Some(idx);
                    }
                    if ui
                        .button("Init")
                        .on_hover_text("reset this path's parameters to their defaults")
                        .clicked()
                    {
                        self.paths[idx].reset_specs();
                    }
                    if ui
                        .button("Init all")
                        .on_hover_text("reset every path's parameters to their defaults")
                        .clicked()
                    {
                        for p in self.paths.iter_mut() {
                            p.reset_specs();
                        }
                    }
                });
                let idx = self.selected_path;
                {
                    let p = &mut self.paths[idx];
                    ui.weak(format!(
                        "{}  (reff={:.3}, nleg={})",
                        p.label, p.reff, p.nleg
                    ));
                    // Per-path fit parameters (the original's per-path Guess/Set
                    // block): N, σ², 3rd/4th cumulant, each an auto-named variable
                    // `<base><pathindex>` when enabled.
                    ui.add_space(2.0);
                    ui.strong("Per-path parameters");
                    for (li, (base, hint)) in PATH_LOCALS.iter().copied().enumerate() {
                        let loc = &mut p.locals[li];
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut loc.enabled, "")
                                .on_hover_text("use this parameter in the fit");
                            ui.add_enabled_ui(loc.enabled, |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(format!("{base}{}", idx + 1))
                                            .monospace(),
                                    )
                                    .selectable(false),
                                )
                                .on_hover_text(hint);
                                egui::ComboBox::from_id_salt(("ploc", idx, li))
                                    .selected_text(kind_name(loc.kind))
                                    .width(56.0)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut loc.kind,
                                            ParamKind::Vary,
                                            "guess",
                                        );
                                        ui.selectable_value(&mut loc.kind, ParamKind::Fixed, "set");
                                        ui.selectable_value(&mut loc.kind, ParamKind::Expr, "expr");
                                    });
                                match loc.kind {
                                    ParamKind::Expr => {
                                        ui.add(
                                            egui::TextEdit::singleline(&mut loc.expr)
                                                .desired_width(110.0),
                                        );
                                    }
                                    _ => {
                                        ui.add(egui::DragValue::new(&mut loc.value).speed(0.001));
                                    }
                                }
                            });
                        });
                    }
                    // Shared wiring: editable expressions over the global
                    // variables / the .dat file. The amplitude (amp·N) is derived
                    // from the N local unless a `.inp` import overrode it.
                    ui.add_space(2.0);
                    ui.strong("Wiring (shared)");
                    egui::Grid::new("feffit_path_wiring")
                        .num_columns(2)
                        .spacing([6.0, 4.0])
                        .show(ui, |ui| {
                            for (name, field) in [
                                ("degen", &mut p.degen),
                                ("e0", &mut p.e0),
                                ("deltar", &mut p.deltar),
                                ("ei", &mut p.ei),
                            ] {
                                ui.label(name);
                                ui.add(egui::TextEdit::singleline(field).desired_width(150.0));
                                ui.end_row();
                            }
                        });
                    if let Some(s02) = &p.s02_override {
                        ui.weak(format!("s02 = {s02}  (from .inp)"));
                    }
                }
                if let Some(i) = remove_path {
                    self.paths.remove(i);
                    if self.selected_path >= self.paths.len() {
                        self.selected_path = self.paths.len().saturating_sub(1);
                    }
                }
            }
        });

        // --- Global variables ---------------------------------------------
        ui.group(|ui| {
            let mut adopt = false;
            let mut undo = false;
            ui.horizontal(|ui| {
                ui.strong("Global variables");
                // The "Add ▾" menu inserts a standard, pre-named variable (so the
                // common parameters need not be remembered and typed), or a blank
                // row to name yourself. A name already in the table is disabled.
                ui.menu_button("Add ⏷", |ui| {
                    for (name, val, hint) in STANDARD_VARS {
                        let exists = self.params.iter().any(|p| p.name == name);
                        if ui
                            .add_enabled(!exists, egui::Button::new(name))
                            .on_hover_text(hint)
                            .clicked()
                        {
                            self.params.push(ParamRow::var(name, val));
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui
                        .button("Custom…")
                        .on_hover_text("add a blank row to name yourself")
                        .clicked()
                    {
                        self.params.push(ParamRow::var("new", 0.0));
                        ui.close();
                    }
                });
                // "Use fit as guess" copies the best-fit values back onto the
                // vary variables as new starting guesses; "Undo" reverts it.
                if ui
                    .add_enabled(self.result.is_some(), egui::Button::new("Use fit as guess"))
                    .on_hover_text("adopt the best-fit values as the new starting guesses")
                    .clicked()
                {
                    adopt = true;
                }
                if ui
                    .add_enabled(self.param_undo.is_some(), egui::Button::new("Undo"))
                    .on_hover_text("revert the last guess update")
                    .clicked()
                {
                    undo = true;
                }
            });
            if adopt {
                self.adopt_fit_as_guess();
            }
            if undo {
                self.undo_guess();
            }
            let mut remove_param = None;
            for (i, row) in self.params.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut row.name).desired_width(72.0));
                    egui::ComboBox::from_id_salt(("pkind", i))
                        .selected_text(kind_name(row.kind))
                        .width(64.0)
                        .show_ui(ui, |ui| {
                            // The original XAFSView's terminology: "guess" = a
                            // refined free variable (larch `vary`), "set" = held
                            // fixed (larch `fixed`).
                            ui.selectable_value(&mut row.kind, ParamKind::Vary, "guess")
                                .on_hover_text("refined free variable");
                            ui.selectable_value(&mut row.kind, ParamKind::Fixed, "set")
                                .on_hover_text("held fixed at its value");
                            ui.selectable_value(&mut row.kind, ParamKind::Expr, "expr")
                                .on_hover_text("a constraint expression");
                        });
                    match row.kind {
                        ParamKind::Vary | ParamKind::Fixed => {
                            ui.add(egui::DragValue::new(&mut row.value).speed(0.01));
                        }
                        ParamKind::Expr => {
                            ui.add(egui::TextEdit::singleline(&mut row.expr).desired_width(120.0));
                        }
                    }
                    if crate::widgets::delete_box(ui).clicked() {
                        remove_param = Some(i);
                    }
                });
            }
            if let Some(i) = remove_param {
                self.params.remove(i);
            }
        });

        // --- User defined functions ---------------------------------------
        // The original's UDF block: `%set name = expr` lines define extra named
        // constants/constraints the path and variable expressions can reference.
        ui.group(|ui| {
            ui.strong("User defined functions");
            ui.weak("one %set per line, e.g.  %set drcorr = alpha*reff");
            ui.add(
                egui::TextEdit::multiline(&mut self.user_funcs)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("%set name = expr"),
            );
        });

        ui.separator();
        ui.horizontal(|ui| {
            // "Only FT" transforms the data alone, so it needs no paths; the other
            // modes need at least one enabled path.
            let can_run = self.fit_mode == FitMode::OnlyFt || self.has_enabled_path();
            if crate::widgets::primary(ui, "Run", crate::widgets::ROW_BTN, can_run).clicked() {
                action = Some(FeffitAction::Run);
            }
            if ui
                .add_enabled(self.plot.is_some(), egui::Button::new("Send to Plot Data"))
                .on_hover_text("open the Plot Data overlay for this group")
                .clicked()
            {
                action = Some(FeffitAction::SendToPlotData);
            }
        });

        // --- Graph item (space) / Graph type (component) ------------------
        ui.horizontal(|ui| {
            ui.label("Graph item");
            for (s, lbl) in [
                (PlotSpace::Q, "Q"),
                (PlotSpace::R, "R"),
                (PlotSpace::K, "K"),
                (PlotSpace::KQ, "K+Q"),
            ] {
                if ui.selectable_value(&mut self.space, s, lbl).clicked() {
                    action.get_or_insert(FeffitAction::Replot);
                }
            }
        });
        // The part selector drives R/Q (and the q half of K+Q); pure k ignores it.
        if self.space != PlotSpace::K {
            ui.horizontal(|ui| {
                ui.label("Graph type");
                for (p, lbl) in [
                    (PlotPart::Re, "Re"),
                    (PlotPart::Im, "Im"),
                    (PlotPart::Mag, "Am"),
                    (PlotPart::Pha, "Ph"),
                ] {
                    if ui.selectable_value(&mut self.part, p, lbl).clicked() {
                        action.get_or_insert(FeffitAction::Replot);
                    }
                }
            });
        }

        // --- Feffit out data (statistics) ---------------------------------
        if let Some(res) = &self.result {
            ui.separator();
            ui.strong("Feffit out data");
            egui::Grid::new("feffit_stats")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("ind. points");
                    ui.monospace(format!("{:.1}", res.n_idp));
                    ui.end_row();
                    ui.label("variable #");
                    ui.monospace(format!("{}", res.nvarys));
                    ui.end_row();
                    ui.label("deg of free");
                    ui.monospace(format!("{}", res.nfree));
                    ui.end_row();
                    stat_row(ui, "red. χ²", res.chi2_reduced);
                    stat_row(ui, "χ²", res.chi_square);
                    stat_row(ui, "R-factor", res.rfactor);
                    stat_row(ui, "AIC", res.aic);
                    stat_row(ui, "BIC", res.bic);
                });
            ui.add_space(4.0);
            ui.strong("Best-fit variables");
            egui::Grid::new("feffit_best")
                .num_columns(2)
                .show(ui, |ui| {
                    for b in &res.best {
                        ui.monospace(&b.name);
                        ui.monospace(format!("{:.5} ± {:.5}", b.value, b.stderr));
                        ui.end_row();
                    }
                    for d in &res.derived {
                        ui.weak(&d.name);
                        ui.weak(format!("{:.5} ± {:.5}", d.value, d.stderr));
                        ui.end_row();
                    }
                });
        }

        // --- View … result reports (the original's "Feffit out data" buttons) -
        ui.add_space(4.0);
        ui.label("View");
        ui.horizontal_wrapped(|ui| {
            // Most reports need a fit result; "Fix values" reads the parameter
            // table, so it is always available.
            let have = self.result.is_some();
            for (kind, enabled) in [
                (ReportKind::Correlations, have),
                (ReportKind::FitValues, have),
                (ReportKind::FixValues, true),
                (ReportKind::FeffitSumm, have),
                (ReportKind::PathSumm, have),
                (ReportKind::ResultsSumm, have),
            ] {
                if ui
                    .add_enabled(enabled, egui::Button::new(kind.title()))
                    .clicked()
                {
                    self.report_view = Some(kind);
                }
            }
        });

        // The selected report's pop-up text window (one shared, re-titled window).
        if let Some(kind) = self.report_view {
            let text = self.report_for(kind);
            if !text_window(ui, "feffit_report", kind.title(), &text) {
                self.report_view = None;
            }
        }

        // --- Bottom file row (the original's Open log / Load / Save result) ---
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .button("Open log")
                .on_hover_text("show the last fit's full report")
                .clicked()
            {
                // Our equivalent of feffit.log is the full fit report.
                let body = self.report_text();
                self.text_view = Some((
                    "Feffit log".to_owned(),
                    if body.is_empty() {
                        "No fit has been run yet.".to_owned()
                    } else {
                        body
                    },
                ));
            }
            if ui
                .button("Load inp")
                .on_hover_text("import a UWXAFS feffit.inp (windows, variables, paths)")
                .clicked()
            {
                action = Some(FeffitAction::LoadInp);
            }
            if ui.button("Load result").clicked() {
                action = Some(FeffitAction::LoadResult);
            }
            if ui
                .add_enabled(self.result.is_some(), egui::Button::new("Save result"))
                .clicked()
            {
                action = Some(FeffitAction::SaveResult);
            }
        });

        // The ad-hoc text pop-up (loaded result file / log).
        if let Some((title, body)) = self.text_view.clone()
            && !text_window(ui, "feffit_text", &title, &body)
        {
            self.text_view = None;
        }

        action
    }
}

/// Render a titled, scrollable, monospace text pop-up. Returns whether the
/// window is still open (the user may have closed it via the title-bar ✕).
fn text_window(ui: &mut egui::Ui, id: &str, title: &str, body: &str) -> bool {
    let mut open = true;
    egui::Window::new(title)
        .id(egui::Id::new(id))
        .open(&mut open)
        .resizable(true)
        .default_size([440.0, 340.0])
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::both().show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(body).monospace())
                        .wrap_mode(egui::TextWrapMode::Extend),
                );
            });
        });
    open
}

/// A statistics grid row.
fn stat_row(ui: &mut egui::Ui, label: &str, value: f64) {
    ui.label(label);
    ui.monospace(format!("{value:.5}"));
    ui.end_row();
}

/// Combo box for choosing an FT window. Bare (no inline label) so it can sit in
/// a labelled grid cell of the "Head param." block.
fn window_combo(ui: &mut egui::Ui, salt: &str, win: &mut Window) {
    egui::ComboBox::from_id_salt(salt)
        .selected_text(window_name(*win))
        .show_ui(ui, |ui| {
            for w in [
                Window::Hanning,
                Window::Kaiser,
                Window::Parzen,
                Window::Welch,
                Window::Sine,
                Window::Gaussian,
            ] {
                ui.selectable_value(win, w, window_name(w));
            }
        });
}

/// The variable-kind labels, in the original XAFSView's terminology: a refined
/// free variable is "guess" (larch `vary`), a held value is "set" (larch
/// `fixed`), and a constraint is "expr".
fn kind_name(k: ParamKind) -> &'static str {
    match k {
        ParamKind::Vary => "guess",
        ParamKind::Fixed => "set",
        ParamKind::Expr => "expr",
    }
}

fn window_name(w: Window) -> &'static str {
    match w {
        Window::Hanning => "Hanning",
        Window::Fha => "Flat-Hanning",
        Window::Parzen => "Parzen",
        Window::Welch => "Welch",
        Window::Kaiser => "Kaiser",
        Window::Bes => "Kaiser (bes)",
        Window::Sine => "Sine",
        Window::Gaussian => "Gaussian",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn path_number_parses_feff_label() {
        assert_eq!(path_number("feff0007.dat"), Some(7));
        assert_eq!(path_number("FEFF0123.DAT"), Some(123));
        assert_eq!(path_number("dir/feff0002.dat"), Some(2));
        assert_eq!(path_number("custom_path.dat"), None);
    }

    #[test]
    fn saved_path_item_maps_keys_and_computes_reff_plus_delr() {
        let sp = SavedPath {
            number: 1,
            reff: 2.50,
            params: vec![
                ("degen".to_owned(), 12.0, 1.5),
                ("deltar".to_owned(), 0.03, 0.004),
                ("sigma2".to_owned(), 0.009, 0.0008),
            ],
        };
        // A plain key returns that parameter's (value, stderr).
        assert_eq!(sp.item("degen"), (12.0, 1.5));
        // reff+Δr ("" key): reff offset by Δr's value, carrying Δr's stderr.
        let (v, e) = sp.item("");
        assert!((v - 2.53).abs() < 1e-12, "reff+delr value {v}");
        assert!((e - 0.004).abs() < 1e-12, "reff+delr err {e}");
        // A parameter not fitted reports the (0, 0) filler.
        assert_eq!(sp.item("e0"), (0.0, 0.0));
    }

    use feffit::feffdat::FeffDatFile;
    use feffit::xasdata::{
        AutobkParams, ColumnFile, MuSpec, PreEdgeParams, XasGroup, autobk_group, build_mu,
        normalize,
    };

    // Workspace fixtures: a real Cu mu(E) and the two first-shell Cu Feff paths.
    const CU_XMU: &str = include_str!("../../feffit/tests/data/cu.xmu");
    const FEFF0001: &str = include_str!("../../feffit/tests/data/feff0001.dat");
    const FEFF0002: &str = include_str!("../../feffit/tests/data/feff0002.dat");

    /// Reduce cu.xmu to (k, chi) the same way the Autobk tab does.
    fn cu_kchi() -> (Vec<f64>, Vec<f64>) {
        let cf = ColumnFile::from_text(CU_XMU).expect("parse cu.xmu");
        let (energy, mu) = build_mu(&cf, &MuSpec::Raw { energy: 0, mu: 1 }).unwrap();
        let mut g = XasGroup::from_mu("cu", energy, mu);
        normalize(&mut g, &PreEdgeParams::default());
        autobk_group(&mut g, &AutobkParams::default(), 0.0);
        (g.k.clone().unwrap(), g.chi.clone().unwrap())
    }

    fn feffit_ui_with_paths() -> FeffitUi {
        // The default mode is `NoFit` (forward preview); the fit tests want the
        // least-squares path, so opt this fixture into `Fit`.
        let mut ui = FeffitUi {
            fit_mode: FitMode::Fit,
            ..FeffitUi::default()
        };
        ui.add_path(
            "feff0001.dat".into(),
            FeffPath::new(FeffDatFile::parse(FEFF0001)),
        );
        ui.add_path(
            "feff0002.dat".into(),
            FeffPath::new(FeffDatFile::parse(FEFF0002)),
        );
        ui
    }

    #[test]
    fn parse_spec_classifies_const_vs_expr() {
        assert!(matches!(parse_spec("  1.5 "), Spec::Const(v) if (v - 1.5).abs() < 1e-12));
        assert!(matches!(parse_spec("0"), Spec::Const(v) if v == 0.0));
        assert!(matches!(parse_spec("amp"), Spec::Expr(s) if s == "amp"));
        assert!(matches!(parse_spec("alpha*reff"), Spec::Expr(s) if s == "alpha*reff"));
    }

    #[test]
    fn parse_user_funcs_extracts_set_definitions() {
        let text = "\
            %set hbar_c = 1973\n\
            %set drcorr = alpha*reff\n\
            % bkg = true\n\
            \n\
            %setx = 5\n\
            not a directive\n\
            %SET Caps_OK = 2.5\n";
        let fns = parse_user_funcs(text);
        assert_eq!(
            fns,
            vec![
                ("hbar_c".to_owned(), "1973".to_owned()),
                ("drcorr".to_owned(), "alpha*reff".to_owned()),
                ("Caps_OK".to_owned(), "2.5".to_owned()),
            ],
            "only well-formed %set lines are taken; comments and %setx are skipped"
        );
    }

    #[test]
    fn seeded_path_wires_default_variables() {
        let row = PathRow::new("p".into(), FeffPath::new(FeffDatFile::parse(FEFF0001)));
        let spec = row.to_pathspec(0);
        // Amplitude is the shared `amp` times the per-path N (fixed at 1); e0/Δr
        // wire to the shared globals; σ² is the per-path `sig1` variable.
        assert!(matches!(spec.s02, Spec::Expr(ref s) if s == "amp*N1"));
        assert!(matches!(spec.e0, Spec::Expr(ref s) if s == "del_e0"));
        assert!(matches!(spec.deltar, Spec::Expr(ref s) if s == "alpha*reff"));
        assert!(matches!(spec.sigma2, Spec::Expr(ref s) if s == "sig1"));
        // degen comes from the file as a constant; cumulants are off (constant 0).
        assert!(matches!(spec.degen, Spec::Const(_)));
        assert!(matches!(spec.third, Spec::Const(v) if v == 0.0));
        assert!(matches!(spec.fourth, Spec::Const(v) if v == 0.0));
    }

    #[test]
    fn per_path_sigma_uses_the_path_index() {
        // The second path's σ² is a distinct variable `sig2`, so two paths refine
        // independent Debye-Waller factors.
        let row = PathRow::new("p".into(), FeffPath::new(FeffDatFile::parse(FEFF0001)));
        let spec1 = row.to_pathspec(1);
        assert!(matches!(spec1.sigma2, Spec::Expr(ref s) if s == "sig2"));
        assert!(matches!(spec1.s02, Spec::Expr(ref s) if s == "amp*N2"));
    }

    #[test]
    fn disabled_local_becomes_a_constant_and_registers_no_variable() {
        let mut row = PathRow::new("p".into(), FeffPath::new(FeffDatFile::parse(FEFF0001)));
        // Turn σ² off: it should fall back to its constant value, not `sig1`.
        row.locals[LOC_SIG].enabled = false;
        row.locals[LOC_SIG].value = 0.005;
        let spec = row.to_pathspec(0);
        assert!(matches!(spec.sigma2, Spec::Const(v) if (v - 0.005).abs() < 1e-12));
        assert!(
            row.locals[LOC_SIG].var("sig", 0).is_none(),
            "a disabled local registers no variable"
        );
        // An expression local references existing variables, so also registers none.
        row.locals[LOC_SIG] = PathLocal {
            kind: ParamKind::Expr,
            value: 0.0,
            expr: "sigm_mcm".into(),
            enabled: true,
        };
        assert!(matches!(row.to_pathspec(0).sigma2, Spec::Expr(ref s) if s == "sigm_mcm"));
        assert!(row.locals[LOC_SIG].var("sig", 0).is_none());
    }

    #[test]
    fn config_clone_copies_config_and_fits_independently() {
        let (k, chi) = cu_kchi();
        let mut template = feffit_ui_with_paths();
        template.run(&k, &chi).expect("template fit");
        assert!(template.result().is_some());

        // The clone carries the configuration (enabled paths) but no result.
        let mut copy = template.config_clone();
        assert!(copy.result().is_none(), "clone must not copy the result");
        assert!(copy.has_enabled_path(), "clone must copy the paths");

        // It fits on its own, and — same config, same data — matches the template
        // (this is exactly the per-group batch's independent-fit guarantee).
        copy.run(&k, &chi).expect("clone fit");
        let a = template.result().unwrap();
        let b = copy.result().unwrap();
        assert_eq!(a.nvarys, b.nvarys);
        assert!(
            (a.rfactor - b.rfactor).abs() < 1e-9,
            "independent fit of the same config diverged: {} vs {}",
            a.rfactor,
            b.rfactor
        );
    }

    #[test]
    fn run_errors_without_paths() {
        // A path-requiring mode (the default "Only FT" transforms data alone).
        let mut ui = FeffitUi {
            fit_mode: FitMode::Fit,
            ..FeffitUi::default()
        };
        let (k, chi) = cu_kchi();
        assert!(ui.run(&k, &chi).is_err(), "no paths must error");
    }

    #[test]
    fn run_errors_without_chi() {
        let mut ui = feffit_ui_with_paths();
        assert!(ui.run(&[], &[]).is_err(), "empty chi must error");
    }

    #[test]
    fn run_fits_cu_first_shell() {
        let (k, chi) = cu_kchi();
        let mut ui = feffit_ui_with_paths();
        let msg = ui.run(&k, &chi).expect("fit should run");
        assert!(msg.contains("χ²ᵣ"), "status summarizes the fit: {msg}");

        let res = ui.result.as_ref().expect("result stored");
        assert!(
            (1..=4).contains(&res.info),
            "MINPACK should report success (info 1-4), got {}",
            res.info
        );
        // Shared amp/del_e0/alpha plus a per-path σ² for each of the two paths
        // (sig1, sig2); the per-path N1/N2 are fixed, so 5 vary.
        assert_eq!(
            res.nvarys, 5,
            "amp, del_e0, alpha + per-path sig1, sig2 all vary"
        );
        assert!(
            res.rfactor.is_finite() && res.rfactor < 0.5,
            "R={}",
            res.rfactor
        );
        assert!(res.chi2_reduced.is_finite());

        // The shared globals and both per-path σ² must appear in the best-fit table.
        for name in ["amp", "del_e0", "alpha", "sig1", "sig2"] {
            assert!(
                res.best.iter().any(|b| b.name == name),
                "missing best-fit var {name}"
            );
        }
        // amp (S0²) should land in a physical range for Cu.
        let amp = res.best.iter().find(|b| b.name == "amp").unwrap().value;
        assert!(
            (0.3..1.5).contains(&amp),
            "amp out of physical range: {amp}"
        );

        // Plot arrays populated and co-indexed in R-space.
        let plot = ui.plot().expect("plot stored");
        assert!(!plot.data.r.is_empty());
        assert_eq!(plot.data.r.len(), plot.data.chir_mag.len());
        assert_eq!(plot.model.r.len(), plot.model.chir_mag.len());
        assert_eq!(plot.data_k.len(), plot.model_chi.len());
        assert!(plot.has_model, "Fit mode carries a model");
    }

    #[test]
    fn default_mode_is_only_ft() {
        // The original XAFSView's "Fit" dropdown opens on "Only FT".
        assert_eq!(FeffitUi::default().fit_mode, FitMode::OnlyFt);
    }

    #[test]
    fn no_fit_forward_model_without_optimising() {
        let (k, chi) = cu_kchi();
        let mut ui = feffit_ui_with_paths();
        ui.fit_mode = FitMode::NoFit;
        let msg = ui.run(&k, &chi).expect("no-fit eval should run");
        assert!(msg.starts_with("No fit"), "status names the mode: {msg}");
        // No optimisation → no statistics, but a model is built and plotted.
        assert!(ui.result().is_none(), "no fit must not produce a result");
        let plot = ui.plot().expect("forward model plotted");
        assert!(plot.has_model, "no fit carries a forward model");
        assert_eq!(plot.data_k.len(), plot.model_chi.len());
        assert!(!plot.model.r.is_empty());
    }

    #[test]
    fn only_ft_transforms_data_without_paths() {
        let (k, chi) = cu_kchi();
        // No paths added — "Only FT" must still run (it transforms the data alone).
        let mut ui = FeffitUi {
            fit_mode: FitMode::OnlyFt,
            ..FeffitUi::default()
        };
        let msg = ui.run(&k, &chi).expect("only-FT should run without paths");
        assert!(msg.starts_with("Only FT"), "status names the mode: {msg}");
        assert!(ui.result().is_none(), "only FT produces no fit result");
        let plot = ui.plot().expect("data transform plotted");
        assert!(!plot.has_model, "only FT has no model");
        assert!(plot.model_chi.is_empty(), "only FT has no model chi");
        assert!(!plot.data.r.is_empty(), "data was transformed to R-space");
        assert_eq!(plot.data.r.len(), plot.data.chir_mag.len());
    }

    #[test]
    fn kq_series_pairs_k_and_q_curves() {
        let (k, chi) = cu_kchi();
        let mut ui = feffit_ui_with_paths();
        ui.run(&k, &chi).expect("fit");
        let plot = ui.plot().expect("plot stored");
        let ((kx, kd, km), (qx, qd, qm)) = plot.kq_series(PlotPart::Mag);
        assert!(!kx.is_empty() && !qx.is_empty(), "both halves present");
        assert_eq!(kx.len(), kd.len());
        assert_eq!(kx.len(), km.len());
        assert_eq!(qx.len(), qd.len());
        assert_eq!(qx.len(), qm.len());
    }

    #[test]
    fn report_for_needs_a_fit_except_fix_values() {
        let ui = FeffitUi::default();
        // Fix values read the parameter table — available without a fit.
        assert!(
            ui.report_for(ReportKind::FixValues)
                .contains("Fixed variables"),
            "fix values render without a fit"
        );
        for kind in [
            ReportKind::FitValues,
            ReportKind::Correlations,
            ReportKind::PathSumm,
            ReportKind::ResultsSumm,
            ReportKind::FeffitSumm,
        ] {
            assert!(
                ui.report_for(kind).contains("Run a fit first"),
                "{} should require a fit",
                kind.title()
            );
        }
    }

    #[test]
    fn report_for_after_fit_has_sections() {
        let (k, chi) = cu_kchi();
        let mut ui = feffit_ui_with_paths();
        ui.run(&k, &chi).expect("fit");
        assert!(
            ui.report_for(ReportKind::FitValues).contains("amp"),
            "fit values list the varied parameters"
        );
        assert!(
            ui.report_for(ReportKind::ResultsSumm)
                .contains("reduced chi^2"),
            "results summary carries the statistics"
        );
        assert!(
            ui.report_for(ReportKind::PathSumm).contains("path "),
            "path summary groups by path"
        );
        assert!(
            ui.report_for(ReportKind::Correlations)
                .contains("Correlations"),
            "correlations report has a header"
        );
    }

    const INP_SAMPLE: &str = r#"%%  feffit.inp
 title  = C:\XAFSView\Feffit\Dummy67.chi    % output
 data   = C:\XAFSView\Autobk\Dummy67k.chi   % input
 Nofit = false
   rmin    = 1.00  rmax    = 3.50  dr   = 0.20
   kmin    = 3.00  kmax    = 12.00  dk   = 0.50
   iwindo  = 1
   kweight = 3
 end
%set  hbar_c = 1973
   set           s02 =       0.901940
%   guess             temp =       0.000000
      guess      e1      =      1.58952200
      guess      delr1      =      -0.00658600
      set      N1      =      1.00000000
      guess      sig1      =      0.00470500
        path    1       ..\Feff8\feff0001.dat
        Id      1       -, r=0.0000, amp=0.0, deg=0, nleg=0
        e0      1       e1
        delR    1       delr1
        s02     1       s02 * N1
        sigma2  1       sig1
        path    2       ..\Feff8\feff0002.dat
        e0      2       e1
        delR    2       delr2
        s02     2       s02 * N1
        sigma2  2       sig2
"#;

    #[test]
    fn parse_feffit_inp_extracts_windows_vars_and_active_paths() {
        let inp = parse_feffit_inp(INP_SAMPLE);

        // Window parameters (multiple `key = value` per line).
        assert_eq!(inp.kmin, Some(3.0));
        assert_eq!(inp.kmax, Some(12.0));
        assert_eq!(inp.dk, Some(0.5));
        assert_eq!(inp.rmin, Some(1.0));
        assert_eq!(inp.rmax, Some(3.5));
        assert_eq!(inp.dr, Some(0.2));
        assert_eq!(inp.kweight, Some(3));
        assert_eq!(inp.iwindo, Some(1));
        assert_eq!(inp.nofit, Some(false));

        // `%set` is a user function, not a comment.
        assert!(inp.user_funcs.contains("hbar_c"));

        // set → fixed, guess → free; the `%`-commented `temp` is skipped.
        assert!(
            inp.vars
                .iter()
                .any(|v| v.name == "s02" && !v.guess && (v.value - 0.901940).abs() < 1e-9),
            "set s02 is a fixed variable"
        );
        assert!(
            inp.vars.iter().any(|v| v.name == "e1" && v.guess),
            "guess e1 is a free variable"
        );
        assert!(
            inp.vars.iter().any(|v| v.name == "N1" && !v.guess),
            "set N1 is a fixed variable"
        );
        assert!(
            !inp.vars.iter().any(|v| v.name == "temp"),
            "the %-commented guess is disabled"
        );

        // Two path entries, sorted by number, with their expression overrides.
        assert_eq!(inp.paths.len(), 2);
        assert_eq!(inp.paths[0].number, 1);
        assert!(inp.paths[0].file.contains("feff0001.dat"));
        assert_eq!(inp.paths[0].e0.as_deref(), Some("e1"));
        assert_eq!(inp.paths[0].deltar.as_deref(), Some("delr1"));
        assert_eq!(inp.paths[0].s02.as_deref(), Some("s02 * N1"));
        assert_eq!(inp.paths[0].sigma2.as_deref(), Some("sig1"));
        assert_eq!(inp.paths[1].number, 2);
        assert!(inp.paths[1].file.contains("feff0002.dat"));
        assert_eq!(inp.paths[1].deltar.as_deref(), Some("delr2"));
    }

    #[test]
    fn apply_inp_replaces_windows_mode_and_variables() {
        let inp = parse_feffit_inp(INP_SAMPLE);
        let mut ui = FeffitUi::default();
        ui.apply_inp(&inp);

        assert_eq!(ui.fit_mode, FitMode::Fit, "Nofit=false → Fit mode");
        assert_eq!(ui.ft.kmin, 3.0);
        assert_eq!(ui.ft.rmax, 3.5);
        assert_eq!(ui.ft.kweight, 3);
        // Variables become parameter rows (set→Fixed, guess→Vary); paths are
        // cleared for the app to reload from the `.dat` files.
        assert!(
            ui.params
                .iter()
                .any(|p| p.name == "e1" && p.kind == ParamKind::Vary)
        );
        assert!(
            ui.params
                .iter()
                .any(|p| p.name == "s02" && p.kind == ParamKind::Fixed)
        );
        assert!(ui.paths.is_empty(), "paths cleared for app-side reload");
    }

    #[test]
    fn add_inp_path_keeps_the_inp_expressions() {
        // The `.inp` carries its own per-path variables (e1/delr1/sig1, amplitude
        // `s02 * N1`); add_inp_path must preserve those expressions verbatim
        // rather than substitute the default amp·N / per-path locals.
        let inp = parse_feffit_inp(INP_SAMPLE);
        let mut ui = FeffitUi::default();
        ui.apply_inp(&inp);
        ui.add_inp_path(
            "feff0001.dat".into(),
            FeffPath::new(FeffDatFile::parse(FEFF0001)),
            &inp.paths[0],
        );
        let spec = ui.paths[0].to_pathspec(0);
        assert!(matches!(spec.e0, Spec::Expr(ref s) if s == "e1"));
        assert!(matches!(spec.deltar, Spec::Expr(ref s) if s == "delr1"));
        assert!(matches!(spec.s02, Spec::Expr(ref s) if s == "s02 * N1"));
        assert!(matches!(spec.sigma2, Spec::Expr(ref s) if s == "sig1"));
        // The amplitude override disables the unused N local (no stray variable).
        assert!(
            ui.paths[0].locals[LOC_N].var("N", 0).is_none(),
            "the .inp drives amplitude, so N registers nothing"
        );
    }

    #[test]
    fn adopt_fit_as_guess_updates_vary_values_and_undo_reverts() {
        let vary_values = |ui: &FeffitUi| -> Vec<(String, f64)> {
            ui.params
                .iter()
                .filter(|r| r.kind == ParamKind::Vary)
                .map(|r| (r.name.clone(), r.value))
                .collect()
        };

        let (k, chi) = cu_kchi();
        let mut ui = feffit_ui_with_paths();
        let before = vary_values(&ui);
        ui.run(&k, &chi).expect("fit");

        let updated = ui.adopt_fit_as_guess();
        assert!(
            updated > 0,
            "at least one vary variable adopted a best-fit value"
        );
        assert_ne!(
            before,
            vary_values(&ui),
            "guesses changed after adopting the fit"
        );

        assert!(ui.undo_guess(), "undo restores the snapshot");
        assert_eq!(
            before,
            vary_values(&ui),
            "undo brings back the starting guesses"
        );
        assert!(!ui.undo_guess(), "no snapshot left after undo");
    }

    #[test]
    fn adopt_fit_as_guess_without_a_result_is_a_noop() {
        let mut ui = FeffitUi::default();
        assert_eq!(ui.adopt_fit_as_guess(), 0);
        assert!(!ui.undo_guess(), "no snapshot recorded without a fit");
    }

    #[test]
    fn standard_vars_cover_the_shared_wiring() {
        // The "Add ▾" menu offers the shared global variables the default path
        // wiring references; σ² is per-path now, not a global.
        let offered: Vec<&str> = STANDARD_VARS.iter().map(|(n, _, _)| *n).collect();
        let row = PathRow::new("p".into(), FeffPath::new(FeffDatFile::parse(FEFF0001)));
        let spec = row.to_pathspec(0);
        let wiring = format!("{:?} {:?} {:?}", spec.s02, spec.e0, spec.deltar);
        for needed in ["amp", "del_e0", "alpha"] {
            assert!(offered.contains(&needed), "Add menu offers {needed}");
            assert!(
                wiring.contains(needed),
                "default path wiring references {needed}"
            );
        }
        assert!(
            !offered.contains(&"sig2"),
            "σ² is a per-path local, not a global"
        );
        // The default variable seed is exactly the three shared params (in order).
        let ui = FeffitUi::default();
        let seeded: Vec<&str> = ui.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(seeded, ["amp", "del_e0", "alpha"]);
    }

    #[test]
    fn saved_paths_number_and_carry_fitted_items() {
        let (k, chi) = cu_kchi();
        let mut ui = feffit_ui_with_paths();
        ui.run(&k, &chi).expect("fit should run");

        let saved = ui.saved_paths();
        // Two enabled paths, numbered from their feff0001/feff0002 labels.
        assert_eq!(saved.len(), 2, "both enabled paths are saved");
        assert_eq!(saved[0].number, 1);
        assert_eq!(saved[1].number, 2);
        for sp in &saved {
            assert!(
                sp.reff > 0.0,
                "reff carried from the feff file: {}",
                sp.reff
            );
            // The standard wiring fits degen, Δr, and σ² on each path.
            let (degen, _) = sp.item("degen");
            assert!(degen > 0.0, "degen present: {degen}");
            let (delr, _) = sp.item("deltar");
            // reff+Δr is the bond distance: reff offset by the fitted Δr.
            let (bond, _) = sp.item("");
            assert!(
                (bond - (sp.reff + delr)).abs() < 1e-9,
                "reff+delr = reff + Δr: {bond} vs {}",
                sp.reff + delr
            );
        }
    }
}
