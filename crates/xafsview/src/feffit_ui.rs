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
use feffdat::FeffPath;
use feffit::{
    DataSet, FeffitResult, FitDataSet, FitSpace, PATH_PNAMES, PathSpec, Spec, Transform,
    XafsOutput, feffit,
};
use params::Parameters;
use xasdata::Window;

/// What the FEFFIT controls need the app to do this frame.
pub enum FeffitAction {
    /// Open a file dialog and add the chosen Feff path file(s).
    AddPath,
    /// Run the fit against the current group's `chi(k)`.
    Run,
    /// Redraw the data-vs-model plot (the plot space/part changed).
    Replot,
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

/// One loaded Feff path plus its eight editable parameter-spec fields (in
/// [`PATH_PNAMES`] order: degen, s02, e0, ei, deltar, sigma2, third, fourth).
#[derive(Clone)]
struct PathRow {
    label: String,
    reff: f64,
    nleg: usize,
    enabled: bool,
    path: FeffPath,
    specs: [String; 8],
}

impl PathRow {
    /// Seed the spec fields from a freshly loaded path with the standard
    /// first-shell wiring (degen from the file; s02/e0/Δr/σ² bound to the
    /// default variables) so a loaded path is ready to fit.
    fn new(label: String, path: FeffPath) -> Self {
        let degen = path.feffdat.degen;
        let reff = path.feffdat.reff;
        let nleg = path.feffdat.nleg;
        Self {
            label,
            reff,
            nleg,
            enabled: true,
            path,
            specs: [
                format!("{degen}"),      // degen
                "amp".to_owned(),        // s02
                "del_e0".to_owned(),     // e0
                "0".to_owned(),          // ei
                "alpha*reff".to_owned(), // deltar
                "sig2".to_owned(),       // sigma2
                "0".to_owned(),          // third
                "0".to_owned(),          // fourth
            ],
        }
    }

    /// Parse the eight spec fields into a [`PathSpec`] (a field that parses as a
    /// number is a constant; otherwise it is an expression string).
    fn to_pathspec(&self) -> PathSpec {
        PathSpec {
            degen: parse_spec(&self.specs[0]),
            s02: parse_spec(&self.specs[1]),
            e0: parse_spec(&self.specs[2]),
            ei: parse_spec(&self.specs[3]),
            deltar: parse_spec(&self.specs[4]),
            sigma2: parse_spec(&self.specs[5]),
            third: parse_spec(&self.specs[6]),
            fourth: parse_spec(&self.specs[7]),
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
}

/// For R/Q space, which component to draw.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlotPart {
    Mag,
    Re,
    Im,
    Pha,
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
}

impl FeffitPlot {
    /// Build the `(x, data_y, model_y, x-label, y-label)` series for a given
    /// plot space and part. For k-space the part is ignored (`kʷ·χ(k)`); for
    /// R/Q the part selects magnitude / real / imag / phase.
    pub fn series(
        &self,
        space: PlotSpace,
        part: PlotPart,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, &'static str, &'static str) {
        match space {
            PlotSpace::K => {
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
    result: Option<FeffitResult>,
    plot: Option<FeffitPlot>,
}

impl Default for FeffitUi {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            // A standard first-shell starter set the seeded path specs reference.
            params: vec![
                ParamRow::var("amp", 0.9),
                ParamRow::var("del_e0", 0.0),
                ParamRow::var("alpha", 0.0),
                ParamRow::var("sig2", 0.003),
            ],
            ft: FtSettings::default(),
            space: PlotSpace::R,
            part: PlotPart::Mag,
            selected_path: 0,
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
            result: None,
            plot: None,
        }
    }

    /// The last fit result, if a fit has been run (for the batch result table).
    pub fn result(&self) -> Option<&FeffitResult> {
        self.result.as_ref()
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

    /// Add a path file the app picked from a dialog.
    pub fn add_path(&mut self, label: String, path: FeffPath) {
        self.paths.push(PathRow::new(label, path));
    }

    /// Whether any enabled path is loaded (the fit needs at least one).
    pub fn has_enabled_path(&self) -> bool {
        self.paths.iter().any(|p| p.enabled)
    }

    /// Assemble and run the fit against `data_k`/`data_chi`, storing the result
    /// and the data-vs-model plot arrays. Returns a one-line status on success
    /// or an error message.
    pub fn run(&mut self, data_k: &[f64], data_chi: &[f64]) -> Result<String, String> {
        if !self.has_enabled_path() {
            return Err("No enabled Feff paths to fit.".to_owned());
        }
        if data_k.is_empty() || data_chi.len() != data_k.len() {
            return Err("Current group has no chi(k) — run AUTOBK first.".to_owned());
        }

        let mut params = Parameters::new();
        for row in &self.params {
            match row.kind {
                ParamKind::Vary => params.add_var(&row.name, row.value),
                ParamKind::Fixed => params.add_fixed(&row.name, row.value),
                ParamKind::Expr => params.add_expr(&row.name, row.expr.trim()),
            }
        }

        let mut feff_paths = Vec::new();
        let mut specs = Vec::new();
        for row in self.paths.iter().filter(|p| p.enabled) {
            feff_paths.push(row.path.clone());
            specs.push(row.to_pathspec());
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

        let res = feffit(&mut params, &mut fds).map_err(|e| format!("fit failed: {e}"))?;

        // Model chi(k) on the data grid, and forward-FT of both data and model.
        let model_chi = fds[0].dataset.model_chi_sum();
        let out = fds[0].dataset.save_outputs(self.ft.rmax + 2.0, false);
        self.plot = Some(FeffitPlot {
            data_k: data_k.to_vec(),
            data_chi: data_chi.to_vec(),
            model_chi,
            data: out.data,
            model: out.model,
            kweight: self.ft.kweight,
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
                    if ui.small_button("✕").clicked() {
                        remove_path = Some(idx);
                    }
                });
                let idx = self.selected_path;
                {
                    let p = &mut self.paths[idx];
                    ui.weak(format!(
                        "{}  (reff={:.3}, nleg={})",
                        p.label, p.reff, p.nleg
                    ));
                    egui::Grid::new("feffit_path_specs")
                        .num_columns(2)
                        .spacing([6.0, 4.0])
                        .show(ui, |ui| {
                            for (j, name) in PATH_PNAMES.iter().enumerate() {
                                ui.label(*name);
                                ui.add(
                                    egui::TextEdit::singleline(&mut p.specs[j])
                                        .desired_width(150.0),
                                );
                                ui.end_row();
                            }
                        });
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
            ui.horizontal(|ui| {
                ui.strong("Global variables");
                if ui.button("Add").clicked() {
                    self.params.push(ParamRow::var("new", 0.0));
                }
            });
            let mut remove_param = None;
            for (i, row) in self.params.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut row.name).desired_width(72.0));
                    egui::ComboBox::from_id_salt(("pkind", i))
                        .selected_text(kind_name(row.kind))
                        .width(64.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut row.kind, ParamKind::Vary, "vary");
                            ui.selectable_value(&mut row.kind, ParamKind::Fixed, "fixed");
                            ui.selectable_value(&mut row.kind, ParamKind::Expr, "expr");
                        });
                    match row.kind {
                        ParamKind::Vary | ParamKind::Fixed => {
                            ui.add(egui::DragValue::new(&mut row.value).speed(0.01));
                        }
                        ParamKind::Expr => {
                            ui.add(egui::TextEdit::singleline(&mut row.expr).desired_width(120.0));
                        }
                    }
                    if ui.small_button("✕").clicked() {
                        remove_param = Some(i);
                    }
                });
            }
            if let Some(i) = remove_param {
                self.params.remove(i);
            }
        });

        ui.separator();
        if ui
            .add_enabled(self.has_enabled_path(), egui::Button::new("Run"))
            .clicked()
        {
            action = Some(FeffitAction::Run);
        }

        // --- Graph item (space) / Graph type (component) ------------------
        ui.horizontal(|ui| {
            ui.label("Graph item");
            for (s, lbl) in [
                (PlotSpace::Q, "Q"),
                (PlotSpace::R, "R"),
                (PlotSpace::K, "K"),
            ] {
                if ui.selectable_value(&mut self.space, s, lbl).clicked() {
                    action.get_or_insert(FeffitAction::Replot);
                }
            }
        });
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

        action
    }
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

fn kind_name(k: ParamKind) -> &'static str {
    match k {
        ParamKind::Vary => "vary",
        ParamKind::Fixed => "fixed",
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
    use feffdat::FeffDatFile;
    use xasdata::{
        AutobkParams, ColumnFile, MuSpec, PreEdgeParams, XasGroup, autobk_group, build_mu,
        normalize,
    };

    // Workspace fixtures: a real Cu mu(E) and the two first-shell Cu Feff paths.
    const CU_XMU: &str = include_str!("../../xasdata/tests/data/cu.xmu");
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
        let mut ui = FeffitUi::default();
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
    fn seeded_path_wires_default_variables() {
        let row = PathRow::new("p".into(), FeffPath::new(FeffDatFile::parse(FEFF0001)));
        let spec = row.to_pathspec();
        assert!(matches!(spec.s02, Spec::Expr(ref s) if s == "amp"));
        assert!(matches!(spec.e0, Spec::Expr(ref s) if s == "del_e0"));
        assert!(matches!(spec.deltar, Spec::Expr(ref s) if s == "alpha*reff"));
        assert!(matches!(spec.sigma2, Spec::Expr(ref s) if s == "sig2"));
        // degen comes from the file as a constant.
        assert!(matches!(spec.degen, Spec::Const(_)));
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
        let mut ui = FeffitUi::default();
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
        assert_eq!(res.nvarys, 4, "amp, del_e0, alpha, sig2 all vary");
        assert!(
            res.rfactor.is_finite() && res.rfactor < 0.5,
            "R={}",
            res.rfactor
        );
        assert!(res.chi2_reduced.is_finite());

        // The four named variables must appear in the best-fit table.
        for name in ["amp", "del_e0", "alpha", "sig2"] {
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
    }
}
