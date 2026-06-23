//! XANES **linear-combination fitting** (LCF) and **principal-component
//! analysis** (PCA) windows.
//!
//! Both reduce to: pick spectra from the session, build the analysis matrix on
//! the reference spectrum's native energy grid over a fit range
//! ([`xasdata::groups2matrix`], cubic interpolation — matching larch), and call
//! the headless engine ([`xasdata::lincombo_fit`] / [`xasdata::pca_train`] +
//! [`xasdata::pca_fit`]). Each window owns its own plot. Spectra are taken
//! as normalized or flattened μ(E); a group must have been normalized first.

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use siplot::YAxis;
use xasdata::{PcaModel, XasGroup, groups2matrix, interp_cubic, lincombo_fit, pca_fit, pca_train};

const BLUE: Color32 = Color32::from_rgb(0x1f, 0x77, 0xb4);
const RED: Color32 = Color32::from_rgb(0xd6, 0x27, 0x28);
const GREEN: Color32 = Color32::from_rgb(0x2c, 0xa0, 0x2c);
const PALETTE: [Color32; 6] = [
    Color32::from_rgb(0x1f, 0x77, 0xb4),
    Color32::from_rgb(0xff, 0x7f, 0x0e),
    Color32::from_rgb(0x2c, 0xa0, 0x2c),
    Color32::from_rgb(0xd6, 0x27, 0x28),
    Color32::from_rgb(0x94, 0x67, 0xbd),
    Color32::from_rgb(0x17, 0xbe, 0xcf),
];

/// The spectrum (energy, array) for a group as normalized or flattened μ(E), or
/// `None` if it hasn't been normalized.
pub(crate) fn array_xy(g: &XasGroup, use_flat: bool) -> Option<(&[f64], &[f64])> {
    let arr = if use_flat {
        g.flat.as_ref()
    } else {
        g.norm.as_ref()
    };
    arr.map(|a| (g.energy.as_slice(), a.as_slice()))
}

/// A group selector backed by a per-group `Vec<bool>`, kept the length of the
/// session's group list. Returns true if the selection changed this frame.
fn group_checkboxes(ui: &mut egui::Ui, groups: &[XasGroup], sel: &mut [bool]) -> bool {
    let mut changed = false;
    for (i, g) in groups.iter().enumerate() {
        let normed = g.norm.is_some();
        let mut on = sel[i];
        let resp = ui.add_enabled(normed, egui::Checkbox::new(&mut on, &g.label));
        if !normed {
            resp.on_hover_text("normalize this group first");
        }
        if on != sel[i] {
            sel[i] = on;
            changed = true;
        }
    }
    changed
}

/// The "normalized vs flattened" array toggle, shared by both windows.
fn array_toggle(ui: &mut egui::Ui, use_flat: &mut bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Array:");
        changed |= ui.selectable_value(use_flat, false, "norm").clicked();
        changed |= ui.selectable_value(use_flat, true, "flat").clicked();
    });
    changed
}

// ===========================================================================
// LCF
// ===========================================================================

/// One LCF result: the fit on the common grid plus the labelled weights.
struct LcfResult {
    grid: Vec<f64>,
    target: Vec<f64>,
    yfit: Vec<f64>,
    weights: Vec<(String, f64)>,
    total: f64,
    rfactor: f64,
    redchi: f64,
}

/// The linear-combination-fitting window.
pub struct LcfWindow {
    pub open: bool,
    plot: crate::plot::Plot,
    target: Option<usize>,
    components: Vec<bool>,
    use_flat: bool,
    emin: f64,
    emax: f64,
    seeded: bool,
    result: Option<LcfResult>,
    error: Option<String>,
    dirty: bool,
}

impl LcfWindow {
    /// Build the window with its own plot (`PlotId` 3).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::Plot::new(render_state, 3);
        plot.set_graph_title("LCF");
        Self {
            open: false,
            plot,
            target: None,
            components: Vec::new(),
            use_flat: true,
            emin: 0.0,
            emax: 0.0,
            seeded: false,
            result: None,
            error: None,
            dirty: true,
        }
    }

    /// Render the window over `groups`.
    pub fn show(&mut self, ctx: &egui::Context, groups: &[XasGroup]) {
        if self.components.len() != groups.len() {
            self.components.resize(groups.len(), false);
        }
        if !self.open {
            return;
        }
        let mut open = self.open;
        crate::window::detached(
            ctx,
            "lcf",
            "LCF — linear combination",
            &mut open,
            [820.0, 560.0],
            |ui| {
                egui::Panel::left("lcf_controls")
                    .resizable(true)
                    .default_size(280.0)
                    .show_inside(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, groups));
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if self.dirty {
                        self.rebuild_plot(groups);
                        self.dirty = false;
                    }
                    crate::plot::show(&mut self.plot, ui);
                    ui.separator();
                    self.results(ui);
                });
            },
        );
        self.open = open;
    }

    fn controls(&mut self, ui: &mut egui::Ui, groups: &[XasGroup]) {
        ui.heading("LCF");
        ui.label("Fit one spectrum as a weighted sum of standards.");

        ui.separator();
        let target_label = self
            .target
            .and_then(|i| groups.get(i))
            .map(|g| g.label.clone())
            .unwrap_or_else(|| "— pick target —".to_owned());
        let mut target_changed = false;
        egui::ComboBox::from_label("Target")
            .selected_text(target_label)
            .show_ui(ui, |ui| {
                for (i, g) in groups.iter().enumerate() {
                    if g.norm.is_some() {
                        target_changed |= ui
                            .selectable_value(&mut self.target, Some(i), &g.label)
                            .clicked();
                    }
                }
            });
        if target_changed {
            self.dirty = true;
        }
        // Seed the fit range from the target's energy span the first time.
        if !self.seeded
            && let Some((e, _)) = self
                .target
                .and_then(|i| groups.get(i))
                .and_then(|g| array_xy(g, self.use_flat))
            && e.len() >= 2
        {
            self.emin = e[0];
            self.emax = e[e.len() - 1];
            self.seeded = true;
        }

        if array_toggle(ui, &mut self.use_flat) {
            self.dirty = true;
        }
        ui.horizontal(|ui| {
            ui.label("E range");
            ui.add(egui::DragValue::new(&mut self.emin).speed(1.0));
            ui.add(egui::DragValue::new(&mut self.emax).speed(1.0));
        });

        ui.separator();
        ui.strong("Standards");
        if group_checkboxes(ui, groups, &mut self.components) {
            self.dirty = true;
        }

        ui.separator();
        let ncomp = self.selected_components(groups).len();
        let can_run = self.target.is_some() && ncomp >= 2;
        if ui
            .add_enabled(can_run, egui::Button::new("Run LCF"))
            .clicked()
        {
            self.run(groups);
        }
        if !can_run {
            ui.weak("Pick a target and ≥2 standards.");
        }
        if let Some(e) = &self.error {
            ui.colored_label(RED, e);
        }
    }

    /// Component group indices that are checked and not the target.
    fn selected_components(&self, groups: &[XasGroup]) -> Vec<usize> {
        self.components
            .iter()
            .enumerate()
            .filter(|(i, on)| **on && Some(*i) != self.target && groups.get(*i).is_some())
            .map(|(i, _)| i)
            .collect()
    }

    fn run(&mut self, groups: &[XasGroup]) {
        self.error = None;
        let Some(ti) = self.target else { return };
        let Some((te, ta)) = groups.get(ti).and_then(|g| array_xy(g, self.use_flat)) else {
            self.error = Some("target not normalized".to_owned());
            return;
        };
        let comp_idx = self.selected_components(groups);
        let mut labels = Vec::new();
        let mut curves: Vec<(&[f64], &[f64])> = vec![(te, ta)];
        for &ci in &comp_idx {
            let g = &groups[ci];
            match array_xy(g, self.use_flat) {
                Some(xy) => {
                    curves.push(xy);
                    labels.push(g.label.clone());
                }
                None => {
                    self.error = Some(format!("standard '{}' not normalized", g.label));
                    return;
                }
            }
        }
        let Some((grid, rows)) = groups2matrix(&curves, self.emin, self.emax) else {
            self.error = Some("empty fit range / no overlap".to_owned());
            return;
        };
        let comps: Vec<Vec<f64>> = rows[1..].to_vec();
        let lc = lincombo_fit(&rows[0], &comps, &Default::default());
        let weights = labels.into_iter().zip(lc.weights.iter().copied()).collect();
        self.result = Some(LcfResult {
            grid,
            target: rows[0].clone(),
            yfit: lc.yfit,
            weights,
            total: lc.total,
            rfactor: lc.rfactor,
            redchi: lc.redchi,
        });
        self.dirty = true;
    }

    fn rebuild_plot(&mut self, groups: &[XasGroup]) {
        self.plot.clear();
        self.plot.set_graph_x_label("Energy (eV)");
        self.plot
            .set_graph_y_label(if self.use_flat { "flat" } else { "norm" }, YAxis::Left);

        // After a fit: the data on the common grid, the fitted combination, and
        // the residual.
        if let Some(r) = &self.result {
            self.plot
                .add_curve_with_legend(&r.grid, &r.target, BLUE, "data");
            self.plot
                .add_curve_with_legend(&r.grid, &r.yfit, RED, "fit");
            let resid: Vec<f64> = r.target.iter().zip(&r.yfit).map(|(d, f)| d - f).collect();
            self.plot
                .add_curve_with_legend(&r.grid, &resid, GREEN, "residual");
            return;
        }

        // No fit yet: show the picked target and the selected standards on their
        // native grids so the data is visible while the fit is being set up (the
        // plot is never blank, even with a single loaded group).
        if let Some((e, a)) = self
            .target
            .and_then(|i| groups.get(i))
            .and_then(|g| array_xy(g, self.use_flat))
        {
            self.plot.add_curve_with_legend(e, a, BLUE, "target");
        }
        for (n, ci) in self.selected_components(groups).into_iter().enumerate() {
            if let Some((e, a)) = groups.get(ci).and_then(|g| array_xy(g, self.use_flat)) {
                let c = PALETTE[(n + 1) % PALETTE.len()];
                self.plot
                    .add_curve_with_legend(e, a, c, groups[ci].label.clone());
            }
        }
    }

    fn results(&mut self, ui: &mut egui::Ui) {
        ui.strong("Weights");
        let Some(r) = &self.result else {
            ui.weak("Run a fit to see weights.");
            return;
        };
        egui::Grid::new("lcf_weights").striped(true).show(ui, |ui| {
            for (name, w) in &r.weights {
                ui.label(name);
                ui.monospace(format!("{w:.4}"));
                ui.end_row();
            }
            ui.strong("sum");
            ui.monospace(format!("{:.4}", r.total));
            ui.end_row();
            ui.label("R-factor");
            ui.monospace(format!("{:.6}", r.rfactor));
            ui.end_row();
            ui.label("reduced χ²");
            ui.monospace(format!("{:.6}", r.redchi));
            ui.end_row();
        });
    }
}

// ===========================================================================
// PCA
// ===========================================================================

/// A trained PCA model with the grid it was trained on.
struct PcaTrained {
    grid: Vec<f64>,
    model: PcaModel,
}

/// A PCA reconstruction of one unknown.
struct PcaReco {
    label: String,
    ydat: Vec<f64>,
    yfit: Vec<f64>,
}

/// The principal-component-analysis window.
pub struct PcaWindow {
    pub open: bool,
    plot: crate::plot::Plot,
    training: Vec<bool>,
    use_flat: bool,
    emin: f64,
    emax: f64,
    seeded: bool,
    ncomps: usize,
    target: Option<usize>,
    trained: Option<PcaTrained>,
    reco: Option<PcaReco>,
    error: Option<String>,
    dirty: bool,
}

impl PcaWindow {
    /// Build the window with its own plot (`PlotId` 4).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::Plot::new(render_state, 4);
        plot.set_graph_title("PCA");
        Self {
            open: false,
            plot,
            training: Vec::new(),
            use_flat: true,
            emin: 0.0,
            emax: 0.0,
            seeded: false,
            ncomps: 2,
            target: None,
            trained: None,
            reco: None,
            error: None,
            dirty: true,
        }
    }

    /// Render the window over `groups`.
    pub fn show(&mut self, ctx: &egui::Context, groups: &[XasGroup]) {
        if self.training.len() != groups.len() {
            self.training.resize(groups.len(), false);
        }
        if !self.open {
            return;
        }
        let mut open = self.open;
        crate::window::detached(
            ctx,
            "pca",
            "PCA — principal components",
            &mut open,
            [820.0, 560.0],
            |ui| {
                egui::Panel::left("pca_controls")
                    .resizable(true)
                    .default_size(280.0)
                    .show_inside(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, groups));
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if self.dirty {
                        self.rebuild_plot(groups);
                        self.dirty = false;
                    }
                    crate::plot::show(&mut self.plot, ui);
                    ui.separator();
                    self.results(ui);
                });
            },
        );
        self.open = open;
    }

    fn controls(&mut self, ui: &mut egui::Ui, groups: &[XasGroup]) {
        ui.heading("PCA");
        ui.label("Decompose a training set; reconstruct an unknown.");

        // Seed range from the first normalized group.
        if !self.seeded
            && let Some((e, _)) = groups.iter().find_map(|g| array_xy(g, self.use_flat))
            && e.len() >= 2
        {
            self.emin = e[0];
            self.emax = e[e.len() - 1];
            self.seeded = true;
        }

        if array_toggle(ui, &mut self.use_flat) {
            self.dirty = true;
        }
        ui.horizontal(|ui| {
            ui.label("E range");
            ui.add(egui::DragValue::new(&mut self.emin).speed(1.0));
            ui.add(egui::DragValue::new(&mut self.emax).speed(1.0));
        });

        ui.separator();
        ui.strong("Training set");
        if group_checkboxes(ui, groups, &mut self.training) {
            self.dirty = true;
        }

        ui.separator();
        let ntrain = self.training.iter().filter(|b| **b).count();
        if ui
            .add_enabled(ntrain >= 2, egui::Button::new("Train"))
            .clicked()
        {
            self.train(groups);
        }
        if ntrain < 2 {
            ui.weak("Select ≥2 training spectra.");
        }

        if let Some(t) = &self.trained {
            let maxc = t.model.components.len().max(1);
            ui.add(egui::Slider::new(&mut self.ncomps, 1..=maxc).text("components"));
            let target_label = self
                .target
                .and_then(|i| groups.get(i))
                .map(|g| g.label.clone())
                .unwrap_or_else(|| "— pick unknown —".to_owned());
            egui::ComboBox::from_label("Unknown")
                .selected_text(target_label)
                .show_ui(ui, |ui| {
                    for (i, g) in groups.iter().enumerate() {
                        if g.norm.is_some() {
                            ui.selectable_value(&mut self.target, Some(i), &g.label);
                        }
                    }
                });
            if ui
                .add_enabled(self.target.is_some(), egui::Button::new("Fit unknown"))
                .clicked()
            {
                self.fit_unknown(groups);
            }
        }
        if let Some(e) = &self.error {
            ui.colored_label(RED, e);
        }
    }

    fn train(&mut self, groups: &[XasGroup]) {
        self.error = None;
        self.reco = None;
        let mut curves: Vec<(&[f64], &[f64])> = Vec::new();
        for (i, on) in self.training.iter().enumerate() {
            if *on && let Some(xy) = groups.get(i).and_then(|g| array_xy(g, self.use_flat)) {
                curves.push(xy);
            }
        }
        let Some((grid, rows)) = groups2matrix(&curves, self.emin, self.emax) else {
            self.error = Some("empty range / no overlap".to_owned());
            return;
        };
        let model = pca_train(&rows);
        self.ncomps = model.nsig.max(1).min(model.components.len().max(1));
        self.trained = Some(PcaTrained { grid, model });
        self.dirty = true;
    }

    fn fit_unknown(&mut self, groups: &[XasGroup]) {
        self.error = None;
        let Some(t) = &self.trained else { return };
        let Some(ti) = self.target else { return };
        let Some((e, a)) = groups.get(ti).and_then(|g| array_xy(g, self.use_flat)) else {
            self.error = Some("unknown not normalized".to_owned());
            return;
        };
        // larch `pca_fit`: slice the unknown to the model's fit range on its own
        // native grid, then cubic-interpolate onto the model's grid.
        let Some((xdat, rows)) = groups2matrix(&[(e, a)], self.emin, self.emax) else {
            self.error = Some("empty range / no overlap".to_owned());
            return;
        };
        let on_grid = interp_cubic(&xdat, &rows[0], &t.grid);
        let fit = pca_fit(&on_grid, &t.model, self.ncomps, true);
        self.reco = Some(PcaReco {
            label: groups[ti].label.clone(),
            ydat: fit.ydat,
            yfit: fit.yfit,
        });
        self.dirty = true;
    }

    fn rebuild_plot(&mut self, groups: &[XasGroup]) {
        self.plot.clear();
        self.plot.set_graph_x_label("Energy (eV)");
        let Some(t) = &self.trained else {
            // Not trained yet: show the selected training spectra on their native
            // grids so the input set is visible (there is no decomposition to draw
            // until Train is pressed).
            self.plot
                .set_graph_y_label(if self.use_flat { "flat" } else { "norm" }, YAxis::Left);
            let train_idx: Vec<usize> = self
                .training
                .iter()
                .enumerate()
                .filter(|(_, on)| **on)
                .map(|(i, _)| i)
                .collect();
            for (n, i) in train_idx.into_iter().enumerate() {
                if let Some((e, a)) = groups.get(i).and_then(|g| array_xy(g, self.use_flat)) {
                    let c = PALETTE[n % PALETTE.len()];
                    self.plot
                        .add_curve_with_legend(e, a, c, groups[i].label.clone());
                }
            }
            return;
        };
        if let Some(r) = &self.reco {
            // Unknown vs reconstruction.
            self.plot.set_graph_y_label("PCA fit", YAxis::Left);
            self.plot
                .add_curve_with_legend(&t.grid, &r.ydat, BLUE, r.label.as_str());
            self.plot
                .add_curve_with_legend(&t.grid, &r.yfit, RED, "reconstruction");
        } else {
            // The mean and leading components.
            self.plot.set_graph_y_label("component", YAxis::Left);
            self.plot
                .add_curve_with_legend(&t.grid, &t.model.mean, Color32::GRAY, "mean");
            let show = t.model.nsig.max(1).min(t.model.components.len());
            for (i, comp) in t.model.components.iter().take(show).enumerate() {
                let c = PALETTE[i % PALETTE.len()];
                self.plot
                    .add_curve_with_legend(&t.grid, comp, c, format!("PC{}", i + 1));
            }
        }
    }

    fn results(&mut self, ui: &mut egui::Ui) {
        ui.strong("Variances / IND");
        let Some(t) = &self.trained else {
            ui.weak("Train to see the scree statistics.");
            return;
        };
        ui.monospace(format!(
            "suggested significant components: {}",
            t.model.nsig
        ));
        egui::Grid::new("pca_scree").striped(true).show(ui, |ui| {
            ui.strong("PC");
            ui.strong("variance");
            ui.strong("IND");
            ui.end_row();
            let n = t.model.variances.len().min(8);
            for i in 0..n {
                ui.monospace(format!("{}", i + 1));
                ui.monospace(format!("{:.5}", t.model.variances[i]));
                ui.monospace(format!(
                    "{:.4e}",
                    t.model.ind.get(i).copied().unwrap_or(0.0)
                ));
                ui.end_row();
            }
        });
    }
}
