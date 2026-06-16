//! The **multi-FEFFIT batch** window: fit several groups, each with its own
//! independent path/parameter configuration.
//!
//! Every included group carries a full [`FeffitUi`] config of its own, seeded
//! from the Feffit tab as a template and then editable per group (different
//! paths, variables, or transform windows). "Run all" fits each group
//! independently (one [`feffit`](feffit::feffit) call per group, *not* a joint
//! multi-dataset fit) and the results are tabulated; the window's own plot shows
//! the selected group's data-vs-model curve.
//!
//! The window is self-contained except for the file dialog used to add a path,
//! which it bubbles up as [`BatchAction::AddPath`] for the app to service.

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use siplot::{Plot1D, YAxis};
use xasdata::XasGroup;

use crate::feffit_ui::{FeffitAction, FeffitUi};

const BLUE: Color32 = Color32::from_rgb(0x1f, 0x77, 0xb4);
const RED: Color32 = Color32::from_rgb(0xd6, 0x27, 0x28);

/// One group's independent fit configuration, result, and last run status.
struct GroupFit {
    /// Index of this group in the session's group list (stable: groups only
    /// ever get appended).
    group_idx: usize,
    label: String,
    ui: FeffitUi,
    /// Outcome of the most recent run (`Ok(summary)` / `Err(message)`), or
    /// `None` if not yet run.
    status: Option<Result<String, String>>,
}

/// What the batch window needs the app to do this frame.
pub enum BatchAction {
    /// Open a file dialog and add the chosen Feff path(s) to the config at this
    /// index (into the batch's own config list).
    AddPath(usize),
}

/// The multi-FEFFIT batch window: a per-group config list, a shared template
/// seed, run-all, a results table, and its own data-vs-model plot.
pub struct FeffitBatch {
    /// Whether the window is shown.
    pub open: bool,
    plot: Plot1D,
    configs: Vec<GroupFit>,
    /// Which config is shown in the editor / plot.
    selected: usize,
    /// Whether the plot needs rebuilding from the selected config.
    dirty: bool,
}

impl FeffitBatch {
    /// Build the window with its own plot (use a distinct `PlotId` from the tabs'
    /// shared plot and the Plot Data window).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = Plot1D::new(render_state, 2);
        plot.set_graph_title("Feffit batch");
        Self {
            open: false,
            plot,
            configs: Vec::new(),
            selected: 0,
            dirty: true,
        }
    }

    /// Add a loaded Feff path to the config at `idx` (the app calls this after a
    /// file dialog services a [`BatchAction::AddPath`]).
    pub fn add_path_to(&mut self, idx: usize, label: String, path: feffdat::FeffPath) {
        if let Some(cfg) = self.configs.get_mut(idx) {
            cfg.ui.add_path(label, path);
        }
    }

    /// Whether the group at `group_idx` is currently in the batch.
    fn position_of(&self, group_idx: usize) -> Option<usize> {
        self.configs.iter().position(|c| c.group_idx == group_idx)
    }

    /// Toggle a group's membership: remove it if present, else add it seeded from
    /// `template`. Keeps `selected` in range.
    fn toggle_group(&mut self, group_idx: usize, groups: &[XasGroup], template: &FeffitUi) {
        if let Some(pos) = self.position_of(group_idx) {
            self.configs.remove(pos);
            self.selected = self.selected.min(self.configs.len().saturating_sub(1));
        } else if let Some(g) = groups.get(group_idx) {
            self.configs.push(GroupFit {
                group_idx,
                label: g.label.clone(),
                ui: template.config_clone(),
                status: None,
            });
            self.selected = self.configs.len() - 1;
        }
        self.dirty = true;
    }

    /// Run the config at `pos` against its group's `chi(k)`, recording the
    /// status. No-op if the group or its `chi(k)` is missing.
    fn run_one(&mut self, pos: usize, groups: &[XasGroup]) {
        let Some(cfg) = self.configs.get_mut(pos) else {
            return;
        };
        match groups.get(cfg.group_idx) {
            Some(g) => match (&g.k, &g.chi) {
                (Some(k), Some(chi)) => cfg.status = Some(cfg.ui.run(k, chi)),
                _ => {
                    cfg.status = Some(Err("no chi(k) — run AUTOBK on this group first".to_owned()));
                }
            },
            None => cfg.status = Some(Err("group no longer exists".to_owned())),
        }
    }

    /// Render the window over `groups`, seeding new configs from `template` (the
    /// Feffit tab). Returns a [`BatchAction`] for app-owned work (a file dialog).
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        groups: &[XasGroup],
        template: &FeffitUi,
    ) -> Option<BatchAction> {
        if !self.open {
            return None;
        }
        let mut bubble = None;
        let mut open = self.open;
        egui::Window::new("Feffit batch (per-group)")
            .open(&mut open)
            .resizable(true)
            .default_width(900.0)
            .default_height(620.0)
            .show(ctx, |ui| {
                egui::Panel::left("feffit_batch_controls")
                    .resizable(true)
                    .default_size(380.0)
                    .show_inside(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            bubble = self.controls(ui, groups, template);
                        });
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if self.dirty {
                        self.rebuild_plot();
                        self.dirty = false;
                    }
                    self.plot.show_toolbar(ui);
                    self.plot.show(ui);
                    ui.separator();
                    self.results_table(ui);
                });
            });
        self.open = open;
        bubble
    }

    /// The left control column: membership list, seed/run-all, and the selected
    /// group's editor.
    fn controls(
        &mut self,
        ui: &mut egui::Ui,
        groups: &[XasGroup],
        template: &FeffitUi,
    ) -> Option<BatchAction> {
        let mut bubble = None;

        ui.heading("Feffit batch");
        ui.label("Each group fits independently with its own configuration.");

        ui.separator();
        ui.strong("Groups in batch");
        if groups.is_empty() {
            ui.weak("No groups loaded.");
        }
        let mut toggle = None;
        for (i, g) in groups.iter().enumerate() {
            let mut inc = self.position_of(i).is_some();
            if ui.checkbox(&mut inc, &g.label).changed() {
                toggle = Some(i);
            }
        }
        if let Some(i) = toggle {
            self.toggle_group(i, groups, template);
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.configs.is_empty(),
                    egui::Button::new("Seed all from Feffit tab"),
                )
                .on_hover_text(
                    "Reset every included group's config to the Feffit tab's current setup",
                )
                .clicked()
            {
                for cfg in &mut self.configs {
                    cfg.ui = template.config_clone();
                    cfg.status = None;
                }
                self.dirty = true;
            }
            if ui
                .add_enabled(!self.configs.is_empty(), egui::Button::new("Run all"))
                .clicked()
            {
                for pos in 0..self.configs.len() {
                    self.run_one(pos, groups);
                }
                self.dirty = true;
            }
        });

        if self.configs.is_empty() {
            return bubble;
        }

        ui.separator();
        self.selected = self.selected.min(self.configs.len() - 1);
        let sel_label = self.configs[self.selected].label.clone();
        egui::ComboBox::from_label("Edit group")
            .selected_text(sel_label)
            .show_ui(ui, |ui| {
                for (ci, cfg) in self.configs.iter().enumerate() {
                    ui.selectable_value(&mut self.selected, ci, &cfg.label);
                }
            });

        ui.separator();
        let sel = self.selected;
        match self.configs[sel].ui.controls(ui) {
            Some(FeffitAction::AddPath) => bubble = Some(BatchAction::AddPath(sel)),
            Some(FeffitAction::Run) => {
                self.run_one(sel, groups);
                self.dirty = true;
            }
            Some(FeffitAction::Replot) => self.dirty = true,
            None => {}
        }

        bubble
    }

    /// Redraw the plot for the selected config's last fit (its chosen space/part).
    fn rebuild_plot(&mut self) {
        self.plot.clear();
        let Some(cfg) = self.configs.get(self.selected) else {
            return;
        };
        let Some(p) = cfg.ui.plot() else {
            return;
        };
        let (space, part) = cfg.ui.plot_selection();
        let (x, dy, my, xlabel, ylabel) = p.series(space, part);
        self.plot.set_graph_x_label(xlabel);
        self.plot.set_graph_y_label(ylabel, YAxis::Left);
        if !x.is_empty() {
            self.plot.add_curve_with_legend(&x, &dy, BLUE, "data");
            self.plot.add_curve_with_legend(&x, &my, RED, "model");
        }
    }

    /// The results grid: one row per group with its run status and key stats.
    fn results_table(&mut self, ui: &mut egui::Ui) {
        ui.strong("Results");
        if self.configs.is_empty() {
            ui.weak("Add groups to the batch and Run all.");
            return;
        }
        egui::Grid::new("feffit_batch_results")
            .striped(true)
            .num_columns(5)
            .show(ui, |ui| {
                ui.strong("group");
                ui.strong("status");
                ui.strong("χ²ᵣ");
                ui.strong("R-factor");
                ui.strong("nvarys");
                ui.end_row();
                for cfg in &self.configs {
                    ui.label(&cfg.label);
                    match &cfg.status {
                        Some(Ok(_)) => {
                            ui.monospace("ok");
                        }
                        Some(Err(e)) => {
                            ui.colored_label(RED, "error").on_hover_text(e);
                        }
                        None => {
                            ui.weak("—");
                        }
                    }
                    match cfg.ui.result() {
                        Some(r) => {
                            ui.monospace(format!("{:.4}", r.chi2_reduced));
                            ui.monospace(format!("{:.5}", r.rfactor));
                            ui.monospace(format!("{}", r.nvarys));
                        }
                        None => {
                            ui.weak("—");
                            ui.weak("—");
                            ui.weak("—");
                        }
                    }
                    ui.end_row();
                }
            });
    }
}
