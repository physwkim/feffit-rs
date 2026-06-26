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

use std::fmt::Write as _;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use feffit::xasdata::XasGroup;
use rayon::prelude::*;
use siplot::YAxis;

use crate::feffit_ui::{FeffitAction, FeffitUi, SavedPath};
use crate::plot::{BLUE, RED};

/// The eight savable "items" of the original Save-Items dialog, each as
/// `(display/file name, FeffitResult path-parameter key)`. An empty key marks
/// the computed `reff + Δr`. Order matches the dialog (그림 1-5-3).
const SAVE_ITEMS: [(&str, &str); 8] = [
    ("e0", "e0"),
    ("delr", "deltar"),
    ("n", "degen"),
    ("sigma2", "sigma2"),
    ("third", "third"),
    ("fourth", "fourth"),
    ("ei", "ei"),
    ("reff+delr", ""),
];

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

/// Fit one config against its group's `chi(k)`, recording the status. No-op-ish
/// (records an error status) if the group or its `chi(k)` is missing. Free of
/// `&self` so "Run all" can drive many of these concurrently across configs —
/// each call mutates only its own `cfg` and reads the shared `groups`.
fn run_config(cfg: &mut GroupFit, groups: &[XasGroup]) {
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

/// What the batch window needs the app to do this frame.
pub enum BatchAction {
    /// Open a file dialog and add the chosen Feff path(s) to the config at this
    /// index (into the batch's own config list).
    AddPath(usize),
    /// Write these `(filename, content)` pairs (one per selected Save-Items
    /// item); the app picks the destination folder and reports the outcome.
    SaveItems(Vec<(String, String)>),
    /// Write these `(filename, content)` pairs — the per-group FEFFIT k/r/q
    /// `.dat`/`.fit` transforms; the app writes them to the work folder (same
    /// destination the single-fit `write_feffit_outputs` uses).
    SaveFeffitOutputs(Vec<(String, String)>),
}

/// The multi-FEFFIT batch window: a per-group config list, a shared template
/// seed, run-all, a results table, and its own data-vs-model plot.
pub struct FeffitBatch {
    /// Whether the window is shown.
    pub open: bool,
    plot: crate::plot::Plot,
    configs: Vec<GroupFit>,
    /// Which config is shown in the editor / plot.
    selected: usize,
    /// Whether the plot needs rebuilding from the selected config.
    dirty: bool,
    /// "Save Items": which of the eight items to write (in [`SAVE_ITEMS`] order).
    save_sel: [bool; 8],
    /// Inclusive path-index range written for each item.
    save_from: usize,
    save_to: usize,
}

impl FeffitBatch {
    /// Build the window with its own plot (use a distinct `PlotId` from the tabs'
    /// shared plot and the Plot Data window).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::Plot::new(render_state, 2);
        plot.set_graph_title("Feffit batch");
        Self {
            open: false,
            plot,
            configs: Vec::new(),
            selected: 0,
            dirty: true,
            // Default to the most-used items (N and the bond distance reff+Δr).
            save_sel: [false, false, true, false, false, false, false, true],
            save_from: 1,
            save_to: 1,
        }
    }

    /// Add a loaded Feff path to the config at `idx` (the app calls this after a
    /// file dialog services a [`BatchAction::AddPath`]).
    pub fn add_path_to(&mut self, idx: usize, label: String, path: feffit::feffdat::FeffPath) {
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
        if let Some(cfg) = self.configs.get_mut(pos) {
            run_config(cfg, groups);
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
        crate::window::detached(
            ctx,
            "feffit_batch",
            "Feffit batch (per-group)",
            &mut open,
            [900.0, 620.0],
            |ui| {
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
                    crate::plot::show(&mut self.plot, ui);
                    ui.separator();
                    self.results_table(ui);
                });
            },
        );
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
                // Each group's fit is independent (own config, own FFT planner,
                // reading only the shared `groups`), so fan the slow fits across
                // cores instead of running them one at a time.
                self.configs
                    .par_iter_mut()
                    .for_each(|cfg| run_config(cfg, groups));
                self.dirty = true;
            }
        });

        if self.configs.is_empty() {
            return bubble;
        }

        ui.separator();
        let any_fit = self.configs.iter().any(|c| c.ui.plot().is_some());
        if ui
            .add_enabled(any_fit, egui::Button::new("Save χ data+fit (all groups)"))
            .on_hover_text(
                "Write every fitted group's FEFFIT transforms to the work folder as \
                 <label>{k,r,q}.dat (data) and <label>{k,r,q}.fit (model).",
            )
            .clicked()
        {
            let files = self.feffit_output_files();
            if !files.is_empty() {
                bubble = Some(BatchAction::SaveFeffitOutputs(files));
            }
        }

        ui.separator();
        egui::CollapsingHeader::new("Save items")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("Items to save (per path index):");
                egui::Grid::new("save_items_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        for (i, (name, _)) in SAVE_ITEMS.iter().enumerate() {
                            ui.checkbox(&mut self.save_sel[i], *name);
                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("path index from");
                    ui.add(egui::DragValue::new(&mut self.save_from).range(1..=99));
                    ui.label("to");
                    ui.add(egui::DragValue::new(&mut self.save_to).range(1..=99));
                });
                let any_item = self.save_sel.iter().any(|s| *s);
                let any_result = self.configs.iter().any(|c| c.ui.result().is_some());
                if ui
                    .add_enabled(
                        any_item && any_result,
                        egui::Button::new("Save items to work folder"),
                    )
                    .on_hover_text(
                        "One file per item in the work folder; rows are groups, columns are \
                         path indices (value and stderr; unused paths are 0).",
                    )
                    .clicked()
                {
                    let files = self.build_saved_items();
                    if !files.is_empty() {
                        bubble = Some(BatchAction::SaveItems(files));
                    }
                }
            });

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
            // The batch window is itself a multi-group view with its own plot and
            // its own "Save Items", so the Feffit tab's single-fit Send-to-Plot /
            // Save-result / Load-result affordances are no-ops here.
            Some(FeffitAction::SendToPlotData)
            | Some(FeffitAction::SaveResult)
            | Some(FeffitAction::LoadResult) => {}
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

    /// Collect the FEFFIT k/r/q `.dat`/`.fit` files for every group that has a
    /// fit result, as `(filename, content)` pairs named from the group label.
    /// The app writes them to the work folder (the same destination the
    /// single-fit `write_feffit_outputs` uses).
    fn feffit_output_files(&self) -> Vec<(String, String)> {
        self.configs
            .iter()
            .filter_map(|cfg| cfg.ui.plot().map(|p| p.output_pairs(&cfg.label)))
            .flatten()
            .collect()
    }

    /// Build one output file per selected Save-Items item: rows are the fitted
    /// groups, columns are path indices `from..=to`. Each cell is the item's
    /// value and propagated stderr for that path, or `0` when a group's fit has
    /// no such path (the original's `n = 0` filler). Returns `(filename,
    /// content)` pairs for the app to write.
    fn build_saved_items(&self) -> Vec<(String, String)> {
        let (from, to) = if self.save_from <= self.save_to {
            (self.save_from, self.save_to)
        } else {
            (self.save_to, self.save_from)
        };
        // Only groups whose last run produced a result contribute rows.
        let groups: Vec<(&str, Vec<SavedPath>)> = self
            .configs
            .iter()
            .filter_map(|c| {
                let sp = c.ui.saved_paths();
                (!sp.is_empty()).then_some((c.label.as_str(), sp))
            })
            .collect();

        let mut files = Vec::new();
        for (sel, (name, key)) in self.save_sel.iter().zip(SAVE_ITEMS) {
            if !*sel {
                continue;
            }
            let mut s = String::new();
            let _ = writeln!(s, "# Save items — {name}");
            let _ = writeln!(
                s,
                "# value and propagated stderr per path index; unused paths are 0"
            );
            let _ = write!(s, "# {:<18}", "group");
            for p in from..=to {
                let _ = write!(
                    s,
                    "{:>16}{:>16}",
                    format!("path{p}"),
                    format!("path{p}_err")
                );
            }
            let _ = writeln!(s);
            for (label, paths) in &groups {
                let _ = write!(s, "  {label:<18}");
                for p in from..=to {
                    let (v, e) = paths
                        .iter()
                        .find(|sp| sp.number == p)
                        .map(|sp| sp.item(key))
                        .unwrap_or((0.0, 0.0));
                    let _ = write!(s, "{v:>16.6}{e:>16.6}");
                }
                let _ = writeln!(s);
            }
            files.push((format!("save_items_{}.txt", sanitize(name)), s));
        }
        files
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

/// A filename-safe form of an item name (`reff+delr` → `reff_delr`).
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_makes_item_names_filename_safe() {
        assert_eq!(sanitize("reff+delr"), "reff_delr");
        assert_eq!(sanitize("sigma2"), "sigma2");
        assert_eq!(sanitize("e0"), "e0");
    }

    #[test]
    fn save_items_covers_all_eight_dialog_items() {
        // The savable set matches the original dialog's eight items, and only
        // reff+Δr is the computed (empty-key) one.
        assert_eq!(SAVE_ITEMS.len(), 8);
        let computed: Vec<&str> = SAVE_ITEMS
            .iter()
            .filter(|(_, key)| key.is_empty())
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(computed, vec!["reff+delr"]);
    }
}
