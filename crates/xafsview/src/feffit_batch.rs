//! The **Feffit batch panel**: run the Feffit tab's current fit configuration
//! against several loaded groups at once.
//!
//! Rendered inline on the right of the Feffit tab (not a separate window). The
//! Feffit tab's single editor *is* the batch configuration — "Run all" applies
//! that one config to every checked group (each fit independent, fanned across
//! cores), tabulates the results, and can save every group's transforms and
//! fitted parameters. There is no per-group config divergence by design: batch
//! fitting shares one model across similar spectra, so the panel manages
//! membership + run + save, while the tab's editor owns the model.

use std::collections::HashSet;
use std::fmt::Write as _;

use eframe::egui;
use feffit::xasdata::XasGroup;
use rayon::prelude::*;

use crate::feffit_ui::{FeffitUi, SavedPath};
use crate::plot::RED;

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

/// One group's batch run: a clone of the Feffit tab's config (the shared
/// template) carrying that group's fit result/plot, plus the run status.
struct GroupRun {
    /// Index of the group in the session's group list.
    group_idx: usize,
    label: String,
    ui: FeffitUi,
    /// Outcome of the run (`Ok(summary)` / `Err(message)`).
    status: Result<String, String>,
}

/// Fit one run's config against its group's `chi(k)`, recording the status.
/// Free of `&self` so "Run all" can fan the slow fits across cores — each call
/// mutates only its own `run` and reads the shared `groups`.
fn run_one(run: &mut GroupRun, groups: &[XasGroup]) {
    run.status = match groups.get(run.group_idx) {
        Some(g) => match (&g.k, &g.chi) {
            (Some(k), Some(chi)) => run.ui.run(k, chi),
            _ => Err("no chi(k) — run AUTOBK on this group first".to_owned()),
        },
        None => Err("group no longer exists".to_owned()),
    };
}

/// What the panel asks the app to do this frame (disk writes the app owns).
pub enum BatchAction {
    /// Write these `(filename, content)` pairs (one per selected Save-Items
    /// item); the app picks the destination folder and reports the outcome.
    SaveItems(Vec<(String, String)>),
    /// Write these `(filename, content)` pairs — the per-group FEFFIT k/r/q
    /// `.dat`/`.fit` transforms; the app writes them and reports the outcome.
    SaveFeffitOutputs(Vec<(String, String)>),
}

/// The Feffit batch panel: a group-membership checklist, "Run all", a results
/// table, and the two batch save actions. Shares the Feffit tab's editor as the
/// fit configuration (passed to [`panel`](Self::panel) as the template).
pub struct FeffitBatch {
    /// Session group indices included in "Run all".
    members: HashSet<usize>,
    /// Anchor row for shift-range selection of `members` (the last plain-clicked
    /// row); `None` until a row is clicked. Stales if the group list changes.
    anchor: Option<usize>,
    /// Results of the last "Run all" (one per member that existed at run time),
    /// in ascending group order.
    runs: Vec<GroupRun>,
    /// "Save Items": which of the eight items to write (in [`SAVE_ITEMS`] order).
    save_sel: [bool; 8],
    /// Inclusive path-index range written for each item.
    save_from: usize,
    save_to: usize,
}

impl Default for FeffitBatch {
    fn default() -> Self {
        Self {
            members: HashSet::new(),
            anchor: None,
            runs: Vec::new(),
            // Default to the most-used items (N and the bond distance reff+Δr).
            save_sel: [false, false, true, false, false, false, false, true],
            save_from: 1,
            save_to: 1,
        }
    }
}

impl FeffitBatch {
    /// Render the panel over `groups`, fitting with `template` (the Feffit tab's
    /// current config). Returns a [`BatchAction`] for app-owned disk writes.
    pub fn panel(
        &mut self,
        ui: &mut egui::Ui,
        groups: &[XasGroup],
        template: &FeffitUi,
    ) -> Option<BatchAction> {
        let mut bubble = None;

        ui.heading("Batch");
        ui.label("Run the current fit setup against every checked group.");

        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Groups in batch");
            ui.weak(format!("({} selected)", self.members.len()));
        });
        if groups.is_empty() {
            ui.weak("No groups loaded.");
        } else {
            ui.weak("Click to toggle; shift-click to extend a range.");
            // Bounded so the list (which can run to hundreds of groups) doesn't
            // push "Run all" and the save actions off-screen.
            egui::ScrollArea::vertical()
                .id_salt("feffit_batch_groups")
                .max_height(260.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    crate::widgets::select_index_list(
                        ui,
                        groups.iter().map(|g| g.label.as_str()),
                        &mut self.members,
                        &mut self.anchor,
                    );
                });
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.members.is_empty(), egui::Button::new("Run all"))
                .on_hover_text("Fit each checked group with the Feffit tab's current config")
                .clicked()
            {
                self.run_all(groups, template);
            }
            if ui
                .add_enabled(!self.runs.is_empty(), egui::Button::new("Clear results"))
                .clicked()
            {
                self.runs.clear();
            }
        });

        ui.separator();
        let any_fit = self.runs.iter().any(|r| r.ui.plot().is_some());
        if ui
            .add_enabled(any_fit, egui::Button::new("Save χ data+fit (all groups)"))
            .on_hover_text(
                "Write every fitted group's FEFFIT transforms as \
                 <label>{k,r,q}.dat (data) and <label>{k,r,q}.fit (model).",
            )
            .clicked()
        {
            let files = self.feffit_output_files(groups, template);
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
                let any_result = self.runs.iter().any(|r| r.ui.result().is_some());
                if ui
                    .add_enabled(
                        any_item && any_result,
                        egui::Button::new("Save items to work folder"),
                    )
                    .on_hover_text(
                        "One file per item; rows are groups, columns are path \
                         indices (value and stderr; unused paths are 0).",
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
        self.results_table(ui);

        bubble
    }

    /// Rebuild the run list from the current members + template and fit each
    /// group. Members are run in ascending group order; the fits are independent
    /// (own config, own FFT planner, reading only the shared `groups`) so they
    /// fan across cores.
    fn run_all(&mut self, groups: &[XasGroup], template: &FeffitUi) {
        let mut idxs: Vec<usize> = self.members.iter().copied().collect();
        idxs.sort_unstable();
        self.runs = idxs
            .into_iter()
            .filter_map(|idx| {
                groups.get(idx).map(|g| GroupRun {
                    group_idx: idx,
                    label: g.label.clone(),
                    ui: template.config_clone(),
                    status: Ok(String::new()),
                })
            })
            .collect();
        self.runs
            .par_iter_mut()
            .for_each(|run| run_one(run, groups));
    }

    /// Collect the FEFFIT k/r/q `.dat`/`.fit` files for every run that has a
    /// fit result, as `(filename, content)` pairs named from the group label.
    /// Each file carries the provenance header built from its source group (for
    /// the reduction params + raw source header) and the shared `template`'s FT
    /// params — the same header the single-fit writer produces.
    fn feffit_output_files(
        &self,
        groups: &[XasGroup],
        template: &FeffitUi,
    ) -> Vec<(String, String)> {
        let (kmin, kmax, kweight, dk) = template.header_ft();
        self.runs
            .iter()
            .filter_map(|run| {
                let plot = run.ui.plot()?;
                let header = match groups.get(run.group_idx) {
                    Some(g) => crate::chi_io::provenance_header(g, kmin, kmax, kweight, dk),
                    None => format!("# {}\r\n", run.label),
                };
                Some(plot.output_pairs(&run.label, &header))
            })
            .flatten()
            .collect()
    }

    /// Build one output file per selected Save-Items item: rows are the fitted
    /// groups, columns are path indices `from..=to`. Each cell is the item's
    /// value and propagated stderr for that path, or `0` when a group's fit has
    /// no such path (the original's `n = 0` filler).
    fn build_saved_items(&self) -> Vec<(String, String)> {
        let (from, to) = if self.save_from <= self.save_to {
            (self.save_from, self.save_to)
        } else {
            (self.save_to, self.save_from)
        };
        // Only groups whose last run produced a result contribute rows.
        let groups: Vec<(&str, Vec<SavedPath>)> = self
            .runs
            .iter()
            .filter_map(|run| {
                let sp = run.ui.saved_paths();
                (!sp.is_empty()).then_some((run.label.as_str(), sp))
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

    /// The results grid: one row per run with its status and key stats.
    fn results_table(&mut self, ui: &mut egui::Ui) {
        ui.strong("Results");
        if self.runs.is_empty() {
            ui.weak("Check groups and Run all.");
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
                for run in &self.runs {
                    ui.label(&run.label);
                    match &run.status {
                        Ok(_) => {
                            ui.monospace("ok");
                        }
                        Err(e) => {
                            ui.colored_label(RED, "error").on_hover_text(e);
                        }
                    }
                    match run.ui.result() {
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

    #[test]
    fn run_all_fits_each_checked_group_with_the_template() {
        use feffit::xasdata::XasGroup;
        // A synthetic χ(k) — enough for the default "Only FT" transform, which
        // needs no Feff paths. Two groups, both checked; one shared template.
        let k: Vec<f64> = (0..400).map(|i| i as f64 * 0.05).collect();
        let chi: Vec<f64> = k.iter().map(|&k| 0.1 * (2.0 * k).sin()).collect();
        let groups = vec![
            XasGroup::from_chi("A", k.clone(), chi.clone()),
            XasGroup::from_chi("B", k.clone(), chi.clone()),
        ];
        let mut b = FeffitBatch::default();
        b.members.insert(0);
        b.members.insert(1);
        b.run_all(&groups, &FeffitUi::default());

        assert_eq!(b.runs.len(), 2, "one run per checked group");
        assert!(
            b.runs.iter().all(|r| r.status.is_ok()),
            "Only FT runs clean"
        );
        assert!(
            b.runs.iter().all(|r| r.ui.plot().is_some()),
            "each run produced a transform"
        );
        // The data .dat transforms are now saveable across the run groups.
        assert!(
            !b.feffit_output_files(&groups, &FeffitUi::default())
                .is_empty()
        );
    }

    #[test]
    fn nothing_to_save_before_any_run() {
        // No runs yet → no fitted transforms to save (mirrors the disabled Save
        // buttons; the builder defends the same invariant).
        let b = FeffitBatch::default();
        assert!(b.feffit_output_files(&[], &FeffitUi::default()).is_empty());
    }
}
