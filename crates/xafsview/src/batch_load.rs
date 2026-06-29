//! The "Make μ(E) from files" staging picker.
//!
//! A Plot Data-style two-pane transfer list for choosing several raw scan files
//! before building a μ(E) group from each. Modelled on the Plot Data file picker
//! (`plot_data::PlotDataWindow::file_picker`) so the loading UX matches: stage
//! files with `=>`, move a wrongly-picked one back out with `<=`, Sort / Clear
//! all, then OK. On OK the app reads each staged file with the active import's
//! column mapping + header-skip and adds the groups (see
//! `app::XafsViewApp::build_xmu_from_paths`).

use std::collections::HashSet;
use std::path::PathBuf;

use eframe::egui;

use crate::widgets::{file_name_of, select_list};

/// The staging picker window and its two-pane selection state.
#[derive(Default)]
pub struct BatchLoadWindow {
    /// Whether the picker is shown.
    pub open: bool,
    /// The folder being browsed.
    dir: Option<PathBuf>,
    /// Raw scan files in `dir` (outputs/dotfiles hidden), minus the staged ones.
    available: Vec<PathBuf>,
    /// Files staged in the "Selected Data" pane, built into groups on OK.
    staged: Vec<PathBuf>,
    /// Multi-select highlight in each pane.
    avail_hi: HashSet<PathBuf>,
    sel_hi: HashSet<PathBuf>,
    /// Anchor row for shift-range selection in each pane (stale on list change).
    avail_anchor: Option<usize>,
    sel_anchor: Option<usize>,
}

/// What the picker asks the app to do after a frame.
pub enum BatchLoadAction {
    /// OK with a non-empty selection: build a μ(E) group from each of these.
    Load(Vec<PathBuf>),
}

impl BatchLoadWindow {
    /// Open the picker on `dir`, clearing any prior staging.
    pub fn open_on(&mut self, dir: Option<PathBuf>) {
        self.open = true;
        // Seed the folder the first time; afterwards keep wherever the user
        // last browsed, like the Plot Data picker.
        if self.dir.is_none() {
            self.dir = dir;
        }
        self.staged.clear();
        self.avail_hi.clear();
        self.sel_hi.clear();
        self.refresh_available();
    }

    /// List the raw scan files in `dir` (hiding the `.xmu` / `.chi` / `.dat` /
    /// `.fit` / `.bkg` outputs we write back into the data folder, plus
    /// dotfiles), excluding the ones already staged. Sorted by name.
    fn refresh_available(&mut self) {
        self.avail_anchor = None;
        self.sel_anchor = None;
        self.available.clear();
        let Some(dir) = self.dir.clone() else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = file_name_of(&path);
            if name.starts_with('.') || is_output_file(&name) {
                continue;
            }
            if self.staged.contains(&path) {
                continue;
            }
            self.available.push(path);
        }
        self.available.sort();
    }

    /// Render the picker as its own OS window; returns an action on OK.
    pub fn ui(&mut self, ctx: &egui::Context) -> Option<BatchLoadAction> {
        if !self.open {
            return None;
        }
        let mut action = None;
        let mut keep_open = true;
        crate::window::detached(
            ctx,
            "batch_load_picker",
            "Make μ(E) — select files",
            &mut keep_open,
            [660.0, 460.0],
            |ui| {
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    // Folder path bar.
                    ui.horizontal(|ui| {
                        if ui.button("📁 Browse…").clicked() {
                            let mut fd = rfd::FileDialog::new();
                            if let Some(dir) = &self.dir {
                                fd = fd.set_directory(dir);
                            }
                            if let Some(dir) = fd.pick_folder() {
                                self.dir = Some(dir);
                                self.avail_hi.clear();
                                self.sel_hi.clear();
                                self.refresh_available();
                            }
                        }
                        match &self.dir {
                            Some(dir) => {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(dir.display().to_string()).weak(),
                                    )
                                    .truncate(),
                                );
                            }
                            None => {
                                ui.weak("(pick a folder)");
                            }
                        }
                    });
                    ui.separator();

                    let mut do_add = false;
                    let mut do_remove = false;
                    let mut do_ok = false;
                    let mut do_sort = false;
                    let mut do_clear = false;

                    let pane_w = ((ui.available_width() - 72.0) * 0.5).max(150.0);
                    let list_h = (ui.available_height() - 36.0).max(160.0);

                    ui.horizontal_top(|ui| {
                        // LEFT — Available Data.
                        ui.vertical(|ui| {
                            ui.set_width(pane_w);
                            ui.strong("Available Data");
                            egui::ScrollArea::vertical()
                                .id_salt("bl_avail")
                                .max_height(list_h)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.set_min_width(pane_w);
                                    let avail = self.available.clone();
                                    select_list(
                                        ui,
                                        &avail,
                                        &mut self.avail_hi,
                                        &mut self.avail_anchor,
                                    );
                                });
                        });

                        // MIDDLE — transfer buttons.
                        ui.vertical(|ui| {
                            ui.add_space(8.0);
                            if ui
                                .button("=>")
                                .on_hover_text("Move highlighted to Selected")
                                .clicked()
                            {
                                do_add = true;
                            }
                            if ui
                                .button("<=")
                                .on_hover_text("Remove highlighted from Selected")
                                .clicked()
                            {
                                do_remove = true;
                            }
                            ui.add_space(10.0);
                            if ui.button("OK").clicked() {
                                do_ok = true;
                            }
                        });

                        // RIGHT — Selected Data, with Sort / Clear all.
                        ui.vertical(|ui| {
                            ui.set_width(pane_w);
                            ui.horizontal(|ui| {
                                ui.strong("Selected Data");
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.button("Clear all").clicked() {
                                            do_clear = true;
                                        }
                                        if ui.button("Sort").clicked() {
                                            do_sort = true;
                                        }
                                    },
                                );
                            });
                            egui::ScrollArea::vertical()
                                .id_salt("bl_sel")
                                .max_height(list_h)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.set_min_width(pane_w);
                                    let staged = self.staged.clone();
                                    select_list(
                                        ui,
                                        &staged,
                                        &mut self.sel_hi,
                                        &mut self.sel_anchor,
                                    );
                                });
                        });
                    });

                    if do_add {
                        let moving: Vec<PathBuf> = self
                            .available
                            .iter()
                            .filter(|p| self.avail_hi.contains(*p))
                            .cloned()
                            .collect();
                        for p in moving {
                            if !self.staged.contains(&p) {
                                self.staged.push(p);
                            }
                        }
                        self.avail_hi.clear();
                        self.refresh_available();
                    }
                    if do_remove {
                        self.staged.retain(|p| !self.sel_hi.contains(p));
                        self.sel_hi.clear();
                        self.refresh_available();
                    }
                    if do_sort {
                        self.staged.sort();
                        self.sel_anchor = None;
                    }
                    if do_clear {
                        self.staged.clear();
                        self.sel_hi.clear();
                        self.refresh_available();
                    }
                    if do_ok {
                        if !self.staged.is_empty() {
                            action = Some(BatchLoadAction::Load(std::mem::take(&mut self.staged)));
                        }
                        self.open = false;
                    }
                });
            },
        );
        // The OS window's close button (keep_open=false) closes the picker too.
        self.open = self.open && keep_open;
        action
    }
}

/// True for the `.xmu` / `.chi` / `.dat` / `.fit` / `.bkg` files this app writes
/// back into the data folder, so the picker lists only raw scan files.
fn is_output_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [".xmu", ".chi", ".dat", ".fit", ".bkg"]
        .iter()
        .any(|ext| lower.ends_with(ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_output_file_hides_our_outputs_keeps_raw_scans() {
        // Our written outputs are hidden.
        assert!(is_output_file("sample.xmu"));
        assert!(is_output_file("samplek.chi"));
        assert!(is_output_file("RDF.dat"));
        assert!(is_output_file("fit.FIT"));
        assert!(is_output_file("samplee.bkg"));
        // Raw scan files (numeric extensions) are kept.
        assert!(!is_output_file("PGT_Mn_S2_NCM50_W_Insitu_000"));
        assert!(!is_output_file("scan.001"));
        assert!(!is_output_file("data.txt"));
    }
}
