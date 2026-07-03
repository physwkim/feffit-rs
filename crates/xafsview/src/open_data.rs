//! The unified "Open data" loader for the Autobk flow.
//!
//! One in-app window that replaces both the native OS file dialog and the old
//! "Make μ(E) from files" picker. It is anchored on the Folders Data folder
//! (Autobk only loads from there) and offers light Up/sub-folder navigation, a
//! Plot Data-style two-pane transfer list (stage raw scan files on the right with
//! `=>`, pull a wrong pick back with `<=`, Sort / Clear all), and — hosted by the
//! app around this browser — the reading-format chooser
//! ([`crate::import::ImportState`]) plus an **Autobk Start** button that builds
//! μ(E) for every staged file and reduces them.
//!
//! It lists **raw scan files only** (the `.xmu`/`.chi`/… outputs this app writes
//! back are hidden by [`is_output_file`]); a precomputed-μ file is loaded through
//! the same panel using the reading-format Mode "Raw μ". The reading-format
//! mapping applies to *every* staged file at build time, so the app previews the
//! [`representative`](OpenDataBrowser::representative) (the first staged file).

use std::collections::HashSet;
use std::path::PathBuf;

use eframe::egui;

use crate::widgets::{file_name_of, select_list};

/// The Open-data loader's browse + staging state. The reading-format chooser and
/// the Autobk Start button live on the app (they need
/// [`crate::import::ImportState`] and the build path); this owns only the folder
/// navigation and the two-pane selection.
#[derive(Default)]
pub struct OpenDataBrowser {
    /// Whether the loader window is shown.
    pub open: bool,
    /// The folder being browsed.
    dir: Option<PathBuf>,
    /// Sub-folders in `dir` (for Up/enter navigation), sorted by name.
    subdirs: Vec<PathBuf>,
    /// Raw scan files in `dir` (outputs/dotfiles hidden), minus the staged ones.
    available: Vec<PathBuf>,
    /// Files staged in the "Selected Data" pane, built into groups on Autobk Start.
    staged: Vec<PathBuf>,
    /// Multi-select highlight in each pane.
    avail_hi: HashSet<PathBuf>,
    sel_hi: HashSet<PathBuf>,
    /// Anchor row for shift-range selection in each pane (stale on list change).
    avail_anchor: Option<usize>,
    sel_anchor: Option<usize>,
}

impl OpenDataBrowser {
    /// Open the loader anchored on `dir` (the Folders Data folder), seeding the
    /// Selected pane with `loaded` — the source files already loaded as groups.
    /// The loader thus shows what is currently loaded (not an empty pane) and
    /// Autobk Start rebuilds only the newly-added files, leaving existing groups
    /// alone. The session's groups are the single source of truth, so re-anchoring
    /// on every open keeps the pane in sync. Re-anchors the folder on every open;
    /// falls back to the working directory when none is set so the window is never
    /// an empty dead-end.
    pub fn open_on(&mut self, dir: Option<PathBuf>, loaded: Vec<PathBuf>) {
        self.open = true;
        if dir.is_some() {
            self.dir = dir;
        }
        if self.dir.is_none() {
            self.dir = std::env::current_dir().ok();
        }
        self.staged = loaded;
        self.avail_hi.clear();
        self.sel_hi.clear();
        self.refresh();
    }

    /// The representative file for the reading-format panel: the first staged
    /// file. All staged files share one column mapping at build time, so the app
    /// previews and seeds columns from this one.
    pub fn representative(&self) -> Option<&PathBuf> {
        self.staged.first()
    }

    /// Whether anything is staged (gates the Autobk Start button).
    pub fn has_staged(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Take the staged paths, leaving the set empty (called on Autobk Start).
    pub fn take_staged(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.staged)
    }

    /// Re-list `dir` into `subdirs` + `available` (raw files minus the staged).
    fn refresh(&mut self) {
        self.avail_anchor = None;
        self.sel_anchor = None;
        self.subdirs.clear();
        self.available.clear();
        let Some(dir) = self.dir.clone() else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = file_name_of(&path);
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                self.subdirs.push(path);
                continue;
            }
            if !path.is_file() || is_output_file(&name) {
                continue;
            }
            if self.staged.contains(&path) {
                continue;
            }
            self.available.push(path);
        }
        self.subdirs.sort();
        self.available.sort();
    }

    /// Render the nav bar, sub-folders, and the two-pane staging list. Mutates the
    /// staged set; the caller reads [`representative`](Self::representative) /
    /// [`has_staged`](Self::has_staged) / [`take_staged`](Self::take_staged).
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // A folder to navigate into (a sub-folder click) or up to (Up); applied
        // after the render below.
        let mut nav: Option<PathBuf> = None;

        ui.horizontal(|ui| {
            let parent = self
                .dir
                .as_ref()
                .and_then(|d| d.parent())
                .map(PathBuf::from);
            if ui
                .add_enabled(parent.is_some(), egui::Button::new("Up"))
                .on_hover_text("Go to the parent folder")
                .clicked()
            {
                nav = parent;
            }
            match &self.dir {
                Some(d) => {
                    ui.add(
                        egui::Label::new(egui::RichText::new(d.display().to_string()).weak())
                            .truncate(),
                    );
                }
                None => {
                    ui.weak("(no Data folder — set one in the Folders tab)");
                }
            }
        });
        if !self.subdirs.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for d in &self.subdirs {
                    if ui.small_button(format!("📁 {}", file_name_of(d))).clicked() {
                        nav = Some(d.clone());
                    }
                }
            });
        }
        ui.separator();

        let mut do_add = false;
        let mut do_remove = false;
        let mut do_sort = false;
        let mut do_clear = false;

        let pane_w = ((ui.available_width() - 72.0) * 0.5).max(150.0);
        let list_h = 180.0;

        ui.horizontal_top(|ui| {
            // LEFT — Available Data.
            ui.vertical(|ui| {
                ui.set_width(pane_w);
                ui.strong("Available Data");
                egui::ScrollArea::vertical()
                    .id_salt("od_avail")
                    .max_height(list_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(pane_w);
                        let avail = self.available.clone();
                        select_list(ui, &avail, &mut self.avail_hi, &mut self.avail_anchor);
                    });
            });

            // MIDDLE — transfer buttons.
            ui.vertical(|ui| {
                ui.add_space(8.0);
                if ui
                    .button("=>")
                    .on_hover_text("Stage highlighted for loading")
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
            });

            // RIGHT — Selected Data, with Sort / Clear all.
            ui.vertical(|ui| {
                ui.set_width(pane_w);
                ui.horizontal(|ui| {
                    ui.strong("Selected Data");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Clear all").clicked() {
                            do_clear = true;
                        }
                        if ui.button("Sort").clicked() {
                            do_sort = true;
                        }
                    });
                });
                egui::ScrollArea::vertical()
                    .id_salt("od_sel")
                    .max_height(list_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(pane_w);
                        let staged = self.staged.clone();
                        select_list(ui, &staged, &mut self.sel_hi, &mut self.sel_anchor);
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
            self.refresh();
        }
        if do_remove {
            self.staged.retain(|p| !self.sel_hi.contains(p));
            self.sel_hi.clear();
            self.refresh();
        }
        if do_sort {
            self.staged.sort();
            self.sel_anchor = None;
        }
        if do_clear {
            self.staged.clear();
            self.sel_hi.clear();
            self.refresh();
        }
        if let Some(target) = nav {
            // Keep the staged set across navigation so files can be gathered from
            // several folders; only the current listing changes.
            self.dir = Some(target);
            self.refresh();
        }
    }
}

/// True for the `.xmu` / `.chi` / `.dat` / `.fit` / `.bkg` files this app writes
/// back into the data folder, so the loader lists only raw scan files.
pub(crate) fn is_output_file(name: &str) -> bool {
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
        assert!(!is_output_file("PGT_Mn_S2_NCM50_W_Insitu.000"));
        assert!(!is_output_file("scan.001"));
        assert!(!is_output_file("data.txt"));
    }
}
