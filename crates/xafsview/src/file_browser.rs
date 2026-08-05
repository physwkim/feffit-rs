//! The browse-and-stage widget shared by the Autobk **Open data** loader and the
//! Plot Data **File selection** picker.
//!
//! Both windows do the same thing to different file sets: navigate a folder,
//! list the files in it that the host cares about, and move a multi-selection
//! into a staged "Selected Data" pane. They differ only in *which* files are
//! listed — [`crate::open_data`] wants raw scans, [`crate::plot_data`] wants the
//! files matching its current graph item — so that is the one thing the host
//! supplies, as the `keep` predicate on [`FileBrowser::refresh`].
//!
//! Keeping the navigation here is what lets both windows browse, and delete,
//! without two copies of the same folder walk drifting apart.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use eframe::egui;

use crate::widgets::{file_name_of, select_list};

/// Folder navigation plus the two-pane transfer list.
///
/// The staged set survives navigation, so files can be gathered from several
/// folders into one selection; only the *Available* listing follows the folder.
pub struct FileBrowser {
    /// The folder being browsed.
    dir: Option<PathBuf>,
    /// Sub-folders of `dir`, sorted by name (the in-window tree step).
    subdirs: Vec<PathBuf>,
    /// Files of `dir` the host's predicate kept, minus the staged ones.
    available: Vec<PathBuf>,
    /// The staged selection ("Selected Data").
    staged: Vec<PathBuf>,
    /// Multi-select highlight in each pane.
    avail_hi: HashSet<PathBuf>,
    sel_hi: HashSet<PathBuf>,
    /// Anchor row for shift-range selection in each pane (stale on list change).
    avail_anchor: Option<usize>,
    sel_anchor: Option<usize>,
    /// egui id prefix, so two browsers on screen keep distinct scroll state.
    id: &'static str,
    /// Files awaiting the delete confirmation, and the deletion errors to show.
    pending_delete: Vec<PathBuf>,
    delete_errors: Vec<String>,
}

impl FileBrowser {
    /// A browser whose egui ids are prefixed with `id` (unique per host window).
    pub fn new(id: &'static str) -> Self {
        Self {
            dir: None,
            subdirs: Vec::new(),
            available: Vec::new(),
            staged: Vec::new(),
            avail_hi: HashSet::new(),
            sel_hi: HashSet::new(),
            avail_anchor: None,
            sel_anchor: None,
            id,
            pending_delete: Vec::new(),
            delete_errors: Vec::new(),
        }
    }

    /// The folder being browsed.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Point the browser at `dir` (ignored when `None`, so a caller can pass an
    /// unset preference without clearing the current folder).
    pub fn set_dir(&mut self, dir: Option<PathBuf>) {
        if dir.is_some() {
            self.dir = dir;
        }
    }

    /// Fall back to the working directory when no folder is set, so the window
    /// is never an empty dead-end.
    pub fn ensure_dir(&mut self) {
        if self.dir.is_none() {
            self.dir = std::env::current_dir().ok();
        }
    }

    /// The staged selection.
    pub fn staged(&self) -> &[PathBuf] {
        &self.staged
    }

    /// Whether anything is staged.
    pub fn has_staged(&self) -> bool {
        !self.staged.is_empty()
    }

    /// Replace the staged selection (e.g. seeding it with what is already loaded).
    pub fn set_staged(&mut self, staged: Vec<PathBuf>) {
        self.staged = staged;
        self.avail_hi.clear();
        self.sel_hi.clear();
    }

    /// Take the staged paths, leaving the set empty.
    pub fn take_staged(&mut self) -> Vec<PathBuf> {
        self.sel_hi.clear();
        self.sel_anchor = None;
        std::mem::take(&mut self.staged)
    }

    /// Re-list the current folder: sub-folders into the nav row, and the files
    /// `keep` accepts (by bare file name) into *Available*, minus the staged.
    ///
    /// Dotfiles are always skipped — macOS AppleDouble `._*` sidecars and
    /// `.DS_Store` are not data and only clutter the list — so `keep` never sees
    /// them and no host has to re-state the rule.
    pub fn refresh(&mut self, keep: impl Fn(&str) -> bool) {
        // The lists are about to change, so any anchored row index is stale.
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
            if !path.is_file() || !keep(&name) || self.staged.contains(&path) {
                continue;
            }
            self.available.push(path);
        }
        self.subdirs.sort();
        self.available.sort();
    }

    /// Render the navigation row: Up, the current path, and a chip per
    /// sub-folder. Returns `true` when the folder changed (the host must
    /// [`refresh`](Self::refresh), which needs its own predicate).
    #[must_use]
    pub fn nav_bar(&mut self, ui: &mut egui::Ui, empty_hint: &str) -> bool {
        // A folder to navigate into (a sub-folder chip) or up to (Up); applied
        // after the render so the lists are not mutated mid-layout.
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
            if ui
                .button("📁 Browse…")
                .on_hover_text("Jump to a folder outside this tree")
                .clicked()
            {
                let mut fd = rfd::FileDialog::new();
                if let Some(dir) = self.dir.as_ref() {
                    fd = fd.set_directory(dir);
                }
                if let Some(dir) = fd.pick_folder() {
                    nav = Some(dir);
                }
            }
            match &self.dir {
                Some(d) => {
                    ui.add(
                        egui::Label::new(egui::RichText::new(d.display().to_string()).weak())
                            .truncate(),
                    );
                }
                None => {
                    ui.weak(empty_hint);
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
        match nav {
            Some(target) => {
                // The staged set survives navigation so files can be gathered
                // from several folders; only the listing follows.
                self.dir = Some(target);
                self.avail_hi.clear();
                true
            }
            None => false,
        }
    }

    /// Render the two panes and the transfer buttons. `keep` is the same
    /// predicate [`refresh`](Self::refresh) takes — the lists are re-listed in
    /// place when a button changes them. Returns whether the staged set changed.
    #[must_use]
    pub fn panes(&mut self, ui: &mut egui::Ui, list_h: f32, keep: impl Fn(&str) -> bool) -> bool {
        let mut do_add = false;
        let mut do_remove = false;
        let mut do_sort = false;
        let mut do_clear = false;
        let mut do_delete = false;

        let pane_w = ((ui.available_width() - 72.0) * 0.5).max(150.0);

        ui.horizontal_top(|ui| {
            // LEFT — Available Data.
            ui.vertical(|ui| {
                ui.set_width(pane_w);
                egui::ScrollArea::vertical()
                    .id_salt(format!("{}_avail", self.id))
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
                ui.add_space(10.0);
                if ui
                    .add_enabled(
                        !self.avail_hi.is_empty(),
                        egui::Button::new(
                            egui::RichText::new("Delete").color(ui.visuals().error_fg_color),
                        ),
                    )
                    .on_hover_text("Delete the highlighted Available files from disk")
                    .on_disabled_hover_text("Highlight files in Available Data first")
                    .clicked()
                {
                    do_delete = true;
                }
            });

            // RIGHT — Selected Data, with Sort / Clear all.
            ui.vertical(|ui| {
                ui.set_width(pane_w);
                ui.horizontal(|ui| {
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
                    .id_salt(format!("{}_sel", self.id))
                    .max_height(list_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_min_width(pane_w);
                        let staged = self.staged.clone();
                        select_list(ui, &staged, &mut self.sel_hi, &mut self.sel_anchor);
                    });
            });
        });

        let mut changed = false;
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
            self.refresh(&keep);
            changed = true;
        }
        if do_remove {
            self.staged.retain(|p| !self.sel_hi.contains(p));
            self.sel_hi.clear();
            self.refresh(&keep);
            changed = true;
        }
        if do_sort {
            self.staged.sort();
            self.sel_anchor = None;
        }
        if do_clear {
            self.staged.clear();
            self.sel_hi.clear();
            self.refresh(&keep);
            changed = true;
        }
        if do_delete {
            // Deleting is not undoable, so the button only *stages* the request;
            // `confirm_delete` renders the modal that actually removes files.
            self.pending_delete = self
                .available
                .iter()
                .filter(|p| self.avail_hi.contains(*p))
                .cloned()
                .collect();
            self.delete_errors.clear();
        }
        changed
    }

    /// Render the delete confirmation modal, if a delete is pending. Returns
    /// `true` when files were actually removed (the host may need to drop the
    /// traces that were reading them).
    ///
    /// Deletion is irreversible and the button sits beside two that merely move
    /// rows between panes, so the confirmation is not optional: nothing leaves
    /// the disk without this dialog being answered.
    #[must_use]
    pub fn confirm_delete(&mut self, ctx: &egui::Context, keep: impl Fn(&str) -> bool) -> bool {
        if self.pending_delete.is_empty() && self.delete_errors.is_empty() {
            return false;
        }
        let mut deleted = false;
        let mut close = false;
        egui::Window::new("Delete files")
            .id(egui::Id::new(format!("{}_delete", self.id)))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if !self.delete_errors.is_empty() {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("{} file(s) could not be deleted:", self.delete_errors.len()),
                    );
                    for e in &self.delete_errors {
                        ui.weak(e);
                    }
                    ui.separator();
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                    return;
                }
                ui.label(format!(
                    "Permanently delete {} file(s) from disk? This cannot be undone.",
                    self.pending_delete.len()
                ));
                egui::ScrollArea::vertical()
                    .id_salt(format!("{}_delete_list", self.id))
                    .max_height(160.0)
                    .show(ui, |ui| {
                        for p in &self.pending_delete {
                            ui.weak(file_name_of(p));
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button(egui::RichText::new("Delete").color(ui.visuals().error_fg_color))
                        .clicked()
                    {
                        for p in std::mem::take(&mut self.pending_delete) {
                            match std::fs::remove_file(&p) {
                                Ok(()) => deleted = true,
                                Err(e) => self
                                    .delete_errors
                                    .push(format!("{}: {e}", file_name_of(&p))),
                            }
                        }
                        // A deleted file can no longer be loaded, so it must not
                        // linger in the staged set either.
                        self.staged.retain(|p| p.exists());
                        self.avail_hi.clear();
                        self.sel_hi.clear();
                        close = self.delete_errors.is_empty();
                    }
                    if ui.button("Cancel").clicked() {
                        self.pending_delete.clear();
                        close = true;
                    }
                });
            });
        if deleted {
            self.refresh(&keep);
        }
        if close {
            self.pending_delete.clear();
            self.delete_errors.clear();
        }
        deleted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xafsview_fb_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("mkdir");
        d
    }

    #[test]
    fn refresh_splits_subfolders_from_files_and_applies_the_predicate() {
        let d = tmpdir("split");
        std::fs::create_dir_all(d.join("sub")).expect("mkdir sub");
        std::fs::write(d.join("scan.000"), "1 2\n").expect("w");
        std::fs::write(d.join("scan.xmu"), "1 2\n").expect("w");
        std::fs::write(d.join(".DS_Store"), "x").expect("w");

        let mut fb = FileBrowser::new("t");
        fb.set_dir(Some(d.clone()));
        fb.refresh(|name| !name.ends_with(".xmu"));

        assert_eq!(fb.subdirs, vec![d.join("sub")]);
        assert_eq!(
            fb.available,
            vec![d.join("scan.000")],
            "predicate drops .xmu; the dotfile never reaches it"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn staged_files_leave_the_available_list_and_survive_navigation() {
        let d = tmpdir("nav");
        let sub = d.join("sub");
        std::fs::create_dir_all(&sub).expect("mkdir sub");
        std::fs::write(d.join("a.000"), "1 2\n").expect("w");
        std::fs::write(sub.join("b.000"), "1 2\n").expect("w");

        let mut fb = FileBrowser::new("t");
        fb.set_dir(Some(d.clone()));
        fb.set_staged(vec![d.join("a.000")]);
        fb.refresh(|_| true);
        assert!(
            fb.available.is_empty(),
            "the one file is staged, so nothing is available"
        );

        // Navigating to the sub-folder relists, but the selection is kept: the
        // point of the staged pane is gathering across folders.
        fb.set_dir(Some(sub.clone()));
        fb.refresh(|_| true);
        assert_eq!(fb.available, vec![sub.join("b.000")]);
        assert_eq!(fb.staged(), [d.join("a.000")]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn take_staged_empties_the_selection() {
        let mut fb = FileBrowser::new("t");
        fb.set_staged(vec![PathBuf::from("/x/a.000"), PathBuf::from("/x/b.000")]);
        assert!(fb.has_staged());
        assert_eq!(fb.take_staged().len(), 2);
        assert!(!fb.has_staged(), "taking leaves the set empty");
    }

    #[test]
    fn set_dir_none_keeps_the_current_folder() {
        let mut fb = FileBrowser::new("t");
        fb.set_dir(Some(PathBuf::from("/x")));
        fb.set_dir(None);
        assert_eq!(fb.dir(), Some(Path::new("/x")));
    }
}
