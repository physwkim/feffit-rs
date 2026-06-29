//! The standalone **Plot Data** window: a *file viewer*. Browse a folder, pick
//! processed output files by *File type* (`*.xmu` / `*.chi` / `*.dat` / `*.fit`)
//! and *Graph item*, and overlay them on one plot — with vertical stacking, an
//! averaged trace, display smoothing, and a multi-peak readout. Mirrors
//! XAFSView's *Plot Data* window, which shows data read from files rather than
//! the in-memory session groups.
//!
//! It owns its own plot (separate from the tabs' shared plot) so it can float
//! independently. Save / zoom / legend come from the siplot toolbar; the data
//! work (averaging, peak finding) is the headless [`xasdata`] code. A Feffit fit
//! can also be sent here for a quick data-vs-model look ([`set_fit_overlay`]).

use std::collections::HashSet;
use std::path::PathBuf;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use feffit::xasdata::{average_curves, peak_in_range, x_at_y};
use siplot::YAxis;

use crate::plot_files::{FileType, GraphItem, LoadedTrace, load_result, load_trace};

/// The "Items" of the Multiple peaks catching window: which feature to locate
/// in the search range. (The original's derivative-based "Peak @ x" needs the
/// raw-data round-trip it describes and is left out.)
#[derive(Clone, Copy, PartialEq, Eq)]
enum PeakMode {
    /// Position of the maximum (the original "Max").
    Max,
    /// Position of the minimum (the original "Min").
    Min,
    /// Interpolated x where the curve crosses a target y — covers the original
    /// "Half step" (y = 0.5 on normalized μ) and "x[i] at y".
    AtY,
}

impl PeakMode {
    const ALL: [PeakMode; 3] = [PeakMode::Max, PeakMode::Min, PeakMode::AtY];

    fn label(self) -> &'static str {
        match self {
            PeakMode::Max => "Max",
            PeakMode::Min => "Min",
            PeakMode::AtY => "x at y",
        }
    }
}

/// Feffit data vs model curve colours, shared by the Feffit tab's own plot
/// ([`replot_feffit`](crate::app)) and this window's "Send to Plot Data" overlay.
pub(crate) const FIT_DATA: Color32 = crate::plot::BLUE;
pub(crate) const FIT_MODEL: Color32 = crate::plot::RED;

/// A Feffit fit handed over from the Feffit tab's "Send to Plot Data": its data
/// and model curves in the chosen space, with that space's axis labels. When
/// shown it takes over the plot (its axes differ from the file items).
struct FitOverlay {
    label: String,
    xlabel: &'static str,
    ylabel: &'static str,
    x: Vec<f64>,
    data: Vec<f64>,
    model: Vec<f64>,
}

/// The Plot Data window state and its own plot.
pub struct PlotDataWindow {
    /// Whether the window is shown.
    pub open: bool,
    plot: crate::plot::Plot,
    /// k-weight applied to any loaded k-space file (their χ is stored unweighted).
    kweight: i32,
    /// Vertical offset added to trace `i` (`i · stack`), in data units.
    stack: f64,
    show_average: bool,
    /// "Average (5 points)": display-smooth each curve (5-point moving average).
    smooth5: bool,
    /// "Change BG color": dark plot background (the original's black/white swap).
    dark_bg: bool,
    peak_lo: f64,
    peak_hi: f64,
    /// Which feature the Multiple peaks catching window locates.
    peak_mode: PeakMode,
    /// Target y for the "x at y" mode (0.5 = normalized half-step).
    peak_target: f64,
    /// One `(file label, x, y)` row per loaded file from the last catch.
    peaks: Vec<(String, f64, f64)>,
    /// A Feffit fit sent here via "Send to Plot Data", if any.
    overlay: Option<FitOverlay>,
    /// Whether the sent fit takes over the plot (vs the loaded files).
    show_overlay: bool,
    /// The legend label of the curve highlighted by a legend click (drawn with a
    /// thicker line, brought to front). `None` = nothing highlighted; clicking
    /// the same entry again clears it. Plot Data only — the shared plot reports
    /// the click, other tabs ignore it.
    highlighted: Option<String>,

    // --- file viewer --------------------------------------------------------
    /// The selected *File type* for browsing/loading files.
    file_type: FileType,
    /// The *Graph item* under `file_type` (its file-name suffix + plot recipe).
    graph_item: GraphItem,
    /// Files loaded for display.
    loaded: Vec<LoadedTrace>,
    /// Whether the "Add / remove data files" picker is open.
    picker_open: bool,
    /// The folder the picker browses.
    pick_dir: Option<PathBuf>,
    /// Files in `pick_dir` matching the current graph item, minus `pick_add`.
    available: Vec<PathBuf>,
    /// Files staged in the picker (the "Selected Data" pane), loaded on OK.
    pick_add: Vec<PathBuf>,
    /// Multi-selection highlight in the "Available Data" pane.
    avail_hi: HashSet<PathBuf>,
    /// Multi-selection highlight in the "Selected Data" pane.
    sel_hi: HashSet<PathBuf>,
    /// Anchor row (the last plain click) in each pane; a shift-click selects the
    /// whole inclusive range between it and the clicked row. `None` until the
    /// first plain click, and reset whenever the list contents change.
    avail_anchor: Option<usize>,
    sel_anchor: Option<usize>,
    /// Outcome of the last load (shown in the Files section).
    pick_status: String,
    /// The configured "Results" folder (`Folders.results_dir`), kept in sync by
    /// the app each frame via [`Self::show`]. "Save in single file" defaults its
    /// dialog here, matching the original XAFSView.
    results_dir: Option<PathBuf>,
    /// The configured "Data" folder (`Folders.data_dir`, under the Sub base),
    /// kept in sync the same way. The file picker opens here and the "Browse…"
    /// dialog defaults here, matching the original XAFSView.
    data_dir: Option<PathBuf>,

    /// Set whenever the overlay needs rebuilding (control change or new files).
    dirty: bool,
}

impl PlotDataWindow {
    /// Build the window with its own plot (use a distinct `PlotId` from the
    /// tabs' shared plot).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::Plot::new(render_state, 1);
        plot.set_graph_title("Plot Data");
        Self {
            open: false,
            plot,
            kweight: 2,
            stack: 0.0,
            show_average: false,
            smooth5: false,
            // Dark by default, matching the cohesive dark canvas every other
            // plot uses (the checkbox still flips to the white "Change BG" mode).
            dark_bg: true,
            peak_lo: 0.0,
            peak_hi: 0.0,
            peak_mode: PeakMode::Max,
            peak_target: 0.5,
            peaks: Vec::new(),
            overlay: None,
            show_overlay: false,
            highlighted: None,
            file_type: FileType::Chi,
            graph_item: FileType::Chi.default_item(),
            loaded: Vec::new(),
            picker_open: false,
            pick_dir: None,
            available: Vec::new(),
            pick_add: Vec::new(),
            avail_hi: HashSet::new(),
            sel_hi: HashSet::new(),
            avail_anchor: None,
            sel_anchor: None,
            pick_status: String::new(),
            results_dir: None,
            data_dir: None,
            dirty: true,
        }
    }

    /// Request a rebuild on the next show.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Take a Feffit fit's data + model curves (the Feffit form's "Send to plot
    /// data"). The window opens showing the fit; untick "Show Feffit fit" or
    /// "Clear fit" to return to the loaded files.
    pub fn set_fit_overlay(
        &mut self,
        label: String,
        xlabel: &'static str,
        ylabel: &'static str,
        x: Vec<f64>,
        data: Vec<f64>,
        model: Vec<f64>,
    ) {
        self.overlay = Some(FitOverlay {
            label,
            xlabel,
            ylabel,
            x,
            data,
            model,
        });
        self.show_overlay = true;
        self.open = true;
        self.dirty = true;
    }

    /// Render the window.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        results_dir: Option<&std::path::Path>,
        data_dir: Option<&std::path::Path>,
    ) {
        // Track the configured Results/Data folders so the save dialog and file
        // picker can default there (kept fresh in case the user reconfigures
        // folders).
        self.results_dir = results_dir.map(std::path::Path::to_path_buf);
        self.data_dir = data_dir.map(std::path::Path::to_path_buf);
        if !self.open {
            return;
        }

        let mut open = self.open;
        crate::window::detached(
            ctx,
            "plot_data",
            "Plot Data",
            &mut open,
            [760.0, 520.0],
            |ui| {
                egui::Panel::left("plot_data_controls")
                    .resizable(true)
                    .default_size(240.0)
                    .show_inside(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            self.controls(ui);
                        });
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if self.dirty {
                        self.rebuild();
                        self.dirty = false;
                    }
                    crate::plot::show(&mut self.plot, ui);
                    // A legend click toggles which curve is highlighted (re-drawn
                    // emphasized on the next rebuild); clicking the active entry
                    // clears it.
                    if let Some(label) = crate::plot::take_legend_click(&mut self.plot) {
                        self.highlighted = if self.highlighted.as_deref() == Some(label.as_str()) {
                            None
                        } else {
                            Some(label)
                        };
                        self.dirty = true;
                    }
                });
            },
        );
        self.open = open;
        // The picker is its own sibling OS viewport (not nested in Plot Data's),
        // so it can be dragged outside the Plot Data window.
        self.file_picker(ctx);
    }

    /// The left-hand control column: the file selectors, stacking, averaging,
    /// and peak search.
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Plot Data");

        // A Feffit fit sent here overrides the file plot while shown.
        if let Some(ov) = &self.overlay {
            let label = ov.label.clone();
            ui.separator();
            ui.strong("Feffit fit");
            if ui
                .checkbox(&mut self.show_overlay, "Show Feffit fit (data + model)")
                .changed()
            {
                self.dirty = true;
            }
            ui.weak(label);
            if ui.button("Clear fit").clicked() {
                self.overlay = None;
                self.show_overlay = false;
                self.dirty = true;
            }
            ui.separator();
        }

        self.files_controls(ui);

        // k-weight re-reads any loaded k-space file (their χ is stored
        // unweighted) so the overlay tracks the slider.
        let kweight_relevant = self.loaded.iter().any(|t| t.item.applies_kweight());
        if kweight_relevant
            && ui
                .add(egui::Slider::new(&mut self.kweight, 0..=3).text("k-weight"))
                .changed()
        {
            self.reload_loaded();
            self.dirty = true;
        }

        ui.separator();
        if ui
            .add(egui::Slider::new(&mut self.stack, 0.0..=5.0).text("stack offset"))
            .changed()
        {
            self.dirty = true;
        }
        if ui
            .checkbox(&mut self.show_average, "Average of loaded")
            .changed()
        {
            self.dirty = true;
        }
        if ui
            .checkbox(&mut self.smooth5, "Average (5 points)")
            .on_hover_text("Display-smooth each curve with a 5-point moving average")
            .changed()
        {
            self.dirty = true;
        }
        if ui
            .checkbox(&mut self.dark_bg, "Change BG color (dark)")
            .changed()
        {
            self.dirty = true;
        }

        ui.separator();
        ui.label("Multiple peak catching");
        egui::ComboBox::from_id_salt("peak_mode")
            .selected_text(self.peak_mode.label())
            .show_ui(ui, |ui| {
                for m in PeakMode::ALL {
                    ui.selectable_value(&mut self.peak_mode, m, m.label());
                }
            });
        ui.horizontal(|ui| {
            ui.label("from");
            ui.add(egui::DragValue::new(&mut self.peak_lo).speed(0.5));
            ui.label("to");
            ui.add(egui::DragValue::new(&mut self.peak_hi).speed(0.5));
        });
        if self.peak_mode == PeakMode::AtY {
            ui.horizontal(|ui| {
                ui.label("target y");
                ui.add(egui::DragValue::new(&mut self.peak_target).speed(0.01));
            });
        }
        if ui.button("Catch peaks (loaded)").clicked() {
            self.catch_peaks();
            self.dirty = true;
        }
        if self.peaks.is_empty() {
            ui.weak("no peaks caught");
        } else {
            egui::ScrollArea::vertical()
                .max_height(140.0)
                .show(ui, |ui| {
                    for (label, x, y) in &self.peaks {
                        ui.monospace(format!("{label}:  x = {x:.4}, y = {y:.5}"));
                    }
                });
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Replot").clicked() {
                self.dirty = true;
            }
            if ui
                .button("Save in single file…")
                .on_hover_text("Write every displayed curve (stacked) to one file")
                .clicked()
            {
                self.save_composite();
            }
        });
    }

    /// Apply the chosen finder to every loaded file over `[peak_lo, peak_hi]`,
    /// collecting one `(label, x, y)` row per file — the original "Multiple peaks
    /// catching", which tabulates a peak position across all plotted spectra. A
    /// marker is drawn at each caught x on rebuild.
    fn catch_peaks(&mut self) {
        self.peaks.clear();
        for idx in 0..self.loaded.len() {
            // Borrow one file briefly: clone what the search needs so the push
            // below does not overlap the `self.loaded` borrow.
            let t = &self.loaded[idx];
            let label = t.label.clone();
            let x = t.x.clone();
            let y = if self.smooth5 {
                smooth5(&t.y)
            } else {
                t.y.clone()
            };
            let found = match self.peak_mode {
                PeakMode::Max => peak_in_range(&x, &y, self.peak_lo, self.peak_hi),
                PeakMode::Min => min_in_range(&x, &y, self.peak_lo, self.peak_hi),
                PeakMode::AtY => {
                    x_at_y_in_range(&x, &y, self.peak_target, self.peak_lo, self.peak_hi)
                        .map(|px| (px, self.peak_target))
                }
            };
            if let Some((px, py)) = found {
                self.peaks.push((label, px, py));
            }
        }
    }

    /// The "Files" control block: file-type / graph-item selectors, the picker
    /// button, the loaded-file list, and Clear Graph.
    fn files_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.strong("Files");

        // The file type / graph item are chosen in the File-selection window
        // (the original keeps the *.DAT picker there); show the current pick here.
        ui.label(format!(
            "Type {} · item {}",
            self.file_type.label(),
            self.graph_item.label()
        ));
        if ui.button("ADD or DEL Data Files…").clicked() {
            self.picker_open = true;
            // Open on the Data folder (under the Sub base) so the picker lists the
            // data files straight away, like the original XAFSView. Only seed it
            // the first time — keep wherever the user last browsed.
            if self.pick_dir.is_none() {
                self.pick_dir = self.data_dir.clone();
            }
            // Seed "Selected Data" with the currently-loaded (plotted) files so
            // they can be removed here: OK reconciles the loaded set to whatever
            // remains selected, so `<=` actually un-plots a file.
            self.pick_add = self.loaded.iter().map(|t| t.path.clone()).collect();
            self.sel_hi.clear();
            self.refresh_available();
        }

        if self.loaded.is_empty() {
            ui.weak("No files loaded — ADD Data Files to plot.");
        } else {
            ui.add_space(2.0);
            ui.label(format!("{} file(s) loaded:", self.loaded.len()));
            let mut remove = None;
            for (i, t) in self.loaded.iter().enumerate() {
                ui.horizontal(|ui| {
                    if crate::widgets::delete_box(ui)
                        .on_hover_text("Remove")
                        .clicked()
                    {
                        remove = Some(i);
                    }
                    ui.label(&t.label).on_hover_text(t.item.label());
                });
            }
            if let Some(i) = remove {
                self.loaded.remove(i);
                self.dirty = true;
            }
            if ui.button("Clear Graph").clicked() {
                self.loaded.clear();
                self.dirty = true;
            }
        }
        if !self.pick_status.is_empty() {
            ui.weak(&self.pick_status);
        }
    }

    /// The original **File selection** window: a two-pane transfer list. The left
    /// "Available Data" pane lists the folder's files of the chosen type/item; the
    /// right "Selected Data" pane holds the staged picks. `=>` / `<=` move the
    /// highlighted rows between panes (multi-select by clicking), `OK` loads the
    /// selection. Shown as its own OS viewport (via [`crate::window::detached`])
    /// so it can be dragged outside the Plot Data window.
    fn file_picker(&mut self, ctx: &egui::Context) {
        if !self.picker_open {
            return;
        }
        let mut keep_open = true;
        crate::window::detached(
            ctx,
            "plot_data_picker",
            "File selection",
            &mut keep_open,
            [660.0, 460.0],
            |ui| {
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    // Folder path bar.
                    ui.horizontal(|ui| {
                        if ui.button("📁 Browse…").clicked() {
                            // Default to where the picker is looking now, else the
                            // Data folder (under the Sub base).
                            let mut fd = rfd::FileDialog::new();
                            if let Some(dir) = self.pick_dir.as_ref().or(self.data_dir.as_ref()) {
                                fd = fd.set_directory(dir);
                            }
                            if let Some(dir) = fd.pick_folder() {
                                self.pick_dir = Some(dir);
                                // Keep the current selection (it may span folders);
                                // only the Available list and highlights follow the
                                // new folder.
                                self.avail_hi.clear();
                                self.sel_hi.clear();
                                self.refresh_available();
                            }
                        }
                        match &self.pick_dir {
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
                        // LEFT — Available Data, with the file-type / graph-item filters.
                        ui.vertical(|ui| {
                            ui.set_width(pane_w);
                            ui.horizontal(|ui| {
                                ui.strong("Available Data");
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        egui::ComboBox::from_id_salt("pd_ftype")
                                            .selected_text(self.file_type.label())
                                            .show_ui(ui, |ui| {
                                                for ft in FileType::ALL {
                                                    if ui
                                                        .selectable_value(
                                                            &mut self.file_type,
                                                            ft,
                                                            ft.label(),
                                                        )
                                                        .changed()
                                                    {
                                                        self.graph_item =
                                                            self.file_type.default_item();
                                                        self.avail_hi.clear();
                                                        self.refresh_available();
                                                    }
                                                }
                                            });
                                    },
                                );
                            });
                            egui::ComboBox::from_id_salt("pd_gitem")
                                .selected_text(self.graph_item.label())
                                .show_ui(ui, |ui| {
                                    for &gi in self.file_type.items() {
                                        if ui
                                            .selectable_value(&mut self.graph_item, gi, gi.label())
                                            .changed()
                                        {
                                            self.avail_hi.clear();
                                            self.refresh_available();
                                        }
                                    }
                                });
                            egui::ScrollArea::vertical()
                                .id_salt("avail_list")
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
                            ui.add_space(28.0);
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
                            // Pad to align the list top with the left pane (two header rows).
                            ui.add_space(
                                ui.spacing().interact_size.y + ui.spacing().item_spacing.y,
                            );
                            egui::ScrollArea::vertical()
                                .id_salt("sel_list")
                                .max_height(list_h)
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.set_min_width(pane_w);
                                    let staged = self.pick_add.clone();
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
                            if !self.pick_add.contains(&p) {
                                self.pick_add.push(p);
                            }
                        }
                        self.avail_hi.clear();
                        self.refresh_available();
                    }
                    if do_remove {
                        self.pick_add.retain(|p| !self.sel_hi.contains(p));
                        self.sel_hi.clear();
                        self.refresh_available();
                    }
                    if do_sort {
                        self.pick_add.sort();
                        self.sel_anchor = None;
                    }
                    if do_clear {
                        self.pick_add.clear();
                        self.sel_hi.clear();
                        self.refresh_available();
                    }
                    if do_ok {
                        self.load_staged();
                        self.avail_hi.clear();
                        self.sel_hi.clear();
                        self.picker_open = false;
                    }
                });
            },
        );
        // The OS window's close button (keep_open=false) closes the picker too.
        self.picker_open = self.picker_open && keep_open;
    }

    /// List the files in `pick_dir` that match the current graph item, excluding
    /// the ones already staged in "Selected Data" (which holds the loaded set
    /// while the picker is open). Sorted by name.
    fn refresh_available(&mut self) {
        // The lists are about to change, so any anchored row index is stale.
        self.avail_anchor = None;
        self.sel_anchor = None;
        self.available.clear();
        let Some(dir) = self.pick_dir.clone() else {
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
            // Skip hidden/metadata files: dotfiles like macOS AppleDouble "._*"
            // sidecars (created when copying to FAT/exFAT/SMB, visible on Windows)
            // and ".DS_Store" are not data and only clutter the list.
            if name.starts_with('.') {
                continue;
            }
            let taken = self.pick_add.contains(&path);
            if self.graph_item.matches(&name) && !taken {
                self.available.push(path);
            }
        }
        self.available.sort();
    }

    /// Reconcile the loaded set to the staged selection (`pick_add`): reuse the
    /// traces still selected (no re-read, so their original graph item is kept),
    /// load the newly-staged files as the current graph item, and drop the ones
    /// removed in the picker. Display order follows the staged list.
    fn load_staged(&mut self) {
        let (item, kw) = (self.graph_item, self.kweight);
        let staged = std::mem::take(&mut self.pick_add);
        // Move the current traces aside; still-selected files are reused from
        // here, anything left over is dropped (un-plotted).
        let mut prev = std::mem::take(&mut self.loaded);
        let mut failed = 0usize;
        for path in staged {
            if item == GraphItem::Result {
                // A `*.result` expands to many curves sharing one path, so the
                // reuse-by-path shortcut does not apply — always (re)expand it.
                match load_result(&path) {
                    Ok(traces) => self.loaded.extend(traces),
                    Err(_) => failed += 1,
                }
            } else if let Some(pos) = prev.iter().position(|t| t.path == path) {
                self.loaded.push(prev.remove(pos));
            } else {
                match load_trace(&path, item, kw) {
                    Ok(t) => self.loaded.push(t),
                    Err(_) => failed += 1,
                }
            }
        }
        let n = self.loaded.len();
        self.pick_status = if failed > 0 {
            format!("Showing {n} file(s); {failed} could not be read.")
        } else {
            format!("Showing {n} file(s).")
        };
        self.refresh_available();
        self.dirty = true;
    }

    /// Re-read every loaded file (e.g. after a k-weight change, since k-space
    /// files store unweighted χ).
    fn reload_loaded(&mut self) {
        let kw = self.kweight;
        for t in &mut self.loaded {
            // `*.result` curves carry final saved values (k-weight already baked
            // in), and one path maps to many curves, so they are not re-read.
            if t.item == GraphItem::Result {
                continue;
            }
            if let Ok(nt) = load_trace(&t.path, t.item, kw) {
                *t = nt;
            }
        }
    }

    /// The current x/y axis labels — from a loaded `*.result`'s saved labels if
    /// present, else the loaded files' own graph item (so they track the data
    /// even if the picker's selector has since moved); falls back to the selector
    /// when nothing is loaded.
    fn axis_labels(&self) -> (String, String) {
        match self.loaded.first() {
            Some(t) => match &t.axis {
                Some((x, y)) => (x.clone(), y.clone()),
                None => (t.item.x_label().to_owned(), t.item.y_label().to_owned()),
            },
            None => (
                self.graph_item.x_label().to_owned(),
                self.graph_item.y_label().to_owned(),
            ),
        }
    }

    /// Build the displayed traces from the loaded files — palette colours and
    /// 5-point smoothing applied, before any stacking offset. One source of
    /// truth for both drawing and saving.
    fn built_traces(&self) -> Vec<(String, Vec<f64>, Vec<f64>, Color32)> {
        // Bright curves on the dark canvas; the muted tab10 on the white "Change
        // BG" canvas, where the bright palette would wash out.
        let palette = if self.dark_bg {
            crate::plot::PALETTE
        } else {
            crate::plot::PALETTE_LIGHT
        };
        let mut traces: Vec<(String, Vec<f64>, Vec<f64>, Color32)> = Vec::new();
        for t in &self.loaded {
            let y = if self.smooth5 {
                smooth5(&t.y)
            } else {
                t.y.clone()
            };
            let color = palette[traces.len() % palette.len()];
            traces.push((t.label.clone(), t.x.clone(), y, color));
        }
        traces
    }

    /// Every displayed curve with its stacking offset applied — exactly what is
    /// drawn (minus the average overlay). Feeds "Save in single file".
    fn displayed_composite(&self) -> Vec<(String, Vec<f64>, Vec<f64>)> {
        self.built_traces()
            .into_iter()
            .enumerate()
            .map(|(idx, (label, x, y, _color))| (label, x, stack_offset_y(y, idx, self.stack)))
            .collect()
    }

    /// "Save in single file": write every displayed (stacked) curve to one file
    /// the user picks.
    fn save_composite(&mut self) {
        let traces = self.displayed_composite();
        if traces.is_empty() {
            self.pick_status = "Nothing to save — load a file first.".to_owned();
            return;
        }
        let mut dlg = rfd::FileDialog::new()
            // The original's wide-table layout (`XANES.dat` / `k3-EXAFS.dat` /
            // `RDF.dat`) is the default; `.result` keeps the round-trippable
            // labelled-block format that `load_result` reads back.
            .add_filter("Results table (wide)", &["dat"])
            .add_filter("Plot Data result", &["result"])
            .set_file_name("plotdata.dat");
        // Default to the Results folder, like the original XAFSView.
        if let Some(dir) = &self.results_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.save_file() else {
            return;
        };
        let (xlabel, ylabel) = self.axis_labels();
        // `.result` → the labelled-block format; any other extension → the
        // original wide table.
        let as_blocks = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("result"));
        let result = if as_blocks {
            write_composite(&path, &traces, &xlabel, &ylabel)
        } else {
            write_composite_table(&path, &traces, &xlabel)
        };
        self.pick_status = match result {
            Ok(()) => format!(
                "Saved {} curve(s) ({}) to {}.",
                traces.len(),
                if as_blocks { "blocks" } else { "table" },
                file_name_of(&path)
            ),
            Err(e) => format!("Save failed: {e}"),
        };
    }

    /// Rebuild every plotted curve from the loaded files and current settings.
    /// Add one trace to the plot, drawn emphasized when it is the curve the user
    /// highlighted by clicking its legend entry (thicker line, on top).
    fn add_trace(&mut self, label: &str, x: &[f64], y: &[f64], color: Color32) {
        if self.highlighted.as_deref() == Some(label) {
            self.plot.add_emphasized_curve(x, y, color, label);
        } else {
            self.plot.add_curve_with_legend(x, y, color, label);
        }
    }

    fn rebuild(&mut self) {
        self.plot.clear();

        // Background colour (the "Change BG color" swap): a dark canvas or the
        // default light one, applied via the shared `set_theme` so the axes and
        // grid track the background too. `fg` is the overlay (average) colour,
        // kept legible against the chosen background.
        let fg = if self.dark_bg {
            Color32::from_gray(0xe0)
        } else {
            Color32::from_rgb(0x20, 0x20, 0x20)
        };
        crate::plot::set_theme(&mut self.plot, !self.dark_bg);

        // A sent Feffit fit takes over the plot (its space/axes differ from the
        // file items), so draw it alone and skip the file traces.
        if self.show_overlay
            && let Some(ov) = &self.overlay
        {
            self.plot.set_graph_x_label(ov.xlabel);
            self.plot.set_graph_y_label(ov.ylabel, YAxis::Left);
            if !ov.x.is_empty() {
                self.plot
                    .add_curve_with_legend(&ov.x, &ov.data, FIT_DATA, "fit data");
                // "Only FT" overlays carry no model curve.
                if !ov.model.is_empty() {
                    self.plot
                        .add_curve_with_legend(&ov.x, &ov.model, FIT_MODEL, "fit model");
                }
            }
            return;
        }

        let (xlabel, ylabel) = self.axis_labels();
        self.plot.set_graph_x_label(xlabel);
        self.plot.set_graph_y_label(ylabel, YAxis::Left);

        // Palette colours and smoothing, before any stacking offset (shared with
        // the "Save in single file" path).
        let traces = self.built_traces();

        // The averaged trace is computed on the un-stacked data, before offsets.
        let avg = if self.show_average && traces.len() > 1 {
            let refs: Vec<(&[f64], &[f64])> = traces
                .iter()
                .map(|(_, x, y, _)| (x.as_slice(), y.as_slice()))
                .collect();
            average_curves(&refs)
        } else {
            None
        };

        // Stack the individual traces (offset i·stack) and draw.
        for (idx, (label, x, y, color)) in traces.into_iter().enumerate() {
            let ys = stack_offset_y(y, idx, self.stack);
            self.add_trace(&label, &x, &ys, color);
        }

        if let Some((x, y)) = avg {
            self.add_trace("average", &x, &y, fg);
        }

        for (_, px, _) in &self.peaks {
            self.plot
                .add_x_marker(*px, Color32::from_rgb(0x80, 0x80, 0x80));
        }
    }
}

/// Render a click-selectable file list with toggle + shift-range multi-select.
///
/// A plain click toggles one row and becomes the new anchor. A shift-click (when
/// an anchor exists) adds every row in the inclusive range between the anchor
/// and the clicked row — the standard "click one, shift-click another, select
/// everything between" behaviour — without moving the anchor, so successive
/// shift-clicks all extend from the same origin.
fn select_list(
    ui: &mut egui::Ui,
    list: &[PathBuf],
    hi: &mut HashSet<PathBuf>,
    anchor: &mut Option<usize>,
) {
    let shift = ui.input(|i| i.modifiers.shift);
    for (idx, path) in list.iter().enumerate() {
        let selected = hi.contains(path);
        if ui.selectable_label(selected, file_name_of(path)).clicked() {
            match (shift, *anchor) {
                (true, Some(a)) => {
                    for p in &list[a.min(idx)..=a.max(idx)] {
                        hi.insert(p.clone());
                    }
                }
                _ => {
                    if !hi.remove(path) {
                        hi.insert(path.clone());
                    }
                    *anchor = Some(idx);
                }
            }
        }
    }
}

/// The bare file name of `path`, for picker lists and legends.
fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Write the displayed curves to one file as labelled two-column (`x  y`)
/// blocks separated by a blank line, so each block parses independently even
/// when the curves are on different x grids.
fn write_composite(
    path: &std::path::Path,
    traces: &[(String, Vec<f64>, Vec<f64>)],
    xlabel: &str,
    ylabel: &str,
) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# XAFSView Plot Data — {} curve(s)", traces.len());
    // Machine-parseable axis lines so `load_result` can restore the labels (they
    // are shared by every curve in the composite).
    let _ = writeln!(s, "# x-axis: {xlabel}");
    let _ = writeln!(s, "# y-axis: {ylabel}");
    for (i, (label, x, y)) in traces.iter().enumerate() {
        // A blank line separates curves (but not the first from the header), so
        // each curve parses back as its own block.
        if i > 0 {
            s.push('\n');
        }
        let _ = writeln!(s, "# curve {}: {label}", i + 1);
        for (xx, yy) in x.iter().zip(y) {
            let _ = writeln!(s, "{xx:14.6}  {yy:18.10}");
        }
    }
    std::fs::write(path, s)
}

/// True when every curve shares a bit-identical x grid (same length and same
/// values). This — not the data type — is what the original uses to pick the
/// wide-table layout: the FT outputs (`k3-EXAFS`, `RDF`) live on one fixed
/// uniform grid and collapse to a shared x column, whereas per-scan XANES
/// energies differ by calibration and stay as separate `(x, y)` pairs.
fn shared_x_grid(traces: &[(String, Vec<f64>, Vec<f64>)]) -> bool {
    let mut grids = traces.iter().map(|(_, x, _)| x);
    match grids.next() {
        Some(first) => grids.all(|x| x == first),
        None => false,
    }
}

/// Write the displayed curves as the original XAFSView "Save in single file"
/// **wide table**: TAB-separated columns, a header label row, and `{:.6}` fixed
/// values, with CRLF line endings — the LabVIEW byte layout of `XANES.dat` /
/// `k3-EXAFS.dat` / `RDF.dat`.
///
/// When every curve shares a bit-identical x grid the table is one shared x
/// column followed by one y column per curve ([`shared_x_grid`] → `k3-EXAFS` /
/// `RDF`); otherwise each curve contributes its own `(x, y)` column pair
/// (`XANES`, whose per-scan energy grids differ). The header row carries a
/// trailing tab and the data rows do not, matching the original exactly; ragged
/// columns (unequal lengths in the pairs layout) are padded with empty cells so
/// the table stays rectangular.
///
/// The x-axis header uses our axis-label convention (e.g. `R (Å)`), not the
/// LabVIEW strings (`R (angstrom)`); the per-curve labels are the file names,
/// the same role the scan names play in the original.
fn write_composite_table(
    path: &std::path::Path,
    traces: &[(String, Vec<f64>, Vec<f64>)],
    xlabel: &str,
) -> std::io::Result<()> {
    let mut s = String::new();

    let mut push_row = |cells: &[String], trailing_tab: bool| {
        s.push_str(&cells.join("\t"));
        if trailing_tab {
            s.push('\t');
        }
        s.push_str("\r\n");
    };

    if shared_x_grid(traces) {
        // Header: x-label then one curve label each, with the trailing tab.
        let mut hdr = vec![xlabel.to_owned()];
        hdr.extend(traces.iter().map(|(label, ..)| label.clone()));
        push_row(&hdr, true);

        // Rows: the shared x value then each curve's y at that index.
        let x = &traces[0].1;
        for (row, &xx) in x.iter().enumerate() {
            let mut cells = vec![format!("{xx:.6}")];
            cells.extend(traces.iter().map(|(_, _, y)| format!("{:.6}", y[row])));
            push_row(&cells, false);
        }
    } else {
        // Header: an (x-label, curve-label) pair per curve, with the trailing tab.
        let mut hdr = Vec::with_capacity(traces.len() * 2);
        for (label, ..) in traces {
            hdr.push(xlabel.to_owned());
            hdr.push(label.clone());
        }
        push_row(&hdr, true);

        // Rows: each curve's (x, y) side by side; shorter curves pad with blanks.
        let nrows = traces.iter().map(|(_, x, _)| x.len()).max().unwrap_or(0);
        for row in 0..nrows {
            let mut cells = Vec::with_capacity(traces.len() * 2);
            for (_, x, y) in traces {
                if row < x.len() {
                    cells.push(format!("{:.6}", x[row]));
                    cells.push(format!("{:.6}", y[row]));
                } else {
                    cells.push(String::new());
                    cells.push(String::new());
                }
            }
            push_row(&cells, false);
        }
    }
    std::fs::write(path, s)
}

/// A 5-point centered moving average (the original's "Average (5 points)"), with
/// the window shrinking at the ends. Display-only smoothing.
fn smooth5(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(2);
            let hi = (i + 2).min(n - 1);
            let win = &y[lo..=hi];
            win.iter().sum::<f64>() / win.len() as f64
        })
        .collect()
}

/// Raise trace `idx`'s y-values by its stacking offset (`idx · stack`); trace 0
/// is left unshifted. The single definition of the waterfall offset, shared by
/// the draw path ([`PlotDataWindow::rebuild`]) and the save path
/// ([`PlotDataWindow::displayed_composite`]) so a "Save in single file" carries
/// exactly the vertical stacking the user sees on screen.
fn stack_offset_y(y: Vec<f64>, idx: usize, stack: f64) -> Vec<f64> {
    let off = idx as f64 * stack;
    if off != 0.0 {
        y.iter().map(|v| v + off).collect()
    } else {
        y
    }
}

/// The minimum `(x, y)` of `y` over `x ∈ [lo, hi]` (inclusive); `None` when no
/// sample falls in the range. Mirrors [`peak_in_range`] for the minimum.
fn min_in_range(x: &[f64], y: &[f64], lo: f64, hi: f64) -> Option<(f64, f64)> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let mut best: Option<(f64, f64)> = None;
    for (&xi, &yi) in x.iter().zip(y) {
        if xi < lo || xi > hi {
            continue;
        }
        match best {
            Some((_, by)) if yi >= by => {}
            _ => best = Some((xi, yi)),
        }
    }
    best
}

/// Interpolated x where `y` first crosses `target`, restricted to the samples
/// with `x ∈ [lo, hi]`. Range-limits the arrays, then defers to [`x_at_y`].
fn x_at_y_in_range(x: &[f64], y: &[f64], target: f64, lo: f64, hi: f64) -> Option<f64> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let (xs, ys): (Vec<f64>, Vec<f64>) = x
        .iter()
        .zip(y)
        .filter(|&(&xi, _)| xi >= lo && xi <= hi)
        .map(|(&xi, &yi)| (xi, yi))
        .unzip();
    x_at_y(&xs, &ys, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_in_range_finds_min_and_respects_range() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [5.0, -2.0, 3.0, -9.0, 1.0];
        // -9 at x=3 is the global min but lies outside [0, 2]; in-range min is -2.
        assert_eq!(min_in_range(&x, &y, 0.0, 2.0), Some((1.0, -2.0)));
        // Full range sees the global minimum.
        assert_eq!(min_in_range(&x, &y, 0.0, 4.0), Some((3.0, -9.0)));
        // Empty range yields nothing.
        assert_eq!(min_in_range(&x, &y, 10.0, 20.0), None);
    }

    #[test]
    fn smooth5_is_a_shrinking_window_moving_average() {
        // The interior point averages its 5-sample window; the ends shrink it.
        let s = smooth5(&[0.0, 0.0, 10.0, 0.0, 0.0]);
        assert!((s[2] - 2.0).abs() < 1e-12, "centre = 10/5: {s:?}");
        assert!((s[0] - 10.0 / 3.0).abs() < 1e-12, "left edge = 10/3: {s:?}");
        // A constant signal is unchanged, and an empty input stays empty.
        let c = smooth5(&[5.0, 5.0, 5.0, 5.0]);
        assert!(c.iter().all(|v| (v - 5.0).abs() < 1e-12), "constant: {c:?}");
        assert!(smooth5(&[]).is_empty());
    }

    #[test]
    fn x_at_y_in_range_restricts_then_interpolates() {
        // A line y = x: crossing y = 2.5 is at x = 2.5 by linear interpolation.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let got = x_at_y_in_range(&x, &y, 2.5, 0.0, 4.0).expect("crossing");
        assert!((got - 2.5).abs() < 1e-9, "got {got}");
        // The same target outside the restricted window is not found.
        assert_eq!(x_at_y_in_range(&x, &y, 2.5, 3.0, 4.0), None);
    }

    #[test]
    fn result_round_trips_write_composite_into_one_trace_per_curve() {
        use crate::plot_files::{GraphItem, load_result};

        let p = std::env::temp_dir().join(format!("xafsview_rt_{}.result", std::process::id()));
        let traces = vec![
            ("first".to_owned(), vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]),
            ("second".to_owned(), vec![0.0, 0.5], vec![9.0, 8.0]),
        ];
        write_composite(&p, &traces, "k (Å⁻¹)", "kʷ·χ(k)").expect("write composite");

        let loaded = load_result(&p).expect("load result");
        assert_eq!(loaded.len(), 2, "one trace per saved curve");
        // Labels and data round-trip per curve.
        assert_eq!(loaded[0].label, "first");
        assert_eq!(loaded[0].x, vec![0.0, 1.0, 2.0]);
        assert_eq!(loaded[0].y, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded[1].label, "second");
        assert_eq!(loaded[1].y, vec![9.0, 8.0]);
        // Both curves are tagged Result and carry the saved axis labels.
        for t in &loaded {
            assert_eq!(t.item, GraphItem::Result);
            assert_eq!(
                t.axis.as_ref().map(|(x, y)| (x.as_str(), y.as_str())),
                Some(("k (Å⁻¹)", "kʷ·χ(k)"))
            );
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn stack_offset_raises_each_trace_and_is_baked_into_the_save() {
        let y = vec![1.0, 2.0, 3.0];
        // Trace 0 is never shifted, regardless of the stack value.
        assert_eq!(stack_offset_y(y.clone(), 0, 2.5), y);
        // Trace i is raised by i·stack — what the draw path and the save path
        // (displayed_composite) both call, so a saved file carries the stacking.
        assert_eq!(stack_offset_y(y.clone(), 1, 2.5), vec![3.5, 4.5, 5.5]);
        assert_eq!(stack_offset_y(y.clone(), 2, 2.5), vec![6.0, 7.0, 8.0]);
        // A zero offset leaves the values untouched (no stacking, no change).
        assert_eq!(stack_offset_y(y.clone(), 3, 0.0), y);
    }

    #[test]
    fn write_composite_blocks_parse_back_independently() {
        use feffit::xasdata::ColumnFile;
        let p = std::env::temp_dir().join(format!("xafsview_composite_{}.dat", std::process::id()));
        let traces = vec![
            ("a".to_owned(), vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]),
            ("b".to_owned(), vec![0.0, 0.5], vec![9.0, 8.0]),
        ];
        write_composite(&p, &traces, "x", "y").expect("write composite");

        let text = std::fs::read_to_string(&p).expect("read back");
        assert!(text.contains("2 curve(s)"), "header: {text}");
        // The blank line separates the two curves into independently parseable
        // two-column blocks (different lengths and grids).
        let blocks: Vec<&str> = text.split("\n\n").collect();
        assert_eq!(blocks.len(), 2, "one blank line between two blocks");
        let b0 = ColumnFile::from_text(blocks[0]).expect("first block");
        assert_eq!(b0.ncols(), 2);
        assert_eq!(b0.column(0).unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(b0.column(1).unwrap(), &[1.0, 2.0, 3.0]);
        let b1 = ColumnFile::from_text(blocks[1]).expect("second block");
        assert_eq!(b1.column(0).unwrap(), &[0.0, 0.5]);
        assert_eq!(b1.column(1).unwrap(), &[9.0, 8.0]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn shared_x_grid_is_bit_equality_across_curves() {
        // Same grid ⇒ shared (k3-EXAFS / RDF); a single differing sample ⇒ pairs
        // (XANES per-scan calibration). Empty set ⇒ not shared.
        let same = vec![
            ("a".to_owned(), vec![0.0, 1.0], vec![1.0, 2.0]),
            ("b".to_owned(), vec![0.0, 1.0], vec![3.0, 4.0]),
        ];
        assert!(shared_x_grid(&same));
        let diff = vec![
            ("a".to_owned(), vec![0.0, 1.0], vec![1.0, 2.0]),
            ("b".to_owned(), vec![0.0, 1.1], vec![3.0, 4.0]),
        ];
        assert!(!shared_x_grid(&diff));
        assert!(!shared_x_grid(&[]));
    }

    #[test]
    fn composite_table_shared_x_writes_one_x_column_and_crlf() {
        let p =
            std::env::temp_dir().join(format!("xafsview_tbl_shared_{}.dat", std::process::id()));
        let traces = vec![
            ("s0".to_owned(), vec![0.05, 0.10], vec![1.0, 2.0]),
            ("s1".to_owned(), vec![0.05, 0.10], vec![3.0, 4.0]),
        ];
        write_composite_table(&p, &traces, "k (Å⁻¹)").expect("write table");

        let text = std::fs::read_to_string(&p).expect("read back");
        let lines: Vec<&str> = text.split("\r\n").collect();
        // CRLF endings with a trailing empty element after the last "\r\n".
        assert_eq!(*lines.last().unwrap(), "", "file ends with CRLF");
        // Header: x-label then one label per curve, with a trailing tab.
        assert_eq!(lines[0], "k (Å⁻¹)\ts0\ts1\t");
        // Rows: shared x column + one y per curve, {:.6}, no trailing tab.
        assert_eq!(lines[1], "0.050000\t1.000000\t3.000000");
        assert_eq!(lines[2], "0.100000\t2.000000\t4.000000");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn composite_table_pairs_when_grids_differ() {
        let p = std::env::temp_dir().join(format!("xafsview_tbl_pairs_{}.dat", std::process::id()));
        // Different x grids (and different lengths) ⇒ (x, y) pair per curve, with
        // the shorter curve padded with blank cells so the table stays rectangular.
        let traces = vec![
            (
                "e0".to_owned(),
                vec![6339.0, 6340.0, 6341.0],
                vec![-0.0015, 0.0, 1.0],
            ),
            ("e1".to_owned(), vec![6339.1, 6340.1], vec![0.5, 0.6]),
        ];
        write_composite_table(&p, &traces, "Energy (eV)").expect("write table");

        let text = std::fs::read_to_string(&p).expect("read back");
        let lines: Vec<&str> = text.split("\r\n").collect();
        // Header: an (x-label, curve-label) pair per curve, trailing tab.
        assert_eq!(lines[0], "Energy (eV)\te0\tEnergy (eV)\te1\t");
        assert_eq!(lines[1], "6339.000000\t-0.001500\t6339.100000\t0.500000");
        assert_eq!(lines[2], "6340.000000\t0.000000\t6340.100000\t0.600000");
        // Third row: second curve exhausted ⇒ its two cells are blank.
        assert_eq!(lines[3], "6341.000000\t1.000000\t\t");
        let _ = std::fs::remove_file(&p);
    }
}
