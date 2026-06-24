//! The standalone **Plot Data** window: overlay any reduction stage of several
//! loaded groups on one plot, with vertical stacking, an averaged trace, and a
//! peak readout. Mirrors XAFSView's *Plot Data* window.
//!
//! It owns its own plot (separate from the tabs' shared plot) so it can
//! float independently. Save / zoom / legend come from the siplot toolbar; the
//! data work (averaging, peak finding) is the headless [`xasdata::batch`] code.

use std::path::PathBuf;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use siplot::YAxis;
use xasdata::{PreEdgeParams, XasGroup, average_curves, normalize, peak_in_range, x_at_y};

use crate::plot_files::{FileType, GraphItem, LoadedTrace, load_trace};

/// Which reduction stage to overlay.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlotItem {
    /// Raw `μ(E)`.
    Mu,
    /// Edge-step normalized `μ(E)`.
    Norm,
    /// Flattened normalized `μ(E)`.
    Flat,
    /// Derivative `dμ/dE`.
    Deriv,
    /// k-weighted `χ(k)`.
    Chik,
    /// `|χ(R)|`.
    ChiR,
}

impl PlotItem {
    const ALL: [PlotItem; 6] = [
        PlotItem::Mu,
        PlotItem::Norm,
        PlotItem::Flat,
        PlotItem::Deriv,
        PlotItem::Chik,
        PlotItem::ChiR,
    ];

    fn label(self) -> &'static str {
        match self {
            PlotItem::Mu => "μ(E)",
            PlotItem::Norm => "normalized μ(E)",
            PlotItem::Flat => "flattened μ(E)",
            PlotItem::Deriv => "dμ/dE",
            PlotItem::Chik => "kʷ·χ(k)",
            PlotItem::ChiR => "|χ(R)|",
        }
    }

    fn x_label(self) -> &'static str {
        match self {
            PlotItem::Mu | PlotItem::Norm | PlotItem::Flat | PlotItem::Deriv => "Energy (eV)",
            PlotItem::Chik => "k (Å⁻¹)",
            PlotItem::ChiR => "R (Å)",
        }
    }

    /// The `(x, y)` arrays for this item from a group, applying k-weighting for
    /// the `χ(k)` view. `None` when the group hasn't reached this stage yet.
    fn xy(self, g: &XasGroup, kweight: i32) -> Option<(Vec<f64>, Vec<f64>)> {
        match self {
            PlotItem::Mu => (!g.energy.is_empty()).then(|| (g.energy.clone(), g.mu.clone())),
            PlotItem::Norm => g.norm.as_ref().map(|v| (g.energy.clone(), v.clone())),
            PlotItem::Flat => g.flat.as_ref().map(|v| (g.energy.clone(), v.clone())),
            PlotItem::Deriv => g.dmude.as_ref().map(|v| (g.energy.clone(), v.clone())),
            PlotItem::Chik => match (&g.k, &g.chi) {
                (Some(k), Some(chi)) => {
                    let y = k
                        .iter()
                        .zip(chi)
                        .map(|(&kk, &c)| c * kk.powi(kweight))
                        .collect();
                    Some((k.clone(), y))
                }
                _ => None,
            },
            PlotItem::ChiR => match (&g.r, &g.chir_mag) {
                (Some(r), Some(mag)) => Some((r.clone(), mag.clone())),
                _ => None,
            },
        }
    }
}

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
/// shown it takes over the plot (its axes differ from the group items).
struct FitOverlay {
    label: String,
    xlabel: &'static str,
    ylabel: &'static str,
    x: Vec<f64>,
    data: Vec<f64>,
    model: Vec<f64>,
}

/// The Plot Data "Normalize options" (그림 1-5-1): override the reduction's
/// pre/post-edge normalization for the Norm/Flat/dμ view, with an optional
/// NEXAFS peak-normalization. Display-only — the session groups are untouched.
struct NormOptions {
    /// Apply this override instead of each group's reduction normalization.
    on: bool,
    /// Let `pre_edge` find E₀ (vs the explicit `e0`).
    e0_auto: bool,
    e0: f64,
    pre1: f64,
    pre2: f64,
    norm1: f64,
    norm2: f64,
    /// NEXAFS: normalize so the largest peak is 1, instead of the edge step.
    maxpoint: bool,
}

impl Default for NormOptions {
    fn default() -> Self {
        // larch's pre/post-edge ranges (eV relative to E₀).
        Self {
            on: false,
            e0_auto: true,
            e0: 0.0,
            pre1: -150.0,
            pre2: -30.0,
            norm1: 150.0,
            norm2: 800.0,
            maxpoint: false,
        }
    }
}

impl NormOptions {
    /// Build the [`PreEdgeParams`] for these options (mirrors the Autobk tab's
    /// `pre_params`).
    fn params(&self) -> PreEdgeParams {
        let mut p = PreEdgeParams::default();
        if !self.e0_auto {
            p.e0 = Some(self.e0);
        }
        p.pre1 = Some(self.pre1);
        p.pre2 = Some(self.pre2);
        p.norm1 = Some(self.norm1);
        p.norm2 = Some(self.norm2);
        p
    }
}

/// The Plot Data window state and its own plot.
pub struct PlotDataWindow {
    /// Whether the window is shown.
    pub open: bool,
    plot: crate::plot::Plot,
    item: PlotItem,
    kweight: i32,
    /// Per-group "show this trace" flags, kept the same length as the session's
    /// group list.
    selected: Vec<bool>,
    /// Vertical offset added to trace `i` (`i · stack`), in data units.
    stack: f64,
    show_average: bool,
    /// "Average (5 points)": display-smooth each curve (5-point moving average).
    smooth5: bool,
    /// "Change BG color": dark plot background (the original's black/white swap).
    dark_bg: bool,
    /// "Normalize options" (그림 1-5-1).
    norm: NormOptions,
    peak_lo: f64,
    peak_hi: f64,
    /// Which feature the Multiple peaks catching window locates.
    peak_mode: PeakMode,
    /// Target y for the "x at y" mode (0.5 = normalized half-step).
    peak_target: f64,
    /// One `(group label, x, y)` row per selected group from the last catch.
    peaks: Vec<(String, f64, f64)>,
    /// A Feffit fit sent here via "Send to Plot Data", if any.
    overlay: Option<FitOverlay>,
    /// Whether the sent fit takes over the plot (vs the group items).
    show_overlay: bool,

    // --- file overlay (the original's file-based Plot Data) -----------------
    /// The selected *File type* for browsing/loading files.
    file_type: FileType,
    /// The *Graph item* under `file_type` (its file-name suffix + plot recipe).
    graph_item: GraphItem,
    /// Files loaded for overlay, drawn additively with the group traces.
    loaded: Vec<LoadedTrace>,
    /// Whether the "Add / remove data files" picker is open.
    picker_open: bool,
    /// The folder the picker browses.
    pick_dir: Option<PathBuf>,
    /// Files in `pick_dir` matching the current graph item, minus `pick_add`.
    available: Vec<PathBuf>,
    /// Files staged in the picker to add on OK.
    pick_add: Vec<PathBuf>,
    /// Sort the available list alphabetically.
    pick_sort: bool,
    /// Outcome of the last load (shown in the Files section).
    pick_status: String,

    /// Set whenever the overlay needs rebuilding (control change or new data).
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
            item: PlotItem::Norm,
            kweight: 2,
            selected: Vec::new(),
            stack: 0.0,
            show_average: false,
            smooth5: false,
            // Dark by default, matching the cohesive dark canvas every other
            // plot uses (the checkbox still flips to the white "Change BG" mode).
            dark_bg: true,
            norm: NormOptions::default(),
            peak_lo: 0.0,
            peak_hi: 0.0,
            peak_mode: PeakMode::Max,
            peak_target: 0.5,
            peaks: Vec::new(),
            overlay: None,
            show_overlay: false,
            file_type: FileType::Chi,
            graph_item: FileType::Chi.default_item(),
            loaded: Vec::new(),
            picker_open: false,
            pick_dir: None,
            available: Vec::new(),
            pick_add: Vec::new(),
            pick_sort: true,
            pick_status: String::new(),
            dirty: true,
        }
    }

    /// Request a rebuild on the next show — call after the loaded groups or their
    /// reduction stages change (e.g. after a batch AUTOBK).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Take a Feffit fit's data + model curves (the Feffit form's "Send to plot
    /// data"). The window opens showing the fit; untick "Show Feffit fit" or
    /// "Clear fit" to return to the group plot.
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

    /// Render the window over `groups` (the session's spectra).
    pub fn show(&mut self, ctx: &egui::Context, groups: &[XasGroup]) {
        // Keep the selection vector aligned with the group list; brand-new groups
        // start selected so they appear without a click.
        if self.selected.len() != groups.len() {
            self.selected.resize(groups.len(), true);
            self.dirty = true;
        }
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
                            self.controls(ui, groups);
                        });
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if self.dirty {
                        self.rebuild(groups);
                        self.dirty = false;
                    }
                    crate::plot::show(&mut self.plot, ui);
                });
                // The file picker floats over the window while open.
                self.file_picker(ui);
            },
        );
        self.open = open;
    }

    /// The left-hand control column: item selector, group checkboxes, stacking,
    /// averaging, and peak search.
    fn controls(&mut self, ui: &mut egui::Ui, groups: &[XasGroup]) {
        ui.heading("Plot Data");

        // A Feffit fit sent here overrides the group items while shown.
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

        egui::ComboBox::from_label("Array")
            .selected_text(self.item.label())
            .show_ui(ui, |ui| {
                for it in PlotItem::ALL {
                    if ui
                        .selectable_value(&mut self.item, it, it.label())
                        .changed()
                    {
                        self.dirty = true;
                    }
                }
            });
        // k-weight is relevant to the group χ(k) view and to any loaded k-space
        // file; changing it re-reads the loaded files (their χ is stored
        // unweighted) so the overlay tracks the slider.
        let kweight_relevant = matches!(self.item, PlotItem::Chik)
            || self.loaded.iter().any(|t| t.item.applies_kweight());
        if kweight_relevant
            && ui
                .add(egui::Slider::new(&mut self.kweight, 0..=3).text("k-weight"))
                .changed()
        {
            self.reload_loaded();
            self.dirty = true;
        }

        self.files_controls(ui);

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Groups");
            if ui.small_button("all").clicked() {
                self.selected.iter_mut().for_each(|s| *s = true);
                self.dirty = true;
            }
            if ui.small_button("none").clicked() {
                self.selected.iter_mut().for_each(|s| *s = false);
                self.dirty = true;
            }
        });
        if groups.is_empty() {
            ui.weak("No groups loaded.");
        }
        for (i, g) in groups.iter().enumerate() {
            if ui.checkbox(&mut self.selected[i], &g.label).changed() {
                self.dirty = true;
            }
        }

        ui.separator();
        if ui
            .add(egui::Slider::new(&mut self.stack, 0.0..=5.0).text("stack offset"))
            .changed()
        {
            self.dirty = true;
        }
        if ui
            .checkbox(&mut self.show_average, "Average of selected")
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
        egui::CollapsingHeader::new("Normalize options")
            .default_open(false)
            .show(ui, |ui| {
                let mut changed = ui
                    .checkbox(&mut self.norm.on, "Apply (override reduction)")
                    .on_hover_text(
                        "Re-normalize the Norm/Flat/dμ view from these settings, \
                         leaving the loaded groups unchanged",
                    )
                    .changed();
                changed |= ui.checkbox(&mut self.norm.e0_auto, "auto E₀").changed();
                if !self.norm.e0_auto {
                    ui.horizontal(|ui| {
                        ui.label("E₀");
                        changed |= ui
                            .add(egui::DragValue::new(&mut self.norm.e0).speed(0.5))
                            .changed();
                    });
                }
                egui::Grid::new("plot_norm_ranges")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("pre1");
                        changed |= ui
                            .add(egui::DragValue::new(&mut self.norm.pre1).speed(1.0))
                            .changed();
                        ui.label("pre2");
                        changed |= ui
                            .add(egui::DragValue::new(&mut self.norm.pre2).speed(1.0))
                            .changed();
                        ui.end_row();
                        ui.label("norm1");
                        changed |= ui
                            .add(egui::DragValue::new(&mut self.norm.norm1).speed(1.0))
                            .changed();
                        ui.label("norm2");
                        changed |= ui
                            .add(egui::DragValue::new(&mut self.norm.norm2).speed(1.0))
                            .changed();
                        ui.end_row();
                    });
                changed |= ui
                    .checkbox(&mut self.norm.maxpoint, "NEXAFS: normalize to max peak")
                    .changed();
                if changed {
                    self.dirty = true;
                }
            });

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
        if ui.button("Catch peaks (selected)").clicked() {
            self.catch_peaks(groups);
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
        if ui.button("Replot").clicked() {
            self.dirty = true;
        }
    }

    /// The `(x, y)` arrays to plot for `g` under the current item, applying the
    /// "Normalize options" override (Norm/Flat/dμ recomputed from a throwaway
    /// clone) when active; otherwise the group's own reduction arrays.
    fn series_for(&self, g: &XasGroup) -> Option<(Vec<f64>, Vec<f64>)> {
        if self.norm.on
            && matches!(self.item, PlotItem::Norm | PlotItem::Flat | PlotItem::Deriv)
            && !g.energy.is_empty()
        {
            let mut tmp = g.clone();
            normalize(&mut tmp, &self.norm.params());
            if self.norm.maxpoint && self.item == PlotItem::Norm {
                return Some((tmp.energy.clone(), peak_normalized(&tmp)));
            }
            return self.item.xy(&tmp, self.kweight);
        }
        self.item.xy(g, self.kweight)
    }

    /// The `(x, y)` actually drawn for `g`: [`series_for`](Self::series_for) plus
    /// the optional 5-point display smoothing. The single source both the trace
    /// loop and [`catch_peaks`](Self::catch_peaks) read, so a caught peak (and its
    /// marker) lands on the same curve the user sees.
    fn displayed_series(&self, g: &XasGroup) -> Option<(Vec<f64>, Vec<f64>)> {
        let (x, y) = self.series_for(g)?;
        let y = if self.smooth5 { smooth5(&y) } else { y };
        Some((x, y))
    }

    /// Apply the chosen finder to every selected group over `[peak_lo, peak_hi]`,
    /// collecting one `(label, x, y)` row per group — the original "Multiple peaks
    /// catching", which tabulates a peak position across all plotted spectra. A
    /// marker is drawn at each caught x on rebuild.
    fn catch_peaks(&mut self, groups: &[XasGroup]) {
        self.peaks.clear();
        for (i, g) in groups.iter().enumerate() {
            if !self.selected.get(i).copied().unwrap_or(false) {
                continue;
            }
            let Some((x, y)) = self.displayed_series(g) else {
                continue;
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
                self.peaks.push((g.label.clone(), px, py));
            }
        }
    }

    /// The "Files" control block: file-type / graph-item selectors, the picker
    /// button, the loaded-file list, and Clear Graph (the original's file-based
    /// Plot Data controls).
    fn files_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.strong("Files");

        egui::ComboBox::from_label("File type")
            .selected_text(self.file_type.label())
            .show_ui(ui, |ui| {
                for ft in FileType::ALL {
                    if ui
                        .selectable_value(&mut self.file_type, ft, ft.label())
                        .changed()
                    {
                        self.graph_item = self.file_type.default_item();
                        self.refresh_available();
                    }
                }
            });
        egui::ComboBox::from_label("Graph item")
            .selected_text(self.graph_item.label())
            .show_ui(ui, |ui| {
                for &gi in self.file_type.items() {
                    if ui
                        .selectable_value(&mut self.graph_item, gi, gi.label())
                        .changed()
                    {
                        self.refresh_available();
                    }
                }
            });

        if ui.button("ADD or DEL Data Files…").clicked() {
            self.picker_open = true;
            self.refresh_available();
        }

        if !self.loaded.is_empty() {
            ui.add_space(2.0);
            ui.label(format!("{} file(s) loaded:", self.loaded.len()));
            let mut remove = None;
            for (i, t) in self.loaded.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui.small_button("✕").on_hover_text("Remove").clicked() {
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

    /// The "Add / remove data files" picker: browse a folder, stage files
    /// matching the current graph item, and load them on OK. A subordinate
    /// `egui::Window` inside the Plot Data viewport.
    fn file_picker(&mut self, ui: &mut egui::Ui) {
        if !self.picker_open {
            return;
        }
        let mut win_open = true;
        egui::Window::new("Add / remove data files")
            .open(&mut win_open)
            .resizable(true)
            .default_size([520.0, 380.0])
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Browse folder…").clicked()
                        && let Some(dir) = rfd::FileDialog::new().pick_folder()
                    {
                        self.pick_dir = Some(dir);
                        self.pick_add.clear();
                        self.refresh_available();
                    }
                    if ui.checkbox(&mut self.pick_sort, "Sort").changed() {
                        self.refresh_available();
                    }
                });
                match &self.pick_dir {
                    Some(dir) => ui.weak(dir.display().to_string()),
                    None => ui.weak("Pick a folder to list its files."),
                };
                ui.label(format!(
                    "Showing {} files matching “{}”.",
                    self.file_type.label(),
                    self.graph_item.label(),
                ));
                ui.separator();

                let mut to_add = None;
                let mut to_remove = None;
                egui::Grid::new("picker_lists")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.strong("Available");
                        ui.strong("Selected");
                        ui.end_row();

                        ui.vertical(|ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("avail")
                                .max_height(240.0)
                                .show(ui, |ui| {
                                    for path in &self.available {
                                        if ui.selectable_label(false, file_name_of(path)).clicked()
                                        {
                                            to_add = Some(path.clone());
                                        }
                                    }
                                });
                        });
                        ui.vertical(|ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("staged")
                                .max_height(240.0)
                                .show(ui, |ui| {
                                    for path in &self.pick_add {
                                        if ui.selectable_label(false, file_name_of(path)).clicked()
                                        {
                                            to_remove = Some(path.clone());
                                        }
                                    }
                                });
                        });
                        ui.end_row();
                    });
                if let Some(p) = to_add {
                    self.pick_add.push(p);
                    self.refresh_available();
                }
                if let Some(p) = to_remove {
                    self.pick_add.retain(|x| x != &p);
                    self.refresh_available();
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Clear all").clicked() {
                        self.pick_add.clear();
                        self.refresh_available();
                    }
                    if ui
                        .add_enabled(!self.pick_add.is_empty(), egui::Button::new("Add to plot"))
                        .clicked()
                    {
                        self.load_staged();
                        self.picker_open = false;
                    }
                    if ui.button("Cancel").clicked() {
                        self.picker_open = false;
                    }
                });
            });
        // The window's [x] also closes the picker.
        self.picker_open = self.picker_open && win_open;
    }

    /// List the files in `pick_dir` that match the current graph item, excluding
    /// already-staged and already-loaded files. Sorted when "Sort" is on.
    fn refresh_available(&mut self) {
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
            let taken = self.pick_add.contains(&path) || self.loaded.iter().any(|t| t.path == path);
            if self.graph_item.matches(&name) && !taken {
                self.available.push(path);
            }
        }
        if self.pick_sort {
            self.available.sort();
        }
    }

    /// Load every staged file as the current graph item, reporting how many
    /// loaded / failed in `pick_status`.
    fn load_staged(&mut self) {
        let (item, kw) = (self.graph_item, self.kweight);
        let staged = std::mem::take(&mut self.pick_add);
        let (mut ok, mut failed) = (0usize, 0usize);
        for path in staged {
            match load_trace(&path, item, kw) {
                Ok(t) => {
                    self.loaded.push(t);
                    ok += 1;
                }
                Err(_) => failed += 1,
            }
        }
        self.pick_status = if failed > 0 {
            format!("Loaded {ok} file(s); {failed} could not be read.")
        } else {
            format!("Loaded {ok} file(s).")
        };
        self.refresh_available();
        self.dirty = true;
    }

    /// Re-read every loaded file (e.g. after a k-weight change, since k-space
    /// files store unweighted χ).
    fn reload_loaded(&mut self) {
        let kw = self.kweight;
        for t in &mut self.loaded {
            if let Ok(nt) = load_trace(&t.path, t.item, kw) {
                *t = nt;
            }
        }
    }

    /// Rebuild every plotted curve from the current selection and settings.
    fn rebuild(&mut self, groups: &[XasGroup]) {
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
        // group items), so draw it alone and skip the group traces.
        if self.show_overlay
            && let Some(ov) = &self.overlay
        {
            self.plot.set_graph_x_label(ov.xlabel);
            self.plot.set_graph_y_label(ov.ylabel, YAxis::Left);
            if !ov.x.is_empty() {
                self.plot
                    .add_curve_with_legend(&ov.x, &ov.data, FIT_DATA, "fit data");
                self.plot
                    .add_curve_with_legend(&ov.x, &ov.model, FIT_MODEL, "fit model");
            }
            return;
        }

        // Axis labels follow the loaded files' graph item when any file is shown
        // (Plot Data is primarily a file viewer); otherwise the group item.
        if self.loaded.is_empty() {
            self.plot.set_graph_x_label(self.item.x_label());
            self.plot.set_graph_y_label(self.item.label(), YAxis::Left);
        } else {
            self.plot.set_graph_x_label(self.graph_item.x_label());
            self.plot
                .set_graph_y_label(self.graph_item.y_label(), YAxis::Left);
        }

        // Selected (x, y) pairs in group order, with their colors. Bright curves
        // on the dark canvas; the muted tab10 on the white "Change BG" canvas,
        // where the bright palette would wash out.
        let palette = if self.dark_bg {
            crate::plot::PALETTE
        } else {
            crate::plot::PALETTE_LIGHT
        };
        let mut traces: Vec<(String, Vec<f64>, Vec<f64>, Color32)> = Vec::new();
        for (i, g) in groups.iter().enumerate() {
            if !self.selected.get(i).copied().unwrap_or(false) {
                continue;
            }
            if let Some((x, y)) = self.displayed_series(g) {
                let color = palette[traces.len() % palette.len()];
                traces.push((g.label.clone(), x, y, color));
            }
        }
        // Loaded files overlay additively with the group traces, sharing the
        // palette and the waterfall offset (the original's file-based view).
        for t in &self.loaded {
            let y = if self.smooth5 {
                smooth5(&t.y)
            } else {
                t.y.clone()
            };
            let color = palette[traces.len() % palette.len()];
            traces.push((t.label.clone(), t.x.clone(), y, color));
        }

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
            let off = idx as f64 * self.stack;
            let ys: Vec<f64> = if off != 0.0 {
                y.iter().map(|v| v + off).collect()
            } else {
                y
            };
            self.plot.add_curve_with_legend(&x, &ys, color, label);
        }

        if let Some((x, y)) = avg {
            self.plot.add_curve_with_legend(&x, &y, fg, "average");
        }

        for (_, px, _) in &self.peaks {
            self.plot
                .add_x_marker(*px, Color32::from_rgb(0x80, 0x80, 0x80));
        }
    }
}

/// The bare file name of `path`, for picker lists and legends.
fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// NEXAFS peak normalization: `(μ − pre-edge) / max(μ − pre-edge)`, so the
/// largest peak sits at 1 (the original's "normalize to the max point"). Expects
/// `g.pre_edge` filled (call [`normalize`] first).
fn peak_normalized(g: &XasGroup) -> Vec<f64> {
    let pre = g.pre_edge.as_deref().unwrap_or(&[]);
    let diff: Vec<f64> = g.mu.iter().zip(pre).map(|(&m, &p)| m - p).collect();
    let peak = diff.iter().copied().fold(f64::MIN, f64::max);
    if peak.abs() < 1e-300 {
        return diff;
    }
    diff.iter().map(|d| d / peak).collect()
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
}
