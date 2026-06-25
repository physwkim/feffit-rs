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

use std::path::PathBuf;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use siplot::YAxis;
use xasdata::{average_curves, peak_in_range, x_at_y};

use crate::plot_files::{FileType, GraphItem, LoadedTrace, load_trace};

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
    /// Files staged in the picker to add on OK.
    pick_add: Vec<PathBuf>,
    /// Sort the available list alphabetically.
    pick_sort: bool,
    /// Outcome of the last load (shown in the Files section).
    pick_status: String,

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
    pub fn show(&mut self, ctx: &egui::Context) {
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
                });
                // The file picker floats over the window while open.
                self.file_picker(ui);
            },
        );
        self.open = open;
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

        if self.loaded.is_empty() {
            ui.weak("No files loaded — ADD Data Files to plot.");
        } else {
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

    /// The current x/y axis labels — the loaded files' graph item.
    fn axis_labels(&self) -> (&'static str, &'static str) {
        (self.graph_item.x_label(), self.graph_item.y_label())
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
            .map(|(idx, (label, x, y, _color))| {
                let off = idx as f64 * self.stack;
                let ys = if off != 0.0 {
                    y.iter().map(|v| v + off).collect()
                } else {
                    y
                };
                (label, x, ys)
            })
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
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("plotdata.dat")
            .save_file()
        else {
            return;
        };
        let (xlabel, ylabel) = self.axis_labels();
        self.pick_status = match write_composite(&path, &traces, xlabel, ylabel) {
            Ok(()) => format!(
                "Saved {} curve(s) to {}.",
                traces.len(),
                file_name_of(&path)
            ),
            Err(e) => format!("Save failed: {e}"),
        };
    }

    /// Rebuild every plotted curve from the loaded files and current settings.
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
                self.plot
                    .add_curve_with_legend(&ov.x, &ov.model, FIT_MODEL, "fit model");
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
    for (i, (label, x, y)) in traces.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        let _ = writeln!(s, "# curve {}: {label}", i + 1);
        let _ = writeln!(s, "#  {xlabel:<14}  {ylabel}");
        for (xx, yy) in x.iter().zip(y) {
            let _ = writeln!(s, "{xx:14.6}  {yy:18.10}");
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
    fn write_composite_blocks_parse_back_independently() {
        use xasdata::ColumnFile;
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
}
