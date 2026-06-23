//! The standalone **Plot Data** window: overlay any reduction stage of several
//! loaded groups on one plot, with vertical stacking, an averaged trace, and a
//! peak readout. Mirrors XAFSView's *Plot Data* window.
//!
//! It owns its own [`Plot1D`] (separate from the tabs' shared plot) so it can
//! float independently. Save / zoom / legend come from the siplot toolbar; the
//! data work (averaging, peak finding) is the headless [`xasdata::batch`] code.

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use siplot::{Plot1D, YAxis};
use xasdata::{XasGroup, average_curves, peak_in_range};

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

/// A tab10-style palette cycled across the overlaid traces.
const PALETTE: [Color32; 8] = [
    Color32::from_rgb(0x1f, 0x77, 0xb4),
    Color32::from_rgb(0xff, 0x7f, 0x0e),
    Color32::from_rgb(0x2c, 0xa0, 0x2c),
    Color32::from_rgb(0xd6, 0x27, 0x28),
    Color32::from_rgb(0x94, 0x67, 0xbd),
    Color32::from_rgb(0x8c, 0x56, 0x4b),
    Color32::from_rgb(0xe3, 0x77, 0xc2),
    Color32::from_rgb(0x17, 0xbe, 0xcf),
];

/// Feffit "Send to Plot Data" overlay colours (data vs model).
const FIT_DATA: Color32 = Color32::from_rgb(0x1f, 0x77, 0xb4);
const FIT_MODEL: Color32 = Color32::from_rgb(0xd6, 0x27, 0x28);

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

/// The Plot Data window state and its own plot.
pub struct PlotDataWindow {
    /// Whether the window is shown.
    pub open: bool,
    plot: Plot1D,
    item: PlotItem,
    kweight: i32,
    /// Per-group "show this trace" flags, kept the same length as the session's
    /// group list.
    selected: Vec<bool>,
    /// Vertical offset added to trace `i` (`i · stack`), in data units.
    stack: f64,
    show_average: bool,
    peak_lo: f64,
    peak_hi: f64,
    peak: Option<(f64, f64)>,
    /// A Feffit fit sent here via "Send to Plot Data", if any.
    overlay: Option<FitOverlay>,
    /// Whether the sent fit takes over the plot (vs the group items).
    show_overlay: bool,
    /// Set whenever the overlay needs rebuilding (control change or new data).
    dirty: bool,
}

impl PlotDataWindow {
    /// Build the window with its own plot (use a distinct `PlotId` from the
    /// tabs' shared plot).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::new_plot1d(render_state, 1);
        plot.set_graph_title("Plot Data");
        Self {
            open: false,
            plot,
            item: PlotItem::Norm,
            kweight: 2,
            selected: Vec::new(),
            stack: 0.0,
            show_average: false,
            peak_lo: 0.0,
            peak_hi: 0.0,
            peak: None,
            overlay: None,
            show_overlay: false,
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
        if matches!(self.item, PlotItem::Chik)
            && ui
                .add(egui::Slider::new(&mut self.kweight, 0..=3).text("k-weight"))
                .changed()
        {
            self.dirty = true;
        }

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

        ui.separator();
        ui.label("Peak search");
        ui.horizontal(|ui| {
            ui.label("from");
            ui.add(egui::DragValue::new(&mut self.peak_lo).speed(0.5));
            ui.label("to");
            ui.add(egui::DragValue::new(&mut self.peak_hi).speed(0.5));
        });
        if ui.button("Find peak (first selected)").clicked() {
            self.find_peak(groups);
            self.dirty = true;
        }
        match self.peak {
            Some((x, y)) => {
                ui.monospace(format!("peak @ x = {x:.4}, y = {y:.5}"));
            }
            None => {
                ui.weak("no peak in range");
            }
        }

        ui.separator();
        if ui.button("Replot").clicked() {
            self.dirty = true;
        }
    }

    /// Find the maximum of the chosen item within `[peak_lo, peak_hi]` on the
    /// first selected group, and store it (a marker is drawn on rebuild).
    fn find_peak(&mut self, groups: &[XasGroup]) {
        self.peak = self
            .first_selected(groups)
            .and_then(|g| self.item.xy(g, self.kweight))
            .and_then(|(x, y)| peak_in_range(&x, &y, self.peak_lo, self.peak_hi));
    }

    fn first_selected<'a>(&self, groups: &'a [XasGroup]) -> Option<&'a XasGroup> {
        self.selected
            .iter()
            .zip(groups)
            .find(|(sel, _)| **sel)
            .map(|(_, g)| g)
    }

    /// Rebuild every plotted curve from the current selection and settings.
    fn rebuild(&mut self, groups: &[XasGroup]) {
        self.plot.clear();

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

        self.plot.set_graph_x_label(self.item.x_label());
        self.plot.set_graph_y_label(self.item.label(), YAxis::Left);

        // Selected (x, y) pairs in group order, with their colors.
        let mut traces: Vec<(String, Vec<f64>, Vec<f64>, Color32)> = Vec::new();
        for (i, g) in groups.iter().enumerate() {
            if !self.selected.get(i).copied().unwrap_or(false) {
                continue;
            }
            if let Some((x, y)) = self.item.xy(g, self.kweight) {
                let color = PALETTE[traces.len() % PALETTE.len()];
                traces.push((g.label.clone(), x, y, color));
            }
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
            self.plot
                .add_curve_with_legend(&x, &y, Color32::from_rgb(0x20, 0x20, 0x20), "average");
        }

        if let Some((px, _)) = self.peak {
            self.plot
                .add_x_marker(px, Color32::from_rgb(0x80, 0x80, 0x80));
        }
    }
}
