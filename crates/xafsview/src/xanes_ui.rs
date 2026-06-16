//! The **XANES tools** window: interactive edge/peak cursors (peak, valley,
//! half-step, half-max, x@y) and an arctangent-step baseline subtraction.
//!
//! All tools read the chosen group's normalized or flattened μ(E) (a group must
//! have been normalized first) and call the headless [`xasdata`] primitives
//! ([`peak`], [`valley`], [`x_at_y`], [`arctan_step`], [`centroid`]). The window
//! owns its own [`Plot1D`].
//!
//! MBACK / NEXAFS normalization (the other XANES-tab item) is intentionally not
//! here: it needs Chantler `f2` from xraydb keyed by element + edge, which lands
//! with the rest of the xraydb-backed UI in Phase 8.

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use siplot::{Plot1D, YAxis};
use xasdata::{XasGroup, arctan_step, centroid, peak, valley, x_at_y};

use crate::analysis_ui::array_xy;

const BLUE: Color32 = Color32::from_rgb(0x1f, 0x77, 0xb4);
const ORANGE: Color32 = Color32::from_rgb(0xff, 0x7f, 0x0e);
const GREEN: Color32 = Color32::from_rgb(0x2c, 0xa0, 0x2c);
const PURPLE: Color32 = Color32::from_rgb(0x94, 0x67, 0xbd);

/// One cursor readout: a label, the energy it marks, and an optional value.
struct Readout {
    label: String,
    x: f64,
    y: Option<f64>,
}

/// The XANES-tools window.
pub struct XanesWindow {
    pub open: bool,
    plot: Plot1D,
    target: Option<usize>,
    use_flat: bool,
    /// Region [rlo, rhi] used by peak / valley / half-max searches.
    rlo: f64,
    rhi: f64,
    /// Target level for the "x @ y" cursor.
    yval: f64,
    seeded: bool,
    /// Cursor results, most recent last (capped).
    readouts: Vec<Readout>,
    // Arctangent-step baseline subtraction.
    arc_on: bool,
    amp: f64,
    center: f64,
    sigma: f64,
    slope: f64,
    intercept: f64,
    /// Centroid of the baseline-subtracted peaks over [rlo, rhi], when computed.
    arc_centroid: Option<f64>,
    dirty: bool,
}

impl XanesWindow {
    /// Build the window with its own plot (`PlotId` 5).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::new_plot1d(render_state, 5);
        plot.set_graph_title("XANES");
        Self {
            open: false,
            plot,
            target: None,
            use_flat: false,
            rlo: 0.0,
            rhi: 0.0,
            yval: 0.5,
            seeded: false,
            readouts: Vec::new(),
            arc_on: false,
            amp: 1.0,
            center: 0.0,
            sigma: 2.0,
            slope: 0.0,
            intercept: 0.0,
            arc_centroid: None,
            dirty: true,
        }
    }

    /// Render the window over `groups`.
    pub fn show(&mut self, ctx: &egui::Context, groups: &[XasGroup]) {
        if !self.open {
            return;
        }
        let mut open = self.open;
        crate::window::detached(
            ctx,
            "xanes",
            "XANES tools",
            &mut open,
            [860.0, 580.0],
            |ui| {
                egui::Panel::left("xanes_controls")
                    .resizable(true)
                    .default_size(300.0)
                    .show_inside(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, groups));
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if self.dirty {
                        self.rebuild_plot(groups);
                        self.dirty = false;
                    }
                    crate::plot::show(&mut self.plot, ui);
                    ui.separator();
                    self.readout_table(ui);
                });
            },
        );
        self.open = open;
    }

    /// The selected group's (energy, array) as normalized or flattened μ(E).
    fn target_xy<'a>(&self, groups: &'a [XasGroup]) -> Option<(&'a [f64], &'a [f64])> {
        self.target
            .and_then(|i| groups.get(i))
            .and_then(|g| array_xy(g, self.use_flat))
    }

    fn controls(&mut self, ui: &mut egui::Ui, groups: &[XasGroup]) {
        ui.heading("XANES");
        ui.label("Edge / peak cursors and arctangent subtraction.");

        ui.separator();
        let target_label = self
            .target
            .and_then(|i| groups.get(i))
            .map(|g| g.label.clone())
            .unwrap_or_else(|| "— pick spectrum —".to_owned());
        egui::ComboBox::from_label("Spectrum")
            .selected_text(target_label)
            .show_ui(ui, |ui| {
                for (i, g) in groups.iter().enumerate() {
                    if g.norm.is_some() {
                        ui.selectable_value(&mut self.target, Some(i), &g.label);
                    }
                }
            });
        if ui
            .horizontal(|ui| {
                ui.label("Array:");
                let a = ui
                    .selectable_value(&mut self.use_flat, false, "norm")
                    .clicked();
                let b = ui
                    .selectable_value(&mut self.use_flat, true, "flat")
                    .clicked();
                a || b
            })
            .inner
        {
            self.dirty = true;
        }

        // Seed region + arctangent center from the spectrum's span / e0 once.
        if !self.seeded
            && let Some((e, _)) = self.target_xy(groups)
            && e.len() >= 2
        {
            self.rlo = e[0];
            self.rhi = e[e.len() - 1];
            let e0 = self
                .target
                .and_then(|i| groups.get(i))
                .and_then(|g| g.e0)
                .unwrap_or((e[0] + e[e.len() - 1]) * 0.5);
            self.center = e0;
            self.seeded = true;
        }

        ui.separator();
        ui.strong("Region (eV)");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.rlo).speed(1.0).prefix("lo "));
            ui.add(egui::DragValue::new(&mut self.rhi).speed(1.0).prefix("hi "));
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Peak").clicked() {
                self.cursor_peak(groups, true);
            }
            if ui.button("Valley").clicked() {
                self.cursor_peak(groups, false);
            }
            if ui.button("Half-step").clicked() {
                self.cursor_half_step(groups);
            }
            if ui.button("Half-max").clicked() {
                self.cursor_half_max(groups);
            }
        });

        ui.separator();
        ui.strong("x @ y");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.yval).speed(0.01));
            if ui.button("Find x at y").clicked() {
                self.cursor_x_at_y(groups);
            }
        });

        ui.separator();
        if ui
            .checkbox(&mut self.arc_on, "Arctangent subtraction")
            .changed()
        {
            self.dirty = true;
        }
        if self.arc_on {
            let mut ch = false;
            egui::Grid::new("xanes_arc").num_columns(2).show(ui, |ui| {
                ui.label("amplitude");
                ch |= ui
                    .add(egui::DragValue::new(&mut self.amp).speed(0.01))
                    .changed();
                ui.end_row();
                ui.label("center (eV)");
                ch |= ui
                    .add(egui::DragValue::new(&mut self.center).speed(0.5))
                    .changed();
                ui.end_row();
                ui.label("sigma (eV)");
                ch |= ui
                    .add(
                        egui::DragValue::new(&mut self.sigma)
                            .speed(0.1)
                            .range(0.05..=500.0),
                    )
                    .changed();
                ui.end_row();
                ui.label("slope");
                ch |= ui
                    .add(egui::DragValue::new(&mut self.slope).speed(1e-5))
                    .changed();
                ui.end_row();
                ui.label("intercept");
                ch |= ui
                    .add(egui::DragValue::new(&mut self.intercept).speed(0.01))
                    .changed();
                ui.end_row();
            });
            if ch {
                self.dirty = true;
            }
            if let Some(c) = self.arc_centroid {
                ui.monospace(format!("peak centroid: {c:.3} eV"));
            }
        }

        ui.separator();
        if ui.button("Clear cursors").clicked() {
            self.readouts.clear();
            self.dirty = true;
        }
        if self.target.is_none() {
            ui.weak("Pick a normalized spectrum.");
        }
    }

    fn push_readout(&mut self, label: String, x: f64, y: Option<f64>) {
        const CAP: usize = 12;
        self.readouts.push(Readout { label, x, y });
        if self.readouts.len() > CAP {
            self.readouts.remove(0);
        }
        self.dirty = true;
    }

    fn cursor_peak(&mut self, groups: &[XasGroup], want_max: bool) {
        let Some((e, y)) = self.target_xy(groups) else {
            return;
        };
        let found = if want_max {
            peak(e, y, self.rlo, self.rhi)
        } else {
            valley(e, y, self.rlo, self.rhi)
        };
        if let Some((px, py)) = found {
            let label = if want_max { "peak" } else { "valley" };
            self.push_readout(label.to_owned(), px, Some(py));
        }
    }

    fn cursor_half_step(&mut self, groups: &[XasGroup]) {
        let Some((e, y)) = self.target_xy(groups) else {
            return;
        };
        // half the edge step: on normalized/flattened μ the step is 1.0.
        if let Some(px) = x_at_y(e, y, 0.5) {
            self.push_readout("half-step".to_owned(), px, Some(0.5));
        }
    }

    fn cursor_half_max(&mut self, groups: &[XasGroup]) {
        let Some((e, y)) = self.target_xy(groups) else {
            return;
        };
        let Some((px, py)) = peak(e, y, self.rlo, self.rhi) else {
            return;
        };
        // Rising-edge crossing of half the peak height: scan from the region low
        // bound up to the peak index.
        let i_peak = e.iter().position(|&x| x >= px).unwrap_or(0);
        let i_lo = e.iter().position(|&x| x >= self.rlo).unwrap_or(0);
        if i_peak > i_lo
            && let Some(hx) = x_at_y(&e[i_lo..=i_peak], &y[i_lo..=i_peak], py * 0.5)
        {
            self.push_readout("half-max".to_owned(), hx, Some(py * 0.5));
        }
    }

    fn cursor_x_at_y(&mut self, groups: &[XasGroup]) {
        let Some((e, y)) = self.target_xy(groups) else {
            return;
        };
        if let Some(px) = x_at_y(e, y, self.yval) {
            self.push_readout(format!("x@{:.3}", self.yval), px, Some(self.yval));
        }
    }

    /// Baseline (arctangent step + line) and the peaks `y - baseline` over the
    /// full spectrum, plus the peaks' centroid over [rlo, rhi].
    fn arc_overlay(&self, e: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let step = arctan_step(e, self.amp, self.center, self.sigma);
        let baseline: Vec<f64> = e
            .iter()
            .zip(&step)
            .map(|(&x, &s)| s + self.slope * x + self.intercept)
            .collect();
        let peaks: Vec<f64> = y.iter().zip(&baseline).map(|(&v, &b)| v - b).collect();
        (baseline, peaks)
    }

    fn rebuild_plot(&mut self, groups: &[XasGroup]) {
        self.plot.clear();
        self.arc_centroid = None;
        let Some((e, y)) = self.target_xy(groups) else {
            return;
        };
        self.plot.set_graph_x_label("Energy (eV)");
        self.plot
            .set_graph_y_label(if self.use_flat { "flat" } else { "norm" }, YAxis::Left);
        self.plot.add_curve_with_legend(e, y, BLUE, "μ(E)");

        if self.arc_on {
            let (baseline, peaks) = self.arc_overlay(e, y);
            self.plot
                .add_curve_with_legend(e, &baseline, ORANGE, "baseline");
            self.plot.add_curve_with_legend(e, &peaks, GREEN, "peaks");
            // centroid over the region.
            let i_lo = e.iter().position(|&x| x >= self.rlo).unwrap_or(0);
            let i_hi = e
                .iter()
                .rposition(|&x| x <= self.rhi)
                .unwrap_or(e.len() - 1);
            if i_hi >= i_lo {
                self.arc_centroid = centroid(&e[i_lo..=i_hi], &peaks[i_lo..=i_hi]);
                if let Some(c) = self.arc_centroid {
                    self.plot.add_x_marker(c, PURPLE);
                }
            }
        }

        for r in &self.readouts {
            self.plot.add_x_marker(r.x, PURPLE);
        }
    }

    fn readout_table(&mut self, ui: &mut egui::Ui) {
        ui.strong("Cursors");
        if self.readouts.is_empty() {
            ui.weak("Use the region tools or x@y to mark edge/peak positions.");
            return;
        }
        egui::Grid::new("xanes_readouts")
            .striped(true)
            .show(ui, |ui| {
                ui.strong("what");
                ui.strong("energy (eV)");
                ui.strong("value");
                ui.end_row();
                for r in &self.readouts {
                    ui.label(&r.label);
                    ui.monospace(format!("{:.3}", r.x));
                    match r.y {
                        Some(v) => ui.monospace(format!("{v:.4}")),
                        None => ui.weak("—"),
                    };
                    ui.end_row();
                }
            });
    }
}
