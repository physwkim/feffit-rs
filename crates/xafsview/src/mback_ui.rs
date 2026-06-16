//! The **MBACK / NEXAFS normalization** window: match μ(E) to the tabulated
//! Chantler `f''(E)` so the edge step is set by atomic data rather than a
//! polynomial fit (Weng, Waldo & Penner-Hahn).
//!
//! This is the one Phase-8 feature that joins both halves of the stack: the
//! Chantler `f2` comes from [`xraydb`] (keyed by element + energy grid) and the
//! fit itself is the headless [`mback_norm`] engine. The
//! window plots the MBACK-normalized spectrum against the group's existing
//! polynomial normalization for comparison.

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use siplot::{Plot1D, YAxis};
use xasdata::{Edge, MbackNorm, MbackNormParams, XasGroup, mback_norm};
use xraydb::XrayDb;

const BLUE: Color32 = Color32::from_rgb(0x1f, 0x77, 0xb4);
const ORANGE: Color32 = Color32::from_rgb(0xff, 0x7f, 0x0e);
const RED: Color32 = Color32::from_rgb(0xd6, 0x27, 0x28);

/// Selectable absorption edges (those MBACK handles).
#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeSel {
    K,
    L1,
    L2,
    L3,
}

impl EdgeSel {
    const ALL: [EdgeSel; 4] = [EdgeSel::K, EdgeSel::L1, EdgeSel::L2, EdgeSel::L3];
    fn label(self) -> &'static str {
        match self {
            EdgeSel::K => "K",
            EdgeSel::L1 => "L1",
            EdgeSel::L2 => "L2",
            EdgeSel::L3 => "L3",
        }
    }
    fn to_engine(self) -> Edge {
        match self {
            EdgeSel::K => Edge::K,
            EdgeSel::L1 => Edge::L1,
            EdgeSel::L2 => Edge::L2,
            EdgeSel::L3 => Edge::L3,
        }
    }
    /// The next-higher edge whose energy caps `norm2` (L2→L1, L3→L2), if any.
    fn capping_edge(self) -> Option<&'static str> {
        match self {
            EdgeSel::L3 => Some("L2"),
            EdgeSel::L2 => Some("L1"),
            EdgeSel::K | EdgeSel::L1 => None,
        }
    }
}

/// One MBACK result on the (deduplicated) energy grid, ready to plot.
struct MbackResult {
    norm: Vec<f64>,
    edge_step: f64,
    edge_step_poly: f64,
}

/// The MBACK / NEXAFS normalization window.
pub struct MbackWindow {
    pub open: bool,
    db: XrayDb,
    plot: Plot1D,
    target: Option<usize>,
    z: u16,
    edge: EdgeSel,
    seeded: bool,
    result: Option<MbackResult>,
    error: Option<String>,
    dirty: bool,
}

impl MbackWindow {
    /// Build the window with its own plot (`PlotId` 6).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::new_plot1d(render_state, 6);
        plot.set_graph_title("MBACK");
        Self {
            open: false,
            db: XrayDb::new(),
            plot,
            target: None,
            z: 26,
            edge: EdgeSel::K,
            seeded: false,
            result: None,
            error: None,
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
            "mback",
            "MBACK / NEXAFS normalization",
            &mut open,
            [820.0, 540.0],
            |ui| {
                egui::Panel::left("mback_controls")
                    .resizable(true)
                    .default_size(280.0)
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
                    self.stats(ui);
                });
            },
        );
        self.open = open;
    }

    fn controls(&mut self, ui: &mut egui::Ui, groups: &[XasGroup]) {
        ui.heading("MBACK");
        ui.label("Match μ(E) to tabulated f″(E) for an atomic-data edge step.");

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
                    if !g.mu.is_empty() {
                        ui.selectable_value(&mut self.target, Some(i), &g.label);
                    }
                }
            });

        // Seed the element from the group's e0 via xraydb's edge guesser, once.
        if !self.seeded
            && let Some(e0) = self.target.and_then(|i| groups.get(i)).and_then(|g| g.e0)
            && let Some((sym, edge)) = self.db.guess_edge(e0, None)
            && let Ok(z) = self.db.atomic_number(&sym)
        {
            self.z = z;
            self.edge = match edge.as_str() {
                "L1" => EdgeSel::L1,
                "L2" => EdgeSel::L2,
                "L3" => EdgeSel::L3,
                _ => EdgeSel::K,
            };
            self.seeded = true;
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("element Z");
            ui.add(egui::DragValue::new(&mut self.z).range(1..=98));
            let sym = self.db.symbol(&self.z.to_string()).unwrap_or("?");
            ui.monospace(sym);
        });
        ui.horizontal(|ui| {
            ui.label("edge");
            for e in EdgeSel::ALL {
                ui.selectable_value(&mut self.edge, e, e.label());
            }
        });

        ui.separator();
        let can_run = self
            .target
            .and_then(|i| groups.get(i))
            .is_some_and(|g| !g.mu.is_empty());
        if ui
            .add_enabled(can_run, egui::Button::new("Run MBACK"))
            .clicked()
        {
            self.run(groups);
        }
        if !can_run {
            ui.weak("Pick a spectrum with μ(E).");
        }
        if let Some(e) = &self.error {
            ui.colored_label(RED, e);
        }
    }

    fn run(&mut self, groups: &[XasGroup]) {
        self.error = None;
        let Some(g) = self.target.and_then(|i| groups.get(i)) else {
            return;
        };
        if g.mu.is_empty() {
            return;
        }
        let zs = self.z.to_string();
        // Chantler f″ on the spectrum's energy grid (xraydb).
        let f2 = match self.db.f2_chantler(&zs, &g.energy) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(format!("f2_chantler: {e}"));
                return;
            }
        };
        // For L2/L3, cap norm2 below the next-higher edge.
        let next_edge_energy = self.edge.capping_edge().and_then(|lbl| {
            self.db
                .xray_edge(&zs, lbl)
                .ok()
                .map(|e: xraydb::XrayEdge| e.energy)
        });
        let params = MbackNormParams {
            e0: g.e0,
            edge: self.edge.to_engine(),
            next_edge_energy,
            ..Default::default()
        };
        let mb: MbackNorm = mback_norm(&g.energy, &g.mu, &f2, &params);
        self.result = Some(MbackResult {
            norm: mb.norm,
            edge_step: mb.edge_step,
            edge_step_poly: mb.edge_step_poly,
        });
        self.dirty = true;
    }

    fn rebuild_plot(&mut self, groups: &[XasGroup]) {
        self.plot.clear();
        let Some(g) = self.target.and_then(|i| groups.get(i)) else {
            return;
        };
        let Some(r) = &self.result else { return };
        self.plot.set_graph_x_label("Energy (eV)");
        self.plot.set_graph_y_label("normalized μ(E)", YAxis::Left);
        // mback_norm dedups the energy grid; plot only when lengths line up.
        if r.norm.len() == g.energy.len() {
            self.plot
                .add_curve_with_legend(&g.energy, &r.norm, BLUE, "MBACK norm");
        }
        if let Some(poly) = &g.norm
            && poly.len() == g.energy.len()
        {
            self.plot
                .add_curve_with_legend(&g.energy, poly, ORANGE, "polynomial norm");
        }
    }

    fn stats(&mut self, ui: &mut egui::Ui) {
        ui.strong("Edge step");
        let Some(r) = &self.result else {
            ui.weak("Run MBACK to compare the atomic-data and polynomial steps.");
            return;
        };
        egui::Grid::new("mback_stats").striped(true).show(ui, |ui| {
            ui.label("MBACK edge step");
            ui.monospace(format!("{:.5}", r.edge_step));
            ui.end_row();
            ui.label("polynomial edge step");
            ui.monospace(format!("{:.5}", r.edge_step_poly));
            ui.end_row();
        });
    }
}
