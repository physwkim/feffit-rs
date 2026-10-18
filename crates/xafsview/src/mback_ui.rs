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
use feffit::xasdata::{Edge, MbackNorm, MbackNormParams, XasGroup, mback_norm};
use rsplot::YAxis;
use xraydb::XrayDb;

use crate::plot::{BLUE, ORANGE, RED};

/// Selectable absorption edges (those MBACK handles).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    /// From xraydb's IUPAC edge label; anything MBACK does not handle (M5)
    /// falls back to K.
    fn from_label(label: &str) -> Self {
        match label {
            "L1" => EdgeSel::L1,
            "L2" => EdgeSel::L2,
            "L3" => EdgeSel::L3,
            _ => EdgeSel::K,
        }
    }
}

/// Element / edge guess for an absorption-edge energy `e0`, as the window's
/// default: the nearest K edge within [`EDGE_TOL`] eV, else the nearest L3 edge
/// within it, else xraydb's nearest edge of any kind. A measured e0 (the
/// derivative maximum) sits a few eV *above* its tabulated edge, so the plain
/// nearest-of-any rule picked whichever other element's edge happened to lie
/// there: Mn K data at 6551 eV came back as Ce L1 (6549 eV) over Mn K
/// (6539 eV). K then L3 is the order in which those edges are measured.
fn guess_element_edge(db: &XrayDb, e0: f64) -> Option<(u16, EdgeSel)> {
    /// Chemical shift + calibration offset of a measured e0 from its tabulated
    /// edge stays well inside this; the nearest K edge of another element is
    /// hundreds of eV away, and the one close K/L3 pair (Ni K 8333 / Er L3
    /// 8358 eV) still resolves for an Er e0 above its edge.
    const EDGE_TOL: f64 = 25.0;
    for edge in ["K", "L3"] {
        if let Some((sym, label)) = db.guess_edge(e0, Some(&[edge]))
            && let Ok(x) = db.xray_edge(&sym, &label)
            && (x.energy - e0).abs() <= EDGE_TOL
            && let Ok(z) = db.atomic_number(&sym)
        {
            return Some((z, EdgeSel::from_label(&label)));
        }
    }
    let (sym, label) = db.guess_edge(e0, None)?;
    let z = db.atomic_number(&sym).ok()?;
    Some((z, EdgeSel::from_label(&label)))
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
    plot: crate::plot::Plot,
    target: Option<usize>,
    z: u16,
    edge: EdgeSel,
    /// The spectrum (`target` index) the element / edge were last seeded for;
    /// a different pick reseeds them from the new spectrum's e0.
    seeded_for: Option<usize>,
    result: Option<MbackResult>,
    error: Option<String>,
    dirty: bool,
}

impl MbackWindow {
    /// Build the window with its own plot (`PlotId` 6).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::Plot::new(render_state, 6);
        plot.set_graph_title("MBACK");
        plot.set_empty_hint("Pick a group to normalize and plot.");
        Self {
            open: false,
            db: XrayDb::new(),
            plot,
            target: None,
            z: 26,
            edge: EdgeSel::K,
            seeded_for: None,
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
        crate::window::detached_panels(
            ctx,
            "mback",
            "MBACK / NEXAFS normalization",
            &mut open,
            [820.0, 540.0],
            |ui| {
                egui::Panel::left("mback_controls")
                    .resizable(true)
                    .default_size(280.0)
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, groups));
                    });
                crate::window::fill_with_footer(
                    ui,
                    "mback",
                    self,
                    |s, ui| s.stats(ui),
                    |s, ui| {
                        if s.dirty {
                            s.rebuild_plot(groups);
                            s.dirty = false;
                        }
                        crate::plot::show(&mut s.plot, ui);
                    },
                );
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

        self.reseed(groups);

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

    /// A newly picked spectrum: the previous spectrum's result no longer
    /// describes what is shown, and the element / edge are re-guessed from
    /// the new e0 (keyed to the picked index, not a one-shot flag — a Mn
    /// seed kept across a switch to Pt ran MBACK with the wrong f″).
    fn reseed(&mut self, groups: &[XasGroup]) {
        if self.seeded_for == self.target {
            return;
        }
        self.seeded_for = self.target;
        self.result = None;
        self.error = None;
        self.dirty = true;
        if let Some(e0) = self.target.and_then(|i| groups.get(i)).and_then(|g| g.e0)
            && let Some((z, edge)) = guess_element_edge(&self.db, e0)
        {
            self.z = z;
            self.edge = edge;
        }
    }

    /// MCP `mback`: the controls in their on-screen order (spectrum pick →
    /// reseed → Z / edge → Run MBACK), then the Edge step table.
    pub fn apply_mcp(
        &mut self,
        groups: &[XasGroup],
        a: &crate::mcp::MbackArgs,
    ) -> Result<String, String> {
        self.open = true;
        if let Some(label) = &a.spectrum {
            let idx = groups
                .iter()
                .position(|g| g.label == *label && !g.mu.is_empty())
                .ok_or_else(|| format!("no group {label:?} with μ(E)"))?;
            self.target = Some(idx);
        }
        self.reseed(groups);
        if let Some(z) = a.z {
            if !(1..=98).contains(&z) {
                return Err(format!("z {z} outside 1..=98"));
            }
            self.z = z;
        }
        if let Some(e) = &a.edge {
            self.edge = match e.to_ascii_uppercase().as_str() {
                "K" => EdgeSel::K,
                "L1" => EdgeSel::L1,
                "L2" => EdgeSel::L2,
                "L3" => EdgeSel::L3,
                other => return Err(format!("unknown edge {other:?}; one of K, L1, L2, L3")),
            };
        }
        let sym = self
            .db
            .symbol(&self.z.to_string())
            .unwrap_or("?")
            .to_owned();
        let target = self
            .target
            .and_then(|i| groups.get(i))
            .map(|g| g.label.clone());
        if a.run.unwrap_or(true) {
            if target.is_none() {
                return Err("Pick a spectrum with μ(E).".to_owned());
            }
            self.run(groups);
            if let Some(e) = &self.error {
                return Err(e.clone());
            }
        }
        let steps = match &self.result {
            Some(r) => format!(
                "MBACK edge step {:.5}, polynomial edge step {:.5}",
                r.edge_step, r.edge_step_poly
            ),
            None => "no result".to_owned(),
        };
        Ok(format!(
            "spectrum {}, Z {} ({sym}) {} edge: {steps}",
            target.as_deref().unwrap_or("none"),
            self.z,
            self.edge.label()
        ))
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
        let mb: MbackNorm = match mback_norm(&g.energy, &g.mu, &f2, &params) {
            Ok(mb) => mb,
            Err(e) => {
                self.error = Some(format!("MBACK: {e}"));
                return;
            }
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// K wins within tolerance even when another element's L edge is nearer
    /// (Mn K 6539 vs Ce L1 6549 for an e0 of 6551); an L3 e0 with no K edge in
    /// range resolves to L3, including the close Ni K / Er L3 pair.
    #[test]
    fn edge_guess_prefers_k_then_l3() {
        let db = XrayDb::new();
        assert_eq!(guess_element_edge(&db, 6551.0), Some((25, EdgeSel::K)));
        assert_eq!(guess_element_edge(&db, 11566.0), Some((78, EdgeSel::L3)));
        assert_eq!(guess_element_edge(&db, 8362.0), Some((68, EdgeSel::L3)));
        assert_eq!(guess_element_edge(&db, 8340.0), Some((28, EdgeSel::K)));
    }
}
