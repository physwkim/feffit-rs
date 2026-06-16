//! The "Edit μ(E)" floating window: deglitch (point/range removal), trim, and
//! smoothing for the current group, plus undo. Mirrors XAFSView's Edit-XMU
//! dialog and Smoothing menu, driving the `xasdata::clean` orchestration.
//!
//! Following the rest of the GUI, the window only *collects* the user's intent
//! into a [`CleanAction`]; the app applies it to the current group and manages
//! the undo stack, so this module borrows no session state.

use eframe::egui;
use xasdata::{RangeSide, SmoothForm};

/// What the dialog is asking the app to do to the current group this frame.
pub enum CleanAction {
    /// Remove the single point nearest this energy (eV).
    DeglitchPoint(f64),
    /// Remove a range relative to the reference energies (eV); the second value
    /// is consulted only for [`RangeSide::Between`].
    DeglitchRange(RangeSide, f64, f64),
    /// Trim to the inclusive `[lo, hi]` energy window (eV).
    Trim(f64, f64),
    /// Smooth `mu(E)` with the given width (eV) and line shape.
    Smooth(f64, SmoothForm),
    /// Undo the most recent edit.
    Undo,
}

/// Editable parameters and visibility for the Edit-μ(E) window.
pub struct EditXmuState {
    /// Whether the window is shown.
    pub open: bool,
    deglitch_e: f64,
    range_side: RangeSide,
    range1: f64,
    range2: f64,
    trim_lo: f64,
    trim_hi: f64,
    smooth_sigma: f64,
    smooth_form: SmoothForm,
    /// True once the energy widgets have been seeded from a group's span.
    seeded: bool,
}

impl Default for EditXmuState {
    fn default() -> Self {
        Self {
            open: false,
            deglitch_e: 0.0,
            range_side: RangeSide::Above,
            range1: 0.0,
            range2: 0.0,
            trim_lo: 0.0,
            trim_hi: 0.0,
            smooth_sigma: 1.0,
            smooth_form: SmoothForm::Lorentzian,
            seeded: false,
        }
    }
}

impl EditXmuState {
    /// Seed the energy-valued widgets from a group's `[lo, hi]` span the first
    /// time the dialog opens against fresh data (no-op once seeded).
    pub fn seed_span(&mut self, lo: f64, hi: f64) {
        if self.seeded {
            return;
        }
        let mid = 0.5 * (lo + hi);
        self.deglitch_e = mid;
        self.range1 = mid;
        self.range2 = mid + (hi - mid) * 0.25;
        self.trim_lo = lo;
        self.trim_hi = hi;
        self.seeded = true;
    }

    /// Allow the next open to re-seed (call after loading/switching data).
    pub fn reset_seed(&mut self) {
        self.seeded = false;
    }

    /// Render the window. Returns a [`CleanAction`] when an action button is
    /// pressed. `npts` is the current group's point count and `can_undo`
    /// enables the undo button.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        has_group: bool,
        npts: usize,
        can_undo: bool,
    ) -> Option<CleanAction> {
        let mut action = None;
        let mut open = self.open;
        crate::window::detached(
            ctx,
            "edit_xmu",
            "Edit μ(E)",
            &mut open,
            [320.0, 520.0],
            |ui| {
                if !has_group {
                    ui.weak("Load a spectrum to edit.");
                    return;
                }
                ui.label(format!("{npts} points"));
                ui.separator();

                // --- Deglitch (point / range removal) ------------------------
                ui.heading("Deglitch");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.deglitch_e)
                            .speed(0.5)
                            .suffix(" eV"),
                    );
                    if ui.button("Remove point").clicked() {
                        action = Some(CleanAction::DeglitchPoint(self.deglitch_e));
                    }
                });
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("deglitch_side")
                        .selected_text(side_name(self.range_side))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.range_side, RangeSide::Above, "above");
                            ui.selectable_value(&mut self.range_side, RangeSide::Below, "below");
                            ui.selectable_value(
                                &mut self.range_side,
                                RangeSide::Between,
                                "between",
                            );
                        });
                    ui.add(
                        egui::DragValue::new(&mut self.range1)
                            .speed(0.5)
                            .suffix(" eV"),
                    );
                    ui.add_enabled(
                        self.range_side == RangeSide::Between,
                        egui::DragValue::new(&mut self.range2)
                            .speed(0.5)
                            .suffix(" eV"),
                    );
                });
                if ui.button("Remove range").clicked() {
                    action = Some(CleanAction::DeglitchRange(
                        self.range_side,
                        self.range1,
                        self.range2,
                    ));
                }

                ui.separator();
                // --- Trim ----------------------------------------------------
                ui.heading("Trim");
                ui.horizontal(|ui| {
                    ui.label("keep");
                    ui.add(
                        egui::DragValue::new(&mut self.trim_lo)
                            .speed(0.5)
                            .suffix(" eV"),
                    );
                    ui.label("…");
                    ui.add(
                        egui::DragValue::new(&mut self.trim_hi)
                            .speed(0.5)
                            .suffix(" eV"),
                    );
                });
                if ui.button("Trim to window").clicked() {
                    action = Some(CleanAction::Trim(self.trim_lo, self.trim_hi));
                }

                ui.separator();
                // --- Smoothing -----------------------------------------------
                ui.heading("Smoothing");
                ui.add(egui::Slider::new(&mut self.smooth_sigma, 0.1..=20.0).text("width σ (eV)"));
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.smooth_form, SmoothForm::Lorentzian, "Lorentzian");
                    ui.radio_value(&mut self.smooth_form, SmoothForm::Gaussian, "Gaussian");
                });
                if ui.button("Smooth μ(E)").clicked() {
                    action = Some(CleanAction::Smooth(self.smooth_sigma, self.smooth_form));
                }

                ui.separator();
                if ui
                    .add_enabled(can_undo, egui::Button::new("Undo last edit"))
                    .clicked()
                {
                    action = Some(CleanAction::Undo);
                }
            },
        );
        self.open = open;
        action
    }
}

/// Display name for a deglitch range side.
fn side_name(s: RangeSide) -> &'static str {
    match s {
        RangeSide::Above => "above",
        RangeSide::Below => "below",
        RangeSide::Between => "between",
    }
}
