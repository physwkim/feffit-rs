//! Detached pop-up windows shown as native OS viewports.
//!
//! An `egui::Window` lives inside the main viewport, so it can never be dragged
//! outside the main application window. To make every pop-up (periodic table,
//! LCF/PCA, XANES, MBACK, the calculators, Plot Data, wavelet, the 3D site
//! viewer, …) a real OS window that can move anywhere on screen, each is shown
//! through [`detached`], which opens an egui *immediate viewport*. Routing them
//! all through one helper keeps the viewport-id and open/close discipline in a
//! single place — a pop-up cannot accidentally stay clamped to the main window.

use eframe::egui;

/// Show `add_contents` in a native OS window (an egui immediate viewport) that
/// can be moved anywhere on screen, independent of the main window.
///
/// `open` is set to `false` when the user closes the viewport with its window
/// close button. `id_source` must be unique per pop-up — it seeds the
/// [`egui::ViewportId`]. `default_size` is the initial inner size in points.
///
/// The body runs every frame the parent paints (an immediate viewport), so it
/// may borrow application state mutably exactly like an `egui::Window` closure.
/// If the backend cannot open separate OS windows, egui falls back to an
/// embedded window inside the main viewport, so the pop-up still works.
pub fn detached(
    ctx: &egui::Context,
    id_source: &str,
    title: &str,
    open: &mut bool,
    default_size: [f32; 2],
    mut add_contents: impl FnMut(&mut egui::Ui),
) {
    if !*open {
        return;
    }
    let viewport_id = egui::ViewportId::from_hash_of(id_source);
    let builder = egui::ViewportBuilder::default()
        .with_title(title)
        .with_inner_size(default_size);
    ctx.show_viewport_immediate(viewport_id, builder, |ui, _class| {
        add_contents(ui);
        if ui.ctx().input(|i| i.viewport().close_requested()) {
            *open = false;
        }
    });
}
