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
///
/// Reopen discipline: an immediate viewport closed via its window button cannot
/// be reliably recreated under the *same* [`egui::ViewportId`] — eframe/winit
/// races the old OS window's teardown against the recreate, so the reopened
/// window immediately closes itself again ("won't reopen after closing"). To
/// avoid that, every closed→open transition bumps a per-`id_source` generation
/// so each fresh open gets a brand-new `ViewportId` with no pending teardown.
/// The generation is stable *while* the window stays open, so it does not
/// recreate (flicker) each frame.
pub fn detached(
    ctx: &egui::Context,
    id_source: &str,
    title: &str,
    open: &mut bool,
    default_size: [f32; 2],
    mut add_contents: impl FnMut(&mut egui::Ui),
) {
    // Bump the open generation on each closed→open edge (see the reopen note
    // above). Kept in egui's own per-id memory so the helper stays stateless.
    let state_id = egui::Id::new(("detached_open_state", id_source));
    let generation = ctx.data_mut(|d| {
        let (was_open, mut g) = d.get_temp::<(bool, u64)>(state_id).unwrap_or((false, 0));
        if *open && !was_open {
            g = g.wrapping_add(1);
        }
        d.insert_temp(state_id, (*open, g));
        g
    });
    if !*open {
        return;
    }
    let viewport_id = egui::ViewportId::from_hash_of((id_source, generation));
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
