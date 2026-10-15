//! Detached pop-up windows shown as native OS viewports.
//!
//! An `egui::Window` lives inside the main viewport, so it can never be dragged
//! outside the main application window. To make every pop-up (periodic table,
//! LCF/PCA, XANES, MBACK, the calculators, Plot Data, wavelet, the 3D site
//! viewer, …) a real OS window that can move anywhere on screen, each is shown
//! through [`detached`] (or [`detached_panels`] for bodies that build their own
//! panel layout), which opens an egui *immediate viewport*. Routing them all
//! through one place keeps the viewport-id, open/close, and margin discipline
//! together — a pop-up cannot accidentally stay clamped to the main window or
//! render its content flush against the window edges.

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
/// The body is wrapped in a default [`egui::CentralPanel`], whose frame carries
/// the standard inner margin — content never hugs the viewport edges. A window
/// that lays out its own `SidePanel`/`CentralPanel` split must use
/// [`detached_panels`] instead (panels are added to the viewport root, and their
/// own frames supply the margins).
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
    detached_panels(ctx, id_source, title, open, default_size, |ui| {
        egui::CentralPanel::default().show(ui, |ui| add_contents(ui));
    });
}

/// As [`detached`] but with `action_row` pinned in its own bottom panel, for a
/// window whose last row is its primary action (OK, Autobk Start). In the
/// single-flow [`detached`] body a tall form pushes that row past the viewport
/// edge, and a detached window has no outer scroll, so the action becomes
/// unreachable; a bottom panel holds its space by construction. The panel
/// renders before `add_contents`, so anything the action row reads (an enabled
/// gate) is one frame stale — fine for display state; apply the action itself
/// from a flag after this call returns.
pub fn detached_with_action(
    ctx: &egui::Context,
    id_source: &str,
    title: &str,
    open: &mut bool,
    default_size: [f32; 2],
    mut add_contents: impl FnMut(&mut egui::Ui),
    mut action_row: impl FnMut(&mut egui::Ui),
) {
    detached_panels(ctx, id_source, title, open, default_size, |ui| {
        egui::Panel::bottom(egui::Id::new((id_source, "action_row"))).show(ui, |ui| action_row(ui));
        egui::CentralPanel::default().show(ui, |ui| add_contents(ui));
    });
}

/// [`detached`] without the [`egui::CentralPanel`] wrap, for windows whose body
/// builds its own `SidePanel`/`CentralPanel` layout on the viewport root (the
/// panels' frames then supply the margins). Every other window goes through
/// [`detached`] so it cannot forget the standard margin.
pub fn detached_panels(
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
