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
/// the standard inner margin — content never hugs the viewport edges — and in a
/// vertical scroll, so a form taller than the viewport scrolls instead of
/// clipping its last rows. A body is therefore *single-flow*: a widget that
/// takes `available_size` (a plot, a 3D scene, a list with `auto_shrink(false)`)
/// sees an infinite height inside the scroll and cannot live here. A window
/// built around such a widget lays out its own panels through
/// [`detached_panels`] (panels are added to the viewport root, and their own
/// frames supply the margins) with [`fill_with_footer`] for anything it shows
/// below the fill widget.
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
        egui::CentralPanel::default().show(ui, |ui| scrolled(ui, &mut add_contents));
    });
}

/// As [`detached`] but with `action_row` pinned in its own bottom panel, for a
/// window whose last row is its primary action (OK, Autobk Start, Save): the
/// action stays in view however long the form above it scrolls. The panel
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
        egui::CentralPanel::default().show(ui, |ui| scrolled(ui, &mut add_contents));
    });
}

/// The single-flow body scroll shared by [`detached`] and
/// [`detached_with_action`]: fills the panel (no auto-shrink, so width-filling
/// widgets keep the panel width) and scrolls vertically past the viewport.
fn scrolled(ui: &mut egui::Ui, add_contents: &mut impl FnMut(&mut egui::Ui)) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| add_contents(ui));
}

/// Lay out `fill` — a widget that takes all remaining space (a plot, a 3D
/// scene, a `auto_shrink(false)` list) — with `footer` pinned in a bottom panel
/// under it. The fill widget reads `available_size`, so anything laid out
/// *after* it in the same flow lands past the viewport edge (a results table
/// below a plot was never visible); a bottom panel reserves its height before
/// the fill widget is drawn. The footer scrolls past 40 % of the height so a
/// long results table cannot starve the fill widget. `state` is handed to both
/// closures in turn so a caller can draw its footer and its plot from the same
/// `&mut self` (two capturing closures could not both borrow it).
pub fn fill_with_footer<S>(
    ui: &mut egui::Ui,
    id_source: &str,
    state: &mut S,
    footer: impl FnOnce(&mut S, &mut egui::Ui),
    fill: impl FnOnce(&mut S, &mut egui::Ui),
) {
    let cap = (ui.available_height() * 0.4).max(60.0);
    egui::Panel::bottom(egui::Id::new((id_source, "footer"))).show(ui, |ui| {
        fixed_viewport_scroll(cap)
            .id_salt((id_source, "footer_scroll"))
            .show(ui, |ui| footer(state, ui));
    });
    egui::CentralPanel::default().show(ui, |ui| fill(state, ui));
}

/// A vertical `ScrollArea` whose viewport is exactly `height` tall, or shorter
/// when its content is. Required wherever the enclosing container sizes itself
/// from its own content — a floating `egui::Area`, a non-resizable
/// `egui::Panel` — because such a container hands its content *last frame's*
/// size as this frame's max rect: a viewport that took the available height
/// (`max_height` alone) could never grow past what it was first laid out as,
/// and settled at egui's 64 pt `min_scrolled` floor for good. Pinning the
/// minimum too takes the available height out of the equation; auto-shrink
/// still fits shorter content.
pub fn fixed_viewport_scroll(height: f32) -> egui::ScrollArea {
    egui::ScrollArea::vertical()
        .min_scrolled_height(height)
        .max_height(height)
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
    let state_id = open_state_id(id_source);
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
    let viewport_id = viewport_id(id_source, generation);
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

/// Bring the detached window `id_source` to the front if it is open. A request
/// to open an already-open pop-up — "Open New file" while the loader sits
/// behind the main window — otherwise changes nothing on screen: the flag is
/// already set and [`detached_panels`] only creates a viewport, never raises
/// one. A no-op while the window is closed (its next open creates it in front).
pub fn focus(ctx: &egui::Context, id_source: &str) {
    if let Some((true, generation)) =
        ctx.data(|d| d.get_temp::<(bool, u64)>(open_state_id(id_source)))
    {
        ctx.send_viewport_cmd_to(
            viewport_id(id_source, generation),
            egui::ViewportCommand::Focus,
        );
    }
}

/// The per-`id_source` memory slot holding `(open last frame, generation)`.
fn open_state_id(id_source: &str) -> egui::Id {
    egui::Id::new(("detached_open_state", id_source))
}

/// The viewport id of `id_source`'s `generation`-th open (see the reopen note
/// on [`detached`]).
fn viewport_id(id_source: &str, generation: u64) -> egui::ViewportId {
    egui::ViewportId::from_hash_of((id_source, generation))
}
