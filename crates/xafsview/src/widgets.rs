//! Shared action-button styling, so buttons match the original XAFSView (LabVIEW)
//! forms instead of being default-sized and scattered: a group of action buttons
//! is uniform width, the form's main verb is an amber *primary* button, and
//! *Exit* is reddish — the colour/size language visible in 그림 1-2-1-1 (Autobk),
//! 그림 1-2-2-2 (Feffit), 그림 1-2-4 (Atoms), 그림 1-2-5 (Feff), 그림 1-2-6 (Folders).

use eframe::egui::{self, Color32, Response, Vec2};

/// Uniform button size for a bottom action row (Feffit / Feff / Atoms forms).
pub const ROW_BTN: Vec2 = Vec2::new(112.0, 28.0);
/// Chunky button size for the Autobk 2×2 cluster (그림 1-2-1-1).
pub const CHUNKY_BTN: Vec2 = Vec2::new(124.0, 40.0);
/// "Browse…" button size for the Folders rows (그림 1-2-6 labelled these "push";
/// relabelled to the conventional Browse verb so the action reads clearly).
pub const BROWSE_BTN: Vec2 = Vec2::new(76.0, 22.0);

/// Reddish "Exit" fill, matching the original Exit buttons.
const EXIT_FILL: Color32 = Color32::from_rgb(0x9c, 0x4a, 0x4a);
/// Amber primary-action fill (Autobk Start / Run / Execute / OK in the originals).
const PRIMARY_FILL: Color32 = Color32::from_rgb(0x9a, 0x7d, 0x2e);

/// A fixed-minimum-size plain button, so a row of action buttons is uniform.
pub fn action(ui: &mut egui::Ui, text: &str, size: Vec2) -> Response {
    ui.add(egui::Button::new(text).min_size(size))
}

/// A fixed-minimum-size button, enabled only when `enabled`.
pub fn action_enabled(ui: &mut egui::Ui, text: &str, size: Vec2, enabled: bool) -> Response {
    ui.add_enabled(enabled, egui::Button::new(text).min_size(size))
}

/// The amber primary-action button (the form's main verb), enabled-gated.
pub fn primary(ui: &mut egui::Ui, text: &str, size: Vec2, enabled: bool) -> Response {
    ui.add_enabled(
        enabled,
        egui::Button::new(text).fill(PRIMARY_FILL).min_size(size),
    )
}

/// The reddish "Exit" button.
pub fn exit(ui: &mut egui::Ui, size: Vec2) -> Response {
    ui.add(egui::Button::new("Exit").fill(EXIT_FILL).min_size(size))
}
