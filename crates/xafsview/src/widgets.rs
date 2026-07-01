//! Shared action-button styling, so buttons match the original XAFSView (LabVIEW)
//! forms instead of being default-sized and scattered: a group of action buttons
//! is uniform width, the form's main verb is an amber *primary* button, and
//! *Exit* is reddish — the colour/size language visible in 그림 1-2-1-1 (Autobk),
//! 그림 1-2-2-2 (Feffit), 그림 1-2-4 (Atoms), 그림 1-2-5 (Feff), 그림 1-2-6 (Folders).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

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

/// The bare file name of `path`, for picker lists and legends.
pub fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Render `list` as a multi-select column of file-name rows: a plain click
/// toggles a row, and a shift-click extends an inclusive range from the last
/// plain-clicked row (`anchor`). `hi` holds the highlighted paths. Ctrl/Cmd+A
/// selects the whole list while the pointer is over it (so in a two-pane picker
/// it targets the pane under the cursor, not both at once). Shared by the Plot
/// Data file picker and the Make-μ(E) batch picker (their two-pane transfer
/// lists). Resetting the lists should clear `anchor`, which stales on change.
pub fn select_list(
    ui: &mut egui::Ui,
    list: &[PathBuf],
    hi: &mut HashSet<PathBuf>,
    anchor: &mut Option<usize>,
) {
    let shift = ui.input(|i| i.modifiers.shift);
    for (idx, path) in list.iter().enumerate() {
        let selected = hi.contains(path);
        if ui.selectable_label(selected, file_name_of(path)).clicked() {
            match (shift, *anchor) {
                (true, Some(a)) => {
                    for p in &list[a.min(idx)..=a.max(idx)] {
                        hi.insert(p.clone());
                    }
                }
                _ => {
                    if !hi.remove(path) {
                        hi.insert(path.clone());
                    }
                    *anchor = Some(idx);
                }
            }
        }
    }
    // Ctrl/Cmd+A selects every item in the list the pointer is over. `command`
    // is Cmd on macOS / Ctrl elsewhere; `ctrl` also matches so a literal Ctrl+A
    // works on macOS too. Gated on hover so the shortcut hits the list under the
    // cursor rather than every `select_list` on screen.
    if ui.ui_contains_pointer()
        && ui.input(|i| (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::A))
    {
        for path in list {
            hi.insert(path.clone());
        }
    }
}

/// Render `labels` as a multi-select column of `selectable_label` rows keyed by
/// index: a plain click toggles a row, a shift-click extends an inclusive range
/// from the last plain-clicked row (`anchor`). `sel` holds the selected indices.
///
/// The index-keyed sibling of [`select_list`] — for a stable, append-only list
/// (e.g. the Feffit batch's session groups) where rows are identified by
/// position rather than by value, and don't move between panes. Resetting the
/// list should clear `anchor`, which stales on change.
pub fn select_index_list<'a>(
    ui: &mut egui::Ui,
    labels: impl Iterator<Item = &'a str>,
    sel: &mut HashSet<usize>,
    anchor: &mut Option<usize>,
) {
    let shift = ui.input(|i| i.modifiers.shift);
    for (idx, label) in labels.enumerate() {
        let selected = sel.contains(&idx);
        if ui.selectable_label(selected, label).clicked() {
            match (shift, *anchor) {
                (true, Some(a)) => {
                    for i in a.min(idx)..=a.max(idx) {
                        sel.insert(i);
                    }
                }
                _ => {
                    if !sel.remove(&idx) {
                        sel.insert(idx);
                    }
                    *anchor = Some(idx);
                }
            }
        }
    }
}

/// A small square "remove" button: a bordered box with an ✕ painted inside.
/// Painted rather than set from a glyph because the default UI fonts lack ✕
/// (U+2715) and every box-with-✕ codepoint (☒/⊠/╳), which would render as a
/// missing-glyph tofu box. Use for any "remove this row" affordance.
pub fn delete_box(ui: &mut egui::Ui) -> Response {
    let side = ui.spacing().interact_size.y.min(18.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(side), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let stroke = ui.style().interact(&resp).fg_stroke;
        let painter = ui.painter();
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(2),
            stroke,
            egui::StrokeKind::Inside,
        );
        let x = rect.shrink(side * 0.3);
        painter.line_segment([x.left_top(), x.right_bottom()], stroke);
        painter.line_segment([x.right_top(), x.left_bottom()], stroke);
    }
    resp
}
