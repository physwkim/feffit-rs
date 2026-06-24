//! Shared `siplot::Plot1D` construction for the GUI.
//!
//! Every plot-bearing tab/window builds its plot through [`new_plot1d`] so they
//! all get the same house data margins. siplot's `DataMargins` default to zero,
//! which draws curves flush against the axis frame (a sample at the data maximum
//! sits exactly on the top frame line and is visually clipped). A uniform margin
//! on every side — matching matplotlib's `axes.margins` default — keeps the data
//! off the frame, and routing every plot through one constructor means no site
//! can forget it.

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use siplot::{DataMargins, ItemHandle, Plot1D, PlotId, Symbol};

/// Fraction of the data range left blank on each side of the data extent, so
/// samples at the extremes are lifted just off the axis frame without wasting
/// visible area (matplotlib's `axes.margins` default of 0.05 leaves too much
/// blank space for the spiky XANES/EXAFS curves seen here).
const DATA_MARGIN: f64 = 0.02;

/// Build a [`Plot1D`] with the house data margins applied. All GUI plots are
/// constructed through this so none can forget the margin — siplot defaults
/// `DataMargins` to zero (data flush against the frame).
pub fn new_plot1d(render_state: &RenderState, id: PlotId) -> Plot1D {
    let mut plot = Plot1D::new(render_state, id);
    plot.plot_mut().set_data_margins(DataMargins {
        x_min: DATA_MARGIN,
        x_max: DATA_MARGIN,
        y_min: DATA_MARGIN,
        y_max: DATA_MARGIN,
    });
    // Hover crosshair + x/y coordinate readout following the pointer over the
    // data area (siplot draws both when the crosshair is on). On by default so
    // every plot reads out the value under the cursor without the user first
    // toggling the toolbar's crosshair button.
    plot.set_graph_cursor(true);
    // House colour scheme: a cohesive dark canvas (see [`set_theme`]) — the plot
    // chrome matches the egui dark panel so it blends in, with a slightly
    // lighter data area, light-grey axes, and a faint grid, so curves and the
    // in-axes legend read clearly without a jarring white box in the dark UI.
    set_theme(&mut plot, false);
    plot
}

/// Apply a self-consistent colour scheme to `plot`: the chrome + data-area
/// background, the axis/frame foreground, and the grid, chosen so curves and
/// labels read clearly *independent of the egui panel theme* (siplot otherwise
/// derives the axis colour from `ui.visuals()`, which would render light axes
/// on a light canvas under a dark UI theme). `light = false` is the cohesive
/// dark canvas used as the house default for every plot; `true` is the white
/// data area with dark axes selected by Plot Data's "Change BG color" toggle.
pub fn set_theme(plot: &mut Plot1D, light: bool) {
    // `(chrome, data, fg, grid)`: the surround behind the axes, the data area
    // itself, the axis/frame/text colour, and the grid lines. `grid` is passed
    // *translucent* on purpose: siplot applies an override grid colour at full
    // opacity (chrome.rs `with_overrides`), so a solid grey would draw crisp
    // lines that fight the curve — a low-alpha tint composites to a faint grid
    // over either canvas, matching siplot's own `with_alpha(text, 28)` default.
    let (chrome, data, fg, grid) = if light {
        (
            egui::Color32::WHITE,
            egui::Color32::WHITE,
            egui::Color32::from_gray(0x20),
            egui::Color32::from_black_alpha(0x16),
        )
    } else {
        // Cohesive dark: chrome == the egui dark panel fill (`from_gray(27)`) so
        // the plot blends into the surrounding panel, the data area one shade
        // lighter so it still reads as a distinct region, light-grey axes, and a
        // faint translucent grid.
        (
            egui::Color32::from_gray(0x1b),
            egui::Color32::from_gray(0x24),
            egui::Color32::from_gray(0xc8),
            egui::Color32::from_white_alpha(0x12),
        )
    };
    plot.set_background_colors(chrome, data);
    plot.set_foreground_colors(fg, grid);
}

/// Bright, dark-background-friendly curve colours, defined once here so every
/// plot-bearing tab draws from the same vivid palette. Matplotlib's tab10 is
/// muted and reads dimly on the dark data area set by [`set_theme`], so these
/// are lifted in luminance and saturation while keeping each hue's identity.
pub const BLUE: egui::Color32 = egui::Color32::from_rgb(0x4f, 0x9b, 0xff);
pub const ORANGE: egui::Color32 = egui::Color32::from_rgb(0xff, 0xa5, 0x3c);
pub const GREEN: egui::Color32 = egui::Color32::from_rgb(0x4a, 0xde, 0x80);
pub const RED: egui::Color32 = egui::Color32::from_rgb(0xff, 0x5d, 0x5d);
pub const PURPLE: egui::Color32 = egui::Color32::from_rgb(0xc0, 0x8c, 0xff);
pub const CYAN: egui::Color32 = egui::Color32::from_rgb(0x34, 0xd6, 0xe6);
pub const BROWN: egui::Color32 = egui::Color32::from_rgb(0xd2, 0x92, 0x6b);
pub const PINK: egui::Color32 = egui::Color32::from_rgb(0xf5, 0x8f, 0xd6);

/// Cyclic palette (tab10 order, brightened) for plots that draw an arbitrary
/// number of curves on the dark canvas. Index with `PALETTE[i % PALETTE.len()]`.
pub const PALETTE: [egui::Color32; 8] = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, CYAN];

/// The muted matplotlib tab10 palette, for the white "Change BG" canvas where
/// the bright [`PALETTE`] would wash out (tab10 is tuned for a light background).
pub const PALETTE_LIGHT: [egui::Color32; 8] = [
    egui::Color32::from_rgb(0x1f, 0x77, 0xb4),
    egui::Color32::from_rgb(0xff, 0x7f, 0x0e),
    egui::Color32::from_rgb(0x2c, 0xa0, 0x2c),
    egui::Color32::from_rgb(0xd6, 0x27, 0x28),
    egui::Color32::from_rgb(0x94, 0x67, 0xbd),
    egui::Color32::from_rgb(0x8c, 0x56, 0x4b),
    egui::Color32::from_rgb(0xe3, 0x77, 0xc2),
    egui::Color32::from_rgb(0x17, 0xbe, 0xcf),
];

/// A [`Plot1D`] bundled with the legend entries for the curves currently on it.
///
/// siplot exposes no per-item colour, and its own `show_legend` draws a
/// visibility-toggle control on every row that the GUI does not want; so each
/// curve's `(label, colour)` is tracked here as it is added and [`show`] draws a
/// plain in-axes legend from it. Curves added through
/// [`Plot::add_curve_with_legend`] are recorded; bare `add_curve` curves (no
/// legend) are not. Every other operation falls through to the inner [`Plot1D`]
/// via `Deref`/`DerefMut`, so all existing plot calls are unchanged.
pub struct Plot {
    inner: Plot1D,
    legend: Vec<(String, egui::Color32)>,
}

impl std::ops::Deref for Plot {
    type Target = Plot1D;
    fn deref(&self) -> &Plot1D {
        &self.inner
    }
}

impl std::ops::DerefMut for Plot {
    fn deref_mut(&mut self) -> &mut Plot1D {
        &mut self.inner
    }
}

impl Plot {
    /// Build a [`Plot`] with the house data margins (see [`new_plot1d`]) and an
    /// empty legend.
    pub fn new(render_state: &RenderState, id: PlotId) -> Self {
        Self {
            inner: new_plot1d(render_state, id),
            legend: Vec::new(),
        }
    }

    /// Add a curve and record its `(label, colour)` for the in-axes legend.
    /// Shadows [`Plot1D::add_curve_with_legend`], so existing call sites record
    /// their legend entry without change.
    pub fn add_curve_with_legend(
        &mut self,
        x: &[f64],
        y: &[f64],
        color: egui::Color32,
        legend: impl Into<String>,
    ) -> ItemHandle {
        let legend = legend.into();
        self.legend.push((legend.clone(), color));
        self.inner.add_curve_with_legend(x, y, color, legend)
    }

    /// Clear all plot items and the recorded legend together. Shadows
    /// [`Plot1D::clear`], so a rebuild that calls `clear()` also empties the
    /// legend (otherwise stale entries would accumulate each frame).
    pub fn clear(&mut self) {
        self.inner.clear();
        self.legend.clear();
    }

    /// Clear curve items and the recorded legend together. Shadows
    /// [`Plot1D::clear_curves`].
    pub fn clear_curves(&mut self) {
        self.inner.clear_curves();
        self.legend.clear();
    }
}

/// Draw `plot`'s standard toolbar plus the house extras on one row, then leave
/// the plot itself to a following `plot.show(ui)`. Every plot-bearing tab/window
/// draws its toolbar through this (instead of `plot.show_toolbar` directly) so
/// they all expose the same controls — currently siplot's `symbol_tool_button`,
/// a "Symbol" menu that toggles data-point markers (size + shape) on every
/// curve, which the bare toolbar omits.
///
/// siplot's ready-made `symbol_tool_button` is a `Plot2D` (image) method, so for
/// the curve `Plot1D` the same control is built here on top of the underlying
/// `PlotWidget::set_all_symbols` / `set_all_symbol_sizes`.
pub fn toolbar(plot: &mut Plot1D, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        plot.show_toolbar(ui);
        symbol_menu(plot, ui);
    });
}

/// Render `plot` as one unit: its toolbar on top, then the plot canvas filling
/// the full width with the legend *overlaid* in the top-right corner of the data
/// area (matplotlib-style), mapping each curve's color/symbol to the name set
/// with `set_item_legend`. Every plot-bearing tab/window draws through this
/// (instead of `toolbar` + `plot.show` by hand) so they all get the same toolbar
/// *and* a visible legend — siplot draws no in-axes legend, so without this call
/// the curve names never appear. The legend floats over the canvas (an egui
/// `Area`) instead of taking a separate column, so it no longer steals width.
pub fn show(plot: &mut Plot, ui: &mut egui::Ui) {
    toolbar(&mut plot.inner, ui);
    // The canvas fills the whole width; `PlotResponse::transform` carries the
    // data-area rectangle (screen points) that we anchor the legend overlay to.
    let area = plot.inner.show(ui).transform.area;

    // No labelled curve — nothing to put in the legend, so skip the overlay.
    if plot.legend.is_empty() {
        return;
    }
    // The legend text colour contrasts with the plot's data-area background
    // (set by `set_theme`), not the egui panel theme, so labels stay legible
    // whether the canvas is light or dark.
    let data_bg = plot.inner.data_background_color();

    // Float the legend over the canvas — by default in the data area's top-right
    // corner, draggable from anywhere on it (no handle): a movable foreground
    // `Area` holds the legend, and egui's hit test routes a *drag* to the movable
    // Area while a *click* still reaches the rows beneath. siplot's rows sense
    // click only, so `hit_test` reports click→row, drag→Area (egui hit_test.rs:
    // "the top Button and the ScrollArea behind it"); the user grabs the labels
    // themselves to move it. egui remembers the dragged spot (keyed by plot id)
    // and `constrain_to` keeps it within the axes.
    const PAD: f32 = 6.0;
    let legend_id = egui::Id::new(plot.inner.backend().plot().id).with("legend_overlay");
    let ctx = ui.ctx().clone();
    egui::Area::new(legend_id)
        .order(egui::Order::Foreground)
        .movable(true)
        .constrain_to(area)
        .default_pos(area.right_top() + egui::vec2(-PAD, PAD))
        .pivot(egui::Align2::RIGHT_TOP)
        .show(&ctx, |ui| {
            // A plain transparent box that only pads the legend off the axis
            // frame — no fill or border of its own.
            egui::Frame::new()
                .inner_margin(egui::Margin::same(PAD as i8))
                .show(ui, |ui| {
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        ui.set_max_width((area.width() * 0.45).clamp(90.0, 200.0));
                        // Disable drag-to-scroll: the ScrollArea sits above the Area
                        // and would otherwise sense the drag and scroll instead of
                        // letting the movable Area move. Scrollbar + wheel still scroll
                        // an overflowing legend (only the `drag` source is turned off).
                        egui::ScrollArea::vertical()
                            .max_height((area.height() - 2.0 * PAD).max(40.0))
                            .scroll_source(egui::scroll_area::ScrollSource {
                                drag: false,
                                ..Default::default()
                            })
                            .show(ui, |ui| {
                                draw_legend(ui, &plot.legend, data_bg);
                            });
                    });
                });
        });
}

/// Draw the plain in-axes legend: one row per entry — a short line swatch in the
/// curve's colour, then its label. No fill, border, or visibility toggle (unlike
/// siplot's `show_legend`, which the GUI deliberately bypasses); just the colour
/// key over the transparent overlay. Rows are non-interactive so a drag anywhere
/// on the legend moves the enclosing `Area` rather than being captured.
fn draw_legend(ui: &mut egui::Ui, entries: &[(String, egui::Color32)], data_bg: egui::Color32) {
    const SWATCH_W: f32 = 22.0;
    const SWATCH_H: f32 = 12.0;
    // Pick a label colour that contrasts with the data-area background (dark
    // text on a light canvas, light text on a dark one) rather than inheriting
    // the egui theme's text colour, which need not match the canvas.
    let luma = 0.299 * data_bg.r() as f32 + 0.587 * data_bg.g() as f32 + 0.114 * data_bg.b() as f32;
    let text = if luma > 128.0 {
        egui::Color32::from_gray(0x20)
    } else {
        egui::Color32::from_gray(0xe0)
    };
    ui.spacing_mut().item_spacing.y = 2.0;
    for (label, color) in entries {
        ui.horizontal(|ui| {
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(SWATCH_W, SWATCH_H), egui::Sense::hover());
            let y = rect.center().y;
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left() + 2.0, y),
                    egui::pos2(rect.right() - 2.0, y),
                ],
                egui::Stroke::new(2.0, *color),
            );
            ui.add_space(4.0);
            ui.colored_label(text, label.as_str());
        });
    }
}

/// A "Symbol" menu that toggles data-point markers (shape + size) on every curve
/// of `plot`, mirroring siplot's `Plot2D::symbol_tool_button` for the curve
/// widget. The chosen size is remembered in egui temp memory, keyed by plot id.
fn symbol_menu(plot: &mut Plot1D, ui: &mut egui::Ui) {
    let size_id = egui::Id::new(plot.backend().plot().id).with("symbol_menu_size");
    let mut size = ui.data(|d| d.get_temp::<f32>(size_id)).unwrap_or(7.0);
    ui.menu_button("Symbol", |ui| {
        ui.horizontal(|ui| {
            ui.label("Size:");
            if ui
                .add(egui::DragValue::new(&mut size).range(1.0..=20.0).speed(0.5))
                .on_hover_text("Marker size for every curve")
                .changed()
            {
                plot.set_all_symbol_sizes(size);
            }
        });
        ui.separator();
        if ui.button("None (line only)").clicked() {
            plot.set_all_symbols(None);
            ui.close();
        }
        for symbol in Symbol::ALL {
            if ui.button(symbol.name()).clicked() {
                plot.set_all_symbols(Some(symbol));
                ui.close();
            }
        }
    });
    ui.data_mut(|d| d.insert_temp(size_id, size));
}
