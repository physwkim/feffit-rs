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
use siplot::{DataMargins, ItemHandle, Plot1D, PlotId, Roi, Symbol};

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
    // Coordinate-readout toggle state. siplot fuses the crosshair guide lines and
    // the coordinate readout behind this single `graph_cursor` flag (drawing both
    // or neither); xafsview instead draws the cursor overlay itself and splits the
    // two controls — `graph_cursor` now backs only the toolbar's "Show cursor
    // coordinates" button as the *readout* toggle, while a separate
    // `Plot::crosshair` field backs our own crosshair-lines toggle (see [`show`],
    // [`toolbar`], and [`draw_cursor_overlay`]). The readout is on by default.
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

/// Colour of the draggable FT-window band ([`set_window`]): a translucent amber
/// that reads against the blue χ curves. siplot fills the band at half this
/// alpha (`color.a()/2`) and outlines the edges at full alpha, so the edges stay
/// clearly grabbable while the interior never hides the data underneath.
const WINDOW_COLOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(0xcc, 0x8c, 0x30, 0xcc);

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

/// A draggable `[min, max]` window on a plot's x-axis, drawn as a shaded
/// vertical band the user can grab by either edge. The reduce tab uses it to
/// expose the FT k-window directly on the kʷ·χ(k) curve: [`set_window`] requests
/// the band before [`show`], and [`take_window_drag`] reports a drag back so the
/// caller can update its parameters and recompute.
#[derive(Clone, Copy, PartialEq)]
pub struct AxisWindow {
    /// Lower bound (left edge), in the plot's x units.
    pub min: f64,
    /// Upper bound (right edge), in the plot's x units.
    pub max: f64,
}

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
    /// The legend overlay's position as a fraction `(x, y)` of the data-area
    /// rectangle (its RIGHT_TOP pivot), so it keeps the same relative spot when
    /// the plot is resized. `None` until first shown (defaults to top-right);
    /// updated whenever the user drags the legend. See [`show`].
    legend_frac: Option<egui::Pos2>,
    /// Whether the crosshair guide lines are drawn under the pointer. Toggled by
    /// the toolbar's own crosshair button ([`crosshair_button`]); independent of
    /// the coordinate readout, which is gated by siplot's `graph_cursor` flag.
    /// On by default. See [`show`] and [`draw_cursor_overlay`].
    crosshair: bool,
    /// The draggable FT-window band, a siplot `VRange` ROI: `Some(index)` once
    /// added. Managed entirely by [`set_window`]/[`show`] — the band is shown
    /// only on frames where [`set_window`] was called and removed otherwise, so
    /// it never leaks onto a graph (or a tab sharing this plot) that has no
    /// window.
    window_roi: Option<usize>,
    /// The bounds last pushed to the window ROI by [`set_window`]. [`show`] tells
    /// a user drag (ROI bounds changed during the inner `show`) from our own sync
    /// by comparing the post-show bounds against this.
    window_target: Option<AxisWindow>,
    /// Set by [`set_window`] each frame it is called; [`show`] removes the band
    /// when it was *not* requested this frame, then clears the flag.
    window_requested: bool,
    /// A user drag of either band edge, detected in [`show`] and taken by
    /// [`take_window_drag`].
    window_dragged: Option<AxisWindow>,
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
            legend_frac: None,
            crosshair: true,
            window_roi: None,
            window_target: None,
            window_requested: false,
            window_dragged: None,
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
        self.rearm_reset_zoom();
    }

    /// Clear curve items and the recorded legend together. Shadows
    /// [`Plot1D::clear_curves`].
    pub fn clear_curves(&mut self) {
        self.inner.clear_curves();
        self.legend.clear();
        self.rearm_reset_zoom();
    }

    /// Re-arm the toolbar's "Reset Zoom" to refit the *current* data.
    ///
    /// siplot captures the home view once on the plot's first show and the
    /// "Reset Zoom" menu item restores exactly that snapshot — it never refreshes
    /// it when the content changes. A long-lived plot whose data, item, or loaded
    /// files change (every window here) would then reset to a stale range from
    /// the first frame. Every rebuild funnels through `clear`/`clear_curves`, so
    /// dropping `home_limits` here makes siplot recapture the freshly auto-fit
    /// view (autoscale refits the live view on the following add) as the new home.
    fn rearm_reset_zoom(&mut self) {
        self.inner.plot_mut().home_limits = None;
    }
}

/// Draw `plot`'s standard toolbar plus the house extras on one row, then leave
/// the plot itself to a following [`show`]. Every plot-bearing tab/window draws
/// its toolbar through this (instead of `plot.show_toolbar` directly) so they all
/// expose the same controls beyond siplot's built-ins:
///
/// - a [`crosshair_button`] that toggles the crosshair guide lines (siplot's own
///   cursor button is repurposed as the coordinate-readout toggle, see [`show`]);
/// - a "Symbol" menu that toggles data-point markers (size + shape) on every
///   curve, which the bare toolbar omits. siplot's ready-made `symbol_tool_button`
///   is a `Plot2D` (image) method, so for the curve `Plot1D` the same control is
///   built here on top of `PlotWidget::set_all_symbols` / `set_all_symbol_sizes`.
pub fn toolbar(plot: &mut Plot, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        plot.inner.show_toolbar(ui);
        crosshair_button(&mut plot.crosshair, ui);
        symbol_menu(&mut plot.inner, ui);
    });
}

/// Request the draggable FT-window band at `window` for this frame, drawn as a
/// shaded vertical band on the active graph's x-axis. Call once per frame
/// *before* [`show`] for every frame the band should appear; a frame with no
/// call removes it (so the band never leaks onto a graph — or a tab sharing this
/// plot — that has no window). The user drags either edge; [`show`] then reports
/// the new bounds through [`take_window_drag`].
///
/// The ROI is only re-synced when `window` differs from the value last pushed —
/// a typed parameter, a graph switch, an Autobk Start — so an in-progress edge
/// drag is left to siplot's own interaction and not overwritten mid-drag.
pub fn set_window(plot: &mut Plot, window: AxisWindow) {
    plot.window_requested = true;
    match plot.window_roi {
        None => {
            let idx = plot.inner.add_roi(Roi::VRange {
                x: (window.min, window.max),
            });
            plot.inner.set_roi_name(idx, "FT window");
            plot.inner.set_roi_color(idx, WINDOW_COLOR);
            if let Some(managed) = plot.inner.rois_mut().get_mut(idx) {
                managed.fill = true;
            }
            plot.window_roi = Some(idx);
            plot.window_target = Some(window);
        }
        Some(idx) => {
            if plot.window_target != Some(window) {
                if let Some(managed) = plot.inner.rois_mut().get_mut(idx) {
                    managed.roi = Roi::VRange {
                        x: (window.min, window.max),
                    };
                }
                plot.window_target = Some(window);
            }
        }
    }
}

/// Take the FT-window drag detected during the last [`show`], if any: the new
/// `[min, max]` bounds the user dragged the band to. Returns `Some` once per drag
/// frame and clears the pending drag, so the caller updates its parameters and
/// recomputes exactly once.
pub fn take_window_drag(plot: &mut Plot) -> Option<AxisWindow> {
    plot.window_dragged.take()
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
    toolbar(plot, ui);

    // siplot fuses the crosshair guide lines and the coordinate readout behind a
    // single `graph_cursor` flag, drawing both or neither. The GUI splits them so
    // each has its own toolbar toggle: `graph_cursor` (the toolbar's "Show cursor
    // coordinates" button) gates the *readout*, and `Plot::crosshair` (our own
    // toolbar button) gates the *lines*. xafsview draws the overlay itself: read
    // the readout toggle off the flag, suppress siplot's own (fused) cursor for
    // this frame's render, then restore the flag so the toolbar button keeps its
    // state. `set_graph_cursor` only flips a bool, so this is cheap.
    let readout_on = plot.inner.graph_cursor();
    plot.inner.set_graph_cursor(false);
    let resp = plot.inner.show(ui);
    plot.inner.set_graph_cursor(readout_on);

    // FT-window band. siplot drives the VRange edge drag inside the inner `show`
    // above, mutating the ROI in place; detect that by comparing the post-show
    // bounds against the value we last pushed (`window_target`). Then keep the
    // band only if it was requested this frame, else remove it so it never leaks
    // onto a windowless graph or another tab sharing this plot. Drain siplot's
    // event buffer (RoiAdded/RoiChanged accumulate per frame) since nothing here
    // consumes it — we read state, not events.
    plot.inner.drain_events();
    if let Some(idx) = plot.window_roi
        && let Some(managed) = plot.inner.rois().get(idx)
        && let Roi::VRange { x: (min, max) } = managed.roi
        && let Some(t) = plot.window_target
        && ((min - t.min).abs() > 1e-9 || (max - t.max).abs() > 1e-9)
    {
        let w = AxisWindow { min, max };
        plot.window_dragged = Some(w);
        plot.window_target = Some(w);
    }
    if !plot.window_requested
        && let Some(idx) = plot.window_roi.take()
    {
        plot.inner.remove_roi(idx);
        plot.window_target = None;
    }
    plot.window_requested = false;

    let crosshair_on = plot.crosshair;
    // `PlotResponse::transform` carries the data-area rectangle (screen points)
    // that we anchor the legend overlay to and map the pointer through.
    let transform = resp.transform;
    let area = transform.area;

    // The readout and legend text colours contrast with the plot's data-area
    // background (set by `set_theme`), not the egui panel theme, so they stay
    // legible whether the canvas is light or dark.
    let data_bg = plot.inner.data_background_color();

    // Cursor overlay following the pointer over the data area: the coordinate
    // readout when its toggle is on, and the crosshair guide lines when theirs
    // is. Skip the hit-test entirely when both are off.
    if (readout_on || crosshair_on)
        && let Some(pos) = resp.response.hover_pos()
        && area.contains(pos)
    {
        let data = transform.pixel_to_data(pos);
        draw_cursor_overlay(ui, area, pos, data, readout_on, crosshair_on, data_bg);
    }

    // No labelled curve — nothing to put in the legend, so skip the overlay.
    if plot.legend.is_empty() {
        return;
    }

    // Float the legend over the canvas, draggable from anywhere on it (no
    // handle): a movable foreground `Area` holds the legend, and egui's hit test
    // routes a *drag* to the movable Area while a *click* still reaches the rows
    // beneath. siplot's rows sense click only, so `hit_test` reports click→row,
    // drag→Area; the user grabs the labels themselves to move it. `constrain_to`
    // keeps it within the axes.
    //
    // The position is held as a *fraction* of the data area (`legend_frac`) and
    // re-applied each frame via `current_pos`, so on resize the legend keeps the
    // same relative spot (moves proportionally) instead of staying at a fixed
    // pixel offset. It defaults to the top-right corner (inset by PAD) and the
    // fraction is re-derived from the Area's actual position after every frame,
    // so a drag sticks and then scales with later resizes.
    const PAD: f32 = 6.0;
    let legend_id = egui::Id::new(plot.inner.backend().plot().id).with("legend_overlay");
    let ctx = ui.ctx().clone();
    let target = match plot.legend_frac {
        Some(f) => egui::pos2(
            area.left() + f.x * area.width(),
            area.top() + f.y * area.height(),
        ),
        None => area.right_top() + egui::vec2(-PAD, PAD),
    };
    egui::Area::new(legend_id)
        .order(egui::Order::Foreground)
        .movable(true)
        .constrain_to(area)
        .current_pos(target)
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

    // Re-derive the fraction from where the legend actually ended up this frame
    // (after any drag and `constrain_to` clamping), so the next frame — and any
    // resize — places it at the same relative spot.
    if area.width() > 0.0
        && area.height() > 0.0
        && let Some(state) = egui::AreaState::load(&ctx, legend_id)
        && let Some(pivot_pos) = state.pivot_pos
    {
        plot.legend_frac = Some(egui::pos2(
            (pivot_pos.x - area.left()) / area.width(),
            (pivot_pos.y - area.top()) / area.height(),
        ));
    }
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

/// Draw the cursor overlay over the data area `area` while the pointer is at
/// `pos`: the `(x, y)` coordinate readout in a small box near the pointer when
/// `readout` is on, and thin guide lines through the pointer spanning the data
/// area when `crosshair` is on. xafsview draws this itself rather than using
/// siplot's `graph_cursor`, which fuses the lines and the readout behind one
/// flag; splitting it lets each get its own toolbar toggle (see [`show`]).
///
/// `data` is the pointer's data-space coordinate (`Transform::pixel_to_data`);
/// the caller passes it pre-computed and only calls this while `pos` is inside
/// `area`. Colours contrast with the data-area background `data_bg`, like the
/// legend, so they read on either the light or dark canvas. The readout box is
/// placed lower-right of the cursor and flips to stay inside `area`, matching
/// siplot's own crosshair readout; values use siplot's `%.7g` number format so
/// the readout matches the axis ticks.
fn draw_cursor_overlay(
    ui: &egui::Ui,
    area: egui::Rect,
    pos: egui::Pos2,
    data: (f64, f64),
    readout: bool,
    crosshair: bool,
    data_bg: egui::Color32,
) {
    let painter = ui.painter();
    let luma = 0.299 * data_bg.r() as f32 + 0.587 * data_bg.g() as f32 + 0.114 * data_bg.b() as f32;
    let dark_canvas = luma <= 128.0;
    let fg = if dark_canvas {
        egui::Color32::from_gray(0xe0)
    } else {
        egui::Color32::from_gray(0x20)
    };

    if crosshair {
        // Semi-transparent so the guide marks the position without burying the
        // curve beneath it (the static grid is fainter still).
        let line = egui::Stroke::new(1.0, fg.gamma_multiply(0.55));
        painter.vline(pos.x, area.y_range(), line);
        painter.hline(area.x_range(), pos.y, line);
    }

    if readout {
        let label = format!(
            "{}, {}",
            siplot::format_value(data.0),
            siplot::format_value(data.1)
        );
        let font = egui::FontId::proportional(11.0);
        let galley = painter.layout_no_wrap(label, font, fg);
        let pad = egui::vec2(4.0, 2.0);
        let size = galley.size() + pad * 2.0;
        // Prefer the lower-right of the cursor; flip to stay inside the data area.
        let mut min = pos + egui::vec2(10.0, 10.0);
        if min.x + size.x > area.right() {
            min.x = pos.x - 10.0 - size.x;
        }
        if min.y + size.y > area.bottom() {
            min.y = pos.y - 10.0 - size.y;
        }
        // A near-opaque box in the opposite tone so the value reads over any curve.
        let box_bg = if dark_canvas {
            egui::Color32::from_black_alpha(0xc8)
        } else {
            egui::Color32::from_white_alpha(0xc8)
        };
        painter.rect_filled(
            egui::Rect::from_min_size(min, size),
            egui::CornerRadius::same(2),
            box_bg,
        );
        painter.galley(min + pad, galley, fg);
    }
}

/// A toolbar toggle button for the crosshair guide lines drawn by
/// [`draw_cursor_overlay`]. siplot's built-in cursor button (a
/// crosshair-with-circle icon) is repurposed as the coordinate-*readout* toggle,
/// so this separate button owns the *lines*. It is styled to match siplot's own
/// icon buttons (same 28×24 footprint, hover/selected fill + stroke, 1.6px icon
/// stroke inset by 5px) and draws a plain crosshair — no circle — to read
/// distinctly from siplot's cursor icon. `on` is [`Plot::crosshair`].
fn crosshair_button(on: &mut bool, ui: &mut egui::Ui) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(28.0, 24.0), egui::Sense::click());
    let response = response.on_hover_text("Toggle crosshair");
    if response.clicked() {
        *on = !*on;
    }
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact_selectable(&response, *on);
        let color = if *on {
            ui.visuals().selection.stroke.color
        } else {
            visuals.fg_stroke.color
        };
        let painter = ui.painter();
        let button_rect = rect.shrink(1.0);
        if *on || response.hovered() || response.has_focus() {
            painter.rect_filled(button_rect, 2.0, visuals.weak_bg_fill);
            painter.rect_stroke(
                button_rect,
                2.0,
                visuals.bg_stroke,
                egui::StrokeKind::Inside,
            );
        }
        let icon = rect.shrink(5.0);
        let c = icon.center();
        let stroke = egui::Stroke::new(1.6, color);
        painter.line_segment(
            [egui::pos2(icon.left(), c.y), egui::pos2(icon.right(), c.y)],
            stroke,
        );
        painter.line_segment(
            [egui::pos2(c.x, icon.top()), egui::pos2(c.x, icon.bottom())],
            stroke,
        );
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
