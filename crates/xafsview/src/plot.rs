//! Shared `rsplot::Plot1D` construction for the GUI.
//!
//! Every plot-bearing tab/window builds its plot through [`new_plot1d`] so they
//! all get the same house data margins. rsplot's `DataMargins` default to zero,
//! which draws curves flush against the axis frame (a sample at the data maximum
//! sits exactly on the top frame line and is visually clipped). A uniform margin
//! on every side — matching matplotlib's `axes.margins` default — keeps the data
//! off the frame, and routing every plot through one constructor means no site
//! can forget it.

use std::collections::{HashMap, VecDeque};

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use rsplot::{CurveSpec, DataMargins, ItemHandle, Plot1D, PlotId, Roi, Symbol};

/// Fraction of the data range left blank on each side of the data extent, so
/// samples at the extremes are lifted just off the axis frame without wasting
/// visible area (matplotlib's `axes.margins` default of 0.05 leaves too much
/// blank space for the spiky XANES/EXAFS curves seen here).
const DATA_MARGIN: f64 = 0.02;

/// Build a [`Plot1D`] with the house data margins applied. All GUI plots are
/// constructed through this so none can forget the margin — rsplot defaults
/// `DataMargins` to zero (data flush against the frame).
pub fn new_plot1d(render_state: &RenderState, id: PlotId) -> Plot1D {
    let mut plot = Plot1D::new(render_state, id);
    plot.plot_mut().set_data_margins(DataMargins {
        x_min: DATA_MARGIN,
        x_max: DATA_MARGIN,
        y_min: DATA_MARGIN,
        y_max: DATA_MARGIN,
    });
    // Disable wheel/scroll zoom on every GUI plot: it fought the macOS trackpad
    // momentum guard repeatedly, so zoom is done through box-drag, the toolbar
    // Home/Zoom buttons, and the context menu instead. (rsplot 0.5.0 `scroll_zoom`.)
    plot.plot_mut().scroll_zoom = false;
    // Coordinate-readout toggle state. rsplot fuses the crosshair guide lines and
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
/// labels read clearly *independent of the egui panel theme* (rsplot otherwise
/// derives the axis colour from `ui.visuals()`, which would render light axes
/// on a light canvas under a dark UI theme). `light = false` is the cohesive
/// dark canvas under the dark UI theme; `true` is the white data area with dark
/// axes under the light theme (Plot Data's "Change BG color" swaps the two).
pub fn set_theme(plot: &mut Plot1D, light: bool) {
    // `(chrome, data, fg, grid)`: the surround behind the axes, the data area
    // itself, the axis/frame/text colour, and the grid lines. `grid` is passed
    // *translucent* on purpose: rsplot applies an override grid colour at full
    // opacity (chrome.rs `with_overrides`), so a solid grey would draw crisp
    // lines that fight the curve — a low-alpha tint composites to a faint grid
    // over either canvas, matching rsplot's own `with_alpha(text, 28)` default.
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
/// that reads against the blue χ curves. rsplot fills the band at half this
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

/// One legend-bearing curve's samples, kept so a graph click can hit-test the
/// plotted data (rsplot exposes no queryable copy of a curve's points). Recorded
/// in lockstep with [`Plot::legend`] and cleared with it.
struct PickCurve {
    /// The curve's legend label (shown in the click readout).
    label: String,
    /// The curve's colour (drawn as the readout's swatch).
    color: egui::Color32,
    /// The curve's `(x, y)` samples, in data coordinates.
    points: Vec<(f64, f64)>,
}

/// The nearest data point to the user's last left-click on a plot's data area,
/// drawn as a pinned bottom readout by [`show`].
struct Pick {
    /// The clicked curve's legend label.
    label: String,
    /// The clicked curve's colour (the readout swatch).
    color: egui::Color32,
    /// The picked vertex's x coordinate.
    x: f64,
    /// The picked vertex's y coordinate.
    y: f64,
}

/// A [`Plot1D`] bundled with the legend entries for the curves currently on it.
///
/// rsplot exposes no per-item colour, and its own `show_legend` draws a
/// visibility-toggle control on every row that the GUI does not want; so each
/// curve's `(label, colour, emphasized)` is tracked here as it is added and
/// [`show`] draws a plain in-axes legend from it. Curves added through
/// [`Plot::add_curve_with_legend`] are recorded; bare `add_curve` curves (no
/// legend) are not. Every other operation falls through to the inner [`Plot1D`]
/// via `Deref`/`DerefMut`, so all existing plot calls are unchanged.
pub struct Plot {
    inner: Plot1D,
    /// Each curve's `(label, colour, emphasized)`. `emphasized` is set for a
    /// curve added via [`add_emphasized_curve`](Self::add_emphasized_curve) so
    /// [`show`] can mark that legend row (mirroring the highlighted curve).
    legend: Vec<(String, egui::Color32, bool)>,
    /// The legend overlay's position as a fraction `(x, y)` of the data-area
    /// rectangle (its RIGHT_TOP pivot), so it keeps the same relative spot when
    /// the plot is resized. `None` until first shown (defaults to top-right);
    /// updated whenever the user drags the legend. See [`show`].
    legend_frac: Option<egui::Pos2>,
    /// Whether the crosshair guide lines are drawn under the pointer. Toggled by
    /// the toolbar's own crosshair button ([`crosshair_button`]); independent of
    /// the coordinate readout, which is gated by rsplot's `graph_cursor` flag.
    /// On by default. See [`show`] and [`draw_cursor_overlay`].
    crosshair: bool,
    /// The draggable FT-window band, a rsplot `VRange` ROI: `Some(index)` once
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
    /// A user drag of either band edge, committed in [`show`] on the frame the
    /// drag ends and taken by [`take_window_drag`].
    window_dragged: Option<AxisWindow>,
    /// True while a band-edge drag is in progress (the ROI has moved under the
    /// pointer but the button is still down). [`set_window`] leaves the ROI to
    /// rsplot's interaction while this holds — re-syncing to the caller's
    /// still-old bounds would snap the band back and fight the drag — and
    /// [`show`] defers the [`window_dragged`] commit until the button is
    /// released, so the reduction recomputes once per drag, not once per frame.
    window_dragging: bool,
    /// The legend entry the user clicked this frame (its label), set in [`show`]
    /// and consumed by [`take_legend_click`]. Lets a caller (Plot Data) drive a
    /// "highlight the clicked curve" selection; plots that never take it are
    /// unaffected.
    legend_clicked: Option<String>,
    /// Every legend-bearing curve's samples, recorded alongside
    /// [`legend`](Self::legend) so a graph *click* can hit-test the plotted data
    /// (rsplot keeps no queryable copy of a curve's samples). Cleared with the
    /// legend; see [`show`]'s click handler and [`last_pick`](Self::last_pick).
    pick_curves: Vec<PickCurve>,
    /// The nearest data point to the user's last left-click on the data area. Set
    /// in [`show`] when a click lands near a curve, cleared when a click lands near
    /// none, and drawn as a pinned bottom readout so the clicked curve's name and
    /// coordinates stay visible.
    last_pick: Option<Pick>,
    /// Set once [`show`] has reset the view for an empty plot, cleared when a
    /// curve is back. The inner plot keeps its last data limits after a
    /// clear (rsplot's content-change refit skips an itemless plot), so a
    /// graph emptied of k-space curves went on showing k ticks under an R
    /// axis label; the one-shot reset puts the empty home view there instead.
    empty_view: bool,
    /// Each legend-bearing curve's inner item handle, in lockstep with
    /// [`legend`](Self::legend) / [`pick_curves`](Self::pick_curves), so a
    /// declared rebuild ([`begin_declare`](Self::begin_declare)) can update or
    /// remove the retained items instead of recreating every curve.
    handles: Vec<ItemHandle>,
    /// Curves recorded since [`begin_declare`](Self::begin_declare) — `Some`
    /// only inside a declare session; [`commit_declare`](Self::commit_declare)
    /// diffs them against the retained curves and resets this to `None`.
    declared: Option<Vec<CurveDecl>>,
    /// Legend fold override: `None` follows the auto rule (folded while more
    /// than [`LEGEND_AUTO_FOLD`] entries), `Some` is the user's explicit choice
    /// from clicking the legend header. See [`show`].
    legend_folded: Option<bool>,
    /// One legend row's height as measured from the rows [`show`] laid out
    /// last frame; sizes the unfolded legend's row-snapped scroll viewport.
    /// `None` until first measured (the first frame uses
    /// [`legend_row_metrics`]'s estimate).
    legend_row_h: Option<f32>,
    /// Handles of the x-markers managed by [`set_x_markers`](Self::set_x_markers),
    /// removed and re-added as a set on each call. Markers added through the
    /// deref'd `Plot1D::add_x_marker` are NOT tracked here — their caller owns
    /// their removal (the XANES tab clears its whole plot per rebuild).
    marker_handles: Vec<ItemHandle>,
    /// Whether the editable X/Y limits row (rsplot's silx `LimitsToolBar`) is
    /// drawn under the toolbar — typed limits for users who control the view
    /// by hand. Toggled by the toolbar's "Limits" button; off by default.
    limits_row: bool,
    /// Hint drawn centered over the data area while the plot has no curves, so
    /// an empty canvas says what fills it instead of looking like missing data.
    /// Set by the owning tab each frame ([`set_empty_hint`](Self::set_empty_hint)).
    empty_hint: Option<String>,
    /// The egui `dark_mode` the canvas palette was last synced to by [`show`].
    /// `None` until first shown, so a saved light preference re-themes the
    /// plot on its first frame.
    theme_synced: Option<bool>,
    /// `Some(light)` once the owner chose the canvas itself
    /// ([`set_light`](Self::set_light), Plot Data's "Change BG color"); such a
    /// plot no longer follows the egui theme.
    theme_override: Option<bool>,
    /// The data-area transform of the last [`show`] frame, so a gesture made
    /// without the pointer ([`gesture`]) maps its data coordinates through
    /// the same pixel space the pointer path uses. `None` until first drawn.
    last_transform: Option<rsplot::Transform>,
}

/// Emphasized-curve styling: the default curve width is 1.0; 3.0 reads as
/// clearly emphasized on either canvas background without depending on colour,
/// and the high z keeps the thick line on top of the other curves.
const EMPH_WIDTH: f32 = 3.0;
const EMPH_Z: f32 = 100.0;

/// How far (egui points) a left-click may land from a curve's nearest vertex
/// and still pick it (see [`show`] and [`gesture`]).
const PICK_THRESHOLD_PX: f32 = 40.0;

/// One curve recorded during a declare session — the desired end state, diffed
/// against the retained curves by legend label on commit (duplicate labels pair
/// in declaration order).
struct CurveDecl {
    label: String,
    color: egui::Color32,
    emphasized: bool,
    x: Vec<f64>,
    y: Vec<f64>,
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
            window_dragging: false,
            legend_clicked: None,
            pick_curves: Vec::new(),
            last_pick: None,
            empty_view: false,
            handles: Vec::new(),
            declared: None,
            legend_folded: None,
            legend_row_h: None,
            marker_handles: Vec::new(),
            limits_row: false,
            empty_hint: None,
            theme_synced: None,
            theme_override: None,
            last_transform: None,
        }
    }

    /// Set the hint [`show`] draws over the data area while no curve is on the
    /// plot. Tabs sharing one `Plot` set their own hint before drawing it.
    pub fn set_empty_hint(&mut self, text: impl Into<String>) {
        self.empty_hint = Some(text.into());
    }

    /// Choose this plot's canvas explicitly ([`set_theme`]) and stop it from
    /// following the egui theme: for an owner with its own light/dark toggle
    /// (Plot Data's "Change BG color"), whose choice would otherwise be
    /// overridden on the next UI theme change.
    pub fn set_light(&mut self, light: bool) {
        set_theme(&mut self.inner, light);
        self.theme_override = Some(light);
    }

    /// Draw x-position markers as a managed set: remove the markers from the
    /// previous call, then add one at each of `xs`. Rebuild-style callers (Plot
    /// Data's caught peaks) re-derive the full set every pass, and neither
    /// `clear_curves` (markers are not curve-like) nor the declare diff (curves
    /// only) removes the old ones — without this, each rebuild stacks a new
    /// copy of every marker over the last.
    pub fn set_x_markers(&mut self, xs: &[f64], color: egui::Color32) {
        for h in self.marker_handles.drain(..) {
            self.inner.remove_marker(h);
        }
        for &x in xs {
            let h = self.inner.add_x_marker(x, color);
            self.marker_handles.push(h);
        }
    }

    /// Add a curve and record its `(label, colour)` for the in-axes legend.
    /// Shadows [`Plot1D::add_curve_with_legend`], so existing call sites record
    /// their legend entry without change. Inside a declare session
    /// ([`begin_declare`](Self::begin_declare)) the curve is recorded for the
    /// commit diff instead of added immediately.
    pub fn add_curve_with_legend(
        &mut self,
        x: &[f64],
        y: &[f64],
        color: egui::Color32,
        legend: impl Into<String>,
    ) {
        self.add_entry(x, y, color, legend.into(), false);
    }

    /// Record a curve's samples for click hit-testing. Kept in lockstep with the
    /// [`legend`](Self::legend) push so [`show`]'s click handler can find the
    /// nearest point without rsplot exposing the drawn data.
    fn record_pick_curve(&mut self, label: &str, color: egui::Color32, x: &[f64], y: &[f64]) {
        let points: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
        self.pick_curves.push(PickCurve {
            label: label.to_owned(),
            color,
            points,
        });
    }

    /// Like [`add_curve_with_legend`](Self::add_curve_with_legend) but draws the
    /// curve emphasized — a thicker line brought to the front — so a selected
    /// curve stands out without dimming the others. Used by Plot Data's
    /// legend-click highlight. The legend entry is recorded as *emphasized* so
    /// [`show`] marks that row too, making the highlighted curve identifiable in
    /// the legend as well as on the canvas.
    pub fn add_emphasized_curve(
        &mut self,
        x: &[f64],
        y: &[f64],
        color: egui::Color32,
        legend: impl Into<String>,
    ) {
        self.add_entry(x, y, color, legend.into(), true);
    }

    /// The shared body of the two add methods: record a declared curve inside a
    /// declare session, or add + bookkeep immediately outside one.
    fn add_entry(&mut self, x: &[f64], y: &[f64], color: egui::Color32, label: String, emph: bool) {
        if let Some(declared) = &mut self.declared {
            declared.push(CurveDecl {
                label,
                color,
                emphasized: emph,
                x: x.to_vec(),
                y: y.to_vec(),
            });
            return;
        }
        self.legend.push((label.clone(), color, emph));
        self.record_pick_curve(&label, color, x, y);
        let handle = Self::push_curve(&mut self.inner, x, y, color, &label, emph);
        self.handles.push(handle);
    }

    /// Add one curve to the inner plot with the wrapper's styling rules (the
    /// emphasized width/z convention of
    /// [`add_emphasized_curve`](Self::add_emphasized_curve)).
    fn push_curve(
        inner: &mut Plot1D,
        x: &[f64],
        y: &[f64],
        color: egui::Color32,
        label: &str,
        emph: bool,
    ) -> ItemHandle {
        let mut spec = CurveSpec::new(x, y, color);
        if emph {
            spec.line_width = EMPH_WIDTH;
        }
        let handle = inner.add_curve_spec(spec);
        inner.set_item_legend(handle, label);
        if emph {
            inner.set_item_z(handle, EMPH_Z);
        }
        handle
    }

    /// Start a declared rebuild: until [`commit_declare`](Self::commit_declare),
    /// the add methods record the desired curves instead of adding them, and
    /// the commit diffs that declaration against the curves already on the
    /// plot — unchanged curves are kept as-is, changed ones update their
    /// existing item in place, and only genuinely new/removed curves are
    /// added/removed. A clear-and-re-add rebuild recreates every curve's GPU
    /// buffers on every call, which dominates interaction cost once tens of
    /// groups are loaded; the diff makes that cost proportional to what
    /// actually changed.
    pub fn begin_declare(&mut self) {
        self.declared = Some(Vec::new());
    }

    /// Commit the declare session started by
    /// [`begin_declare`](Self::begin_declare). Returns whether the plot's
    /// content actually changed (any curve added, removed, or updated) — a
    /// no-op commit leaves the view alone, so callers adjusting limits after a
    /// rebuild can preserve the user's zoom the same way the diff itself does.
    pub fn commit_declare(&mut self) -> bool {
        let Some(decls) = self.declared.take() else {
            return false;
        };
        // Pair each declared curve with a retained one by label, in order
        // (duplicate labels pair first-to-first).
        let mut by_label: HashMap<&str, VecDeque<usize>> = HashMap::new();
        for (j, (label, _, _)) in self.legend.iter().enumerate() {
            by_label.entry(label.as_str()).or_default().push_back(j);
        }
        let matched: Vec<Option<usize>> = decls
            .iter()
            .map(|d| {
                by_label
                    .get_mut(d.label.as_str())
                    .and_then(|q| q.pop_front())
            })
            .collect();
        let mut keep = vec![false; self.handles.len()];
        for &j in matched.iter().flatten() {
            keep[j] = true;
        }
        let mut changed = false;
        for (j, kept) in keep.iter().enumerate() {
            if !kept {
                self.inner.remove_curve(self.handles[j]);
                changed = true;
            }
        }
        let old_legend = std::mem::take(&mut self.legend);
        let mut old_picks = std::mem::take(&mut self.pick_curves);
        let old_handles = std::mem::take(&mut self.handles);
        for (i, (d, m)) in decls.into_iter().zip(matched).enumerate() {
            let handle = match m {
                Some(j) => {
                    let (_, old_color, old_emph) = old_legend[j];
                    let same_style = old_color == d.color && old_emph == d.emphasized;
                    let old_points = &old_picks[j].points;
                    let same_data = old_points.len() == d.x.len()
                        && old_points
                            .iter()
                            .zip(d.x.iter().zip(&d.y))
                            .all(|(p, (&x, &y))| p.0 == x && p.1 == y);
                    let handle = old_handles[j];
                    if !(same_style && same_data) {
                        let mut spec = CurveSpec::new(&d.x, &d.y, d.color);
                        if d.emphasized {
                            spec.line_width = EMPH_WIDTH;
                        }
                        self.inner.update_curve_spec(handle, spec);
                        changed = true;
                    }
                    if same_data {
                        // Move the recorded hit-test samples over instead of
                        // copying them again.
                        let points = std::mem::take(&mut old_picks[j].points);
                        self.pick_curves.push(PickCurve {
                            label: d.label.clone(),
                            color: d.color,
                            points,
                        });
                    } else {
                        self.record_pick_curve(&d.label, d.color, &d.x, &d.y);
                    }
                    handle
                }
                None => {
                    let handle = Self::push_curve(
                        &mut self.inner,
                        &d.x,
                        &d.y,
                        d.color,
                        &d.label,
                        d.emphasized,
                    );
                    self.record_pick_curve(&d.label, d.color, &d.x, &d.y);
                    changed = true;
                    handle
                }
            };
            // Keep paint order = declaration order even for retained items (a
            // kept curve would otherwise stay below items added after it). The
            // fractional step keeps every normal curve well under EMPH_Z.
            let base = if d.emphasized { EMPH_Z } else { 0.0 };
            self.inner.set_item_z(handle, base + i as f32 * 0.01);
            self.legend.push((d.label, d.color, d.emphasized));
            self.handles.push(handle);
        }
        self.last_pick = None;
        changed
    }

    /// The last clicked data point (the pinned readout) as `(curve label, x)`,
    /// if a click has landed on a curve. `x` is in the data space of the
    /// current graph — the caller knows whether that axis is energy, k, or R —
    /// and the label is the clicked curve's legend entry, so a caller that
    /// overlays several groups can tell whose curve was picked.
    pub fn last_pick(&self) -> Option<(&str, f64)> {
        self.last_pick.as_ref().map(|p| (p.label.as_str(), p.x))
    }

    /// Y bounds `(min, max)` over every curve's samples whose x lies within
    /// `[xmin, xmax]` (bounds accepted in either order; non-finite y skipped).
    /// `None` when no sample falls inside the window.
    pub fn y_bounds_in_x_window(&self, xmin: f64, xmax: f64) -> Option<(f64, f64)> {
        let (lo, hi) = if xmin <= xmax {
            (xmin, xmax)
        } else {
            (xmax, xmin)
        };
        let mut bounds: Option<(f64, f64)> = None;
        for c in &self.pick_curves {
            for &(x, y) in &c.points {
                if x >= lo && x <= hi && y.is_finite() {
                    bounds = Some(match bounds {
                        None => (y, y),
                        Some((bl, bh)) => (bl.min(y), bh.max(y)),
                    });
                }
            }
        }
        bounds
    }

    /// Clear all plot items and the recorded legend together. Shadows
    /// [`Plot1D::clear`], so a rebuild that calls `clear()` also empties the
    /// legend (otherwise stale entries would accumulate each frame).
    pub fn clear(&mut self) {
        self.inner.clear();
        self.legend.clear();
        self.pick_curves.clear();
        self.handles.clear();
        self.declared = None;
        self.last_pick = None;
        // `Plot1D::clear` removed the marker items; drop the stale handles.
        // (`clear_curves` below spares markers, so it keeps the tracking.)
        self.marker_handles.clear();
    }

    /// Clear curve items and the recorded legend together. Shadows
    /// [`Plot1D::clear_curves`].
    pub fn clear_curves(&mut self) {
        self.inner.clear_curves();
        self.legend.clear();
        self.pick_curves.clear();
        self.handles.clear();
        self.declared = None;
        self.last_pick = None;
    }
}

/// Draw `plot`'s standard toolbar plus the house extras on one row, then leave
/// the plot itself to a following [`show`]. Every plot-bearing tab/window draws
/// its toolbar through this (instead of `plot.show_toolbar` directly) so they all
/// expose the same controls beyond rsplot's built-ins:
///
/// - a [`crosshair_button`] that toggles the crosshair guide lines (rsplot's own
///   cursor button is repurposed as the coordinate-readout toggle, see [`show`]);
/// - a "Symbol" menu that toggles data-point markers (size + shape) on every
///   curve, which the bare toolbar omits. rsplot's ready-made `symbol_tool_button`
///   is a `Plot2D` (image) method, so for the curve `Plot1D` the same control is
///   built here on top of `PlotWidget::set_all_symbols` / `set_all_symbol_sizes`.
pub fn toolbar(plot: &mut Plot, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        plot.inner.show_toolbar(ui);
        crosshair_button(&mut plot.crosshair, ui);
        symbol_menu(&mut plot.inner, ui);
        if ui
            .selectable_label(plot.limits_row, "Limits")
            .on_hover_text("Edit the axis limits numerically")
            .clicked()
        {
            plot.limits_row = !plot.limits_row;
        }
    });
    if plot.limits_row {
        plot.inner.show_limits_toolbar(ui);
    }
}

/// Request the draggable FT-window band at `window` for this frame, drawn as a
/// shaded vertical band on the active graph's x-axis. Call once per frame
/// *before* [`show`] for every frame the band should appear; a frame with no
/// call removes it (so the band never leaks onto a graph — or a tab sharing this
/// plot — that has no window). The user drags either edge; [`show`] then reports
/// the new bounds through [`take_window_drag`].
///
/// The ROI is only re-synced when `window` differs from the value last pushed
/// *and* no edge-drag is in progress — a typed parameter, a graph switch, an
/// Autobk Start — so an in-progress edge drag is left to rsplot's own
/// interaction and not overwritten mid-drag (the caller's bounds stay at the
/// pre-drag value until the drag commits on release).
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
            if !plot.window_dragging && plot.window_target != Some(window) {
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

/// Take the FT-window drag committed during the last [`show`], if any: the new
/// `[min, max]` bounds the user dragged the band to. Returns `Some` exactly once
/// per drag — on the frame the edge-drag ends (button released) — so the caller
/// updates its parameters and recomputes once, not on every mid-drag frame. The
/// band still follows the cursor live during the drag ([`show`]); only the
/// (expensive) recompute is deferred to release.
pub fn take_window_drag(plot: &mut Plot) -> Option<AxisWindow> {
    plot.window_dragged.take()
}

/// Take the legend entry (its label) the user clicked since the last call, if
/// any. Plot Data uses it to toggle which curve is highlighted; callers that
/// never take it leave the click inert.
pub fn take_legend_click(plot: &mut Plot) -> Option<String> {
    plot.legend_clicked.take()
}

/// A pointer gesture on a plot, named so it can be made without the pointer
/// (the MCP `graph` tool). Coordinates are in the plot's data space.
#[derive(Debug, Clone, PartialEq)]
pub enum Gesture {
    /// The toolbar's Zoom in: shrink the view about its centre.
    ZoomIn,
    /// The toolbar's Zoom out: grow the view about its centre.
    ZoomOut,
    /// A left-drag rectangle in Zoom mode, released: zoom to the box.
    ZoomBox { x0: f64, y0: f64, x1: f64, y1: f64 },
    /// A pan drag that moves the view by `(dx, dy)` data units.
    Pan { dx: f64, dy: f64 },
    /// The toolbar's Zoom back: the previous view, or the data bounds when
    /// there is none.
    ZoomBack,
    /// The toolbar's Reset zoom: the data bounds.
    Reset,
    /// A left-click on the data area at `(x, y)`: pins the nearest curve's
    /// readout, or clears it when no curve is within reach.
    Click { x: f64, y: f64 },
    /// An FT-window band edge drag released with the band at `[min, max]`.
    DragWindow { min: f64, max: f64 },
    /// A click on the legend row `label`.
    LegendClick { label: String },
    /// No gesture: report the view only.
    View,
}

/// Make gesture `g` on `plot` through the code its pointer version runs —
/// rsplot's toolbar actions and gesture maths for the view, [`show`]'s own
/// click, band-drag and legend-click handling for the rest — then report the
/// view: the axis limits, the pinned pick and the band. The owner consumes the
/// result on its next frame exactly as after a real gesture
/// ([`take_window_drag`], [`take_legend_click`], [`Plot::last_pick`]). The
/// plot must have been drawn once so its data↔pixel transform is known.
pub fn gesture(plot: &mut Plot, g: Gesture) -> Result<String, String> {
    let t = plot
        .last_transform
        .ok_or("the plot has not been drawn yet; open its window first")?;
    let finite = |v: f64, name: &str| {
        v.is_finite()
            .then_some(v)
            .ok_or_else(|| format!("`{name}` must be finite"))
    };
    match g {
        Gesture::ZoomIn => rsplot::actions::control::zoom_in(&mut plot.inner),
        Gesture::ZoomOut => rsplot::actions::control::zoom_out(&mut plot.inner),
        Gesture::ZoomBack => {
            rsplot::actions::control::zoom_back(&mut plot.inner);
        }
        Gesture::Reset => plot.inner.reset_zoom(),
        Gesture::ZoomBox { x0, y0, x1, y1 } => {
            for (v, n) in [(x0, "x"), (y0, "y"), (x1, "x2"), (y1, "y2")] {
                finite(v, n)?;
            }
            // rsplot takes a box-zoom release only when the dragged rectangle
            // covers 5 px² on screen (silx SURFACE_THRESHOLD).
            let (a, b) = (t.data_to_pixel(x0, y0), t.data_to_pixel(x1, y1));
            if ((a.x - b.x) * (a.y - b.y)).abs() < 5.0 {
                return Err(
                    "the zoom box covers less than 5 px² on screen; a drag that small is ignored"
                        .to_owned(),
                );
            }
            let (current, y2, keep_aspect, zoom_x, zoom_y) = {
                let core = plot.inner.plot_mut();
                core.push_limits();
                (
                    core.limits,
                    core.y2,
                    core.keep_aspect,
                    core.zoom_x_enabled(),
                    core.zoom_y_enabled(),
                )
            };
            let boxed = rsplot::widget::interaction::box_zoom(x0, y0, x1, y1);
            let (nx0, nx1, ny0, ny1) = if keep_aspect {
                boxed
            } else {
                rsplot::widget::interaction::constrain_zoom_axes(boxed, current, zoom_x, zoom_y)
            };
            plot.inner.set_limits(nx0, nx1, ny0, ny1, y2);
        }
        Gesture::Pan { dx, dy } => {
            finite(dx, "dx")?;
            finite(dy, "dy")?;
            // The drag that carries the view by (dx, dy): the content moves the
            // other way, so the pixel delta runs from the shifted point back
            // to the original one.
            let (cx, cy) = t.pixel_to_data(t.area.center());
            let delta = t.data_to_pixel(cx, cy) - t.data_to_pixel(cx + dx, cy + dy);
            let (current, y2, x_scale, y_scale) = {
                let core = plot.inner.plot_mut();
                core.push_limits();
                (core.limits, core.y2, core.x_scale, core.y_scale)
            };
            let ((nx0, nx1, ny0, ny1), next_y2) = rsplot::widget::interaction::pan_view(
                (current, y2),
                t.area,
                delta,
                x_scale,
                y_scale,
            );
            plot.inner.set_limits(nx0, nx1, ny0, ny1, next_y2);
        }
        Gesture::Click { x, y } => {
            finite(x, "x")?;
            finite(y, "y")?;
            let pos = t.data_to_pixel(x, y);
            if !t.area.contains(pos) {
                return Err(format!("({x}, {y}) lies outside the data area"));
            }
            plot.last_pick = nearest_pick(&plot.pick_curves, &t, pos, PICK_THRESHOLD_PX);
        }
        Gesture::DragWindow { min, max } => {
            finite(min, "min")?;
            finite(max, "max")?;
            let idx = plot.window_roi.ok_or(
                "no FT-window band on this graph (turn on Show k/R range on a k or R graph)",
            )?;
            // What rsplot's edge drag does to the band, then what `show` does
            // when the button is released.
            if let Some(managed) = plot.inner.rois_mut().get_mut(idx) {
                managed.roi = Roi::VRange { x: (min, max) };
            }
            let w = AxisWindow { min, max };
            plot.window_target = Some(w);
            plot.window_dragged = Some(w);
            plot.window_dragging = false;
        }
        Gesture::LegendClick { label } => {
            if !plot.legend.iter().any(|(l, _, _)| *l == label) {
                let names: Vec<&str> = plot.legend.iter().map(|(l, _, _)| l.as_str()).collect();
                return Err(format!(
                    "no legend entry {label:?}; entries: {}",
                    names.join(", ")
                ));
            }
            plot.legend_clicked = Some(label);
        }
        Gesture::View => {}
    }
    Ok(view_report(plot))
}

/// The view as one line: the axis limits, the pinned pick, the FT-window band
/// and the curve count.
fn view_report(plot: &Plot) -> String {
    use std::fmt::Write;
    let (xmin, xmax) = plot.inner.x_limits();
    let (ymin, ymax) = plot
        .inner
        .y_limits(rsplot::YAxis::Left)
        .unwrap_or((f64::NAN, f64::NAN));
    let mut s = format!("x = [{xmin:.6}, {xmax:.6}], y = [{ymin:.6}, {ymax:.6}]");
    match &plot.last_pick {
        Some(p) => {
            let _ = write!(s, "; pick: {} at ({:.6}, {:.6})", p.label, p.x, p.y);
        }
        None => s.push_str("; pick: none"),
    }
    if let Some(w) = plot.window_target {
        let _ = write!(s, "; window: [{:.4}, {:.4}]", w.min, w.max);
    }
    let _ = write!(s, "; {} curve(s)", plot.legend.len());
    s
}

/// Render `plot` as one unit: its toolbar on top, then the plot canvas filling
/// the full width with the legend *overlaid* in the top-right corner of the data
/// area (matplotlib-style), mapping each curve's color/symbol to the name set
/// with `set_item_legend`. Every plot-bearing tab/window draws through this
/// (instead of `toolbar` + `plot.show` by hand) so they all get the same toolbar
/// *and* a visible legend — rsplot draws no in-axes legend, so without this call
/// the curve names never appear. The legend floats over the canvas (an egui
/// `Area`) instead of taking a separate column, so it no longer steals width.
pub fn show(plot: &mut Plot, ui: &mut egui::Ui) {
    // Follow the egui theme: Settings → "Toggle dark / light" (and a saved light
    // preference) only switch the egui visuals, so a plot themed once at
    // construction kept light-grey tick labels and titles on the now-white
    // panel. Re-apply the house palette whenever the UI theme changes, unless
    // the owner picked the canvas itself ([`Plot::set_light`]).
    let dark = ui.visuals().dark_mode;
    if plot.theme_override.is_none() && plot.theme_synced != Some(dark) {
        set_theme(&mut plot.inner, !dark);
        plot.theme_synced = Some(dark);
    }

    toolbar(plot, ui);

    // An emptied plot shows the empty home view, not the last curves' limits
    // (see `empty_view`). Once per emptying, so a later manual zoom on an
    // empty canvas is left alone; a plot that is cleared and refilled within
    // one frame never passes through here empty.
    if plot.legend.is_empty() {
        if !plot.empty_view {
            plot.inner.reset_zoom_to_data();
            plot.empty_view = true;
        }
    } else {
        plot.empty_view = false;
    }

    // rsplot fuses the crosshair guide lines and the coordinate readout behind a
    // single `graph_cursor` flag, drawing both or neither. The GUI splits them so
    // each has its own toolbar toggle: `graph_cursor` (the toolbar's "Show cursor
    // coordinates" button) gates the *readout*, and `Plot::crosshair` (our own
    // toolbar button) gates the *lines*. xafsview draws the overlay itself: read
    // the readout toggle off the flag, suppress rsplot's own (fused) cursor for
    // this frame's render, then restore the flag so the toolbar button keeps its
    // state. `set_graph_cursor` only flips a bool, so this is cheap.
    let readout_on = plot.inner.graph_cursor();
    plot.inner.set_graph_cursor(false);
    let resp = plot.inner.show(ui);
    plot.inner.set_graph_cursor(readout_on);

    // FT-window band. rsplot drives the VRange edge drag inside the inner `show`
    // above, mutating the ROI in place; detect that by comparing the post-show
    // bounds against the value we last pushed (`window_target`). Then keep the
    // band only if it was requested this frame, else remove it so it never leaks
    // onto a windowless graph or another tab sharing this plot. Drain rsplot's
    // event buffer (RoiAdded/RoiChanged accumulate per frame) since nothing here
    // consumes it — we read state, not events.
    plot.inner.drain_events();
    // A user edge-drag moves the ROI away from the bounds we last pushed. Track
    // the live position each frame so the band follows the cursor, but do NOT
    // commit yet: recomputing the reduction on every mid-drag frame is what made
    // the FT window sluggish. rsplot exposes no ROI-drag-finished event, so the
    // drag-end signal is egui's pointer release.
    if let Some(idx) = plot.window_roi
        && let Some(managed) = plot.inner.rois().get(idx)
        && let Roi::VRange { x: (min, max) } = managed.roi
        && let Some(t) = plot.window_target
        && ((min - t.min).abs() > 1e-9 || (max - t.max).abs() > 1e-9)
    {
        plot.window_target = Some(AxisWindow { min, max });
        plot.window_dragging = true;
    }
    // Commit exactly once, on the frame the drag ends (primary button released).
    if plot.window_dragging && ui.input(|i| i.pointer.primary_released()) {
        plot.window_dragged = plot.window_target;
        plot.window_dragging = false;
    }
    if !plot.window_requested
        && let Some(idx) = plot.window_roi.take()
    {
        plot.inner.remove_roi(idx);
        plot.window_target = None;
        // Band gone (e.g. graph switched mid-drag): drop any in-progress drag so
        // a later release cannot commit a stale window onto another graph.
        plot.window_dragging = false;
    }
    plot.window_requested = false;

    let crosshair_on = plot.crosshair;
    // `PlotResponse::transform` carries the data-area rectangle (screen points)
    // that we anchor the legend overlay to and map the pointer through.
    let transform = resp.transform;
    plot.last_transform = Some(transform);
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

    // Left-click on the data area pins the nearest curve's name and (x, y) as a
    // bottom readout (a *separate* area from the pointer-following cursor overlay,
    // per the request). The hit-test runs in pixel space via rsplot's
    // `nearest_point`, so "nearest" matches what the eye sees regardless of how
    // the two axes' data scales differ. A click that lands near no curve clears it.
    if resp.response.clicked()
        && let Some(pos) = resp.response.interact_pointer_pos()
        && area.contains(pos)
    {
        plot.last_pick = nearest_pick(&plot.pick_curves, &transform, pos, PICK_THRESHOLD_PX);
    }
    if let Some(pick) = &plot.last_pick {
        draw_pick_readout(ui, area, pick, data_bg);
    }

    // No labelled curve — draw the empty-state hint (what fills this plot)
    // instead of leaving a bare grid that reads as missing data, and skip the
    // legend overlay. Text colour contrasts with the data-area background, like
    // the readout/legend text.
    if plot.legend.is_empty() {
        if let Some(hint) = &plot.empty_hint {
            let luma = 0.299 * data_bg.r() as f32
                + 0.587 * data_bg.g() as f32
                + 0.114 * data_bg.b() as f32;
            let color = if luma > 128.0 {
                egui::Color32::from_gray(0x60)
            } else {
                egui::Color32::from_gray(0x90)
            };
            // Wrapped to the data-area width and clipped to it: on a narrow
            // plot the sentence folds into lines instead of spilling over the
            // panels beside the graph (the clip covers the degenerate case of
            // an area smaller than even the wrapped text).
            let painter = ui.painter().with_clip_rect(area);
            let galley = painter.layout(
                hint.clone(),
                egui::FontId::proportional(15.0),
                color,
                (area.width() - 16.0).max(0.0),
            );
            painter.galley(area.center() - galley.size() / 2.0, galley, color);
        }
        return;
    }

    // Float the legend over the canvas, draggable from anywhere on it (no
    // handle): a movable foreground `Area` holds the legend, and egui's hit test
    // routes a *drag* to the movable Area while a *click* still reaches the rows
    // beneath. rsplot's rows sense click only, so `hit_test` reports click→row,
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
    // With many curves (a batch of scans) a full legend would blanket the
    // canvas, so past LEGEND_AUTO_FOLD entries it starts folded to a one-line
    // "N curves" header; clicking the header toggles, and the user's choice
    // sticks (`legend_folded`). A long unfolded legend is capped to a fraction
    // of the plot height and scrolls, instead of spanning the full canvas.
    const LEGEND_AUTO_FOLD: usize = 15;
    let folded = plot
        .legend_folded
        .unwrap_or(plot.legend.len() > LEGEND_AUTO_FOLD);
    let foldable = plot.legend.len() > LEGEND_AUTO_FOLD;
    let mut legend_click: Option<String> = None;
    let mut fold_click = false;
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
                        // Wide enough for a ~28-char curve name on one line
                        // (batch scan names differ only in their suffix, so
                        // truncation must be the fallback, not the norm). The
                        // 90px readability floor still loses to a plot too
                        // narrow to hold it: a column wider than the constrain
                        // rect cannot fit, and its overflow clips mid-label.
                        let cap = (area.width() * 0.45)
                            .clamp(90.0, 280.0)
                            .min((area.width() - 4.0 * PAD).max(0.0));
                        ui.set_max_width(cap);
                        if foldable {
                            fold_click = draw_legend_header(ui, plot.legend.len(), folded, data_bg);
                        }
                        if folded && foldable {
                            return;
                        }
                        // Fix the column width before entering the ScrollArea:
                        // the rows' truncating labels have no intrinsic width,
                        // and auto-shrink would collapse them all to "…". Only
                        // the unfolded legend gets the width — the folded
                        // header keeps its natural (narrow) size so the
                        // invisible `Area` doesn't blanket the plot's corner.
                        ui.set_width(legend_intrinsic_width(ui, &plot.legend).min(cap));
                        // Disable drag-to-scroll: the ScrollArea sits above the Area
                        // and would otherwise sense the drag and scroll instead of
                        // letting the movable Area move. Scrollbar + wheel still scroll
                        // an overflowing legend (only the `drag` source is turned off).
                        // Snap the viewport down to whole rows: an arbitrary
                        // height cap slices the bottom row in half, which reads
                        // as a broken label rather than as an overflow.
                        // The row height comes from the rows as they were
                        // actually laid out last frame (`legend_row_h`); the
                        // metrics estimate only seeds the first frame. An
                        // estimate alone under-counted the pitch (egui's
                        // horizontal-row minimum height, frame margins) and
                        // capped a three-curve legend at two visible rows.
                        let (est_row_h, row_gap) = legend_row_metrics(ui);
                        let row_h = plot.legend_row_h.unwrap_or(est_row_h);
                        let cap_h = (area.height() * 0.4).max(80.0);
                        let rows = (((cap_h + row_gap) / (row_h + row_gap)).floor()).max(1.0);
                        // Pinned from both sides (`fixed_viewport_scroll`):
                        // the floating `Area` feeds its own last size back as
                        // the available height, and a viewport taking that
                        // sat at the 64 pt min_scrolled floor for good — two
                        // rows of three.
                        let viewport_h = rows * row_h + (rows - 1.0) * row_gap;
                        let out = crate::window::fixed_viewport_scroll(viewport_h)
                            .scroll_source(egui::scroll_area::ScrollSource {
                                drag: egui::scroll_area::DragScroll::Never,
                                ..Default::default()
                            })
                            .show(ui, |ui| {
                                legend_click = draw_legend(ui, &plot.legend, data_bg);
                            });
                        // Measure: N rows separated by N−1 gaps fill the content.
                        let n = plot.legend.len() as f32;
                        let measured = (out.content_size.y - (n - 1.0) * row_gap) / n;
                        if measured.is_finite()
                            && measured > 0.0
                            && plot.legend_row_h != Some(measured)
                        {
                            plot.legend_row_h = Some(measured);
                            // Re-cap with the real pitch right away instead of
                            // showing a sliced bottom row for one frame.
                            ui.ctx().request_repaint();
                        }
                    });
                });
        });
    if fold_click {
        plot.legend_folded = Some(!folded);
    }

    // A legend row was clicked this frame — record it for `take_legend_click`.
    if legend_click.is_some() {
        plot.legend_clicked = legend_click;
    }

    // Re-derive the fraction from where the legend actually ended up this frame
    // (after any drag and `constrain_to` clamping), so the next frame — and any
    // resize — places it at the same relative spot.
    if area.width() > 0.0
        && area.height() > 0.0
        && let Some(state) = egui::AreaState::load(&ctx, legend_id)
        && let Some(pivot_pos) = state.pivot_pos
    {
        // Clamped to [0, 1]: on a frame where the legend cannot fit the area
        // (a mid-resize sliver), the derived fraction would land outside and
        // keep placing the legend off-plot after the window grows back.
        plot.legend_frac = Some(egui::pos2(
            ((pivot_pos.x - area.left()) / area.width()).clamp(0.0, 1.0),
            ((pivot_pos.y - area.top()) / area.height()).clamp(0.0, 1.0),
        ));
    }
}

/// The fold header shown above a long legend: a painted fold triangle plus
/// "N curves" in the legend's contrast colour. Senses a click only (like the
/// rows), so a drag anywhere on it still moves the enclosing `Area`. Returns
/// `true` when clicked.
fn draw_legend_header(
    ui: &mut egui::Ui,
    count: usize,
    folded: bool,
    data_bg: egui::Color32,
) -> bool {
    let luma = 0.299 * data_bg.r() as f32 + 0.587 * data_bg.g() as f32 + 0.114 * data_bg.b() as f32;
    let text = if luma > 128.0 {
        egui::Color32::from_gray(0x20)
    } else {
        egui::Color32::from_gray(0xe0)
    };
    let row = ui
        .horizontal(|ui| {
            // The fold arrow is painted (a triangle), not a text glyph, so its
            // look does not depend on which fallback font supplies ▸/▾.
            let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            let r = rect.shrink(1.5);
            let points = if folded {
                vec![r.left_top(), r.right_center(), r.left_bottom()]
            } else {
                vec![r.left_top(), r.right_top(), r.center_bottom()]
            };
            ui.painter().add(egui::Shape::convex_polygon(
                points,
                text,
                egui::Stroke::NONE,
            ));
            ui.label(
                egui::RichText::new(format!("{count} curves"))
                    .color(text)
                    .strong(),
            );
        })
        .response
        .interact(egui::Sense::click())
        .on_hover_text(if folded {
            "Show the legend"
        } else {
            "Fold the legend"
        });
    if row.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    row.clicked()
}

/// Width of the legend rows' line-swatch column (shared by the width estimate
/// and the row renderer).
const SWATCH_W: f32 = 22.0;
/// Height of a legend row's colour swatch (see [`draw_legend`]).
const SWATCH_H: f32 = 12.0;

/// First-frame estimate of one legend row's height, and the gap between rows,
/// as [`draw_legend`] lays them out: a row is the swatch/label line plus the
/// row frame's 1px top and bottom margins, and rows are separated by the 2px
/// `item_spacing` the legend sets. [`show`] snaps the unfolded legend's scroll
/// viewport to a whole number of rows so the cap never slices the bottom row
/// in half; from the second frame on it uses the measured row height
/// (`Plot::legend_row_h`), since the estimate ignores egui's horizontal-row
/// minimum height. The gap is exact.
fn legend_row_metrics(ui: &egui::Ui) -> (f32, f32) {
    let text_h = ui.text_style_height(&egui::TextStyle::Body);
    (text_h.max(SWATCH_H) + 2.0, 2.0)
}

/// The width the legend's widest row wants: swatch + spacing + the widest
/// label laid out un-wrapped, plus row margins and scrollbar room. The rows'
/// labels truncate (see [`draw_legend`]) and so have no intrinsic width of
/// their own — egui's auto-shrink would collapse them to "…" — so [`show`]
/// sets the legend column to this width (capped) before entering the
/// `ScrollArea`.
fn legend_intrinsic_width(ui: &egui::Ui, entries: &[(String, egui::Color32, bool)]) -> f32 {
    let body = egui::TextStyle::Body.resolve(ui.style());
    let painter = ui.painter();
    let widest = entries
        .iter()
        .map(|(l, _, _)| {
            painter
                .layout_no_wrap(l.clone(), body.clone(), egui::Color32::WHITE)
                .rect
                .width()
        })
        .fold(0.0f32, f32::max);
    // swatch + spacing(4) + frame margins(8) + bold slack for the emphasized
    // row + scrollbar width when the list overflows.
    SWATCH_W + widest + 36.0
}

/// Characters that end a token inside a curve label; an affix cut is snapped to
/// one of these (or to a letter/digit transition) so the shared stem and the
/// distinguishing core split on a token boundary: "Insitu_" + "064" + ".dat",
/// not "Insitu_06" + "4" + ".dat".
const TOKEN_SEPARATORS: &[char] = &['_', '-', '.', ' '];

/// Is byte index `i` of `s` a token boundary: the start or end, or a point
/// where a separator or a letter/digit transition sits?
fn is_token_boundary(s: &str, i: usize) -> bool {
    if i == 0 || i == s.len() {
        return true;
    }
    if !s.is_char_boundary(i) {
        return false;
    }
    let prev = s[..i].chars().next_back().unwrap_or(' ');
    let next = s[i..].chars().next().unwrap_or(' ');
    TOKEN_SEPARATORS.contains(&prev)
        || TOKEN_SEPARATORS.contains(&next)
        || prev.is_ascii_digit() != next.is_ascii_digit()
}

/// The byte lengths of the prefix and suffix that every label in `labels`
/// shares, each snapped to a token boundary (the prefix back, the suffix
/// forward). Batch scan names are a long shared stem around a short
/// distinguishing core, and this is what separates the two. `(0, 0)` with fewer
/// than two labels; the two affixes never overlap within any label.
fn common_affixes(labels: &[&str]) -> (usize, usize) {
    let [first, rest @ ..] = labels else {
        return (0, 0);
    };
    if rest.is_empty() {
        return (0, 0);
    }
    let mut prefix = first.len();
    let mut suffix = first.len();
    for l in rest {
        prefix = first
            .bytes()
            .zip(l.bytes())
            .take(prefix)
            .take_while(|(a, b)| a == b)
            .count();
        suffix = first
            .bytes()
            .rev()
            .zip(l.bytes().rev())
            .take(suffix)
            .take_while(|(a, b)| a == b)
            .count();
    }
    // Snap the prefix back to a token boundary. Identical bytes have identical
    // char boundaries, so a boundary in `first` is one in every label.
    while prefix > 0 && !is_token_boundary(first, prefix) {
        prefix -= 1;
    }
    // The suffix may not eat into any label's prefix; then snap it forward.
    let shortest = labels.iter().map(|l| l.len()).min().unwrap_or(0);
    suffix = suffix.min(shortest - prefix);
    while suffix > 0 && !is_token_boundary(first, first.len() - suffix) {
        suffix -= 1;
    }
    (prefix, suffix)
}

/// `label` as a legend row shows it under the width check `fits`: whole when
/// it fits; otherwise its core — what is left after the legend's common prefix
/// and suffix `affixes` — kept intact, with the room beyond it given back to
/// the stem (the prefix's end first, then the suffix's start) and ellipses
/// marking what was dropped, so rows that differ only in "064" / "065" stay
/// distinguishable and a long name still reads as "…Insitu_064 data" rather
/// than a bare "…data"; and when even the core is too wide, the core elided
/// from the middle so a long distinguishing part keeps both its ends. A label
/// that is nothing but stem (an empty core) is treated as all core.
fn elide_label(label: &str, affixes: (usize, usize), fits: impl Fn(&str) -> bool) -> String {
    if fits(label) {
        return label.to_owned();
    }
    let (prefix, suffix) = affixes;
    let core = if prefix + suffix < label.len() {
        prefix..label.len() - suffix
    } else {
        0..label.len()
    };
    let head: Vec<char> = label[..core.start].chars().collect();
    let tail: Vec<char> = label[core.end..].chars().collect();
    let core: Vec<char> = label[core].chars().collect();
    // "…<prefix end><core><suffix start>…" with `keep_h` / `keep_t` stem chars
    // shown on each side; an ellipsis marks a side that dropped anything.
    let assemble = |keep_h: usize, core_s: &str, keep_t: usize| -> String {
        let h: String = head[head.len() - keep_h..].iter().collect();
        let t: String = tail[..keep_t].iter().collect();
        let lead = if keep_h < head.len() { "…" } else { "" };
        let trail = if keep_t < tail.len() { "…" } else { "" };
        format!("{lead}{h}{core_s}{t}{trail}")
    };
    let whole: String = core.iter().collect();
    if fits(&assemble(0, &whole, 0)) {
        let keep_h = largest(head.len(), |k| fits(&assemble(k, &whole, 0)));
        let keep_t = largest(tail.len(), |k| fits(&assemble(keep_h, &whole, k)));
        return assemble(keep_h, &whole, keep_t);
    }
    // Narrower than the core: each step drops one more char from the core's
    // middle, alternating sides so both ends survive as long as possible.
    for keep in (0..core.len()).rev() {
        let (a, b) = (keep.div_ceil(2), keep / 2);
        let start: String = core[..a].iter().collect();
        let end: String = core[core.len() - b..].iter().collect();
        let cand = assemble(0, &format!("{start}…{end}"), 0).replace("……", "…");
        if fits(&cand) {
            return cand;
        }
    }
    "…".to_owned()
}

/// The largest `k` in `0..=n` with `ok(k)`, for an `ok` that holds at `0` and
/// stays true up to the answer (a width check on a string that grows with
/// `k`); binary search.
fn largest(n: usize, ok: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Draw the plain in-axes legend: one row per entry — a short line swatch in the
/// curve's colour, then its label. No fill, border, or visibility toggle (unlike
/// rsplot's `show_legend`, which the GUI deliberately bypasses); just the colour
/// key over the transparent overlay. Rows sense a *click* (returned as the
/// clicked label) so a caller can highlight that curve, but not a *drag*, so a
/// drag anywhere on the legend still moves the enclosing `Area`. An emphasized
/// entry (its `bool` set — the highlighted curve) is drawn with a faint
/// selection background, a bold label, and a thicker swatch, so the selected
/// curve is identifiable in the legend. Returns the label clicked this frame, if
/// any.
fn draw_legend(
    ui: &mut egui::Ui,
    entries: &[(String, egui::Color32, bool)],
    data_bg: egui::Color32,
) -> Option<String> {
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
    // The emphasized (highlighted) row gets a clear selection fill plus an
    // outline that reads on either canvas — light on a dark background, dark on a
    // light one. A near-invisible tint (alpha 28) was the old value; this is
    // strong enough to spot at a glance.
    let sel_bg = if luma > 128.0 {
        egui::Color32::from_black_alpha(110)
    } else {
        egui::Color32::from_white_alpha(110)
    };
    let body = egui::TextStyle::Body.resolve(ui.style());
    // What every row shares, so a row too long to show whole keeps the part
    // that tells it from the others (see `elide_label`).
    let labels: Vec<&str> = entries.iter().map(|(l, _, _)| l.as_str()).collect();
    let affixes = common_affixes(&labels);
    let mut clicked = None;
    for (label, color, emph) in entries {
        let mut frame = egui::Frame::new()
            .inner_margin(egui::Margin::symmetric(4, 1))
            .corner_radius(egui::CornerRadius::same(3));
        if *emph {
            frame = frame.fill(sel_bg).stroke(egui::Stroke::new(1.0, text));
        }
        let row = frame
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (rect, _) = ui
                        .allocate_exact_size(egui::vec2(SWATCH_W, SWATCH_H), egui::Sense::hover());
                    let y = rect.center().y;
                    // Thicken the emphasized entry's swatch to mirror the
                    // emphasized curve (3.0 line width) it stands for.
                    let swatch_w = if *emph { 3.0 } else { 2.0 };
                    ui.painter().line_segment(
                        [
                            egui::pos2(rect.left() + 2.0, y),
                            egui::pos2(rect.right() - 2.0, y),
                        ],
                        egui::Stroke::new(swatch_w, *color),
                    );
                    ui.add_space(4.0);
                    // Elide to the available width instead of overflowing: a
                    // label inside `horizontal` neither wraps nor clips by
                    // default, so a long curve name would push the row past
                    // the legend's width cap. The emphasized row renders bold
                    // (wider), hence its slack.
                    let avail = ui.available_width() - if *emph { 8.0 } else { 0.0 };
                    let painter = ui.painter();
                    let fits = |s: &str| {
                        painter
                            .layout_no_wrap(s.to_owned(), body.clone(), egui::Color32::WHITE)
                            .rect
                            .width()
                            <= avail
                    };
                    let shown = elide_label(label, affixes, fits);
                    let rich = egui::RichText::new(shown).color(text);
                    ui.label(if *emph { rich.strong() } else { rich });
                });
            })
            .response
            // Add click sense to the whole row without capturing drags, so the
            // legend `Area` still moves when dragged from a row.
            .interact(egui::Sense::click());
        if row.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if row.clicked() {
            clicked = Some(label.clone());
        }
    }
    clicked
}

/// Draw the cursor overlay over the data area `area` while the pointer is at
/// `pos`: the `(x, y)` coordinate readout in a small box near the pointer when
/// `readout` is on, and thin guide lines through the pointer spanning the data
/// area when `crosshair` is on. xafsview draws this itself rather than using
/// rsplot's `graph_cursor`, which fuses the lines and the readout behind one
/// flag; splitting it lets each get its own toolbar toggle (see [`show`]).
///
/// `data` is the pointer's data-space coordinate (`Transform::pixel_to_data`);
/// the caller passes it pre-computed and only calls this while `pos` is inside
/// `area`. Colours contrast with the data-area background `data_bg`, like the
/// legend, so they read on either the light or dark canvas. The readout box is
/// placed lower-right of the cursor and flips to stay inside `area`, matching
/// rsplot's own crosshair readout; values use rsplot's `%.7g` number format so
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
    // Clipped to the data area: the readout box flips to stay inside, but on a
    // plot narrower than the label the flip alone cannot keep it in.
    let painter = ui.painter().with_clip_rect(area);
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
            rsplot::format_value(data.0),
            rsplot::format_value(data.1)
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

/// The vertex nearest to `cursor` (screen pixels) across every curve in `curves`,
/// within `threshold_px`, or `None` if no curve has a vertex that close. Each
/// curve is hit-tested in pixel space through `transform` (rsplot's
/// [`nearest_point`](rsplot::nearest_point)), so "nearest" is visual, not skewed
/// by the two axes' differing data scales; the global minimum across curves wins
/// (ties break to the earliest curve). Factored out of [`show`] so the
/// multi-curve reduction is unit-testable without an egui frame.
fn nearest_pick(
    curves: &[PickCurve],
    transform: &rsplot::Transform,
    cursor: egui::Pos2,
    threshold_px: f32,
) -> Option<Pick> {
    let mut best: Option<(f32, Pick)> = None;
    for c in curves {
        if let Some(p) = rsplot::nearest_point(&c.points, transform, cursor, threshold_px)
            && best.as_ref().is_none_or(|(d, _)| p.dist_px < *d)
        {
            best = Some((
                p.dist_px,
                Pick {
                    label: c.label.clone(),
                    color: c.color,
                    x: p.x,
                    y: p.y,
                },
            ));
        }
    }
    best.map(|(_, pick)| pick)
}

/// Draw the last graph-click pick — the clicked curve's colour swatch, its name,
/// and the picked `(x, y)` — pinned at the bottom-left of the data area `area`, so
/// it stays put as a *separate* readout (unlike the pointer-following cursor
/// overlay). `data_bg` is the canvas colour the text/box tones contrast against.
fn draw_pick_readout(ui: &egui::Ui, area: egui::Rect, pick: &Pick, data_bg: egui::Color32) {
    // Clipped to the data area: a long curve name on a narrow plot must not
    // spill over the panels beside the graph.
    let painter = ui.painter().with_clip_rect(area);
    let luma = 0.299 * data_bg.r() as f32 + 0.587 * data_bg.g() as f32 + 0.114 * data_bg.b() as f32;
    let dark_canvas = luma <= 128.0;
    let fg = if dark_canvas {
        egui::Color32::from_gray(0xe0)
    } else {
        egui::Color32::from_gray(0x20)
    };
    let font = egui::FontId::proportional(11.0);
    let pad = egui::vec2(4.0, 2.0);
    // A colour swatch keys the readout to the picked curve (matching its legend row).
    const SWATCH: f32 = 8.0;
    const GAP: f32 = 5.0;
    // Pin to the bottom-left of the data area, inset so it clears the axes.
    const INSET: f32 = 4.0;
    // The coordinates are the point of the readout, so they are laid out
    // first and the curve name gets whatever width is left, elided with an
    // ellipsis — a long scan name on a narrow plot used to push "y = …" past
    // the clip edge.
    let coords = painter.layout_no_wrap(
        format!(
            "   x = {}, y = {}",
            rsplot::format_value(pick.x),
            rsplot::format_value(pick.y)
        ),
        font.clone(),
        fg,
    );
    let label_w =
        (area.width() - 2.0 * INSET - SWATCH - GAP - pad.x * 2.0 - coords.size().x).max(0.0);
    let mut job = egui::text::LayoutJob::simple_singleline(pick.label.clone(), font, fg);
    job.wrap = egui::text::TextWrapping::truncate_at_width(label_w);
    let label = painter.layout_job(job);
    let text_w = label.size().x + coords.size().x;
    let text_h = label.size().y.max(coords.size().y);
    let size = egui::vec2(text_w + SWATCH + GAP + pad.x * 2.0, text_h + pad.y * 2.0);
    let min = egui::pos2(area.left() + INSET, area.bottom() - INSET - size.y);
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
    let swatch_center = egui::pos2(min.x + pad.x + SWATCH * 0.5, min.y + size.y * 0.5);
    painter.circle_filled(swatch_center, SWATCH * 0.5, pick.color);
    let text_min = egui::pos2(min.x + pad.x + SWATCH + GAP, min.y + pad.y);
    let label_w = label.size().x;
    painter.galley(text_min, label, fg);
    painter.galley(text_min + egui::vec2(label_w, 0.0), coords, fg);
}

/// A toolbar toggle button for the crosshair guide lines drawn by
/// [`draw_cursor_overlay`]. rsplot's built-in cursor button (a
/// crosshair-with-circle icon) is repurposed as the coordinate-*readout* toggle,
/// so this separate button owns the *lines*. It is styled to match rsplot's own
/// icon buttons (same 28×24 footprint, hover/selected fill + stroke, 1.6px icon
/// stroke inset by 5px) and draws a plain crosshair — no circle — to read
/// distinctly from rsplot's cursor icon. `on` is [`Plot::crosshair`].
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
/// of `plot`, mirroring rsplot's `Plot2D::symbol_tool_button` for the curve
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

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{Color32, pos2, vec2};

    #[test]
    fn common_affixes_split_on_token_boundaries() {
        // The shared stem ends at the "_" before the scan number and resumes at
        // the "." of the extension, not inside the digits.
        assert_eq!(
            common_affixes(&["PGT_Mn_S2_Insitu_064.dat", "PGT_Mn_S2_Insitu_065.dat"]),
            (17, 4)
        );
        // A digit run is one token: "scan" + "001".
        assert_eq!(common_affixes(&["scan001", "scan002"]), (4, 0));
        // One label is nothing but stem: the suffix stops at its prefix.
        assert_eq!(common_affixes(&["a_b.dat", "a_b_c.dat"]), (3, 4));
        assert_eq!(common_affixes(&["x.dat", "y.dat"]), (0, 4));
        // Fewer than two labels share nothing.
        assert_eq!(common_affixes(&["alone"]), (0, 0));
        assert_eq!(common_affixes(&[]), (0, 0));
    }

    #[test]
    fn elide_label_keeps_the_distinguishing_core() {
        // Width as a character budget.
        let fits = |n: usize| move |s: &str| s.chars().count() <= n;
        let label = "PGT_Mn_S2_Insitu_064.dat";
        let affixes = (17, 4);
        assert_eq!(elide_label(label, affixes, fits(30)), label);
        // Too long: the stem goes, the scan number stays.
        assert_eq!(elide_label(label, affixes, fits(5)), "…064…");
        // Room beyond the core goes back to the stem, from the prefix's end,
        // then to the suffix once the whole prefix fits.
        assert_eq!(elide_label(label, affixes, fits(12)), "…Insitu_064…");
        assert_eq!(elide_label(label, affixes, fits(15)), "…S2_Insitu_064…");
        assert_eq!(
            elide_label(label, affixes, fits(23)),
            "PGT_Mn_S2_Insitu_064.d…"
        );
        // A lone group's "data" / "fit" pair keeps its name, not just the role.
        assert_eq!(
            elide_label("PGT_Mn_S2_NCM50_W_Insitu_000 data", (29, 0), fits(16)),
            "…Insitu_000 data"
        );
        // Narrower than the core: the core itself gives from the middle.
        assert_eq!(elide_label(label, affixes, fits(4)), "…0…");
        // An all-stem label is elided as a whole, from the middle.
        assert_eq!(elide_label("a_b.dat", (3, 4), fits(5)), "a_…at");
        // Nothing shared (a lone curve): middle elision keeps both ends.
        assert_eq!(elide_label("abcdef", (0, 0), fits(4)), "ab…f");
        assert_eq!(elide_label("abcdef", (0, 0), fits(0)), "…");
    }

    /// Data `[0,10] × [0,10]` mapped onto a 100×100 px area, so `data_to_pixel`
    /// gives round pixel coordinates for the fixtures below.
    fn transform_10x10() -> rsplot::Transform {
        rsplot::Transform::new(
            0.0,
            10.0,
            0.0,
            10.0,
            egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 100.0)),
        )
    }

    fn curve(label: &str, points: &[(f64, f64)]) -> PickCurve {
        PickCurve {
            label: label.to_owned(),
            color: Color32::WHITE,
            points: points.to_vec(),
        }
    }

    #[test]
    fn nearest_pick_takes_the_global_minimum_across_curves_not_the_first() {
        // Curve "far" is listed first but "near" holds the closest vertex, so the
        // pick must cross curves to the global minimum, not stop at the first hit.
        let t = transform_10x10();
        let curves = vec![curve("far", &[(1.0, 1.0)]), curve("near", &[(5.0, 5.0)])];
        let cursor = t.data_to_pixel(5.0, 5.0) + vec2(2.0, 0.0);
        let pick = nearest_pick(&curves, &t, cursor, 10.0).expect("a pick");
        assert_eq!(pick.label, "near");
        assert_eq!((pick.x, pick.y), (5.0, 5.0));
    }

    #[test]
    fn nearest_pick_respects_the_threshold_boundary() {
        // A single vertex exactly 5 px from the cursor: inside a 6 px threshold,
        // outside a 4 px one.
        let t = transform_10x10();
        let curves = vec![curve("a", &[(5.0, 5.0)])];
        let cursor = t.data_to_pixel(5.0, 5.0) + vec2(5.0, 0.0);
        assert!(nearest_pick(&curves, &t, cursor, 6.0).is_some());
        assert!(nearest_pick(&curves, &t, cursor, 4.0).is_none());
    }

    #[test]
    fn nearest_pick_is_none_with_no_curves() {
        let t = transform_10x10();
        assert!(nearest_pick(&[], &t, pos2(50.0, 50.0), 100.0).is_none());
    }
}
