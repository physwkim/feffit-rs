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
use siplot::{DataMargins, Plot1D, PlotId, Symbol};

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
    plot
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

/// Render `plot` as one unit: its toolbar on top, the plot canvas filling the
/// width, and a legend list down the right edge mapping each curve's
/// color/symbol to the name set with `set_item_legend`. Every plot-bearing
/// tab/window draws through this (instead of `toolbar` + `plot.show` by hand)
/// so they all get the same toolbar *and* a visible legend — siplot draws no
/// in-axes legend, so without this call the curve names never appear.
pub fn show(plot: &mut Plot1D, ui: &mut egui::Ui) {
    toolbar(plot, ui);
    ui.horizontal_top(|ui| {
        let avail = ui.available_size();
        // A narrow legend strip on the right; the plot canvas fills the rest.
        let legend_w = (avail.x * 0.22).clamp(90.0, 180.0).min(avail.x);
        let plot_w = (avail.x - legend_w).max(0.0);
        ui.allocate_ui(egui::vec2(plot_w, avail.y), |ui| {
            plot.show(ui);
        });
        ui.allocate_ui(egui::vec2(legend_w, avail.y), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                plot.show_legend(ui);
            });
        });
    });
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
