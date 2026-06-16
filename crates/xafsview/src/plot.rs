//! Shared `siplot::Plot1D` construction for the GUI.
//!
//! Every plot-bearing tab/window builds its plot through [`new_plot1d`] so they
//! all get the same house data margins. siplot's `DataMargins` default to zero,
//! which draws curves flush against the axis frame (a sample at the data maximum
//! sits exactly on the top frame line and is visually clipped). A uniform margin
//! on every side — matching matplotlib's `axes.margins` default — keeps the data
//! off the frame, and routing every plot through one constructor means no site
//! can forget it.

use eframe::egui_wgpu::RenderState;
use siplot::{DataMargins, Plot1D, PlotId};

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
    plot
}
