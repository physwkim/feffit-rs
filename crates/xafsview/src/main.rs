//! XAFSView — a modern Rust GUI for XAFS data reduction and FEFF fitting.
//!
//! Re-implements the feature set of the old LabVIEW `XAFSView` on top of the
//! `feffit-rs` engines (pre-edge/normalize, AUTOBK, FEFFIT, LCF/PCA, FEFF8L/
//! FEFF10) and `siplot` for plotting. This binary is the GUI shell; the data
//! model lives in the `xasdata` crate and the math in the other workspace crates.

mod analysis_ui;
mod app;
mod atoms_ui;
mod calc_ui;
mod clean_ui;
mod feffit_batch;
mod feffit_ui;
mod import;
mod mback_ui;
mod plot_data;
mod reduce_ui;
mod wavelet;
mod xanes_ui;

use app::XafsViewApp;

fn main() -> eframe::Result {
    // siplot paints through an egui-wgpu callback, so the wgpu renderer is
    // mandatory (the OpenGL backend has no RenderState to install into).
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("XAFSView")
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "XAFSView",
        options,
        Box::new(|cc| Ok(Box::new(XafsViewApp::new(cc)) as Box<dyn eframe::App>)),
    )
}
