//! XAFSView — a modern Rust GUI for XAFS data reduction and FEFF fitting.
//!
//! Re-implements the feature set of the old LabVIEW `XAFSView` on top of the
//! `feffit-rs` engines (pre-edge/normalize, AUTOBK, FEFFIT, LCF/PCA, FEFF8L/
//! FEFF10) and `siplot` for plotting. This binary is the GUI shell; the data
//! model lives in the `xasdata` crate and the math in the other workspace crates.

mod analysis_ui;
mod app;
mod atoms_ui;
mod batch_load;
mod calc_ui;
mod chi_io;
mod clean_ui;
mod feffit_batch;
mod feffit_ui;
mod fonts;
mod import;
mod mback_ui;
mod plot;
mod plot_data;
mod plot_files;
mod reduce_ui;
mod timeres_ui;
mod wavelet;
mod widgets;
mod window;
mod xanes_ui;

use std::sync::Arc;

use app::XafsViewApp;

/// Window / taskbar icon, vendored as a 256×256 PNG and embedded so the binary
/// stays self-contained (no sidecar file to ship). Decoded once at startup.
/// (macOS uses the `.app` bundle icon for the dock, so this is mainly visible on
/// Windows/Linux — which is where a distributed build needs it.)
static ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

fn main() -> eframe::Result {
    let icon =
        eframe::icon_data::from_png_bytes(ICON_PNG).expect("bundled icon.png is a valid PNG");

    // siplot paints through an egui-wgpu callback, so the wgpu renderer is
    // mandatory (the OpenGL backend has no RenderState to install into).
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("XAFSView")
            .with_inner_size([1200.0, 800.0])
            .with_icon(Arc::new(icon)),
        ..Default::default()
    };

    eframe::run_native(
        "XAFSView",
        options,
        Box::new(|cc| Ok(Box::new(XafsViewApp::new(cc)) as Box<dyn eframe::App>)),
    )
}
