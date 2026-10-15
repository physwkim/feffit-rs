//! XAFSView — a modern Rust GUI for XAFS data reduction and FEFF fitting.
//!
//! Re-implements the feature set of the old LabVIEW `XAFSView` on top of the
//! `feffit-rs` engines (pre-edge/normalize, AUTOBK, FEFFIT, LCF/PCA, FEFF8L/
//! FEFF10) and `rsplot` for plotting. This binary is the GUI shell; the data
//! model lives in the `xasdata` crate and the math in the other workspace crates.

mod analysis_ui;
mod app;
mod atoms_ui;
mod calc_ui;
mod chi_io;
mod clean_ui;
mod feffit_batch;
mod feffit_ui;
mod file_browser;
mod fonts;
mod import;
mod insitu_ui;
mod jobs;
mod mback_ui;
mod mcp;
mod open_data;
mod peakfit;
mod plot;
mod plot_data;
mod plot_files;
mod reduce_ui;
mod status_log;
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

/// Parse `--mcp [PORT]` / `--mcp=PORT` from the command line: start the
/// embedded MCP server on that port ([`mcp::DEFAULT_PORT`] when none is given).
/// Other arguments are ignored, as before.
fn mcp_port_arg() -> Option<u16> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    for (i, a) in args.iter().enumerate() {
        if a == "--mcp" {
            let port = args.get(i + 1).and_then(|p| p.parse().ok());
            return Some(port.unwrap_or(mcp::DEFAULT_PORT));
        }
        if let Some(p) = a.strip_prefix("--mcp=") {
            return Some(p.parse().unwrap_or(mcp::DEFAULT_PORT));
        }
    }
    None
}

fn main() -> eframe::Result {
    let mcp_port = mcp_port_arg();
    let icon =
        eframe::icon_data::from_png_bytes(ICON_PNG).expect("bundled icon.png is a valid PNG");

    // rsplot paints through an egui-wgpu callback, so the wgpu renderer is
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
        Box::new(move |cc| Ok(Box::new(XafsViewApp::new(cc, mcp_port)) as Box<dyn eframe::App>)),
    )
}
