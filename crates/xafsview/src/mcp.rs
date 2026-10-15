//! GUI-embedded MCP server: lets an MCP client (Claude Code, Codex, …) drive
//! the *running* XAFSView — load data, set parameters, run AUTOBK and Feffit,
//! read results, and capture screenshots — over MCP streamable HTTP on
//! localhost. Enabled with `--mcp [port]` or the Settings-menu toggle.
//!
//! Architecture: rmcp's streamable-HTTP layer is a plain `tower` service, so a
//! background thread runs a small tokio runtime serving it through hyper. Each
//! tool call becomes an [`McpRequest`] carrying a oneshot reply sender; the
//! request goes over an `mpsc` channel to the egui thread (plus a
//! `request_repaint` to wake an idle event loop), where the app handles it at
//! the top of the frame against the live state — the same methods the buttons
//! call — and answers through the oneshot. Screenshots ride the same channel
//! but resolve later, when the `ViewportCommand::Screenshot` reply event
//! arrives in a subsequent frame.

use std::sync::{Arc, mpsc};

use eframe::egui;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{
    ErrorData, ServerHandler,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    schemars, tool, tool_handler, tool_router,
};
use tokio_util::sync::CancellationToken;

/// Default listen port for `--mcp` / the Settings toggle.
pub const DEFAULT_PORT: u16 = 8787;

/// Reply channel for the text-outcome tools: `Ok` is a success message
/// (status line or report), `Err` the failure message.
pub type TextReply = tokio::sync::oneshot::Sender<Result<String, String>>;

/// Reply channel for `screenshot`: encoded PNG bytes.
pub type PngReply = tokio::sync::oneshot::Sender<Result<Vec<u8>, String>>;

/// One tool call, handed to the egui thread. Every variant carries the oneshot
/// that resolves the awaiting HTTP request.
pub enum McpRequest {
    LoadData {
        path: std::path::PathBuf,
        reply: TextReply,
    },
    SetParams {
        args: Box<SetParamsArgs>,
        reply: TextReply,
    },
    RunAutobk {
        reply: TextReply,
    },
    RunFit {
        reply: TextReply,
    },
    GetResults {
        reply: TextReply,
    },
    Screenshot {
        reply: PngReply,
    },
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct LoadDataArgs {
    /// Path to a XAS data file (beamline column file or .xmu). Columns are
    /// auto-detected from the header; μ(E) is built and added as a new group,
    /// which becomes the current group.
    pub path: String,
}

/// Parameters for `set_params`. Every field is optional; only the given ones
/// change, exactly as if edited in the GUI.
#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SetParamsArgs {
    /// AUTOBK / normalization / FT parameters (the Autobk tab).
    pub autobk: Option<AutobkPatch>,
    /// Feffit fit configuration (the Feffit tab).
    pub feffit: Option<FeffitPatch>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct AutobkPatch {
    /// Manual E₀ (eV). Setting it turns E₀ auto-detection off.
    pub e0: Option<f64>,
    /// Re-enable automatic E₀ detection.
    pub e0_auto: Option<bool>,
    /// AUTOBK Rbkg (Å).
    pub rbkg: Option<f64>,
    /// k-weight for the background FT.
    pub kweight: Option<i32>,
    /// Forward-FT window lower bound (Å⁻¹).
    pub kmin: Option<f64>,
    /// Forward-FT window upper bound (Å⁻¹).
    pub kmax: Option<f64>,
    /// Forward-FT window taper width (Å⁻¹).
    pub dk: Option<f64>,
    /// Reverse-FT R-window lower bound (Å).
    pub rmin: Option<f64>,
    /// Reverse-FT R-window upper bound (Å).
    pub rmax: Option<f64>,
    /// Reverse-FT R-window taper width (Å).
    pub dr: Option<f64>,
    /// Pre-edge fit lower bound (eV, relative to E₀). Setting any of
    /// pre1/pre2/norm1/norm2/nnorm turns pre/norm auto-ranging off.
    pub pre1: Option<f64>,
    /// Pre-edge fit upper bound (eV, relative to E₀).
    pub pre2: Option<f64>,
    /// Normalization fit lower bound (eV, relative to E₀).
    pub norm1: Option<f64>,
    /// Normalization fit upper bound (eV, relative to E₀).
    pub norm2: Option<f64>,
    /// Post-edge normalization polynomial order (0–3).
    pub nnorm: Option<usize>,
    /// Re-enable automatic pre-edge / normalization ranges.
    pub pre_norm_auto: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct FeffitPatch {
    /// Global fit parameters to add or update by name.
    pub params: Option<Vec<ParamSpec>>,
    /// Fit-FT window lower bound (Å⁻¹).
    pub kmin: Option<f64>,
    /// Fit-FT window upper bound (Å⁻¹).
    pub kmax: Option<f64>,
    /// Fit k-weight.
    pub kweight: Option<i32>,
    /// Fit-FT window taper width (Å⁻¹).
    pub dk: Option<f64>,
    /// Fit R-range lower bound (Å).
    pub rmin: Option<f64>,
    /// Fit R-range upper bound (Å).
    pub rmax: Option<f64>,
    /// Fit R-window taper width (Å).
    pub dr: Option<f64>,
    /// Fit space: "R", "k", or "q".
    pub fitspace: Option<String>,
    /// What "Run" does: "fit" (least squares), "nofit" (evaluate the model at
    /// the current guesses), or "only_ft" (transform the data, no model).
    pub fit_mode: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ParamSpec {
    /// Parameter name (e.g. "amp", "del_e0", "alpha").
    pub name: String,
    /// "vary" (free, starting at `value`), "fixed" (held at `value`), or
    /// "expr" (constraint expression over the other parameters).
    pub kind: String,
    /// Starting / fixed value, for "vary" and "fixed".
    pub value: Option<f64>,
    /// Constraint expression, for "expr".
    pub expr: Option<String>,
}

/// The per-connection MCP handler: forwards each tool call to the egui thread
/// and awaits the oneshot. Cheap to clone (a channel sender + an egui context).
#[derive(Clone)]
pub struct XafsMcp {
    tx: mpsc::Sender<McpRequest>,
    egui_ctx: egui::Context,
}

impl XafsMcp {
    /// Send a request to the egui thread and wake its event loop — without the
    /// repaint an idle app would not poll the channel until the next input.
    fn send(&self, req: McpRequest) -> Result<(), ErrorData> {
        self.tx
            .send(req)
            .map_err(|_| ErrorData::internal_error("the XAFSView app is shutting down", None))?;
        self.egui_ctx.request_repaint();
        Ok(())
    }

    /// Round-trip a text-outcome request: app `Ok`/`Err` becomes a tool
    /// success/error result; a dropped reply (app closed) is a protocol error.
    async fn roundtrip(
        &self,
        build: impl FnOnce(TextReply) -> McpRequest,
    ) -> Result<CallToolResult, ErrorData> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.send(build(tx))?;
        match rx.await {
            Ok(Ok(msg)) => Ok(CallToolResult::success(vec![ContentBlock::text(msg)])),
            Ok(Err(e)) => Ok(CallToolResult::error(vec![ContentBlock::text(e)])),
            Err(_) => Err(ErrorData::internal_error(
                "the XAFSView app dropped the request",
                None,
            )),
        }
    }
}

#[tool_router]
impl XafsMcp {
    #[tool(
        description = "Load a XAS data file (beamline column file or .xmu): columns are auto-detected, μ(E) is built and added as a new group, and that group becomes current."
    )]
    async fn load_data(
        &self,
        Parameters(a): Parameters<LoadDataArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::LoadData {
            path: a.path.into(),
            reply,
        })
        .await
    }

    #[tool(
        description = "Set AUTOBK / normalization / FT parameters and/or the Feffit fit configuration (global parameters, fit window, fit space, fit mode). Only the given fields change."
    )]
    async fn set_params(
        &self,
        Parameters(a): Parameters<SetParamsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::SetParams {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Run AUTOBK background removal + Fourier transforms on the current group (or every checked group) with the current parameters, writing the χ output files."
    )]
    async fn run_autobk(&self) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::RunAutobk { reply })
            .await
    }

    #[tool(
        description = "Run the Feffit tab's configured fit against the current group's χ(k) and return the fit report. Requires χ(k) (run_autobk first) and, for fit_mode \"fit\"/\"nofit\", at least one loaded Feff path."
    )]
    async fn run_fit(&self) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::RunFit { reply }).await
    }

    #[tool(
        description = "Report the session state: loaded groups (points, E₀, χ availability), the current group, and the last Feffit result (statistics, variables, path parameters)."
    )]
    async fn get_results(&self) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::GetResults { reply })
            .await
    }

    #[tool(description = "Capture a screenshot of the XAFSView main window as a PNG image.")]
    async fn screenshot(&self) -> Result<CallToolResult, ErrorData> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.send(McpRequest::Screenshot { reply: tx })?;
        match rx.await {
            Ok(Ok(png)) => {
                use base64::Engine as _;
                let b64 = base64::engine::general_purpose::STANDARD.encode(png);
                Ok(CallToolResult::success(vec![ContentBlock::image(
                    b64,
                    "image/png",
                )]))
            }
            Ok(Err(e)) => Ok(CallToolResult::error(vec![ContentBlock::text(e)])),
            Err(_) => Err(ErrorData::internal_error(
                "the XAFSView app dropped the request",
                None,
            )),
        }
    }
}

#[tool_handler(
    name = "xafsview",
    instructions = "Remote control of a running XAFSView (XAFS data reduction + FEFF fitting GUI). Typical flow: load_data → set_params → run_autobk → run_fit → get_results; screenshot shows the current window."
)]
impl ServerHandler for XafsMcp {}

/// The running server: the listener thread plus its shutdown token. Dropping
/// (or [`stop`](Self::stop)) cancels the sessions and joins the thread.
pub struct McpServer {
    /// The bound port (useful when the requested port was 0).
    pub port: u16,
    cancel: CancellationToken,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl McpServer {
    pub fn stop(&mut self) {
        self.cancel.cancel();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Bind `127.0.0.1:port` and serve MCP on a background thread. Returns the
/// server handle and the request receiver the egui thread must poll (see
/// `XafsViewApp::poll_mcp`). The bind happens here, synchronously, so a port
/// conflict surfaces as the `Err` instead of a dead thread.
pub fn start(
    port: u16,
    egui_ctx: egui::Context,
) -> std::io::Result<(McpServer, mpsc::Receiver<McpRequest>)> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();
    let (tx, rx) = mpsc::channel();
    let cancel = CancellationToken::new();
    let cancel_serve = cancel.clone();
    let thread = std::thread::Builder::new()
        .name("mcp-server".into())
        .spawn(move || {
            // Two workers: one can be parked in a long poll (the SSE stream)
            // while the other keeps accepting connections.
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("tokio runtime for the MCP server");
            rt.block_on(serve(listener, tx, egui_ctx, cancel_serve));
            // Dropping `rt` here aborts the in-flight connection tasks.
        })?;
    Ok((
        McpServer {
            port,
            cancel,
            thread: Some(thread),
        },
        rx,
    ))
}

/// Accept loop: each connection gets rmcp's streamable-HTTP handler via
/// hyper/http1 (SSE streaming needs nothing more). Ends on cancellation, which
/// also terminates the MCP sessions (the config carries the same token).
async fn serve(
    listener: std::net::TcpListener,
    tx: mpsc::Sender<McpRequest>,
    egui_ctx: egui::Context,
    cancel: CancellationToken,
) {
    let Ok(listener) = tokio::net::TcpListener::from_std(listener) else {
        return;
    };
    let service = StreamableHttpService::new(
        move || {
            Ok(XafsMcp {
                tx: tx.clone(),
                egui_ctx: egui_ctx.clone(),
            })
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default().with_cancellation_token(cancel.clone()),
    );
    loop {
        let stream = tokio::select! {
            _ = cancel.cancelled() => break,
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(_) => continue,
            },
        };
        let service = service.clone();
        tokio::spawn(async move {
            let svc = hyper::service::service_fn(move |req| {
                let service = service.clone();
                async move { Ok::<_, std::convert::Infallible>(service.handle(req).await) }
            });
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(hyper_util::rt::TokioIo::new(stream), svc)
                .await;
        });
    }
}

/// Encode an egui screenshot as PNG. The capture is opaque (alpha 255), so the
/// premultiplied-sRGBA pixel bytes are valid straight RGBA.
pub fn encode_png(img: &egui::ColorImage) -> Result<Vec<u8>, String> {
    let [w, h] = img.size;
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(img.as_raw())
        .map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(out)
}
