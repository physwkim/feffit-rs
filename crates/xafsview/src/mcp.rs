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
        paths: Vec<std::path::PathBuf>,
        reply: TextReply,
    },
    SetParams {
        args: Box<SetParamsArgs>,
        reply: TextReply,
    },
    AddPath {
        args: Box<AddPathArgs>,
        reply: TextReply,
    },
    SelectGroups {
        indices: Vec<usize>,
        reply: TextReply,
    },
    RemoveGroups {
        indices: Vec<usize>,
        reply: TextReply,
    },
    RemovePaths {
        indices: Vec<usize>,
        reply: TextReply,
    },
    SetGraph {
        args: SetGraphArgs,
        reply: TextReply,
    },
    SetZoom {
        args: SetZoomArgs,
        reply: TextReply,
    },
    SetTab {
        tab: String,
        reply: TextReply,
    },
    Deglitch {
        nsig: f64,
        apply: bool,
        reply: TextReply,
    },
    RunAutobk {
        reply: TextReply,
    },
    RunFit {
        mode: String,
        reply: TextReply,
    },
    GetResults {
        reply: TextReply,
    },
    InsituMap {
        args: Box<InsituMapArgs>,
        reply: TextReply,
    },
    Screenshot {
        reply: PngReply,
    },
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct LoadDataArgs {
    /// Path to one XAS data file (beamline column file or .xmu). Columns are
    /// auto-detected from the header; μ(E) is built and added as a new group,
    /// which becomes the current group. Give `path` or `paths`, not both.
    pub path: Option<String>,
    /// Several files loaded as one batch (the in-situ series flow): the first
    /// file's auto-detected column mapping applies to every file, each becomes
    /// a group, and files already loaded are skipped.
    pub paths: Option<Vec<String>>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct AddPathArgs {
    /// Path to a Feff path file (feffNNNN.dat).
    pub path: String,
    /// Path label (defaults to the file name).
    pub label: Option<String>,
    /// Wiring overrides: each is an expression over the global parameters (a
    /// plain number is a constant). Unset fields keep the standard first-shell
    /// starter wiring: degen from the file, e0 = del_e0, deltar = alpha*reff,
    /// amplitude = amp·N (N fixed at 1), and a varied per-path σ².
    pub degen: Option<String>,
    pub e0: Option<String>,
    pub ei: Option<String>,
    pub deltar: Option<String>,
    /// Overrides the whole amplitude (replacing amp·N), e.g. "s02*N1".
    pub s02: Option<String>,
    /// The N local (amplitude factor in amp·N).
    pub n: Option<String>,
    pub sigma2: Option<String>,
    pub third: Option<String>,
    pub fourth: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SelectGroupsArgs {
    /// Group indices (as shown by get_results) to check for Run — run_autobk
    /// and run_fit then target exactly these groups; the first index becomes
    /// the current group. An empty list clears the checks, so runs target only
    /// the current group.
    pub indices: Vec<usize>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct RemoveGroupsArgs {
    /// Group indices (as shown by get_results) to remove. Indices above a
    /// removed group shift down by one, exactly as in the Data window.
    pub indices: Vec<usize>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct RemovePathsArgs {
    /// Feffit path-list indices (0-based, in add order) to remove. Indices
    /// above a removed path shift down by one.
    pub indices: Vec<usize>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SetGraphArgs {
    /// Graph type by its menu label (case-insensitive): "XMU + BKG", "*K.CHI",
    /// "*R.CHI", "*Q.CHI", "XMU' + XMU''", "BKG(E)'", "BKG(k)'",
    /// "XMU' + BKG(E)'", "BKG(E)''", or "Normalize".
    pub graph: Option<String>,
    /// Overlay every loaded group's curve for the graph type, not just the
    /// current one (the "Show all groups" checkbox).
    pub show_all_groups: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SetZoomArgs {
    /// X axis display limits — give both or neither.
    pub xmin: Option<f64>,
    pub xmax: Option<f64>,
    /// Y axis (left) display limits — give both or neither.
    pub ymin: Option<f64>,
    pub ymax: Option<f64>,
    /// `true` resets the view to the data bounds (the toolbar's Reset Zoom);
    /// no limits may be given alongside it.
    pub reset: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct DeglitchArgs {
    /// Spike threshold in robust σ of the 7-point rolling window (default 10):
    /// points deviating from their rolling median by more than this are
    /// glitches.
    pub nsig: Option<f64>,
    /// `true` removes the candidate points from every targeted group (re-run
    /// run_autobk afterwards); `false`/absent is a dry run that only reports
    /// each group's candidates.
    pub apply: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SetTabArgs {
    /// Tab by its strip label (case-insensitive): "Folders", "Autobk",
    /// "Feffit", "Fit report" (alias "Feffit_txt"), "Atoms", or "Feff".
    pub tab: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct RunFitArgs {
    /// Fit mode for this run, set on the Feffit tab before running: "fit"
    /// (default — the full least-squares fit), "nofit" (evaluate the model
    /// without varying), or "only_ft" (Fourier transforms only, no model).
    pub mode: Option<String>,
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

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct InsituMapArgs {
    /// Map channel built from the loaded groups, in series (load) order:
    /// "rdf" (default — each group's |χ(R)| from the forward FT) or "xanes"
    /// (normalized μ(E) interpolated onto a common energy grid).
    pub channel: Option<String>,
    /// Instead of the loaded groups, open a BL10C series .dat file: RDF
    /// (x + one column per scan) or XANES ((E, μ) column pairs).
    pub file: Option<String>,
    /// Reference scan (1-based, series order): the map shows value − reference
    /// (the Sub map, diverging colors). Omit for the plain map.
    pub reference: Option<usize>,
    /// XANES grid start (eV); seeded from the data when omitted.
    pub e_start: Option<f64>,
    /// XANES grid stop (eV); seeded from the data when omitted.
    pub e_stop: Option<f64>,
    /// XANES grid point count; seeded from the data when omitted.
    pub e_steps: Option<usize>,
    /// XANES rows from the flattened spectrum (true, the window's default) or
    /// the plain normalized one (false — what the legacy BL10C XANES.dat holds).
    pub flat: Option<bool>,
    /// Write the displayed map to this path as a tab-separated matrix
    /// (x first column, one named column per scan) — OriginPro-importable and
    /// re-openable via `file`.
    pub save: Option<String>,
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
        description = "Load XAS data (beamline column files or .xmu): columns are auto-detected, μ(E) is built and each file becomes a group. `path` loads one file (it becomes current); `paths` loads a batch with the first file's column mapping."
    )]
    async fn load_data(
        &self,
        Parameters(a): Parameters<LoadDataArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let paths: Vec<std::path::PathBuf> = match (a.path, a.paths) {
            (Some(p), None) => vec![p.into()],
            (None, Some(v)) if !v.is_empty() => v.into_iter().map(Into::into).collect(),
            _ => {
                return Err(ErrorData::invalid_params(
                    "give exactly one of `path` (one file) or `paths` (a non-empty batch)",
                    None,
                ));
            }
        };
        self.roundtrip(|reply| McpRequest::LoadData { paths, reply })
            .await
    }

    #[tool(
        description = "Add a Feff path file (feffNNNN.dat) to the Feffit model, optionally overriding its parameter wiring with expressions over the global parameters (see set_params)."
    )]
    async fn add_path(
        &self,
        Parameters(a): Parameters<AddPathArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::AddPath {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Check groups for Run by index (see get_results): run_autobk and run_fit then target exactly these groups, and the first becomes the current group. An empty list clears the checks."
    )]
    async fn select_groups(
        &self,
        Parameters(a): Parameters<SelectGroupsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::SelectGroups {
            indices: a.indices,
            reply,
        })
        .await
    }

    #[tool(
        description = "Remove loaded groups by index (as shown by get_results). Remaining groups' indices shift down, exactly as in the Data window."
    )]
    async fn remove_groups(
        &self,
        Parameters(a): Parameters<RemoveGroupsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::RemoveGroups {
            indices: a.indices,
            reply,
        })
        .await
    }

    #[tool(
        description = "Remove Feff paths from the Feffit model by path-list index (0-based, in add order)."
    )]
    async fn remove_paths(
        &self,
        Parameters(a): Parameters<RemovePathsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::RemovePaths {
            indices: a.indices,
            reply,
        })
        .await
    }

    #[tool(
        description = "Switch the plotted graph type (XMU + BKG, *K.CHI, *R.CHI, *Q.CHI, derivatives, Normalize) and/or toggle the all-groups overlay — what screenshot then shows."
    )]
    async fn set_graph(
        &self,
        Parameters(a): Parameters<SetGraphArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::SetGraph { args: a, reply })
            .await
    }

    #[tool(
        description = "Zoom the main graph: set explicit x and/or y display limits, or reset:true to refit the view to the data. Reports the resulting limits."
    )]
    async fn set_zoom(
        &self,
        Parameters(a): Parameters<SetZoomArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::SetZoom { args: a, reply })
            .await
    }

    #[tool(
        description = "Automatic glitch removal on the checked groups (or the current one): points deviating from a rolling median by nsig robust σ. Dry run by default (reports per-group candidates); apply=true removes them — run_autobk afterwards to recompute χ."
    )]
    async fn deglitch(
        &self,
        Parameters(a): Parameters<DeglitchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::Deglitch {
            nsig: a.nsig.unwrap_or(10.0),
            apply: a.apply.unwrap_or(false),
            reply,
        })
        .await
    }

    #[tool(
        description = "Switch the main window's active tab (Folders, Autobk, Feffit, Fit report, Atoms, Feff) — what screenshot then shows."
    )]
    async fn set_tab(
        &self,
        Parameters(a): Parameters<SetTabArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::SetTab { tab: a.tab, reply })
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
        description = "Run a Feffit fit against the current group's χ(k) and return the fit report. Requires χ(k) (run_autobk first) and, for mode \"fit\"/\"nofit\" (default \"fit\"), at least one loaded Feff path; mode \"only_ft\" transforms without a model."
    )]
    async fn run_fit(
        &self,
        Parameters(a): Parameters<RunFitArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::RunFit {
            mode: a.mode.unwrap_or_else(|| "fit".to_owned()),
            reply,
        })
        .await
    }

    #[tool(
        description = "Report the session state: loaded groups (points, E₀, χ availability), the current group, and the last Feffit result (statistics, variables, path parameters)."
    )]
    async fn get_results(&self) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::GetResults { reply })
            .await
    }

    #[tool(
        description = "Build and show the In-situ map (scan-series image, the BL10C IMAGE view): channel \"rdf\" (|χ(R)| rows) or \"xanes\" (normalized μ(E) on a common grid) from the loaded groups, or open a BL10C series .dat via `file`. `reference` (1-based) subtracts that scan (the Sub map); `save` writes the displayed map as an OriginPro-ready tab-separated matrix."
    )]
    async fn insitu_map(
        &self,
        Parameters(a): Parameters<InsituMapArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::InsituMap {
            args: Box::new(a),
            reply,
        })
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
