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
    OpenWindow {
        name: String,
        reply: TextReply,
    },
    CloseWindow {
        name: String,
        reply: TextReply,
    },
    SetTheme {
        dark: bool,
        reply: TextReply,
    },
    PlotData {
        args: Box<PlotDataArgs>,
        reply: TextReply,
    },
    FeffitGraph {
        args: FeffitGraphArgs,
        reply: TextReply,
    },
    Xanes {
        args: Box<XanesArgs>,
        reply: TextReply,
    },
    Mback {
        args: MbackArgs,
        reply: TextReply,
    },
    Lcf {
        args: Box<LcfArgs>,
        reply: TextReply,
    },
    Pca {
        args: Box<PcaArgs>,
        reply: TextReply,
    },
    Wavelet {
        args: WaveletArgs,
        reply: TextReply,
    },
    EditMu {
        args: Box<EditMuArgs>,
        reply: TextReply,
    },
    BuildFeffInp {
        args: Box<AtomsArgs>,
        reply: TextReply,
    },
    RunFeff {
        backend: Option<String>,
        reply: TextReply,
    },
    Graph {
        args: Box<GraphArgs>,
        reply: TextReply,
    },
    Screenshot {
        reply: PngReply,
    },
}

/// The pop-up windows `open_window` / `close_window` address, by name — the
/// same windows the Tools menu, the top bar and the Log / Data buttons open.
pub const WINDOW_NAMES: [&str; 19] = [
    "open_data",
    "data",
    "log",
    "plot_data",
    "edit_mu",
    "periodic_table",
    "ion_chamber",
    "powder",
    "xanes",
    "mback",
    "wavelet",
    "ke_convert",
    "lcf",
    "pca",
    "time_resolved",
    "insitu_map",
    "plot_sites",
    "feffit_batch",
    "about",
];

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
    /// Show the draggable FT-window band — kmin/kmax on *K.CHI, rmin/rmax on
    /// *R.CHI (the "Show k/R range" checkbox); `graph` action "drag_window"
    /// then moves it.
    pub show_range: Option<bool>,
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
pub struct GraphArgs {
    /// Which window's graph: "main" (default — the shared Autobk / Feffit
    /// plot), "plot_data", "xanes", "mback", "lcf", "pca" or "insitu_map" (the
    /// profile plot under the map).
    pub window: Option<String>,
    /// The gesture: "view" (report only); "zoom_in" / "zoom_out" (the toolbar
    /// buttons, about the view centre); "zoom_box" (a Zoom-mode drag from
    /// (`x`, `y`) to (`x2`, `y2`)); "pan" (a drag that moves the view by `dx`,
    /// `dy`); "zoom_back"; "reset"; "click" (a left-click at (`x`, `y`): pins
    /// the nearest curve's readout, which edit_mu remove_point then uses);
    /// "drag_window" (drag the FT-window band's edges to [`min`, `max`]);
    /// "legend_click" (click the legend row `label`, e.g. Plot Data's
    /// highlight). Coordinates are data units of the graph's axes.
    pub action: String,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub x2: Option<f64>,
    pub y2: Option<f64>,
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub label: Option<String>,
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
    /// "rdf" (each group's |χ(R)| from the forward FT) or "xanes" (normalized
    /// μ(E) interpolated onto a common energy grid). Omitted while a map is
    /// shown, that map is kept (so `reference`, `scan`, `x` and `save` act on
    /// it); with no map yet, "rdf" is built.
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
    /// XANES rows from the flattened spectrum (true) or the plain normalized
    /// one (false, the window's default — what the legacy BL10C XANES.dat holds).
    pub flat: Option<bool>,
    /// Write the displayed map to this path as a tab-separated matrix
    /// (x first column, one named column per scan) — OriginPro-importable and
    /// re-openable via `file`.
    pub save: Option<String>,
    /// Select a map cell as a click on the map does: `scan` (1-based, series
    /// order) picks the row and `x` the nearest column; the profile plot under
    /// the map follows. Either alone keeps the other coordinate of the current
    /// selection (the first row / column when there is none).
    pub scan: Option<usize>,
    pub x: Option<f64>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct WindowArgs {
    /// Window name (see [`WINDOW_NAMES`]): "open_data", "data", "log",
    /// "plot_data", "edit_mu", "periodic_table", "ion_chamber", "powder",
    /// "xanes", "mback", "wavelet", "ke_convert", "lcf", "pca",
    /// "time_resolved", "insitu_map", "plot_sites", "feffit_batch", "about".
    pub name: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SetThemeArgs {
    /// `true` for the dark UI theme, `false` for the light one (Settings ›
    /// Toggle dark / light). Every plot canvas follows the theme.
    pub dark: bool,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct PlotDataArgs {
    /// A button of the window: "add_groups" (Add loaded groups — overlay every
    /// session group as the graph item), "clear" (Clear All), "clear_fit"
    /// (drop the sent Feffit fit), or "send_fit" (the Feffit tab's Send to Plot
    /// Data: the last fit's data + model in the Feffit graph's space).
    pub action: Option<String>,
    /// Graph item by its combo label — "f", "norm", "f'", "f''" (xmu), "k.chi",
    /// "r.chi", "k.dat", "r.dat", "q.dat", "k.fit", "r.fit", "q.fit", "e.bkg",
    /// "k.bkg", … — selecting its file type too. Applied before `action`.
    pub graph_item: Option<String>,
    /// "Show Feffit fit (data + model)" for a sent fit.
    pub show_fit: Option<bool>,
    /// "Change BG color": swap the canvas against the UI theme.
    pub swap_bg: Option<bool>,
    /// G_kweight (0–3) applied to k-space curves.
    pub kweight: Option<i32>,
    /// Stacking offset between curves (data units, 0–5).
    pub stack: Option<f64>,
    /// "Same height": scale every curve to one peak-to-peak height.
    pub same_height: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct FeffitGraphArgs {
    /// The Feffit tab's graph space: "k", "r", "q" or "kq" (K+Q overlay).
    pub space: String,
    /// The complex-χ component for R / q: "re", "im", "mag" (Am) or "pha" (Ph).
    pub part: Option<String>,
    /// Show the draggable fit-range band — kmin/kmax on k / q, rmin/rmax on R
    /// (the "Show k/R range" checkbox); `graph` action "drag_window" then
    /// moves it.
    pub show_range: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct XanesArgs {
    /// The spectrum to work on, by group label (must be normalized).
    pub spectrum: Option<String>,
    /// Which array: "norm" or "flat".
    pub array: Option<String>,
    /// Region low bound (eV) for the peak / valley / half-max searches.
    pub rlo: Option<f64>,
    /// Region high bound (eV).
    pub rhi: Option<f64>,
    /// A cursor button: "peak", "valley", "half_step", "half_max", "x_at_y"
    /// (needs `y`), or "clear" (Clear cursors). Applied after the settings.
    pub cursor: Option<String>,
    /// The target level for the "x_at_y" cursor.
    pub y: Option<f64>,
    /// "Arctangent subtraction" on / off.
    pub arctan: Option<bool>,
    /// Arctangent amplitude.
    pub amp: Option<f64>,
    /// Arctangent center (eV).
    pub center: Option<f64>,
    /// Arctangent sigma (eV, 0.05–500).
    pub sigma: Option<f64>,
    /// Linear-baseline slope.
    pub slope: Option<f64>,
    /// Linear-baseline intercept.
    pub intercept: Option<f64>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct MbackArgs {
    /// The spectrum to normalize, by group label (needs μ(E)). Picking one
    /// re-guesses the element and edge from its E₀, like the combo.
    pub spectrum: Option<String>,
    /// Element atomic number Z (1–98), after the guess.
    pub z: Option<u16>,
    /// Edge: "K", "L1", "L2" or "L3", after the guess.
    pub edge: Option<String>,
    /// Press "Run MBACK" (default true).
    pub run: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct LcfArgs {
    /// The target spectrum, by group label (must be normalized). Picking one
    /// reseeds the E range from its span.
    pub target: Option<String>,
    /// The standards to check, by group label — exactly this set (others are
    /// unchecked).
    pub standards: Option<Vec<String>>,
    /// Which array: "norm" or "flat".
    pub array: Option<String>,
    /// Fit range low bound (eV).
    pub emin: Option<f64>,
    /// Fit range high bound (eV).
    pub emax: Option<f64>,
    /// Press "Run LCF" (default true).
    pub run: Option<bool>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct PcaArgs {
    /// The training spectra to check, by group label — exactly this set.
    pub training: Option<Vec<String>>,
    /// Which array: "norm" or "flat".
    pub array: Option<String>,
    /// E range low bound (eV).
    pub emin: Option<f64>,
    /// E range high bound (eV).
    pub emax: Option<f64>,
    /// "train" (Train on the checked set) or "fit" (Fit unknown: needs a
    /// trained model and `unknown`). Applied after the settings.
    pub action: Option<String>,
    /// Number of components for the reconstruction (1 … trained components).
    pub ncomps: Option<usize>,
    /// The unknown spectrum to reconstruct, by group label.
    pub unknown: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct WaveletArgs {
    /// k-weight (0–3).
    pub kweight: Option<i32>,
    /// R min (Å, 0.1–4).
    pub rmin: Option<f64>,
    /// R max (Å, 1–10).
    pub rmax: Option<f64>,
    /// R samples (32–512).
    pub nr: Option<usize>,
    /// Morlet σ (0.2–8).
    pub bandwidth: Option<f64>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct EditMuArgs {
    /// The Edit μ(E) button, on the current group: "find" (mark automatic
    /// glitch candidates), "remove_auto" (remove them), "remove_point"
    /// (`energy`), "remove_range" (`side` above / below / between, `e1`, `e2`),
    /// "trim" (`lo`, `hi`), "smooth" (`sigma`, `form`), or "undo". The batch
    /// "Remove in N checked groups" is the `deglitch` tool with apply = true.
    pub action: String,
    /// Spike threshold in robust σ for find / remove_auto (default 10).
    pub nsig: Option<f64>,
    /// Energy (eV) of the point to remove (remove_point).
    pub energy: Option<f64>,
    /// Range side for remove_range: "above", "below" or "between" (default above).
    pub side: Option<String>,
    /// Reference energy (eV) for remove_range.
    pub e1: Option<f64>,
    /// Second reference energy (eV) for remove_range "between".
    pub e2: Option<f64>,
    /// Trim window low bound (eV).
    pub lo: Option<f64>,
    /// Trim window high bound (eV).
    pub hi: Option<f64>,
    /// Smoothing width σ (eV, default 1).
    pub sigma: Option<f64>,
    /// Smoothing line shape: "lorentzian" (default) or "gaussian".
    pub form: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SiteArg {
    /// Element symbol.
    pub element: String,
    /// Fractional x.
    pub x: f64,
    /// Fractional y.
    pub y: f64,
    /// Fractional z.
    pub z: f64,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct AtomsArgs {
    /// TITLE for the feff.inp.
    pub title: Option<String>,
    /// Lattice a (Å).
    pub a: Option<f64>,
    /// Lattice b (Å).
    pub b: Option<f64>,
    /// Lattice c (Å).
    pub c: Option<f64>,
    /// Lattice α (°).
    pub alpha: Option<f64>,
    /// Lattice β (°).
    pub beta: Option<f64>,
    /// Lattice γ (°).
    pub gamma: Option<f64>,
    /// Absorption edge: "K", "L1", "L2" or "L3".
    pub edge: Option<String>,
    /// International space-group number (1–230; 1 = P1, no expansion).
    pub space_group: Option<u32>,
    /// Cluster radius (Å, 1–12).
    pub cluster_size: Option<f64>,
    /// Index of the absorber among `sites` (0-based).
    pub absorber: Option<usize>,
    /// The asymmetric unit — replaces the site table when given.
    pub sites: Option<Vec<SiteArg>>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct RunFeffArgs {
    /// FEFF backend: "feff10" (in-process, default) or "feff8l" (external).
    pub backend: Option<String>,
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
            (Some(p), None) => vec![native_path(&p)],
            (None, Some(v)) if !v.is_empty() => v.iter().map(|p| native_path(p)).collect(),
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
        description = "Switch the plotted graph type (XMU + BKG, *K.CHI, *R.CHI, *Q.CHI, derivatives, Normalize), toggle the all-groups overlay, and/or `show_range` (the draggable k/R FT-window band) — what screenshot then shows."
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

    #[tool(
        description = "Make a pointer gesture on a graph without the mouse — zoom in / out, box zoom, pan (drag), zoom back / reset, a click on the data (pins the nearest-point readout), an FT-window band drag, or a legend click — on the main plot or a pop-up window's plot, through the same code the mouse runs; reports the view (axis limits, pinned pick, band). `action` \"view\" only reports."
    )]
    async fn graph(
        &self,
        Parameters(a): Parameters<GraphArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::Graph {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Open a pop-up window by name (the Tools menu / top-bar / Log / Data windows), or raise it when it is already open. Names: open_data, data, log, plot_data, edit_mu, periodic_table, ion_chamber, powder, xanes, mback, wavelet, ke_convert, lcf, pca, time_resolved, insitu_map, plot_sites, feffit_batch, about."
    )]
    async fn open_window(
        &self,
        Parameters(a): Parameters<WindowArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::OpenWindow {
            name: a.name,
            reply,
        })
        .await
    }

    #[tool(description = "Close a pop-up window by name (same names as open_window).")]
    async fn close_window(
        &self,
        Parameters(a): Parameters<WindowArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::CloseWindow {
            name: a.name,
            reply,
        })
        .await
    }

    #[tool(
        description = "Switch the UI theme (Settings › Toggle dark / light): dark = true or false. Every plot canvas follows."
    )]
    async fn set_theme(
        &self,
        Parameters(a): Parameters<SetThemeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::SetTheme {
            dark: a.dark,
            reply,
        })
        .await
    }

    #[tool(
        description = "Drive the Plot Data window (opens it): `action` add_groups / clear / clear_fit / send_fit, `graph_item` by combo label (k.chi, r.chi, norm, …), and the display controls show_fit, swap_bg, kweight, stack, same_height. Only the given fields change."
    )]
    async fn plot_data(
        &self,
        Parameters(a): Parameters<PlotDataArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::PlotData {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Select the Feffit tab's graph: space k / r / q / kq and, for r / q, the component re / im / mag / pha; `show_range` toggles the draggable fit-range band. The Feffit graph replots (and send_fit uses this space)."
    )]
    async fn feffit_graph(
        &self,
        Parameters(a): Parameters<FeffitGraphArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::FeffitGraph { args: a, reply })
            .await
    }

    #[tool(
        description = "Drive the XANES tools window (opens it): pick the spectrum / array / region, the arctangent-subtraction settings, and press a cursor button (peak, valley, half_step, half_max, x_at_y, clear). Returns the cursor table and the arctangent centroid."
    )]
    async fn xanes(
        &self,
        Parameters(a): Parameters<XanesArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::Xanes {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Drive the MBACK / NEXAFS normalization window (opens it): pick the spectrum, element Z and edge, and run MBACK. Returns the MBACK and polynomial edge steps."
    )]
    async fn mback(
        &self,
        Parameters(a): Parameters<MbackArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::Mback { args: a, reply })
            .await
    }

    #[tool(
        description = "Drive the LCF window (opens it): pick the target, the standards, the array and the E range, and run the linear-combination fit. Returns the weights, their sum, the R-factor and reduced χ²."
    )]
    async fn lcf(&self, Parameters(a): Parameters<LcfArgs>) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::Lcf {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Drive the PCA window (opens it): pick the training set, the array and the E range, then action \"train\" (returns the suggested component count and the scree table) or \"fit\" (reconstruct `unknown` with `ncomps` components)."
    )]
    async fn pca(&self, Parameters(a): Parameters<PcaArgs>) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::Pca {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Set the wavelet window's parameters (k-weight, R range, R samples, Morlet σ) and press Compute on the current group's χ(k); opens the window with the |W(k,R)| map."
    )]
    async fn wavelet(
        &self,
        Parameters(a): Parameters<WaveletArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::Wavelet { args: a, reply })
            .await
    }

    #[tool(
        description = "Press an Edit μ(E) button on the current group: find / remove_auto (automatic glitches at `nsig` σ), remove_point (`energy`, or the graph's pinned pick when omitted), remove_range (`side`, `e1`, `e2`), trim (`lo`, `hi`), smooth (`sigma`, `form`), or undo. Each edit re-reduces the current group as a live preview (E₀ and k-point count in the reply; no χ file is written) — run_autobk writes the files."
    )]
    async fn edit_mu(
        &self,
        Parameters(a): Parameters<EditMuArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::EditMu {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Set the Atoms tab's cell (lattice, edge, space group, cluster size, absorber, sites — only the given fields change) and press Build feff.inp; the Feff tab then holds the input. Returns the build status."
    )]
    async fn build_feff_inp(
        &self,
        Parameters(a): Parameters<AtomsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::BuildFeffInp {
            args: Box::new(a),
            reply,
        })
        .await
    }

    #[tool(
        description = "Press Run FEFF on the Feff tab's feff.inp with the chosen backend (feff10 in-process by default, or feff8l) and wait for it; returns the work folder and the feffNNNN.dat files written."
    )]
    async fn run_feff(
        &self,
        Parameters(a): Parameters<RunFeffArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.roundtrip(|reply| McpRequest::RunFeff {
            backend: a.backend,
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
    instructions = "Remote control of a running XAFSView (XAFS data reduction + FEFF fitting GUI). Typical flow: load_data → set_params → run_autobk → run_fit → get_results; screenshot shows the main window. open_window / close_window / set_theme / plot_data / feffit_graph / xanes / mback / lcf / pca / wavelet / edit_mu / build_feff_inp / run_feff drive the pop-up windows and the tabs without the mouse; graph makes the mouse gestures on a plot (zoom, box zoom, pan, click, band drag, legend click)."
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

/// A filesystem path from an MCP string argument, in the platform's own
/// spelling. Clients hand over paths as they hold them — on Windows typically
/// with forward slashes (`C:/data/x.dat`) — and that spelling would otherwise
/// travel into the session verbatim: the Folders tab showed such a Data folder
/// with slashes. Re-joining the parsed components uses the native separator
/// (and drops `.` components); on Unix this is the identity. Every MCP argument
/// that names a file goes through here.
pub fn native_path(s: &str) -> std::path::PathBuf {
    std::path::Path::new(s).components().collect()
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

#[cfg(test)]
mod tests {
    use super::native_path;
    use std::path::PathBuf;

    #[test]
    fn native_path_respells_separators_and_drops_dot_components() {
        // Compare spellings, not `Path` equality (which is component-wise and
        // would call `C:/x` and `C:\x` equal).
        let (win, unix) = (native_path("C:/data/./x.dat"), native_path("/d/./x.dat"));
        if cfg!(windows) {
            // The forward slashes an MCP client sends come out native.
            assert_eq!(win.as_os_str(), r"C:\data\x.dat");
            assert_eq!(unix.as_os_str(), r"\d\x.dat");
        } else {
            // Nothing to respell; the `.` component still goes.
            assert_eq!(unix.as_os_str(), "/d/x.dat");
            assert_eq!(native_path("rel/x.dat"), PathBuf::from("rel/x.dat"));
        }
    }
}
