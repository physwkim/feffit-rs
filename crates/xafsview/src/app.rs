//! The application shell: window layout, the tab bar, the menu bar, and the
//! shared `rsplot` plot. Individual tabs fill in their panels in later phases;
//! Phase 0 wires the skeleton and makes the **Folders** tab functional.

use eframe::egui;
use feffit::xasdata::{ColumnFile, Folders, Session, XasGroup};
use rsplot::widget::form;

use crate::analysis_ui::{LcfWindow, PcaWindow};
use crate::atoms_ui::{AtomsAction, AtomsTab, FeffAction, FeffTab, PlotSitesWindow};
use crate::calc_ui::{IonChamberWindow, KeConvertWindow, PeriodicTableWindow, PowderWindow};
use crate::clean_ui::{CleanAction, EditXmuState};
use crate::feffit_batch::{BatchAction, FeffitBatch};
use crate::feffit_ui::{FeffitAction, FeffitPlot, FeffitUi, FitMode};
use crate::import::ImportState;
use crate::insitu_ui::{InsituAction, InsituWindow};
use crate::mback_ui::MbackWindow;
use crate::mcp;
use crate::open_data::OpenDataBrowser;
use crate::plot_data::PlotDataWindow;
use crate::reduce_ui::{ControlsChange, GraphType, ReductionUi, RefitKey, TheoryStd};
use crate::timeres_ui::TimeResolvedWindow;
use crate::wavelet::{WaveletAction, WaveletWindow, morlet_cwt};
use crate::xanes_ui::XanesWindow;

/// The top-level tabs, mirroring XAFSView's tab strip. Most are placeholders in
/// Phase 0 and are filled in by their respective phases.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    /// Working-directory configuration — the entry point of the analysis flow.
    Folders,
    /// Pre-edge / normalize / AUTOBK background removal (P1–P2).
    Autobk,
    /// FEFFIT path fitting (P4) — kept next to Autobk because reduce→fit is the
    /// every-session path; the structure-building tabs that feed it (Atoms/Feff)
    /// run less often and sit after it.
    Feffit,
    /// FEFFIT text/parameter view (P4).
    FeffitTxt,
    /// Crystal → feff.inp (P9) — only when FEFF paths must be (re)generated.
    Atoms,
    /// Edit feff.inp / run FEFF (P9) — only when FEFF paths must be (re)generated.
    Feff,
}

impl Tab {
    /// All tabs in strip order — the EXAFS analysis flow, left to right:
    /// set up folders → reduce data → fit; the structure-building tabs
    /// (Atoms/Feff), used only when FEFF paths must be regenerated, follow the fit.
    const ALL: [Tab; 6] = [
        Tab::Folders,
        Tab::Autobk,
        Tab::Feffit,
        Tab::FeffitTxt,
        Tab::Atoms,
        Tab::Feff,
    ];

    /// The label shown on the tab button.
    fn label(self) -> &'static str {
        match self {
            Tab::Autobk => "Autobk",
            Tab::Feffit => "Feffit",
            Tab::FeffitTxt => "Fit report",
            Tab::Atoms => "Atoms",
            Tab::Feff => "Feff",
            Tab::Folders => "Folders",
        }
    }
}

/// The eframe application: the data [`Session`], the active tab, and the shared
/// plot (one `rsplot::Plot1D` reused across the plot-bearing tabs).
pub struct XafsViewApp {
    session: Session,
    tab: Tab,
    plot: crate::plot::Plot,
    /// The in-progress data import (column → role mapping), if a raw file is open.
    import: Option<ImportState>,
    /// Reduction (normalize/autobk/FT) parameters and the active graph type.
    reduction: ReductionUi,
    /// Autobk-graph view mode: when `true`, overlay every loaded group's
    /// principal curve for the selected graph type (the current group in full,
    /// the others for comparison) instead of only the current group. Turned on by
    /// the `Multiple_data` batch actions so a whole batch is shown at once; a
    /// "Show all groups" checkbox by the group selector is its visible owner.
    show_all_groups: bool,
    /// Autobk μ(E)+background view: when `false`, the Autobk-computed background
    /// μ₀(E) is hidden so only the raw μ(E) data files are shown. The per-group
    /// μ(E) comparison overlays (under [`show_all_groups`]) are never touched by
    /// this toggle — they are data files too. Only meaningful for
    /// [`GraphType::MuBkg`] (the only view that draws a separate background), so
    /// its "Show background" checkbox is shown only there.
    ///
    /// [`show_all_groups`]: Self::show_all_groups
    show_bkg: bool,
    /// Waterfall offset (data units) between the group comparison curves under
    /// [`show_all_groups`], mirroring Plot Data's "Stacking": the focused group
    /// sits at the baseline and the j-th comparison overlay is lifted by
    /// `(j+1) · stack`. `0.0` leaves every curve at its true level.
    ///
    /// [`show_all_groups`]: Self::show_all_groups
    stack: f64,
    /// FEFFIT tab state: paths, variables, transform, and last fit result.
    feffit: FeffitUi,
    /// Label of the group the last main-tab Feffit fit was run on, so "Send to
    /// Plot Data" names the fitted group even after the current group changes.
    feffit_fit_group: Option<String>,
    /// Every fit from the last Feffit tab "Run" (`(group label, plot)`), so a
    /// multi-group run can overlay all fits on the Feffit graph the way AUTOBK's
    /// "Show all groups" overlays every reduced group. The focused group's fit is
    /// drawn prominently by the base curves and skipped in the overlay.
    feffit_run_overlays: Vec<(String, FeffitPlot)>,
    /// Whether the last Feffit tab run's fits are overlaid on the graph — the
    /// Feffit-tab twin of [`show_all_groups`]. A multi-group Run turns it on; a
    /// "Show all fits" checkbox by the Run button is its visible owner.
    feffit_show_all_fits: bool,
    /// The Feffit batch state (shown in the detached "Feffit — Batch" window,
    /// opened from the Feffit tab's "Batch…" button): runs the tab's config
    /// against several checked groups, tabulates and saves the results.
    feffit_batch: FeffitBatch,
    /// The detached "Open data" loader: the in-app raw-scan browser + two-pane
    /// staging that replaces both the OS file dialog and the old batch picker.
    /// Only the browse/staging state lives here; the app hosts the reading-format
    /// chooser ([`import`](Self::import)) and the Autobk Start build around it.
    open_data: OpenDataBrowser,
    /// Whether the detached "Data" window (the loaded-group manager: the list,
    /// "Add data files…", and "Clear all") is shown. The Autobk sidebar keeps
    /// only a compact current-group combo; the full list lives here.
    data_window_open: bool,
    /// Scroll the Data window's list to the current group's row: set whenever
    /// the window is opened, cleared once that row has been brought into view
    /// (a 66-scan list opened on scan 40 otherwise showed scans 1–12).
    data_window_reveal: bool,
    /// Last plain-clicked Run-checkbox row in the Data window: the anchor a
    /// shift-click extends from ([`widgets::select_index_list`]'s convention).
    /// Cleared when rows are removed, which would shift the indices under it.
    ///
    /// [`widgets::select_index_list`]: crate::widgets::select_index_list
    data_check_anchor: Option<usize>,
    /// Whether the detached "Feffit — Batch" window is shown. The Feffit tab
    /// keeps only a "Batch…" button (left controls); the batch panel itself —
    /// an occasional bulk op — opens on demand so the central plot keeps the
    /// full width instead of a permanent right panel.
    feffit_batch_open: bool,
    /// The Edit-μ(E) window (deglitch / trim / smooth) state.
    edit_xmu: EditXmuState,
    /// The Morlet wavelet-transform window (|W(k,R)| heatmap of χ(k)).
    wavelet: WaveletWindow,
    /// The Plot Data window: a detached OS window with its own graph and a
    /// right-hand control column, overlaying processed files loaded from disk
    /// (opened from the top bar or Feffit's "Send to Plot Data").
    plot_data: PlotDataWindow,
    /// The linear-combination-fitting window (its own plot).
    lcf: LcfWindow,
    /// The principal-component-analysis window (its own plot).
    pca: PcaWindow,
    /// The XANES-tools window: edge/peak cursors + arctangent subtraction.
    xanes: XanesWindow,
    /// The MBACK / NEXAFS normalization window (its own plot).
    mback: MbackWindow,
    /// The periodic-table + atom-data window.
    periodic: PeriodicTableWindow,
    /// The ion-chamber / gas-absorption calculator.
    ion_chamber: IonChamberWindow,
    /// The powder-weight calculator.
    powder: PowderWindow,
    /// The k ↔ E conversion calculator.
    ke_convert: KeConvertWindow,
    /// The Extract-XAS-measured-time window (time-resolved series timing).
    time_resolved: TimeResolvedWindow,
    /// The In-situ map window (scan-series image, the BL10C IMAGE view).
    insitu: InsituWindow,
    /// The Atoms tab state (crystal cell → feff.inp).
    atoms_tab: AtomsTab,
    /// The Feff tab state (edit feff.inp / run FEFF).
    feff_tab: FeffTab,
    /// The 3D Plot Sites cluster viewer (its own `rsplot` scene).
    plot_sites: PlotSitesWindow,
    /// The shared `feff.inp` text, written by Atoms and edited/run by Feff.
    feff_inp: String,
    /// Pre-edit group snapshots for undoing cleanup edits (most recent last).
    clean_undo: Vec<XasGroup>,
    /// The `(current group, parameters)` the last committed-edit refit ran
    /// with, so a commit that changed nothing (field entered and left) skips
    /// the recompute. `None` until the first such refit.
    last_refit_key: Option<(Option<usize>, RefitKey)>,
    /// The tab shown on the previous frame, to detect tab switches for replot.
    last_tab: Tab,
    /// Whether the About window (Help menu) is open.
    about_open: bool,
    /// The status strip's line plus the cumulative session message log — every
    /// user-visible outcome funnels through it (see [`crate::status_log`]).
    status: crate::status_log::StatusLog,
    /// The single background-job slot for the batch operations (Multiple
    /// AUTOBK, batch μ(E), Feffit runs) — see [`crate::jobs`].
    jobs: crate::jobs::JobRunner<Self>,
    /// Working value for the Settings "Font size" slider (the egui zoom factor).
    /// The slider edits this freely; the live zoom is applied only when the drag
    /// ends, so the UI doesn't reflow under the cursor on every drag tick.
    ui_scale: f32,
    /// Mirror of the active visuals' dark/light choice, kept by the Settings
    /// toggle so [`Prefs::capture`] can persist it (eframe's `save` has no
    /// `Context` to read the live visuals from).
    dark_mode: bool,
    /// The embedded MCP server when enabled (`--mcp` or the Settings toggle):
    /// the listener handle plus the channel its tool requests arrive on.
    /// Dropping it (toggle off) stops the server. See [`crate::mcp`].
    mcp: Option<(mcp::McpServer, std::sync::mpsc::Receiver<mcp::McpRequest>)>,
    /// Port the Settings toggle (re)starts the MCP server on: the `--mcp` port
    /// when one was given, else [`mcp::DEFAULT_PORT`].
    mcp_port: u16,
    /// An MCP reply parked while the request's background job runs (`run_fit`,
    /// batch `load_data` — the single job slot holds at most one); answered by
    /// the job's finisher ([`finish_feffit`](Self::finish_feffit) /
    /// [`finish_build_xmu`](Self::finish_build_xmu)) or the job-failure path.
    mcp_job_reply: Option<mcp::TextReply>,
    /// MCP `screenshot` replies awaiting the next [`egui::Event::Screenshot`].
    mcp_shot_replies: Vec<mcp::PngReply>,
    /// Armed by the In-situ window's "Re-reduce & build" when its reduction
    /// job starts; [`finish_reduction`](Self::finish_reduction) consumes it to
    /// rebuild the map from the freshly reduced groups (cleared if the job
    /// dies, so a later unrelated reduction doesn't rebuild unasked).
    insitu_rebuild_after_reduce: bool,
}

/// Default overall UI zoom (scales text *and* spacing together). Set >1.0 so
/// the app reads larger than egui's stock size out of the box; adjustable live
/// from the Settings menu. egui's `Context` holds the live value, so it persists
/// across frames within a session without any extra app state.
const DEFAULT_UI_SCALE: f32 = 1.2;

/// What survives a restart, through eframe's storage (written on exit and on
/// its periodic autosave): the Settings-menu choices and the working folders.
/// The MCP toggle is deliberately left out — silently re-opening a localhost
/// server on launch would be surprising, so `--mcp` / the Settings checkbox
/// opt in per run. Window geometry persists separately via eframe itself.
#[derive(serde::Serialize, serde::Deserialize)]
struct Prefs {
    ui_scale: f32,
    dark: bool,
    sub_base: Option<std::path::PathBuf>,
    data_dir: Option<std::path::PathBuf>,
    autobk_dir: Option<std::path::PathBuf>,
    feffit_dir: Option<std::path::PathBuf>,
    atoms_dir: Option<std::path::PathBuf>,
    results_dir: Option<std::path::PathBuf>,
}

/// eframe storage key for [`Prefs`].
const PREFS_KEY: &str = "prefs";

impl Prefs {
    fn capture(app: &XafsViewApp) -> Self {
        let f = &app.session.folders;
        Self {
            ui_scale: app.ui_scale,
            dark: app.dark_mode,
            sub_base: f.sub_base.clone(),
            data_dir: f.data_dir.clone(),
            autobk_dir: f.autobk_dir.clone(),
            feffit_dir: f.feffit_dir.clone(),
            atoms_dir: f.atoms_dir.clone(),
            results_dir: f.results_dir.clone(),
        }
    }

    fn apply_folders(&self, f: &mut Folders) {
        f.sub_base = self.sub_base.clone();
        f.data_dir = self.data_dir.clone();
        f.autobk_dir = self.autobk_dir.clone();
        f.feffit_dir = self.feffit_dir.clone();
        f.atoms_dir = self.atoms_dir.clone();
        f.results_dir = self.results_dir.clone();
    }
}

/// Widen egui's stock spacing to desktop-form density (the Qt layout metrics
/// the original XAFSView forms were designed around): egui's defaults are
/// tuned for compact dev tooling — 18 px interact rows, 1 px button padding,
/// 3 px between rows — which reads cramped next to a QFormLayout. Values are
/// pre-zoom; [`DEFAULT_UI_SCALE`] scales them with the text. Applied to every
/// theme so the light/dark toggle keeps the same metrics.
fn apply_form_density(ctx: &egui::Context) {
    ctx.all_styles_mut(|style| {
        let s = &mut style.spacing;
        s.item_spacing = egui::vec2(8.0, 6.0);
        s.button_padding = egui::vec2(10.0, 4.0);
        s.interact_size.y = 24.0;
        s.icon_width = 16.0;
        s.icon_spacing = 6.0;
        s.window_margin = egui::Margin::same(8);
        s.menu_margin = egui::Margin::same(8);
        // Combo popups (group / spectrum / target pickers) list every loaded
        // scan; egui's default caps the popup at 200 px, about seven rows, so a
        // 66-scan series was mostly behind a scrollbar.
        s.combo_height = 480.0;
    });
}

/// Refusal message when a batch trigger fires while the single job slot is
/// taken ([`crate::jobs::JobRunner::start`] returned `false`).
const JOB_BUSY: &str = "A background job is already running — wait for it to finish.";

/// Colour of the auto-deglitch candidate markers on the main plot (orange, so
/// they read as "flagged" rather than as data).
const GLITCH_MARKER: egui::Color32 = egui::Color32::from_rgb(0xff, 0x8c, 0x00);

/// One Feffit-job target's outcome, from worker to apply: the group's index and
/// label, with `Ok((run summary, the fitted config clone))` or `Err(message)`.
type FeffitJobFit = (usize, String, Result<(String, FeffitUi), String>);

impl XafsViewApp {
    /// Build the app. Requires the wgpu render state (see [`main`](crate::main)).
    /// `mcp_port` starts the embedded MCP server (`--mcp [port]`).
    pub fn new(cc: &eframe::CreationContext<'_>, mcp_port: Option<u16>) -> Self {
        // Install the glyph-fallback font before anything draws, so both the egui
        // UI and rsplot axis labels can render the superscript/subscript math
        // glyphs the default font lacks (Å⁻¹, kʷ, χ²ᵣ).
        crate::fonts::install_fallback(&cc.egui_ctx);

        // Restore the persisted Settings choices before anything draws; first
        // run (or no storage) falls back to the defaults.
        let prefs: Option<Prefs> = cc.storage.and_then(|s| eframe::get_value(s, PREFS_KEY));
        let ui_scale = prefs.as_ref().map_or(DEFAULT_UI_SCALE, |p| p.ui_scale);
        let dark_mode = prefs.as_ref().is_none_or(|p| p.dark);
        cc.egui_ctx.set_zoom_factor(ui_scale);
        if !dark_mode {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
        }
        apply_form_density(&cc.egui_ctx);

        let mut session = Session::new();
        if let Some(p) = &prefs {
            p.apply_folders(&mut session.folders);
        }

        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must use the wgpu renderer (NativeOptions.renderer = Wgpu)");

        let mut plot = crate::plot::Plot::new(render_state, 0);
        plot.set_graph_title("XAFSView");
        plot.set_graph_x_label("Energy (eV)");
        plot.set_graph_y_label("μ(E)", rsplot::YAxis::Left);

        // A placeholder curve so the GPU plot path is exercised before any data
        // is loaded; replaced by the active group's spectrum from Phase 1 on.
        let x: Vec<f64> = (0..400).map(|i| 8900.0 + i as f64 * 0.5).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&e| {
                // A crude arctangent edge at ~8979 eV, just for display.
                0.5 + 0.5 * ((e - 8979.0) / 8.0).tanh()
            })
            .collect();
        plot.add_curve_with_legend(&x, &y, crate::plot::BLUE, "demo edge");

        let plot_data = PlotDataWindow::new(render_state);
        let lcf = LcfWindow::new(render_state);
        let pca = PcaWindow::new(render_state);
        let xanes = XanesWindow::new(render_state);
        let mback = MbackWindow::new(render_state);
        let insitu = InsituWindow::new(render_state);
        let periodic = PeriodicTableWindow::default();
        let ion_chamber = IonChamberWindow::default();
        let powder = PowderWindow::default();
        let plot_sites = PlotSitesWindow::new(render_state);

        let mut app = Self {
            session,
            // Open on Folders — the flow entry point where the working
            // directories are set before any data is loaded.
            tab: Tab::Folders,
            plot,
            import: None,
            reduction: ReductionUi::default(),
            show_all_groups: false,
            show_bkg: true,
            stack: 0.0,
            feffit: FeffitUi::default(),
            feffit_fit_group: None,
            feffit_run_overlays: Vec::new(),
            feffit_show_all_fits: false,
            feffit_batch: FeffitBatch::default(),
            open_data: OpenDataBrowser::default(),
            data_window_open: false,
            data_window_reveal: false,
            data_check_anchor: None,
            feffit_batch_open: false,
            edit_xmu: EditXmuState::default(),
            wavelet: WaveletWindow::default(),
            plot_data,
            lcf,
            pca,
            xanes,
            mback,
            periodic,
            ion_chamber,
            powder,
            ke_convert: KeConvertWindow::default(),
            time_resolved: TimeResolvedWindow::default(),
            insitu,
            atoms_tab: AtomsTab::default(),
            feff_tab: FeffTab::default(),
            plot_sites,
            feff_inp: String::new(),
            clean_undo: Vec::new(),
            last_refit_key: None,
            last_tab: Tab::Atoms,
            about_open: false,
            status: Default::default(),
            jobs: Default::default(),
            ui_scale,
            dark_mode,
            mcp: None,
            mcp_port: mcp_port.unwrap_or(mcp::DEFAULT_PORT),
            mcp_job_reply: None,
            mcp_shot_replies: Vec::new(),
            insitu_rebuild_after_reduce: false,
        };
        if mcp_port.is_some() {
            app.start_mcp(&cc.egui_ctx);
        }
        app
    }

    /// Start the embedded MCP server on [`mcp_port`](Self::mcp_port), reporting
    /// the endpoint (or the bind failure) on the status line.
    fn start_mcp(&mut self, ctx: &egui::Context) {
        match mcp::start(self.mcp_port, ctx.clone()) {
            Ok((server, rx)) => {
                self.status.info(format!(
                    "MCP server listening on http://127.0.0.1:{}/mcp",
                    server.port
                ));
                self.mcp = Some((server, rx));
            }
            Err(e) => self.status.error(format!(
                "MCP server failed to start on port {}: {e}",
                self.mcp_port
            )),
        }
    }

    /// Resolve parked MCP screenshot requests when the capture event arrives,
    /// then handle every queued tool request. Runs at the top of each frame.
    fn poll_mcp(&mut self, ctx: &egui::Context) {
        if !self.mcp_shot_replies.is_empty() {
            let image = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(image) = image {
                let png = mcp::encode_png(&image);
                for reply in self.mcp_shot_replies.drain(..) {
                    let _ = reply.send(png.clone());
                }
            }
        }
        let Some((_, rx)) = &self.mcp else { return };
        let requests: Vec<mcp::McpRequest> = rx.try_iter().collect();
        for req in requests {
            self.handle_mcp_request(req, ctx);
        }
    }

    /// Handle one MCP tool request against the live state — through the same
    /// methods the GUI buttons call, so an agent and a user produce identical
    /// sessions (including the written output files).
    fn handle_mcp_request(&mut self, req: mcp::McpRequest, ctx: &egui::Context) {
        match req {
            mcp::McpRequest::LoadData { paths, reply } => self.mcp_load_data(paths, reply),
            mcp::McpRequest::SetParams { args, reply } => {
                let _ = reply.send(self.apply_mcp_params(*args));
            }
            mcp::McpRequest::AddPath { args, reply } => {
                let _ = reply.send(self.mcp_add_path(*args));
            }
            mcp::McpRequest::SelectGroups { indices, reply } => {
                let n = self.session.groups.len();
                if let Some(&bad) = indices.iter().find(|&&i| i >= n) {
                    let _ = reply.send(Err(format!(
                        "group index {bad} out of range ({n} group(s) loaded)"
                    )));
                } else {
                    self.session.selected = indices.iter().copied().collect();
                    if let Some(&first) = indices.first() {
                        self.session.current = Some(first);
                    }
                    self.plot_data.mark_dirty();
                    self.replot_active_tab();
                    let mut sorted = indices;
                    sorted.sort_unstable();
                    let _ = reply.send(Ok(format!(
                        "checked {sorted:?}; current group = {:?}",
                        self.session.current
                    )));
                }
            }
            mcp::McpRequest::RemoveGroups { indices, reply } => {
                let n = self.session.groups.len();
                if let Some(&bad) = indices.iter().find(|&&i| i >= n) {
                    let _ = reply.send(Err(format!(
                        "group index {bad} out of range ({n} group(s) loaded)"
                    )));
                } else {
                    // Descending order so each removal leaves the smaller
                    // indices still pointing at the groups the caller named.
                    let mut sorted = indices;
                    sorted.sort_unstable();
                    sorted.dedup();
                    for &i in sorted.iter().rev() {
                        self.session.remove_group(i);
                    }
                    self.data_check_anchor = None;
                    self.plot_data.mark_dirty();
                    self.replot_active_tab();
                    let _ = reply.send(Ok(format!(
                        "removed {} group(s); {} remain; current group = {:?}",
                        sorted.len(),
                        self.session.groups.len(),
                        self.session.current
                    )));
                }
            }
            mcp::McpRequest::RemovePaths { indices, reply } => {
                let n = self.feffit.path_count();
                if let Some(&bad) = indices.iter().find(|&&i| i >= n) {
                    let _ = reply.send(Err(format!(
                        "path index {bad} out of range ({n} path(s) loaded)"
                    )));
                } else {
                    let mut sorted = indices;
                    sorted.sort_unstable();
                    sorted.dedup();
                    let removed: Vec<String> = sorted
                        .iter()
                        .rev()
                        .filter_map(|&i| self.feffit.remove_path(i))
                        .collect();
                    let _ = reply.send(Ok(format!(
                        "removed path(s) {:?}; {} remain",
                        removed,
                        self.feffit.path_count()
                    )));
                }
            }
            mcp::McpRequest::SetGraph { args, reply } => {
                let _ = reply.send(self.apply_mcp_graph(args));
            }
            mcp::McpRequest::SetZoom { args, reply } => {
                let _ = reply.send(self.apply_mcp_zoom(args));
            }
            mcp::McpRequest::Deglitch { nsig, apply, reply } => {
                let _ = reply.send(self.apply_mcp_deglitch(nsig, apply));
            }
            mcp::McpRequest::SetTab { tab, reply } => {
                // "Feffit_txt" stays accepted: the tab's label until the Fit
                // report rename, so existing MCP drivers keep working.
                let found = Tab::ALL
                    .into_iter()
                    .find(|t| t.label().eq_ignore_ascii_case(&tab))
                    .or_else(|| {
                        tab.eq_ignore_ascii_case("Feffit_txt")
                            .then_some(Tab::FeffitTxt)
                    });
                let _ = reply.send(match found {
                    Some(t) => {
                        self.tab = t;
                        Ok(format!("tab = {:?}", t.label()))
                    }
                    None => {
                        let labels: Vec<&str> = Tab::ALL.iter().map(|t| t.label()).collect();
                        Err(format!("unknown tab {tab:?}; one of {labels:?}"))
                    }
                });
            }
            mcp::McpRequest::RunAutobk { reply } => {
                if self.jobs.busy() {
                    let _ = reply.send(Err(JOB_BUSY.to_owned()));
                } else if self.run_reduction() {
                    // Started: park the reply for finish_reduction (or the
                    // job-failure path in the frame hook).
                    self.mcp_job_reply = Some(reply);
                } else {
                    // Refused synchronously; the status line carries the reason.
                    let _ = reply.send(Err(self.status.line().to_owned()));
                }
            }
            mcp::McpRequest::RunFit { mode, reply } => {
                if self.jobs.busy() {
                    let _ = reply.send(Err(JOB_BUSY.to_owned()));
                } else {
                    match Self::parse_fit_mode(&mode) {
                        Err(e) => {
                            let _ = reply.send(Err(e));
                        }
                        Ok(m) => {
                            self.feffit.set_fit_mode(m);
                            self.run_feffit();
                            if self.jobs.busy() {
                                // Started: park the reply for finish_feffit (or
                                // the job-failure path in the frame hook).
                                self.mcp_job_reply = Some(reply);
                            } else {
                                // run_feffit refused synchronously; the status
                                // line carries its reason.
                                let _ = reply.send(Err(self.status.line().to_owned()));
                            }
                        }
                    }
                }
            }
            mcp::McpRequest::GetResults { reply } => {
                let _ = reply.send(Ok(self.mcp_results_text()));
            }
            mcp::McpRequest::InsituMap { args, reply } => {
                let _ = reply.send(self.insitu.apply_mcp(&self.session.groups, &args));
            }
            mcp::McpRequest::Screenshot { reply } => {
                self.mcp_shot_replies.push(reply);
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
        }
    }

    /// MCP `load_data`: one path goes through the single-file open (import
    /// seeded from its header, `calc_xmu` builds + adds the group); several go
    /// through the GUI's staged batch ([`build_xmu_from_paths`]) on the job
    /// slot, with the reply parked for [`finish_build_xmu`]. Already-loaded
    /// files are skipped, like the Open-data window skips them.
    ///
    /// [`build_xmu_from_paths`]: Self::build_xmu_from_paths
    /// [`finish_build_xmu`]: Self::finish_build_xmu
    fn mcp_load_data(&mut self, paths: Vec<std::path::PathBuf>, reply: mcp::TextReply) {
        if paths.len() == 1 {
            let path = &paths[0];
            let before = self.session.groups.len();
            match ColumnFile::from_path(path) {
                Ok(cf) => {
                    // The Open-data flow with this file as representative: the
                    // import seeds the column mapping from the header, calc_xmu
                    // builds μ(E), adds the group (making it current), and
                    // writes the .xmu.
                    self.import = Some(ImportState::new(cf));
                    self.set_data_dir_from(path);
                    self.calc_xmu();
                    let line = self.status.line().to_owned();
                    let _ = reply.send(if self.session.groups.len() > before {
                        Ok(line)
                    } else {
                        Err(line)
                    });
                }
                Err(e) => {
                    let _ = reply.send(Err(format!("Open failed: {e}")));
                }
            }
            return;
        }
        if self.jobs.busy() {
            let _ = reply.send(Err(JOB_BUSY.to_owned()));
            return;
        }
        let loaded: std::collections::HashSet<std::path::PathBuf> = self
            .session
            .groups
            .iter()
            .filter_map(|g| g.filename.clone())
            .collect();
        let to_build: Vec<std::path::PathBuf> = paths
            .iter()
            .filter(|p| !loaded.contains(*p))
            .cloned()
            .collect();
        if to_build.is_empty() {
            let _ = reply.send(Ok(format!("all {} file(s) already loaded", paths.len())));
            return;
        }
        match ColumnFile::from_path(&to_build[0]) {
            Ok(cf) => {
                self.import = Some(ImportState::new(cf));
                self.set_data_dir_from(&to_build[0]);
                self.build_xmu_from_paths(to_build);
                if self.jobs.busy() {
                    self.mcp_job_reply = Some(reply);
                } else {
                    let _ = reply.send(Err(self.status.line().to_owned()));
                }
            }
            Err(e) => {
                let _ = reply.send(Err(format!("Open failed (first file): {e}")));
            }
        }
    }

    /// MCP `add_path`: load the Feff path file and add it to the Feffit model
    /// with the given wiring overrides.
    fn mcp_add_path(&mut self, args: mcp::AddPathArgs) -> Result<String, String> {
        let path = std::path::Path::new(&args.path);
        let fp = feffit::feffdat::FeffPath::from_path(path)
            .map_err(|e| format!("Path load failed: {e}"))?;
        let label = args.label.unwrap_or_else(|| {
            path.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "path".to_owned())
        });
        let (reff, nleg, degen) = (fp.feffdat.reff, fp.feffdat.nleg, fp.feffdat.degen);
        let wiring = crate::feffit_ui::PathWiring {
            degen: args.degen,
            e0: args.e0,
            ei: args.ei,
            deltar: args.deltar,
            s02: args.s02,
            n: args.n,
            sigma2: args.sigma2,
            third: args.third,
            fourth: args.fourth,
        };
        self.feffit.add_path_wired(label.clone(), fp, wiring);
        self.status.info(format!("Added Feff path {label}."));
        Ok(format!(
            "added path \"{label}\": reff = {reff:.4} Å, nleg = {nleg}, degen = {degen}"
        ))
    }

    /// MCP `set_graph`: switch the plotted graph type by its menu label and/or
    /// toggle the all-groups overlay, then replot — the same state the Graph
    /// box's combo and checkbox mutate.
    fn apply_mcp_graph(&mut self, args: mcp::SetGraphArgs) -> Result<String, String> {
        if let Some(name) = &args.graph {
            let Some(g) = GraphType::ALL
                .into_iter()
                .find(|g| g.label().eq_ignore_ascii_case(name))
            else {
                let labels: Vec<&str> = GraphType::ALL.iter().map(|g| g.label()).collect();
                return Err(format!("unknown graph type {name:?}; one of {labels:?}"));
            };
            self.reduction.graph = g;
        }
        if let Some(all) = args.show_all_groups {
            self.show_all_groups = all;
        }
        self.replot_active_tab();
        Ok(format!(
            "graph = {:?}; show all groups = {}",
            self.reduction.graph.label(),
            self.show_all_groups
        ))
    }

    /// MCP `set_zoom`: explicit display limits on the main graph (each axis
    /// pair independently), or a reset to the data bounds. Reports the
    /// resulting limits either way.
    fn apply_mcp_zoom(&mut self, args: mcp::SetZoomArgs) -> Result<String, String> {
        let pair = |name: &str, min: Option<f64>, max: Option<f64>| match (min, max) {
            (None, None) => Ok(None),
            (Some(lo), Some(hi)) => {
                if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                    return Err(format!("{name} limits must be finite with min < max"));
                }
                Ok(Some((lo, hi)))
            }
            _ => Err(format!("give both {name}min and {name}max, or neither")),
        };
        let x = pair("x", args.xmin, args.xmax)?;
        let y = pair("y", args.ymin, args.ymax)?;
        if args.reset == Some(true) {
            if x.is_some() || y.is_some() {
                return Err("reset takes no limits alongside it".to_owned());
            }
            self.plot.reset_zoom();
        } else if x.is_none() && y.is_none() {
            return Err("nothing to do: give x/y limits or reset:true".to_owned());
        }
        if let Some((lo, hi)) = x {
            self.plot.set_graph_x_limits(lo, hi);
        }
        if let Some((lo, hi)) = y {
            self.plot.set_graph_y_limits(lo, hi, rsplot::YAxis::Left);
        }
        let (xmin, xmax) = self.plot.x_limits();
        let (ymin, ymax) = self
            .plot
            .y_limits(rsplot::YAxis::Left)
            .unwrap_or((f64::NAN, f64::NAN));
        Ok(format!(
            "x = [{xmin:.6}, {xmax:.6}], y = [{ymin:.6}, {ymax:.6}]"
        ))
    }

    /// MCP `deglitch`: a dry run lists each Run-target group's auto-glitch
    /// candidates; `apply` removes them through [`Self::deglitch_targets`],
    /// the same owner the GUI batch button uses.
    fn apply_mcp_deglitch(&mut self, nsig: f64, apply: bool) -> Result<String, String> {
        if !nsig.is_finite() || nsig <= 0.0 {
            return Err(format!("nsig must be a positive number, got {nsig}"));
        }
        if apply {
            let (total, groups, targeted) = self.deglitch_targets(nsig);
            if targeted == 0 {
                return Err("no data to deglitch".to_owned());
            }
            return Ok(format!(
                "removed {total} point(s) across {groups}/{targeted} group(s) at {nsig}σ — \
                 run_autobk to recompute χ"
            ));
        }
        let targets = self.session.run_targets();
        if targets.is_empty() {
            return Err("no data to deglitch".to_owned());
        }
        let mut lines = Vec::new();
        let mut total = 0usize;
        for &idx in &targets {
            let Some(g) = self.session.groups.get(idx) else {
                continue;
            };
            let energies = feffit::xasdata::find_glitches(g, nsig);
            if energies.is_empty() {
                continue;
            }
            total += energies.len();
            let shown: Vec<String> = energies.iter().take(6).map(|e| format!("{e:.1}")).collect();
            let more = if energies.len() > shown.len() {
                ", …"
            } else {
                ""
            };
            lines.push(format!(
                "  [{idx}] {}: {} at {} eV{more}",
                g.label,
                energies.len(),
                shown.join(", "),
            ));
        }
        if total == 0 {
            return Ok(format!(
                "no candidates at {nsig}σ in {} group(s)",
                targets.len()
            ));
        }
        // A dry run is the MCP counterpart of the GUI "Find": when the current
        // group is targeted, preview its candidates as markers on the graph.
        if let Some(cur) = self.session.current
            && targets.contains(&cur)
        {
            self.auto_deglitch_find(nsig);
        }
        Ok(format!(
            "{total} candidate(s) at {nsig}σ in {}/{} group(s) (dry run — apply=true removes):\n{}",
            lines.len(),
            targets.len(),
            lines.join("\n"),
        ))
    }

    /// Apply an MCP `set_params` patch to the Autobk and Feffit configuration.
    /// Returns a summary naming every field set; on a bad field the error names
    /// it (fields listed before it in the patch are already applied).
    fn apply_mcp_params(&mut self, args: mcp::SetParamsArgs) -> Result<String, String> {
        let mut set: Vec<String> = Vec::new();
        if let Some(a) = args.autobk {
            let r = &mut self.reduction;
            if let Some(v) = a.e0 {
                r.e0 = v;
                r.e0_auto = false;
                set.push(format!("e0={v}"));
            }
            if let Some(v) = a.e0_auto {
                r.e0_auto = v;
                set.push(format!("e0_auto={v}"));
            }
            if let Some(v) = a.rbkg {
                r.rbkg = v;
                set.push(format!("rbkg={v}"));
            }
            if let Some(v) = a.kweight {
                r.kweight = v;
                set.push(format!("kweight={v}"));
            }
            if let Some(v) = a.kmin {
                r.kmin = v;
                set.push(format!("kmin={v}"));
            }
            if let Some(v) = a.kmax {
                r.kmax = v;
                set.push(format!("kmax={v}"));
            }
            if let Some(v) = a.dk {
                r.dk = v;
                set.push(format!("dk={v}"));
            }
            if let Some(v) = a.rmin {
                r.rmin = v;
                set.push(format!("rmin={v}"));
            }
            if let Some(v) = a.rmax {
                r.rmax = v;
                set.push(format!("rmax={v}"));
            }
            if let Some(v) = a.dr {
                r.dr = v;
                set.push(format!("dr={v}"));
            }
            // Any explicit pre/norm range turns auto-ranging off, like typing
            // in the GUI fields would take effect only with Auto unchecked.
            let manual_prenorm =
                a.pre1.is_some() || a.pre2.is_some() || a.norm1.is_some() || a.norm2.is_some();
            if let Some(v) = a.pre1 {
                r.pre1 = v;
                set.push(format!("pre1={v}"));
            }
            if let Some(v) = a.pre2 {
                r.pre2 = v;
                set.push(format!("pre2={v}"));
            }
            if let Some(v) = a.norm1 {
                r.norm1 = v;
                set.push(format!("norm1={v}"));
            }
            if let Some(v) = a.norm2 {
                r.norm2 = v;
                set.push(format!("norm2={v}"));
            }
            if let Some(v) = a.nnorm {
                if v > 3 {
                    return Err(format!("nnorm={v} out of range (0–3)"));
                }
                r.nnorm = v;
                set.push(format!("nnorm={v}"));
            }
            if manual_prenorm || a.nnorm.is_some() {
                r.pre_norm_auto = false;
            }
            if let Some(v) = a.pre_norm_auto {
                r.pre_norm_auto = v;
                set.push(format!("pre_norm_auto={v}"));
            }
        }
        if let Some(f) = args.feffit {
            if let Some(params) = f.params {
                for p in params {
                    self.feffit.set_global_param(
                        &p.name,
                        &p.kind,
                        p.value.unwrap_or(0.0),
                        p.expr.as_deref().unwrap_or(""),
                    )?;
                    set.push(format!("param {}={}", p.name, p.kind));
                }
            }
            {
                let ft = self.feffit.ft_mut();
                if let Some(v) = f.kmin {
                    ft.kmin = v;
                    set.push(format!("fit kmin={v}"));
                }
                if let Some(v) = f.kmax {
                    ft.kmax = v;
                    set.push(format!("fit kmax={v}"));
                }
                if let Some(v) = f.kweight {
                    ft.kweight = v;
                    set.push(format!("fit kweight={v}"));
                }
                if let Some(v) = f.dk {
                    ft.dk = v;
                    set.push(format!("fit dk={v}"));
                }
                if let Some(v) = f.rmin {
                    ft.rmin = v;
                    set.push(format!("fit rmin={v}"));
                }
                if let Some(v) = f.rmax {
                    ft.rmax = v;
                    set.push(format!("fit rmax={v}"));
                }
                if let Some(v) = f.dr {
                    ft.dr = v;
                    set.push(format!("fit dr={v}"));
                }
                if let Some(s) = f.fitspace {
                    ft.fitspace = match s.as_str() {
                        "R" | "r" => feffit::FitSpace::R,
                        "k" | "K" => feffit::FitSpace::K,
                        "q" | "Q" => feffit::FitSpace::Q,
                        other => return Err(format!("unknown fitspace \"{other}\" (R, k, or q)")),
                    };
                    set.push(format!("fitspace={s}"));
                }
            }
            if let Some(s) = f.fit_mode {
                self.feffit.set_fit_mode(Self::parse_fit_mode(&s)?);
                set.push(format!("fit_mode={s}"));
            }
        }
        if set.is_empty() {
            return Ok("nothing to set (empty patch)".to_owned());
        }
        Ok(format!("set: {}", set.join(", ")))
    }

    /// Parse an MCP fit-mode string — the one mapping shared by `set_params`
    /// and `run_fit`.
    fn parse_fit_mode(s: &str) -> Result<FitMode, String> {
        match s {
            "fit" => Ok(FitMode::Fit),
            "nofit" => Ok(FitMode::NoFit),
            "only_ft" => Ok(FitMode::OnlyFt),
            other => Err(format!(
                "unknown fit_mode \"{other}\" (fit, nofit, or only_ft)"
            )),
        }
    }

    /// The MCP `get_results` report: the loaded groups, the current group, and
    /// the last Feffit report (the same text as "View fit").
    fn mcp_results_text(&self) -> String {
        let mut out = String::from("=== Groups ===\n");
        if self.session.groups.is_empty() {
            out.push_str("  (none loaded)\n");
        }
        for (i, g) in self.session.groups.iter().enumerate() {
            let cur = if Some(i) == self.session.current {
                '*'
            } else {
                ' '
            };
            let chk = if self.session.selected.contains(&i) {
                "[x]"
            } else {
                "[ ]"
            };
            let e0 =
                g.e0.map_or_else(|| "-".to_owned(), |e| format!("{e:.2} eV"));
            let chi = if g.k.is_some() && g.chi.is_some() {
                "yes"
            } else {
                "no"
            };
            out.push_str(&format!(
                "{cur}{chk} [{i}] {}: {} points, E0 = {e0}, chi(k): {chi}\n",
                g.label,
                g.len()
            ));
        }
        match self.feffit.result() {
            Some(_) => {
                out.push('\n');
                // The report describes the group the fit ran on, which the
                // current-group marker above no longer identifies once the
                // current group has been switched.
                if let Some(g) = &self.feffit_fit_group {
                    out.push_str(&format!("Feffit result for group: {g}\n"));
                }
                out.push_str(&self.feffit.report_text());
                if self.feffit.result_is_stale() {
                    out.push_str("\n(parameters changed since this fit — run_fit again)\n");
                }
            }
            None => out.push_str("\nNo Feffit result yet.\n"),
        }
        out
    }

    /// Open a beamline column file via a native dialog and start an import.
    /// Point the Folders "Data folder" at the folder a just-opened data file
    /// came from. This is where the file dialogs default and where AUTOBK/FEFFIT
    /// outputs land when no Work folder is set, so loading a file makes the
    /// output location explicit in the Folders tab instead of silently falling
    /// back to the launch directory. A Work folder, if set, still takes priority.
    fn set_data_dir_from(&mut self, path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            self.session.folders.data_dir = Some(parent.to_path_buf());
        }
    }

    /// "Open New file" / "Add data files…": open the unified [`open_data_window`]
    /// loader (in-app raw-scan browser + two-pane staging + reading-format
    /// chooser). Autobk loads raw scans from the Folders Data folder; the chooser
    /// starts fresh and is seeded from the first staged file.
    ///
    /// [`open_data_window`]: Self::open_data_window
    fn open_new_file(&mut self) {
        self.import = None;
        // Seed the Selected pane with the files already loaded as groups, so the
        // loader shows the current session rather than an empty pane.
        let loaded: Vec<std::path::PathBuf> = self
            .session
            .groups
            .iter()
            .filter_map(|g| g.filename.clone())
            .collect();
        self.open_data
            .open_on(self.session.folders.data_dir.clone(), loaded);
    }

    /// The detached "Open data" loader window: the in-app browser + two-pane
    /// staging ([`open_data`](Self::open_data)), the reading-format chooser
    /// ([`import`](Self::import), previewing the first staged file), and an
    /// **Autobk Start** button that builds μ(E) for every staged file and reduces
    /// them ([`build_xmu_from_paths`](Self::build_xmu_from_paths)).
    fn open_data_window(&mut self, ctx: &egui::Context) {
        if !self.open_data.open {
            return;
        }
        let mut keep_open = true;
        // Deferred out of the render closure (which reads disjoint `self` fields):
        // the representative re-read and `build_xmu_from_paths` borrow all of self.
        let mut do_start = false;
        // Read before the closures: the pinned action row renders first (one
        // frame stale on the stage-first-file frame) and cannot borrow
        // `open_data` while the content closure holds it mutably.
        let has_staged = self.open_data.has_staged();
        crate::window::detached_with_action(
            ctx,
            "open_data",
            "Open data",
            &mut keep_open,
            [640.0, 720.0],
            |ui| {
                self.open_data.ui(ui);
                ui.separator();
                match self.import.as_mut() {
                    Some(import) => import.ui(ui),
                    None => {
                        ui.weak("Stage a raw scan file above to set its columns.");
                    }
                }
            },
            |ui| {
                let start = crate::widgets::primary(
                    ui,
                    "Autobk Start",
                    crate::widgets::ROW_BTN,
                    has_staged,
                )
                .on_hover_text("Build μ(E) from every staged file, then run AUTOBK");
                if start.clicked() {
                    do_start = true;
                }
            },
        );
        self.open_data.open = self.open_data.open && keep_open;
        // The browser's delete confirmation is a top-level window so the detached
        // viewport cannot clip it.
        self.open_data.confirm_delete(ctx);

        // Keep the reading-format chooser pointed at the representative (first
        // staged) file: re-read only when it changes, so the user's column picks
        // persist while the representative stays put.
        let rep = self.open_data.representative().cloned();
        let cur = self.import.as_ref().and_then(|i| i.file.path.clone());
        if rep != cur {
            match rep {
                Some(path) => match ColumnFile::from_path(&path) {
                    Ok(cf) => self.import = Some(ImportState::new(cf)),
                    Err(e) => self.status.error(format!("Open failed: {e}")),
                },
                None => self.import = None,
            }
        }

        // Build + reduce every staged file with the representative's mapping.
        // Guard on a seeded import so a failed representative read never drops the
        // staged set (the re-read above seeds it whenever anything is staged).
        if do_start && self.import.is_some() {
            let staged = self.open_data.take_staged();
            // The Selected pane is seeded with the already-loaded files, so build
            // only the newly-staged ones — rebuilding an existing group would
            // duplicate it. Groups are keyed by their source path.
            let loaded: std::collections::HashSet<std::path::PathBuf> = self
                .session
                .groups
                .iter()
                .filter_map(|g| g.filename.clone())
                .collect();
            let to_build: Vec<std::path::PathBuf> =
                staged.into_iter().filter(|p| !loaded.contains(p)).collect();
            // Track the folder the data came from (as the old single-file open
            // did), so outputs and the next open default to it.
            if let Some(first) = to_build.first() {
                self.set_data_dir_from(first);
            }
            self.open_data.open = false;
            if !to_build.is_empty() {
                self.build_xmu_from_paths(to_build);
            }
        }
    }

    /// Load a FEFF `chi.dat` as the **Theory** standard — the `k_std`/`chi_std`
    /// constraint applied by AUTOBK so the background does not absorb real
    /// first-shell amplitude. Entry point is the "Theory" row in the Autobk tab's
    /// Data box.
    fn open_theory_file(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("FEFF chi.dat", &["dat"]);
        if let Some(dir) = &self.session.folders.data_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.pick_file() else {
            return;
        };
        match feffit::xasdata::read_chi_dat(&path) {
            Ok((k, chi)) => {
                let n = k.len();
                self.reduction.theory = Some(TheoryStd { path, k, chi });
                self.status
                    .info(format!("Loaded theory χ(k) standard: {n} points"));
            }
            Err(e) => self.status.error(format!("Theory open failed: {e}")),
        }
    }

    /// Build `mu(E)` from the current import selection and add it as a group.
    /// Used by the Autobk tab's "Autobk Start" auto-calc fallback (a single
    /// representative import). The Open-data loader builds staged files directly
    /// through [`build_xmu_from_paths`](Self::build_xmu_from_paths).
    fn calc_xmu(&mut self) {
        let Some(import) = self.import.as_ref() else {
            return;
        };
        let spec = import.to_spec();
        let monitors = import.monitor_policy();
        let input_path = import.file.path.clone();
        // Keep the raw source header so the AUTOBK / FEFFIT writers can echo the
        // original beamline metadata into their provenance block.
        let source_header = import.file.header.clone();
        match feffit::xasdata::build_mu(&import.file, &spec, monitors) {
            Ok((energy, mu)) => {
                let label = input_path
                    .as_deref()
                    .and_then(feffit::xasdata::group_name_from_path)
                    .unwrap_or_else(|| format!("group{}", self.session.groups.len() + 1));
                let n = energy.len();
                // Persist μ(E) as a .xmu next to the source (the original's
                // "Output file"); a single Calc XMU is never numbered.
                let xmu = match self.write_xmu_output(
                    input_path.as_deref(),
                    &label,
                    &energy,
                    &mu,
                    None,
                ) {
                    Ok(name) => format!(" · wrote {name}"),
                    Err(e) => format!(" · .xmu not written: {e}"),
                };
                // Seed the Autobk "Output file" chi base from the loaded name.
                self.reduction.seed_output_file(Some(&label));
                // Record the source path on the group, so the "Data File" field
                // shows it and AUTOBK's χ-file writer can fall back to the source
                // folder instead of the launch directory.
                let mut g = XasGroup::from_mu(label, energy, mu);
                g.filename = input_path;
                g.source_header = source_header;
                self.session.add_group(g);
                self.status.info(format!("Built μ(E): {n} points{xmu}"));
                self.reduction.graph = GraphType::MuBkg;
                // New spectrum: drop stale undo history and re-seed the editor.
                self.clean_undo.clear();
                self.edit_xmu.reset_seed();
                self.plot_data.mark_dirty();
                self.replot_active_tab();
            }
            Err(e) => self.status.error(format!("Calc XMU failed: {e}")),
        }
    }

    /// Persist a μ(E) spectrum as a `.xmu` text file next to its source file (or
    /// in a configured folder). `index`, when set, appends a zero-padded sequence
    /// number to the stem ("Output file numbering"). Returns the written file
    /// name on success, or a reason on failure.
    fn write_xmu_output(
        &self,
        input: Option<&std::path::Path>,
        label: &str,
        energy: &[f64],
        mu: &[f64],
        index: Option<usize>,
    ) -> Result<String, String> {
        let Some(dir) = input
            .and_then(|p| p.parent())
            .filter(|d| !d.as_os_str().is_empty())
            .map(std::path::Path::to_path_buf)
            .or_else(|| self.session.folders.data_dir.clone())
            .or_else(|| self.session.folders.autobk_dir.clone())
        else {
            return Err("no output folder".to_owned());
        };
        let stem = input
            .and_then(feffit::xasdata::group_name_from_path)
            .unwrap_or_else(|| label.to_owned());
        let name = match index {
            Some(i) => format!("{stem}{i:04}.xmu"),
            None => format!("{stem}.xmu"),
        };
        let out = dir.join(&name);
        // Never overwrite the source itself (e.g. Raw mode re-reading a `.xmu`,
        // where the derived output resolves to the same file — case-insensitively
        // on macOS).
        if let Some(inp) = input
            && same_file(inp, &out)
        {
            return Err("would overwrite the source file".to_owned());
        }
        write_xmu(&out, label, energy, mu)
            .map(|()| name)
            .map_err(|e| e.to_string())
    }

    /// Run normalize → autobk → xftf on the current group, then redraw.
    /// Recompute the reduction (normalize → AUTOBK → FT) for the current group
    /// and refresh the plots. This is the live-preview path used when an Autobk
    /// parameter changes: it does NOT write χ files to disk — that stays the
    /// explicit "Autobk Start" action. Returns `(E₀, k-point count)`, or `None`
    /// when there is no μ(E) to reduce.
    fn recompute_reduction(&mut self) -> Option<(f64, usize)> {
        let pre = self.reduction.pre_params();
        let bk = self.reduction.autobk_params();
        let ft = self.reduction.ft_params();
        let xr = self.reduction.xftr_params();
        let info = {
            let g = self.session.current_group_mut()?;
            if g.mu.is_empty() {
                return None;
            }
            // normalize / autobk_group also record the pre-edge range and rbkg
            // they used onto the group (for the output provenance header).
            feffit::xasdata::normalize(g, &pre);
            feffit::xasdata::autobk_group(g, &bk, 1.0);
            feffit::xasdata::xftf_group(g, &ft);
            feffit::xasdata::xftr_group(g, &xr);
            (g.e0.unwrap_or(0.0), g.k.as_ref().map_or(0, |k| k.len()))
        };
        self.plot_data.mark_dirty();
        self.replot_active_tab();
        Some(info)
    }

    /// Run AUTOBK on every group in the run set ([`Session::run_targets`]: the
    /// checked groups, or the current group when nothing is checked) and write
    /// each group's χ files. A single target keeps the original single-Autobk
    /// behaviour (the editable "Output file" stem, detailed status); several use
    /// each group's own label so the outputs stay distinct.
    ///
    /// The reduction runs on a background job (one progress tick per group, the
    /// UI stays live — a 66-scan series froze the frame for seconds when this
    /// ran inline); [`finish_reduction`](Self::finish_reduction) applies the
    /// results, writes the files, and answers a parked MCP reply. Returns
    /// whether the job started — `false` means the refusal reason is already on
    /// the status line (nothing to reduce, no μ(E), or the job slot is busy).
    fn run_reduction(&mut self) -> bool {
        let targets = self.session.run_targets();
        if targets.is_empty() {
            self.status.error("No data to reduce.");
            return false;
        }
        let pre = self.reduction.pre_params();
        let bk = self.reduction.autobk_params();
        let ft = self.reduction.ft_params();
        let xr = self.reduction.xftr_params();
        // Snapshot the targets with a μ(E) for the worker; skip any without
        // (e.g. a directly-loaded χ(k) group), as the inline version did.
        let items: Vec<XasGroup> = targets
            .iter()
            .filter_map(|&idx| self.session.groups.get(idx))
            .filter(|g| !g.mu.is_empty())
            .cloned()
            .collect();
        if items.is_empty() {
            self.status
                .error("No μ(E) in the selected group(s) — Calc XMU first.");
            return false;
        }
        let current_label = self
            .session
            .current
            .and_then(|i| self.session.groups.get(i))
            .map(|g| g.label.clone());
        let targets_len = targets.len();
        let total = items.len();
        let started = self.jobs.start("AUTOBK", total, move |job| {
            use rayon::prelude::*;
            let mut items = items;
            let done = std::sync::atomic::AtomicUsize::new(0);
            items.par_iter_mut().for_each(|g| {
                feffit::xasdata::normalize(g, &pre);
                feffit::xasdata::autobk_group(g, &bk, 1.0);
                feffit::xasdata::xftf_group(g, &ft);
                feffit::xasdata::xftr_group(g, &xr);
                job.progress(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
            });
            Box::new(move |app: &mut Self| app.finish_reduction(items, targets_len, current_label))
        });
        if !started {
            self.status.error(JOB_BUSY);
        }
        started
    }

    /// Apply a finished AUTOBK job: write each reduced group's χ outputs, put
    /// the groups back into the session (matched by label — the session may
    /// have changed while the job ran; vanished groups are dropped with a log
    /// entry), refresh the plots, answer a parked MCP `run_autobk` reply, and
    /// rebuild the In-situ map when its Re-reduce & build asked for this run.
    fn finish_reduction(
        &mut self,
        reduced_groups: Vec<XasGroup>,
        targets_len: usize,
        current_label: Option<String>,
    ) {
        let reduced = reduced_groups.len();
        let single = targets_len == 1;
        let mut current_info: Option<(f64, usize)> = None;
        let mut dropped = 0usize;
        let mut saved = 0usize;
        let mut write_failed = 0usize;
        let mut single_paths = None;
        for g in reduced_groups {
            if self.session.groups.iter().all(|s| s.label != g.label) {
                dropped += 1;
                self.status.detail_error(format!(
                    "{}: group removed while AUTOBK ran — its result was dropped",
                    g.label
                ));
                continue;
            }
            if current_label.as_deref() == Some(g.label.as_str()) {
                current_info = Some((g.e0.unwrap_or(0.0), g.k.as_ref().map_or(0, |k| k.len())));
            }
            // Persist χ(k)/χ(R) (+ background): original single-Autobk "Output
            // file" behaviour for one target; per-group labels for several, so
            // the batch outputs stay distinct.
            if g.k.as_deref().is_some_and(|k| !k.is_empty()) && g.r.is_some() {
                let stem = if single {
                    self.reduction.output_stem_for(&g.label)
                } else {
                    g.label.clone()
                };
                match self.write_chi_files(&g, &stem) {
                    Ok(paths) => {
                        saved += 1;
                        if single {
                            single_paths = Some(paths);
                        }
                    }
                    Err(e) => {
                        write_failed += 1;
                        // Per-file reason into the log; the summary line below
                        // only carries the count.
                        self.status
                            .detail_error(format!("{stem}: χ write failed: {e}"));
                    }
                }
            }
            if let Some(slot) = self.session.groups.iter_mut().find(|s| s.label == g.label) {
                *slot = g;
            }
        }
        let fail = if write_failed > 0 {
            format!(", {write_failed} write error(s)")
        } else {
            String::new()
        };
        let drop_note = if dropped > 0 {
            format!(", {dropped} dropped (group list changed)")
        } else {
            String::new()
        };
        let msg = match (single, current_info, single_paths) {
            (true, Some((e0, npts)), Some((kp, rp))) => format!(
                "AUTOBK done: E₀ = {e0:.2} eV, {npts} k-points · saved {} + {}",
                file_name_of(&kp),
                file_name_of(&rp)
            ),
            (true, Some((e0, npts)), None) => {
                format!("AUTOBK done: E₀ = {e0:.2} eV, {npts} k-points{fail}")
            }
            _ => format!(
                "AUTOBK: reduced {reduced} group(s); saved {saved} χ pair(s){fail}{drop_note}"
            ),
        };
        let ok = write_failed + dropped == 0;
        if ok {
            self.status.info(msg);
        } else {
            self.status.error(msg);
        }
        // Show the whole batch at once when several were reduced.
        if targets_len > 1 {
            self.show_all_groups = true;
        }
        self.plot_data.mark_dirty();
        self.replot_active_tab();
        // A parked MCP run_autobk reply belongs to this job (single job slot).
        if let Some(reply) = self.mcp_job_reply.take() {
            let line = self.status.line().to_owned();
            let _ = reply.send(if ok { Ok(line) } else { Err(line) });
        }
        // The In-situ window's Re-reduce & build: the groups now carry the
        // fresh reduction, so rebuild its map. A write failure doesn't block
        // this — the in-memory χ the map reads is valid regardless.
        if std::mem::take(&mut self.insitu_rebuild_after_reduce) {
            self.insitu.build_from_groups(&self.session.groups);
        }
    }

    /// Write the AUTOBK outputs the original's "Output file" produces: χ(k) →
    /// `<stem>k.chi` (k, χ), χ(R) → `<stem>r.chi` (R, |χ|, Re, Im), and the
    /// background pair μ0(E) → `<stem>e.bkg` and μ0−μ in k → `<stem>k.bkg`, into
    /// the Autobk folder (falling back to the data folder, then the source
    /// folder). The single-group path passes the editable "Output file" stem; the
    /// Multiple-autobk path passes each group's own label so the outputs stay
    /// distinct. Returns the two χ paths (for the status line).
    fn write_chi_files(
        &self,
        g: &XasGroup,
        stem: &str,
    ) -> std::io::Result<(std::path::PathBuf, std::path::PathBuf)> {
        let stem = match stem.trim() {
            "" => "chi",
            s => s,
        };
        let dir = self
            .session
            .folders
            .autobk_dir
            .clone()
            .or_else(|| self.session.folders.data_dir.clone())
            .or_else(|| {
                g.filename
                    .as_ref()
                    .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
            })
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let k_path = dir.join(format!("{stem}k.chi"));
        let r_path = dir.join(format!("{stem}r.chi"));
        // AUTOBK χ files carry the same provenance header as the FEFFIT outputs,
        // built from the group plus the reduce-tab FT params (this is the FT the
        // χ(R) here came from).
        let header = crate::chi_io::provenance_header(
            g,
            self.reduction.kmin,
            self.reduction.kmax,
            self.reduction.kweight,
            self.reduction.dk,
        );
        write_chik(
            &k_path,
            &header,
            g.k.as_deref().unwrap_or(&[]),
            g.chi.as_deref().unwrap_or(&[]),
        )?;
        write_chir(
            &r_path,
            &header,
            g.r.as_deref().unwrap_or(&[]),
            g.chir_mag.as_deref().unwrap_or(&[]),
            g.chir_re.as_deref().unwrap_or(&[]),
            g.chir_im.as_deref().unwrap_or(&[]),
        )?;
        // Original XAFSView also writes the AUTOBK background: μ0(E) → <stem>e.bkg
        // and the background EXAFS in k → <stem>k.bkg. The k-space background is
        // μ0−μ on the k grid, i.e. −edge_step·χ (χ = (μ−μ0)/edge_step).
        if let Some(bkg) = g.bkg.as_deref()
            && !g.energy.is_empty()
            && bkg.len() == g.energy.len()
        {
            write_file(
                &dir.join(format!("{stem}e.bkg")),
                crate::chi_io::xy_string(&header, " energy            bkg", &g.energy, bkg),
            )?;
        }
        if let (Some(k), Some(chi), Some(step)) = (g.k.as_deref(), g.chi.as_deref(), g.edge_step) {
            let bkg_k: Vec<f64> = chi.iter().map(|&c| -step * c).collect();
            write_file(
                &dir.join(format!("{stem}k.bkg")),
                crate::chi_io::xy_string(&header, " k                 bkg", k, &bkg_k),
            )?;
        }
        Ok((k_path, r_path))
    }

    /// Write the FEFFIT transforms the original's Plot Data reads: the data in
    /// k/r/q space → `<stem>k.dat`/`<stem>r.dat`/`<stem>q.dat` and the model →
    /// `<stem>k.fit`/`<stem>r.fit`/`<stem>q.fit`, into the Feffit folder (falling
    /// back to the data folder). Returns the number of files written (6). The
    /// provenance header is built from the group at `group_idx` — the group this
    /// `plot` was fitted from — so a multi-group Feffit Run writes each group's
    /// own provenance, not the current selection's.
    fn write_feffit_outputs(
        &self,
        plot: &FeffitPlot,
        stem: &str,
        group_idx: usize,
    ) -> std::io::Result<usize> {
        let stem = match stem.trim() {
            "" => "feffit",
            s => s,
        };
        let dir = self.feffit_output_dir();
        let (kmin, kmax, kweight, dk) = self.feffit.header_ft();
        let header = match self.session.groups.get(group_idx) {
            Some(g) => crate::chi_io::provenance_header(g, kmin, kmax, kweight, dk),
            None => format!("# {stem}\r\n"),
        };
        let files = plot.output_pairs(stem, &header);
        let n = files.len();
        for (name, content) in files {
            write_file(&dir.join(name), content)?;
        }
        Ok(n)
    }

    /// The folder the FEFFIT `.dat`/`.fit` transforms are written to: the Feffit
    /// folder, falling back to the data folder, then the working directory. The
    /// single owner of "where FEFFIT transforms go", shared by the single-fit
    /// writer and the batch panel's "Save χ data+fit" so both land where the
    /// original XAFSView writes them (the Feffit sub-folder).
    fn feffit_output_dir(&self) -> std::path::PathBuf {
        self.session
            .folders
            .feffit_dir
            .clone()
            .or_else(|| self.session.folders.data_dir.clone())
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    }

    /// Write the batch panel's FEFFIT `.dat`/`.fit` transforms (one set per
    /// fitted group) to the Feffit folder — the same destination the single-fit
    /// `write_feffit_outputs` uses — reporting the outcome.
    fn write_feffit_output_files(&mut self, files: Vec<(String, String)>) {
        let dir = self.feffit_output_dir();
        let mut written = 0usize;
        let mut failed = 0usize;
        for (name, content) in &files {
            match write_file(&dir.join(name), content) {
                Ok(()) => written += 1,
                Err(e) => {
                    failed += 1;
                    // Per-file reason into the log; the summary line below only
                    // carries the count.
                    self.status.detail_error(format!("{name}: {e}"));
                }
            }
        }
        let extra = if failed > 0 {
            format!(" ({failed} failed)")
        } else {
            String::new()
        };
        let msg = format!(
            "Saved {written} χ data+fit file(s) to {}{extra}.",
            dir.display()
        );
        if failed > 0 {
            self.status.error(msg);
        } else {
            self.status.info(msg);
        }
    }

    /// Run normalize → AUTOBK → FT on every loaded group with one shared set of
    /// parameters (the Autobk tab's current settings) on a background job, then
    /// refresh the plots. The reduction runs on cloned groups off-thread (per
    /// group, in parallel, with progress ticks); [`finish_multi_autobk`] applies
    /// them back on the main thread.
    ///
    /// [`finish_multi_autobk`]: Self::finish_multi_autobk
    fn run_multi_autobk(&mut self) {
        if self.session.groups.is_empty() {
            self.status.error("No groups loaded.");
            return;
        }
        let pre = self.reduction.pre_params();
        let bk = self.reduction.autobk_params();
        let ft = self.reduction.ft_params();
        let mut groups = self.session.groups.clone();
        let total = groups.len();
        let started = self.jobs.start("Multiple AUTOBK", total, move |job| {
            use rayon::prelude::*;
            // `reduce_all` on one-group slices: each group reduces independently
            // (the function's own contract), so the per-group calls keep the
            // core fan-out while giving a progress tick per finished group.
            let reduced = std::sync::atomic::AtomicUsize::new(0);
            let done = std::sync::atomic::AtomicUsize::new(0);
            groups.par_iter_mut().for_each(|g| {
                let n = feffit::xasdata::reduce_all(std::slice::from_mut(g), &pre, &bk, &ft, 1.0);
                reduced.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                job.progress(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
            });
            let n = reduced.into_inner();
            Box::new(move |app: &mut Self| app.finish_multi_autobk(groups, n))
        });
        if !started {
            self.status.error(JOB_BUSY);
        }
    }

    /// Apply a finished Multiple-AUTOBK job: put the reduced groups back into
    /// the session (matched by label — the session may have changed while the
    /// job ran; vanished groups are dropped with a log entry), write each
    /// group's χ outputs, and refresh the plots.
    fn finish_multi_autobk(&mut self, reduced_groups: Vec<XasGroup>, n: usize) {
        let mut dropped = 0usize;
        for g in reduced_groups {
            match self.session.groups.iter_mut().find(|s| s.label == g.label) {
                Some(slot) => *slot = g,
                None => {
                    dropped += 1;
                    self.status.detail_error(format!(
                        "{}: group removed while Multiple AUTOBK ran — its result was dropped",
                        g.label
                    ));
                }
            }
        }
        // Per-group χ outputs, the original Multiple-autobk behaviour: each group
        // writes its own <label>k.chi / <label>r.chi (its label, not the shared
        // "Output file", so batch outputs stay distinct).
        let mut saved = 0usize;
        let mut write_failed = 0usize;
        for g in &self.session.groups {
            if g.k.as_deref().is_some_and(|k| !k.is_empty()) && g.r.is_some() {
                match self.write_chi_files(g, &g.label) {
                    Ok(_) => saved += 1,
                    Err(e) => {
                        write_failed += 1;
                        // Per-group reason into the log; the summary line below
                        // only carries the count.
                        self.status
                            .detail_error(format!("{}: χ write failed: {e}", g.label));
                    }
                }
            }
        }
        let fail = if write_failed > 0 {
            format!(", {write_failed} write error(s)")
        } else {
            String::new()
        };
        let drop_note = if dropped > 0 {
            format!(", {dropped} dropped (group list changed)")
        } else {
            String::new()
        };
        let msg = format!(
            "Multiple AUTOBK: reduced {n} group(s); saved {saved} χ pair(s){fail}{drop_note}."
        );
        if write_failed + dropped > 0 {
            self.status.error(msg);
        } else {
            self.status.info(msg);
        }
        // Show the whole batch at once, not just the current group.
        self.show_all_groups = true;
        self.plot_data.mark_dirty();
        self.replot_active_tab();
    }

    /// Build a μ(E) group from each path using the active import's column
    /// mapping and header-skip, add them to the session, and write each `.xmu`.
    /// The staging picker's OK routes here.
    fn build_xmu_from_paths(&mut self, paths: Vec<std::path::PathBuf>) {
        let Some(spec) = self.import.as_ref().map(|i| i.to_spec()) else {
            self.status
                .error("Open one file and choose its columns first, then batch the rest.");
            return;
        };
        // Apply the chooser's header-skip override to every batch file (None =
        // each file auto-detects its own header).
        let skip = self.import.as_ref().and_then(|i| i.header_skip);
        // One monitor-reading rule for the whole batch, like the column mapping.
        let monitors = self
            .import
            .as_ref()
            .map_or_else(Default::default, |i| i.monitor_policy());
        // "Output file numbering": append a sequence number to each .xmu so the
        // batch outputs stay distinct (the original toggles this in the dialog).
        let number = self.import.as_ref().is_some_and(|i| i.number_outputs);
        let pre = self.reduction.pre_params();
        let bk = self.reduction.autobk_params();
        let ft = self.reduction.ft_params();
        let total = paths.len();
        let started = self.jobs.start("Batch μ(E)", total, move |job| {
            // Read + build + auto-reduce on the worker; the built groups are
            // owned here (nothing shared), so the main-thread apply only writes
            // the .xmu files and appends them to the session. Failures carry
            // their per-file reason back for the log.
            let mut files = Vec::with_capacity(paths.len());
            let mut read_fail: Vec<String> = Vec::new();
            for path in paths {
                match feffit::xasdata::ColumnFile::from_path_skip(&path, skip) {
                    Ok(cf) => files.push(cf),
                    Err(e) => read_fail.push(format!("{}: unreadable: {e}", path.display())),
                }
            }
            let mut built: Vec<(XasGroup, Option<usize>)> = Vec::new();
            let mut build_fail: Vec<String> = Vec::new();
            for (i, result) in feffit::xasdata::make_xmu_batch(&files, &spec, monitors)
                .into_iter()
                .enumerate()
            {
                match result {
                    Ok(mut group) => {
                        // Record the source path (as the single-file paths do) so
                        // each group's later χ-file write falls back to its own
                        // folder; the .xmu write reads it back in the apply.
                        group.filename = files.get(i).and_then(|f| f.path.clone());
                        built.push((group, number.then_some(i + 1)));
                    }
                    Err(e) => {
                        let name = files
                            .get(i)
                            .and_then(|f| f.path.as_deref())
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| format!("file {}", i + 1));
                        build_fail.push(format!("{name}: μ(E) build failed: {e}"));
                    }
                }
            }
            // Auto-reduce the just-built groups: normalize → AUTOBK → forward-FT
            // so background/EXAFS graphs show them immediately (not just μ(E)).
            // The reduce dominates the job's time, so it carries the progress.
            use rayon::prelude::*;
            let reduced = std::sync::atomic::AtomicUsize::new(0);
            let done = std::sync::atomic::AtomicUsize::new(0);
            built.par_iter_mut().for_each(|(g, _)| {
                let n = feffit::xasdata::reduce_all(std::slice::from_mut(g), &pre, &bk, &ft, 1.0);
                reduced.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                job.progress(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
            });
            let reduced = reduced.into_inner();
            Box::new(move |app: &mut Self| {
                app.finish_build_xmu(built, read_fail, build_fail, reduced)
            })
        });
        if !started {
            self.status.error(JOB_BUSY);
        }
    }

    /// Apply a finished batch-μ(E) job: log the per-file failures, write each
    /// built group's `.xmu` (the χ(k)/χ(R) files are not written — that stays
    /// the explicit "Multiple AUTOBK" action), append the groups to the
    /// session, and refresh the plots.
    fn finish_build_xmu(
        &mut self,
        groups: Vec<(XasGroup, Option<usize>)>,
        read_fail: Vec<String>,
        build_fail: Vec<String>,
        reduced: usize,
    ) {
        let (read_errors, build_errors) = (read_fail.len(), build_fail.len());
        for m in read_fail.into_iter().chain(build_fail) {
            self.status.detail_error(m);
        }
        let built = groups.len();
        let mut written = 0usize;
        let mut write_failed = 0usize;
        for (group, index) in groups {
            let input = group.filename.clone();
            match self.write_xmu_output(
                input.as_deref(),
                &group.label,
                &group.energy,
                &group.mu,
                index,
            ) {
                Ok(_) => written += 1,
                Err(e) => {
                    write_failed += 1;
                    self.status
                        .detail_error(format!("{}: .xmu not written: {e}", group.label));
                }
            }
            self.session.add_group(group);
        }
        if built > 0 {
            self.reduction.graph = GraphType::MuBkg;
            self.clean_undo.clear();
            self.edit_xmu.reset_seed();
            // Show every μ(E) just built at once, not just the last group.
            self.show_all_groups = true;
            self.plot_data.mark_dirty();
            self.replot_active_tab();
        }
        let msg = format!(
            "Batch μ(E): built {built}, wrote {written} .xmu \
             ({write_failed} not written), build errors {build_errors}, \
             unreadable {read_errors}, AUTOBK reduced {reduced}."
        );
        let failed = write_failed + build_errors + read_errors > 0;
        // Answer a parked MCP batch load_data with the same summary.
        if let Some(reply) = self.mcp_job_reply.take() {
            let _ = reply.send(if failed {
                Err(msg.clone())
            } else {
                Ok(msg.clone())
            });
        }
        if failed {
            self.status.error(msg);
        } else {
            self.status.info(msg);
        }
    }

    /// Open the Edit-μ(E) window; its energy widgets are seeded per frame
    /// against the current group right before the window is shown.
    fn open_edit_xmu(&mut self) {
        self.edit_xmu.open = true;
    }

    /// Compute the Morlet wavelet transform of the current group's `χ(k)` using
    /// the wavelet window's parameters, and hand the result back to the window.
    fn run_wavelet(&mut self) {
        let p = self.wavelet.params();
        let computed = self
            .session
            .current_group()
            .and_then(|g| match (&g.k, &g.chi) {
                (Some(k), Some(chi)) if k.len() >= 2 => {
                    let kstep = k[1] - k[0];
                    Some((morlet_cwt(k, chi, kstep, &p), kstep))
                }
                _ => None,
            });
        match computed {
            Some((Some(wt), kstep)) => {
                let info = format!(
                    "{} R × {} k samples, k-step {:.3} Å⁻¹",
                    wt.r.len(),
                    wt.k.len(),
                    kstep
                );
                self.wavelet.set_result(Some(wt), info);
                self.status.info("Wavelet transform computed.");
            }
            Some((None, _)) => {
                self.wavelet.set_result(
                    None,
                    "Transform failed — check R range vs k-step (Nyquist).".to_owned(),
                );
                self.status
                    .error("Wavelet transform failed (R range exceeds Nyquist?).");
            }
            None => {
                self.wavelet.set_result(
                    None,
                    "No χ(k) — run AUTOBK on the current group first.".to_owned(),
                );
                self.status
                    .error("No χ(k) for the wavelet transform — run AUTOBK first.");
            }
        }
    }

    /// Save the last wavelet `|W(k,R)|` result as a CSV matrix via a save dialog,
    /// reporting the outcome. No-op (with a status hint) when nothing has been
    /// computed yet or the user cancels the dialog.
    fn export_wavelet_csv(&mut self) {
        let Some(csv) = self.wavelet.result_csv() else {
            self.status
                .error("No wavelet result to export — Compute first.");
            return;
        };
        let mut dlg = rfd::FileDialog::new()
            .add_filter("CSV", &["csv"])
            .set_file_name("wavelet.csv");
        if let Some(dir) = self
            .session
            .folders
            .results_dir
            .clone()
            .or_else(|| self.session.folders.data_dir.clone())
        {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.save_file() else {
            return;
        };
        match std::fs::write(&path, csv) {
            Ok(()) => self
                .status
                .info(format!("Wavelet CSV saved to {}.", path.display())),
            Err(e) => self.status.error(format!("Wavelet CSV save failed: {e}")),
        }
    }

    /// Apply a cleanup edit (or undo) to the current group, recording an undo
    /// snapshot for edits that actually changed the spectrum.
    fn apply_clean_action(&mut self, action: CleanAction) {
        const UNDO_CAP: usize = 32;

        if let CleanAction::Undo = action {
            // The snapshot names its own group (its label): restore *that*
            // group, wherever it now sits — the current index may point at a
            // different group than the one edited.
            if let Some(prev) = self.clean_undo.pop() {
                match self
                    .session
                    .groups
                    .iter()
                    .position(|g| g.label == prev.label)
                {
                    Some(idx) => {
                        self.status
                            .info(format!("Undid last edit on {}.", prev.label));
                        self.session.groups[idx] = prev;
                        self.plot.set_x_markers(&[], GLITCH_MARKER);
                        self.plot_data.mark_dirty();
                        self.replot_active_tab();
                    }
                    None => self.status.error(format!(
                        "Undo target \"{}\" is no longer loaded.",
                        prev.label
                    )),
                }
            }
            return;
        }

        let Some(g) = self.session.current_group_mut() else {
            return;
        };
        if g.mu.is_empty() {
            return;
        }
        let snapshot = g.clone();
        let (changed, msg) = match action {
            CleanAction::AutoDeglitch(nsig) => {
                let n = feffit::xasdata::auto_deglitch(g, nsig);
                if n > 0 {
                    (true, format!("Auto deglitch: removed {n} point(s)"))
                } else {
                    (false, format!("Auto deglitch: no candidates at {nsig:.0}σ"))
                }
            }
            CleanAction::DeglitchPoint(e) => {
                let n = feffit::xasdata::deglitch_point(g, e);
                (n > 0, format!("Deglitch: removed {n} point(s)"))
            }
            CleanAction::DeglitchRange(side, e1, e2) => {
                let n = feffit::xasdata::deglitch_range(g, side, e1, e2);
                (n > 0, format!("Deglitch range: removed {n} point(s)"))
            }
            CleanAction::Trim(lo, hi) => {
                let n = feffit::xasdata::trim(g, lo, hi);
                (n > 0, format!("Trim: removed {n} point(s)"))
            }
            CleanAction::Smooth(sigma, form) => {
                let ok = feffit::xasdata::smooth_mu(g, sigma, form);
                let m = if ok {
                    format!("Smoothed μ(E) (σ = {sigma:.2} eV)")
                } else {
                    "Smoothing skipped (grid too short or not increasing).".to_owned()
                };
                (ok, m)
            }
            CleanAction::Undo => unreachable!("handled above"),
            CleanAction::AutoFind(_) | CleanAction::AutoDeglitchChecked(_) => {
                unreachable!("handled by the caller")
            }
        };

        // `changed = false` means the edit did not apply (e.g. grid too short) —
        // that reads as a failure, so it logs as one.
        if changed {
            self.status.info(msg);
        } else {
            self.status.error(msg);
        }
        if changed {
            self.clean_undo.push(snapshot);
            if self.clean_undo.len() > UNDO_CAP {
                self.clean_undo.remove(0);
            }
            // Any edit moves the data the candidate markers pointed at.
            self.plot.set_x_markers(&[], GLITCH_MARKER);
            self.plot_data.mark_dirty();
            self.replot_active_tab();
        }
    }

    /// Preview automatic glitch candidates on the current group: switch the
    /// graph to the E-space view, mark each candidate energy, and report the
    /// count. Removes nothing — "Remove" (or the batch button) does that.
    fn auto_deglitch_find(&mut self, nsig: f64) {
        let Some(g) = self.session.current_group() else {
            return;
        };
        if g.mu.is_empty() {
            return;
        }
        let energies = feffit::xasdata::find_glitches(g, nsig);
        if energies.is_empty() {
            self.plot.set_x_markers(&[], GLITCH_MARKER);
            self.status
                .info(format!("Auto deglitch: no candidates at {nsig:.0}σ."));
            return;
        }
        // The markers are energies — show them on the E-space graph, which is
        // the Autobk tab's (the Edit μ(E) window can be driven from any tab).
        self.tab = Tab::Autobk;
        self.reduction.graph = GraphType::MuBkg;
        self.replot_graph();
        self.plot.set_x_markers(&energies, GLITCH_MARKER);
        let shown: Vec<String> = energies.iter().take(6).map(|e| format!("{e:.1}")).collect();
        let more = if energies.len() > shown.len() {
            ", …"
        } else {
            ""
        };
        self.status.info(format!(
            "Auto deglitch: {} candidate(s) at {} eV{more} — Remove drops them.",
            energies.len(),
            shown.join(", "),
        ));
    }

    /// Auto-deglitch every Run-target group at `nsig`; the single owner of the
    /// batch edit (the Edit-μ(E) button and the MCP `deglitch` tool both come
    /// through here). Returns `(points removed, groups changed, groups
    /// targeted)`. On any removal it clears the single-group undo stack (the
    /// snapshots no longer describe one edit), drops stale candidate markers,
    /// and replots.
    fn deglitch_targets(&mut self, nsig: f64) -> (usize, usize, usize) {
        let targets = self.session.run_targets();
        let mut total = 0usize;
        let mut groups = 0usize;
        for &idx in &targets {
            let Some(g) = self.session.groups.get_mut(idx) else {
                continue;
            };
            if g.mu.is_empty() {
                continue;
            }
            let n = feffit::xasdata::auto_deglitch(g, nsig);
            if n > 0 {
                total += n;
                groups += 1;
            }
        }
        if total > 0 {
            self.clean_undo.clear();
            self.plot.set_x_markers(&[], GLITCH_MARKER);
            self.plot_data.mark_dirty();
            self.replot_active_tab();
        }
        (total, groups, targets.len())
    }

    /// Auto-deglitch every group checked for Run (the batch counterpart of
    /// [`CleanAction::AutoDeglitch`], for a whole scan series in one pass).
    fn auto_deglitch_checked(&mut self, nsig: f64) {
        let (total, groups, targeted) = self.deglitch_targets(nsig);
        if targeted == 0 {
            self.status.error("No data to deglitch.");
        } else if total == 0 {
            self.status.info(format!(
                "Auto deglitch: no candidates at {nsig:.0}σ in {targeted} group(s)."
            ));
        } else {
            self.status.info(format!(
                "Auto deglitch: removed {total} point(s) across {groups}/{targeted} group(s) — \
                 re-run Autobk. Undo unavailable after a batch edit.",
            ));
        }
    }

    /// Repopulate the shared plot for whichever tab is showing it. The Autobk and
    /// Feffit tabs draw different content onto one `Plot`, so every path that
    /// changes shared state from outside a tab (the Data window, the Open data
    /// loader, Edit μ(E), background AUTOBK completions, MCP calls) must replot
    /// through here: calling `replot_graph` directly while the Feffit tab is
    /// active overwrote the fit graph with the Autobk μ(E) overlay until the
    /// next Run. Tabs without the shared plot leave it alone.
    fn replot_active_tab(&mut self) {
        match self.tab {
            Tab::Autobk => self.replot_graph(),
            Tab::Feffit => self.replot_feffit(),
            Tab::Folders | Tab::FeffitTxt | Tab::Atoms | Tab::Feff => {}
        }
    }

    /// Redraw the shared plot for the active group according to the selected
    /// [`GraphType`]. Curves that need a stage not yet computed are skipped.
    /// Autobk-tab content: callers outside the Autobk tab's own controls go
    /// through [`replot_active_tab`](Self::replot_active_tab).
    fn replot_graph(&mut self) {
        // The Autobk tab's own graph: the current group's base curves, plus the
        // other loaded groups overlaid after Multiple_data (the current group is
        // already drawn in full by `replot_graph_base`). Plot Data is a separate
        // window now and draws its loaded files onto its own graph.
        //
        // Declared rebuild instead of clear-and-re-add: with tens of groups
        // loaded, recreating every curve's GPU buffers on each replot dominates
        // the cost of every interaction; the declare diff touches only the
        // curves that actually changed.
        self.plot.begin_declare();
        self.replot_graph_base();
        if self.show_all_groups {
            self.overlay_other_groups();
        }
        let changed = self.plot.commit_declare();
        // On the k-weighted χ graphs, y-autoscale over the FT k-window only:
        // the AUTOBK spline can hook sharply at the data end (parity with the
        // original tool), and fitting y to the hook squashes the oscillations
        // the analysis actually uses. Only on a content change, so a no-op
        // replot preserves the user's zoom like the declare diff does — and
        // only while the toolbar's Y-autoscale toggle is on, like rsplot's own
        // content-change refit (an off axis keeps the user's manual limits).
        if changed
            && self.plot.plot().y_autoscale()
            && matches!(self.reduction.graph, GraphType::KChi | GraphType::ChiQ)
            && let Some((lo, hi)) = self
                .plot
                .y_bounds_in_x_window(self.reduction.kmin, self.reduction.kmax)
        {
            let pad = if hi > lo { 0.05 * (hi - lo) } else { 1.0 };
            self.plot
                .set_graph_y_limits(lo - pad, hi + pad, rsplot::YAxis::Left);
        }
    }

    /// The single "principal" curve `(x, y)` a group contributes for `graph` — the
    /// one curve used when overlaying several groups (the same quantity
    /// [`replot_graph_base`] draws as that group's main curve). `None` when the
    /// stage it needs has not been computed yet.
    fn group_principal_curve(
        g: &XasGroup,
        graph: GraphType,
        kweight: i32,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        use feffit::xasproc::mathutils::dmude as deriv;
        let kw_chi = |k: &[f64], chi: &[f64]| {
            k.iter()
                .zip(chi)
                .map(|(&kk, &c)| c * kk.powi(kweight))
                .collect()
        };
        match graph {
            GraphType::MuBkg => (!g.energy.is_empty()).then(|| (g.energy.clone(), g.mu.clone())),
            GraphType::Norm => {
                let flat = g.flat.as_ref().or(g.norm.as_ref())?;
                (!g.energy.is_empty()).then(|| (g.energy.clone(), flat.clone()))
            }
            GraphType::KChi | GraphType::ChiQ => {
                let (k, chi) = (g.k.as_ref()?, g.chi.as_ref()?);
                Some((k.clone(), kw_chi(k, chi)))
            }
            GraphType::ChiR => {
                let (r, mag) = (g.r.as_ref()?, g.chir_mag.as_ref()?);
                Some((r.clone(), mag.clone()))
            }
            GraphType::MuDeriv | GraphType::MuBkgEDeriv => (!g.energy.is_empty()
                && !g.mu.is_empty())
            .then(|| (g.energy.clone(), deriv(&g.mu, &g.energy))),
            GraphType::BkgEDeriv => {
                let bkg = g.bkg.as_ref()?;
                Some((g.energy.clone(), deriv(bkg, &g.energy)))
            }
            GraphType::BkgKDeriv => {
                let (k, chi, step) = (g.k.as_ref()?, g.chi.as_ref()?, g.edge_step?);
                let bkg_k: Vec<f64> = chi.iter().map(|&c| -step * c).collect();
                Some((k.clone(), deriv(&bkg_k, k)))
            }
            GraphType::BkgEDeriv2 => {
                let bkg = g.bkg.as_ref()?;
                let d1 = deriv(bkg, &g.energy);
                Some((g.energy.clone(), deriv(&d1, &g.energy)))
            }
        }
    }

    /// Overlay the *other* loaded groups' principal curve for the selected graph
    /// type onto the shared plot — the batch-comparison view behind
    /// [`show_all_groups`]. The current group is skipped (already drawn in full by
    /// [`replot_graph_base`]); the others use palette colours past the current
    /// group's BLUE / ORANGE, legended by group label.
    ///
    /// [`show_all_groups`]: Self::show_all_groups
    fn overlay_other_groups(&mut self) {
        let graph = self.reduction.graph;
        let kweight = self.reduction.kweight;
        let stack = self.stack;
        let cur = self.session.current;
        let palette = &crate::plot::PALETTE;
        // On an energy axis a group at another edge (Mn at 6.5 keV next to Pt
        // at 11.5 keV) squeezes both onto slivers of a 5 keV span, so only the
        // groups whose energy range overlaps the focused group's are compared.
        // k, R and q axes are shared by every group and need no such cut.
        let cur_x: Option<Vec<f64>> = if graph.x_is_energy() {
            cur.and_then(|i| self.session.groups.get(i))
                .and_then(|g| Self::group_principal_curve(g, graph, kweight))
                .map(|(x, _)| x)
        } else {
            None
        };
        let mut j = 0usize;
        for (i, g) in self.session.groups.iter().enumerate() {
            if Some(i) == cur {
                continue;
            }
            if let Some((x, y)) = Self::group_principal_curve(g, graph, kweight) {
                if cur_x.as_deref().is_some_and(|cx| !spans_overlap(cx, &x)) {
                    continue;
                }
                // Skip BLUE / ORANGE (the current group's colours) so a comparison
                // curve never shares the focused group's colour.
                let color = palette[2 + j % (palette.len() - 2)];
                // Waterfall: lift the j-th comparison curve by (j+1)·stack above
                // the focused group (baseline, offset 0), mirroring Plot Data's
                // per-trace idx·stack. stack == 0 leaves the curve at its level.
                let off = (j + 1) as f64 * stack;
                let y: Vec<f64> = if off != 0.0 {
                    y.iter().map(|v| v + off).collect()
                } else {
                    y
                };
                self.plot.add_curve_with_legend(&x, &y, color, &g.label);
                j += 1;
            }
        }
    }

    /// Whether a click landing on the Autobk graph's curve `label` picked a
    /// point of the current group (its principal curve, or an auxiliary one
    /// such as the background) rather than another group's Show-all overlay.
    /// Overlays are legended by their own group's label
    /// ([`overlay_other_groups`](Self::overlay_other_groups)), so any other
    /// group's label marks one; a label the current group shares with another
    /// group is taken as the current group's own curve.
    fn pick_is_current_group(groups: &[XasGroup], cur: Option<usize>, label: &str) -> bool {
        let is_current = cur
            .and_then(|i| groups.get(i))
            .is_some_and(|g| g.label == label);
        is_current
            || !groups
                .iter()
                .enumerate()
                .any(|(i, g)| Some(i) != cur && g.label == label)
    }

    /// Draw the Autobk tab's own base curves for the selected graph type into the
    /// (already-cleared) shared plot. Called only by [`replot_graph`], which then
    /// appends the Plot Data overlay.
    fn replot_graph_base(&mut self) {
        use crate::plot::{BLUE, ORANGE};

        let graph = self.reduction.graph;
        let kweight = self.reduction.kweight;
        let show_bkg = self.show_bkg;
        let Some(g) = self.session.current_group() else {
            self.plot.set_graph_title("XAFSView");
            return;
        };
        // Title the graph with the scan it shows: with tens of groups loaded,
        // the axis labels say *what* is plotted but not *which* scan.
        self.plot.set_graph_title(g.label.as_str());

        // Derivative w.r.t. x the way the engine computes dμ/dE
        // (`np.gradient(y)/np.gradient(x)`), shared by the derivative graph types.
        use feffit::xasproc::mathutils::dmude as deriv;

        match graph {
            GraphType::MuBkg => {
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot.set_graph_y_label("μ(E)", rsplot::YAxis::Left);
                if !g.energy.is_empty() {
                    // Legend the μ(E) trace with the data's own name (the y-axis
                    // already says it is μ(E)), so the AUTOBK view identifies which
                    // scan is shown rather than repeating "μ(E)".
                    self.plot
                        .add_curve_with_legend(&g.energy, &g.mu, BLUE, &g.label);
                }
                if show_bkg && let Some(bkg) = &g.bkg {
                    self.plot
                        .add_curve_with_legend(&g.energy, bkg, ORANGE, "background");
                }
            }
            GraphType::KChi => {
                self.plot.set_graph_x_label("k (Å⁻¹)");
                self.plot.set_graph_y_label("kʷ·χ(k)", rsplot::YAxis::Left);
                if let (Some(k), Some(chi)) = (&g.k, &g.chi) {
                    let y: Vec<f64> = k
                        .iter()
                        .zip(chi)
                        .map(|(&kk, &c)| c * kk.powi(kweight))
                        .collect();
                    self.plot.add_curve_with_legend(k, &y, BLUE, &g.label);
                }
            }
            GraphType::ChiR => {
                self.plot.set_graph_x_label("R (Å)");
                self.plot.set_graph_y_label("|χ(R)|", rsplot::YAxis::Left);
                if let (Some(r), Some(mag)) = (&g.r, &g.chir_mag) {
                    self.plot.add_curve_with_legend(r, mag, BLUE, &g.label);
                }
            }
            GraphType::ChiQ => {
                // Fourier-filtered EXAFS: the R-windowed back-transform Re[χ(q)]
                // (carrying the forward FT's k-weight) overlaid on the original
                // kʷ·χ(k), so the R-window's effect reads directly against the
                // unfiltered signal on the shared k≈q axis.
                self.plot.set_graph_x_label("q (Å⁻¹)");
                self.plot.set_graph_y_label("kʷ·χ(q)", rsplot::YAxis::Left);
                if let (Some(k), Some(chi)) = (&g.k, &g.chi) {
                    let y: Vec<f64> = k
                        .iter()
                        .zip(chi)
                        .map(|(&kk, &c)| c * kk.powi(kweight))
                        .collect();
                    self.plot.add_curve_with_legend(k, &y, BLUE, &g.label);
                }
                if let (Some(q), Some(re)) = (&g.q, &g.chiq_re) {
                    self.plot
                        .add_curve_with_legend(q, re, ORANGE, "Re[χ(q)] (filtered)");
                }
            }
            GraphType::MuDeriv => {
                // XMU' + XMU'': first and second derivative of the raw μ(E).
                // Works on just-loaded data — no AUTOBK needed.
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot
                    .set_graph_y_label("dμ/dE  ·  d²μ/dE²", rsplot::YAxis::Left);
                if !g.energy.is_empty() && !g.mu.is_empty() {
                    let d1 = deriv(&g.mu, &g.energy);
                    let d2 = deriv(&d1, &g.energy);
                    self.plot
                        .add_curve_with_legend(&g.energy, &d1, BLUE, &g.label);
                    self.plot
                        .add_curve_with_legend(&g.energy, &d2, ORANGE, "XMU''");
                }
            }
            GraphType::BkgEDeriv => {
                // BKG(E)': first derivative of the background μ0(E).
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot
                    .set_graph_y_label("d(BKG)/dE", rsplot::YAxis::Left);
                if let Some(bkg) = &g.bkg {
                    let d1 = deriv(bkg, &g.energy);
                    self.plot
                        .add_curve_with_legend(&g.energy, &d1, BLUE, &g.label);
                }
            }
            GraphType::BkgKDeriv => {
                // BKG(k)': derivative in k of the k-space background. The k-space
                // background is μ0−μ = −edge_step·χ on the k grid (the `k.bkg`
                // file's quantity).
                self.plot.set_graph_x_label("k (Å⁻¹)");
                self.plot
                    .set_graph_y_label("d(BKG)/dk", rsplot::YAxis::Left);
                if let (Some(k), Some(chi), Some(step)) = (&g.k, &g.chi, g.edge_step) {
                    let bkg_k: Vec<f64> = chi.iter().map(|&c| -step * c).collect();
                    let d1 = deriv(&bkg_k, k);
                    self.plot.add_curve_with_legend(k, &d1, BLUE, &g.label);
                }
            }
            GraphType::MuBkgEDeriv => {
                // XMU' + BKG(E)': raw-μ derivative and background derivative
                // overlaid (compare where the spline tracks the edge).
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot
                    .set_graph_y_label("dμ/dE  ·  d(BKG)/dE", rsplot::YAxis::Left);
                if !g.energy.is_empty() && !g.mu.is_empty() {
                    let dmu = deriv(&g.mu, &g.energy);
                    self.plot
                        .add_curve_with_legend(&g.energy, &dmu, BLUE, &g.label);
                }
                if let Some(bkg) = &g.bkg {
                    let dbkg = deriv(bkg, &g.energy);
                    self.plot
                        .add_curve_with_legend(&g.energy, &dbkg, ORANGE, "BKG(E)'");
                }
            }
            GraphType::BkgEDeriv2 => {
                // BKG(E)'': second derivative of the background μ0(E).
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot
                    .set_graph_y_label("d²(BKG)/dE²", rsplot::YAxis::Left);
                if let Some(bkg) = &g.bkg {
                    let d1 = deriv(bkg, &g.energy);
                    let d2 = deriv(&d1, &g.energy);
                    self.plot
                        .add_curve_with_legend(&g.energy, &d2, BLUE, &g.label);
                }
            }
            GraphType::Norm => {
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot
                    .set_graph_y_label("normalized μ(E)", rsplot::YAxis::Left);
                // Athena/XAFSView convention: the "normalized" view shows the
                // *flattened* μ (post-edge curvature removed, lifted to ~1),
                // not the plain edge-step normalization (which keeps the
                // post-edge slope). Fall back to `norm` only if flattening is
                // somehow unavailable — `reduce` always sets them together.
                if let Some(flat) = g.flat.as_ref().or(g.norm.as_ref()) {
                    self.plot
                        .add_curve_with_legend(&g.energy, flat, BLUE, &g.label);
                }
            }
        }
    }

    /// The Feffit tab: fit controls on the left, the data-vs-model plot filling
    /// the centre. The single editor *is* the batch configuration; "Batch…" opens
    /// the batch in a detached window. Plot Data is its own detached window
    /// (opened from the top bar / "Send to Plot Data"), independent of this graph.
    fn feffit_tab(&mut self, ui: &mut egui::Ui) {
        let mut feffit_action = None;
        // Hide Log / Load result don't map to the engine and are omitted per the
        // functional-only field rule; "Send to plot data" opens the group's Plot
        // Data overlay.
        egui::Panel::left("feffit_controls")
            .resizable(true)
            .default_size(380.0)
            .show(ui, |ui| {
                // The action rows, pinned below the scrolling form so the
                // primary Run stays visible however long the form is (the
                // Autobk tab pins its Start row the same way).
                egui::Panel::bottom("feffit_actions").show(ui, |ui| {
                    ui.add_space(4.0);
                    feffit_action = self.feffit.action_row(ui);
                    ui.add_space(2.0);
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        // `controls` hosts the Graph section's "Show all fits" checkbox
                        // (shown only for >1 group); computing the group count up front
                        // lets it gate that checkbox and this Data… row alike.
                        let n_groups = self.session.groups.len();
                        // The pinned row rendered first; don't let a None from the
                        // scrolling form overwrite its click.
                        feffit_action = self
                            .feffit
                            .controls(
                                ui,
                                &mut self.feffit_show_all_fits,
                                n_groups > 1,
                                self.feffit_fit_group.as_deref(),
                                self.session.current_group().map(|g| g.label.as_str()),
                            )
                            .or(feffit_action.take());
                        // Exit sits directly below the controls (그림 1-2-2-2), no gap.
                        ui.add_space(6.0);
                        // Run fits every checked group (shared with the Autobk tab); this
                        // row shows and opens that selection.
                        if n_groups > 0 {
                            ui.horizontal(|ui| {
                                if ui
                                    .button("Data…")
                                    .on_hover_text(
                                        "Choose which groups Run fits (shared with the Autobk tab)",
                                    )
                                    .clicked()
                                {
                                    self.data_window_open = true;
                                    self.data_window_reveal = true;
                                }
                                ui.weak(format!(
                                    "{}/{n_groups} checked for Run",
                                    self.session.selected.len()
                                ));
                            });
                        }
                        ui.horizontal(|ui| {
                            // The batch — an occasional bulk op — opens in its own
                            // window (data_manager-style) so the central plot keeps
                            // the full width.
                            if ui
                                .button("Batch…")
                                .on_hover_text(
                                    "Run this fit setup against several checked groups, \
                                 in a detached window",
                                )
                                .clicked()
                            {
                                self.feffit_batch_open = true;
                            }
                        });
                    });
                });
            });
        // Draggable FT-window band, the Feffit twin of the Autobk tab's band:
        // kmin/kmax on the k / q graphs, rmin/rmax on the R graph. Shown only
        // while "Show k/R range" is on (off by default). A finished edge drag
        // writes the new bounds back onto the FT window (same as editing the
        // kmin/kmax/rmin/rmax fields); the values apply on the next Run.
        if self.feffit.show_range {
            let (min, max) = self.feffit.range_window();
            crate::plot::set_window(&mut self.plot, crate::plot::AxisWindow { min, max });
        }
        self.plot
            .set_empty_hint("No fit yet — add Feff paths and press Run.");
        egui::CentralPanel::default().show(ui, |ui| {
            crate::plot::show(&mut self.plot, ui);
        });
        if self.feffit.show_range
            && let Some(w) = crate::plot::take_window_drag(&mut self.plot)
        {
            // A left-edge drag past the right (or vice versa) inverts the pair;
            // re-order so min < max and keep the lower bound off negative k/R.
            let (lo, hi) = if w.min <= w.max {
                (w.min, w.max)
            } else {
                (w.max, w.min)
            };
            self.feffit.set_range_window(lo.max(0.0), hi);
        }

        match feffit_action {
            Some(FeffitAction::AddPath) => self.add_feff_path(),
            Some(FeffitAction::Run) => self.run_feffit(),
            Some(FeffitAction::Replot) => self.replot_feffit(),
            Some(FeffitAction::SendToPlotData) => self.send_feffit_to_plot_data(),
            Some(FeffitAction::SaveResult) => self.save_feffit_result(),
            Some(FeffitAction::LoadResult) => self.load_feffit_result(),
            Some(FeffitAction::LoadInp) => self.load_feffit_inp(),
            None => {}
        }
    }

    /// The detached "Feffit — Batch" window: the batch panel moved out of the
    /// Feffit tab's right side (the tab keeps only a "Batch…" button). Runs the
    /// Feffit tab's current config against the checked groups and tabulates /
    /// saves the results. Returns a [`BatchAction`] for app-owned disk writes.
    fn feffit_batch_window(&mut self, ctx: &egui::Context) -> Option<BatchAction> {
        if !self.feffit_batch_open {
            return None;
        }
        let mut keep_open = true;
        let mut action = None;
        crate::window::detached(
            ctx,
            "feffit_batch",
            "Feffit — Batch",
            &mut keep_open,
            [360.0, 520.0],
            |ui| {
                action = self.feffit_batch.panel(
                    ui,
                    &self.session.groups,
                    &mut self.session.selected,
                    &self.feffit,
                );
            },
        );
        self.feffit_batch_open = keep_open;
        action
    }

    /// The Fit report tab (the original's Feffit_txt): a plain-text report of
    /// the last fit.
    fn feffit_txt_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Fit report");
            ui.add_space(8.0);
            ui.weak("Text report of the last FEFFIT fit");
        });
        ui.separator();
        let report = self.feffit.report_text();
        if report.is_empty() {
            ui.weak("Run a fit on the Feffit tab to see the report.");
            if ui.button("Go to Feffit").clicked() {
                self.tab = Tab::Feffit;
            }
            return;
        }
        egui::ScrollArea::both().show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut report.as_str())
                    .font(egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY)
                    .desired_rows(28),
            );
        });
    }

    /// The About window (Help menu): program/version information.
    fn about_window(&mut self, ctx: &egui::Context) {
        crate::window::detached(
            ctx,
            "about",
            "About XAFSView",
            &mut self.about_open,
            [440.0, 200.0],
            |ui| {
                ui.heading("XAFSView");
                ui.label(format!(
                    "feffit-rs port · version {}",
                    env!("CARGO_PKG_VERSION")
                ));
                ui.separator();
                ui.label(
                    "A Rust reimplementation of the XAFSView GUI on the feffit-rs engines \
                     (pre-edge / normalize, AUTOBK, FEFFIT, LCF / PCA, FEFF8L / FEFF10), \
                     with larch-parity math.",
                );
            },
        );
    }

    /// Open a dialog and add the chosen Feff path file(s) to the fit.
    fn add_feff_path(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("FEFF path (feffNNNN.dat)", &["dat"]);
        if let Some(dir) = &self.session.folders.atoms_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(paths) = dlg.pick_files() else {
            return;
        };
        let mut added = 0usize;
        for path in paths {
            match feffit::feffdat::FeffPath::from_path(&path) {
                Ok(fp) => {
                    let label = path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "path".to_owned());
                    self.feffit.add_path(label, fp);
                    added += 1;
                }
                Err(e) => self.status.error(format!("Path load failed: {e}")),
            }
        }
        if added > 0 {
            self.status.info(format!("Added {added} Feff path(s)."));
        }
    }

    /// Write the batch "Save Items" files to the Results folder (the original's
    /// results folder; falls back to the data folder), reporting the outcome.
    fn write_saved_items(&mut self, files: Vec<(String, String)>) {
        let Some(dir) = self
            .session
            .folders
            .results_dir
            .clone()
            .or_else(|| self.session.folders.data_dir.clone())
        else {
            self.status
                .error("Set a Results folder (Folders tab) before saving items.");
            return;
        };
        let mut written = 0usize;
        let mut failed = 0usize;
        for (name, content) in &files {
            match write_file(&dir.join(name), content) {
                Ok(()) => written += 1,
                Err(e) => {
                    failed += 1;
                    // Per-file reason into the log; the summary line below only
                    // carries the count.
                    self.status.detail_error(format!("{name}: {e}"));
                }
            }
        }
        let extra = if failed > 0 {
            format!(" ({failed} failed)")
        } else {
            String::new()
        };
        let msg = format!("Saved {written} item file(s) to {}{extra}.", dir.display());
        if failed > 0 {
            self.status.error(msg);
        } else {
            self.status.info(msg);
        }
    }

    /// Run the FEFFIT fit on the current group's `chi(k)` and redraw.
    /// Fit the Feffit tab's current config against every group in the run set
    /// ([`Session::run_targets`]: the checked groups, or the current group when
    /// nothing is checked), writing each group's `.dat`/`.fit` transforms. Every
    /// fit runs on an independent config clone on a background job (the editor
    /// stays usable meanwhile); [`finish_feffit`] then adopts the focused fit —
    /// the current group when it is in the set, else the first target — into
    /// `self.feffit` so the tab graph shows it in detail.
    ///
    /// [`finish_feffit`]: Self::finish_feffit
    fn run_feffit(&mut self) {
        let targets = self.session.run_targets();
        if targets.is_empty() {
            self.status
                .error("No group to fit — load data and run AUTOBK first.");
            return;
        }
        let focused = match self.session.current {
            Some(c) if targets.contains(&c) => c,
            _ => targets[0],
        };
        // Snapshot each target's χ(k) for the worker; the ones without stay
        // behind as a count for the summary.
        let mut items: Vec<(usize, String, Vec<f64>, Vec<f64>)> = Vec::new();
        let mut no_chi = 0usize;
        for &idx in &targets {
            match self
                .session
                .groups
                .get(idx)
                .and_then(|g| match (&g.k, &g.chi) {
                    (Some(k), Some(chi)) => Some((g.label.clone(), k.clone(), chi.clone())),
                    _ => None,
                }) {
                Some((label, k, chi)) => items.push((idx, label, k, chi)),
                None => no_chi += 1,
            }
        }
        if items.is_empty() {
            self.status.error(if targets.len() == 1 {
                "No χ(k) for the group — run AUTOBK first.".to_owned()
            } else {
                format!(
                    "Feffit: fit 0/{} group(s); {no_chi} without χ(k)",
                    targets.len()
                )
            });
            return;
        }
        let template = self.feffit.config_clone();
        let targets_len = targets.len();
        let total = items.len();
        let started = self.jobs.start("Feffit", total, move |job| {
            // Each target fits an independent clone of the config (same
            // guarantee the batch panel relies on) — including the focused one,
            // whose fitted clone the apply adopts into the editor.
            let mut fits: Vec<FeffitJobFit> = Vec::with_capacity(items.len());
            for (done, (idx, label, k, chi)) in items.into_iter().enumerate() {
                let mut cfg = template.config_clone();
                let outcome = cfg.run(&k, &chi).map(|msg| (msg, cfg));
                fits.push((idx, label, outcome));
                job.progress(done + 1);
            }
            Box::new(move |app: &mut Self| app.finish_feffit(focused, targets_len, no_chi, fits))
        });
        if !started {
            self.status.error(JOB_BUSY);
        }
    }

    /// Apply a finished Feffit job: adopt the focused fit into the editor,
    /// write each fitted group's `.dat`/`.fit` transforms, rebuild the overlay
    /// list, and report the summary.
    fn finish_feffit(
        &mut self,
        focused: usize,
        targets_len: usize,
        no_chi: usize,
        fits: Vec<FeffitJobFit>,
    ) {
        // Collect every fitted group's plot so a multi-group run can overlay all
        // fits (the AUTOBK "Show all groups" twin). Rebuilt each run.
        self.feffit_run_overlays.clear();
        let mut fit_ok = 0usize;
        let mut saved = 0usize;
        let mut first_err: Option<String> = None;
        // For a single-target run, keep the original detailed status (fit summary
        // plus the write outcome); a multi-target run summarises counts.
        let mut focused_msg = String::new();
        let mut focused_note = String::new();
        for (idx, label, outcome) in fits {
            // The provenance header reads the group by index — re-resolve by
            // label in case the group list changed while the job ran.
            let idx_now = self
                .session
                .groups
                .iter()
                .position(|g| g.label == label)
                .unwrap_or(idx);
            match outcome {
                Ok((msg, cfg)) => {
                    fit_ok += 1;
                    // The data .dat files always write; "Only FT" has no model,
                    // so output_pairs omits the .fit files. Keep a plot snapshot
                    // for the overlay (owned, so the write + push don't clash).
                    let plot = cfg.plot().cloned();
                    if idx == focused {
                        focused_msg = msg;
                        self.feffit_fit_group = Some(label.clone());
                        // The fit ran on a clone; adopt its result (and its
                        // freshness snapshot — the stale badge stays correct
                        // even if the config was edited while the fit ran).
                        self.feffit.adopt_run(cfg);
                    }
                    if let Some(plot) = plot {
                        match self.write_feffit_outputs(&plot, &label, idx_now) {
                            Ok(n) => {
                                saved += 1;
                                if idx == focused {
                                    focused_note = format!(" · saved {n} .dat/.fit files");
                                }
                            }
                            Err(e) => {
                                // The fit succeeded but its files did not land —
                                // classify the run as failed.
                                let m = format!(".dat/.fit not written: {e}");
                                self.status.detail_error(format!("{label}: {m}"));
                                if idx == focused {
                                    focused_note = format!(" · {m}");
                                }
                                first_err.get_or_insert(m);
                            }
                        }
                        self.feffit_run_overlays.push((label.clone(), plot));
                    }
                }
                Err(e) => {
                    if idx == focused {
                        focused_msg = e.clone();
                    }
                    // Per-group reason into the log; the summary line below
                    // only carries the first error.
                    self.status.detail_error(format!("{label}: {e}"));
                    first_err.get_or_insert(e);
                }
            }
        }
        let failed = first_err.is_some() || (targets_len > 1 && no_chi > 0);
        let msg = if targets_len == 1 {
            if fit_ok == 1 {
                format!("{focused_msg}{focused_note}")
            } else {
                first_err
                    .clone()
                    .unwrap_or_else(|| "No χ(k) for the group — run AUTOBK first.".to_owned())
            }
        } else {
            let mut m = format!(
                "Feffit: fit {fit_ok}/{targets_len} group(s); saved {saved} .dat/.fit set(s)"
            );
            if no_chi > 0 {
                m.push_str(&format!("; {no_chi} without χ(k)"));
            }
            if let Some(e) = &first_err {
                m.push_str(&format!("; first error: {e}"));
            }
            m
        };
        // Answer a parked MCP run_fit with the summary line plus, when a fit
        // produced one, the full report.
        if let Some(reply) = self.mcp_job_reply.take() {
            let _ = reply.send(if failed || fit_ok == 0 {
                Err(msg.clone())
            } else {
                let report = self.feffit.report_text();
                Ok(if report.is_empty() {
                    msg.clone()
                } else {
                    format!("{msg}\n\n{report}")
                })
            });
        }
        if failed || fit_ok == 0 {
            self.status.error(msg);
        } else {
            self.status.info(msg);
        }
        // Show the whole batch at once when several were fit (AUTOBK parity).
        if targets_len > 1 {
            self.feffit_show_all_fits = true;
        }
        self.replot_feffit();
    }

    /// The batch panel's "Run all" on the background-job slot: fit each checked
    /// group with the Feffit tab's current config on the worker
    /// ([`feffit_batch::compute_runs`]), then adopt the runs on the main thread
    /// (`set_runs` marks the graph overlay dirty; the app replots it).
    ///
    /// [`feffit_batch::compute_runs`]: crate::feffit_batch::compute_runs
    fn start_feffit_batch(&mut self) {
        // The batch is the session's checked-for-Run set; "Run all" is disabled
        // while it is empty, so the current-group fallback of run_targets never
        // widens it here.
        let idxs = self.session.run_targets();
        if idxs.is_empty() {
            return;
        }
        let groups = self.session.groups.clone();
        let template = self.feffit.config_clone();
        let total = idxs.len();
        let started = self.jobs.start("Feffit batch", total, move |job| {
            let runs = crate::feffit_batch::compute_runs(idxs, &groups, &template, &|done| {
                job.progress(done)
            });
            Box::new(move |app: &mut Self| {
                let (total, errors) = app.feffit_batch.set_runs(runs);
                let msg = format!("Feffit batch: fit {}/{total} group(s).", total - errors);
                if errors > 0 {
                    app.status.error(msg);
                } else {
                    app.status.info(msg);
                }
            })
        });
        if !started {
            self.status.error(JOB_BUSY);
        }
    }

    /// Redraw the shared plot with the last fit's data vs model in the selected
    /// space/part.
    fn replot_feffit(&mut self) {
        // The Feffit tab's own graph: the focused fit's data-vs-model base curves,
        // plus every batch run overlaid after a Feffit batch (like Multiple_data
        // on Autobk). Plot Data is a separate window now and draws its loaded files
        // onto its own graph.
        self.plot.clear_curves();
        self.replot_feffit_base();
        // The Feffit-tab "Show all fits" overlay (AUTOBK "Show all groups" twin):
        // every group fitted by the last tab Run.
        if self.feffit_show_all_fits {
            self.overlay_tab_run_fits();
        }
        if self.feffit_batch.show_on_graph() {
            self.overlay_feffit_runs();
        }
    }

    /// Overlay every Feffit-batch run's data + model curves for the selected
    /// space/part onto the shared graph — the batch-comparison view. The tab's own
    /// focused fit is already drawn by [`replot_feffit_base`], so the run matching
    /// its group is skipped; the others use palette colours past the focused fit's,
    /// each legended by group label (`"<label> data"` / `"<label> fit"`).
    fn overlay_feffit_runs(&mut self) {
        let (space, part) = self.feffit.plot_selection();
        let focused = self.feffit_fit_group.as_deref();
        Self::overlay_fit_plots(
            &mut self.plot,
            space,
            part,
            focused,
            self.feffit_batch.runs_with_plots(),
        );
    }

    /// Overlay every fit from the last Feffit tab "Run" — the AUTOBK "Show all
    /// groups" twin for the Feffit tab. The focused group is drawn prominently by
    /// [`replot_feffit_base`] and skipped here.
    fn overlay_tab_run_fits(&mut self) {
        let (space, part) = self.feffit.plot_selection();
        let focused = self.feffit_fit_group.as_deref();
        Self::overlay_fit_plots(
            &mut self.plot,
            space,
            part,
            focused,
            self.feffit_run_overlays
                .iter()
                .map(|(l, p)| (l.as_str(), p)),
        );
    }

    /// Overlay a set of fits `(label, plot)` onto the shared Feffit graph in the
    /// selected `space`/`part`, each in a palette colour and legended by label.
    /// The `focused` group is skipped — its base curves are already drawn. Shared
    /// by the tab-Run overlay ([`overlay_tab_run_fits`]) and the batch overlay
    /// ([`overlay_feffit_runs`]).
    fn overlay_fit_plots<'a>(
        plot: &mut crate::plot::Plot,
        space: crate::feffit_ui::PlotSpace,
        part: crate::feffit_ui::PlotPart,
        focused: Option<&str>,
        runs: impl Iterator<Item = (&'a str, &'a FeffitPlot)>,
    ) {
        let palette = &crate::plot::PALETTE;
        let mut j = 0usize;
        for (label, p) in runs {
            if Some(label) == focused {
                continue;
            }
            let (x, data_y, model_y, _xl, _yl) = p.series(space, part);
            if x.is_empty() {
                continue;
            }
            // Skip the first two palette slots (the focused fit's data/model
            // colours) so an overlaid curve never matches the focused fit.
            let color = palette[2 + j % (palette.len() - 2)];
            plot.add_curve_with_legend(&x, &data_y, color, format!("{label} data"));
            if p.has_model {
                plot.add_curve_with_legend(&x, &model_y, color, format!("{label} fit"));
            }
            j += 1;
        }
    }

    /// Draw the Feffit tab's own data-vs-model base curves into the
    /// (already-cleared) shared plot. Called only by [`replot_feffit`], which
    /// then appends the Plot Data overlay.
    fn replot_feffit_base(&mut self) {
        use crate::feffit_ui::PlotSpace;
        use crate::plot::{GREEN, ORANGE};
        use crate::plot_data::{FIT_DATA, FIT_MODEL};

        let (space, part) = self.feffit.plot_selection();
        let Some(p) = self.feffit.plot() else {
            self.plot.set_graph_title("XAFSView");
            // No fit yet: still label the axes for the selected fit space, so
            // the empty plot doesn't carry the previous tab's μ(E)/Energy axes.
            let (xl, yl) = crate::feffit_ui::space_labels(space, part);
            self.plot.set_graph_x_label(xl);
            self.plot.set_graph_y_label(yl, rsplot::YAxis::Left);
            return;
        };
        // Same orientation rule as the Autobk graph: name the focused fit.
        self.plot
            .set_graph_title(self.feffit_fit_group.as_deref().unwrap_or("Feffit"));
        let has_model = p.has_model;

        // K+Q: kʷ·χ(k) and χ(q) overlaid on the shared Å⁻¹ axis (four curves).
        if space == PlotSpace::KQ {
            let ((kx, kd, km), (qx, qd, qm)) = p.kq_series(part);
            let (xl, yl) = crate::feffit_ui::space_labels(space, part);
            self.plot.set_graph_x_label(xl);
            self.plot.set_graph_y_label(yl, rsplot::YAxis::Left);
            // Legend each focused curve with the group name (like the overlaid
            // fits), keeping the k/q + data/fit qualifier to tell the four apart.
            let (dk, mk, dq, mq) = match self.feffit_fit_group.as_deref() {
                Some(n) => (
                    format!("{n} data k"),
                    format!("{n} fit k"),
                    format!("{n} data q"),
                    format!("{n} fit q"),
                ),
                None => (
                    "data k".to_owned(),
                    "model k".to_owned(),
                    "data q".to_owned(),
                    "model q".to_owned(),
                ),
            };
            if !kx.is_empty() {
                self.plot.add_curve_with_legend(&kx, &kd, FIT_DATA, dk);
                if has_model {
                    self.plot.add_curve_with_legend(&kx, &km, FIT_MODEL, mk);
                }
            }
            if !qx.is_empty() {
                self.plot.add_curve_with_legend(&qx, &qd, GREEN, dq);
                if has_model {
                    self.plot.add_curve_with_legend(&qx, &qm, ORANGE, mq);
                }
            }
            return;
        }

        let (x, data_y, model_y, xlabel, ylabel) = p.series(space, part);
        self.plot.set_graph_x_label(xlabel);
        self.plot.set_graph_y_label(ylabel, rsplot::YAxis::Left);
        // Legend the focused fit with its group name (like the overlaid fits), so
        // every trace is identified by file name — not a bare "data"/"model".
        let (data_legend, model_legend) = match self.feffit_fit_group.as_deref() {
            Some(n) => (format!("{n} data"), format!("{n} fit")),
            None => ("data".to_owned(), "model".to_owned()),
        };
        if !x.is_empty() {
            self.plot
                .add_curve_with_legend(&x, &data_y, FIT_DATA, data_legend);
            // "Only FT" mode has no model — draw the data alone.
            if has_model {
                self.plot
                    .add_curve_with_legend(&x, &model_y, FIT_MODEL, model_legend);
            }
        }
    }

    /// Hand the last Feffit fit's data + model curves to the Plot Data panel
    /// (the Feffit form's "Send to plot data"), in the currently-selected space.
    fn send_feffit_to_plot_data(&mut self) {
        let label = match &self.feffit_fit_group {
            Some(g) => format!("Feffit fit — {g}"),
            None => "Feffit fit".to_owned(),
        };
        let (space, part) = self.feffit.plot_selection();
        let Some(p) = self.feffit.plot() else {
            self.status
                .error("Run Feffit first, then send it to Plot Data.");
            return;
        };
        let has_model = p.has_model;
        // Plot Data's own graph owns the axis labels; the fit's own labels are
        // dropped — it is drawn on top of the loaded files on that window's axes.
        let (x, data_y, model_y, _xlabel, _ylabel) = p.series(space, part);
        // "Only FT" has no model — send an empty model so Plot Data draws data alone.
        let model_y = if has_model { model_y } else { Vec::new() };
        // A k-space series is kʷ·χ(k) at the fit's k-weight; Plot Data re-weights
        // it to its own G_kweight (the K+Q selection sends the k series).
        use crate::feffit_ui::PlotSpace;
        let kweight = matches!(space, PlotSpace::K | PlotSpace::KQ).then_some(p.kweight);
        self.plot_data
            .set_fit_overlay(label, x, data_y, model_y, kweight);
        // Pop the Plot Data window so the sent fit is visible right away.
        self.plot_data.open();
        self.status
            .info("Sent the Feffit fit (data + model) to Plot Data.");
    }

    /// Save the current fit report to a text file (the original's "Save result").
    fn save_feffit_result(&mut self) {
        let text = self.feffit.report_text();
        if text.is_empty() {
            self.status
                .error("Run a fit first — there is no result to save.");
            return;
        }
        let stem = self.feffit_fit_group.as_deref().unwrap_or("feffit");
        let mut dlg = rfd::FileDialog::new()
            .add_filter("Text", &["txt", "log", "out"])
            .set_file_name(format!("{stem}.txt"));
        if let Some(dir) = self
            .session
            .folders
            .feffit_dir
            .clone()
            .or_else(|| self.session.folders.results_dir.clone())
        {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.save_file() else {
            self.status.info("Save result cancelled.");
            return;
        };
        match std::fs::write(&path, text) {
            Ok(()) => self
                .status
                .info(format!("Saved fit result to {}", path.display())),
            Err(e) => self.status.error(format!("Could not save result: {e}")),
        }
    }

    /// Load a saved result/text file into the Feffit pop-up viewer ("Load result").
    fn load_feffit_result(&mut self) {
        let mut dlg =
            rfd::FileDialog::new().add_filter("Text", &["txt", "log", "out", "fit", "dat"]);
        if let Some(dir) = self
            .session
            .folders
            .feffit_dir
            .clone()
            .or_else(|| self.session.folders.results_dir.clone())
        {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.pick_file() else {
            self.status.info("Load result cancelled.");
            return;
        };
        match feffit::textio::read_to_string_lenient(&path) {
            Ok(content) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "result".to_owned());
                self.feffit.show_text(format!("Loaded: {name}"), content);
                self.status
                    .info(format!("Loaded result from {}", path.display()));
            }
            Err(e) => self.status.error(format!("Could not load result: {e}")),
        }
    }

    /// Import a UWXAFS `feffit.inp`: parse it, apply its FT windows / fit mode /
    /// user functions / variables to the Feffit tab, then load each path's
    /// `feffNNNN.dat` (resolved relative to the `.inp`, falling back to the Atoms
    /// folder by file name) with its per-parameter expression overrides.
    fn load_feffit_inp(&mut self) {
        let mut dlg = rfd::FileDialog::new().add_filter("Feffit input", &["inp"]);
        if let Some(dir) = self
            .session
            .folders
            .feffit_dir
            .clone()
            .or_else(|| self.session.folders.sub_base.clone())
        {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.pick_file() else {
            self.status.info("Load inp cancelled.");
            return;
        };
        let content = match feffit::textio::read_to_string_lenient(&path) {
            Ok(c) => c,
            Err(e) => {
                self.status.error(format!("Could not read inp: {e}"));
                return;
            }
        };

        let inp = crate::feffit_ui::parse_feffit_inp(&content);
        let nvars = inp.vars.len();
        let npaths = inp.paths.len();
        let base = path
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();
        self.feffit.apply_inp(&inp);

        let mut loaded = 0usize;
        let mut missing = 0usize;
        for ip in &inp.paths {
            let candidate = base.join(ip.file.replace('\\', "/"));
            let resolved = if candidate.exists() {
                Some(candidate)
            } else {
                // Fall back to the Atoms folder, matched by file name.
                candidate
                    .file_name()
                    .and_then(|name| {
                        self.session
                            .folders
                            .atoms_dir
                            .as_ref()
                            .map(|d| d.join(name))
                    })
                    .filter(|p| p.exists())
            };
            match resolved.and_then(|p| {
                feffit::feffdat::FeffPath::from_path(&p)
                    .ok()
                    .map(|fp| (p, fp))
            }) {
                Some((p, fp)) => {
                    let label = p
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| format!("path{}", ip.number));
                    self.feffit.add_inp_path(label, fp, ip);
                    loaded += 1;
                }
                None => {
                    missing += 1;
                    // Per-path reason into the log; the summary line below only
                    // carries the count.
                    self.status
                        .detail_error(format!("{}: feff path file not found", ip.file));
                }
            }
        }

        if missing == 0 {
            self.status.info(format!(
                "Loaded inp: {nvars} variable(s), {loaded}/{npaths} path(s)."
            ));
        } else {
            self.status.error(format!(
                "Loaded inp: {nvars} variable(s), {loaded}/{npaths} path(s) ({missing} file(s) not found)."
            ));
        }
    }

    /// The compact current-group selector in the Autobk sidebar: a combo to
    /// switch the active spectrum inline (the most frequent action during
    /// reduction) plus a "Data…" button that opens the full [`data_manager_window`]
    /// (the loaded-group list + Add + Clear all). Returns `true` when the
    /// selection changed (caller replots).
    ///
    /// [`data_manager_window`]: Self::data_manager_window
    fn current_group_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let n = self.session.groups.len();
        let mut changed = false;
        ui.horizontal(|ui| {
            form::label(ui, INFO_LABEL_W, "Group");
            let current = self.session.current;
            let text = current
                .and_then(|i| self.session.groups.get(i))
                .map(|g| g.label.as_str())
                .unwrap_or("— no data —");
            let mut select = None;
            egui::ComboBox::from_id_salt("current_group_combo")
                .selected_text(text)
                .width(180.0)
                .show_ui(ui, |ui| {
                    for (i, g) in self.session.groups.iter().enumerate() {
                        if ui.selectable_label(current == Some(i), &g.label).clicked() {
                            select = Some(i);
                        }
                    }
                });
            if let Some(i) = select {
                self.session.current = Some(i);
                changed = true;
            }
            if ui
                .button(format!("Data… ({n})"))
                .on_hover_text("Open the data manager: add, remove, clear, or check groups for Run")
                .clicked()
            {
                self.data_window_open = true;
                self.data_window_reveal = true;
            }
        });
        // The shared Run scope, so it is visible without opening the Data window.
        // (The graph-display toggles — Show all groups / Stacking / Show background
        // — moved to the Autobk tab's "Graph" box, mirroring the Feffit panel.)
        if n > 0 {
            ui.weak(format!(
                "{}/{n} checked for Run",
                self.session.selected.len()
            ));
        }
        changed
    }

    /// The detached "Data" window: the loaded-group manager moved out of the
    /// Autobk sidebar (which keeps only the compact [`current_group_selector`]).
    /// Hosts "Add data files…", every loaded group as a click-to-select row with
    /// a ✕ to remove it, and "Clear all". Groups are otherwise append-only, so
    /// this is the one place to add, switch, or drop them. Returns `true` when
    /// the selection changed (caller replots).
    ///
    /// [`current_group_selector`]: Self::current_group_selector
    fn data_manager_window(&mut self, ctx: &egui::Context) -> bool {
        if !self.data_window_open {
            return false;
        }
        let mut keep_open = true;
        let mut changed = false;
        let mut add_clicked = false;
        let reveal = self.data_window_reveal;
        // Own panel layout: the group list fills the window, so Add + the
        // checked-count row go in a top panel and "Clear all" in a pinned footer
        // (in one flow the list pushed "Clear all" past the viewport edge).
        crate::window::detached_panels(
            ctx,
            "data_manager",
            "Data",
            &mut keep_open,
            [320.0, 420.0],
            |ui| {
                let n = self.session.groups.len();
                let current = self.session.current;
                let sel_count = self.session.selected.len();
                // Snapshot the checked state so the row loop can read it while
                // iterating `groups` (a distinct field); mutations are deferred.
                let checked: Vec<bool> = (0..n).map(|i| self.session.is_selected(i)).collect();
                let mut select = None;
                let mut remove = None;
                let mut toggle = None;
                let mut select_all = false;
                let mut clear_sel = false;
                let mut clear_all = false;
                let mut revealed = false;
                egui::Panel::top("data_manager_header").show(ui, |ui| {
                    // ADD opens the Plot Data-style staging picker (needs an open
                    // import for the column mapping); same affordance as Plot Data.
                    if ui
                        .button("Add data files…")
                        .on_hover_text("Stage several raw scan files, then build μ(E) for each")
                        .clicked()
                    {
                        add_clicked = true;
                    }
                    if n > 0 {
                        ui.separator();
                        // The shared "selected file list": which groups each tab's
                        // Run (Autobk reduce / Feffit fit) acts on. Check none →
                        // Run falls back to the current group.
                        ui.horizontal(|ui| {
                            ui.strong(format!("{sel_count} of {n} checked for Run"));
                            if ui.small_button("All").clicked() {
                                select_all = true;
                            }
                            if ui.small_button("None").clicked() {
                                clear_sel = true;
                            }
                        });
                    }
                });
                if n == 0 {
                    egui::CentralPanel::default().show(ui, |ui| {
                        ui.weak("No data — open a file, then Add data files.");
                    });
                    return;
                }
                crate::window::fill_with_footer(
                    ui,
                    "data_manager",
                    self,
                    |_, ui| {
                        if ui.button("Clear all").clicked() {
                            clear_all = true;
                        }
                    },
                    |s, ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("data_manager_list")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                for (i, g) in s.session.groups.iter().enumerate() {
                                    ui.push_id(i, |ui| {
                                        ui.horizontal(|ui| {
                                            if crate::widgets::delete_box(ui)
                                                .on_hover_text("Remove this data")
                                                .clicked()
                                            {
                                                remove = Some(i);
                                            }
                                            let mut on = checked[i];
                                            if ui
                                                .checkbox(&mut on, "")
                                                .on_hover_text(
                                                    "Include this group in Run — shift-click \
                                                     checks/unchecks the whole range from the \
                                                     last clicked row",
                                                )
                                                .changed()
                                            {
                                                toggle = Some((i, on));
                                            }
                                            let row =
                                                ui.selectable_label(current == Some(i), &g.label);
                                            if reveal && current == Some(i) {
                                                row.scroll_to_me(Some(egui::Align::Center));
                                                revealed = true;
                                            }
                                            if row.clicked() {
                                                select = Some(i);
                                            }
                                        });
                                    });
                                }
                            });
                    },
                );
                if revealed {
                    self.data_window_reveal = false;
                }
                if clear_all {
                    self.session.groups.clear();
                    self.session.current = None;
                    self.session.clear_selected();
                    self.data_check_anchor = None;
                    self.plot_data.mark_dirty();
                    changed = true;
                }
                // Apply the row/scope actions after the read-only iteration above.
                if select_all {
                    self.session.select_all();
                    changed = true;
                }
                if clear_sel {
                    self.session.clear_selected();
                    changed = true;
                }
                if let Some((i, on)) = toggle {
                    // Shift extends the clicked checkbox's new state over the
                    // inclusive range from the anchor (select_index_list's
                    // convention, as a checkbox: shift-uncheck clears a range).
                    let shift = ui.input(|inp| inp.modifiers.shift);
                    match (shift, self.data_check_anchor) {
                        (true, Some(a)) if a < n => {
                            for j in a.min(i)..=a.max(i) {
                                self.session.set_selected(j, on);
                            }
                        }
                        _ => {
                            self.session.set_selected(i, on);
                            self.data_check_anchor = Some(i);
                        }
                    }
                    changed = true;
                }
                if let Some(i) = select {
                    self.session.current = Some(i);
                    changed = true;
                }
                if let Some(i) = remove {
                    self.session.remove_group(i);
                    // Indices shifted under the anchor; a later shift-click
                    // would range over the wrong rows.
                    self.data_check_anchor = None;
                    self.plot_data.mark_dirty();
                    changed = true;
                }
            },
        );
        self.data_window_open = keep_open;
        if add_clicked {
            self.open_new_file();
        }
        changed
    }

    /// The Autobk tab: import + reduction controls on the left, plot on the right.
    fn autobk_tab(&mut self, ui: &mut egui::Ui) {
        let mut open_clicked = false;
        let mut start_clicked = false;
        let mut edit_clicked = false;
        let mut multi_autobk_clicked = false;
        let mut theory_pick = false;
        let mut theory_clear = false;
        let mut change = ControlsChange::default();

        // μ(E)-dependent actions (Autobk Start, Edit μ(E)) need a real spectrum,
        // not a directly-loaded χ(k) group.
        let has_mu = self
            .session
            .current_group()
            .is_some_and(|g| !g.mu.is_empty());
        // "Autobk Start" can run even before Calc XMU: with a file open but no
        // μ(E) yet, it builds μ(E) first (auto Calc XMU) so one click does the
        // whole reduction. Edit μ(E) keeps the stricter `has_mu` gate — it edits
        // an existing spectrum, so there is nothing to auto-build.
        let import_open = self.import.is_some();
        let start_autocalc = !has_mu && import_open;
        let can_start = has_mu || import_open;
        let data_file = self
            .session
            .current_group()
            .and_then(|g| g.filename.as_ref())
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned());
        let theory_name = self
            .reduction
            .theory
            .as_ref()
            .and_then(|t| t.path.file_name())
            .map(|s| s.to_string_lossy().into_owned());

        egui::Panel::left("autobk_controls")
            .resizable(true)
            .default_size(360.0)
            .show(ui, |ui| {
                // The action row, pinned below the scrolling form so the
                // primary Start stays visible however long the form is.
                // Wrapped: at a narrow panel width the buttons flow onto a
                // second row instead of clipping.
                egui::Panel::bottom("autobk_actions").show(ui, |ui| {
                    ui.add_space(4.0);
                    use crate::widgets::{self, ROW_BTN};
                    ui.horizontal_wrapped(|ui| {
                        if widgets::action(ui, "Open New file", ROW_BTN)
                            .on_hover_text(
                                "Open the raw-scan loader: stage files, set columns, \
                                 then Autobk Start builds μ(E) for each and reduces.",
                            )
                            .clicked()
                        {
                            open_clicked = true;
                        }
                        let start = widgets::primary(ui, "Autobk Start", ROW_BTN, can_start);
                        let start = if start_autocalc {
                            start.on_hover_text("No μ(E) yet — builds it, then AUTOBK")
                        } else {
                            start
                        };
                        if start.clicked() {
                            start_clicked = true;
                        }
                        if widgets::action_enabled(ui, "Edit μ(E)", ROW_BTN, has_mu).clicked() {
                            edit_clicked = true;
                        }
                        // Batch reduce: normalize → AUTOBK → FT every loaded
                        // group and save each χ(k)/χ(R) (moved here from the
                        // former "Multiple_data" menu).
                        let many = self.session.groups.len() > 1;
                        if widgets::action_enabled(ui, "Multiple AUTOBK", ROW_BTN, many)
                            .on_hover_text(
                                "Reduce every loaded group and save each \
                                 χ(k)/χ(R) file",
                            )
                            .clicked()
                        {
                            multi_autobk_clicked = true;
                        }
                    });
                    ui.add_space(2.0);
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Autobk");

                        // --- Data: active group, per-group info, loading mode ---
                        // Compact current-group selector; the full loaded-group
                        // manager (add / remove / clear) lives in the detached
                        // "Data" window opened from here. A switch repopulates the
                        // plot (deferred to `group_switched` below so the replot is
                        // not called from inside the box closure).
                        let mut group_switched = false;
                        rsplot::widget::form::group(ui, |ui| {
                            ui.strong("Data");
                            group_switched = self.current_group_selector(ui);

                            // info rows: Title (group label) / Data File / Theory —
                            // form rows with a fixed label column (QFormLayout
                            // look): the field grows with the panel.
                            ui.horizontal(|ui| {
                                form::label(ui, INFO_LABEL_W, "Title");
                                match self.session.current_group_mut() {
                                    Some(g) => {
                                        // The label is the graph title and the
                                        // group's legend entry; redraw as it is
                                        // typed so the plot never shows the old name.
                                        if ui
                                            .add(
                                                egui::TextEdit::singleline(&mut g.label)
                                                    .desired_width(ui.available_width()),
                                            )
                                            .changed()
                                        {
                                            change.replot = true;
                                        }
                                    }
                                    None => {
                                        ui.weak("— no data —");
                                    }
                                }
                            });
                            ui.horizontal(|ui| {
                                form::label(ui, INFO_LABEL_W, "Data File");
                                match &data_file {
                                    Some(name) => {
                                        ui.monospace(name.as_str());
                                    }
                                    None => {
                                        ui.weak("—");
                                    }
                                }
                            });
                            ui.horizontal(|ui| {
                                form::label(ui, INFO_LABEL_W, "Output file");
                                // The field follows the current group (it was
                                // seeded only on load, so after loading Pt and
                                // switching back to a Mn scan, Autobk Start on
                                // the Mn scan wrote Pt-named χ files).
                                self.reduction.seed_output_file(
                                    self.session.current_group().map(|g| g.label.as_str()),
                                );
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.reduction.output_file)
                                        .hint_text("(auto)")
                                        .desired_width(ui.available_width()),
                                )
                                .on_hover_text(
                                    "AUTOBK Start writes <stem>k.chi (χ(k)) and \
                                     <stem>r.chi (χ(R)) into the Autobk folder",
                                );
                            });
                            ui.horizontal(|ui| {
                                form::label(ui, INFO_LABEL_W, "Theory");
                                if ui
                                    .button("Load…")
                                    .on_hover_text(
                                        "FEFF chi.dat χ(k) standard that constrains \
                                         the AUTOBK background (k_std/chi_std)",
                                    )
                                    .clicked()
                                {
                                    theory_pick = true;
                                }
                                match &theory_name {
                                    Some(name) => {
                                        ui.monospace(name.as_str());
                                        if crate::widgets::delete_box(ui).clicked() {
                                            theory_clear = true;
                                        }
                                    }
                                    None => {
                                        ui.weak("(none)");
                                    }
                                }
                            });
                        }); // end Data box
                        if group_switched {
                            self.replot_graph();
                        }

                        // The reading-format column chooser moved out of here into
                        // the detached "Open data" window (see `open_data_window`),
                        // so "Open New file" opens an in-app browser + chooser
                        // instead of an OS dialog + inline panel.

                        // --- Autobk parameters (already wrapped in its own group) -
                        let detected_e0 = self.session.current_group().and_then(|g| g.e0);
                        change = self.reduction.controls(ui, detected_e0);

                        // --- Graph: graph type + comparison-overlay display options
                        // (Feffit's panel gathers its graph controls the same way).
                        rsplot::widget::form::group(ui, |ui| {
                            ui.strong("Graph");
                            let n = self.session.groups.len();
                            ui.horizontal(|ui| {
                                ui.label("Graph type");
                                egui::ComboBox::from_id_salt("autobk_graph")
                                    .selected_text(self.reduction.graph.label())
                                    .show_ui(ui, |ui| {
                                        for g in GraphType::ALL {
                                            if ui
                                                .selectable_value(
                                                    &mut self.reduction.graph,
                                                    g,
                                                    g.label(),
                                                )
                                                .clicked()
                                            {
                                                change.replot = true;
                                            }
                                        }
                                    });
                            });
                            // Overlay every loaded group's curve for this graph type.
                            if n > 1
                                && ui
                                    .checkbox(&mut self.show_all_groups, "Show all groups")
                                    .on_hover_text(
                                        "Overlay every loaded group's curve for this \
                                         graph type, not just the selected one",
                                    )
                                    .changed()
                            {
                                change.replot = true;
                            }
                            // Waterfall the overlays (Plot Data's "Stacking"); only
                            // shown once the overlays are on screen, since it offsets
                            // those curves.
                            if n > 1
                                && self.show_all_groups
                                && ui
                                    .add(
                                        egui::Slider::new(&mut self.stack, 0.0..=5.0)
                                            .text("Stacking"),
                                    )
                                    .on_hover_text(
                                        "Vertically offset each comparison overlay so \
                                         stacked curves don't overlap (the focused group \
                                         stays at the baseline)",
                                    )
                                    .changed()
                            {
                                change.replot = true;
                            }
                            // The Autobk-computed background μ₀(E) is only drawn in
                            // the μ(E)+background view, so its toggle is offered there.
                            if self.reduction.graph == GraphType::MuBkg
                                && ui
                                    .checkbox(&mut self.show_bkg, "Show background")
                                    .on_hover_text(
                                        "Hide the Autobk-computed background μ₀(E) to \
                                         see only the raw μ(E) data files",
                                    )
                                    .changed()
                            {
                                change.replot = true;
                            }
                            // Reveal the draggable FT-window band on the kʷ·χ(k)
                            // (kmin/kmax) and |χ(R)| (rmin/rmax) graphs — the only
                            // two that draw it (see `active_window` below), so the
                            // toggle is disabled elsewhere rather than silently
                            // doing nothing.
                            let band_graph =
                                matches!(self.reduction.graph, GraphType::KChi | GraphType::ChiR);
                            ui.add_enabled(
                                band_graph,
                                egui::Checkbox::new(
                                    &mut self.reduction.show_range,
                                    "Show k/R range",
                                ),
                            )
                            .on_hover_text(
                                "Drag the kmin/kmax (kʷ·χ(k) graph) or rmin/rmax \
                                 (|χ(R)| graph) FT-window band to set the range",
                            )
                            .on_disabled_hover_text(
                                "The FT-window band is drawn on the *K.CHI and *R.CHI \
                                 graphs only",
                            );
                        });

                        // --- Analysis: openers for the windows that consume this
                        // tab's products (χ(k) → wavelet; the reduced scan series →
                        // in-situ map; the raw-file series → measured time), so the
                        // next pipeline stage is reachable where its input appears.
                        // The same entries stay in the Tools menu.
                        rsplot::widget::form::group(ui, |ui| {
                            ui.strong("Analysis");
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button("Wavelet |W(k,R)|…")
                                    .on_hover_text(
                                        "Morlet wavelet transform of the current \
                                         group's χ(k)",
                                    )
                                    .clicked()
                                {
                                    self.wavelet.open = true;
                                }
                                if ui
                                    .button("In-situ map…")
                                    .on_hover_text(
                                        "Scan-series heatmap (XANES μ(E) or RDF \
                                         |χ(R)|) over the loaded groups",
                                    )
                                    .clicked()
                                {
                                    self.insitu.open = true;
                                }
                                if ui
                                    .button("Measured time…")
                                    .on_hover_text(
                                        "Extract each scan's measured time from a \
                                         raw-file series (time-resolved XAS)",
                                    )
                                    .clicked()
                                {
                                    self.time_resolved.open = true;
                                }
                            });
                        });
                    });
                });
            });
        // Draggable FT-window band over the curve, shown only while "Show k/R
        // range" is on (off by default): the forward k-window (kmin/kmax) on the
        // kʷ·χ(k) graph, the reverse R-window (rmin/rmax) on the |χ(R)| graph.
        // With the toggle off, or on any other graph, no band — `set_window` is
        // simply not called and `show` removes any stale band. The band tracks the cursor
        // while an edge is dragged, but `take_window_drag` reports the new bounds
        // only when the drag ends (button released), so the recompute runs once
        // per drag — the same `change.refit` path a finished DragValue edit takes
        // (kmin/kmax feed both the AUTOBK background window and the FT, so a full
        // recompute is the correct cost either way).
        let active_window = if self.reduction.show_range {
            match self.reduction.graph {
                GraphType::KChi => Some((self.reduction.kmin, self.reduction.kmax)),
                GraphType::ChiR => Some((self.reduction.rmin, self.reduction.rmax)),
                _ => None,
            }
        } else {
            None
        };
        if let Some((min, max)) = active_window {
            crate::plot::set_window(&mut self.plot, crate::plot::AxisWindow { min, max });
        }
        self.plot
            .set_empty_hint("No data yet — load scans with Open New file, then Autobk Start.");
        egui::CentralPanel::default().show(ui, |ui| {
            crate::plot::show(&mut self.plot, ui);
        });
        if let Some(w) = crate::plot::take_window_drag(&mut self.plot) {
            // A left-edge drag past the right (or vice versa) inverts the pair;
            // re-order so min < max, and keep the lower bound off negative k/R.
            let (lo, hi) = if w.min <= w.max {
                (w.min, w.max)
            } else {
                (w.max, w.min)
            };
            let lo = lo.max(0.0);
            match self.reduction.graph {
                GraphType::KChi => {
                    self.reduction.kmin = lo;
                    self.reduction.kmax = hi;
                }
                GraphType::ChiR => {
                    self.reduction.rmin = lo;
                    self.reduction.rmax = hi;
                }
                _ => {}
            }
            change.refit = true;
        }

        if open_clicked {
            self.open_new_file();
        }
        // "Autobk Start" runs the full reduction *and* writes the χ files. A
        // finished Autobk parameter edit (change.refit) instead auto-runs a live
        // preview — recompute + replot only, no file write — so the graph tracks
        // the new value the moment the drag/edit ends. A bare graph-type switch is
        // the cheap replot. Start wins when it and an edit land on the same frame.
        let refit_key = (self.session.current, self.reduction.refit_key());
        if start_clicked {
            // No μ(E) for the current group but a file is open → build it first
            // (the original requires a separate Calc XMU click), then run the
            // full reduction + χ-file write on the freshly-built spectrum.
            if start_autocalc {
                self.calc_xmu();
            }
            self.run_reduction();
            self.last_refit_key = Some(refit_key);
        } else if has_mu && change.refit {
            // Skip the recompute when the committed edit changed nothing —
            // entering a field and leaving fires the same commit events as a
            // real edit. The explicit Start above always runs.
            if self.last_refit_key.as_ref() != Some(&refit_key) {
                self.recompute_reduction();
                self.last_refit_key = Some(refit_key);
            }
        } else if change.replot {
            self.replot_graph();
        }
        if edit_clicked {
            self.open_edit_xmu();
        }
        if multi_autobk_clicked {
            self.run_multi_autobk();
        }
        if theory_pick {
            self.open_theory_file();
        }
        if theory_clear {
            self.reduction.theory = None;
        }
    }

    /// The top menu bar (XAFSView's menus; most entries are stubs until their
    /// phase lands).
    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            let mut edit_clicked = false;
            ui.menu_button("Periodic table", |ui| {
                if ui.button("Periodic table + atom data…").clicked() {
                    self.periodic.open = true;
                    ui.close();
                }
            });
            // Grouped by analysis stage, following the data flow: edit μ(E) /
            // experiment / sample prep → XANES (energy) → EXAFS (k/R) →
            // multi-spectrum → structure.
            ui.menu_button("Tools", |ui| {
                if ui.button("Edit μ(E) (deglitch / trim / smooth)…").clicked() {
                    edit_clicked = true;
                    ui.close();
                }
                ui.separator();
                // Experiment & sample preparation.
                if ui.button("Ion chamber / gas absorption…").clicked() {
                    self.ion_chamber.open = true;
                    ui.close();
                }
                if ui.button("Powder weight…").clicked() {
                    self.powder.open = true;
                    ui.close();
                }
                ui.separator();
                // XANES (energy-domain) analysis.
                if ui
                    .button("XANES tools (peak / cursors / arctan)…")
                    .clicked()
                {
                    self.xanes.open = true;
                    ui.close();
                }
                if ui.button("MBACK / NEXAFS normalization…").clicked() {
                    self.mback.open = true;
                    ui.close();
                }
                ui.separator();
                // EXAFS (k / R-domain) analysis.
                if ui.button("Wavelet transform |W(k,R)|…").clicked() {
                    self.wavelet.open = true;
                    ui.close();
                }
                if ui.button("k ↔ E conversion…").clicked() {
                    self.ke_convert.open = true;
                    ui.close();
                }
                ui.separator();
                // Multi-spectrum / time-series analysis.
                if ui.button("LCF (linear combination)…").clicked() {
                    self.lcf.open = true;
                    ui.close();
                }
                if ui.button("PCA (principal components)…").clicked() {
                    self.pca.open = true;
                    ui.close();
                }
                if ui
                    .button("Extract XAS measured time (time-resolved)…")
                    .clicked()
                {
                    self.time_resolved.open = true;
                    ui.close();
                }
                if ui.button("In-situ map (scan series)…").clicked() {
                    self.insitu.open = true;
                    ui.close();
                }
                ui.separator();
                // Structure visualization.
                if ui.button("Plot Sites (3D cluster)…").clicked() {
                    self.plot_sites.open = true;
                    ui.close();
                }
            });
            ui.menu_button("Settings", |ui| {
                ui.set_min_width(200.0);
                // Overall UI scale (font size): scales text and spacing together
                // via egui's zoom factor, which the Context keeps across frames.
                ui.label("Font size");
                let resp = ui.add(egui::Slider::new(&mut self.ui_scale, 0.8..=2.0).step_by(0.05));
                // Apply only when the adjustment finishes — the drag is released
                // or a typed value is committed — so the whole UI doesn't rescale
                // (and the slider slip out from under the cursor) on every tick.
                if resp.drag_stopped() || resp.lost_focus() {
                    ui.ctx().set_zoom_factor(self.ui_scale);
                }
                if ui.button("Reset to default").clicked() {
                    self.ui_scale = DEFAULT_UI_SCALE;
                    ui.ctx().set_zoom_factor(DEFAULT_UI_SCALE);
                }
                ui.separator();
                // Background colour: toggle the whole egui UI between the dark and
                // light visuals (moved here from the former "Change BG" menu).
                if ui.button("Toggle dark / light").clicked() {
                    let dark = ui.ctx().global_style().visuals.dark_mode;
                    ui.ctx().set_visuals(if dark {
                        egui::Visuals::light()
                    } else {
                        egui::Visuals::dark()
                    });
                    self.dark_mode = !dark;
                }
                ui.separator();
                // The embedded MCP server (also startable with `--mcp [port]`),
                // for driving the app from an MCP client (Claude Code, Codex…).
                let mut mcp_on = self.mcp.is_some();
                if ui
                    .checkbox(
                        &mut mcp_on,
                        format!("MCP server (localhost:{})", self.mcp_port),
                    )
                    .on_hover_text(
                        "Expose load/params/autobk/fit/results/screenshot as MCP \
                         tools over streamable HTTP, for AI-agent control",
                    )
                    .changed()
                {
                    if mcp_on {
                        let ctx = ui.ctx().clone();
                        self.start_mcp(&ctx);
                    } else {
                        // Dropping the handle cancels the sessions and joins
                        // the listener thread.
                        self.mcp = None;
                        self.status.info("MCP server stopped.");
                    }
                }
            });
            ui.menu_button("Help", |ui| {
                if ui.button("About XAFSView…").clicked() {
                    self.about_open = true;
                    ui.close();
                }
            });
            if edit_clicked {
                self.tab = Tab::Autobk;
                self.open_edit_xmu();
            }
        });
    }

    /// Point the working folders at a project root and report the outcome to the
    /// status line. The folder creation + field mapping lives in the headless
    /// [`set_project_folders`] so it can be unit-tested.
    fn apply_sub_base(&mut self, root: std::path::PathBuf) {
        let errors = set_project_folders(&mut self.session.folders, &root);
        let made = 5 - errors.len();
        if errors.is_empty() {
            self.status.info(format!(
                "Sub base set to {} — {made} working folders ready.",
                root.display()
            ));
        } else {
            self.status.error(format!(
                "Sub base {} — {made} folder(s) ready, errors: {}",
                root.display(),
                errors.join("; ")
            ));
        }
    }

    /// The Folders tab: pick a project Sub base folder to create and fill the five
    /// working folders (Data / Autobk / Feffit / Atoms / Results); each is also
    /// editable individually.
    fn folders_panel(&mut self, ui: &mut egui::Ui) {
        let mut sub_base_pick: Option<std::path::PathBuf> = None;
        egui::CentralPanel::default().show(ui, |ui| {
            // `auto_shrink` off on the horizontal axis: otherwise the scroll area
            // collapses its width to the content, which clamps the path fields'
            // `desired_width` down to a sliver (the "folder name too small" bug).
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.heading("Folders");
                    ui.label(
                        "Pick a Sub base (project) folder: it creates and points the five \
                         working folders at <sub base>/{Data, Autobk, Feffit, Atoms, Results}. \
                         Each can still be changed individually below.",
                    );
                    ui.add_space(8.0);

                    // Form rows: fixed label column, path field filling up to the
                    // trailing Browse. Cap the row width so a maximized window
                    // doesn't sprawl the fields.
                    ui.scope(|ui| {
                        ui.set_max_width(960.0);

                        // Sub base row: a Browse pick — or a committed typed path that
                        // already exists and differs from the current one — creates the
                        // five subfolders and fills the rows below. A half-typed path is
                        // not turned into a tree: only an existing directory is accepted.
                        ui.horizontal(|ui| {
                            form::label(ui, FOLDER_LABEL_W, "Sub base folder");
                            let mut text = self
                                .session
                                .folders
                                .sub_base
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_default();
                            let resp = form::fill(
                                ui,
                                |ui| {
                                    if crate::widgets::action(
                                        ui,
                                        "Browse…",
                                        crate::widgets::BROWSE_BTN,
                                    )
                                    .clicked()
                                        && let Some(picked) = rfd::FileDialog::new().pick_folder()
                                    {
                                        sub_base_pick = Some(picked);
                                    }
                                },
                                |ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut text)
                                            .hint_text("(pick a project folder)")
                                            .desired_width(ui.available_width()),
                                    )
                                },
                            );
                            if resp.lost_focus() && !text.trim().is_empty() {
                                let p = std::path::PathBuf::from(text.trim());
                                let unchanged =
                                    self.session.folders.sub_base.as_deref() == Some(p.as_path());
                                if !unchanged && p.is_dir() {
                                    sub_base_pick = Some(p);
                                }
                            }
                        });

                        ui.add_space(6.0);
                        ui.weak("Working folders (auto-filled from the sub base; each editable):");
                        ui.add_space(4.0);

                        folder_row(ui, "Data folder", &mut self.session.folders.data_dir);
                        folder_row(ui, "Autobk folder", &mut self.session.folders.autobk_dir);
                        folder_row(ui, "Feffit folder", &mut self.session.folders.feffit_dir);
                        folder_row(ui, "Atoms folder", &mut self.session.folders.atoms_dir);
                        folder_row(ui, "Results folder", &mut self.session.folders.results_dir);
                    });
                });
        });
        if let Some(root) = sub_base_pick {
            self.apply_sub_base(root);
        }
    }
}

impl eframe::App for XafsViewApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PREFS_KEY, &Prefs::capture(self));
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Apply a finished background job's results first, so this frame draws
        // the post-job state (the Apply closure runs on this thread — the only
        // place shared state is mutated).
        match self.jobs.poll() {
            Some(crate::jobs::Polled::Apply(apply)) => apply(self),
            Some(crate::jobs::Polled::Failed(label)) => {
                let msg = format!("{label}: the background job died unexpectedly — results lost.");
                // The single job slot means a parked MCP reply belongs to this
                // job — answer it so the client is not left hanging. Same for
                // the In-situ rebuild armed on a reduction job: disarm it.
                if let Some(reply) = self.mcp_job_reply.take() {
                    let _ = reply.send(Err(msg.clone()));
                }
                self.insitu_rebuild_after_reduce = false;
                self.status.error(msg);
            }
            None => {}
        }
        // While a job runs, keep repainting so its progress ticks show without
        // waiting for user input.
        if self.jobs.busy() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Handle MCP tool requests after the job poll, so a just-finished job
        // is applied before a new request checks the job slot.
        self.poll_mcp(ui.ctx());

        egui::Panel::top("menubar").show(ui, |ui| {
            self.menu_bar(ui);
        });

        // Horizontal tab strip directly under the menu bar, in analysis-flow
        // order (Folders … Feff) rather than a vertical side list.
        egui::Panel::top("tabs").show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                for tab in Tab::ALL {
                    ui.selectable_value(&mut self.tab, tab, tab.label());
                }
                // Plot Data is a detached viewer, not a tab: set it off past a
                // separator after the last tab, with the window-opening ellipsis,
                // so it reads as a window opener rather than another tab.
                ui.separator();
                if ui
                    .button("Plot Data…")
                    .on_hover_text("Open the Plot Data file-overlay window")
                    .clicked()
                {
                    self.plot_data.open();
                }
            });
            ui.add_space(2.0);
        });

        // On a tab switch, repopulate the shared plot for the newly active tab
        // so it shows that tab's curves rather than the previous tab's.
        if self.tab != self.last_tab {
            self.replot_active_tab();
            self.last_tab = self.tab;
        }

        // The common error/status strip, shown on every tab. At the bottom of
        // the window — where users scan for status by convention.
        egui::Panel::bottom("status").show(ui, |ui| {
            // Right-to-left so the Log button and group count claim their width
            // first; the status line then truncates into whatever remains and
            // cannot run under them on a narrow window.
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // The strip shows only the latest line; the Log window keeps
                    // the whole session's messages (its error count stays on the
                    // button so overwritten failures remain discoverable).
                    let errors = self.status.error_count();
                    let label = if errors > 0 {
                        format!("Log ({errors} ⚠)")
                    } else {
                        "Log".to_owned()
                    };
                    if ui
                        .button(label)
                        .on_hover_text("open the session message log")
                        .clicked()
                    {
                        self.status.open = true;
                    }
                    ui.weak(format!("{} group(s) loaded", self.session.groups.len()));
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        // A running background job shows its spinner + progress
                        // ahead of the status line.
                        if let Some((label, done, total)) = self.jobs.status() {
                            ui.spinner();
                            ui.label(format!("{label}: {done}/{total}"));
                            ui.separator();
                        }
                        ui.add(egui::Label::new(self.status.line()).truncate());
                    });
                });
            });
        });

        // Keep the Plot Data window's folder defaults (file-picker start dir,
        // save-dialog default) fresh before it renders below.
        self.plot_data.set_dirs(
            self.session.folders.results_dir.as_deref(),
            self.session.folders.data_dir.as_deref(),
        );

        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Folders => self.folders_panel(ui),
            Tab::Autobk => self.autobk_tab(ui),
            Tab::Feffit => self.feffit_tab(ui),
            Tab::FeffitTxt => self.feffit_txt_tab(ui),
            Tab::Atoms => match self.atoms_tab.ui(ui, &mut self.feff_inp) {
                // Hand off to the Feff tab so the user can run the new input.
                Some(AtomsAction::BuiltFeffInp) => self.tab = Tab::Feff,
                None => {}
            },
            Tab::Feff => {
                match self.feff_tab.ui(
                    ui,
                    &mut self.feff_inp,
                    self.session.folders.atoms_dir.as_deref(),
                ) {
                    Some(FeffAction::ViewStructure) => self.plot_sites.open = true,
                    None => {}
                }
            }
        });

        // The About window (Help menu).
        self.about_window(ui.ctx());

        // The Edit-μ(E) window floats above the panels and is driven from the
        // Autobk tab button / Tools menu. Seed its energy widgets for the
        // group they will edit — a group switch while the window is open
        // re-seeds, so another scan's stale energies never target this one.
        if self.edit_xmu.open
            && let Some(g) = self.session.current_group()
            && !g.energy.is_empty()
        {
            self.edit_xmu
                .seed_span(&g.label, g.energy[0], *g.energy.last().unwrap());
        }
        let has_group = self
            .session
            .current_group()
            .is_some_and(|g| !g.mu.is_empty());
        let npts = self.session.current_group().map_or(0, |g| g.len());
        let can_undo = !self.clean_undo.is_empty();
        let n_checked = self.session.selected.len();
        // "use clicked" takes the last clicked data point only when it is an
        // energy of the group being edited: the shared plot must be showing
        // the Autobk tab's graph (the Feffit tab draws k/R/q on the same
        // plot) in an energy space, and the clicked curve must be the current
        // group's, not another group's Show-all overlay.
        let picked = (self.tab == Tab::Autobk && self.reduction.graph.x_is_energy())
            .then(|| self.plot.last_pick())
            .flatten()
            .filter(|(label, _)| {
                !self.show_all_groups
                    || Self::pick_is_current_group(
                        &self.session.groups,
                        self.session.current,
                        label,
                    )
            })
            .map(|(_, x)| x);
        if let Some(action) =
            self.edit_xmu
                .show(ui.ctx(), has_group, npts, can_undo, n_checked, picked)
        {
            match action {
                // The auto actions that reach beyond the current group (preview
                // markers, batch over checked groups) are handled here; the
                // single-group edits go through the undo-snapshot path.
                CleanAction::AutoFind(nsig) => self.auto_deglitch_find(nsig),
                CleanAction::AutoDeglitchChecked(nsig) => self.auto_deglitch_checked(nsig),
                other => self.apply_clean_action(other),
            }
        }

        // The wavelet-transform window floats above the panels; Compute reads the
        // current group's χ(k). Enabled only when the active group has χ(k).
        let has_chi = self
            .session
            .current_group()
            .is_some_and(|g| g.k.is_some() && g.chi.is_some());
        match self.wavelet.show(ui.ctx(), has_chi) {
            Some(WaveletAction::Compute) => self.run_wavelet(),
            Some(WaveletAction::ExportCsv) => self.export_wavelet_csv(),
            None => {}
        }

        // The detached Plot Data window (its own graph + control column, like the
        // original XAFSView) and its file-selection sub-window — both render here,
        // independent of the active tab, and only when opened.
        self.plot_data.show_window(ui.ctx(), &self.session.groups);
        self.plot_data.file_picker(ui.ctx(), &mut self.status);

        // The detached Log window: the session's cumulative status/error
        // messages behind the strip's "Log" button.
        self.status.show_window(ui.ctx());

        // The detached "Data" window: the loaded-group manager (list + Add +
        // Clear all) moved out of the Autobk sidebar. A select/remove/clear
        // here repopulates the plot.
        let ctx = ui.ctx().clone();
        if self.data_manager_window(&ctx) {
            self.replot_active_tab();
        }

        // The detached "Open data" loader: the in-app raw-scan browser + two-pane
        // staging + reading-format chooser that "Open New file" / "Add data files…"
        // open; its Autobk Start builds μ(E) for every staged file and reduces.
        self.open_data_window(&ctx);

        // The detached "Feffit — Batch" window: the batch panel moved out of the
        // Feffit tab's right side. Its disk-write actions are app-owned.
        match self.feffit_batch_window(&ctx) {
            Some(BatchAction::RunAll) => self.start_feffit_batch(),
            Some(BatchAction::SaveItems(files)) => self.write_saved_items(files),
            Some(BatchAction::SaveFeffitOutputs(files)) => self.write_feffit_output_files(files),
            None => {}
        }
        // A batch run / clear / overlay-toggle rebuilds the Feffit graph overlay,
        // but only while the Feffit tab is the one showing the shared plot (the
        // batch window renders regardless of the active tab).
        if self.tab == Tab::Feffit && self.feffit_batch.take_graph_dirty() {
            self.replot_feffit();
        }

        // The LCF and PCA windows each overlay several groups onto one common
        // energy grid and call the headless engines; both own their own plot.
        self.lcf.show(ui.ctx(), &self.session.groups);
        self.pca.show(ui.ctx(), &self.session.groups);

        // The XANES-tools window reads one chosen spectrum and owns its own plot.
        self.xanes.show(ui.ctx(), &self.session.groups);

        // MBACK normalization (xraydb f2 + the headless mback_norm engine) reads
        // one spectrum and owns its own plot.
        self.mback.show(ui.ctx(), &self.session.groups);

        // The Phase-8 atomic-data calculators are self-contained (no session,
        // no plot) — each is backed by the bundled xraydb database.
        self.periodic.show(ui.ctx());
        self.ion_chamber.show(ui.ctx());
        self.powder.show(ui.ctx());
        // k ↔ E conversion: seeded from the active group's edge energy E₀.
        let group_e0 = self.session.current_group().and_then(|g| g.e0);
        self.ke_convert.show(ui.ctx(), group_e0);
        // Extract XAS measured time: self-contained (picks its own file series).
        self.time_resolved.show(ui.ctx());
        // In-situ map: the scan-series image over the loaded groups (or a
        // legacy BL10C series file it opens itself). Re-reduce & build starts
        // the Autobk-tab reduction job here (the window has no parameters of
        // its own); the armed flag makes finish_reduction rebuild the map. A
        // synchronous refusal keeps the window's map and status untouched.
        if let Some(InsituAction::ReduceAndBuild) = self.insitu.show(ui.ctx(), &self.session.groups)
            && self.run_reduction()
        {
            self.insitu_rebuild_after_reduce = true;
        }

        // The Plot Sites 3D viewer parses the shared feff.inp into a point cloud.
        // It needs the wgpu render state each frame (unlike the 2D Plot1D, which
        // caches it), so it is fed from the eframe Frame here.
        if let Some(rs) = frame.wgpu_render_state() {
            let path_files = self.feff_tab.last_path_files();
            self.plot_sites
                .show(ui.ctx(), rs, &self.feff_inp, &path_files);
        }
    }
}

/// Create `<root>/{Data,Autobk,Feffit,Atoms,Results}` (idempotent — existing
/// folders are reused) and point the matching [`Folders`] fields at them, in the
/// fixed order Data→`data_dir`, Autobk→`autobk_dir`, Feffit→`feffit_dir`,
/// Atoms→`atoms_dir`, Results→`results_dir`. A subfolder whose creation fails
/// keeps the field's previous value; `sub_base` is set to `root` regardless.
/// Returns one `"<Name>: <error>"` string per failed creation. Mirrors the
/// original XAFSView "Sub base folder" auto-fill.
fn set_project_folders(folders: &mut Folders, root: &std::path::Path) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();
    let mut make = |name: &str, slot: &mut Option<std::path::PathBuf>| {
        let dir = root.join(name);
        match std::fs::create_dir_all(&dir) {
            Ok(()) => *slot = Some(dir),
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    };
    make("Data", &mut folders.data_dir);
    make("Autobk", &mut folders.autobk_dir);
    make("Feffit", &mut folders.feffit_dir);
    make("Atoms", &mut folders.atoms_dir);
    make("Results", &mut folders.results_dir);
    folders.sub_base = Some(root.to_path_buf());
    errors
}

/// Label-column width of the Folders form rows ("Sub base folder" is the
/// widest label).
const FOLDER_LABEL_W: f32 = 110.0;

/// Label-column width of the Autobk Data box rows (Group / Title / Data File /
/// Output file / Theory share one aligned column).
const INFO_LABEL_W: f32 = 80.0;

/// One labelled folder row — fixed label column, a path field filling up to
/// the trailing Browse button that opens a native folder picker.
fn folder_row(ui: &mut egui::Ui, label: &str, dir: &mut Option<std::path::PathBuf>) {
    ui.horizontal(|ui| {
        form::label(ui, FOLDER_LABEL_W, label);
        // Editable path text (mirrors the original's typeable folder field), kept in
        // sync with the PathBuf; an empty string clears it.
        let mut text = dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let resp = form::fill(
            ui,
            |ui| {
                if crate::widgets::action(ui, "Browse…", crate::widgets::BROWSE_BTN).clicked()
                    && let Some(picked) = rfd::FileDialog::new().pick_folder()
                {
                    *dir = Some(picked);
                }
            },
            |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut text)
                        .hint_text("(not set)")
                        .desired_width(ui.available_width()),
                )
            },
        );
        if resp.changed() {
            *dir = if text.trim().is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(text.clone()))
            };
        }
        // A deep path can still outrun the field; show the whole path on hover
        // so it stays readable.
        if !text.is_empty() {
            resp.on_hover_text(text);
        }
    });
    // A typed path is kept as typed (the writers create a missing output folder
    // on first save), but a path that does not exist yet is said so here, under
    // its field, instead of a silently accepted typo.
    if dir.as_ref().is_some_and(|d| !d.is_dir()) {
        ui.horizontal(|ui| {
            ui.add_space(FOLDER_LABEL_W + ui.spacing().item_spacing.x);
            ui.colored_label(ui.visuals().warn_fg_color, "⚠ folder does not exist yet");
        });
    }
}

/// Whether the x ranges of two curves overlap (touching counts). An empty or
/// all-NaN curve overlaps nothing.
fn spans_overlap(a: &[f64], b: &[f64]) -> bool {
    fn span(x: &[f64]) -> Option<(f64, f64)> {
        let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo <= hi).then_some((lo, hi))
    }
    match (span(a), span(b)) {
        (Some((alo, ahi)), Some((blo, bhi))) => alo <= bhi && blo <= ahi,
        _ => false,
    }
}

/// Whether two paths point to the same file. Canonicalizes both (so case-folding
/// and `..` are resolved on macOS); falls back to a path compare when `b` does
/// not exist yet (in which case it is a different, new file).
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}

/// Write a μ(E) spectrum as a two-column `.xmu` text file (energy, μ) with a
/// short header — the format the column reader reads back.
fn write_xmu(
    path: &std::path::Path,
    label: &str,
    energy: &[f64],
    mu: &[f64],
) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(energy.len() * 32 + 64);
    let _ = writeln!(s, "# {label}");
    let _ = writeln!(s, "#  energy            xmu");
    for (&e, &m) in energy.iter().zip(mu) {
        let _ = writeln!(s, "{e:14.6}  {m:18.10}");
    }
    write_file(path, s)
}

/// Serialize χ(k) as a two-column UWXAFS-style `.chi` file: `k` (Å⁻¹) and the
/// unweighted `χ(k)` (k-weighting is applied at plot/FT time, not stored).
fn write_chik(path: &std::path::Path, header: &str, k: &[f64], chi: &[f64]) -> std::io::Result<()> {
    write_file(path, crate::chi_io::chik_string(header, k, chi))
}

/// Serialize a complex transform as a five-column UWXAFS file: the axis grid,
/// `real`, `imag`, `ampl`, `phase`, all on the same grid. `sym` names the space
/// (`"r"` for R-space, `"k"` for the q transform's k-grid) so r- and q-space
/// files are labelled correctly.
fn write_complex5(
    path: &std::path::Path,
    header: &str,
    sym: &str,
    x: &[f64],
    mag: &[f64],
    re: &[f64],
    im: &[f64],
) -> std::io::Result<()> {
    write_file(
        path,
        crate::chi_io::complex5_string(header, sym, x, mag, re, im),
    )
}

/// Serialize χ(R) as a five-column `.chi` file: `r` (Å), `real`, `imag`, `ampl`,
/// `phase`, all on the same `R` grid (`phase = atan2(imag, real)`).
fn write_chir(
    path: &std::path::Path,
    header: &str,
    r: &[f64],
    mag: &[f64],
    re: &[f64],
    im: &[f64],
) -> std::io::Result<()> {
    write_complex5(path, header, "r", r, mag, re, im)
}

/// Write `content` to `path`, creating the parent directory tree first. The
/// single write primitive for our output files (`.xmu` / `.chi` / `.bkg` /
/// `.dat` / `.fit` / Save-Items), so "the destination folder exists" holds by
/// construction at every writer — a configured-but-missing Feffit / Autobk /
/// Results folder is created on demand instead of the write failing. The rfd
/// save-dialog writers stay on `std::fs::write`: the user picked an existing
/// directory there.
fn write_file(path: &std::path::Path, content: impl AsRef<[u8]>) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)
}

/// The file-name portion of a path as an owned string (for status messages).
fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_principal_curve_matches_the_selected_graph_type() {
        // μ(E) view → the raw (energy, μ) of the group.
        let gm = XasGroup::from_mu("m", vec![7000.0, 7001.0, 7002.0], vec![0.1, 0.2, 0.3]);
        let (x, y) =
            XafsViewApp::group_principal_curve(&gm, GraphType::MuBkg, 2).expect("μ(E) principal");
        assert_eq!(x, vec![7000.0, 7001.0, 7002.0]);
        assert_eq!(y, vec![0.1, 0.2, 0.3]);

        // kʷ·χ(k) view → χ weighted by kʷ (here w = 2), so y = χ·k².
        let gk = XasGroup::from_chi("k", vec![1.0, 2.0, 3.0], vec![0.5, 0.5, 0.5]);
        let (xk, yk) =
            XafsViewApp::group_principal_curve(&gk, GraphType::KChi, 2).expect("kʷ·χ principal");
        assert_eq!(xk, vec![1.0, 2.0, 3.0]);
        assert_eq!(yk, vec![0.5, 2.0, 4.5]);

        // A μ(E)-only group has no χ(R) stage → nothing to overlay in the R view.
        assert!(XafsViewApp::group_principal_curve(&gm, GraphType::ChiR, 2).is_none());
    }

    #[test]
    fn spans_overlap_requires_a_shared_x_range() {
        // Mn (6.3–7.1 keV) and Pt (11.4–12.5 keV) never overlap; a shared edge
        // does; touching ranges count; an empty curve overlaps nothing.
        assert!(!spans_overlap(&[6339.0, 7100.0], &[11464.0, 12563.0]));
        assert!(spans_overlap(&[6339.0, 7100.0], &[6400.0, 6900.0]));
        assert!(spans_overlap(&[7100.0, 6339.0], &[7100.0, 7500.0]));
        assert!(!spans_overlap(&[], &[1.0, 2.0]));
    }

    #[test]
    fn pick_is_current_group_rejects_other_groups_overlay_labels() {
        let groups = vec![
            XasGroup::from_mu("a", vec![1.0], vec![0.1]),
            XasGroup::from_mu("b", vec![1.0], vec![0.1]),
        ];
        // The current group's own curve and its auxiliary curves pass …
        assert!(XafsViewApp::pick_is_current_group(&groups, Some(1), "b"));
        assert!(XafsViewApp::pick_is_current_group(
            &groups,
            Some(1),
            "background"
        ));
        // … another group's overlay (legended by that group's label) does not.
        assert!(!XafsViewApp::pick_is_current_group(&groups, Some(1), "a"));
        // A label the current group shares with another group is the current
        // group's own curve, not an overlay.
        let dup = vec![
            XasGroup::from_mu("a", vec![1.0], vec![0.1]),
            XasGroup::from_mu("a", vec![1.0], vec![0.1]),
        ];
        assert!(XafsViewApp::pick_is_current_group(&dup, Some(0), "a"));
    }

    #[test]
    fn write_xmu_roundtrips_through_the_column_reader() {
        // A written .xmu must read back as a 2-column energy/μ file (the format
        // the import path itself consumes).
        let path =
            std::env::temp_dir().join(format!("xafsview_write_xmu_{}.xmu", std::process::id()));
        let energy = [7000.0, 7001.5, 7003.0];
        let mu = [0.10, 0.25, 0.42];
        write_xmu(&path, "roundtrip", &energy, &mu).expect("write .xmu");

        let cf = ColumnFile::from_path(&path).expect("read .xmu back");
        assert_eq!(cf.nrows(), 3);
        assert_eq!(cf.ncols(), 2);
        let e = cf.column(0).expect("energy column");
        let m = cf.column(1).expect("μ column");
        assert!((e[0] - 7000.0).abs() < 1e-3, "energy {e:?}");
        assert!((m[2] - 0.42).abs() < 1e-6, "mu {m:?}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_file_creates_missing_parent_dirs() {
        // The output writers route through `write_file`; a configured-but-missing
        // destination folder (e.g. a Feffit folder set in the Folders tab that
        // doesn't exist yet) must be created on demand rather than failing the
        // write — the bug behind "Only FT produced no .dat files".
        let root = std::env::temp_dir().join(format!("xafsview_write_file_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("Feffit").join("missing").join("out.dat");
        assert!(
            !nested.parent().unwrap().exists(),
            "parent absent before write"
        );

        write_file(&nested, "k r q\n").expect("create parents and write");
        assert_eq!(std::fs::read_to_string(&nested).unwrap(), "k r q\n");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_chik_is_two_columns_and_write_chir_is_five() {
        let pid = std::process::id();
        let kp = std::env::temp_dir().join(format!("xafsview_chik_{pid}k.chi"));
        let rp = std::env::temp_dir().join(format!("xafsview_chir_{pid}r.chi"));

        let k = [2.0, 2.05, 2.10];
        let chi = [0.012, -0.008, 0.003];
        write_chik(&kp, "# demo\r\n", &k, &chi).expect("write k.chi");

        let r = [0.0, 0.05, 0.10, 0.15];
        let mag = [1.0e-2, 2.0e-2, 1.5e-2, 5.0e-3];
        let re = [1.0e-2, 0.0, -1.5e-2, 5.0e-3];
        let im = [0.0, 2.0e-2, 0.0, 0.0];
        write_chir(&rp, "# demo\r\n", &r, &mag, &re, &im).expect("write r.chi");

        let ck = ColumnFile::from_path(&kp).expect("read k.chi");
        assert_eq!(ck.ncols(), 2, "χ(k) file is k + chi");
        assert_eq!(ck.nrows(), 3);
        assert!((ck.column(0).unwrap()[0] - 2.0).abs() < 1e-3, "k grid");
        assert!((ck.column(1).unwrap()[1] + 0.008).abs() < 1e-6, "χ value");

        let cr = ColumnFile::from_path(&rp).expect("read r.chi");
        assert_eq!(cr.ncols(), 5, "χ(R) file is r + real + imag + ampl + phase");
        assert_eq!(cr.nrows(), 4);
        assert!((cr.column(0).unwrap()[2] - 0.10).abs() < 1e-3, "R grid");
        // The amplitude column is index 3 (real=1, imag=2, ampl=3, phase=4).
        assert!(
            (cr.column(3).unwrap()[1] - 2.0e-2).abs() < 1e-6,
            "|χ(R)| is column 3 in the 5-column layout"
        );

        let _ = std::fs::remove_file(&kp);
        let _ = std::fs::remove_file(&rp);
    }

    #[test]
    fn same_file_guards_against_overwriting_the_source() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let a = dir.join(format!("xafsview_same_a_{pid}.xmu"));
        let b = dir.join(format!("xafsview_same_b_{pid}.xmu"));
        std::fs::write(&a, "x").expect("write a");
        std::fs::write(&b, "y").expect("write b");

        assert!(same_file(&a, &a), "a path is the same file as itself");
        assert!(!same_file(&a, &b), "distinct existing files differ");
        // A not-yet-existing output beside an existing source is a new file.
        let c = dir.join(format!("xafsview_same_c_{pid}.xmu"));
        assert!(!same_file(&a, &c), "non-existent output is distinct");

        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }

    #[test]
    fn set_project_folders_creates_and_maps_the_five_subfolders() {
        // A fresh project root: set_project_folders must create all five named
        // subfolders and map each to the right field (Data→data_dir, …).
        let root = std::env::temp_dir().join(format!("xafsview_subbase_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("make project root");

        let mut folders = Folders::default();
        let errors = set_project_folders(&mut folders, &root);
        assert!(errors.is_empty(), "no creation errors: {errors:?}");

        assert_eq!(folders.sub_base.as_deref(), Some(root.as_path()));
        assert_eq!(folders.data_dir, Some(root.join("Data")));
        assert_eq!(folders.autobk_dir, Some(root.join("Autobk")));
        assert_eq!(folders.feffit_dir, Some(root.join("Feffit")));
        assert_eq!(folders.atoms_dir, Some(root.join("Atoms")));
        assert_eq!(folders.results_dir, Some(root.join("Results")));
        for name in ["Data", "Autobk", "Feffit", "Atoms", "Results"] {
            assert!(root.join(name).is_dir(), "{name} subfolder created");
        }

        // Idempotent: a second call on the same root reuses the folders (no
        // error) and leaves the mapping unchanged.
        let again = set_project_folders(&mut folders, &root);
        assert!(again.is_empty(), "second call has no errors: {again:?}");
        assert_eq!(folders.data_dir, Some(root.join("Data")));

        let _ = std::fs::remove_dir_all(&root);
    }
}
