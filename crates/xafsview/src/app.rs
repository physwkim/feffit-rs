//! The application shell: window layout, the tab bar, the menu bar, and the
//! shared `siplot` plot. Individual tabs fill in their panels in later phases;
//! Phase 0 wires the skeleton and makes the **Folders** tab functional.

use eframe::egui;
use egui::Color32;
use xasdata::{ColumnFile, Session, XasGroup};

use crate::analysis_ui::{LcfWindow, PcaWindow};
use crate::atoms_ui::{AtomsAction, AtomsTab, FeffAction, FeffTab, PlotSitesWindow};
use crate::calc_ui::{IonChamberWindow, KeConvertWindow, PeriodicTableWindow, PowderWindow};
use crate::clean_ui::{CleanAction, EditXmuState};
use crate::feffit_batch::{BatchAction, FeffitBatch};
use crate::feffit_ui::{FeffitAction, FeffitUi};
use crate::import::{ImportAction, ImportState};
use crate::mback_ui::MbackWindow;
use crate::plot_data::PlotDataWindow;
use crate::reduce_ui::{GraphType, LoadingType, ReductionUi, TheoryStd};
use crate::timeres_ui::TimeResolvedWindow;
use crate::wavelet::{WaveletAction, WaveletWindow, morlet_cwt};
use crate::xanes_ui::XanesWindow;

/// The top-level tabs, mirroring XAFSView's tab strip. Most are placeholders in
/// Phase 0 and are filled in by their respective phases.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    /// Pre-edge / normalize / AUTOBK background removal (P1–P2).
    Autobk,
    /// FEFFIT path fitting (P4).
    Feffit,
    /// FEFFIT text/parameter view (P4).
    FeffitTxt,
    /// Crystal → feff.inp (P9).
    Atoms,
    /// Edit feff.inp / run FEFF (P9).
    Feff,
    /// Working-directory configuration (this phase).
    Folders,
    /// Program/version information (mirrors XAFSView's About tab).
    About,
}

impl Tab {
    /// All tabs in strip order.
    const ALL: [Tab; 7] = [
        Tab::Autobk,
        Tab::Feffit,
        Tab::FeffitTxt,
        Tab::Atoms,
        Tab::Feff,
        Tab::Folders,
        Tab::About,
    ];

    /// The label shown on the tab button.
    fn label(self) -> &'static str {
        match self {
            Tab::Autobk => "Autobk",
            Tab::Feffit => "Feffit",
            Tab::FeffitTxt => "Feffit_txt",
            Tab::Atoms => "Atoms",
            Tab::Feff => "Feff",
            Tab::Folders => "Folders",
            Tab::About => "About",
        }
    }
}

/// The eframe application: the data [`Session`], the active tab, and the shared
/// plot (one `siplot::Plot1D` reused across the plot-bearing tabs).
pub struct XafsViewApp {
    session: Session,
    tab: Tab,
    plot: crate::plot::Plot,
    /// The in-progress data import (column → role mapping), if a raw file is open.
    import: Option<ImportState>,
    /// Reduction (normalize/autobk/FT) parameters and the active graph type.
    reduction: ReductionUi,
    /// FEFFIT tab state: paths, variables, transform, and last fit result.
    feffit: FeffitUi,
    /// Label of the group the last main-tab Feffit fit was run on, so "Send to
    /// Plot Data" names the fitted group even after the current group changes.
    feffit_fit_group: Option<String>,
    /// The multi-FEFFIT batch window (one independent fit config per group).
    feffit_batch: FeffitBatch,
    /// The Edit-μ(E) window (deglitch / trim / smooth) state.
    edit_xmu: EditXmuState,
    /// The Morlet wavelet-transform window (|W(k,R)| heatmap of χ(k)).
    wavelet: WaveletWindow,
    /// The standalone Plot Data window (multi-group overlay), with its own plot.
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
    /// The Atoms tab state (crystal cell → feff.inp).
    atoms_tab: AtomsTab,
    /// The Feff tab state (edit feff.inp / run FEFF).
    feff_tab: FeffTab,
    /// The 3D Plot Sites cluster viewer (its own `siplot` scene).
    plot_sites: PlotSitesWindow,
    /// The shared `feff.inp` text, written by Atoms and edited/run by Feff.
    feff_inp: String,
    /// Pre-edit group snapshots for undoing cleanup edits (most recent last).
    clean_undo: Vec<XasGroup>,
    /// The tab shown on the previous frame, to detect tab switches for replot.
    last_tab: Tab,
    /// Status line shown at the bottom of the window.
    status: String,
}

impl XafsViewApp {
    /// Build the app. Requires the wgpu render state (see [`main`](crate::main)).
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Install the glyph-fallback font before anything draws, so both the egui
        // UI and siplot axis labels can render the superscript/subscript math
        // glyphs the default font lacks (Å⁻¹, kʷ, χ²ᵣ).
        crate::fonts::install_fallback(&cc.egui_ctx);

        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must use the wgpu renderer (NativeOptions.renderer = Wgpu)");

        let mut plot = crate::plot::Plot::new(render_state, 0);
        plot.set_graph_title("XAFSView");
        plot.set_graph_x_label("Energy (eV)");
        plot.set_graph_y_label("μ(E)", siplot::YAxis::Left);

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
        plot.add_curve_with_legend(&x, &y, Color32::from_rgb(0x1f, 0x77, 0xb4), "demo edge");

        let plot_data = PlotDataWindow::new(render_state);
        let feffit_batch = FeffitBatch::new(render_state);
        let lcf = LcfWindow::new(render_state);
        let pca = PcaWindow::new(render_state);
        let xanes = XanesWindow::new(render_state);
        let mback = MbackWindow::new(render_state);
        let periodic = PeriodicTableWindow::default();
        let ion_chamber = IonChamberWindow::default();
        let powder = PowderWindow::default();
        let plot_sites = PlotSitesWindow::new(render_state);

        Self {
            session: Session::new(),
            tab: Tab::Autobk,
            plot,
            import: None,
            reduction: ReductionUi::default(),
            feffit: FeffitUi::default(),
            feffit_fit_group: None,
            feffit_batch,
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
            atoms_tab: AtomsTab::default(),
            feff_tab: FeffTab::default(),
            plot_sites,
            feff_inp: String::new(),
            clean_undo: Vec::new(),
            last_tab: Tab::Autobk,
            status: "Ready.".to_owned(),
        }
    }

    /// Open a beamline column file via a native dialog and start an import.
    fn open_file(&mut self) {
        let mut dlg = rfd::FileDialog::new();
        if let Some(dir) = &self.session.folders.data_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.pick_file() else {
            return;
        };
        match ColumnFile::from_path(&path) {
            Ok(cf) => {
                self.status = format!(
                    "Loaded {} — {} columns × {} rows",
                    path.display(),
                    cf.ncols(),
                    cf.nrows()
                );
                self.import = Some(ImportState::new(cf));
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    /// "Open New file": route the chosen file by the Loading-file-type ring.
    /// `chi.dat` loads χ(k) directly; the μ(E) modes go through the column
    /// chooser (which adapts to a raw multi-column file or a 2-column μ file).
    fn open_new_file(&mut self) {
        match self.reduction.loading {
            LoadingType::ChiDat => self.open_chi_dat(),
            LoadingType::CalcXmu | LoadingType::LoadXmu => self.open_file(),
        }
    }

    /// Load a FEFF `chi.dat` directly as a χ(k)-only group (no μ(E)).
    fn open_chi_dat(&mut self) {
        let mut dlg = rfd::FileDialog::new();
        if let Some(dir) = &self.session.folders.data_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.pick_file() else {
            return;
        };
        match xasdata::read_chi_dat(&path) {
            Ok((k, chi)) => {
                let label = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("chi{}", self.session.groups.len() + 1));
                let n = k.len();
                let mut g = XasGroup::from_chi(label, k, chi);
                g.filename = Some(path.clone());
                self.session.add_group(g);
                self.reduction.graph = GraphType::KChi;
                self.plot_data.mark_dirty();
                self.replot_graph();
                self.status = format!("Loaded χ(k): {n} points from {}", path.display());
            }
            Err(e) => self.status = format!("chi.dat open failed: {e}"),
        }
    }

    /// Load a FEFF `chi.dat` as the **Theory** standard — the `k_std`/`chi_std`
    /// constraint applied by AUTOBK so the background does not absorb real
    /// first-shell amplitude.
    fn open_theory_file(&mut self) {
        let mut dlg = rfd::FileDialog::new();
        if let Some(dir) = &self.session.folders.data_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.pick_file() else {
            return;
        };
        match xasdata::read_chi_dat(&path) {
            Ok((k, chi)) => {
                let n = k.len();
                self.reduction.theory = Some(TheoryStd { path, k, chi });
                self.status = format!("Loaded theory χ(k) standard: {n} points");
            }
            Err(e) => self.status = format!("Theory open failed: {e}"),
        }
    }

    /// Build `mu(E)` from the current import selection and add it as a group.
    fn calc_xmu(&mut self) {
        let Some(import) = self.import.as_ref() else {
            return;
        };
        let spec = import.to_spec();
        let input_path = import.file.path.clone();
        match xasdata::build_mu(&import.file, &spec) {
            Ok((energy, mu)) => {
                let label = input_path
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .map(|s| s.to_string_lossy().into_owned())
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
                self.session.add_group(XasGroup::from_mu(label, energy, mu));
                self.status = format!("Built μ(E): {n} points{xmu}");
                self.reduction.graph = GraphType::MuBkg;
                // New spectrum: drop stale undo history and re-seed the editor.
                self.clean_undo.clear();
                self.edit_xmu.reset_seed();
                self.plot_data.mark_dirty();
                self.replot_graph();
            }
            Err(e) => self.status = format!("Calc XMU failed: {e}"),
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
            .or_else(|| self.session.folders.work_dir.clone())
        else {
            return Err("no output folder".to_owned());
        };
        let stem = input
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
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
    fn run_reduction(&mut self) {
        let pre = self.reduction.pre_params();
        let bk = self.reduction.autobk_params();
        let ft = self.reduction.ft_params();
        let info = {
            let Some(g) = self.session.current_group_mut() else {
                return;
            };
            if g.mu.is_empty() {
                return;
            }
            xasdata::normalize(g, &pre);
            xasdata::autobk_group(g, &bk, 1.0);
            xasdata::xftf_group(g, &ft);
            (g.e0.unwrap_or(0.0), g.k.as_ref().map_or(0, |k| k.len()))
        };
        self.status = format!("Reduction done: E₀ = {:.2} eV, {} k-points", info.0, info.1);
        self.plot_data.mark_dirty();
        self.replot_graph();
    }

    /// Run normalize → AUTOBK → FT on every loaded group with one shared set of
    /// parameters (the Autobk tab's current settings), then refresh the plots.
    fn run_multi_autobk(&mut self) {
        if self.session.groups.is_empty() {
            self.status = "No groups loaded.".to_owned();
            return;
        }
        let pre = self.reduction.pre_params();
        let bk = self.reduction.autobk_params();
        let ft = self.reduction.ft_params();
        let n = xasdata::reduce_all(&mut self.session.groups, &pre, &bk, &ft, 1.0);
        self.status = format!("Multiple AUTOBK: reduced {n} group(s).");
        self.plot_data.mark_dirty();
        self.replot_graph();
    }

    /// Pick several files of the same column layout and build a μ(E) group from
    /// each using the active import's column mapping (XAFSView batch make-xmu).
    fn make_xmu_from_files(&mut self) {
        let Some(spec) = self.import.as_ref().map(|i| i.to_spec()) else {
            self.status =
                "Open one file and choose its columns first, then batch the rest.".to_owned();
            return;
        };
        let mut dlg = rfd::FileDialog::new();
        if let Some(dir) = &self.session.folders.data_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(paths) = dlg.pick_files() else {
            return;
        };

        let mut files = Vec::with_capacity(paths.len());
        let mut read_errors = 0usize;
        for path in paths {
            match xasdata::ColumnFile::from_path(&path) {
                Ok(cf) => files.push(cf),
                Err(_) => read_errors += 1,
            }
        }

        // "Output file numbering": append a sequence number to each .xmu so the
        // batch outputs stay distinct (the original toggles this in the dialog).
        let number = self.import.as_ref().is_some_and(|i| i.number_outputs);
        let mut built = 0usize;
        let mut build_errors = 0usize;
        let mut written = 0usize;
        let mut write_failed = 0usize;
        for (i, result) in xasdata::make_xmu_batch(&files, &spec)
            .into_iter()
            .enumerate()
        {
            match result {
                Ok(group) => {
                    let index = number.then_some(i + 1);
                    let input = files.get(i).and_then(|f| f.path.as_deref());
                    match self.write_xmu_output(
                        input,
                        &group.label,
                        &group.energy,
                        &group.mu,
                        index,
                    ) {
                        Ok(_) => written += 1,
                        Err(_) => write_failed += 1,
                    }
                    self.session.add_group(group);
                    built += 1;
                }
                Err(_) => build_errors += 1,
            }
        }

        if built > 0 {
            self.reduction.graph = GraphType::MuBkg;
            self.clean_undo.clear();
            self.edit_xmu.reset_seed();
            self.plot_data.mark_dirty();
            self.replot_graph();
        }
        self.status = format!(
            "Batch μ(E): built {built}, wrote {written} .xmu \
             ({write_failed} not written), build errors {build_errors}, \
             unreadable {read_errors}."
        );
    }

    /// Open the Edit-μ(E) window, seeding its energy widgets from the current
    /// group's span the first time it is shown against fresh data.
    fn open_edit_xmu(&mut self) {
        if let Some((lo, hi)) = self
            .session
            .current_group()
            .filter(|g| !g.energy.is_empty())
            .map(|g| (g.energy[0], *g.energy.last().unwrap()))
        {
            self.edit_xmu.seed_span(lo, hi);
        }
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
                self.status = "Wavelet transform computed.".to_owned();
            }
            Some((None, _)) => {
                self.wavelet.set_result(
                    None,
                    "Transform failed — check R range vs k-step (Nyquist).".to_owned(),
                );
                self.status = "Wavelet transform failed (R range exceeds Nyquist?).".to_owned();
            }
            None => {
                self.wavelet.set_result(
                    None,
                    "No χ(k) — run AUTOBK on the current group first.".to_owned(),
                );
                self.status = "No χ(k) for the wavelet transform — run AUTOBK first.".to_owned();
            }
        }
    }

    /// Apply a cleanup edit (or undo) to the current group, recording an undo
    /// snapshot for edits that actually changed the spectrum.
    fn apply_clean_action(&mut self, action: CleanAction) {
        const UNDO_CAP: usize = 32;

        if let CleanAction::Undo = action {
            match (self.clean_undo.pop(), self.session.current) {
                (Some(prev), Some(idx)) if idx < self.session.groups.len() => {
                    self.session.groups[idx] = prev;
                    self.status = "Undid last edit.".to_owned();
                    self.plot_data.mark_dirty();
                    self.replot_graph();
                }
                _ => {}
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
            CleanAction::DeglitchPoint(e) => {
                let n = xasdata::deglitch_point(g, e);
                (n > 0, format!("Deglitch: removed {n} point(s)"))
            }
            CleanAction::DeglitchRange(side, e1, e2) => {
                let n = xasdata::deglitch_range(g, side, e1, e2);
                (n > 0, format!("Deglitch range: removed {n} point(s)"))
            }
            CleanAction::Trim(lo, hi) => {
                let n = xasdata::trim(g, lo, hi);
                (n > 0, format!("Trim: removed {n} point(s)"))
            }
            CleanAction::Smooth(sigma, form) => {
                let ok = xasdata::smooth_mu(g, sigma, form);
                let m = if ok {
                    format!("Smoothed μ(E) (σ = {sigma:.2} eV)")
                } else {
                    "Smoothing skipped (grid too short or not increasing).".to_owned()
                };
                (ok, m)
            }
            CleanAction::Undo => unreachable!("handled above"),
        };

        self.status = msg;
        if changed {
            self.clean_undo.push(snapshot);
            if self.clean_undo.len() > UNDO_CAP {
                self.clean_undo.remove(0);
            }
            self.plot_data.mark_dirty();
            self.replot_graph();
        }
    }

    /// Redraw the shared plot for the active group according to the selected
    /// [`GraphType`]. Curves that need a stage not yet computed are skipped.
    fn replot_graph(&mut self) {
        const BLUE: Color32 = Color32::from_rgb(0x1f, 0x77, 0xb4);
        const ORANGE: Color32 = Color32::from_rgb(0xff, 0x7f, 0x0e);

        self.plot.clear_curves();
        let graph = self.reduction.graph;
        let kweight = self.reduction.kweight;
        let Some(g) = self.session.current_group() else {
            return;
        };

        match graph {
            GraphType::MuBkg => {
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot.set_graph_y_label("μ(E)", siplot::YAxis::Left);
                if !g.energy.is_empty() {
                    self.plot
                        .add_curve_with_legend(&g.energy, &g.mu, BLUE, "μ(E)");
                }
                if let Some(bkg) = &g.bkg {
                    self.plot
                        .add_curve_with_legend(&g.energy, bkg, ORANGE, "background");
                }
            }
            GraphType::Norm => {
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot
                    .set_graph_y_label("normalized μ(E)", siplot::YAxis::Left);
                // Athena/XAFSView convention: the "normalized" view shows the
                // *flattened* μ (post-edge curvature removed, lifted to ~1),
                // not the plain edge-step normalization (which keeps the
                // post-edge slope). Fall back to `norm` only if flattening is
                // somehow unavailable — `reduce` always sets them together.
                if let Some(flat) = g.flat.as_ref().or(g.norm.as_ref()) {
                    self.plot
                        .add_curve_with_legend(&g.energy, flat, BLUE, "norm");
                }
            }
            GraphType::Deriv => {
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot.set_graph_y_label("d μ / dE", siplot::YAxis::Left);
                if let Some(d) = &g.dmude {
                    self.plot.add_curve_with_legend(&g.energy, d, BLUE, "dμ/dE");
                }
            }
            GraphType::KChi => {
                self.plot.set_graph_x_label("k (Å⁻¹)");
                self.plot.set_graph_y_label("kʷ·χ(k)", siplot::YAxis::Left);
                if let (Some(k), Some(chi)) = (&g.k, &g.chi) {
                    let y: Vec<f64> = k
                        .iter()
                        .zip(chi)
                        .map(|(&kk, &c)| c * kk.powi(kweight))
                        .collect();
                    self.plot.add_curve_with_legend(k, &y, BLUE, "kʷ·χ(k)");
                }
            }
            GraphType::ChiR => {
                self.plot.set_graph_x_label("R (Å)");
                self.plot.set_graph_y_label("|χ(R)|", siplot::YAxis::Left);
                if let (Some(r), Some(mag)) = (&g.r, &g.chir_mag) {
                    self.plot.add_curve_with_legend(r, mag, BLUE, "|χ(R)|");
                }
            }
        }
    }

    /// The Feffit tab: fit controls on the left, data-vs-model plot on the right.
    fn feffit_tab(&mut self, ui: &mut egui::Ui) {
        let mut feffit_action = None;
        // The original Feffit form's bottom "Exit" button is tab chrome, not part
        // of the reusable control set the batch window also renders, so it lives
        // in this wrapper rather than in `FeffitUi::controls`. Hide Log / Load
        // result don't map to the engine and are omitted per the functional-only
        // field rule; "Send to plot data" opens the group's Plot Data overlay.
        let mut exit = false;
        egui::Panel::left("feffit_controls")
            .resizable(true)
            .default_size(380.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    feffit_action = self.feffit.controls(ui);
                    // Exit sits directly below the controls (그림 1-2-2-2), no gap.
                    ui.add_space(6.0);
                    if crate::widgets::exit(ui, crate::widgets::ROW_BTN).clicked() {
                        exit = true;
                    }
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            crate::plot::show(&mut self.plot, ui);
        });

        if exit {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        match feffit_action {
            Some(FeffitAction::AddPath) => self.add_feff_path(),
            Some(FeffitAction::Run) => self.run_feffit(),
            Some(FeffitAction::Replot) => self.replot_feffit(),
            Some(FeffitAction::SendToPlotData) => self.send_feffit_to_plot_data(),
            None => {}
        }
    }

    /// The Feffit_txt tab: a plain-text report of the last fit.
    fn feffit_txt_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Feffit_txt");
            ui.add_space(8.0);
            ui.weak("Text report of the last FEFFIT fit");
            // The original Feffit_txt form's "Exit" button (this view is the
            // functional equivalent of its feffit text output).
            ui.add_space(12.0);
            if crate::widgets::exit(ui, crate::widgets::ROW_BTN).clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
        ui.separator();
        let report = self.feffit.report_text();
        if report.is_empty() {
            ui.weak("Run a fit on the Feffit tab to see the report.");
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

    /// The About tab: program/version information, mirroring XAFSView's About.
    fn about_tab(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(8.0);
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
        });
    }

    /// Open a dialog and add the chosen Feff path file(s) to the fit.
    fn add_feff_path(&mut self) {
        let mut dlg = rfd::FileDialog::new();
        if let Some(dir) = &self.session.folders.feff_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(paths) = dlg.pick_files() else {
            return;
        };
        let mut added = 0usize;
        for path in paths {
            match feffdat::FeffPath::from_path(&path) {
                Ok(fp) => {
                    let label = path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "path".to_owned());
                    self.feffit.add_path(label, fp);
                    added += 1;
                }
                Err(e) => self.status = format!("Path load failed: {e}"),
            }
        }
        if added > 0 {
            self.status = format!("Added {added} Feff path(s).");
        }
    }

    /// Open a dialog and add the chosen Feff path file(s) to one batch config.
    fn add_feff_path_to_batch(&mut self, idx: usize) {
        let mut dlg = rfd::FileDialog::new();
        if let Some(dir) = &self.session.folders.feff_dir {
            dlg = dlg.set_directory(dir);
        }
        let Some(paths) = dlg.pick_files() else {
            return;
        };
        let mut added = 0usize;
        for path in paths {
            match feffdat::FeffPath::from_path(&path) {
                Ok(fp) => {
                    let label = path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "path".to_owned());
                    self.feffit_batch.add_path_to(idx, label, fp);
                    added += 1;
                }
                Err(e) => self.status = format!("Path load failed: {e}"),
            }
        }
        if added > 0 {
            self.status = format!("Added {added} Feff path(s) to batch group.");
        }
    }

    /// Write the batch "Save Items" files to the work folder (the original's
    /// results folder; falls back to the data folder), reporting the outcome.
    fn write_saved_items(&mut self, files: Vec<(String, String)>) {
        let Some(dir) = self
            .session
            .folders
            .work_dir
            .clone()
            .or_else(|| self.session.folders.data_dir.clone())
        else {
            self.status = "Set a work folder (Folders tab) before saving items.".to_owned();
            return;
        };
        let mut written = 0usize;
        let mut failed = 0usize;
        for (name, content) in &files {
            match std::fs::write(dir.join(name), content) {
                Ok(()) => written += 1,
                Err(_) => failed += 1,
            }
        }
        let extra = if failed > 0 {
            format!(" ({failed} failed)")
        } else {
            String::new()
        };
        self.status = format!("Saved {written} item file(s) to {}{extra}.", dir.display());
    }

    /// Run the FEFFIT fit on the current group's `chi(k)` and redraw.
    fn run_feffit(&mut self) {
        let Some((label, k, chi)) =
            self.session
                .current_group()
                .and_then(|g| match (&g.k, &g.chi) {
                    (Some(k), Some(chi)) => Some((g.label.clone(), k.clone(), chi.clone())),
                    _ => None,
                })
        else {
            self.status = "No chi(k) for the current group — run AUTOBK first.".to_owned();
            return;
        };
        match self.feffit.run(&k, &chi) {
            Ok(msg) => {
                self.feffit_fit_group = Some(label);
                self.status = msg;
                self.replot_feffit();
            }
            Err(e) => self.status = e,
        }
    }

    /// Redraw the shared plot with the last fit's data vs model in the selected
    /// space/part.
    fn replot_feffit(&mut self) {
        use crate::plot_data::{FIT_DATA, FIT_MODEL};

        self.plot.clear_curves();
        let (space, part) = self.feffit.plot_selection();
        let Some(p) = self.feffit.plot() else {
            return;
        };

        let (x, data_y, model_y, xlabel, ylabel) = p.series(space, part);
        self.plot.set_graph_x_label(xlabel);
        self.plot.set_graph_y_label(ylabel, siplot::YAxis::Left);
        if !x.is_empty() {
            self.plot
                .add_curve_with_legend(&x, &data_y, FIT_DATA, "data");
            self.plot
                .add_curve_with_legend(&x, &model_y, FIT_MODEL, "model");
        }
    }

    /// Hand the last Feffit fit's data + model curves to the Plot Data window
    /// (the Feffit form's "Send to plot data"), in the currently-selected space.
    fn send_feffit_to_plot_data(&mut self) {
        let label = match &self.feffit_fit_group {
            Some(g) => format!("Feffit fit — {g}"),
            None => "Feffit fit".to_owned(),
        };
        let (space, part) = self.feffit.plot_selection();
        let Some(p) = self.feffit.plot() else {
            self.status = "Run a fit first, then send it to Plot Data.".to_owned();
            return;
        };
        let (x, data_y, model_y, xlabel, ylabel) = p.series(space, part);
        self.plot_data
            .set_fit_overlay(label, xlabel, ylabel, x, data_y, model_y);
        self.status = "Sent the Feffit fit (data + model) to Plot Data.".to_owned();
    }

    /// The Autobk tab: import + reduction controls on the left, plot on the right.
    fn autobk_tab(&mut self, ui: &mut egui::Ui) {
        let mut open_clicked = false;
        let mut start_clicked = false;
        let mut exit_clicked = false;
        let mut edit_clicked = false;
        let mut theory_pick = false;
        let mut theory_clear = false;
        let mut import_action = None;
        let mut replot = false;

        // μ(E)-dependent actions (Autobk Start, Edit μ(E)) need a real spectrum,
        // not a directly-loaded χ(k) group.
        let has_mu = self
            .session
            .current_group()
            .is_some_and(|g| !g.mu.is_empty());
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
            .show_inside(ui, |ui| {
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Autobk");

                        // info rows: Title (group label) / Data File / Theory
                        egui::Grid::new("autobk_info")
                            .num_columns(2)
                            .spacing([6.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Title");
                                match self.session.current_group_mut() {
                                    Some(g) => {
                                        ui.text_edit_singleline(&mut g.label);
                                    }
                                    None => {
                                        ui.weak("— no data —");
                                    }
                                }
                                ui.end_row();

                                ui.label("Data File");
                                match &data_file {
                                    Some(name) => {
                                        ui.monospace(name.as_str());
                                    }
                                    None => {
                                        ui.weak("—");
                                    }
                                }
                                ui.end_row();

                                ui.label("Theory");
                                ui.horizontal(|ui| {
                                    if ui
                                        .button("Load…")
                                        .on_hover_text(
                                            "FEFF chi.dat standard for the background constraint",
                                        )
                                        .clicked()
                                    {
                                        theory_pick = true;
                                    }
                                    match &theory_name {
                                        Some(name) => {
                                            ui.monospace(name.as_str());
                                            if ui.small_button("✕").clicked() {
                                                theory_clear = true;
                                            }
                                        }
                                        None => {
                                            ui.weak("(none)");
                                        }
                                    }
                                });
                                ui.end_row();
                            });

                        // column chooser, shown while a raw / μ file is open
                        if let Some(import) = self.import.as_mut() {
                            ui.separator();
                            import_action = import.ui(ui);
                        }

                        // the "Autobk parameters" grid (+ loading mode + graph type)
                        ui.separator();
                        replot = self.reduction.controls(ui);

                        // The 2×2 action block (그림 1-2-1-1) sits directly below
                        // the parameters — no pinned-bottom gap.
                        ui.add_space(8.0);
                        use crate::widgets::{self, CHUNKY_BTN};
                        ui.horizontal(|ui| {
                            if widgets::action(ui, "Open New file", CHUNKY_BTN).clicked() {
                                open_clicked = true;
                            }
                            if widgets::primary(ui, "Autobk Start", CHUNKY_BTN, has_mu).clicked() {
                                start_clicked = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            if widgets::exit(ui, CHUNKY_BTN).clicked() {
                                exit_clicked = true;
                            }
                            if widgets::action_enabled(ui, "Edit μ(E)", CHUNKY_BTN, has_mu)
                                .clicked()
                            {
                                edit_clicked = true;
                            }
                        });
                    });
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            crate::plot::show(&mut self.plot, ui);
        });

        if open_clicked {
            self.open_new_file();
        }
        if theory_pick {
            self.open_theory_file();
        }
        if theory_clear {
            self.reduction.theory = None;
        }
        if let Some(ImportAction::CalcXmu) = import_action {
            self.calc_xmu();
        }
        if start_clicked {
            self.run_reduction();
        }
        if edit_clicked {
            self.open_edit_xmu();
        }
        if exit_clicked {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if replot {
            self.replot_graph();
        }
    }

    /// The top menu bar (XAFSView's menus; most entries are stubs until their
    /// phase lands).
    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            let mut open_clicked = false;
            ui.menu_button("File", |ui| {
                if ui.button("Open data file…").clicked() {
                    open_clicked = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            if open_clicked {
                self.tab = Tab::Autobk;
                self.open_file();
            }
            let mut open_plot_data = false;
            let mut run_multi_autobk = false;
            let mut make_xmu_files = false;
            let mut open_feffit_batch = false;
            ui.menu_button("Multiple_data", |ui| {
                if ui.button("Plot Data…").clicked() {
                    open_plot_data = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Multiple AUTOBK (reduce all loaded)").clicked() {
                    run_multi_autobk = true;
                    ui.close();
                }
                if ui.button("Make μ(E) from files…").clicked() {
                    make_xmu_files = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Feffit batch (per-group)…").clicked() {
                    open_feffit_batch = true;
                    ui.close();
                }
            });
            if open_feffit_batch {
                self.feffit_batch.open = true;
            }
            if open_plot_data {
                self.plot_data.open = true;
                self.plot_data.mark_dirty();
            }
            if run_multi_autobk {
                self.run_multi_autobk();
            }
            if make_xmu_files {
                self.tab = Tab::Autobk;
                self.make_xmu_from_files();
            }
            let mut edit_clicked = false;
            ui.menu_button("Smoothing", |ui| {
                if ui.button("Edit μ(E) (deglitch / trim / smooth)…").clicked() {
                    edit_clicked = true;
                    ui.close();
                }
            });
            if edit_clicked {
                self.tab = Tab::Autobk;
                self.open_edit_xmu();
            }
            ui.menu_button("Periodic table", |ui| {
                if ui.button("Periodic table + atom data…").clicked() {
                    self.periodic.open = true;
                    ui.close();
                }
            });
            ui.menu_button("Tools", |ui| {
                if ui.button("Wavelet transform |W(k,R)|…").clicked() {
                    self.wavelet.open = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("LCF (linear combination)…").clicked() {
                    self.lcf.open = true;
                    ui.close();
                }
                if ui.button("PCA (principal components)…").clicked() {
                    self.pca.open = true;
                    ui.close();
                }
                ui.separator();
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
                if ui.button("Ion chamber / gas absorption…").clicked() {
                    self.ion_chamber.open = true;
                    ui.close();
                }
                if ui.button("Powder weight…").clicked() {
                    self.powder.open = true;
                    ui.close();
                }
                if ui.button("k ↔ E conversion…").clicked() {
                    self.ke_convert.open = true;
                    ui.close();
                }
                if ui
                    .button("Extract XAS measured time (time-resolved)…")
                    .clicked()
                {
                    self.time_resolved.open = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Plot Sites (3D cluster)…").clicked() {
                    self.plot_sites.open = true;
                    ui.close();
                }
            });
            ui.menu_button("Change BG", |ui| {
                if ui.button("Toggle dark / light").clicked() {
                    let dark = ui.ctx().global_style().visuals.dark_mode;
                    ui.ctx().set_visuals(if dark {
                        egui::Visuals::light()
                    } else {
                        egui::Visuals::dark()
                    });
                }
            });
        });
    }

    /// The Folders tab: configure the data/work/feff working directories.
    fn folders_panel(&mut self, ui: &mut egui::Ui) {
        let mut exit_clicked = false;
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Folders");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.strong("Change folder");
                    });
                });
                ui.label(
                    "Working directories used for file dialogs and output. Type a path or push.",
                );
                ui.add_space(8.0);

                // Only the folders this app actually uses (functional fields only):
                // the original's Base / Sub base / Autobk / Feffit_txt / IE Explorer
                // / Results folders have no engine mapping here.
                egui::Grid::new("folders_grid")
                    .num_columns(3)
                    .spacing([8.0, 8.0])
                    .show(ui, |ui| {
                        folder_row(ui, "Data folder", &mut self.session.folders.data_dir);
                        ui.end_row();
                        folder_row(ui, "Work folder", &mut self.session.folders.work_dir);
                        ui.end_row();
                        folder_row(ui, "FEFF folder", &mut self.session.folders.feff_dir);
                        ui.end_row();
                    });

                // Exit sits directly below the folder rows (그림 1-2-6), no gap.
                ui.add_space(8.0);
                if crate::widgets::exit(ui, crate::widgets::ROW_BTN).clicked() {
                    exit_clicked = true;
                }
            });
        });
        if exit_clicked {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

impl eframe::App for XafsViewApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        egui::Panel::top("menubar").show_inside(ui, |ui| {
            self.menu_bar(ui);
        });

        // Horizontal tab strip directly under the menu bar, mirroring XAFSView's
        // top tab row (Autobk … About) rather than a vertical side list.
        egui::Panel::top("tabs").show_inside(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                for tab in Tab::ALL {
                    ui.selectable_value(&mut self.tab, tab, tab.label());
                }
            });
            ui.add_space(2.0);
        });

        // On a tab switch, repopulate the shared plot for the newly active tab
        // so it shows that tab's curves rather than the previous tab's.
        if self.tab != self.last_tab {
            match self.tab {
                Tab::Autobk => self.replot_graph(),
                Tab::Feffit => self.replot_feffit(),
                _ => {}
            }
            self.last_tab = self.tab;
        }

        // The common error/status strip at the top of the body (below the tabs),
        // matching XAFSView's shared text line shown on every tab.
        egui::Panel::top("status").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(format!("{} group(s) loaded", self.session.groups.len()));
                });
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| match self.tab {
            Tab::Folders => self.folders_panel(ui),
            Tab::Autobk => self.autobk_tab(ui),
            Tab::Feffit => self.feffit_tab(ui),
            Tab::FeffitTxt => self.feffit_txt_tab(ui),
            Tab::Atoms => match self.atoms_tab.ui(ui, &mut self.feff_inp) {
                // Hand off to the Feff tab so the user can run the new input.
                Some(AtomsAction::BuiltFeffInp) => self.tab = Tab::Feff,
                Some(AtomsAction::Exit) => ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close),
                None => {}
            },
            Tab::Feff => {
                match self.feff_tab.ui(
                    ui,
                    &mut self.feff_inp,
                    self.session.folders.work_dir.as_deref(),
                ) {
                    Some(FeffAction::ViewStructure) => self.plot_sites.open = true,
                    Some(FeffAction::Exit) => {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close)
                    }
                    None => {}
                }
            }
            Tab::About => self.about_tab(ui),
        });

        // The Edit-μ(E) window floats above the panels and is driven from the
        // Autobk tab button / Smoothing menu.
        let has_group = self
            .session
            .current_group()
            .is_some_and(|g| !g.mu.is_empty());
        let npts = self.session.current_group().map_or(0, |g| g.len());
        let can_undo = !self.clean_undo.is_empty();
        if let Some(action) = self.edit_xmu.show(ui.ctx(), has_group, npts, can_undo) {
            self.apply_clean_action(action);
        }

        // The wavelet-transform window floats above the panels; Compute reads the
        // current group's χ(k). Enabled only when the active group has χ(k).
        let has_chi = self
            .session
            .current_group()
            .is_some_and(|g| g.k.is_some() && g.chi.is_some());
        if let Some(WaveletAction::Compute) = self.wavelet.show(ui.ctx(), has_chi) {
            self.run_wavelet();
        }

        // The Plot Data window overlays several groups; it owns its own plot and
        // reads (but does not mutate) the session's groups.
        self.plot_data.show(ui.ctx(), &self.session.groups);

        // The multi-FEFFIT batch window: each group has its own fit config seeded
        // from the Feffit tab (the template). The add-path dialog and Save Items
        // file writes bubble up for the app to service.
        match self
            .feffit_batch
            .show(ui.ctx(), &self.session.groups, &self.feffit)
        {
            Some(BatchAction::AddPath(idx)) => self.add_feff_path_to_batch(idx),
            Some(BatchAction::SaveItems(files)) => self.write_saved_items(files),
            None => {}
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

/// One labelled folder row: the current path (or "(not set)") and a Browse
/// button that opens a native folder picker.
fn folder_row(ui: &mut egui::Ui, label: &str, dir: &mut Option<std::path::PathBuf>) {
    ui.label(label);
    // Editable path text (mirrors the original's typeable folder field), kept in
    // sync with the PathBuf; an empty string clears it.
    let mut text = dir
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    if ui
        .add(
            egui::TextEdit::singleline(&mut text)
                .desired_width(300.0)
                .hint_text("(not set)"),
        )
        .changed()
    {
        *dir = if text.trim().is_empty() {
            None
        } else {
            Some(std::path::PathBuf::from(text))
        };
    }
    if crate::widgets::action(ui, "push", crate::widgets::PUSH_BTN).clicked()
        && let Some(picked) = rfd::FileDialog::new().pick_folder()
    {
        *dir = Some(picked);
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
    std::fs::write(path, s)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
