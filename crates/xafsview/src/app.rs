//! The application shell: window layout, the tab bar, the menu bar, and the
//! shared `siplot` plot. Individual tabs fill in their panels in later phases;
//! Phase 0 wires the skeleton and makes the **Folders** tab functional.

use eframe::egui;
use egui::Color32;
use siplot::Plot1D;
use xasdata::{ColumnFile, Session, XasGroup};

use crate::analysis_ui::{LcfWindow, PcaWindow};
use crate::atoms_ui::{AtomsAction, AtomsTab, FeffTab, PlotSitesWindow};
use crate::calc_ui::{IonChamberWindow, PeriodicTableWindow, PowderWindow};
use crate::clean_ui::{CleanAction, EditXmuState};
use crate::feffit_batch::{BatchAction, FeffitBatch};
use crate::feffit_ui::{FeffitAction, FeffitUi};
use crate::import::{ImportAction, ImportState};
use crate::mback_ui::MbackWindow;
use crate::plot_data::PlotDataWindow;
use crate::reduce_ui::{GraphType, ReductionAction, ReductionUi};
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
}

impl Tab {
    /// All tabs in strip order.
    const ALL: [Tab; 6] = [
        Tab::Autobk,
        Tab::Feffit,
        Tab::FeffitTxt,
        Tab::Atoms,
        Tab::Feff,
        Tab::Folders,
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
        }
    }
}

/// The eframe application: the data [`Session`], the active tab, and the shared
/// plot (one `siplot::Plot1D` reused across the plot-bearing tabs).
pub struct XafsViewApp {
    session: Session,
    tab: Tab,
    plot: Plot1D,
    /// The in-progress data import (column → role mapping), if a raw file is open.
    import: Option<ImportState>,
    /// Reduction (normalize/autobk/FT) parameters and the active graph type.
    reduction: ReductionUi,
    /// FEFFIT tab state: paths, variables, transform, and last fit result.
    feffit: FeffitUi,
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
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must use the wgpu renderer (NativeOptions.renderer = Wgpu)");

        let mut plot = crate::plot::new_plot1d(render_state, 0);
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
        let h = plot.add_curve(&x, &y, Color32::from_rgb(0x1f, 0x77, 0xb4));
        plot.set_item_legend(h, "demo edge");

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

    /// Build `mu(E)` from the current import selection and add it as a group.
    fn calc_xmu(&mut self) {
        let Some(import) = self.import.as_ref() else {
            return;
        };
        let spec = import.to_spec();
        match xasdata::build_mu(&import.file, &spec) {
            Ok((energy, mu)) => {
                let label = import
                    .file
                    .path
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("group{}", self.session.groups.len() + 1));
                let n = energy.len();
                self.session.add_group(XasGroup::from_mu(label, energy, mu));
                self.status = format!("Built μ(E): {n} points");
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

        let mut built = 0usize;
        let mut build_errors = 0usize;
        for result in xasdata::make_xmu_batch(&files, &spec) {
            match result {
                Ok(group) => {
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
            "Batch μ(E): built {built}, build errors {build_errors}, unreadable {read_errors}."
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
                    let h = self.plot.add_curve(&g.energy, &g.mu, BLUE);
                    self.plot.set_item_legend(h, "μ(E)");
                }
                if let Some(bkg) = &g.bkg {
                    let h = self.plot.add_curve(&g.energy, bkg, ORANGE);
                    self.plot.set_item_legend(h, "background");
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
                    let h = self.plot.add_curve(&g.energy, flat, BLUE);
                    self.plot.set_item_legend(h, "norm");
                }
            }
            GraphType::Deriv => {
                self.plot.set_graph_x_label("Energy (eV)");
                self.plot.set_graph_y_label("d μ / dE", siplot::YAxis::Left);
                if let Some(d) = &g.dmude {
                    let h = self.plot.add_curve(&g.energy, d, BLUE);
                    self.plot.set_item_legend(h, "dμ/dE");
                }
            }
            GraphType::KChi => {
                self.plot.set_graph_x_label("k (Å⁻¹)");
                self.plot.set_graph_y_label("k^w·χ(k)", siplot::YAxis::Left);
                if let (Some(k), Some(chi)) = (&g.k, &g.chi) {
                    let y: Vec<f64> = k
                        .iter()
                        .zip(chi)
                        .map(|(&kk, &c)| c * kk.powi(kweight))
                        .collect();
                    let h = self.plot.add_curve(k, &y, BLUE);
                    self.plot.set_item_legend(h, "k^w·χ(k)");
                }
            }
            GraphType::ChiR => {
                self.plot.set_graph_x_label("R (Å)");
                self.plot.set_graph_y_label("|χ(R)|", siplot::YAxis::Left);
                if let (Some(r), Some(mag)) = (&g.r, &g.chir_mag) {
                    let h = self.plot.add_curve(r, mag, BLUE);
                    self.plot.set_item_legend(h, "|χ(R)|");
                }
            }
        }
    }

    /// The Feffit tab: fit controls on the left, data-vs-model plot on the right.
    fn feffit_tab(&mut self, ui: &mut egui::Ui) {
        let mut feffit_action = None;
        egui::Panel::left("feffit_controls")
            .resizable(true)
            .default_size(380.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    feffit_action = self.feffit.controls(ui);
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            crate::plot::toolbar(&mut self.plot, ui);
            self.plot.show(ui);
        });

        match feffit_action {
            Some(FeffitAction::AddPath) => self.add_feff_path(),
            Some(FeffitAction::Run) => self.run_feffit(),
            Some(FeffitAction::Replot) => self.replot_feffit(),
            None => {}
        }
    }

    /// The Feffit_txt tab: a plain-text report of the last fit.
    fn feffit_txt_tab(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Feffit_txt");
            ui.add_space(8.0);
            ui.weak("Text report of the last FEFFIT fit");
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

    /// Run the FEFFIT fit on the current group's `chi(k)` and redraw.
    fn run_feffit(&mut self) {
        let Some((k, chi)) = self
            .session
            .current_group()
            .and_then(|g| match (&g.k, &g.chi) {
                (Some(k), Some(chi)) => Some((k.clone(), chi.clone())),
                _ => None,
            })
        else {
            self.status = "No chi(k) for the current group — run AUTOBK first.".to_owned();
            return;
        };
        match self.feffit.run(&k, &chi) {
            Ok(msg) => {
                self.status = msg;
                self.replot_feffit();
            }
            Err(e) => self.status = e,
        }
    }

    /// Redraw the shared plot with the last fit's data vs model in the selected
    /// space/part.
    fn replot_feffit(&mut self) {
        const BLUE: Color32 = Color32::from_rgb(0x1f, 0x77, 0xb4);
        const RED: Color32 = Color32::from_rgb(0xd6, 0x27, 0x28);

        self.plot.clear_curves();
        let (space, part) = self.feffit.plot_selection();
        let Some(p) = self.feffit.plot() else {
            return;
        };

        let (x, data_y, model_y, xlabel, ylabel) = p.series(space, part);
        self.plot.set_graph_x_label(xlabel);
        self.plot.set_graph_y_label(ylabel, siplot::YAxis::Left);
        if !x.is_empty() {
            let hd = self.plot.add_curve(&x, &data_y, BLUE);
            self.plot.set_item_legend(hd, "data");
            let hm = self.plot.add_curve(&x, &model_y, RED);
            self.plot.set_item_legend(hm, "model");
        }
    }

    /// The Autobk tab: import + reduction controls on the left, plot on the right.
    fn autobk_tab(&mut self, ui: &mut egui::Ui) {
        let mut open_clicked = false;
        let mut edit_clicked = false;
        let mut import_action = None;
        let mut reduction_action = None;
        let has_group = self
            .session
            .current_group()
            .is_some_and(|g| !g.mu.is_empty());

        egui::Panel::left("autobk_controls")
            .resizable(true)
            .default_size(340.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Autobk");
                    ui.horizontal(|ui| {
                        if ui.button("Open data file…").clicked() {
                            open_clicked = true;
                        }
                        if ui
                            .add_enabled(has_group, egui::Button::new("Edit μ(E)…"))
                            .clicked()
                        {
                            edit_clicked = true;
                        }
                    });
                    ui.separator();
                    match self.import.as_mut() {
                        Some(import) => import_action = import.ui(ui),
                        None => {
                            ui.weak("Open a beamline column file to build μ(E).");
                        }
                    }
                    if has_group {
                        ui.separator();
                        reduction_action = self.reduction.controls(ui);
                    }
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            crate::plot::toolbar(&mut self.plot, ui);
            self.plot.show(ui);
        });

        if open_clicked {
            self.open_file();
        }
        if edit_clicked {
            self.open_edit_xmu();
        }
        if let Some(ImportAction::CalcXmu) = import_action {
            self.calc_xmu();
        }
        match reduction_action {
            Some(ReductionAction::Run) => self.run_reduction(),
            Some(ReductionAction::Replot) => self.replot_graph(),
            None => {}
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
        ui.heading("Folders");
        ui.label("Configure the working directories used for file dialogs and output.");
        ui.add_space(8.0);

        egui::Grid::new("folders_grid")
            .num_columns(3)
            .spacing([12.0, 8.0])
            .show(ui, |ui| {
                folder_row(ui, "Data folder", &mut self.session.folders.data_dir);
                ui.end_row();
                folder_row(ui, "Work folder", &mut self.session.folders.work_dir);
                ui.end_row();
                folder_row(ui, "FEFF folder", &mut self.session.folders.feff_dir);
                ui.end_row();
            });
    }
}

impl eframe::App for XafsViewApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        egui::Panel::top("menubar").show_inside(ui, |ui| {
            self.menu_bar(ui);
        });

        egui::Panel::left("tabs")
            .resizable(false)
            .exact_size(120.0)
            .show_inside(ui, |ui| {
                ui.add_space(4.0);
                for tab in Tab::ALL {
                    ui.selectable_value(&mut self.tab, tab, tab.label());
                }
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

        egui::Panel::bottom("status").show_inside(ui, |ui| {
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
            Tab::Atoms => {
                if let Some(AtomsAction::BuiltFeffInp) = self.atoms_tab.ui(ui, &mut self.feff_inp) {
                    // Hand off to the Feff tab so the user can run the new input.
                    self.tab = Tab::Feff;
                }
            }
            Tab::Feff => self.feff_tab.ui(
                ui,
                &mut self.feff_inp,
                self.session.folders.work_dir.as_deref(),
            ),
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
        // from the Feffit tab (the template). Only the add-path dialog bubbles up.
        if let Some(BatchAction::AddPath(idx)) =
            self.feffit_batch
                .show(ui.ctx(), &self.session.groups, &self.feffit)
        {
            self.add_feff_path_to_batch(idx);
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

        // The Plot Sites 3D viewer parses the shared feff.inp into a point cloud.
        // It needs the wgpu render state each frame (unlike the 2D Plot1D, which
        // caches it), so it is fed from the eframe Frame here.
        if let Some(rs) = frame.wgpu_render_state() {
            self.plot_sites.show(ui.ctx(), rs, &self.feff_inp);
        }
    }
}

/// One labelled folder row: the current path (or "(not set)") and a Browse
/// button that opens a native folder picker.
fn folder_row(ui: &mut egui::Ui, label: &str, dir: &mut Option<std::path::PathBuf>) {
    ui.label(label);
    match dir.as_ref() {
        Some(p) => ui.monospace(p.display().to_string()),
        None => ui.weak("(not set)"),
    };
    if ui.button("Browse…").clicked()
        && let Some(picked) = rfd::FileDialog::new().pick_folder()
    {
        *dir = Some(picked);
    }
}
