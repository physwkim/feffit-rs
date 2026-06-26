//! Autobk-tab reduction controls: edge/background/FT parameters laid out as the
//! original XAFSView "Autobk parameters" 3-column grid, plus the file-loading
//! mode and graph selector. [`ReductionUi`] holds the editable values and
//! renders the parameter grid; the app turns its
//! [`pre_params`](ReductionUi::pre_params) /
//! [`autobk_params`](ReductionUi::autobk_params) / [`ft_params`](ReductionUi::ft_params)
//! into engine calls via `feffit::xasdata::reduce`, and reads [`graph`](ReductionUi::graph)
//! to decide what to plot. The Title / Data File / Theory rows and the
//! Open / Start / Exit / Edit button cluster live in the app (they need session
//! and file-dialog access).

use std::path::PathBuf;

use eframe::egui;
use feffit::xasdata::{AutobkParams, FtParams, PreEdgeParams, Window};

/// Which reduction stage to display on the plot.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    /// `XMU + BKG`: raw `mu(E)` with the AUTOBK background overlaid.
    MuBkg,
    /// `*K.CHI`: k-weighted EXAFS `kʷ·χ(k)`.
    KChi,
    /// `*R.CHI`: magnitude of the Fourier transform `|χ(R)|`.
    ChiR,
    /// `XMU' + XMU''`: first and second derivative of `mu(E)`.
    MuDeriv,
    /// `BKG(E)'`: first derivative of the background `μ0(E)`.
    BkgEDeriv,
    /// `BKG(k)'`: first derivative of the k-space background (`μ0−μ`) in `k`.
    BkgKDeriv,
    /// `XMU' + BKG(E)'`: `mu'(E)` and the background derivative `μ0'(E)` overlaid.
    MuBkgEDeriv,
    /// `BKG(E)''`: second derivative of the background `μ0(E)`.
    BkgEDeriv2,
    /// `Normalize`: edge-step normalized (flattened) `mu(E)`.
    Norm,
}

impl GraphType {
    /// All graph types, in the original XAFSView "Graph type" order.
    pub const ALL: [GraphType; 9] = [
        GraphType::MuBkg,
        GraphType::KChi,
        GraphType::ChiR,
        GraphType::MuDeriv,
        GraphType::BkgEDeriv,
        GraphType::BkgKDeriv,
        GraphType::MuBkgEDeriv,
        GraphType::BkgEDeriv2,
        GraphType::Norm,
    ];

    /// Short menu label (matches the original's "Graph type" radio text).
    pub fn label(self) -> &'static str {
        match self {
            GraphType::MuBkg => "XMU + BKG",
            GraphType::KChi => "*K.CHI",
            GraphType::ChiR => "*R.CHI",
            GraphType::MuDeriv => "XMU' + XMU''",
            GraphType::BkgEDeriv => "BKG(E)'",
            GraphType::BkgKDeriv => "BKG(k)'",
            GraphType::MuBkgEDeriv => "XMU' + BKG(E)'",
            GraphType::BkgEDeriv2 => "BKG(E)''",
            GraphType::Norm => "Normalize",
        }
    }
}

/// How the "Open New file" button interprets the chosen file (the original
/// "Loading file type" ring: Calc. XMU / Load XMU / chi.dat).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LoadingType {
    /// Raw beamline columns → build μ(E) via the column chooser.
    CalcXmu,
    /// An already-built μ(E) file → load via the same column chooser (it adapts
    /// to a 2-column energy/μ file).
    LoadXmu,
    /// A FEFF `chi.dat` (k, χ) → load χ(k) directly as the working data.
    ChiDat,
}

impl LoadingType {
    /// All modes in display order.
    pub const ALL: [LoadingType; 3] = [
        LoadingType::CalcXmu,
        LoadingType::LoadXmu,
        LoadingType::ChiDat,
    ];

    /// Menu label, matching the original ring text.
    pub fn label(self) -> &'static str {
        match self {
            LoadingType::CalcXmu => "Calc. XMU",
            LoadingType::LoadXmu => "Load XMU",
            LoadingType::ChiDat => "chi.dat",
        }
    }
}

/// A loaded theoretical χ(k) standard (a FEFF `chi.dat`) used to constrain the
/// background fit via autobk's `k_std`/`chi_std` — the original "Theory" field.
pub struct TheoryStd {
    /// Source path (shown in the Theory row).
    pub path: PathBuf,
    /// k grid of the standard.
    pub k: Vec<f64>,
    /// χ values of the standard.
    pub chi: Vec<f64>,
}

/// Editable reduction parameters plus the active graph type, loading mode, and
/// the optional theory standard.
pub struct ReductionUi {
    /// Let `pre_edge`/`autobk` find E0 automatically.
    pub e0_auto: bool,
    /// Manual E0 (eV), used when `e0_auto` is off.
    pub e0: f64,
    /// AUTOBK `Rbkg` (Å).
    pub rbkg: f64,
    /// k-weight for the background FT and the kʷ·χ(k) plot.
    pub kweight: i32,
    /// FT window lower bound (Å⁻¹).
    pub kmin: f64,
    /// FT window upper bound (Å⁻¹).
    pub kmax: f64,
    /// FT window taper width (Å⁻¹).
    pub dk: f64,
    /// FT window function.
    pub window: Window,
    /// Low-energy spline clamp weight.
    pub clamp_lo: f64,
    /// High-energy spline clamp weight.
    pub clamp_hi: f64,
    /// Active graph type.
    pub graph: GraphType,
    /// Let `pre_edge` auto-determine the pre-edge / normalization ranges; when
    /// off, the `pre1`/`pre2`/`norm1`/`norm2` values below are used.
    pub pre_norm_auto: bool,
    /// Pre-edge fit lower bound (eV, relative to E0) — original "Pre1".
    pub pre1: f64,
    /// Pre-edge fit upper bound (eV, relative to E0) — original "Pre2".
    pub pre2: f64,
    /// Normalization fit lower bound (eV, relative to E0) — original "Nor1".
    pub norm1: f64,
    /// Normalization fit upper bound (eV, relative to E0) — original "Nor2".
    pub norm2: f64,
    /// Post-edge normalization polynomial order (0=constant … 3=cubic), used
    /// when `pre_norm_auto` is off. XAFSView's normalization is quadratic (2);
    /// the auto rule picks 1 for a <300 eV norm window, which underestimates the
    /// edge step by ~1.6% on the KSW Mn data — exposing this lets the user match.
    pub nnorm: usize,
    /// How "Open New file" interprets the chosen file.
    pub loading: LoadingType,
    /// Optional theoretical χ(k) standard constraining the background fit.
    pub theory: Option<TheoryStd>,
    /// The "Output file" chi base name (original Autobk "Output file" field):
    /// AUTOBK Start writes χ(k) to `<stem>k.chi` and χ(R) to `<stem>r.chi`.
    /// Auto-filled from the loaded file's stem on load; editable. Empty falls
    /// back to the group label at write time.
    pub output_file: String,
}

impl Default for ReductionUi {
    fn default() -> Self {
        Self {
            e0_auto: true,
            e0: 0.0,
            rbkg: 1.0,
            kweight: 2,
            kmin: 2.0,
            kmax: 14.0,
            dk: 1.0,
            window: Window::Hanning,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
            graph: GraphType::MuBkg,
            pre_norm_auto: true,
            // Manual-range fallbacks default to the original XAFSView pre-edge /
            // normalization windows (Pre1/Pre2/Nor1/Nor2 = -200/-50/150/400), so
            // unchecking "auto" reproduces the XAFSView edge step (~1.1% of the
            // UWXAFS value on the KSW Mn data) instead of needing a manual retype.
            pre1: -200.0,
            pre2: -50.0,
            norm1: 150.0,
            norm2: 400.0,
            nnorm: 2,
            loading: LoadingType::CalcXmu,
            theory: None,
            output_file: String::new(),
        }
    }
}

impl ReductionUi {
    /// Pre-edge parameters for this selection.
    pub fn pre_params(&self) -> PreEdgeParams {
        let mut p = PreEdgeParams::default();
        if !self.e0_auto {
            p.e0 = Some(self.e0);
        }
        if !self.pre_norm_auto {
            p.pre1 = Some(self.pre1);
            p.pre2 = Some(self.pre2);
            p.norm1 = Some(self.norm1);
            p.norm2 = Some(self.norm2);
            p.nnorm = Some(self.nnorm);
        }
        p
    }

    /// AUTOBK parameters for this selection, including the theory standard (the
    /// `k_std`/`chi_std` background constraint) when one is loaded.
    pub fn autobk_params(&self) -> AutobkParams {
        AutobkParams {
            rbkg: self.rbkg,
            ek0: (!self.e0_auto).then_some(self.e0),
            kmin: self.kmin,
            kmax: Some(self.kmax),
            kweight: self.kweight,
            dk: self.dk,
            win: self.window,
            clamp_lo: self.clamp_lo,
            clamp_hi: self.clamp_hi,
            k_std: self.theory.as_ref().map(|t| t.k.clone()),
            chi_std: self.theory.as_ref().map(|t| t.chi.clone()),
            ..AutobkParams::default()
        }
    }

    /// Forward-FT parameters for the χ(R) plot.
    pub fn ft_params(&self) -> FtParams {
        FtParams {
            kmin: self.kmin,
            kmax: self.kmax,
            kweight: self.kweight,
            dk: self.dk,
            window: self.window,
            ..FtParams::default()
        }
    }

    /// Render the "Autobk parameters" grid plus the loading-mode and graph-type
    /// selectors, mirroring the original 3-column layout. The returned
    /// [`ControlsChange`] reports whether a reduction parameter's edit just
    /// finished (`refit`, re-run Autobk) and/or only the graph type changed
    /// (`replot`, a cheap re-render).
    pub fn controls(&mut self, ui: &mut egui::Ui) -> ControlsChange {
        let mut change = ControlsChange::default();
        ui.group(|ui| {
            ui.strong("Autobk parameters");
            ui.horizontal_top(|ui| {
                // column 1: edge energy + pre-edge / normalization ranges
                egui::Grid::new("autobk_col1")
                    .num_columns(2)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Eo");
                        ui.horizontal(|ui| {
                            change.refit |= ui.checkbox(&mut self.e0_auto, "auto").changed();
                            let r = ui.add_enabled(
                                !self.e0_auto,
                                egui::DragValue::new(&mut self.e0).speed(0.1).suffix(" eV"),
                            );
                            change.refit |= r.drag_stopped() || r.lost_focus();
                        });
                        ui.end_row();

                        ui.label("Rbkg");
                        let r = ui.add(
                            egui::DragValue::new(&mut self.rbkg)
                                .speed(0.01)
                                .range(0.2..=2.5)
                                .suffix(" Å"),
                        );
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();

                        ui.label("ranges");
                        change.refit |= ui
                            .checkbox(&mut self.pre_norm_auto, "auto pre/norm")
                            .changed();
                        ui.end_row();

                        let manual = !self.pre_norm_auto;
                        for (label, value) in [
                            ("Pre1", &mut self.pre1),
                            ("Pre2", &mut self.pre2),
                            ("Nor1", &mut self.norm1),
                            ("Nor2", &mut self.norm2),
                        ] {
                            ui.label(label);
                            let r = ui.add_enabled(
                                manual,
                                egui::DragValue::new(value).speed(1.0).suffix(" eV"),
                            );
                            change.refit |= r.drag_stopped() || r.lost_focus();
                            ui.end_row();
                        }

                        ui.label("Nnorm");
                        let r = ui
                            .add_enabled(manual, egui::DragValue::new(&mut self.nnorm).range(0..=3))
                            .on_hover_text(
                                "post-edge normalization order (0=const … 3=cubic); \
                                 XAFSView uses 2",
                            );
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();
                    });

                // A `ui.separator()` here is a vertical divider, and inside
                // `horizontal_top` (whose initial height is the full available
                // height, unlike `ui.horizontal`'s one-row height) it grabs the
                // whole panel height — inflating this group so the Autobk action
                // buttons below it are pushed past the scroll fold. Plain spacing
                // keeps the columns at their natural (content) height.
                ui.add_space(16.0);
                // column 2: k-range, FT window, spline clamps
                egui::Grid::new("autobk_col2")
                    .num_columns(2)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("kmin");
                        let r = ui.add(
                            egui::DragValue::new(&mut self.kmin)
                                .speed(0.1)
                                .range(0.0..=6.0),
                        );
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();
                        ui.label("kmax");
                        let r = ui.add(
                            egui::DragValue::new(&mut self.kmax)
                                .speed(0.1)
                                .range(5.0..=20.0),
                        );
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();
                        ui.label("dk");
                        let r = ui.add(
                            egui::DragValue::new(&mut self.dk)
                                .speed(0.1)
                                .range(0.0..=3.0),
                        );
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();
                        ui.label("kweight");
                        let r = ui.add(egui::DragValue::new(&mut self.kweight).range(0..=3));
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();
                        ui.label("Window");
                        egui::ComboBox::from_id_salt("autobk_window")
                            .selected_text(window_name(self.window))
                            .show_ui(ui, |ui| {
                                for w in [
                                    Window::Hanning,
                                    Window::Kaiser,
                                    Window::Parzen,
                                    Window::Welch,
                                    Window::Sine,
                                    Window::Gaussian,
                                ] {
                                    if ui
                                        .selectable_value(&mut self.window, w, window_name(w))
                                        .changed()
                                    {
                                        change.refit = true;
                                    }
                                }
                            });
                        ui.end_row();
                        ui.label("clamp lo");
                        let r = ui.add(
                            egui::DragValue::new(&mut self.clamp_lo)
                                .speed(0.5)
                                .range(0.0..=50.0),
                        );
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();
                        ui.label("clamp hi");
                        let r = ui.add(
                            egui::DragValue::new(&mut self.clamp_hi)
                                .speed(0.5)
                                .range(0.0..=50.0),
                        );
                        change.refit |= r.drag_stopped() || r.lost_focus();
                        ui.end_row();
                    });

                ui.add_space(16.0);
                // column 3: file-loading mode + graph type
                egui::Grid::new("autobk_col3")
                    .num_columns(2)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Loading");
                        egui::ComboBox::from_id_salt("autobk_loading")
                            .selected_text(self.loading.label())
                            .show_ui(ui, |ui| {
                                for l in LoadingType::ALL {
                                    ui.selectable_value(&mut self.loading, l, l.label());
                                }
                            });
                        ui.end_row();
                        ui.label("Graph");
                        egui::ComboBox::from_id_salt("autobk_graph")
                            .selected_text(self.graph.label())
                            .show_ui(ui, |ui| {
                                for g in GraphType::ALL {
                                    if ui.selectable_value(&mut self.graph, g, g.label()).clicked()
                                    {
                                        change.replot = true;
                                    }
                                }
                            });
                        ui.end_row();
                    });
            });
        });
        change
    }
}

/// What [`ReductionUi::controls`] detected on a frame: a finished parameter edit
/// (re-run Autobk) and/or a graph-type switch (cheap replot, no refit).
#[derive(Default, Clone, Copy)]
pub struct ControlsChange {
    /// A reduction parameter's edit finished this frame — the drag was released
    /// or a typed value committed — so Autobk should re-run with the new value.
    pub refit: bool,
    /// Only the graph type changed → re-render the current data, no refit.
    pub replot: bool,
}

/// Display name for a window type.
fn window_name(w: Window) -> &'static str {
    match w {
        Window::Hanning => "Hanning",
        Window::Fha => "Flat-Hanning",
        Window::Parzen => "Parzen",
        Window::Welch => "Welch",
        Window::Kaiser => "Kaiser",
        Window::Bes => "Kaiser (bes)",
        Window::Sine => "Sine",
        Window::Gaussian => "Gaussian",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Graph-type selector must mirror the original XAFSView list: the nine
    /// entries, in order, with the original radio labels — every label distinct
    /// and non-empty (drives the combo box via `GraphType::ALL`).
    #[test]
    fn graph_types_mirror_the_original_nine() {
        let labels: Vec<&str> = GraphType::ALL.iter().map(|g| g.label()).collect();
        assert_eq!(
            labels,
            [
                "XMU + BKG",
                "*K.CHI",
                "*R.CHI",
                "XMU' + XMU''",
                "BKG(E)'",
                "BKG(k)'",
                "XMU' + BKG(E)'",
                "BKG(E)''",
                "Normalize",
            ]
        );
        assert!(labels.iter().all(|l| !l.is_empty()));
        let mut uniq = labels.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), labels.len(), "labels must be distinct");
    }
}
