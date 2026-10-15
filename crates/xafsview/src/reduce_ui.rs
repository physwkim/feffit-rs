//! Autobk-tab reduction controls: edge/background/FT parameters laid out as the
//! original XAFSView "Autobk parameters" 3-column grid, plus the graph selector.
//! [`ReductionUi`] holds the editable values and
//! renders the parameter grid; the app turns its
//! [`pre_params`](ReductionUi::pre_params) /
//! [`autobk_params`](ReductionUi::autobk_params) / [`ft_params`](ReductionUi::ft_params)
//! into engine calls via `feffit::xasdata::reduce`, and reads [`graph`](ReductionUi::graph)
//! to decide what to plot. The Title / Data File / Theory rows and the
//! Open / Start / Edit button cluster live in the app (they need session
//! and file-dialog access).

use std::path::PathBuf;

use eframe::egui;
use feffit::xasdata::{AutobkParams, FtParams, PreEdgeParams, Window, XftrParams};

/// Which reduction stage to display on the plot.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    /// `XMU + BKG`: raw `mu(E)` with the AUTOBK background overlaid.
    MuBkg,
    /// `*K.CHI`: k-weighted EXAFS `kʷ·χ(k)`.
    KChi,
    /// `*R.CHI`: magnitude of the Fourier transform `|χ(R)|`.
    ChiR,
    /// `*Q.CHI`: Fourier-filtered EXAFS `Re[χ(q)]` from the R-windowed reverse
    /// FT, overlaid on the original `kʷ·χ(k)` for comparison.
    ChiQ,
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
    pub const ALL: [GraphType; 10] = [
        GraphType::MuBkg,
        GraphType::KChi,
        GraphType::ChiR,
        GraphType::ChiQ,
        GraphType::MuDeriv,
        GraphType::BkgEDeriv,
        GraphType::BkgKDeriv,
        GraphType::MuBkgEDeriv,
        GraphType::BkgEDeriv2,
        GraphType::Norm,
    ];

    /// Whether this graph's x axis is energy (eV) — the spaces where a plot
    /// click can seed the Edit-μ(E) energy fields.
    pub fn x_is_energy(self) -> bool {
        !matches!(
            self,
            GraphType::KChi | GraphType::ChiR | GraphType::ChiQ | GraphType::BkgKDeriv
        )
    }

    /// Short menu label (matches the original's "Graph type" radio text).
    pub fn label(self) -> &'static str {
        match self {
            GraphType::MuBkg => "XMU + BKG",
            GraphType::KChi => "*K.CHI",
            GraphType::ChiR => "*R.CHI",
            GraphType::ChiQ => "*Q.CHI",
            GraphType::MuDeriv => "XMU' + XMU''",
            GraphType::BkgEDeriv => "BKG(E)'",
            GraphType::BkgKDeriv => "BKG(k)'",
            GraphType::MuBkgEDeriv => "XMU' + BKG(E)'",
            GraphType::BkgEDeriv2 => "BKG(E)''",
            GraphType::Norm => "Normalize",
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

/// The parameter state a refit depends on, comparable so the app can skip a
/// committed edit that changed nothing — entering a field and leaving fires
/// the same commit events (`drag_stopped`/`lost_focus`) as a real edit. Built
/// by [`ReductionUi::refit_key`]; display-only state (graph type, band
/// visibility) is deliberately absent.
#[derive(PartialEq, Clone)]
pub struct RefitKey {
    e0_auto: bool,
    e0: f64,
    rbkg: f64,
    kweight: i32,
    kmin: f64,
    kmax: f64,
    dk: f64,
    window: Window,
    rmin: f64,
    rmax: f64,
    dr: f64,
    clamp_lo: f64,
    clamp_hi: f64,
    pre_norm_auto: bool,
    pre1: f64,
    pre2: f64,
    norm1: f64,
    norm2: f64,
    nnorm: usize,
    theory: Option<PathBuf>,
}

/// Editable reduction parameters plus the active graph type and the optional
/// theory standard.
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
    /// FT window function. Shared by the forward k-window and the reverse
    /// R-window taper.
    pub window: Window,
    /// Reverse-FT R-window lower bound (Å) — the back-transform peak selection
    /// for the Fourier-filtered χ(q).
    pub rmin: f64,
    /// Reverse-FT R-window upper bound (Å).
    pub rmax: f64,
    /// Reverse-FT R-window taper width (Å).
    pub dr: f64,
    /// Low-energy spline clamp weight.
    pub clamp_lo: f64,
    /// High-energy spline clamp weight.
    pub clamp_hi: f64,
    /// Active graph type.
    pub graph: GraphType,
    /// When set, draw a draggable FT-window band on the kʷ·χ(k) (kmin/kmax) and
    /// |χ(R)| (rmin/rmax) graphs so the range can be set on the graph. Off by
    /// default; the Feffit tab carries the same toggle.
    pub show_range: bool,
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
    /// Optional theoretical χ(k) standard constraining the background fit.
    pub theory: Option<TheoryStd>,
    /// The "Output file" chi base name (original Autobk "Output file" field):
    /// AUTOBK Start writes χ(k) to `<stem>k.chi` and χ(R) to `<stem>r.chi`.
    /// Auto-filled from the loaded file's stem on load; editable. Empty falls
    /// back to the group label at write time.
    pub output_file: String,
    /// The group label `output_file` was seeded for (and edited under): the
    /// single-target AUTOBK write honours the field only for that group
    /// ([`output_stem_for`](Self::output_stem_for)), so a stem seeded from the
    /// last-loaded scan can never name another scan's χ files.
    pub output_file_for: Option<String>,
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
            rmin: 1.0,
            rmax: 3.0,
            dr: 0.2,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
            graph: GraphType::MuBkg,
            show_range: false,
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
            theory: None,
            output_file: String::new(),
            output_file_for: None,
        }
    }
}

impl ReductionUi {
    /// Point the "Output file" field at `label`'s scan (`<label>.chi`) unless
    /// it already belongs to that group — an edit made under it is kept — and
    /// blank it when there is no group (`None`). Called on load and every
    /// frame for the current group, so the field follows the group switch
    /// instead of the last-loaded file.
    pub fn seed_output_file(&mut self, label: Option<&str>) {
        match label {
            Some(label) if self.output_file_for.as_deref() != Some(label) => {
                self.output_file = format!("{label}.chi");
                self.output_file_for = Some(label.to_owned());
            }
            Some(_) => {}
            None => {
                self.output_file.clear();
                self.output_file_for = None;
            }
        }
    }

    /// The χ-file stem for `label`'s single-target AUTOBK outputs: the field's
    /// stem when the field belongs to that group and is not blank, else the
    /// label itself (a batch always uses the labels).
    pub fn output_stem_for(&self, label: &str) -> String {
        if self.output_file_for.as_deref() == Some(label)
            && let Some(stem) = std::path::Path::new(self.output_file.trim())
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .filter(|s| !s.is_empty())
        {
            return stem;
        }
        label.to_owned()
    }

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

    /// The comparable refit-relevant parameter state (see [`RefitKey`]).
    pub fn refit_key(&self) -> RefitKey {
        RefitKey {
            e0_auto: self.e0_auto,
            e0: self.e0,
            rbkg: self.rbkg,
            kweight: self.kweight,
            kmin: self.kmin,
            kmax: self.kmax,
            dk: self.dk,
            window: self.window,
            rmin: self.rmin,
            rmax: self.rmax,
            dr: self.dr,
            clamp_lo: self.clamp_lo,
            clamp_hi: self.clamp_hi,
            pre_norm_auto: self.pre_norm_auto,
            pre1: self.pre1,
            pre2: self.pre2,
            norm1: self.norm1,
            norm2: self.norm2,
            nnorm: self.nnorm,
            theory: self.theory.as_ref().map(|t| t.path.clone()),
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

    /// Reverse-FT parameters for the Fourier-filtered χ(q) plot. `qmax_out`
    /// follows larch's `kmax + 2`; the R-window shares the FT window function.
    pub fn xftr_params(&self) -> XftrParams {
        XftrParams {
            rmin: self.rmin,
            rmax: self.rmax,
            dr: self.dr,
            window: self.window,
            qmax_out: self.kmax + 2.0,
            ..XftrParams::default()
        }
    }

    /// Leaving "auto": start the manual E0 at the value the reduction found for
    /// the current group (`detected_e0`), not at whatever the field last held
    /// (0.0 on a fresh session), so the next Run does not silently fit with
    /// E0 = 0. A group without a detected E0 keeps the field as is.
    fn seed_manual_e0(&mut self, detected_e0: Option<f64>) {
        if !self.e0_auto
            && let Some(e0) = detected_e0
        {
            self.e0 = e0;
        }
    }

    /// Render the "Autobk parameters" grid plus the graph-type selector,
    /// mirroring the original 3-column layout. `detected_e0` is the current
    /// group's E0 as found by the reduction: shown in the disabled field while
    /// "auto" is on and copied into the manual field when "auto" is unchecked.
    /// The returned [`ControlsChange`] reports whether a reduction parameter's
    /// edit just finished (`refit`, re-run Autobk) and/or only the graph type
    /// changed (`replot`, a cheap re-render).
    pub fn controls(&mut self, ui: &mut egui::Ui, detected_e0: Option<f64>) -> ControlsChange {
        let mut change = ControlsChange::default();
        rsplot::widget::form::group(ui, |ui| {
            ui.strong("Autobk parameters");
            ui.horizontal_top(|ui| {
                // column 1: edge energy + pre-edge / normalization ranges
                egui::Grid::new("autobk_col1")
                    .num_columns(2)
                    .min_col_width(crate::widgets::NUM_FIELD_W)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        crate::widgets::uniform_num_fields(ui);
                        ui.label("Eo");
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut self.e0_auto, "auto").changed() {
                                self.seed_manual_e0(detected_e0);
                                change.refit = true;
                            }
                            if self.e0_auto {
                                // Read-only: the value the reduction found.
                                let mut shown = detected_e0.unwrap_or(self.e0);
                                ui.add_enabled(
                                    false,
                                    egui::DragValue::new(&mut shown).speed(0.1).suffix(" eV"),
                                );
                            } else {
                                let r = ui.add(
                                    egui::DragValue::new(&mut self.e0).speed(0.1).suffix(" eV"),
                                );
                                change.refit |= r.drag_stopped() || r.lost_focus();
                            }
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
                    .min_col_width(crate::widgets::NUM_FIELD_W)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        crate::widgets::uniform_num_fields(ui);
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
                                    Window::Fha,
                                    Window::Kaiser,
                                    Window::Bes,
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
                // column 3: the reverse-FT R-window. (The graph-type / k-R-range
                // display controls moved to the Autobk tab's "Graph" box, mirroring
                // the Feffit panel's grouping.)
                egui::Grid::new("autobk_col3")
                    .num_columns(2)
                    .min_col_width(crate::widgets::NUM_FIELD_W)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        crate::widgets::uniform_num_fields(ui);
                        // Reverse-FT R-window (the back-transform peak selection
                        // for the *Q.CHI graph; also the band draggable on *R.CHI).
                        for (label, value, speed, range) in [
                            ("rmin", &mut self.rmin, 0.05, 0.0..=8.0),
                            ("rmax", &mut self.rmax, 0.05, 0.0..=8.0),
                            ("dr", &mut self.dr, 0.05, 0.0..=2.0),
                        ] {
                            ui.label(label);
                            let r = ui.add(
                                egui::DragValue::new(value)
                                    .speed(speed)
                                    .range(range)
                                    .suffix(" Å"),
                            );
                            change.refit |= r.drag_stopped() || r.lost_focus();
                            ui.end_row();
                        }
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

    /// Unchecking "auto" seeds the manual E0 from the detected value; checking
    /// it back, or a group without a detected E0, leaves the field alone.
    #[test]
    fn manual_e0_seeds_from_the_detected_value() {
        let mut r = ReductionUi::default();
        assert!(r.e0_auto);
        r.e0_auto = false;
        r.seed_manual_e0(Some(6551.2));
        assert_eq!(r.e0, 6551.2);
        r.seed_manual_e0(None);
        assert_eq!(r.e0, 6551.2);
        r.e0_auto = true;
        r.seed_manual_e0(Some(11564.3));
        assert_eq!(r.e0, 6551.2);
        assert_eq!(r.autobk_params().ek0, None);
        r.e0_auto = false;
        assert_eq!(r.autobk_params().ek0, Some(6551.2));
    }

    /// The "Output file" stem belongs to the group it was seeded for: it
    /// follows a group switch, keeps an edit made under the same group, and
    /// never names another group's χ files (the single-target write falls
    /// back to that group's label).
    #[test]
    fn output_file_follows_the_current_group() {
        let mut r = ReductionUi::default();
        assert_eq!(r.output_stem_for("Mn_000"), "Mn_000");
        r.seed_output_file(Some("Pt_000"));
        assert_eq!(r.output_file, "Pt_000.chi");
        assert_eq!(r.output_stem_for("Pt_000"), "Pt_000");
        assert_eq!(r.output_stem_for("Mn_000"), "Mn_000");
        r.output_file = "pt_foil_run2.chi".into();
        r.seed_output_file(Some("Pt_000"));
        assert_eq!(r.output_stem_for("Pt_000"), "pt_foil_run2");
        r.output_file = "  ".into();
        assert_eq!(r.output_stem_for("Pt_000"), "Pt_000");
        r.seed_output_file(Some("Mn_000"));
        assert_eq!(r.output_file, "Mn_000.chi");
        assert_eq!(r.output_stem_for("Mn_000"), "Mn_000");
        r.seed_output_file(None);
        assert!(r.output_file.is_empty());
        r.seed_output_file(Some("Mn_000"));
        assert_eq!(r.output_file, "Mn_000.chi");
    }

    /// The Graph-type selector must mirror the original XAFSView list: the nine
    /// entries, in order, with the original radio labels — every label distinct
    /// and non-empty (drives the combo box via `GraphType::ALL`).
    #[test]
    fn graph_types_cover_the_original_nine_plus_back_transform() {
        // The original XAFSView Autobk has nine graph types; `*Q.CHI` is an
        // addition beyond parity — the Fourier-filtered χ(q) view the original
        // lacks here, needed so the reverse-FT R-window has a visible result.
        let labels: Vec<&str> = GraphType::ALL.iter().map(|g| g.label()).collect();
        assert_eq!(
            labels,
            [
                "XMU + BKG",
                "*K.CHI",
                "*R.CHI",
                "*Q.CHI",
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
