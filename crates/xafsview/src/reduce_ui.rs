//! Autobk-tab reduction controls: edge/background/FT parameters and the graph
//! selector. [`ReductionUi`] holds the editable values and renders the control
//! panel; the app turns its [`pre_params`](ReductionUi::pre_params) /
//! [`autobk_params`](ReductionUi::autobk_params) / [`ft_params`](ReductionUi::ft_params)
//! into engine calls via `xasdata::reduce`, and reads [`graph`](ReductionUi::graph)
//! to decide what to plot.

use eframe::egui;
use xasdata::{AutobkParams, FtParams, PreEdgeParams, Window};

/// Which reduction stage to display on the plot.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    /// Raw `mu(E)` with the AUTOBK background overlaid.
    MuBkg,
    /// Edge-step normalized `mu(E)`.
    Norm,
    /// First derivative `d(mu)/dE`.
    Deriv,
    /// k-weighted EXAFS `kʷ·χ(k)`.
    KChi,
    /// Magnitude of the Fourier transform `|χ(R)|`.
    ChiR,
}

impl GraphType {
    /// All graph types in display order.
    pub const ALL: [GraphType; 5] = [
        GraphType::MuBkg,
        GraphType::Norm,
        GraphType::Deriv,
        GraphType::KChi,
        GraphType::ChiR,
    ];

    /// Short button label.
    pub fn label(self) -> &'static str {
        match self {
            GraphType::MuBkg => "μ + bkg",
            GraphType::Norm => "norm",
            GraphType::Deriv => "deriv",
            GraphType::KChi => "kʷ·χ(k)",
            GraphType::ChiR => "χ(R)",
        }
    }
}

/// What the controls are asking the app to do this frame.
pub enum ReductionAction {
    /// Re-run normalize → autobk → xftf for the current group.
    Run,
    /// Re-draw the current group with the (possibly changed) graph type.
    Replot,
}

/// Editable reduction parameters plus the active graph type.
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
    /// High-energy spline clamp weight.
    pub clamp_hi: f64,
    /// Active graph type.
    pub graph: GraphType,
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
            clamp_hi: 1.0,
            graph: GraphType::MuBkg,
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
        p
    }

    /// AUTOBK parameters for this selection.
    pub fn autobk_params(&self) -> AutobkParams {
        AutobkParams {
            rbkg: self.rbkg,
            ek0: (!self.e0_auto).then_some(self.e0),
            kmin: self.kmin,
            kmax: Some(self.kmax),
            kweight: self.kweight,
            dk: self.dk,
            win: self.window,
            clamp_hi: self.clamp_hi,
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

    /// Render the control panel. Returns [`ReductionAction::Run`] when the run
    /// button is pressed and [`ReductionAction::Replot`] when only the graph
    /// type changed.
    pub fn controls(&mut self, ui: &mut egui::Ui) -> Option<ReductionAction> {
        let mut action = None;

        ui.heading("Reduction");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.e0_auto, "auto E₀");
            ui.add_enabled(
                !self.e0_auto,
                egui::DragValue::new(&mut self.e0).speed(0.1).suffix(" eV"),
            );
        });
        ui.add(egui::Slider::new(&mut self.rbkg, 0.2..=2.5).text("Rbkg (Å)"));
        ui.horizontal(|ui| {
            ui.label("k-weight");
            ui.add(egui::DragValue::new(&mut self.kweight).range(0..=3));
        });
        ui.add(egui::Slider::new(&mut self.kmin, 0.0..=6.0).text("k min"));
        ui.add(egui::Slider::new(&mut self.kmax, 5.0..=20.0).text("k max"));
        ui.add(egui::Slider::new(&mut self.dk, 0.0..=3.0).text("dk"));
        egui::ComboBox::from_label("window")
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
                    ui.selectable_value(&mut self.window, w, window_name(w));
                }
            });
        ui.add(egui::Slider::new(&mut self.clamp_hi, 0.0..=50.0).text("clamp hi"));

        ui.add_space(4.0);
        if ui.button("Run reduction").clicked() {
            action = Some(ReductionAction::Run);
        }

        ui.separator();
        ui.label("Graph:");
        ui.horizontal_wrapped(|ui| {
            for g in GraphType::ALL {
                if ui.selectable_value(&mut self.graph, g, g.label()).clicked() {
                    action.get_or_insert(ReductionAction::Replot);
                }
            }
        });

        action
    }
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
