//! Column → role mapping for the Autobk tab's "Open file → Calc XMU" flow.
//!
//! After a [`ColumnFile`] is read, [`ImportState`] holds the user's choice of
//! energy column, measurement mode, and monitor columns (seeded from the file's
//! own [`RoleGuess`](feffit::xasdata::RoleGuess)). Its [`ui`](ImportState::ui) renders
//! the chooser and returns [`ImportAction::CalcXmu`] when the user is ready to
//! build `mu(E)`; [`to_spec`](ImportState::to_spec) turns the choices into a
//! [`MuSpec`] for [`feffit::xasdata::build_mu`].

use eframe::egui;
use feffit::xasdata::{ColumnFile, MuSpec};

/// The measurement geometry the user selects.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImportMode {
    /// `mu = ln(I0 / It)`.
    Transmission,
    /// `mu = Σ(channels) / I0`.
    Fluorescence,
    /// `mu = ln(It / Iref)`.
    Reference,
    /// A precomputed mu column, used as-is.
    Raw,
}

/// What the chooser is asking the app to do.
pub enum ImportAction {
    /// Build `mu(E)` from the current selection.
    CalcXmu,
}

/// The loaded file plus the in-progress column-role selection.
pub struct ImportState {
    /// The parsed column file.
    pub file: ColumnFile,
    /// Selected measurement mode.
    pub mode: ImportMode,
    /// Energy column index.
    pub energy: usize,
    /// I0 column index.
    pub i0: usize,
    /// It column index.
    pub it: usize,
    /// Iref column index.
    pub iref: usize,
    /// Precomputed-mu column index (Raw mode).
    pub mu_col: usize,
    /// Per-column flag: include this column in the fluorescence sum.
    pub channels: Vec<bool>,
    /// "Output file numbering": when batching several files into μ(E), append a
    /// sequential number to each written `.xmu` so the outputs stay distinct.
    pub number_outputs: bool,
}

impl ImportState {
    /// Seed a chooser from a freshly-read file, guessing roles from its labels.
    pub fn new(file: ColumnFile) -> Self {
        let roles = file.guess_roles();
        let ncols = file.ncols().max(1);
        let mut channels = vec![false; ncols];
        if let Some(f) = roles.iflu {
            channels[f] = true;
        }
        // Default to Raw when the file already carries a mu column and no
        // transmission monitor; otherwise transmission.
        let mode = if roles.mu.is_some() && roles.it.is_none() {
            ImportMode::Raw
        } else {
            ImportMode::Transmission
        };
        // When the labels don't name a role, seed the canonical transmission
        // layout — Energy=col1, I0=col2, It=col3 — clamped to the file's width,
        // instead of collapsing every role onto column 1.
        let last = ncols - 1;
        Self {
            mode,
            energy: roles.energy.unwrap_or(0),
            i0: roles.i0.unwrap_or(1).min(last),
            it: roles.it.or(roles.mu).unwrap_or(2).min(last),
            iref: roles.iref.unwrap_or(0),
            mu_col: roles.mu.unwrap_or(0),
            channels,
            number_outputs: false,
            file,
        }
    }

    /// Turn the current selection into a [`MuSpec`].
    pub fn to_spec(&self) -> MuSpec {
        match self.mode {
            ImportMode::Transmission => MuSpec::Transmission {
                energy: self.energy,
                i0: self.i0,
                it: self.it,
            },
            ImportMode::Fluorescence => MuSpec::Fluorescence {
                energy: self.energy,
                i0: self.i0,
                channels: self
                    .channels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &on)| on.then_some(i))
                    .collect(),
            },
            ImportMode::Reference => MuSpec::Reference {
                energy: self.energy,
                it: self.it,
                iref: self.iref,
            },
            ImportMode::Raw => MuSpec::Raw {
                energy: self.energy,
                mu: self.mu_col,
            },
        }
    }

    /// Render the chooser; returns an action when the user clicks "Calc XMU".
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<ImportAction> {
        let labels = self.file.labels.clone();

        if let Some(p) = self.file.path.as_ref() {
            ui.label(format!("File: {}", p.display()));
        }
        // The original "Change reading format" readout: header-line count and
        // data-point count, the two numbers XAFSView shows for the loaded file.
        ui.weak(format!(
            "{} header lines · {} columns × {} data points",
            self.file.header.len(),
            self.file.ncols(),
            self.file.nrows()
        ));

        // File preview (the original "File preview" pane): the header metadata
        // block plus the first few data rows, each line-numbered as XAFSView
        // does, so the right columns can be picked by inspecting the header.
        egui::CollapsingHeader::new("File preview")
            .default_open(true)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(150.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for (i, line) in self.file.header.iter().enumerate() {
                            ui.monospace(format!("{:>3}  {}", i + 1, line));
                        }
                        let base = self.file.header.len();
                        let ncols = self.file.ncols();
                        for r in 0..self.file.nrows().min(6) {
                            let mut row = String::new();
                            for c in 0..ncols {
                                if let Some(col) = self.file.column(c) {
                                    row.push_str(&format!("{:>12.5}", col[r]));
                                }
                            }
                            ui.monospace(format!("{:>3} {row}", base + r + 1));
                        }
                    });
            });
        ui.separator();

        ui.horizontal_wrapped(|ui| {
            ui.label("Mode:");
            ui.radio_value(&mut self.mode, ImportMode::Transmission, "Transmission");
            ui.radio_value(&mut self.mode, ImportMode::Fluorescence, "Fluorescence");
            ui.radio_value(&mut self.mode, ImportMode::Reference, "Reference");
            ui.radio_value(&mut self.mode, ImportMode::Raw, "Raw μ");
        });
        ui.add_space(4.0);

        column_combo(ui, "imp_energy", "Energy", &mut self.energy, &labels);
        match self.mode {
            ImportMode::Transmission => {
                column_combo(ui, "imp_i0", "I₀", &mut self.i0, &labels);
                column_combo(ui, "imp_it", "Iₜ", &mut self.it, &labels);
            }
            ImportMode::Fluorescence => {
                column_combo(ui, "imp_i0", "I₀", &mut self.i0, &labels);
                ui.label("Fluorescence channels (summed):");
                egui::ScrollArea::vertical()
                    .max_height(140.0)
                    .show(ui, |ui| {
                        for (i, lbl) in labels.iter().enumerate() {
                            ui.checkbox(&mut self.channels[i], lbl.as_str());
                        }
                    });
            }
            ImportMode::Reference => {
                column_combo(ui, "imp_it", "Iₜ", &mut self.it, &labels);
                column_combo(ui, "imp_iref", "I_ref", &mut self.iref, &labels);
            }
            ImportMode::Raw => {
                column_combo(ui, "imp_mu", "μ column", &mut self.mu_col, &labels);
            }
        }

        ui.separator();
        ui.checkbox(
            &mut self.number_outputs,
            "Output file numbering (batch outputs)",
        )
        .on_hover_text(
            "When making μ(E) from several files at once, append 0001, 0002, … to each \
             written .xmu so the outputs stay distinct.",
        );
        ui.weak("Calc XMU writes a .xmu next to each source file.");

        ui.separator();
        ui.button("Calc XMU")
            .clicked()
            .then_some(ImportAction::CalcXmu)
    }
}

/// A labelled combo box that selects a column index by its label.
fn column_combo(
    ui: &mut egui::Ui,
    id_salt: &str,
    label: &str,
    current: &mut usize,
    labels: &[String],
) {
    ui.horizontal(|ui| {
        ui.label(label);
        let selected = labels.get(*current).cloned().unwrap_or_default();
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                for (i, l) in labels.iter().enumerate() {
                    ui.selectable_value(current, i, l.as_str());
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transmission_file_seeds_transmission_spec() {
        // energy, i0, itrans, mutrans — both a transmission monitor and a mu
        // column exist, so transmission wins and the mu column is ignored.
        let cf = ColumnFile::from_text(
            "# Column.1: energy eV\n# Column.2: i0\n# Column.3: itrans\n# Column.4: mutrans\n\
             10 1 0.5 -0.69\n20 2 0.9 -0.80\n",
        )
        .unwrap();
        let st = ImportState::new(cf);
        assert!(matches!(st.mode, ImportMode::Transmission));
        match st.to_spec() {
            MuSpec::Transmission { energy, i0, it } => {
                assert_eq!((energy, i0, it), (0, 1, 2));
            }
            _ => panic!("expected Transmission spec"),
        }
    }

    #[test]
    fn unlabeled_file_seeds_positional_energy_i0_it() {
        // No column name matches a role keyword, so the seed falls back to the
        // canonical transmission layout: Energy=col1, I0=col2, It=col3.
        let cf = ColumnFile::from_text("# a b c\n10 1 0.5\n20 2 0.9\n").unwrap();
        let st = ImportState::new(cf);
        assert!(matches!(st.mode, ImportMode::Transmission));
        match st.to_spec() {
            MuSpec::Transmission { energy, i0, it } => {
                assert_eq!((energy, i0, it), (0, 1, 2));
            }
            _ => panic!("expected Transmission spec"),
        }
    }

    #[test]
    fn mu_only_file_seeds_raw_spec() {
        let cf = ColumnFile::from_text("# energy xmu\n10 0.1\n20 0.2\n").unwrap();
        let st = ImportState::new(cf);
        assert!(matches!(st.mode, ImportMode::Raw));
        match st.to_spec() {
            MuSpec::Raw { energy, mu } => assert_eq!((energy, mu), (0, 1)),
            _ => panic!("expected Raw spec"),
        }
    }

    #[test]
    fn fluorescence_spec_collects_checked_channels() {
        let cf = ColumnFile::from_text("# e i0 c1 c2 c3\n1 100 1 2 3\n2 200 4 5 6\n").unwrap();
        let mut st = ImportState::new(cf);
        st.mode = ImportMode::Fluorescence;
        st.i0 = 1;
        st.channels = vec![false, false, true, false, true]; // c1 and c3
        match st.to_spec() {
            MuSpec::Fluorescence {
                energy,
                i0,
                channels,
            } => {
                assert_eq!((energy, i0), (0, 1));
                assert_eq!(channels, vec![2, 4]);
            }
            _ => panic!("expected Fluorescence spec"),
        }
    }
}
