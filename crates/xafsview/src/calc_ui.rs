//! Phase-8 auxiliary calculators, all backed by the pure-Rust [`xraydb`] atomic
//! database (the Rust mirror of Python `xraydb`):
//!
//! - [`PeriodicTableWindow`] — a clickable periodic table + per-element atom data
//!   (mass, density, absorption edges, emission lines, core-hole widths).
//! - [`IonChamberWindow`] — ion-chamber / gas-absorption fluxes
//!   ([`XrayDb::ionchamber_fluxes`]).
//! - [`PowderWindow`] — sample powder weight for a target absorbance from a
//!   chemical formula ([`XrayDb::material_mu`]).
//!
//! These windows are pure read-outs (no plot, no session mutation), so they take
//! no [`siplot`](siplot) handle.

use eframe::egui;
use egui::Color32;
use xraydb::{CrossSectionKind, XrayDb};

const RED: Color32 = Color32::from_rgb(0xd6, 0x27, 0x28);

/// Element symbols indexed by `Z - 1` (Z = 1..=118).
#[rustfmt::skip]
const SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
    "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
    "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

/// (row, col), both 1-based, of element `z` in the standard 18-column periodic
/// table. Lanthanides land on row 9 and actinides on row 10 (cols 3..=17).
fn pt_cell(z: u16) -> (usize, usize) {
    match z {
        1 => (1, 1),
        2 => (1, 18),
        3..=4 => (2, (z - 2) as usize),
        5..=10 => (2, (z + 8) as usize),
        11..=12 => (3, (z - 10) as usize),
        13..=18 => (3, z as usize),
        19..=36 => (4, (z - 18) as usize),
        37..=54 => (5, (z - 36) as usize),
        55..=56 => (6, (z - 54) as usize),
        57..=71 => (9, (z - 57 + 3) as usize),
        72..=86 => (6, (z - 68) as usize),
        87..=88 => (7, (z - 86) as usize),
        89..=103 => (10, (z - 89 + 3) as usize),
        104..=118 => (7, (z - 100) as usize),
        _ => (0, 0),
    }
}

/// The clickable periodic table plus an atom-data panel for the selected element.
pub struct PeriodicTableWindow {
    pub open: bool,
    db: XrayDb,
    /// Atomic number of the selected element (1-based), or `None`.
    selected: Option<u16>,
    /// `grid[row-1][col-1]` = `Some(Z)` if an element sits in that cell.
    grid: [[Option<u16>; 18]; 10],
}

impl Default for PeriodicTableWindow {
    fn default() -> Self {
        let mut grid = [[None; 18]; 10];
        for (i, _) in SYMBOLS.iter().enumerate() {
            let z = (i + 1) as u16;
            let (r, c) = pt_cell(z);
            if r >= 1 && c >= 1 {
                grid[r - 1][c - 1] = Some(z);
            }
        }
        Self {
            open: false,
            db: XrayDb::new(),
            selected: None,
            grid,
        }
    }
}

impl PeriodicTableWindow {
    /// Render the window.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        let mut open = self.open;
        egui::Window::new("Periodic table — atom data")
            .open(&mut open)
            .resizable(true)
            .default_width(720.0)
            .show(ctx, |ui| {
                self.table(ui);
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(280.0)
                    .show(ui, |ui| self.atom_data(ui));
            });
        self.open = open;
    }

    fn table(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("periodic_table")
            .spacing([2.0, 2.0])
            .min_col_width(26.0)
            .show(ui, |ui| {
                for row in 0..10 {
                    // blank spacer row between the main body and the f-block.
                    for col in 0..18 {
                        match self.grid[row][col] {
                            Some(z) => {
                                let sym = SYMBOLS[(z - 1) as usize];
                                let selected = self.selected == Some(z);
                                let btn = egui::Button::new(sym)
                                    .min_size(egui::vec2(26.0, 22.0))
                                    .selected(selected);
                                if ui.add(btn).on_hover_text(format!("Z = {z}")).clicked() {
                                    self.selected = Some(z);
                                }
                            }
                            None => {
                                ui.allocate_space(egui::vec2(26.0, 22.0));
                            }
                        }
                    }
                    ui.end_row();
                }
            });
    }

    fn atom_data(&mut self, ui: &mut egui::Ui) {
        let Some(z) = self.selected else {
            ui.weak("Click an element to see its atom data.");
            return;
        };
        let zs = z.to_string();
        let sym = SYMBOLS[(z - 1) as usize];
        let name = self.db.atomic_name(&zs).unwrap_or_default();
        let mass = self.db.molar_mass(&zs).unwrap_or(f64::NAN);
        let dens = self.db.density(&zs).unwrap_or(f64::NAN);
        ui.heading(format!("{sym} — {name}  (Z = {z})"));
        ui.monospace(format!("molar mass : {mass:.4} g/mol"));
        ui.monospace(format!("density    : {dens:.4} g/cm³"));

        ui.separator();
        ui.strong("Absorption edges");
        match self.db.xray_edges(&zs) {
            Ok(map) => {
                let mut edges: Vec<_> = map.into_iter().collect();
                edges.sort_by(|a, b| b.1.energy.total_cmp(&a.1.energy));
                egui::Grid::new("atom_edges").striped(true).show(ui, |ui| {
                    ui.strong("edge");
                    ui.strong("energy (eV)");
                    ui.strong("fluor. yield");
                    ui.strong("jump");
                    ui.end_row();
                    for (label, e) in edges {
                        ui.monospace(label);
                        ui.monospace(format!("{:.2}", e.energy));
                        ui.monospace(format!("{:.4}", e.fluorescence_yield));
                        ui.monospace(format!("{:.3}", e.jump_ratio));
                        ui.end_row();
                    }
                });
            }
            Err(e) => {
                ui.colored_label(RED, format!("no edge data: {e}"));
            }
        }

        ui.separator();
        ui.strong("Strong emission lines");
        if let Ok(map) = self.db.xray_lines(&zs, None, None) {
            let mut lines: Vec<_> = map
                .into_iter()
                .filter(|(_, l)| l.intensity >= 0.01)
                .collect();
            lines.sort_by(|a, b| b.1.intensity.total_cmp(&a.1.intensity));
            lines.truncate(8);
            egui::Grid::new("atom_lines").striped(true).show(ui, |ui| {
                ui.strong("line");
                ui.strong("energy (eV)");
                ui.strong("rel. int.");
                ui.end_row();
                for (label, l) in lines {
                    ui.monospace(label);
                    ui.monospace(format!("{:.2}", l.energy));
                    ui.monospace(format!("{:.4}", l.intensity));
                    ui.end_row();
                }
            });
        }
    }
}

/// One selectable fill gas with its mixture fraction.
struct GasRow {
    name: &'static str,
    on: bool,
    frac: f64,
}

/// Ion-chamber / gas-absorption flux calculator.
pub struct IonChamberWindow {
    pub open: bool,
    db: XrayDb,
    gases: Vec<GasRow>,
    energy: f64,
    length_cm: f64,
    volts: f64,
    sensitivity: f64,
    with_compton: bool,
    both_carriers: bool,
    result: Option<Result<xraydb::IonChamberFluxes, String>>,
}

impl Default for IonChamberWindow {
    fn default() -> Self {
        Self {
            open: false,
            db: XrayDb::new(),
            gases: vec![
                GasRow {
                    name: "helium",
                    on: false,
                    frac: 1.0,
                },
                GasRow {
                    name: "nitrogen",
                    on: true,
                    frac: 1.0,
                },
                GasRow {
                    name: "argon",
                    on: false,
                    frac: 1.0,
                },
                GasRow {
                    name: "krypton",
                    on: false,
                    frac: 1.0,
                },
                GasRow {
                    name: "xenon",
                    on: false,
                    frac: 1.0,
                },
            ],
            energy: 10000.0,
            length_cm: 15.0,
            volts: 1.0,
            sensitivity: 1e-6,
            with_compton: true,
            both_carriers: true,
            result: None,
        }
    }
}

impl IonChamberWindow {
    /// Render the window.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        let mut open = self.open;
        egui::Window::new("Ion chamber / gas absorption")
            .open(&mut open)
            .resizable(true)
            .default_width(420.0)
            .show(ctx, |ui| self.body(ui));
        self.open = open;
    }

    fn body(&mut self, ui: &mut egui::Ui) {
        ui.strong("Fill gases (mixture fractions)");
        egui::Grid::new("ic_gases").show(ui, |ui| {
            for g in &mut self.gases {
                ui.checkbox(&mut g.on, g.name);
                ui.add_enabled(
                    g.on,
                    egui::DragValue::new(&mut g.frac)
                        .speed(0.05)
                        .range(0.0..=1.0),
                );
                ui.end_row();
            }
        });

        ui.separator();
        egui::Grid::new("ic_params").num_columns(2).show(ui, |ui| {
            ui.label("energy (eV)");
            ui.add(
                egui::DragValue::new(&mut self.energy)
                    .speed(10.0)
                    .range(100.0..=800_000.0),
            );
            ui.end_row();
            ui.label("length (cm)");
            ui.add(
                egui::DragValue::new(&mut self.length_cm)
                    .speed(0.5)
                    .range(0.01..=200.0),
            );
            ui.end_row();
            ui.label("voltage (V)");
            ui.add(egui::DragValue::new(&mut self.volts).speed(0.05));
            ui.end_row();
            ui.label("sensitivity (A/V)");
            ui.add(
                egui::DragValue::new(&mut self.sensitivity)
                    .speed(1e-8)
                    .range(1e-12..=1.0)
                    .custom_formatter(|n, _| format!("{n:.2e}")),
            );
            ui.end_row();
        });
        ui.checkbox(&mut self.with_compton, "include Compton electrons");
        ui.checkbox(&mut self.both_carriers, "count both carriers");

        ui.separator();
        if ui.button("Compute fluxes").clicked() {
            self.compute();
        }
        match &self.result {
            Some(Ok(f)) => {
                egui::Grid::new("ic_out").striped(true).show(ui, |ui| {
                    ui.label("incident");
                    ui.monospace(format!("{:.4e} Hz", f.incident));
                    ui.end_row();
                    ui.label("transmitted");
                    ui.monospace(format!("{:.4e} Hz", f.transmitted));
                    ui.end_row();
                    ui.label("photo");
                    ui.monospace(format!("{:.4e} Hz", f.photo));
                    ui.end_row();
                    ui.label("incoherent");
                    ui.monospace(format!("{:.4e} Hz", f.incoherent));
                    ui.end_row();
                    ui.label("coherent");
                    ui.monospace(format!("{:.4e} Hz", f.coherent));
                    ui.end_row();
                });
            }
            Some(Err(e)) => {
                ui.colored_label(RED, e);
            }
            None => {
                ui.weak("Set the gas mix and parameters, then compute.");
            }
        }
    }

    fn compute(&mut self) {
        let gases: Vec<(&str, f64)> = self
            .gases
            .iter()
            .filter(|g| g.on && g.frac > 0.0)
            .map(|g| (g.name, g.frac))
            .collect();
        if gases.is_empty() {
            self.result = Some(Err("select at least one gas".to_owned()));
            return;
        }
        self.result = Some(
            self.db
                .ionchamber_fluxes(
                    &gases,
                    self.volts,
                    self.length_cm,
                    self.energy,
                    self.sensitivity,
                    self.with_compton,
                    self.both_carriers,
                )
                .map_err(|e| e.to_string()),
        );
    }
}

/// Sample powder-weight calculator: grams of a compound to reach a target
/// absorbance over a given area.
pub struct PowderWindow {
    pub open: bool,
    db: XrayDb,
    formula: String,
    energy: f64,
    area_cm2: f64,
    absorbance: f64,
    result: Option<Result<(f64, f64), String>>,
}

impl Default for PowderWindow {
    fn default() -> Self {
        Self {
            open: false,
            db: XrayDb::new(),
            formula: "Fe2O3".to_owned(),
            energy: 7200.0,
            area_cm2: 1.327, // 13 mm diameter pellet
            absorbance: 2.5,
            result: None,
        }
    }
}

impl PowderWindow {
    /// Render the window.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        let mut open = self.open;
        egui::Window::new("Powder weight")
            .open(&mut open)
            .resizable(true)
            .default_width(380.0)
            .show(ctx, |ui| self.body(ui));
        self.open = open;
    }

    fn body(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("pw_params").num_columns(2).show(ui, |ui| {
            ui.label("formula");
            ui.text_edit_singleline(&mut self.formula);
            ui.end_row();
            ui.label("energy (eV)");
            ui.add(
                egui::DragValue::new(&mut self.energy)
                    .speed(10.0)
                    .range(100.0..=800_000.0),
            );
            ui.end_row();
            ui.label("area (cm²)");
            ui.add(
                egui::DragValue::new(&mut self.area_cm2)
                    .speed(0.05)
                    .range(0.001..=100.0),
            );
            ui.end_row();
            ui.label("target μx (absorbance)");
            ui.add(
                egui::DragValue::new(&mut self.absorbance)
                    .speed(0.05)
                    .range(0.01..=20.0),
            );
            ui.end_row();
        });
        ui.weak("μx is the total absorbance (≈2.5 for transmission samples).");

        ui.separator();
        if ui.button("Compute weight").clicked() {
            self.compute();
        }
        match &self.result {
            Some(Ok((mass_mg, mu_mass))) => {
                egui::Grid::new("pw_out").striped(true).show(ui, |ui| {
                    ui.label("μ/ρ at energy");
                    ui.monospace(format!("{mu_mass:.3} cm²/g"));
                    ui.end_row();
                    ui.label("sample mass");
                    ui.monospace(format!("{mass_mg:.3} mg"));
                    ui.end_row();
                    ui.label("areal density");
                    ui.monospace(format!("{:.4} mg/cm²", mass_mg / self.area_cm2));
                    ui.end_row();
                });
            }
            Some(Err(e)) => {
                ui.colored_label(RED, e);
            }
            None => {
                ui.weak("Enter a formula and compute.");
            }
        }
    }

    fn compute(&mut self) {
        self.result = Some(powder_weight(
            &self.db,
            &self.formula,
            self.energy,
            self.area_cm2,
            self.absorbance,
        ));
    }
}

/// Sample mass `(mass_mg, μ/ρ)` to reach `absorbance` over `area_cm2` for a
/// compound at `energy`. `material_mu` at unit density gives the mass
/// attenuation μ/ρ [cm²/g]; then `absorbance = (μ/ρ)·(mass/area)`.
fn powder_weight(
    db: &XrayDb,
    formula: &str,
    energy: f64,
    area_cm2: f64,
    absorbance: f64,
) -> Result<(f64, f64), String> {
    let mu_mass = match db.material_mu(formula, 1.0, &[energy], CrossSectionKind::Total) {
        Ok(v) if !v.is_empty() && v[0] > 0.0 => v[0],
        Ok(_) => return Err("zero attenuation at this energy".to_owned()),
        Err(e) => return Err(e.to_string()),
    };
    let mass_g = absorbance * area_cm2 / mu_mass;
    Ok((mass_g * 1000.0, mu_mass))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_match_xraydb() {
        // The hand-typed 118-symbol table must agree with the database for every
        // element the database covers (off-by-one / typo guard).
        let db = XrayDb::new();
        for (i, &sym) in SYMBOLS.iter().enumerate() {
            let z = (i + 1) as u16;
            if let Ok(dbsym) = db.symbol(&z.to_string()) {
                assert_eq!(sym, dbsym, "symbol mismatch at Z={z}");
            }
        }
    }

    #[test]
    fn pt_cells_unique_and_valid() {
        // Every element lands in a distinct, in-bounds periodic-table cell.
        let mut seen = std::collections::HashSet::new();
        for z in 1u16..=118 {
            let (r, c) = pt_cell(z);
            assert!(
                (1..=10).contains(&r) && (1..=18).contains(&c),
                "Z={z} → ({r},{c})"
            );
            assert!(seen.insert((r, c)), "duplicate cell ({r},{c}) at Z={z}");
        }
        // Spot-check the corners and f-block placement.
        assert_eq!(pt_cell(1), (1, 1)); // H
        assert_eq!(pt_cell(2), (1, 18)); // He
        assert_eq!(pt_cell(26), (4, 8)); // Fe
        assert_eq!(pt_cell(57), (9, 3)); // La (lanthanide row)
        assert_eq!(pt_cell(89), (10, 3)); // Ac (actinide row)
        assert_eq!(pt_cell(118), (7, 18)); // Og
    }

    #[test]
    fn powder_weight_relation_holds() {
        // mass = absorbance·area/(μ/ρ): doubling the target absorbance doubles
        // the mass, and μ/ρ is positive for a real compound at an XAS energy.
        let db = XrayDb::new();
        let (m1, mu) = powder_weight(&db, "Fe2O3", 7200.0, 1.0, 1.0).unwrap();
        assert!(mu > 0.0, "μ/ρ should be positive");
        let (m2, _) = powder_weight(&db, "Fe2O3", 7200.0, 1.0, 2.0).unwrap();
        assert!(
            (m2 - 2.0 * m1).abs() < 1e-9 * m2.max(1.0),
            "mass should scale with absorbance"
        );
        // mass_mg = absorbance·area·1000/(μ/ρ)
        assert!((m1 - 1000.0 / mu).abs() < 1e-6, "mass formula");
    }
}
