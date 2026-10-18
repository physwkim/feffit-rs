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
//! no [`rsplot`] handle.

use eframe::egui;
use xraydb::{CrossSectionKind, XrayDb};

use crate::plot::RED;

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
        crate::window::detached(
            ctx,
            "periodic_table",
            "Periodic table — atom data",
            &mut open,
            [720.0, 600.0],
            |ui| {
                self.table(ui);
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(280.0)
                    .show(ui, |ui| self.atom_data(ui));
            },
        );
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
        let d = self.atom_data_of(z);
        ui.heading(format!("{} — {}  (Z = {z})", d.symbol, d.name));
        ui.monospace(format!("molar mass : {:.4} g/mol", d.mass));
        ui.monospace(format!("density    : {:.4} g/cm³", d.density));

        ui.separator();
        ui.strong("Absorption edges");
        match &d.edges {
            Ok(edges) => {
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
        egui::Grid::new("atom_lines").striped(true).show(ui, |ui| {
            ui.strong("line");
            ui.strong("energy (eV)");
            ui.strong("rel. int.");
            ui.end_row();
            for (label, l) in &d.lines {
                ui.monospace(label);
                ui.monospace(format!("{:.2}", l.energy));
                ui.monospace(format!("{:.4}", l.intensity));
                ui.end_row();
            }
        });
    }

    /// The atom-data panel's content for element `z`: identity, mass, density,
    /// the absorption edges (highest first) and the strong emission lines
    /// (intensity ≥ 0.01, strongest first, at most eight) — one fetch shared by
    /// the panel and MCP `periodic_table`.
    fn atom_data_of(&self, z: u16) -> AtomData {
        let zs = z.to_string();
        let edges = self.db.xray_edges(&zs).map(|map| {
            let mut edges: Vec<_> = map.into_iter().collect();
            edges.sort_by(|a, b| b.1.energy.total_cmp(&a.1.energy));
            edges
        });
        let mut lines: Vec<_> = self
            .db
            .xray_lines(&zs, None, None)
            .map(|map| {
                map.into_iter()
                    .filter(|(_, l)| l.intensity >= 0.01)
                    .collect()
            })
            .unwrap_or_default();
        lines.sort_by(|a, b| b.1.intensity.total_cmp(&a.1.intensity));
        lines.truncate(8);
        AtomData {
            symbol: SYMBOLS[(z - 1) as usize],
            name: self.db.atomic_name(&zs).unwrap_or_default().to_owned(),
            mass: self.db.molar_mass(&zs).unwrap_or(f64::NAN),
            density: self.db.density(&zs).unwrap_or(f64::NAN),
            edges: edges.map_err(|e| e.to_string()),
            lines,
        }
    }

    /// MCP `periodic_table`: click `element` (a symbol or an atomic number)
    /// when given, then the atom-data panel of the selection as text.
    pub fn apply_mcp(&mut self, element: Option<&str>) -> Result<String, String> {
        self.open = true;
        if let Some(e) = element {
            self.selected = Some(parse_element(e)?);
        }
        let Some(z) = self.selected else {
            return Ok("no element selected".to_owned());
        };
        let d = self.atom_data_of(z);
        let mut s = format!(
            "{} — {}  (Z = {z})\nmolar mass : {:.4} g/mol\ndensity    : {:.4} g/cm³\n",
            d.symbol, d.name, d.mass, d.density
        );
        s.push_str("absorption edges (edge, energy eV, fluorescence yield, jump):\n");
        match &d.edges {
            Ok(edges) => {
                for (label, e) in edges {
                    s.push_str(&format!(
                        "  {label:<4} {:>10.2} {:>8.4} {:>7.3}\n",
                        e.energy, e.fluorescence_yield, e.jump_ratio
                    ));
                }
            }
            Err(e) => s.push_str(&format!("  no edge data: {e}\n")),
        }
        s.push_str("strong emission lines (line, energy eV, relative intensity):\n");
        for (label, l) in &d.lines {
            s.push_str(&format!(
                "  {label:<6} {:>10.2} {:>8.4}\n",
                l.energy, l.intensity
            ));
        }
        Ok(s)
    }
}

/// The atom-data panel's content for one element (see
/// [`PeriodicTableWindow::atom_data_of`]).
struct AtomData {
    symbol: &'static str,
    name: String,
    mass: f64,
    density: f64,
    /// `(edge label, edge)` highest energy first, or the database's error.
    edges: Result<Vec<(String, xraydb::XrayEdge)>, String>,
    /// `(line label, line)` strongest first.
    lines: Vec<(String, xraydb::XrayLine)>,
}

/// An element given as a symbol (case-insensitive) or an atomic number.
fn parse_element(s: &str) -> Result<u16, String> {
    let t = s.trim();
    if let Ok(z) = t.parse::<u16>() {
        return if (1..=118).contains(&z) {
            Ok(z)
        } else {
            Err(format!("atomic number {z} outside 1..=118"))
        };
    }
    SYMBOLS
        .iter()
        .position(|sym| sym.eq_ignore_ascii_case(t))
        .map(|i| (i + 1) as u16)
        .ok_or_else(|| format!("unknown element {s:?}"))
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
        crate::window::detached(
            ctx,
            "ion_chamber",
            "Ion chamber / gas absorption",
            &mut open,
            [420.0, 520.0],
            |ui| self.body(ui),
        );
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
            crate::widgets::uniform_num_fields(ui);
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

    /// MCP `ion_chamber`: the gas mix and parameter fields (range-checked like
    /// the widgets), then "Compute fluxes" and the result table as text.
    pub fn apply_mcp(&mut self, a: &crate::mcp::IonChamberArgs) -> Result<String, String> {
        self.open = true;
        if let Some(mix) = &a.gases {
            for g in mix {
                if !self.gases.iter().any(|row| row.name == g.name) {
                    return Err(format!(
                        "unknown gas {:?}; one of helium, nitrogen, argon, krypton, xenon",
                        g.name
                    ));
                }
                if !(0.0..=1.0).contains(&g.frac) {
                    return Err(format!("gas {} fraction {} outside 0..=1", g.name, g.frac));
                }
            }
            for row in &mut self.gases {
                match mix.iter().find(|g| g.name == row.name) {
                    Some(g) => {
                        row.on = true;
                        row.frac = g.frac;
                    }
                    None => row.on = false,
                }
            }
        }
        if let Some(v) = check(a.energy, 100.0, 800_000.0, "energy")? {
            self.energy = v;
        }
        if let Some(v) = check(a.length_cm, 0.01, 200.0, "length_cm")? {
            self.length_cm = v;
        }
        if let Some(v) = a.volts {
            self.volts = v;
        }
        if let Some(v) = check(a.sensitivity, 1e-12, 1.0, "sensitivity")? {
            self.sensitivity = v;
        }
        if let Some(v) = a.with_compton {
            self.with_compton = v;
        }
        if let Some(v) = a.both_carriers {
            self.both_carriers = v;
        }
        self.compute();
        let gases: Vec<String> = self
            .gases
            .iter()
            .filter(|g| g.on)
            .map(|g| format!("{} {:.3}", g.name, g.frac))
            .collect();
        let setup = format!(
            "gases: {}; energy {} eV; length {} cm; {} V; sensitivity {:.2e} A/V; compton {}; both carriers {}",
            gases.join(", "),
            self.energy,
            self.length_cm,
            self.volts,
            self.sensitivity,
            self.with_compton,
            self.both_carriers
        );
        match &self.result {
            Some(Ok(f)) => Ok(format!(
                "{setup}\nincident    {:.4e} Hz\ntransmitted {:.4e} Hz\nphoto       {:.4e} Hz\nincoherent  {:.4e} Hz\ncoherent    {:.4e} Hz",
                f.incident, f.transmitted, f.photo, f.incoherent, f.coherent
            )),
            Some(Err(e)) => Err(format!("{setup}\n{e}")),
            None => Err(setup),
        }
    }
}

/// Range check for an optional MCP value, reported as the field name.
fn check<T: PartialOrd + std::fmt::Display + Copy>(
    v: Option<T>,
    lo: T,
    hi: T,
    name: &str,
) -> Result<Option<T>, String> {
    match v {
        Some(x) if x < lo || x > hi => Err(format!("{name} {x} outside {lo}..={hi}")),
        other => Ok(other),
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
        crate::window::detached(
            ctx,
            "powder",
            "Powder weight",
            &mut open,
            [380.0, 480.0],
            |ui| self.body(ui),
        );
        self.open = open;
    }

    fn body(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("pw_params").num_columns(2).show(ui, |ui| {
            crate::widgets::uniform_num_fields(ui);
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

    /// MCP `powder`: the fields (range-checked like the widgets), then
    /// "Compute weight" and the result table as text.
    pub fn apply_mcp(&mut self, a: &crate::mcp::PowderArgs) -> Result<String, String> {
        self.open = true;
        if let Some(f) = &a.formula {
            self.formula = f.trim().to_owned();
        }
        if let Some(v) = check(a.energy, 100.0, 800_000.0, "energy")? {
            self.energy = v;
        }
        if let Some(v) = check(a.area_cm2, 0.001, 100.0, "area_cm2")? {
            self.area_cm2 = v;
        }
        if let Some(v) = check(a.absorbance, 0.01, 20.0, "absorbance")? {
            self.absorbance = v;
        }
        self.compute();
        let setup = format!(
            "{} at {} eV, area {} cm², target μx {}",
            self.formula, self.energy, self.area_cm2, self.absorbance
        );
        match &self.result {
            Some(Ok((mass_mg, mu_mass))) => Ok(format!(
                "{setup}\nμ/ρ at energy  {mu_mass:.3} cm²/g\nsample mass    {mass_mg:.3} mg\nareal density  {:.4} mg/cm²",
                mass_mg / self.area_cm2
            )),
            Some(Err(e)) => Err(format!("{setup}\n{e}")),
            None => Err(setup),
        }
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

/// **k ↔ E conversion** (the original XAFSView Tools menu item): convert between
/// photoelectron wavenumber `k` [Å⁻¹] and photon energy `E` [eV] about an
/// absorption-edge energy `E₀`, using larch's `KTOE` (`E − E₀ = KTOE·k²`, with
/// [`feffit::feffdat::ktoe`]/[`feffit::feffdat::etok`]). Handy for locating a glitch or Bragg
/// peak seen in one space while working in the other. `E₀` is seeded from the
/// active group's edge energy when the window opens and can be overridden.
pub struct KeConvertWindow {
    pub open: bool,
    /// Edge energy `E₀` [eV].
    e0: f64,
    /// Whether `E₀` has been seeded from the active group since the last open.
    seeded: bool,
    /// Absolute photon energy [eV] for the E→k direction.
    energy: f64,
    /// Photoelectron wavenumber [Å⁻¹] for the k→E direction.
    k: f64,
}

impl Default for KeConvertWindow {
    fn default() -> Self {
        Self {
            open: false,
            e0: 0.0,
            seeded: false,
            energy: 0.0,
            k: 0.0,
        }
    }
}

impl KeConvertWindow {
    /// Render the window. `group_e0` is the active group's edge energy; it seeds
    /// `E₀` the first time the window opens (the user may then override it).
    pub fn show(&mut self, ctx: &egui::Context, group_e0: Option<f64>) {
        if !self.open {
            // Re-seed from the active group on the next open.
            self.seeded = false;
            return;
        }
        self.seed(group_e0);

        let mut open = self.open;
        crate::window::detached(
            ctx,
            "ke_convert",
            "k ↔ E conversion",
            &mut open,
            [380.0, 210.0],
            |ui| {
                ui.horizontal(|ui| {
                    ui.label("Edge E₀ (eV)");
                    ui.add(egui::DragValue::new(&mut self.e0).speed(0.1));
                    if let Some(e0) = group_e0
                        && ui
                            .small_button("use group E₀")
                            .on_hover_text(format!("active group E₀ = {e0:.3} eV"))
                            .clicked()
                    {
                        self.e0 = e0;
                    }
                });
                ui.weak("E − E₀ = KTOE·k²   (KTOE = 3.809982 eV·Å²)");
                ui.separator();

                // E → k
                ui.horizontal(|ui| {
                    ui.label("E");
                    ui.add(
                        egui::DragValue::new(&mut self.energy)
                            .speed(0.5)
                            .suffix(" eV"),
                    );
                    ui.label("⇒");
                    if let Some(k) = k_from_energy(self.energy, self.e0) {
                        ui.strong(format!("k = {k:.4} Å⁻¹"));
                    } else {
                        ui.strong("k = 0 Å⁻¹");
                        ui.weak("(at/below edge)");
                    }
                });

                // k → E
                ui.horizontal(|ui| {
                    ui.label("k");
                    ui.add(
                        egui::DragValue::new(&mut self.k)
                            .speed(0.05)
                            .range(0.0..=1000.0)
                            .suffix(" Å⁻¹"),
                    );
                    ui.label("⇒");
                    ui.strong(format!(
                        "E = {:.3} eV",
                        self.e0 + feffit::feffdat::ktoe(self.k)
                    ));
                });
            },
        );
        self.open = open;
    }
}

impl KeConvertWindow {
    /// Seed `E₀` (and an unset `E`) from the active group's edge energy once
    /// per open; later calls keep the user's override.
    fn seed(&mut self, group_e0: Option<f64>) {
        if self.seeded {
            return;
        }
        if let Some(e0) = group_e0 {
            self.e0 = e0;
            if self.energy <= 0.0 {
                self.energy = e0;
            }
        }
        self.seeded = true;
    }

    /// MCP `ke_convert`: seed as an open does, apply `E₀` / `E` / `k`, and
    /// return both conversion rows as text. `group_e0` is the active group's
    /// edge energy.
    pub fn apply_mcp(
        &mut self,
        a: &crate::mcp::KeConvertArgs,
        group_e0: Option<f64>,
    ) -> Result<String, String> {
        self.open = true;
        self.seed(group_e0);
        if let Some(v) = a.e0 {
            self.e0 = v;
        }
        if let Some(v) = a.energy {
            self.energy = v;
        }
        if let Some(v) = check(a.k, 0.0, 1000.0, "k")? {
            self.k = v;
        }
        let e_to_k = match k_from_energy(self.energy, self.e0) {
            Some(k) => format!("k = {k:.4} Å⁻¹"),
            None => "k = 0 Å⁻¹ (at/below edge)".to_owned(),
        };
        Ok(format!(
            "E₀ = {:.3} eV\nE = {:.3} eV ⇒ {e_to_k}\nk = {:.4} Å⁻¹ ⇒ E = {:.3} eV",
            self.e0,
            self.energy,
            self.k,
            self.e0 + feffit::feffdat::ktoe(self.k)
        ))
    }
}

/// Photoelectron wavenumber [Å⁻¹] for photon energy `energy` about edge `e0`;
/// `None` at or below the edge. `feffit::feffdat::etok` already returns `k`
/// (`sqrt(ΔE·ETOK)`), so no further root is taken here.
fn k_from_energy(energy: f64, e0: f64) -> Option<f64> {
    let de = energy - e0;
    (de > 0.0).then(|| feffit::feffdat::etok(de))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_from_energy_matches_ktoe_inverse() {
        // ΔE = 100 eV → k = sqrt(100 / 3.809982) = 5.1232 Å⁻¹ (not the 2.2633 a
        // second sqrt would give).
        let k = k_from_energy(11664.3, 11564.3).unwrap();
        assert!((k - 5.1232).abs() < 5e-5, "k = {k}");
        assert!((feffit::feffdat::ktoe(k) - 100.0).abs() < 1e-9);
        assert_eq!(k_from_energy(11564.3, 11564.3), None);
        assert_eq!(k_from_energy(11000.0, 11564.3), None);
    }

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
    fn parse_element_accepts_symbol_or_z() {
        assert_eq!(parse_element("Fe"), Ok(26));
        assert_eq!(parse_element("fe"), Ok(26));
        assert_eq!(parse_element(" 29 "), Ok(29));
        assert!(parse_element("0").is_err());
        assert!(parse_element("119").is_err());
        assert!(parse_element("Xx").is_err());
    }

    #[test]
    fn periodic_table_mcp_reports_the_clicked_element() {
        let mut w = PeriodicTableWindow::default();
        assert_eq!(w.apply_mcp(None).unwrap(), "no element selected");
        let text = w.apply_mcp(Some("Cu")).unwrap();
        assert!(text.starts_with("Cu — "), "{text}");
        assert!(text.contains("K   ") && text.contains("8979"), "{text}");
        assert_eq!(w.selected, Some(29));
        assert!(w.open);
    }

    #[test]
    fn ion_chamber_mcp_replaces_the_gas_mix_and_computes() {
        let mut w = IonChamberWindow::default();
        let a = crate::mcp::IonChamberArgs {
            gases: Some(vec![crate::mcp::GasArg {
                name: "argon".into(),
                frac: 0.5,
            }]),
            energy: Some(9000.0),
            length_cm: None,
            volts: None,
            sensitivity: None,
            with_compton: None,
            both_carriers: None,
        };
        let text = w.apply_mcp(&a).unwrap();
        assert!(text.starts_with("gases: argon 0.500;"), "{text}");
        assert!(text.contains("incident"), "{text}");
        assert!(matches!(w.result, Some(Ok(_))));
        let bad = crate::mcp::IonChamberArgs {
            gases: Some(vec![crate::mcp::GasArg {
                name: "neon".into(),
                frac: 1.0,
            }]),
            energy: None,
            length_cm: None,
            volts: None,
            sensitivity: None,
            with_compton: None,
            both_carriers: None,
        };
        assert!(w.apply_mcp(&bad).unwrap_err().contains("unknown gas"));
    }

    #[test]
    fn ke_convert_mcp_seeds_from_the_group_then_keeps_an_override() {
        let mut w = KeConvertWindow::default();
        let a = crate::mcp::KeConvertArgs {
            e0: None,
            energy: Some(11664.3),
            k: Some(5.1232),
        };
        let text = w.apply_mcp(&a, Some(11564.3)).unwrap();
        assert!(text.starts_with("E₀ = 11564.300 eV"), "{text}");
        assert!(text.contains("k = 5.1232 Å⁻¹"), "{text}");
        let b = crate::mcp::KeConvertArgs {
            e0: Some(8979.0),
            energy: None,
            k: None,
        };
        let text = w.apply_mcp(&b, Some(11564.3)).unwrap();
        assert!(text.starts_with("E₀ = 8979.000 eV"), "{text}");
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
