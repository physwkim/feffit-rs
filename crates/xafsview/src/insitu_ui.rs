//! The **In-situ map** window — the series-image viewer ported from the
//! BL10C_XAFS (PyMca-based) beamline app's IMAGE tab.
//!
//! A scan series (an in-situ run) renders as a 2-D map: one row per scan in
//! series order (first scan at the top), the x axis either energy (XANES) or
//! `R` (RDF, `|χ(R)|`). A reference scan can be subtracted from every row (the
//! original's "Sub" checkbox — its hard-coded Δμ-vs-first-scan XANES map is
//! `reference = scan 1` here), rendered on a diverging colormap since the
//! difference is signed. Clicking the map extracts a profile: the clicked
//! scan's curve, or the time trace of the clicked x across the series (the
//! original's profile-selection widget).
//!
//! Two data sources: the session's loaded groups (XANES uses the normalized
//! μ(E) interpolated onto a common energy grid; RDF uses each group's `|χ(R)|`
//! from the forward FT), or a legacy BL10C series `.dat` file
//! ([`parse_series_dat`]). "Save map" writes the *displayed* values (reference
//! subtraction applied) as a tab-separated matrix — x in the first column, one
//! named column per scan, exactly the layout of the original's `*_Sub_*.dat`
//! exports — which OriginPro imports directly (first row = column long names;
//! worksheet → matrix for a contour plot) and [`parse_series_dat`] re-opens.

use std::fmt::Write as _;

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use feffit::xasdata::XasGroup;
use feffit::xasproc::mathutils::interp_linear;
use rsplot::YAxis;

use crate::plot::BLUE;
use crate::wavelet::plasma;

/// Which physical axis a series map lives on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SeriesKind {
    /// μ(E) vs energy (eV).
    Xanes,
    /// `|χ(R)|` vs `R` (Å).
    Rdf,
}

impl SeriesKind {
    /// The x-axis label, spelled as the BL10C app's series files spell it so
    /// exported maps round-trip through [`parse_series_dat`].
    fn x_label(self) -> &'static str {
        match self {
            SeriesKind::Xanes => "Energy (eV)",
            SeriesKind::Rdf => "R (angstrom)",
        }
    }
}

/// A scan series on a common x grid: `rows[i][j]` is scan `i` (series order) at
/// `x[j]`; `labels[i]` names scan `i`. Every row has `x.len()` values.
pub struct SeriesMap {
    pub kind: SeriesKind,
    pub x: Vec<f64>,
    pub labels: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

impl SeriesMap {
    /// The displayed value at `(row, col)`: the stored value minus the
    /// reference scan's value at the same x, when a reference is set.
    fn display(&self, reference: Option<usize>, row: usize, col: usize) -> f64 {
        let v = self.rows[row][col];
        match reference {
            Some(r) => v - self.rows[r][col],
            None => v,
        }
    }

    /// `(min, max)` over all displayed values.
    fn display_range(&self, reference: Option<usize>) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for i in 0..self.rows.len() {
            for j in 0..self.x.len() {
                let v = self.display(reference, i, j);
                min = min.min(v);
                max = max.max(v);
            }
        }
        (min, max)
    }

    /// Serialize the displayed map as the tab-separated series layout: header
    /// `x-label \t scan names…`, then one line per x with the x value and one
    /// column per scan. This is the BL10C `*_Sub_*.dat` layout, re-openable by
    /// [`parse_series_dat`] and directly importable in OriginPro (first row as
    /// column long names; worksheet → matrix → contour for the image).
    pub fn to_dat(&self, reference: Option<usize>) -> String {
        let mut s = String::from(self.kind.x_label());
        for l in &self.labels {
            s.push('\t');
            s.push_str(l);
        }
        s.push('\n');
        for j in 0..self.x.len() {
            let _ = write!(s, "{:.6}", self.x[j]);
            for i in 0..self.rows.len() {
                let _ = write!(s, "\t{:.6}", self.display(reference, i, j));
            }
            s.push('\n');
        }
        s
    }
}

/// `n` evenly spaced values over `[a, b]` (`n ≥ 2`).
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| a + step * i as f64).collect()
}

/// Parse a BL10C series `.dat` file. Tab-separated header, whitespace-separated
/// numeric rows. Three layouts, dispatched on the header:
///
/// - first cell starts with `R` → RDF: `x` + one column per scan;
/// - first cell starts with `E` and the third repeats it → XANES `(E, μ)`
///   column pairs, one pair per scan, interpolated onto a common grid spanning
///   the first pair's energies (what the original's `_processData` did —
///   except its hard-coded Δμ-vs-first-scan, which the reference control
///   replaces here);
/// - first cell starts with `E` otherwise → XANES already on one grid
///   (`x` + one column per scan) — the layout [`SeriesMap::to_dat`] writes.
///
/// Returns the series with the skipped-line count folded into the error when
/// a row cannot be read.
pub fn parse_series_dat(text: &str) -> Result<SeriesMap, String> {
    let mut lines = text.lines().enumerate();
    let (_, header) = lines
        .by_ref()
        .find(|(_, l)| !l.trim().is_empty())
        .ok_or("empty file")?;
    let header: Vec<&str> = header.split('\t').map(str::trim).collect();
    if header.len() < 2 {
        return Err("header has no scan columns (expected tab-separated names)".into());
    }
    let first = header[0].to_ascii_uppercase();
    let (kind, paired) = if first.starts_with('R') {
        (SeriesKind::Rdf, false)
    } else if first.starts_with('E') {
        // (E, μ) pairs repeat the energy label at every even position.
        (
            SeriesKind::Xanes,
            header.len() >= 4 && header[2] == header[0],
        )
    } else {
        return Err(format!(
            "unrecognized series header '{}' (expected an R… or E… first column)",
            header[0]
        ));
    };

    let labels: Vec<String> = if paired {
        header
            .iter()
            .skip(1)
            .step_by(2)
            .map(|s| s.to_string())
            .collect()
    } else {
        header[1..].iter().map(|s| s.to_string()).collect()
    };
    let ncols = if paired {
        2 * labels.len()
    } else {
        1 + labels.len()
    };

    // Parse the numeric block: data[r] is one file line.
    let mut data: Vec<Vec<f64>> = Vec::new();
    for (ln, line) in lines {
        if line.trim().is_empty() {
            continue;
        }
        let row: Result<Vec<f64>, _> = line.split_whitespace().map(str::parse::<f64>).collect();
        let row = row.map_err(|e| format!("line {}: {e}", ln + 1))?;
        if row.len() != ncols {
            return Err(format!(
                "line {}: {} values, expected {ncols}",
                ln + 1,
                row.len()
            ));
        }
        data.push(row);
    }
    if data.len() < 2 {
        return Err("no data rows".into());
    }

    if !paired {
        let x: Vec<f64> = data.iter().map(|r| r[0]).collect();
        let rows: Vec<Vec<f64>> = (0..labels.len())
            .map(|i| data.iter().map(|r| r[1 + i]).collect())
            .collect();
        return Ok(SeriesMap {
            kind,
            x,
            labels,
            rows,
        });
    }

    // Column pairs: common grid over the first pair's energy span, every scan
    // interpolated onto it.
    let e0: Vec<f64> = data.iter().map(|r| r[0]).collect();
    let (lo, hi) = e0
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if hi <= lo {
        return Err("the first energy column has no span".into());
    }
    let x = linspace(lo, hi, e0.len());
    let rows: Vec<Vec<f64>> = (0..labels.len())
        .map(|i| {
            let e: Vec<f64> = data.iter().map(|r| r[2 * i]).collect();
            let y: Vec<f64> = data.iter().map(|r| r[2 * i + 1]).collect();
            interp_linear(&x, &e, &y)
        })
        .collect();
    Ok(SeriesMap {
        kind,
        x,
        labels,
        rows,
    })
}

/// Build the XANES map from the loaded groups: each group's normalized μ(E)
/// (flattened if available, else normalized, else raw μ) interpolated onto
/// `steps` points over `[start, stop]` eV. Returns the map and the labels of
/// groups skipped for having no spectrum.
pub fn map_from_groups_xanes(
    groups: &[XasGroup],
    start: f64,
    stop: f64,
    steps: usize,
) -> Result<(SeriesMap, Vec<String>), String> {
    if stop <= start || steps < 2 {
        return Err("energy grid needs stop > start and at least 2 steps".into());
    }
    let x = linspace(start, stop, steps);
    let mut labels = Vec::new();
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for g in groups {
        let y = g.flat.as_ref().or(g.norm.as_ref()).unwrap_or(&g.mu);
        if g.energy.len() < 2 || y.len() != g.energy.len() {
            skipped.push(g.label.clone());
            continue;
        }
        labels.push(g.label.clone());
        rows.push(interp_linear(&x, &g.energy, y));
    }
    if rows.is_empty() {
        return Err("no loaded group has a μ(E) spectrum".into());
    }
    Ok((
        SeriesMap {
            kind: SeriesKind::Xanes,
            x,
            labels,
            rows,
        },
        skipped,
    ))
}

/// Build the RDF map from the loaded groups: each group's `|χ(R)|` on the
/// first reduced group's R grid (other grids are interpolated onto it).
/// Returns the map and the labels of groups skipped for having no FT yet.
pub fn map_from_groups_rdf(groups: &[XasGroup]) -> Result<(SeriesMap, Vec<String>), String> {
    let mut x: Option<Vec<f64>> = None;
    let mut labels = Vec::new();
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for g in groups {
        let (Some(r), Some(mag)) = (g.r.as_ref(), g.chir_mag.as_ref()) else {
            skipped.push(g.label.clone());
            continue;
        };
        if r.len() < 2 || mag.len() != r.len() {
            skipped.push(g.label.clone());
            continue;
        }
        let grid = x.get_or_insert_with(|| r.clone());
        labels.push(g.label.clone());
        if grid == r {
            rows.push(mag.clone());
        } else {
            rows.push(interp_linear(grid, r, mag));
        }
    }
    let Some(x) = x else {
        return Err("no loaded group has |χ(R)| — run AUTOBK / reduce first".into());
    };
    Ok((
        SeriesMap {
            kind: SeriesKind::Rdf,
            x,
            labels,
            rows,
        },
        skipped,
    ))
}

/// A diverging blue–white–red map for signed (reference-subtracted) values,
/// `t ∈ [0, 1]` with the zero of the data at `t = 0.5`.
fn diverging(t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8, f: f64| (a as f64 + (b as f64 - a as f64) * f).round() as u8;
    let (c0, c1, f) = if t < 0.5 {
        ([33u8, 102, 172], [247u8, 247, 247], t * 2.0)
    } else {
        ([247u8, 247, 247], [178u8, 24, 43], (t - 0.5) * 2.0)
    };
    Color32::from_rgb(
        lerp(c0[0], c1[0], f),
        lerp(c0[1], c1[1], f),
        lerp(c0[2], c1[2], f),
    )
}

/// Which profile a click on the map extracts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProfileMode {
    /// The clicked scan's full curve vs x.
    Scan,
    /// The clicked x position's value across the series (vs scan number).
    Trace,
}

/// The In-situ map window.
pub struct InsituWindow {
    pub open: bool,
    map: Option<SeriesMap>,
    /// Channel for "Build from loaded groups".
    channel: SeriesKind,
    /// XANES common-grid fields; `e_steps == 0` means "seed from the data".
    e_start: f64,
    e_stop: f64,
    e_steps: usize,
    /// Reference scan (row index) subtracted from every row, if any.
    reference: Option<usize>,
    autoscale: bool,
    vmin: f64,
    vmax: f64,
    profile: ProfileMode,
    /// The clicked map cell `(row, col)` the profile follows.
    sel: Option<(usize, usize)>,
    plot: crate::plot::Plot,
    plot_dirty: bool,
    texture: Option<egui::TextureHandle>,
    status: String,
}

impl InsituWindow {
    /// Build the window with its own profile plot (`PlotId` 8).
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = crate::plot::Plot::new(render_state, 8);
        plot.set_graph_title("Profile");
        plot.set_empty_hint("Click the map to extract a profile.");
        Self {
            open: false,
            map: None,
            channel: SeriesKind::Rdf,
            e_start: 0.0,
            e_stop: 0.0,
            e_steps: 0,
            reference: None,
            autoscale: true,
            vmin: 0.0,
            vmax: 1.0,
            profile: ProfileMode::Scan,
            sel: None,
            plot,
            plot_dirty: false,
            texture: None,
            status: String::new(),
        }
    }

    /// Install a freshly built series, resetting the per-series view state.
    fn set_map(&mut self, map: SeriesMap, status: String) {
        self.map = Some(map);
        self.reference = None;
        self.sel = None;
        self.texture = None;
        self.plot_dirty = true;
        self.status = status;
    }

    /// Apply one `insitu_map` MCP call against the loaded groups: build the map
    /// (from a series file or the groups), optionally set the reference and
    /// save the displayed map, and open the window. Returns the report line.
    pub fn apply_mcp(
        &mut self,
        groups: &[XasGroup],
        args: &crate::mcp::InsituMapArgs,
    ) -> Result<String, String> {
        if let Some(file) = &args.file {
            let text = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
            let map = parse_series_dat(&text)?;
            let n = map.labels.len();
            self.set_map(map, format!("Opened {file}: {n} scan(s)."));
        } else {
            let channel = args.channel.as_deref().unwrap_or("rdf");
            match channel.to_ascii_lowercase().as_str() {
                "rdf" => self.channel = SeriesKind::Rdf,
                "xanes" => self.channel = SeriesKind::Xanes,
                other => return Err(format!("unknown channel '{other}' (rdf or xanes)")),
            }
            if self.channel == SeriesKind::Xanes {
                if let Some(v) = args.e_start {
                    self.e_start = v;
                }
                if let Some(v) = args.e_stop {
                    self.e_stop = v;
                }
                if let Some(v) = args.e_steps {
                    self.e_steps = v;
                }
                if self.e_steps == 0 || self.e_stop <= self.e_start {
                    self.seed_grid(groups);
                }
            }
            self.build_from_groups(groups);
            if self.map.is_none() {
                return Err(self.status.clone());
            }
        }
        if let Some(r) = args.reference {
            let n = self.map.as_ref().map_or(0, |m| m.labels.len());
            if r == 0 || r > n {
                return Err(format!("reference {r} out of range (1..={n})"));
            }
            self.reference = Some(r - 1);
            self.texture = None;
            self.plot_dirty = true;
        }
        let map = self.map.as_ref().expect("map was just set");
        let mut report = format!(
            "{} map: {} scan(s) × {} point(s), x {:.4}–{:.4}",
            match map.kind {
                SeriesKind::Xanes => "XANES",
                SeriesKind::Rdf => "RDF",
            },
            map.labels.len(),
            map.x.len(),
            map.x.first().copied().unwrap_or(0.0),
            map.x.last().copied().unwrap_or(0.0),
        );
        let (lo, hi) = map.display_range(self.reference);
        let _ = write!(report, ", value {lo:.4}–{hi:.4}");
        if let Some(r) = self.reference {
            let _ = write!(report, ", reference scan {} ({})", r + 1, map.labels[r]);
        }
        if let Some(path) = &args.save {
            let dat = map.to_dat(self.reference);
            std::fs::write(path, dat).map_err(|e| format!("save {path}: {e}"))?;
            let _ = write!(report, "; wrote {path}");
        }
        if !self.status.is_empty() {
            let _ = write!(report, " [{}]", self.status);
        }
        self.open = true;
        Ok(report)
    }

    /// Seed the XANES grid from the loaded groups: the intersection of their
    /// energy ranges, sampled at the largest group's point count.
    fn seed_grid(&mut self, groups: &[XasGroup]) {
        let mut start = f64::NEG_INFINITY;
        let mut stop = f64::INFINITY;
        let mut steps = 0usize;
        for g in groups {
            if g.energy.len() < 2 {
                continue;
            }
            start = start.max(g.energy[0]);
            stop = stop.min(*g.energy.last().expect("non-empty"));
            steps = steps.max(g.energy.len());
        }
        if steps == 0 {
            return;
        }
        if stop <= start {
            // Disjoint ranges: fall back to the union so the fields stay usable.
            start = f64::INFINITY;
            stop = f64::NEG_INFINITY;
            for g in groups {
                if g.energy.len() < 2 {
                    continue;
                }
                start = start.min(g.energy[0]);
                stop = stop.max(*g.energy.last().expect("non-empty"));
            }
        }
        self.e_start = start;
        self.e_stop = stop;
        self.e_steps = steps.clamp(50, 2000);
    }

    /// Build the map from the loaded groups with the current channel settings.
    fn build_from_groups(&mut self, groups: &[XasGroup]) {
        let built = match self.channel {
            SeriesKind::Xanes => {
                map_from_groups_xanes(groups, self.e_start, self.e_stop, self.e_steps)
            }
            SeriesKind::Rdf => map_from_groups_rdf(groups),
        };
        match built {
            Ok((map, skipped)) => {
                let mut status = format!("Built from {} group(s).", map.labels.len());
                if !skipped.is_empty() {
                    let _ = write!(status, " Skipped (no data): {}.", skipped.join(", "));
                }
                self.set_map(map, status);
            }
            Err(e) => self.status = e,
        }
    }

    /// Render the window over `groups`.
    pub fn show(&mut self, ctx: &egui::Context, groups: &[XasGroup]) {
        if !self.open {
            return;
        }
        let mut open = self.open;
        crate::window::detached_panels(
            ctx,
            "insitu",
            "In-situ map (scan series)",
            &mut open,
            [920.0, 640.0],
            |ui| {
                egui::Panel::left("insitu_controls")
                    .resizable(true)
                    .default_size(300.0)
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, groups));
                    });
                egui::CentralPanel::default().show(ui, |ui| self.map_pane(ui));
            },
        );
        self.open = open;
    }

    fn controls(&mut self, ui: &mut egui::Ui, groups: &[XasGroup]) {
        ui.heading("In-situ map");
        ui.label("One row per scan in series order (first scan at the top).");

        ui.separator();
        ui.strong("Build from loaded groups");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.channel, SeriesKind::Rdf, "RDF |χ(R)|");
            ui.selectable_value(&mut self.channel, SeriesKind::Xanes, "XANES μ(E)");
        });
        if self.channel == SeriesKind::Xanes {
            if self.e_steps == 0 && !groups.is_empty() {
                self.seed_grid(groups);
            }
            ui.horizontal(|ui| {
                ui.label("Start [eV]");
                ui.add(egui::DragValue::new(&mut self.e_start).speed(1.0));
                ui.label("Stop [eV]");
                ui.add(egui::DragValue::new(&mut self.e_stop).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Steps");
                ui.add(egui::DragValue::new(&mut self.e_steps).range(2..=20000));
                if ui
                    .button("From data")
                    .on_hover_text("Reset the grid to the loaded groups' common energy range")
                    .clicked()
                {
                    self.seed_grid(groups);
                }
            });
        }
        if ui
            .add_enabled(!groups.is_empty(), egui::Button::new("Build map"))
            .clicked()
        {
            self.build_from_groups(groups);
        }
        if groups.is_empty() {
            ui.weak("Load a scan series first (Open data / batch load).");
        }

        ui.separator();
        if ui
            .button("Open series file (.dat)…")
            .on_hover_text(
                "A BL10C series file: RDF (x + one column per scan) or \
                 XANES ((E, μ) column pairs)",
            )
            .clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("series data", &["dat", "txt"])
                .pick_file()
        {
            match std::fs::read_to_string(&path).map_err(|e| e.to_string()) {
                Ok(text) => match parse_series_dat(&text) {
                    Ok(map) => {
                        let n = map.labels.len();
                        self.set_map(map, format!("Opened {}: {n} scan(s).", path.display()));
                    }
                    Err(e) => self.status = format!("{}: {e}", path.display()),
                },
                Err(e) => self.status = format!("{}: {e}", path.display()),
            }
        }

        let Some(map) = &self.map else {
            if !self.status.is_empty() {
                ui.separator();
                ui.weak(&self.status);
            }
            return;
        };

        ui.separator();
        ui.label(format!(
            "{} scan(s) × {} point(s)",
            map.labels.len(),
            map.x.len()
        ));

        let ref_label = match self.reference {
            Some(r) => format!("{}: {}", r + 1, map.labels[r]),
            None => "None".to_owned(),
        };
        let before = self.reference;
        egui::ComboBox::from_label("Reference (subtract)")
            .selected_text(ref_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.reference, None, "None");
                for (i, l) in map.labels.iter().enumerate() {
                    ui.selectable_value(&mut self.reference, Some(i), format!("{}: {l}", i + 1));
                }
            });
        if self.reference != before {
            self.texture = None;
            self.plot_dirty = true;
        }

        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.autoscale, "Autoscale").changed() {
                self.texture = None;
            }
            let step = ((self.vmax - self.vmin).abs() / 200.0).max(1e-6);
            let lo = ui.add_enabled(
                !self.autoscale,
                egui::DragValue::new(&mut self.vmin)
                    .speed(step)
                    .prefix("min "),
            );
            let hi = ui.add_enabled(
                !self.autoscale,
                egui::DragValue::new(&mut self.vmax)
                    .speed(step)
                    .prefix("max "),
            );
            if lo.changed() || hi.changed() {
                self.texture = None;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Profile");
            let before = self.profile;
            ui.selectable_value(&mut self.profile, ProfileMode::Scan, "Scan curve");
            ui.selectable_value(&mut self.profile, ProfileMode::Trace, "Time trace");
            if self.profile != before {
                self.plot_dirty = true;
            }
        });

        ui.separator();
        if ui
            .button("Save map (.dat, Origin-ready)…")
            .on_hover_text(
                "Tab-separated matrix of the displayed values: x in the first \
                 column, one named column per scan — imports into OriginPro \
                 (first row = long names) and re-opens here",
            )
            .clicked()
            && let Some(path) = rfd::FileDialog::new()
                .set_file_name("insitu_map.dat")
                .save_file()
        {
            let dat = map.to_dat(self.reference);
            self.status = match std::fs::write(&path, dat) {
                Ok(()) => format!("Saved {}.", path.display()),
                Err(e) => format!("save {}: {e}", path.display()),
            };
        }
        if !self.status.is_empty() {
            ui.weak(&self.status);
        }
    }

    /// The central pane: the map image (clickable) above the profile plot.
    fn map_pane(&mut self, ui: &mut egui::Ui) {
        let Some(map) = &self.map else {
            ui.weak("Build a map from the loaded groups, or open a series .dat file.");
            return;
        };

        // (Re)build the heatmap texture when the data, reference, or scale
        // changed. Autoscale writes the realized range back into vmin/vmax so
        // the manual fields start from it.
        if self.texture.is_none() {
            if self.autoscale {
                let (lo, hi) = map.display_range(self.reference);
                if self.reference.is_some() {
                    // Signed Δ map: symmetric range so zero sits on white.
                    let m = lo.abs().max(hi.abs()).max(f64::MIN_POSITIVE);
                    self.vmin = -m;
                    self.vmax = m;
                } else {
                    self.vmin = lo;
                    self.vmax = hi;
                }
            }
            let span = (self.vmax - self.vmin).max(f64::MIN_POSITIVE);
            let w = map.x.len();
            let h = map.rows.len();
            let signed = self.reference.is_some();
            let mut pixels = Vec::with_capacity(w * h);
            for i in 0..h {
                for j in 0..w {
                    let t = (map.display(self.reference, i, j) - self.vmin) / span;
                    pixels.push(if signed { diverging(t) } else { plasma(t) });
                }
            }
            let image = egui::ColorImage::new([w, h], pixels);
            self.texture = Some(ui.ctx().load_texture(
                "insitu_map",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }

        let axis = format!(
            "{}: {:.3} … {:.3}   (scan 1 at the top, scan {} at the bottom)",
            map.kind.x_label(),
            map.x.first().copied().unwrap_or(0.0),
            map.x.last().copied().unwrap_or(0.0),
            map.rows.len(),
        );

        let tex = self.texture.as_ref().expect("texture was just built");
        let w = ui.available_width().clamp(200.0, 1200.0);
        let img_h = (ui.available_height() * 0.5).clamp(120.0, 420.0);
        let resp = ui.add(
            egui::Image::new(tex)
                .fit_to_exact_size(egui::vec2(w, img_h))
                .maintain_aspect_ratio(false)
                .sense(egui::Sense::click()),
        );

        // Map the pointer to a (row, col) cell for the readout and the click.
        let cell = resp.hover_pos().map(|pos| {
            let rel = pos - resp.rect.min;
            let col = ((rel.x / resp.rect.width()) * map.x.len() as f32) as usize;
            let row = ((rel.y / resp.rect.height()) * map.rows.len() as f32) as usize;
            (row.min(map.rows.len() - 1), col.min(map.x.len() - 1))
        });
        match cell {
            Some((i, j)) => ui.weak(format!(
                "{axis}   |   scan {} ({}), x = {:.4}, value = {:.5}",
                i + 1,
                map.labels[i],
                map.x[j],
                map.display(self.reference, i, j),
            )),
            None => ui.weak(axis),
        };
        if resp.clicked()
            && let Some(c) = cell
        {
            self.sel = Some(c);
            self.plot_dirty = true;
        }

        ui.separator();
        if self.plot_dirty {
            self.rebuild_profile();
            self.plot_dirty = false;
        }
        crate::plot::show(&mut self.plot, ui);
    }

    /// Rebuild the profile plot from the current selection.
    fn rebuild_profile(&mut self) {
        self.plot.clear();
        let Some(map) = &self.map else { return };
        let Some((i, j)) = self.sel else { return };
        let y_label = if self.reference.is_some() {
            "Δ value"
        } else {
            "value"
        };
        match self.profile {
            ProfileMode::Scan => {
                let y: Vec<f64> = (0..map.x.len())
                    .map(|c| map.display(self.reference, i, c))
                    .collect();
                self.plot.set_graph_x_label(map.kind.x_label());
                self.plot.set_graph_y_label(y_label, YAxis::Left);
                self.plot
                    .add_curve_with_legend(&map.x, &y, BLUE, &map.labels[i]);
            }
            ProfileMode::Trace => {
                let xs: Vec<f64> = (1..=map.rows.len()).map(|n| n as f64).collect();
                let y: Vec<f64> = (0..map.rows.len())
                    .map(|r| map.display(self.reference, r, j))
                    .collect();
                self.plot.set_graph_x_label("scan #");
                self.plot.set_graph_y_label(y_label, YAxis::Left);
                self.plot
                    .add_curve_with_legend(&xs, &y, BLUE, format!("x = {:.4}", map.x[j]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rdf_series_layout() {
        let text = "R (angstrom)\ta_000\ta_001\n\
                    0\t0.1\t0.2\n\
                    0.5\t0.3\t0.4\n\
                    1.0\t0.5\t0.6\n";
        let m = parse_series_dat(text).expect("rdf");
        assert!(matches!(m.kind, SeriesKind::Rdf));
        assert_eq!(m.labels, vec!["a_000", "a_001"]);
        assert_eq!(m.x, vec![0.0, 0.5, 1.0]);
        assert_eq!(m.rows[0], vec![0.1, 0.3, 0.5]);
        assert_eq!(m.rows[1], vec![0.2, 0.4, 0.6]);
    }

    #[test]
    fn parses_xanes_pair_layout_onto_common_grid() {
        // Two scans on slightly different energy grids (the beamline reality):
        // the common grid spans the FIRST pair's energies.
        let text = "Energy (eV)\ts0\tEnergy (eV)\ts1\n\
                    6339.0\t1.0\t6339.5\t2.0\n\
                    6341.0\t2.0\t6341.5\t3.0\n\
                    6343.0\t3.0\t6343.5\t4.0\n";
        let m = parse_series_dat(text).expect("xanes pairs");
        assert!(matches!(m.kind, SeriesKind::Xanes));
        assert_eq!(m.labels, vec!["s0", "s1"]);
        assert_eq!(m.x, vec![6339.0, 6341.0, 6343.0]);
        // Scan 0 lands exactly on its own grid.
        assert_eq!(m.rows[0], vec![1.0, 2.0, 3.0]);
        // Scan 1 is linear in E, so interpolation reproduces the line
        // (clamped at the left edge below its first point).
        assert_eq!(m.rows[1], vec![2.0, 2.75, 3.75]);
    }

    #[test]
    fn export_round_trips_through_the_parser() {
        let m = SeriesMap {
            kind: SeriesKind::Xanes,
            x: vec![6339.0, 6341.0, 6343.0],
            labels: vec!["s0".into(), "s1".into()],
            rows: vec![vec![1.0, 2.0, 3.0], vec![2.0, 2.75, 3.75]],
        };
        // The single-x XANES export must NOT re-parse as the pair layout.
        let back = parse_series_dat(&m.to_dat(None)).expect("round trip");
        assert!(matches!(back.kind, SeriesKind::Xanes));
        assert_eq!(back.labels, m.labels);
        assert_eq!(back.x, m.x);
        for (a, b) in back.rows.iter().flatten().zip(m.rows.iter().flatten()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn export_applies_the_reference_subtraction() {
        let m = SeriesMap {
            kind: SeriesKind::Rdf,
            x: vec![0.0, 1.0],
            labels: vec!["a".into(), "b".into()],
            rows: vec![vec![1.0, 2.0], vec![1.5, 2.25]],
        };
        let back = parse_series_dat(&m.to_dat(Some(0))).expect("round trip");
        // Row a minus itself is zero; row b carries the difference — the
        // original's *_Sub_*.dat export.
        assert_eq!(back.rows[0], vec![0.0, 0.0]);
        for (v, want) in back.rows[1].iter().zip([0.5, 0.25]) {
            assert!((v - want).abs() < 1e-6);
        }
    }

    #[test]
    fn parser_rejects_ragged_and_unknown_layouts() {
        assert!(parse_series_dat("").is_err());
        assert!(parse_series_dat("Q\ta\n1 2\n3 4\n").is_err());
        // 3 header names but a 2-value data row: the error names the line.
        let e = parse_series_dat("R\ta\tb\n0 1\n").err().unwrap();
        assert!(e.contains("line 2"), "{e}");
    }

    #[test]
    fn groups_xanes_map_interpolates_onto_the_grid() {
        let g0 = XasGroup::from_mu("g0", vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
        let mut g1 = XasGroup::from_mu("g1", vec![0.0, 2.0], vec![4.0, 0.0]);
        g1.norm = Some(vec![1.0, 3.0]);
        let empty = XasGroup::from_mu("none", vec![], vec![]);
        let (m, skipped) = map_from_groups_xanes(&[g0, g1, empty], 0.0, 2.0, 3).expect("xanes map");
        assert_eq!(m.labels, vec!["g0", "g1"]);
        assert_eq!(skipped, vec!["none"]);
        // g0 uses μ directly; g1 prefers its normalized spectrum.
        assert_eq!(m.rows[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(m.rows[1], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn groups_rdf_map_skips_unreduced_groups() {
        let mut g0 = XasGroup::from_mu("g0", vec![0.0, 1.0], vec![0.0, 0.0]);
        g0.r = Some(vec![0.0, 0.5, 1.0]);
        g0.chir_mag = Some(vec![0.1, 0.2, 0.3]);
        let raw = XasGroup::from_mu("raw", vec![0.0, 1.0], vec![0.0, 0.0]);
        let (m, skipped) = map_from_groups_rdf(&[g0, raw]).expect("rdf map");
        assert_eq!(m.labels, vec!["g0"]);
        assert_eq!(skipped, vec!["raw"]);
        assert_eq!(m.x, vec![0.0, 0.5, 1.0]);
        assert_eq!(m.rows[0], vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn display_range_and_reference_subtraction() {
        let m = SeriesMap {
            kind: SeriesKind::Rdf,
            x: vec![0.0, 1.0],
            labels: vec!["a".into(), "b".into()],
            rows: vec![vec![1.0, 2.0], vec![0.5, 3.0]],
        };
        assert_eq!(m.display_range(None), (0.5, 3.0));
        // vs reference a: a → 0, b → (-0.5, 1.0).
        assert_eq!(m.display_range(Some(0)), (-0.5, 1.0));
        assert_eq!(m.display(Some(0), 1, 1), 1.0);
    }
}
