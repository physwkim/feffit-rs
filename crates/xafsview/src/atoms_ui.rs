//! Phase-9 UI: the **Atoms** tab (crystal cell → `feff.inp`), the **Feff** tab
//! (edit `feff.inp` and run FEFF), and the **Plot Sites** 3D cluster viewer.
//!
//! All three share one `feff.inp` text buffer owned by the app: Atoms *writes*
//! it, Feff *edits and runs* it, Plot Sites *reads* it. The crystal/cluster math
//! and `feff.inp` parsing live in the headless [`feffinp`] crate; running FEFF is
//! [`feffrun`] (default in-process FEFF10 backend, so no external executables are
//! required); the 3D scene is `siplot`'s [`SceneWidget`].
//!
//! **Scope note.** The Atoms tab does *not* apply space-group symmetry — enter
//! the full unit cell (every atom), not the asymmetric unit. See [`feffinp`]'s
//! crate docs for the deferred space-group expansion.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use eframe::egui;
use eframe::egui_wgpu::RenderState;
use egui::Color32;
use feffinp::{Crystal, Edge, FeffInp, Lattice, Site};
use siplot::{Colormap, ColormapName, PointMarker, Scatter3D, Scene3dGeometry, SceneWidget, Vec3};

const RED: Color32 = Color32::from_rgb(0xd6, 0x27, 0x28);
const GREEN: Color32 = Color32::from_rgb(0x2c, 0xa0, 0x2c);

/// The four selectable edges, paired with their labels.
const EDGES: [(Edge, &str); 4] = [
    (Edge::K, "K"),
    (Edge::L1, "L1"),
    (Edge::L2, "L2"),
    (Edge::L3, "L3"),
];

// ---------------------------------------------------------------------------
// Atoms tab: crystal cell → feff.inp
// ---------------------------------------------------------------------------

/// One editable cell-site row (element + fractional coordinates).
struct SiteRow {
    element: String,
    x: f64,
    y: f64,
    z: f64,
}

/// Signals the app should act after the Atoms tab renders.
pub enum AtomsAction {
    /// A `feff.inp` was built into the shared buffer; switch to the Feff tab.
    BuiltFeffInp,
}

/// The **Atoms** tab state: a unit cell plus the absorber/edge/cluster choices.
pub struct AtomsTab {
    title: String,
    a: f64,
    b: f64,
    c: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    edge: Edge,
    /// International space-group number (1‥230); 1 = P1, no expansion.
    space_group: u32,
    cluster_size: f64,
    absorber: usize,
    sites: Vec<SiteRow>,
    status: Option<Result<String, String>>,
}

impl Default for AtomsTab {
    fn default() -> Self {
        // Seed with fcc Cu (a = 3.61) as its asymmetric unit — one Cu atom plus
        // space group Fm-3m (No. 225) — so "Build feff.inp" exercises the
        // space-group expansion and yields the canonical 12-neighbour shell.
        Self {
            title: "copper (fcc)".to_owned(),
            a: 3.61,
            b: 3.61,
            c: 3.61,
            alpha: 90.0,
            beta: 90.0,
            gamma: 90.0,
            edge: Edge::K,
            space_group: 225,
            cluster_size: 6.0,
            absorber: 0,
            sites: vec![SiteRow {
                element: "Cu".into(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }],
            status: None,
        }
    }
}

impl AtomsTab {
    /// Render the Atoms tab. On "Build feff.inp" the generated input is written
    /// into `feff_inp` and [`AtomsAction::BuiltFeffInp`] is returned.
    pub fn ui(&mut self, ui: &mut egui::Ui, feff_inp: &mut String) -> Option<AtomsAction> {
        let mut action = None;
        ui.horizontal(|ui| {
            ui.heading("Atoms");
            ui.add_space(8.0);
            ui.weak("Crystal cell + space group → feff.inp (enter the asymmetric unit)");
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Title");
                ui.add(egui::TextEdit::singleline(&mut self.title).desired_width(280.0));
            });

            ui.add_space(4.0);
            ui.strong("Lattice (Å, degrees)");
            egui::Grid::new("atoms_lattice")
                .striped(true)
                .show(ui, |ui| {
                    ui.label("a");
                    ui.add(
                        egui::DragValue::new(&mut self.a)
                            .speed(0.01)
                            .range(0.1..=100.0),
                    );
                    ui.label("b");
                    ui.add(
                        egui::DragValue::new(&mut self.b)
                            .speed(0.01)
                            .range(0.1..=100.0),
                    );
                    ui.label("c");
                    ui.add(
                        egui::DragValue::new(&mut self.c)
                            .speed(0.01)
                            .range(0.1..=100.0),
                    );
                    ui.end_row();
                    ui.label("α");
                    ui.add(
                        egui::DragValue::new(&mut self.alpha)
                            .speed(0.1)
                            .range(1.0..=179.0),
                    );
                    ui.label("β");
                    ui.add(
                        egui::DragValue::new(&mut self.beta)
                            .speed(0.1)
                            .range(1.0..=179.0),
                    );
                    ui.label("γ");
                    ui.add(
                        egui::DragValue::new(&mut self.gamma)
                            .speed(0.1)
                            .range(1.0..=179.0),
                    );
                    ui.end_row();
                });

            ui.add_space(6.0);
            ui.strong("Sites (fractional coordinates)");
            let mut remove: Option<usize> = None;
            egui::Grid::new("atoms_sites").striped(true).show(ui, |ui| {
                ui.label("abs");
                ui.label("element");
                ui.label("x");
                ui.label("y");
                ui.label("z");
                ui.label("");
                ui.end_row();
                for (i, row) in self.sites.iter_mut().enumerate() {
                    ui.radio_value(&mut self.absorber, i, "");
                    ui.add(egui::TextEdit::singleline(&mut row.element).desired_width(44.0));
                    ui.add(
                        egui::DragValue::new(&mut row.x)
                            .speed(0.001)
                            .range(-1.0..=2.0),
                    );
                    ui.add(
                        egui::DragValue::new(&mut row.y)
                            .speed(0.001)
                            .range(-1.0..=2.0),
                    );
                    ui.add(
                        egui::DragValue::new(&mut row.z)
                            .speed(0.001)
                            .range(-1.0..=2.0),
                    );
                    if ui.button("✕").clicked() {
                        remove = Some(i);
                    }
                    ui.end_row();
                }
            });
            if let Some(i) = remove {
                self.sites.remove(i);
                if self.absorber >= self.sites.len() {
                    self.absorber = 0;
                }
            }
            if ui.button("➕ Add site").clicked() {
                self.sites.push(SiteRow {
                    element: "O".into(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                });
            }

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label("Space group (No.)");
                ui.add(egui::DragValue::new(&mut self.space_group).range(1..=230))
                    .on_hover_text("International number 1–230; 1 = P1 (no expansion)");
            });
            ui.horizontal(|ui| {
                ui.label("Edge");
                for (e, lbl) in EDGES {
                    ui.selectable_value(&mut self.edge, e, lbl);
                }
                ui.add_space(12.0);
                ui.label("Cluster size (Å)");
                ui.add(
                    egui::DragValue::new(&mut self.cluster_size)
                        .speed(0.1)
                        .range(1.0..=12.0),
                );
            });

            ui.add_space(8.0);
            if ui.add(egui::Button::new("Build feff.inp")).clicked() {
                action = self.build(feff_inp);
            }
            match &self.status {
                Some(Ok(msg)) => {
                    ui.colored_label(GREEN, msg);
                }
                Some(Err(e)) => {
                    ui.colored_label(RED, e);
                }
                None => {}
            }
        });
        action
    }

    /// Expand the asymmetric unit by the space group, then build the cluster and
    /// write its `feff.inp` into `feff_inp`.
    fn build(&mut self, feff_inp: &mut String) -> Option<AtomsAction> {
        if self.sites.is_empty() {
            self.status = Some(Err("Add at least one site.".to_owned()));
            return None;
        }
        let asym = Crystal {
            lattice: Lattice {
                a: self.a,
                b: self.b,
                c: self.c,
                alpha: self.alpha,
                beta: self.beta,
                gamma: self.gamma,
            },
            sites: self
                .sites
                .iter()
                .map(|r| Site::new(r.element.trim(), r.x, r.y, r.z))
                .collect(),
        };
        // Expand the asymmetric unit to the full cell.
        let full = match asym.expand(self.space_group) {
            Ok(f) => f,
            Err(e) => {
                self.status = Some(Err(format!("Expand failed: {e}")));
                return None;
            }
        };
        // The chosen absorber is an asymmetric-unit site; expansion reorders the
        // list, so locate its image (same element + wrapped position) in the
        // expanded cell.
        let asite = &self.sites[self.absorber.min(self.sites.len() - 1)];
        let target = [wrap01(asite.x), wrap01(asite.y), wrap01(asite.z)];
        let abs_idx = full
            .sites
            .iter()
            .position(|s| {
                s.element.eq_ignore_ascii_case(asite.element.trim())
                    && (0..3).all(|d| {
                        let mut dd = s.frac[d] - target[d];
                        dd -= dd.round();
                        dd.abs() < 1.0e-3
                    })
            })
            .unwrap_or(0);

        match full.cluster(abs_idx, self.cluster_size, self.edge) {
            Ok(mut cluster) => {
                cluster.title = self.title.clone();
                *feff_inp = cluster.to_feff_inp();
                self.status = Some(Ok(format!(
                    "SG {} expanded {} site(s) → {} in cell; built feff.inp: {} atoms, {} potentials.",
                    self.space_group,
                    self.sites.len(),
                    full.sites.len(),
                    cluster.atoms.len(),
                    cluster.potentials.len()
                )));
                Some(AtomsAction::BuiltFeffInp)
            }
            Err(e) => {
                self.status = Some(Err(format!("Build failed: {e}")));
                None
            }
        }
    }
}

/// Wrap a fractional coordinate into `[0, 1)`.
fn wrap01(x: f64) -> f64 {
    let v = x - x.floor();
    if v >= 1.0 { 0.0 } else { v }
}

// ---------------------------------------------------------------------------
// Feff tab: edit feff.inp + run FEFF
// ---------------------------------------------------------------------------

/// Which FEFF backend to run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendSel {
    /// In-process FEFF10 (no external executables).
    Feff10,
    /// External FEFF8L subprocess pipeline (`FEFF8L_DIR`/`PATH`).
    Feff8l,
}

impl BackendSel {
    fn to_engine(self) -> feffrun::Backend {
        match self {
            BackendSel::Feff10 => feffrun::Backend::Feff10,
            BackendSel::Feff8l => feffrun::Backend::Feff8l,
        }
    }
}

/// A finished run's summary (the bits we display).
struct RunSummary {
    workdir: PathBuf,
    dat_names: Vec<String>,
}

/// Signals the app should act after the Feff tab renders (the original Feff
/// form's "View Structure" and "Exit" buttons, which the tab itself can't honor
/// — they touch the app-owned Plot Sites window / the viewport).
pub enum FeffAction {
    /// Open the Plot Sites 3D cluster viewer (original "View Structure").
    ViewStructure,
    /// Close the application (original "Exit").
    Exit,
}

/// The **Feff** tab state: the backend choice and the in-flight/last run.
pub struct FeffTab {
    backend: BackendSel,
    running: bool,
    rx: Option<Receiver<Result<RunSummary, String>>>,
    last: Option<Result<RunSummary, String>>,
}

impl Default for FeffTab {
    fn default() -> Self {
        Self {
            backend: BackendSel::Feff10,
            running: false,
            rx: None,
            last: None,
        }
    }
}

impl FeffTab {
    /// Render the Feff tab; edits `feff_inp` in place and can run FEFF. Returns a
    /// [`FeffAction`] for the buttons the app must service (View Structure, Exit).
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        feff_inp: &mut String,
        work_dir: Option<&Path>,
    ) -> Option<FeffAction> {
        self.poll();
        let mut action = None;

        ui.horizontal(|ui| {
            ui.heading("Feff");
            ui.add_space(8.0);
            ui.weak("Edit feff.inp and run FEFF → feffNNNN.dat");
        });
        ui.separator();

        // Original Feff form (그림 1-2-5) button row: Exit / Select Feff Version
        // / View Structure / Execute, plus our Load/Save for the input file.
        ui.horizontal(|ui| {
            if ui.button("Load…").clicked()
                && let Some(text) = load_feff_inp(work_dir)
            {
                *feff_inp = text;
            }
            if ui.button("Save…").clicked() {
                save_feff_inp(feff_inp, work_dir);
            }
            ui.separator();
            ui.label("Feff version");
            ui.selectable_value(&mut self.backend, BackendSel::Feff10, "FEFF10 (in-process)");
            ui.selectable_value(&mut self.backend, BackendSel::Feff8l, "FEFF8L (external)");
            ui.separator();
            if ui
                .add_enabled(
                    !feff_inp.trim().is_empty(),
                    egui::Button::new("View Structure"),
                )
                .clicked()
            {
                action = Some(FeffAction::ViewStructure);
            }
            let can_run = !self.running && !feff_inp.trim().is_empty();
            if ui
                .add_enabled(can_run, egui::Button::new("Run FEFF"))
                .clicked()
            {
                self.start_run(feff_inp, work_dir);
            }
            ui.separator();
            if ui.button("Exit").clicked() {
                action = Some(FeffAction::Exit);
            }
            if self.running {
                ui.spinner();
                ui.label("running…");
                ui.ctx().request_repaint();
            }
        });

        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("feff_inp_editor")
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(feff_inp)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(24),
                );
            });

        ui.separator();
        match &self.last {
            Some(Ok(s)) => {
                ui.colored_label(
                    GREEN,
                    format!(
                        "Ran OK in {} — {} feffNNNN.dat",
                        s.workdir.display(),
                        s.dat_names.len()
                    ),
                );
                if !s.dat_names.is_empty() {
                    ui.label(s.dat_names.join(", "));
                }
            }
            Some(Err(e)) => {
                ui.colored_label(RED, format!("Run failed: {e}"));
            }
            None => {
                ui.weak("No run yet.");
            }
        }

        action
    }

    /// Drain a finished background run, if any.
    fn poll(&mut self) {
        if let Some(rx) = &self.rx
            && let Ok(result) = rx.try_recv()
        {
            self.last = Some(result);
            self.running = false;
            self.rx = None;
        }
    }

    /// Spawn the FEFF run on a worker thread so the UI stays responsive.
    fn start_run(&mut self, feff_inp: &str, work_dir: Option<&Path>) {
        let workdir = work_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::env::temp_dir().join("xafsview_feff"));
        if let Err(e) = std::fs::create_dir_all(&workdir) {
            self.last = Some(Err(format!("create {}: {e}", workdir.display())));
            return;
        }
        let backend = self.backend.to_engine();
        let text = feff_inp.to_owned();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let result = backend
                .run(&text, &workdir)
                .map(|out| RunSummary {
                    workdir: out.workdir,
                    dat_names: out
                        .dat_files
                        .iter()
                        .filter_map(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
                        .collect(),
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.rx = Some(rx);
        self.running = true;
        self.last = None;
    }
}

/// Open a `feff.inp` via a native dialog and return its text.
fn load_feff_inp(work_dir: Option<&Path>) -> Option<String> {
    let mut dlg = rfd::FileDialog::new().add_filter("feff input", &["inp"]);
    if let Some(d) = work_dir {
        dlg = dlg.set_directory(d);
    }
    let path = dlg.pick_file()?;
    std::fs::read_to_string(path).ok()
}

/// Save the current `feff.inp` text via a native dialog.
fn save_feff_inp(text: &str, work_dir: Option<&Path>) {
    let mut dlg = rfd::FileDialog::new()
        .add_filter("feff input", &["inp"])
        .set_file_name("feff.inp");
    if let Some(d) = work_dir {
        dlg = dlg.set_directory(d);
    }
    if let Some(path) = dlg.save_file() {
        let _ = std::fs::write(path, text);
    }
}

// ---------------------------------------------------------------------------
// Plot Sites: 3D cluster viewer
// ---------------------------------------------------------------------------

/// The **Plot Sites** window: a 3D scatter of the `feff.inp` cluster, coloured
/// by FEFF potential index.
pub struct PlotSitesWindow {
    pub open: bool,
    scene: SceneWidget,
    /// The `feff.inp` text the current scene was built from (rebuild on change).
    source: String,
    /// `(ipot, tag, count)` for the legend.
    legend: Vec<(usize, String, usize)>,
    error: Option<String>,
}

impl PlotSitesWindow {
    /// Build the window with its own 3D scene (`Scene3dId` 0).
    pub fn new(render_state: &RenderState) -> Self {
        Self {
            open: false,
            scene: SceneWidget::new(render_state, 0),
            source: String::new(),
            legend: Vec::new(),
            error: None,
        }
    }

    /// Render the window over the shared `feff_inp`, rebuilding the scene when
    /// the text changes.
    pub fn show(&mut self, ctx: &egui::Context, render_state: &RenderState, feff_inp: &str) {
        if !self.open {
            return;
        }
        if self.source != feff_inp {
            self.rebuild(render_state, feff_inp);
            self.source = feff_inp.to_owned();
        }

        let mut open = self.open;
        crate::window::detached(
            ctx,
            "plot_sites",
            "Plot Sites (3D cluster)",
            &mut open,
            [640.0, 560.0],
            |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Reset view").clicked() {
                        self.scene.reset_camera();
                    }
                    siplot::viewpoint_menu(ui, &mut self.scene);
                    ui.separator();
                    if let Some(e) = &self.error {
                        ui.colored_label(RED, e);
                    } else {
                        let n: usize = self.legend.iter().map(|(_, _, c)| c).sum();
                        ui.weak(format!("{n} atoms"));
                    }
                });
                ui.separator();
                self.scene.show(ui);
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    for (ipot, tag, count) in &self.legend {
                        ui.label(format!("ipot {ipot} = {tag} ({count})"));
                        ui.add_space(8.0);
                    }
                });
            },
        );
        self.open = open;
    }

    /// Parse `feff_inp` and upload a fresh 3D scene.
    fn rebuild(&mut self, render_state: &RenderState, feff_inp: &str) {
        self.error = None;
        self.legend.clear();
        let parsed = match FeffInp::parse(feff_inp) {
            Ok(p) => p,
            Err(e) => {
                self.error = Some(e.to_string());
                self.scene
                    .set_geometry(render_state, Scene3dGeometry::new());
                return;
            }
        };
        if parsed.atoms.is_empty() {
            self.error = Some("no ATOMS in feff.inp".to_owned());
            self.scene
                .set_geometry(render_state, Scene3dGeometry::new());
            return;
        }

        let max_ipot = parsed.atoms.iter().map(|a| a.ipot).max().unwrap_or(0);
        // Legend: count per (ipot, tag), in ipot order.
        for ipot in 0..=max_ipot {
            let count = parsed.atoms.iter().filter(|a| a.ipot == ipot).count();
            if count == 0 {
                continue;
            }
            let tag = parsed
                .atoms
                .iter()
                .find(|a| a.ipot == ipot)
                .map(|a| a.tag.clone())
                .unwrap_or_default();
            self.legend.push((ipot, tag, count));
        }

        // Split absorber (ipot 0) from the rest so it can be emphasised.
        let (mut ax, mut ay, mut az, mut av) = (vec![], vec![], vec![], vec![]);
        let (mut nx, mut ny, mut nz, mut nv) = (vec![], vec![], vec![], vec![]);
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for atom in &parsed.atoms {
            let p = [atom.xyz[0] as f32, atom.xyz[1] as f32, atom.xyz[2] as f32];
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
            if atom.ipot == 0 {
                ax.push(p[0]);
                ay.push(p[1]);
                az.push(p[2]);
                av.push(0.0);
            } else {
                nx.push(p[0]);
                ny.push(p[1]);
                nz.push(p[2]);
                nv.push(atom.ipot as f64);
            }
        }

        let vmax = max_ipot.max(1) as f64;
        let cmap = || Colormap::new(ColormapName::Viridis, 0.0, vmax);
        let mut geometry = Scene3dGeometry::new();
        if !nx.is_empty() {
            Scatter3D::new()
                .with_data(&nx, &ny, &nz, &nv)
                .with_colormap(cmap())
                .with_marker(PointMarker::Circle)
                .with_size(10.0)
                .append_to(&mut geometry);
        }
        if !ax.is_empty() {
            // The absorber, larger and as a diamond, to mark the cluster centre.
            Scatter3D::new()
                .with_data(&ax, &ay, &az, &av)
                .with_colormap(cmap())
                .with_marker(PointMarker::Diamond)
                .with_size(18.0)
                .append_to(&mut geometry);
        }

        // Pad bounds so markers near the hull are not clipped.
        let pad = 0.5;
        let bounds = (
            Vec3::new(lo[0] - pad, lo[1] - pad, lo[2] - pad),
            Vec3::new(hi[0] + pad, hi[1] + pad, hi[2] + pad),
        );
        self.scene.set_bounds(render_state, bounds);
        self.scene.set_geometry(render_state, geometry);
    }
}
