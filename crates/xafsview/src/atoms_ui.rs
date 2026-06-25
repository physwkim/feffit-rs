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
use feffit::feffdat::FeffDatFile;
use feffit::feffinp::{Crystal, Edge, FeffInp, Lattice, Site};
use siplot::{Colormap, ColormapName, PointMarker, Scatter3D, Scene3dGeometry, SceneWidget, Vec3};

use crate::plot::{GREEN, RED};

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
    /// Close the application (the original Atoms form's "Exit").
    Exit,
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

        egui::CentralPanel::default().show_inside(ui, |ui| {
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
                        if crate::widgets::delete_box(ui).clicked() {
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

                // Execute / Exit row + build status, directly below the cell
                // inputs (그림 1-2-4) — no pinned-bottom gap. Reload (re-read
                // atoms.inp from disk) has no file source for the structured
                // builder, so it is omitted per the functional-only field rule.
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if crate::widgets::exit(ui, crate::widgets::ROW_BTN).clicked() {
                        action = Some(AtomsAction::Exit);
                    }
                    if crate::widgets::primary(ui, "Execute", crate::widgets::ROW_BTN, true)
                        .on_hover_text("Build feff.inp from the cell")
                        .clicked()
                    {
                        action = self.build(feff_inp);
                    }
                });
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
    fn to_engine(self) -> feffit::feffrun::Backend {
        match self {
            BackendSel::Feff10 => feffit::feffrun::Backend::Feff10,
            BackendSel::Feff8l => feffit::feffrun::Backend::Feff8l,
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

        // Top: the input/output file handling + "Select Feff Version" selector
        // (그림 1-2-5: Input file / Output path rows + version ring). The action
        // buttons live in the bottom row below, as in the original form.
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
            ui.label("Select Feff Version");
            ui.selectable_value(&mut self.backend, BackendSel::Feff10, "FEFF10 (in-process)");
            ui.selectable_value(&mut self.backend, BackendSel::Feff8l, "FEFF8L (external)");
        });

        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("feff_inp_editor")
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(feff_inp)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(22),
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

        // Bottom action row (그림 1-2-5: Exit / Select Feff Version / View
        // Structure / Execute), uniform width, with "Execute" (= Run FEFF) as the
        // amber primary action.
        ui.separator();
        ui.horizontal(|ui| {
            if crate::widgets::exit(ui, crate::widgets::ROW_BTN).clicked() {
                action = Some(FeffAction::Exit);
            }
            let has_inp = !feff_inp.trim().is_empty();
            if crate::widgets::action_enabled(
                ui,
                "View Structure",
                crate::widgets::ROW_BTN,
                has_inp,
            )
            .clicked()
            {
                action = Some(FeffAction::ViewStructure);
            }
            let can_run = !self.running && has_inp;
            if crate::widgets::primary(ui, "Execute", crate::widgets::ROW_BTN, can_run)
                .on_hover_text("Run FEFF → feffNNNN.dat")
                .clicked()
            {
                self.start_run(feff_inp, work_dir);
            }
            if self.running {
                ui.spinner();
                ui.label("running…");
                ui.ctx().request_repaint();
            }
        });

        action
    }

    /// Full paths to the `feffNNNN.dat` files produced by the last successful
    /// run, for the Plot Sites "View Path" overlay. Empty if no run succeeded.
    pub fn last_path_files(&self) -> Vec<PathBuf> {
        match &self.last {
            Some(Ok(s)) => s
                .dat_names
                .iter()
                .filter(|n| n.starts_with("feff") && n.ends_with(".dat"))
                .map(|n| s.workdir.join(n))
                .collect(),
            _ => Vec::new(),
        }
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

/// Two shell radii closer than this (Å) are merged into one coordination shell.
const SHELL_TOL: f32 = 0.05;

/// A cached cluster atom: position, FEFF potential index, and distance from the
/// absorber (for shell grouping / filtering).
struct ClusterAtom {
    pos: [f32; 3],
    ipot: usize,
    r: f32,
}

/// The **Plot Sites** window: a 3D scatter of the `feff.inp` cluster coloured by
/// FEFF potential index, with shell filtering (그림 1-4-1 "Shell #") and a
/// "View Path" overlay that draws a chosen `feffNNNN.dat` scattering path as a
/// red dotted route through its atoms, with its geometry table (paths.dat).
pub struct PlotSitesWindow {
    pub open: bool,
    scene: SceneWidget,
    /// The `feff.inp` text the cached cluster was parsed from (re-parse on change).
    source: String,
    /// `(ipot, tag, count)` for the legend.
    legend: Vec<(usize, String, usize)>,
    error: Option<String>,
    /// Cluster atoms (cached on `feff.inp` change), for shell filtering + bounds.
    atoms: Vec<ClusterAtom>,
    /// Absorber (ipot 0) position in cluster coordinates.
    absorber: [f32; 3],
    /// Highest potential index (sets the colormap range).
    max_ipot: usize,
    /// Sorted unique shell radii (Å) from the absorber (≤ 10, as the original).
    shells: Vec<f32>,
    /// How many shells to draw (1..=`shells.len()`); `all_shells` overrides.
    shell_count: usize,
    /// Draw every shell regardless of `shell_count`.
    all_shells: bool,
    /// Atom marker size.
    point_size: f32,
    /// Parsed `feffNNNN.dat` path geometries from the last FEFF run (View Path).
    paths: Vec<FeffDatFile>,
    /// The path-file list `paths` was parsed from (re-parse on change).
    paths_source: Vec<PathBuf>,
    /// Overlay the selected scattering path as a red dotted route.
    view_path: bool,
    /// Selected path index into `paths`.
    path_idx: usize,
    /// The GPU geometry needs rebuilding (a control or the source changed).
    dirty: bool,
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
            atoms: Vec::new(),
            absorber: [0.0; 3],
            max_ipot: 0,
            shells: Vec::new(),
            shell_count: 1,
            all_shells: true,
            point_size: 10.0,
            paths: Vec::new(),
            paths_source: Vec::new(),
            view_path: false,
            path_idx: 0,
            dirty: false,
        }
    }

    /// Render the window over the shared `feff_inp` and the last FEFF run's
    /// `feffNNNN.dat` files. Re-parses the cluster / paths when either changes
    /// and rebuilds the GPU scene when a control or the source changes.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        render_state: &RenderState,
        feff_inp: &str,
        path_files: &[PathBuf],
    ) {
        if !self.open {
            return;
        }
        if self.source != feff_inp {
            self.parse_cluster(feff_inp);
            self.source = feff_inp.to_owned();
            self.dirty = true;
        }
        if self.paths_source.as_slice() != path_files {
            self.parse_paths(path_files);
            self.paths_source = path_files.to_vec();
            self.dirty = true;
        }

        let mut open = self.open;
        crate::window::detached(
            ctx,
            "plot_sites",
            "Plot Sites (3D cluster)",
            &mut open,
            [680.0, 620.0],
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
                        ui.weak(format!("{} atoms", self.atoms.len()));
                    }
                });

                // Marker size + shell selection (그림 1-4-1: PointSize / Shell #).
                ui.horizontal(|ui| {
                    ui.label("Point size");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.point_size)
                                .range(2.0..=30.0)
                                .speed(0.2),
                        )
                        .changed()
                    {
                        self.dirty = true;
                    }
                    ui.separator();
                    if ui.checkbox(&mut self.all_shells, "All shells").changed() {
                        self.dirty = true;
                    }
                    if !self.all_shells && !self.shells.is_empty() {
                        ui.label("Shell #");
                        let nsh = self.shells.len();
                        if ui
                            .add(egui::DragValue::new(&mut self.shell_count).range(1..=nsh))
                            .changed()
                        {
                            self.dirty = true;
                        }
                        let i = self.shell_count.clamp(1, nsh) - 1;
                        ui.weak(format!("≤ {:.3} Å", self.shells[i]));
                    }
                });

                // View Path overlay (그림 1-4-2): pick a feffNNNN.dat path and
                // trace it as a red dotted route through its atoms.
                if self.paths.is_empty() {
                    ui.weak("Run FEFF (Feff tab) to enable View Path.");
                } else {
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut self.view_path, "View Path").changed() {
                            self.dirty = true;
                        }
                        if self.view_path {
                            ui.label("Path #");
                            let last = self.paths.len() - 1;
                            if ui
                                .add(egui::DragValue::new(&mut self.path_idx).range(0..=last))
                                .changed()
                            {
                                self.dirty = true;
                            }
                            if let Some(p) = self.paths.get(self.path_idx) {
                                ui.weak(format!(
                                    "reff={:.3} Å · nleg={} · degen={} · {}",
                                    p.reff,
                                    p.nleg,
                                    p.degen,
                                    path_label(p)
                                ));
                            }
                        }
                    });
                }

                if self.dirty {
                    self.build_scene(render_state);
                    self.dirty = false;
                }

                ui.separator();
                self.scene.show(ui);
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    for (ipot, tag, count) in &self.legend {
                        ui.label(format!("ipot {ipot} = {tag} ({count})"));
                        ui.add_space(8.0);
                    }
                });

                // The selected path's geometry, as the original's paths.dat table.
                if self.view_path
                    && let Some(p) = self.paths.get(self.path_idx)
                {
                    egui::CollapsingHeader::new("Scattering path geometry (paths.dat)")
                        .default_open(false)
                        .show(ui, |ui| {
                            egui::Grid::new("path_geom")
                                .num_columns(5)
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.strong("atom");
                                    ui.strong("ipot");
                                    ui.strong("x");
                                    ui.strong("y");
                                    ui.strong("z");
                                    ui.end_row();
                                    for g in &p.geom {
                                        ui.label(&g.label);
                                        ui.monospace(format!("{}", g.ipot));
                                        ui.monospace(format!("{:.4}", g.x));
                                        ui.monospace(format!("{:.4}", g.y));
                                        ui.monospace(format!("{:.4}", g.z));
                                        ui.end_row();
                                    }
                                });
                        });
                }
            },
        );
        self.open = open;
    }

    /// Parse `feff_inp` into the cached cluster (`atoms`, `absorber`, `shells`,
    /// `legend`). Sets `error` and clears the cluster on a parse failure.
    fn parse_cluster(&mut self, feff_inp: &str) {
        self.error = None;
        self.legend.clear();
        self.atoms.clear();
        self.shells.clear();
        self.absorber = [0.0; 3];
        self.max_ipot = 0;
        let parsed = match FeffInp::parse(feff_inp) {
            Ok(p) => p,
            Err(e) => {
                self.error = Some(e.to_string());
                return;
            }
        };
        if parsed.atoms.is_empty() {
            self.error = Some("no ATOMS in feff.inp".to_owned());
            return;
        }

        // Absorber (ipot 0) anchors the shell radii; default to the origin.
        if let Some(a) = parsed.atoms.iter().find(|a| a.ipot == 0) {
            self.absorber = [a.xyz[0] as f32, a.xyz[1] as f32, a.xyz[2] as f32];
        }
        self.max_ipot = parsed.atoms.iter().map(|a| a.ipot).max().unwrap_or(0);

        // Legend: count per (ipot, tag), in ipot order.
        for ipot in 0..=self.max_ipot {
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

        // Cache atoms with their distance from the absorber, then derive the
        // sorted unique shell radii (merging near-equal radii; ≤ 10 shells, as
        // the original limits the Shell # selector).
        let mut radii: Vec<f32> = Vec::new();
        for a in &parsed.atoms {
            let pos = [a.xyz[0] as f32, a.xyz[1] as f32, a.xyz[2] as f32];
            let (dx, dy, dz) = (
                pos[0] - self.absorber[0],
                pos[1] - self.absorber[1],
                pos[2] - self.absorber[2],
            );
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            self.atoms.push(ClusterAtom {
                pos,
                ipot: a.ipot,
                r,
            });
            if a.ipot != 0 {
                radii.push(r);
            }
        }
        self.shells = coordination_shells(&radii);
        self.shell_count = self.shells.len().max(1);
    }

    /// Parse the last run's `feffNNNN.dat` files into path geometries; skips any
    /// that fail to read. Keeps `path_idx` in range.
    fn parse_paths(&mut self, path_files: &[PathBuf]) {
        self.paths.clear();
        for pf in path_files {
            if let Ok(f) = FeffDatFile::from_path(pf) {
                self.paths.push(f);
            }
        }
        if self.path_idx >= self.paths.len() {
            self.path_idx = 0;
        }
        if self.paths.is_empty() {
            self.view_path = false;
        }
    }

    /// Rebuild the GPU scene from the cached cluster (with the active shell
    /// filter) plus the View Path overlay.
    fn build_scene(&mut self, render_state: &RenderState) {
        if self.atoms.is_empty() {
            self.scene
                .set_geometry(render_state, Scene3dGeometry::new());
            return;
        }

        // Shell radius cutoff (absorber is always drawn).
        let limit = if self.all_shells || self.shells.is_empty() {
            f32::MAX
        } else {
            let i = self.shell_count.clamp(1, self.shells.len()) - 1;
            self.shells[i] + SHELL_TOL
        };

        let (mut ax, mut ay, mut az, mut av) = (vec![], vec![], vec![], vec![]);
        let (mut nx, mut ny, mut nz, mut nv) = (vec![], vec![], vec![], vec![]);
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for atom in &self.atoms {
            if atom.ipot != 0 && atom.r > limit {
                continue;
            }
            expand_bounds(atom.pos, &mut lo, &mut hi);
            if atom.ipot == 0 {
                ax.push(atom.pos[0]);
                ay.push(atom.pos[1]);
                az.push(atom.pos[2]);
                av.push(0.0);
            } else {
                nx.push(atom.pos[0]);
                ny.push(atom.pos[1]);
                nz.push(atom.pos[2]);
                nv.push(atom.ipot as f64);
            }
        }

        let vmax = self.max_ipot.max(1) as f64;
        let cmap = || Colormap::new(ColormapName::Viridis, 0.0, vmax);
        let mut geometry = Scene3dGeometry::new();
        if !nx.is_empty() {
            Scatter3D::new()
                .with_data(&nx, &ny, &nz, &nv)
                .with_colormap(cmap())
                .with_marker(PointMarker::Circle)
                .with_size(self.point_size)
                .append_to(&mut geometry);
        }
        if !ax.is_empty() {
            // The absorber, larger and as a diamond, to mark the cluster centre.
            Scatter3D::new()
                .with_data(&ax, &ay, &az, &av)
                .with_colormap(cmap())
                .with_marker(PointMarker::Diamond)
                .with_size(self.point_size * 1.8)
                .append_to(&mut geometry);
        }

        // View Path: trace the selected path as a red dotted route (interpolated
        // points along each leg), a closed loop absorber → scatterers → absorber.
        if self.view_path
            && let Some(p) = self.paths.get(self.path_idx)
        {
            let c = self.absorber;
            let mut route: Vec<[f32; 3]> = Vec::with_capacity(p.geom.len() + 2);
            route.push(c);
            for g in &p.geom {
                route.push([c[0] + g.x as f32, c[1] + g.y as f32, c[2] + g.z as f32]);
            }
            route.push(c);

            let (mut rx, mut ry, mut rz) = (vec![], vec![], vec![]);
            const STEPS: usize = 16;
            for seg in route.windows(2) {
                let (a, b) = (seg[0], seg[1]);
                for s in 0..STEPS {
                    let t = s as f32 / STEPS as f32;
                    let pt = [
                        a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t,
                    ];
                    expand_bounds(pt, &mut lo, &mut hi);
                    rx.push(pt[0]);
                    ry.push(pt[1]);
                    rz.push(pt[2]);
                }
            }
            if let Some(red) = Colormap::from_colors(
                &[[0xd6, 0x27, 0x28, 0xff], [0xd6, 0x27, 0x28, 0xff]],
                0.0,
                1.0,
            ) {
                let vals = vec![0.5_f64; rx.len()];
                Scatter3D::new()
                    .with_data(&rx, &ry, &rz, &vals)
                    .with_colormap(red)
                    .with_marker(PointMarker::Circle)
                    .with_size((self.point_size * 0.45).max(2.0))
                    .append_to(&mut geometry);
            }
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

/// Grow `lo`/`hi` to include point `p`.
fn expand_bounds(p: [f32; 3], lo: &mut [f32; 3], hi: &mut [f32; 3]) {
    for d in 0..3 {
        lo[d] = lo[d].min(p[d]);
        hi[d] = hi[d].max(p[d]);
    }
}

/// Sorted unique coordination-shell radii from per-atom distances: radii closer
/// than [`SHELL_TOL`] merge into one shell, and the list is capped at 10 (the
/// original limits the Shell # selector to ten shells).
fn coordination_shells(radii: &[f32]) -> Vec<f32> {
    let mut sorted = radii.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mut shells: Vec<f32> = Vec::new();
    for r in sorted {
        if shells.last().is_none_or(|&last| r - last > SHELL_TOL) {
            shells.push(r);
        }
    }
    shells.truncate(10);
    shells
}

/// The path's leg-atom labels joined as a route, e.g. `"Cu – O – Cu"`.
fn path_label(p: &FeffDatFile) -> String {
    p.geom
        .iter()
        .map(|g| g.label.as_str())
        .collect::<Vec<_>>()
        .join(" – ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use feffit::feffdat::GeomAtom;

    #[test]
    fn coordination_shells_merge_near_equal_and_cap() {
        // Three near-equal radii (within SHELL_TOL) collapse to one shell each;
        // distinct radii stay separate; the merged radius is the first seen.
        let shells = coordination_shells(&[2.55, 2.56, 2.54, 3.61, 3.62, 4.42]);
        assert_eq!(shells.len(), 3);
        assert!((shells[0] - 2.54).abs() < 1e-6, "{shells:?}");
        assert!((shells[1] - 3.61).abs() < 1e-6, "{shells:?}");
        assert!((shells[2] - 4.42).abs() < 1e-6, "{shells:?}");

        // More than ten distinct shells are capped at ten.
        let many: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(coordination_shells(&many).len(), 10);

        // No atoms → no shells.
        assert!(coordination_shells(&[]).is_empty());
    }

    #[test]
    fn path_label_joins_leg_atoms() {
        let mut f = FeffDatFile::default();
        let leg = |label: &str| GeomAtom {
            label: label.to_owned(),
            iz: 0,
            ipot: 0,
            mass: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        f.geom = vec![leg("Cu"), leg("O"), leg("Cu")];
        assert_eq!(path_label(&f), "Cu – O – Cu");
    }
}
