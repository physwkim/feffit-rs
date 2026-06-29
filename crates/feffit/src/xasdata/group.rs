//! The in-memory model shared by the GUI and the headless batch drivers.
//!
//! [`XasGroup`] holds **one spectrum end-to-end** — from the raw `mu(E)` through
//! normalization, AUTOBK background removal, and the Fourier transform — mirror‑
//! ing a larch `Group`. Derived stages are `Option`, populated as the user runs
//! each step; an absent field means "not computed yet". The struct deliberately
//! does *not* embed fit results: keeping FEFFIT output out of here lets `xasdata`
//! stay free of the heavy fitting crates, so the data model is lightweight and
//! unit-testable. The GUI keeps fit state alongside the group, in its own state.
//!
//! [`Session`] is the data half of the application state: the loaded groups, the
//! current selection, and the configured working [`Folders`]. The eframe app
//! wraps a `Session` and adds GUI-only state (the plot, active tab, dialogs).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One XAS spectrum and every reduction stage computed from it.
///
/// Invariant: `energy` and `mu` always have the same length once a group is
/// built. Each derived vector (when `Some`) matches the length of the grid it
/// lives on — `pre_edge`/`post_edge`/`norm`/`flat`/`bkg` on the `energy` grid,
/// `chi` on the `k` grid, and the `chir_*` magnitudes/parts on the `r` grid.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct XasGroup {
    /// Short display name, normally the source file stem.
    pub label: String,
    /// Source file this group was read from, if any.
    pub filename: Option<PathBuf>,

    // --- raw spectrum -------------------------------------------------------
    /// Photon energy, eV.
    pub energy: Vec<f64>,
    /// Absorption coefficient `mu(E)` (the working spectrum).
    pub mu: Vec<f64>,

    // --- source columns kept for re-deriving `mu` ---------------------------
    /// Incident-beam monitor `I0`, if `mu` was built from raw columns.
    pub i0: Option<Vec<f64>>,
    /// Transmitted-beam monitor `It` (transmission channel).
    pub it: Option<Vec<f64>>,
    /// Fluorescence monitor `If` (fluorescence channel).
    pub iflu: Option<Vec<f64>>,
    /// Reference-foil transmitted monitor `Iref`, for energy alignment.
    pub iref: Option<Vec<f64>>,

    // --- normalization (pre_edge) ------------------------------------------
    /// Edge energy `E0`, eV.
    pub e0: Option<f64>,
    /// Edge step (jump) used to normalize.
    pub edge_step: Option<f64>,
    /// Pre-edge line evaluated on `energy`.
    pub pre_edge: Option<Vec<f64>>,
    /// Post-edge normalization polynomial evaluated on `energy`.
    pub post_edge: Option<Vec<f64>>,
    /// Edge-step–normalized `mu`.
    pub norm: Option<Vec<f64>>,
    /// Flattened normalized `mu` (post-edge curvature removed above `E0`).
    pub flat: Option<Vec<f64>>,
    /// First derivative `d(mu)/dE` on the `energy` grid (for derivative plots).
    pub dmude: Option<Vec<f64>>,

    // --- background removal (autobk) ---------------------------------------
    /// Smooth post-edge background `mu0(E)` on the `energy` grid.
    pub bkg: Option<Vec<f64>>,
    /// 1σ uncertainty in `bkg` on the `energy` grid (for an uncertainty band).
    pub delta_bkg: Option<Vec<f64>>,
    /// Photoelectron wavenumber grid, Å⁻¹.
    pub k: Option<Vec<f64>>,
    /// EXAFS `chi(k)` on the `k` grid (not k-weighted).
    pub chi: Option<Vec<f64>>,
    /// 1σ uncertainty in `chi(k)` on the `k` grid (larch `delta_chi` units).
    pub delta_chi: Option<Vec<f64>>,

    // --- Fourier transform (xftf) ------------------------------------------
    /// Radial grid `R`, Å.
    pub r: Option<Vec<f64>>,
    /// `|chi(R)|`.
    pub chir_mag: Option<Vec<f64>>,
    /// `Re chi(R)`.
    pub chir_re: Option<Vec<f64>>,
    /// `Im chi(R)`.
    pub chir_im: Option<Vec<f64>>,

    // --- reverse Fourier transform (xftr) ----------------------------------
    /// Back-transform wavenumber grid `q`, Å⁻¹ (the Fourier-filtered EXAFS, from
    /// an R-windowed reverse FT of `chi(R)`).
    pub q: Option<Vec<f64>>,
    /// `|chi(q)|`.
    pub chiq_mag: Option<Vec<f64>>,
    /// `Re chi(q)`.
    pub chiq_re: Option<Vec<f64>>,
    /// `Im chi(q)`.
    pub chiq_im: Option<Vec<f64>>,

    // --- provenance (for the output-file headers) --------------------------
    /// Raw comment/header lines of the source file this group was read from,
    /// kept verbatim so the `.chi`/`.dat`/`.fit` writers can echo the original
    /// beamline header into their provenance block. Empty when the group was
    /// not built from a file with a header (e.g. `from_chi`).
    #[serde(default)]
    pub source_header: Vec<String>,
    /// Pre-edge fit range `[pre1, pre2]` (eV, relative to `e0`) used by the last
    /// normalize, recorded for the output provenance header. `None` until
    /// normalize has run.
    #[serde(default)]
    pub pre1: Option<f64>,
    #[serde(default)]
    pub pre2: Option<f64>,
    /// `rbkg` (Å) used by the last AUTOBK, recorded for the output provenance
    /// header. `None` until AUTOBK has run.
    #[serde(default)]
    pub rbkg: Option<f64>,
}

impl XasGroup {
    /// A bare group from an `energy`/`mu` pair, labelled `label`.
    pub fn from_mu(label: impl Into<String>, energy: Vec<f64>, mu: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            energy,
            mu,
            ..Default::default()
        }
    }

    /// A group holding only `χ(k)` (e.g. a loaded FEFF `chi.dat`), labelled
    /// `label`. `energy`/`mu` stay empty; the FT and FEFFIT can still run on the
    /// `k`/`chi` arrays.
    pub fn from_chi(label: impl Into<String>, k: Vec<f64>, chi: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            k: Some(k),
            chi: Some(chi),
            ..Default::default()
        }
    }

    /// Number of points in the raw spectrum.
    pub fn len(&self) -> usize {
        self.energy.len()
    }

    /// Drop every derived reduction stage, leaving only the raw spectrum
    /// (`energy`/`mu` and the source columns). Called after any edit that
    /// changes the raw spectrum — deglitch, trim, smooth — so stale
    /// normalize/AUTOBK/FT results are never shown against the new data; the
    /// user re-runs reduction to repopulate them.
    pub fn clear_derived(&mut self) {
        self.e0 = None;
        self.edge_step = None;
        // Reduction-parameter provenance (recorded by normalize/AUTOBK); the raw
        // `source_header` is kept — it describes the source file, not a stage.
        self.pre1 = None;
        self.pre2 = None;
        self.rbkg = None;
        self.pre_edge = None;
        self.post_edge = None;
        self.norm = None;
        self.flat = None;
        self.dmude = None;
        self.bkg = None;
        self.delta_bkg = None;
        self.k = None;
        self.chi = None;
        self.delta_chi = None;
        self.r = None;
        self.chir_mag = None;
        self.chir_re = None;
        self.chir_im = None;
    }

    /// True when no spectrum has been loaded.
    pub fn is_empty(&self) -> bool {
        self.energy.is_empty()
    }
}

/// Working directories the user configures on the Folders tab.
///
/// Mirrors the original XAFSView project layout: a chosen "Sub base" project
/// root holds the five working folders (`Data`, `Autobk`, `Feffit`, `Atoms`,
/// `Results`). Selecting the sub base creates all five at once and points each
/// field below at `<sub_base>/<Name>`, so file dialogs read from and outputs are
/// written to the right place; each can still be overridden individually.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Folders {
    /// Project root; selecting it creates and fills the five folders below with
    /// `<sub_base>/{Data,Autobk,Feffit,Atoms,Results}`.
    pub sub_base: Option<PathBuf>,
    /// `Data` — raw/reduced data files are read from here.
    pub data_dir: Option<PathBuf>,
    /// `Autobk` — AUTOBK χ(k)/χ(R) outputs are written here.
    pub autobk_dir: Option<PathBuf>,
    /// `Feffit` — FEFFIT data/model (`.dat`/`.fit`) outputs are written here.
    pub feffit_dir: Option<PathBuf>,
    /// `Atoms` — `feff.inp` and the `feffNNNN.dat` path files live here.
    pub atoms_dir: Option<PathBuf>,
    /// `Results` — saved items and exports are written here.
    pub results_dir: Option<PathBuf>,
}

/// The data half of the application state: loaded groups, current selection, and
/// configured folders. Serializable so a whole session can be saved/loaded.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Session {
    /// All loaded spectra, in load order.
    pub groups: Vec<XasGroup>,
    /// Index of the active group in `groups`, if any.
    pub current: Option<usize>,
    /// Configured working directories.
    pub folders: Folders,
}

impl Session {
    /// An empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a group and make it the current selection; returns its index.
    pub fn add_group(&mut self, group: XasGroup) -> usize {
        self.groups.push(group);
        let idx = self.groups.len() - 1;
        self.current = Some(idx);
        idx
    }

    /// Remove the group at `idx` (no-op, returning `None`, when out of range),
    /// keeping `current` valid by construction:
    /// - a selection *after* the removed slot shifts down by one,
    /// - a selection *before* it is unchanged,
    /// - removing the selected group falls to the group that slid into its slot
    ///   (or the new last group when it was last),
    /// - emptying the session clears the selection.
    pub fn remove_group(&mut self, idx: usize) -> Option<XasGroup> {
        if idx >= self.groups.len() {
            return None;
        }
        let removed = self.groups.remove(idx);
        self.current = match self.current {
            None => None,
            Some(_) if self.groups.is_empty() => None,
            Some(c) if c > idx => Some(c - 1),
            Some(c) if c < idx => Some(c),
            // c == idx: the selected group was removed; keep the slot, clamped to
            // the new last index (so a removed *last* group selects its predecessor).
            Some(c) => Some(c.min(self.groups.len() - 1)),
        };
        Some(removed)
    }

    /// The currently selected group, if any.
    pub fn current_group(&self) -> Option<&XasGroup> {
        self.current.and_then(|i| self.groups.get(i))
    }

    /// Mutable access to the currently selected group, if any.
    pub fn current_group_mut(&mut self) -> Option<&mut XasGroup> {
        match self.current {
            Some(i) => self.groups.get_mut(i),
            None => None,
        }
    }

    /// Serialize the whole session to a pretty JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Restore a session from a JSON string written by [`Session::to_json`].
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_group_sets_current() {
        let mut s = Session::new();
        assert!(s.current_group().is_none());
        let i = s.add_group(XasGroup::from_mu("a", vec![1.0, 2.0], vec![0.1, 0.2]));
        assert_eq!(i, 0);
        assert_eq!(s.current, Some(0));
        assert_eq!(s.current_group().unwrap().label, "a");
        let j = s.add_group(XasGroup::from_mu("b", vec![3.0], vec![0.3]));
        assert_eq!(j, 1);
        assert_eq!(s.current_group().unwrap().label, "b");
    }

    #[test]
    fn remove_group_keeps_current_valid_at_every_boundary() {
        let mk = |n: &str| XasGroup::from_mu(n, vec![1.0], vec![0.1]);
        let setup = || {
            let mut s = Session::new();
            s.add_group(mk("a")); // 0
            s.add_group(mk("b")); // 1
            s.add_group(mk("c")); // 2
            s
        };

        // Out of range: no-op, returns None.
        let mut s = setup();
        assert!(s.remove_group(9).is_none());
        assert_eq!(s.groups.len(), 3);

        // Remove BEFORE current (current=2): selection shifts down to 1.
        let mut s = setup();
        s.current = Some(2);
        assert_eq!(s.remove_group(0).unwrap().label, "a");
        assert_eq!(s.current, Some(1));
        assert_eq!(s.current_group().unwrap().label, "c");

        // Remove AFTER current (current=0): selection unchanged.
        let mut s = setup();
        s.current = Some(0);
        s.remove_group(2);
        assert_eq!(s.current, Some(0));
        assert_eq!(s.current_group().unwrap().label, "a");

        // Remove the SELECTED middle group: slot falls to the group that slid in.
        let mut s = setup();
        s.current = Some(1);
        s.remove_group(1);
        assert_eq!(s.current, Some(1));
        assert_eq!(s.current_group().unwrap().label, "c");

        // Remove the SELECTED LAST group: slot clamps to the new last.
        let mut s = setup();
        s.current = Some(2);
        s.remove_group(2);
        assert_eq!(s.current, Some(1));
        assert_eq!(s.current_group().unwrap().label, "b");

        // Remove the only remaining group: selection clears.
        let mut s = Session::new();
        s.add_group(mk("solo"));
        s.remove_group(0);
        assert!(s.groups.is_empty());
        assert_eq!(s.current, None);
    }

    #[test]
    fn group_len_and_empty() {
        let g = XasGroup::from_mu("x", vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]);
        assert_eq!(g.len(), 3);
        assert!(!g.is_empty());
        assert!(XasGroup::default().is_empty());
    }

    #[test]
    fn session_json_roundtrip() {
        let mut s = Session::new();
        let mut g = XasGroup::from_mu("cu", vec![8900.0, 8901.0], vec![0.5, 1.5]);
        g.e0 = Some(8979.0);
        g.edge_step = Some(1.0);
        s.add_group(g);
        s.folders.data_dir = Some(PathBuf::from("/tmp/data"));

        let json = s.to_json().expect("serialize");
        let back = Session::from_json(&json).expect("deserialize");
        assert_eq!(back.groups.len(), 1);
        assert_eq!(back.current, Some(0));
        assert_eq!(back.groups[0].label, "cu");
        assert_eq!(back.groups[0].e0, Some(8979.0));
        assert_eq!(back.folders.data_dir, Some(PathBuf::from("/tmp/data")));
    }
}
