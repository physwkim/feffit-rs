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
/// XAFSView kept separate folders for data, scratch/work output, and FEFF runs;
/// we keep the same split so file dialogs and batch output can default sensibly.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Folders {
    /// Where raw/reduced data files are read from.
    pub data_dir: Option<PathBuf>,
    /// Where reduced output and projects are written.
    pub work_dir: Option<PathBuf>,
    /// Where `feff.inp` / `feffNNNN.dat` live for fitting.
    pub feff_dir: Option<PathBuf>,
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
