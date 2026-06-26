//! File-type model and readers for the Plot Data window's **file overlay**.
//!
//! The original XAFSView Plot Data is a file viewer: the user picks a *File
//! type* (`*.xmu`, `*.chi`, `*.dat`, `*.fit`) and a *Graph item* under it (e.g.
//! `*.dat` → `k.dat` / `r.dat` / `q.dat`), then overlays the matching files from
//! a folder. This module maps each (file type, graph item) pair to the file-name
//! suffix it selects and the recipe for turning the file's columns into an
//! `(x, y)` curve, and reads one file into a [`LoadedTrace`] via
//! [`feffit::xasdata::ColumnFile`].
//!
//! Colours are assigned at draw time (in `plot_data::rebuild`), not stored here,
//! so a [`LoadedTrace`] is just the labelled data.

use std::path::{Path, PathBuf};

use feffit::xasdata::ColumnFile;

/// The four file-overlay types of the original's *File type* selector. (The
/// analysis-mode types — `Normalize`, `*.result`, `res. all`, `*_Dearctan.dat`,
/// `raw data`, `*.bkg` — are deferred.)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// `μ(E)` files: `energy`, `μ`.
    Xmu,
    /// AUTOBK output: `<stem>k.chi` (k, χ) and `<stem>r.chi` (R, |χ|, re, im).
    Chi,
    /// FEFFIT data: `<stem>{k,r,q}.dat`.
    Dat,
    /// FEFFIT model: `<stem>{k,r,q}.fit`.
    Fit,
    /// AUTOBK background: `<stem>e.bkg` (energy, μ₀) and `<stem>k.bkg` (k, μ₀−μ).
    Bkg,
}

impl FileType {
    /// All file types, in selector order.
    pub const ALL: [FileType; 5] = [
        FileType::Xmu,
        FileType::Chi,
        FileType::Dat,
        FileType::Fit,
        FileType::Bkg,
    ];

    /// The selector label (the original's glob form).
    pub fn label(self) -> &'static str {
        match self {
            FileType::Xmu => "*.xmu",
            FileType::Chi => "*.chi",
            FileType::Dat => "*.dat",
            FileType::Fit => "*.fit",
            FileType::Bkg => "*.bkg",
        }
    }

    /// The graph items selectable under this file type, in order.
    pub fn items(self) -> &'static [GraphItem] {
        match self {
            FileType::Xmu => &[GraphItem::XmuF, GraphItem::XmuD1, GraphItem::XmuD2],
            FileType::Chi => &[GraphItem::ChiK, GraphItem::ChiR],
            FileType::Dat => &[GraphItem::DatK, GraphItem::DatR, GraphItem::DatQ],
            FileType::Fit => &[GraphItem::FitK, GraphItem::FitR, GraphItem::FitQ],
            FileType::Bkg => &[GraphItem::BkgE, GraphItem::BkgK],
        }
    }

    /// The first graph item of this file type (the default when the type changes).
    pub fn default_item(self) -> GraphItem {
        self.items()[0]
    }
}

/// A graph item under a [`FileType`]: which file-name suffix it selects in the
/// folder and how the file's columns become a curve.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GraphItem {
    /// `μ(E)` itself.
    XmuF,
    /// `dμ/dE`.
    XmuD1,
    /// `d²μ/dE²`.
    XmuD2,
    /// `<stem>k.chi`: kʷ·χ(k).
    ChiK,
    /// `<stem>r.chi`: |χ(R)|.
    ChiR,
    /// `<stem>k.dat`: kʷ·χ(k) data.
    DatK,
    /// `<stem>r.dat`: |χ(R)| data.
    DatR,
    /// `<stem>q.dat`: |χ(q)| data.
    DatQ,
    /// `<stem>k.fit`: kʷ·χ(k) model.
    FitK,
    /// `<stem>r.fit`: |χ(R)| model.
    FitR,
    /// `<stem>q.fit`: |χ(q)| model.
    FitQ,
    /// `<stem>e.bkg`: the AUTOBK background μ₀(E).
    BkgE,
    /// `<stem>k.bkg`: the AUTOBK background in k, μ₀−μ.
    BkgK,
}

impl GraphItem {
    /// The combo label.
    pub fn label(self) -> &'static str {
        match self {
            GraphItem::XmuF => "f",
            GraphItem::XmuD1 => "f'",
            GraphItem::XmuD2 => "f''",
            GraphItem::ChiK => "k.chi",
            GraphItem::ChiR => "r.chi",
            GraphItem::DatK => "k.dat",
            GraphItem::DatR => "r.dat",
            GraphItem::DatQ => "q.dat",
            GraphItem::FitK => "k.fit",
            GraphItem::FitR => "r.fit",
            GraphItem::FitQ => "q.fit",
            GraphItem::BkgE => "e.bkg",
            GraphItem::BkgK => "k.bkg",
        }
    }

    /// The file-name suffix this item selects (case-insensitively) in a folder.
    pub fn suffix(self) -> &'static str {
        match self {
            GraphItem::XmuF | GraphItem::XmuD1 | GraphItem::XmuD2 => ".xmu",
            GraphItem::ChiK => "k.chi",
            GraphItem::ChiR => "r.chi",
            GraphItem::DatK => "k.dat",
            GraphItem::DatR => "r.dat",
            GraphItem::DatQ => "q.dat",
            GraphItem::FitK => "k.fit",
            GraphItem::FitR => "r.fit",
            GraphItem::FitQ => "q.fit",
            GraphItem::BkgE => "e.bkg",
            GraphItem::BkgK => "k.bkg",
        }
    }

    /// Whether `name` (a bare file name) is selected by this item.
    pub fn matches(self, name: &str) -> bool {
        name.to_ascii_lowercase().ends_with(self.suffix())
    }

    /// The x-axis label for this item's space.
    pub fn x_label(self) -> &'static str {
        match self {
            GraphItem::XmuF | GraphItem::XmuD1 | GraphItem::XmuD2 => "Energy (eV)",
            GraphItem::ChiK | GraphItem::DatK | GraphItem::FitK => "k (Å⁻¹)",
            GraphItem::ChiR | GraphItem::DatR | GraphItem::FitR => "R (Å)",
            GraphItem::DatQ | GraphItem::FitQ => "q (Å⁻¹)",
            GraphItem::BkgE => "Energy (eV)",
            GraphItem::BkgK => "k (Å⁻¹)",
        }
    }

    /// The y-axis label for this item.
    pub fn y_label(self) -> &'static str {
        match self {
            GraphItem::XmuF => "μ(E)",
            GraphItem::XmuD1 => "dμ/dE",
            GraphItem::XmuD2 => "d²μ/dE²",
            GraphItem::ChiK | GraphItem::DatK | GraphItem::FitK => "kʷ·χ(k)",
            GraphItem::ChiR | GraphItem::DatR | GraphItem::FitR => "|χ(R)|",
            GraphItem::DatQ | GraphItem::FitQ => "|χ(q)|",
            GraphItem::BkgE => "μ₀(E)",
            GraphItem::BkgK => "μ₀−μ",
        }
    }

    /// k-weight applies to the k-space items (their files store unweighted χ).
    pub fn applies_kweight(self) -> bool {
        matches!(self, GraphItem::ChiK | GraphItem::DatK | GraphItem::FitK)
    }

    /// Derivative order of the y column (0/1/2 — only the `*.xmu` items differ).
    fn deriv_order(self) -> u8 {
        match self {
            GraphItem::XmuD1 => 1,
            GraphItem::XmuD2 => 2,
            _ => 0,
        }
    }
}

/// One file loaded for overlay: its `(x, y)` curve and the item it came from.
pub struct LoadedTrace {
    /// Source file.
    pub path: PathBuf,
    /// Legend label (the file name).
    pub label: String,
    /// The graph item this was read as (carries the axis labels).
    pub item: GraphItem,
    /// x grid (energy / k / R / q).
    pub x: Vec<f64>,
    /// y curve, with k-weighting / derivative already applied.
    pub y: Vec<f64>,
}

/// Read `path` as `item`, applying `kweight` to k-space items. The x grid is
/// column 0 and the value is column 1; `*.xmu` `f'`/`f''` differentiate the
/// value against the energy grid.
pub fn load_trace(path: &Path, item: GraphItem, kweight: i32) -> Result<LoadedTrace, String> {
    let name = file_name(path);
    let cf = ColumnFile::from_path(path).map_err(|e| format!("{name}: {e}"))?;
    let x = cf
        .column(0)
        .ok_or_else(|| format!("{name}: no data columns"))?
        .to_vec();
    let v = cf
        .column(1)
        .ok_or_else(|| format!("{name}: needs at least two columns"))?;
    let y = match item.deriv_order() {
        1 => derivative(&x, v),
        2 => derivative(&x, &derivative(&x, v)),
        _ if item.applies_kweight() => x
            .iter()
            .zip(v)
            .map(|(&k, &c)| c * k.powi(kweight))
            .collect(),
        _ => v.to_vec(),
    };
    Ok(LoadedTrace {
        path: path.to_path_buf(),
        label: name,
        item,
        x,
        y,
    })
}

/// The bare file name of `path` (for legends and error messages).
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Central-difference derivative `dy/dx`, one-sided at the ends. Returns zeros
/// for fewer than two points or where the x step collapses (duplicate x).
fn derivative(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return vec![0.0; n];
    }
    (0..n)
        .map(|i| {
            let (a, b) = match i {
                0 => (0, 1),
                i if i == n - 1 => (n - 2, n - 1),
                i => (i - 1, i + 1),
            };
            let dx = x[b] - x[a];
            if dx == 0.0 { 0.0 } else { (y[b] - y[a]) / dx }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chi_io::{chik_string, complex4_string};

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("xafsview_pf_{}_{name}", std::process::id()))
    }

    #[test]
    fn item_suffixes_are_mutually_exclusive_within_a_type() {
        // q.dat must not be picked up by the k.dat or r.dat filters.
        assert!(GraphItem::DatQ.matches("FeS2q.dat"));
        assert!(!GraphItem::DatK.matches("FeS2q.dat"));
        assert!(!GraphItem::DatR.matches("FeS2q.dat"));
        // case-insensitive on the extension
        assert!(GraphItem::XmuF.matches("SAMPLE.XMU"));
    }

    #[test]
    fn load_r_dat_reads_magnitude_column() {
        let p = tmp("r.dat");
        let r = [0.0, 0.1, 0.2];
        let mag = [1.0, 2.0, 3.0];
        std::fs::write(
            &p,
            complex4_string("t", "R", &r, &mag, &[0.0; 3], &[0.0; 3]),
        )
        .expect("write r.dat");

        let t = load_trace(&p, GraphItem::DatR, 2).expect("load r.dat");
        assert_eq!(t.x, r);
        assert_eq!(t.y, mag, "y is the |χ(R)| column, untouched by k-weight");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn load_k_chi_applies_kweight() {
        let p = tmp("k.chi");
        let k = [1.0, 2.0, 4.0];
        let chi = [1.0, 1.0, 1.0];
        std::fs::write(&p, chik_string("t", &k, &chi)).expect("write k.chi");

        let t = load_trace(&p, GraphItem::ChiK, 2).expect("load k.chi");
        // k²·χ with χ ≡ 1 ⇒ y = k².
        assert!((t.y[0] - 1.0).abs() < 1e-9);
        assert!((t.y[1] - 4.0).abs() < 1e-9);
        assert!((t.y[2] - 16.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn bkg_items_select_their_files_and_load_the_value_untouched() {
        // e.bkg / k.bkg suffixes are exclusive of each other and of the χ items.
        assert!(GraphItem::BkgK.matches("FeS2k.bkg"));
        assert!(GraphItem::BkgE.matches("FeS2e.bkg"));
        assert!(!GraphItem::BkgK.matches("FeS2e.bkg"));
        assert!(!GraphItem::ChiK.matches("FeS2k.bkg"));

        // The background is stored ready-to-plot: no k-weighting, no derivative.
        let p = tmp("k.bkg");
        let k = [1.0, 2.0, 4.0];
        let bkg = [0.5, 0.4, 0.3];
        std::fs::write(&p, chik_string("t", &k, &bkg)).expect("write k.bkg");
        let t = load_trace(&p, GraphItem::BkgK, 3).expect("load k.bkg");
        assert_eq!(t.x, k);
        assert_eq!(t.y, bkg, "k.bkg plots μ₀−μ as written, ignoring G_kweight");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn xmu_first_derivative_of_a_line_is_its_slope() {
        let p = tmp("line.xmu");
        // y = 3x ⇒ dy/dx ≡ 3 everywhere (incl. the one-sided ends).
        let e = [0.0, 1.0, 2.0, 3.0];
        let mu = [0.0, 3.0, 6.0, 9.0];
        std::fs::write(&p, chik_string("t", &e, &mu)).expect("write xmu");

        let t = load_trace(&p, GraphItem::XmuD1, 0).expect("load xmu f'");
        for d in &t.y {
            assert!((d - 3.0).abs() < 1e-9, "slope 3, got {d}");
        }
        let _ = std::fs::remove_file(&p);
    }
}
