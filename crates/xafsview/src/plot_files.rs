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
use feffit::xasproc::{PreEdgeParams, pre_edge};

/// The file-overlay types of the original's *File type* selector. (The remaining
/// analysis-mode / beamline-specific types — `Normalize`, `res. all`,
/// `*_Dearctan.dat`, `raw data`, the intensity ratios, `*.XRF`, `XRD`, `UV-VIS`,
/// … — are deferred.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// A saved Plot Data composite (`*.result`): the multi-curve file written by
    /// "Save in single file", read back as one curve per saved trace.
    Result,
}

impl FileType {
    /// All file types, in selector order.
    pub const ALL: [FileType; 6] = [
        FileType::Xmu,
        FileType::Chi,
        FileType::Dat,
        FileType::Fit,
        FileType::Bkg,
        FileType::Result,
    ];

    /// The selector label (the original's glob form).
    pub fn label(self) -> &'static str {
        match self {
            FileType::Xmu => "*.xmu",
            FileType::Chi => "*.chi",
            FileType::Dat => "*.dat",
            FileType::Fit => "*.fit",
            FileType::Bkg => "*.bkg",
            FileType::Result => "*.result",
        }
    }

    /// The graph items selectable under this file type, in order.
    pub fn items(self) -> &'static [GraphItem] {
        match self {
            FileType::Xmu => &[
                GraphItem::XmuF,
                GraphItem::XmuNorm,
                GraphItem::XmuD1,
                GraphItem::XmuD2,
            ],
            FileType::Chi => &[GraphItem::ChiK, GraphItem::ChiR],
            FileType::Dat => &[GraphItem::DatK, GraphItem::DatR, GraphItem::DatQ],
            FileType::Fit => &[GraphItem::FitK, GraphItem::FitR, GraphItem::FitQ],
            FileType::Bkg => &[GraphItem::BkgE, GraphItem::BkgK],
            FileType::Result => &[GraphItem::Result],
        }
    }

    /// The first graph item of this file type (the default when the type changes).
    pub fn default_item(self) -> GraphItem {
        self.items()[0]
    }
}

/// A graph item under a [`FileType`]: which file-name suffix it selects in the
/// folder and how the file's columns become a curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphItem {
    /// `μ(E)` itself.
    XmuF,
    /// Edge-step **normalized** `μ(E)` (pre-edge subtracted, divided by the edge
    /// step) — the original XAFSView "Normalize" view, reproducing `XANES.dat`.
    XmuNorm,
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
    /// A curve from a saved `*.result` composite (axis labels come from the
    /// file, not from this item).
    Result,
}

impl GraphItem {
    /// The combo label.
    pub fn label(self) -> &'static str {
        match self {
            GraphItem::XmuF => "f",
            GraphItem::XmuNorm => "norm",
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
            GraphItem::Result => "result",
        }
    }

    /// The file-name suffix this item selects (case-insensitively) in a folder.
    pub fn suffix(self) -> &'static str {
        match self {
            GraphItem::XmuF | GraphItem::XmuNorm | GraphItem::XmuD1 | GraphItem::XmuD2 => ".xmu",
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
            GraphItem::Result => ".result",
        }
    }

    /// Whether `name` (a bare file name) is selected by this item.
    pub fn matches(self, name: &str) -> bool {
        name.to_ascii_lowercase().ends_with(self.suffix())
    }

    /// The x-axis label for this item's space.
    pub fn x_label(self) -> &'static str {
        match self {
            GraphItem::XmuF | GraphItem::XmuNorm | GraphItem::XmuD1 | GraphItem::XmuD2 => {
                "Energy (eV)"
            }
            GraphItem::ChiK | GraphItem::DatK | GraphItem::FitK => "k (Å⁻¹)",
            GraphItem::ChiR | GraphItem::DatR | GraphItem::FitR => "R (Å)",
            GraphItem::DatQ | GraphItem::FitQ => "q (Å⁻¹)",
            GraphItem::BkgE => "Energy (eV)",
            GraphItem::BkgK => "k (Å⁻¹)",
            // Restored from the saved file by `load_result`; this is the fallback.
            GraphItem::Result => "x",
        }
    }

    /// The y-axis label for this item.
    pub fn y_label(self) -> &'static str {
        match self {
            GraphItem::XmuF => "μ(E)",
            GraphItem::XmuNorm => "norm μ(E)",
            GraphItem::XmuD1 => "dμ/dE",
            GraphItem::XmuD2 => "d²μ/dE²",
            GraphItem::ChiK | GraphItem::DatK | GraphItem::FitK => "kʷ·χ(k)",
            GraphItem::ChiR | GraphItem::DatR | GraphItem::FitR => "|χ(R)|",
            GraphItem::DatQ | GraphItem::FitQ => "|χ(q)|",
            GraphItem::BkgE => "μ₀(E)",
            GraphItem::BkgK => "μ₀−μ",
            GraphItem::Result => "y",
        }
    }

    /// k-weight applies to the k-space items (their files store unweighted χ).
    pub fn applies_kweight(self) -> bool {
        matches!(self, GraphItem::ChiK | GraphItem::DatK | GraphItem::FitK)
    }

    /// The 0-based column holding this item's plotted value. The R/q-space
    /// complex files store `axis, real, imag, ampl, phase`, so their magnitude
    /// is column 3; every other file type plots column 1 (the value beside the
    /// x grid). This must track the writer column order in
    /// [`crate::chi_io::complex5_string`].
    fn value_col(self) -> usize {
        match self {
            GraphItem::ChiR
            | GraphItem::DatR
            | GraphItem::DatQ
            | GraphItem::FitR
            | GraphItem::FitQ => 3,
            _ => 1,
        }
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
    /// Axis labels restored from a `*.result` file, overriding `item`'s labels;
    /// `None` for ordinary single-item files (which use `item`).
    pub axis: Option<(String, String)>,
    /// x grid (energy / k / R / q).
    pub x: Vec<f64>,
    /// y curve, with k-weighting / derivative already applied.
    pub y: Vec<f64>,
}

/// Read `path` as `item`, applying `kweight` to k-space items. The x grid is
/// column 0 and the value is [`GraphItem::value_col`] (column 1 for most files,
/// column 3 — `ampl` — for the R/q-space complex files); `*.xmu` `f'`/`f''`
/// differentiate the value against the energy grid.
pub fn load_trace(path: &Path, item: GraphItem, kweight: i32) -> Result<LoadedTrace, String> {
    let name = file_name(path);
    let cf = ColumnFile::from_path(path).map_err(|e| format!("{name}: {e}"))?;
    let x = cf
        .column(0)
        .ok_or_else(|| format!("{name}: no data columns"))?
        .to_vec();
    let vcol = item.value_col();
    let v = cf
        .column(vcol)
        .ok_or_else(|| format!("{name}: needs at least {} columns", vcol + 1))?;
    // Normalized μ(E) is not a column transform — it runs the pre-edge / edge-step
    // normalization (larch auto parameters) to reproduce the original's XANES.dat.
    if item == GraphItem::XmuNorm {
        let pe = pre_edge(&x, v, &PreEdgeParams::default());
        return Ok(LoadedTrace {
            path: path.to_path_buf(),
            label: name,
            item,
            axis: None,
            x: pe.energy,
            y: pe.norm,
        });
    }
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
        axis: None,
        x,
        y,
    })
}

/// Read a `*.result` composite — the multi-curve file written by Plot Data's
/// "Save in single file" — into one [`LoadedTrace`] per saved curve. The file's
/// `# x-axis:` / `# y-axis:` header lines restore the axis labels (shared by
/// every curve), and each curve's `# curve N: <label>` comment restores its
/// legend label. Curves are separated by a blank line and carry two numeric
/// columns; a `# k-weight` etc. is *not* re-applied — the saved values are final.
pub fn load_result(path: &Path) -> Result<Vec<LoadedTrace>, String> {
    let name = file_name(path);
    let text = std::fs::read_to_string(path).map_err(|e| format!("{name}: {e}"))?;

    // Axis labels are written once, in the file header.
    let header_value = |key: &str| {
        text.lines().find_map(|l| {
            l.trim_start()
                .strip_prefix(key)
                .map(|v| v.trim().to_owned())
        })
    };
    let axis = match (header_value("# x-axis:"), header_value("# y-axis:")) {
        (Some(x), Some(y)) => Some((x, y)),
        _ => None,
    };

    let mut out: Vec<LoadedTrace> = Vec::new();
    // Blocks are separated by a blank line; the first block also carries the
    // file header comments (ignored by the column reader).
    for block in text.split("\n\n") {
        if block.trim().is_empty() {
            continue;
        }
        let label = block
            .lines()
            .find_map(|l| l.trim_start().strip_prefix("# curve"))
            .and_then(|rest| rest.split_once(':').map(|(_, lab)| lab.trim().to_owned()))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("{name} [{}]", out.len() + 1));

        // A header-only block (no numeric rows) is skipped, not an error.
        let Ok(cf) = ColumnFile::from_text(block) else {
            continue;
        };
        let (Some(x), Some(y)) = (cf.column(0), cf.column(1)) else {
            continue;
        };
        out.push(LoadedTrace {
            path: path.to_path_buf(),
            label,
            item: GraphItem::Result,
            axis: axis.clone(),
            x: x.to_vec(),
            y: y.to_vec(),
        });
    }

    if out.is_empty() {
        return Err(format!("{name}: no curves found"));
    }
    Ok(out)
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
    use crate::chi_io::{chik_string, complex5_string};

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
        // re/im chosen so |χ(R)| = ampl column (index 3) is exactly `mag`.
        let re = [1.0, 0.0, -3.0];
        let im = [0.0, 2.0, 0.0];
        let mag = [1.0, 2.0, 3.0];
        std::fs::write(&p, complex5_string("# t\r\n", "r", &r, &mag, &re, &im))
            .expect("write r.dat");

        let t = load_trace(&p, GraphItem::DatR, 2).expect("load r.dat");
        assert_eq!(t.x, r);
        assert_eq!(
            t.y, mag,
            "y is the ampl column (index 3), untouched by k-weight"
        );
        let _ = std::fs::remove_file(&p);
    }

    /// A file carrying the real multi-line `#` provenance header — including a
    /// verbatim source-header line with embedded tabs — must still parse: every
    /// `#` line (reduction params + echoed beamline block + separator) is skipped
    /// as a comment and the numeric columns read back unchanged. This locks the
    /// integration between [`crate::chi_io::provenance_header`] and the reader.
    #[test]
    fn provenance_header_block_is_skipped_and_columns_round_trip() {
        use crate::chi_io::provenance_header;
        use feffit::xasdata::XasGroup;

        let p = tmp("prov_r.dat");
        let r = [0.0, 0.1, 0.2];
        let re = [1.0, 0.0, -3.0];
        let im = [0.0, 2.0, 0.0];
        let mag = [1.0, 2.0, 3.0];

        let mut g = XasGroup::from_chi("sample", vec![1.0], vec![0.1]);
        g.e0 = Some(6539.0);
        g.edge_step = Some(1.553);
        g.pre1 = Some(-200.0);
        g.pre2 = Some(-50.0);
        g.rbkg = Some(1.2);
        // A verbatim source line with tabs — the riskiest header content for the
        // numeric reader (it must NOT mistake the tab-split tokens for data).
        g.source_header = vec![
            "Data were taken at HFXAFS in PLS-II\tNumber of Points : 459\tEo : 6539.0".to_owned(),
        ];
        let header = provenance_header(&g, 0.0, 12.2, 3, 0.0);
        // Sanity: the header really is multi-line and carries the tab line.
        assert!(header.lines().count() > 5, "multi-line header: {header}");
        assert!(header.contains('\t'), "verbatim tab line present");

        std::fs::write(&p, complex5_string(&header, "r", &r, &mag, &re, &im))
            .expect("write prov_r.dat");

        let t = load_trace(&p, GraphItem::DatR, 2).expect("load prov_r.dat");
        assert_eq!(t.x, r, "axis column read past the provenance block");
        assert_eq!(t.y, mag, "ampl column read past the provenance block");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn load_k_chi_applies_kweight() {
        let p = tmp("k.chi");
        let k = [1.0, 2.0, 4.0];
        let chi = [1.0, 1.0, 1.0];
        std::fs::write(&p, chik_string("# t\r\n", &k, &chi)).expect("write k.chi");

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
        std::fs::write(&p, chik_string("# t\r\n", &k, &bkg)).expect("write k.bkg");
        let t = load_trace(&p, GraphItem::BkgK, 3).expect("load k.bkg");
        assert_eq!(t.x, k);
        assert_eq!(t.y, bkg, "k.bkg plots μ₀−μ as written, ignoring G_kweight");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn xmu_norm_runs_pre_edge_and_flattens_the_step_to_unity() {
        // A logistic edge at 6539 eV: flat ≈0.1 below, flat ≈1.1 above ⇒ an edge
        // step of ≈1.0. `norm` is (μ − pre-edge line)/step, so it must sit near 0
        // well below the edge and near 1 well above it — the XANES.dat curve.
        let p = tmp("step.xmu");
        let e: Vec<f64> = (0..201).map(|i| 6400.0 + i as f64 * 2.0).collect();
        let mu: Vec<f64> = e
            .iter()
            .map(|&x| 0.1 + 1.0 / (1.0 + (-(x - 6539.0) / 3.0).exp()))
            .collect();
        std::fs::write(&p, chik_string("# t\r\n", &e, &mu)).expect("write xmu");

        let t = load_trace(&p, GraphItem::XmuNorm, 0).expect("load xmu norm");
        assert_eq!(t.item, GraphItem::XmuNorm);
        assert_eq!(t.x.len(), e.len());
        assert!(t.y[0] < 0.2, "pre-edge norm ≈0, got {}", t.y[0]);
        let last = *t.y.last().unwrap();
        assert!((last - 1.0).abs() < 0.2, "post-edge norm ≈1, got {last}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn xmu_first_derivative_of_a_line_is_its_slope() {
        let p = tmp("line.xmu");
        // y = 3x ⇒ dy/dx ≡ 3 everywhere (incl. the one-sided ends).
        let e = [0.0, 1.0, 2.0, 3.0];
        let mu = [0.0, 3.0, 6.0, 9.0];
        std::fs::write(&p, chik_string("# t\r\n", &e, &mu)).expect("write xmu");

        let t = load_trace(&p, GraphItem::XmuD1, 0).expect("load xmu f'");
        for d in &t.y {
            assert!((d - 3.0).abs() < 1e-9, "slope 3, got {d}");
        }
        let _ = std::fs::remove_file(&p);
    }
}
