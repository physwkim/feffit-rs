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

use feffit::xasdata::{ColumnFile, MonitorPolicy};
use feffit::xasproc::{PreEdgeParams, pre_edge};

use crate::open_data::raw_scans;
use crate::widgets::file_name_of;

/// The file-overlay types of the original's *File type* selector. (The remaining
/// analysis-mode / beamline-specific types — `Normalize`, `res. all`,
/// `*_Dearctan.dat`, `*.XRF`, `XRD`, `UV-VIS`, … — are deferred.)
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
    /// Un-reduced beamline scans: the monitor channels and the intensity ratios
    /// built from them, plotted against energy without building μ(E) or running
    /// AUTOBK. The scan-quality view — is I₀ flat, did a channel drop out.
    Raw,
}

impl FileType {
    /// All file types, in selector order.
    pub const ALL: [FileType; 7] = [
        FileType::Xmu,
        FileType::Chi,
        FileType::Dat,
        FileType::Fit,
        FileType::Bkg,
        FileType::Result,
        FileType::Raw,
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
            FileType::Raw => "raw data",
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
            FileType::Raw => RAW_ITEMS,
        }
    }

    /// The first graph item of this file type (the default when the type changes).
    pub fn default_item(self) -> GraphItem {
        self.items()[0]
    }
}

/// A channel or intensity ratio read straight from a raw beamline scan, under
/// [`FileType::Raw`].
///
/// The columns are found by *label*, through
/// [`ColumnFile::role_index`](feffit::xasdata::ColumnFile::role_index), so the
/// same lookup that seeds the Autobk column chooser serves here — a scan whose
/// channels the importer resolves is a scan this plots, with no second naming
/// convention to keep in sync.
///
/// The per-element entries cover a 4-element detector (the ME-4 the beamline
/// runs). A wider array's extra elements are not selectable here; its summed
/// channel is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawChannel {
    /// `ln(I₀ / Iₜ)` — transmission μ, unnormalized.
    Transmission,
    /// `I_f / I₀` — fluorescence yield from the single fluorescence channel.
    Fluorescence,
    /// `ln(Iₜ / I_ref)` — the reference foil, for energy alignment.
    Reference,
    /// `Σ(elements) / I₀` — fluorescence yield from the multi-element detector.
    FluorescenceSdd,
    /// A single monitor channel, plotted as recorded counts.
    Monitor(Monitor),
    /// The multi-element detector's summed channel, as counts.
    SddSum,
    /// One element of the multi-element detector (1-based), as counts.
    SddElement(u8),
}

/// A single monitor channel of a raw scan, plotted as recorded counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monitor {
    /// Incident beam.
    I0,
    /// Transmitted beam.
    It,
    /// Fluorescence.
    If,
    /// Reference foil.
    Ir,
    /// A fourth monitor some beamlines record after the reference.
    Im,
}

impl Monitor {
    /// The label keys this monitor's column may carry.
    fn keys(self) -> &'static [&'static str] {
        match self {
            Monitor::I0 => &["i0", "io"],
            Monitor::It => &["it", "itrans", "i1", "trans"],
            Monitor::If => &["if", "iflu", "ifluor"],
            Monitor::Ir => &["iref", "ir", "i2"],
            Monitor::Im => &["im", "i3"],
        }
    }

    /// The selector / axis label.
    fn label(self) -> &'static str {
        match self {
            Monitor::I0 => "I₀",
            Monitor::It => "Iₜ",
            Monitor::If => "I_f",
            Monitor::Ir => "I_ref",
            Monitor::Im => "I_m",
        }
    }
}

/// The label keys a multi-element detector's summed column may carry.
const SDD_SUM_KEYS: &[&str] = &["all ch", "allch", "me-4", "all channels"];

/// The raw-scan items, in selector order — the original's intensity-ratio and
/// per-channel list.
const RAW_ITEMS: &[GraphItem] = &[
    GraphItem::Raw(RawChannel::Transmission),
    GraphItem::Raw(RawChannel::Fluorescence),
    GraphItem::Raw(RawChannel::Reference),
    GraphItem::Raw(RawChannel::Monitor(Monitor::I0)),
    GraphItem::Raw(RawChannel::Monitor(Monitor::It)),
    GraphItem::Raw(RawChannel::Monitor(Monitor::If)),
    GraphItem::Raw(RawChannel::Monitor(Monitor::Ir)),
    GraphItem::Raw(RawChannel::Monitor(Monitor::Im)),
    GraphItem::Raw(RawChannel::FluorescenceSdd),
    GraphItem::Raw(RawChannel::SddSum),
    GraphItem::Raw(RawChannel::SddElement(1)),
    GraphItem::Raw(RawChannel::SddElement(2)),
    GraphItem::Raw(RawChannel::SddElement(3)),
    GraphItem::Raw(RawChannel::SddElement(4)),
];

impl RawChannel {
    /// The selector label.
    fn label(self) -> String {
        match self {
            RawChannel::Transmission => "Transmission".into(),
            RawChannel::Fluorescence => "Fluorescence".into(),
            RawChannel::Reference => "Reference".into(),
            RawChannel::FluorescenceSdd => "Fluorescence (SDD)".into(),
            RawChannel::Monitor(m) => m.label().into(),
            RawChannel::SddSum => "SDD all ch".into(),
            RawChannel::SddElement(n) => format!("SDD ch{n:02}"),
        }
    }

    /// The y-axis label.
    fn y_label(self) -> String {
        match self {
            RawChannel::Transmission => "ln(I₀/Iₜ)".into(),
            RawChannel::Fluorescence => "I_f/I₀".into(),
            RawChannel::Reference => "ln(Iₜ/I_ref)".into(),
            RawChannel::FluorescenceSdd => "ΣSDD/I₀".into(),
            _ => format!("{} (counts)", self.label()),
        }
    }

    /// This channel's column in `cf`, by label. `None` when the scan does not
    /// carry it — a transmission-only scan has no SDD columns.
    fn column(self, cf: &ColumnFile) -> Option<&[f64]> {
        let idx = match self {
            RawChannel::Monitor(m) => cf.role_index(m.keys())?,
            RawChannel::SddSum | RawChannel::FluorescenceSdd => cf.role_index(SDD_SUM_KEYS)?,
            RawChannel::SddElement(n) => {
                // `Ch #-01` is what the beamline writes; `mca1` is the XDI
                // spelling of the same per-element column.
                let keys = [format!("ch #-{n:02}"), format!("mca{n}")];
                let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
                cf.role_index(&keys)?
            }
            // The ratios read two columns; `column` is for the single-column items.
            RawChannel::Transmission | RawChannel::Fluorescence | RawChannel::Reference => {
                return None;
            }
        };
        cf.column(idx)
    }

    /// Build this channel's curve from `cf`, reading monitors under `monitors`.
    /// `Err` names the channel the scan is missing.
    fn values(self, cf: &ColumnFile, monitors: MonitorPolicy) -> Result<Vec<f64>, String> {
        let need = |ch: RawChannel| -> Result<Vec<f64>, String> {
            ch.column(cf)
                .map(|c| monitors.apply(c).into_owned())
                .ok_or_else(|| format!("no {} column", ch.label()))
        };
        match self {
            RawChannel::Transmission => {
                let i0 = need(RawChannel::Monitor(Monitor::I0))?;
                let it = need(RawChannel::Monitor(Monitor::It))?;
                Ok(feffit::xasdata::transmission_mu(&i0, &it))
            }
            RawChannel::Reference => {
                let it = need(RawChannel::Monitor(Monitor::It))?;
                let ir = need(RawChannel::Monitor(Monitor::Ir))?;
                Ok(feffit::xasdata::transmission_mu(&it, &ir))
            }
            RawChannel::Fluorescence => {
                let iff = need(RawChannel::Monitor(Monitor::If))?;
                let i0 = need(RawChannel::Monitor(Monitor::I0))?;
                Ok(feffit::xasdata::fluorescence_mu(&iff, &i0))
            }
            RawChannel::FluorescenceSdd => {
                let sdd = need(RawChannel::SddSum)?;
                let i0 = need(RawChannel::Monitor(Monitor::I0))?;
                Ok(feffit::xasdata::fluorescence_mu(&sdd, &i0))
            }
            // A bare channel is plotted as recorded — the point of the view is to
            // see what the detector actually wrote, so the clamp stays off it.
            single => single
                .column(cf)
                .map(<[f64]>::to_vec)
                .ok_or_else(|| format!("no {} column", single.label())),
        }
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
    /// A channel or intensity ratio of a raw beamline scan.
    Raw(RawChannel),
}

impl GraphItem {
    /// The combo label.
    pub fn label(self) -> String {
        if let GraphItem::Raw(ch) = self {
            return ch.label();
        }
        self.fixed_label().into()
    }

    /// The combo label of the fixed (non-raw) items.
    fn fixed_label(self) -> &'static str {
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
            GraphItem::Raw(_) => "raw",
        }
    }

    /// The file-name suffix this item selects (case-insensitively) in a folder,
    /// or `None` when the item is not selected by name at all.
    ///
    /// Every item we *write* carries a suffix we chose. A raw scan is named by
    /// the beamline — `sample.000`, `sample.txt` — so there is no suffix to key
    /// on and the selection is stated the other way round (see [`Self::matches`]).
    /// Returning `Option` rather than `""` keeps that from collapsing into
    /// `ends_with("")`, which is true of every file including our own outputs.
    pub fn suffix(self) -> Option<&'static str> {
        Some(match self {
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
            GraphItem::Raw(_) => return None,
        })
    }

    /// Whether the file at `path` is selected by this item.
    pub fn matches(self, path: &Path) -> bool {
        match self.suffix() {
            Some(sfx) => file_name_of(path).to_ascii_lowercase().ends_with(sfx),
            // Raw scans list by the Open-data loader's rule: not one of our
            // own outputs, and content that reads as a scan table.
            None => raw_scans(path),
        }
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
            GraphItem::Raw(_) => "Energy (eV)",
        }
    }

    /// The y-axis label for this item.
    pub fn y_label(self) -> String {
        if let GraphItem::Raw(ch) = self {
            return ch.y_label();
        }
        self.fixed_y_label().into()
    }

    /// The y-axis label of the fixed (non-raw) items.
    fn fixed_y_label(self) -> &'static str {
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
            GraphItem::Raw(_) => "y",
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

/// Read `path` as `item`, applying `kweight` to k-space items and reading raw
/// scans' monitors under `monitors`. The x grid is column 0 and the value is
/// [`GraphItem::value_col`] (column 1 for most files, column 3 — `ampl` — for
/// the R/q-space complex files); `*.xmu` `f'`/`f''` differentiate the value
/// against the energy grid. A [`GraphItem::Raw`] item instead resolves its
/// columns by label — see [`RawChannel`].
pub fn load_trace(
    path: &Path,
    item: GraphItem,
    kweight: i32,
    monitors: MonitorPolicy,
) -> Result<LoadedTrace, String> {
    let name = file_name(path);
    let cf = ColumnFile::from_path(path).map_err(|e| format!("{name}: {e}"))?;
    if let GraphItem::Raw(ch) = item {
        // A raw scan's energy column is found by label like every other channel;
        // column 0 is the fallback for a scan whose labels don't name it.
        let ex = cf.guess_roles().energy.unwrap_or(0);
        let x = cf
            .column(ex)
            .ok_or_else(|| format!("{name}: no data columns"))?
            .to_vec();
        let y = ch
            .values(&cf, monitors)
            .map_err(|e| format!("{name}: {e}"))?;
        return Ok(LoadedTrace {
            path: path.to_path_buf(),
            label: name,
            item,
            axis: None,
            x,
            y,
        });
    }
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
pub(crate) fn derivative(x: &[f64], y: &[f64]) -> Vec<f64> {
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
        assert!(GraphItem::DatQ.matches(Path::new("FeS2q.dat")));
        assert!(!GraphItem::DatK.matches(Path::new("FeS2q.dat")));
        assert!(!GraphItem::DatR.matches(Path::new("FeS2q.dat")));
        // case-insensitive on the extension
        assert!(GraphItem::XmuF.matches(Path::new("SAMPLE.XMU")));
    }

    /// A PLS-II transmission scan with an SDD array: tab-separated labels whose
    /// channel names carry spaces, and a dropped I₀ point on the last row.
    fn pls2_scan() -> String {
        "PLS-II 7D XAFS\n\
         Read-Energy(eV)\tCh1 (I0)\tCh2 (It)\tCh3 (If)\tCh4 (Ir)\tCh5 (Im)\t\
         All Ch (ME-4)\tCh #-01\tCh #-02\tCh #-03\tCh #-04\n\
         8900.0\t100000\t50000\t1200\t48000\t900\t4800\t1000\t1200\t1300\t1300\n\
         8901.0\t100100\t50100\t1210\t48100\t910\t4840\t1010\t1210\t1310\t1310\n\
         8902.0\t0\t50200\t1220\t48200\t920\t4880\t1020\t1220\t1320\t1320\n"
            .to_string()
    }

    #[test]
    fn raw_items_select_scans_and_exclude_our_own_outputs() {
        let raw = GraphItem::Raw(RawChannel::Transmission);
        // The raw rule sniffs content, so the positives are real files.
        let d = std::env::temp_dir().join(format!("xafsview_pf_raw_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("mkdir");
        for name in ["PGT_Mn_S2_NCM50_W_Insitu.000", "scan.001", "data.txt"] {
            let p = d.join(name);
            std::fs::write(&p, "header\n6539.0 100000 50000\n").expect("w");
            assert!(raw.matches(&p), "{name}");
        }
        // A non-table file in the same folder is not offered as a scan.
        let png = d.join("shot.png");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").expect("w");
        assert!(!raw.matches(&png));
        // Not our outputs — otherwise a raw overlay would list every .xmu the
        // app has written into the same folder (excluded by name alone, so no
        // file is needed).
        assert!(!raw.matches(Path::new("sample.xmu")));
        assert!(!raw.matches(Path::new("samplek.chi")));
        assert!(!raw.matches(Path::new("samplee.bkg")));
        assert!(!raw.matches(Path::new("fit.FIT")));
        // And the suffix-keyed items still do not claim a raw scan.
        assert!(!GraphItem::XmuF.matches(Path::new("scan.001")));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn raw_ratios_resolve_pls2_channels_by_label() {
        let p = tmp("scan.000");
        std::fs::write(&p, pls2_scan()).expect("write scan");

        let trans = load_trace(
            &p,
            GraphItem::Raw(RawChannel::Transmission),
            0,
            MonitorPolicy::AsRecorded,
        )
        .expect("transmission");
        assert_eq!(trans.x, [8900.0, 8901.0, 8902.0], "energy by label");
        assert!((trans.y[0] - (100000.0f64 / 50000.0).ln()).abs() < 1e-12);

        let refr = load_trace(
            &p,
            GraphItem::Raw(RawChannel::Reference),
            0,
            MonitorPolicy::AsRecorded,
        )
        .expect("reference");
        assert!((refr.y[0] - (50000.0f64 / 48000.0).ln()).abs() < 1e-12);

        let flu = load_trace(
            &p,
            GraphItem::Raw(RawChannel::Fluorescence),
            0,
            MonitorPolicy::AsRecorded,
        )
        .expect("fluorescence");
        assert!((flu.y[0] - 1200.0 / 100000.0).abs() < 1e-12);

        let sdd = load_trace(
            &p,
            GraphItem::Raw(RawChannel::FluorescenceSdd),
            0,
            MonitorPolicy::AsRecorded,
        )
        .expect("sdd fluorescence");
        assert!((sdd.y[0] - 4800.0 / 100000.0).abs() < 1e-12);

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn raw_bare_channels_are_the_recorded_counts() {
        let p = tmp("scan.001");
        std::fs::write(&p, pls2_scan()).expect("write scan");

        for (ch, want) in [
            (RawChannel::Monitor(Monitor::I0), 100000.0),
            (RawChannel::Monitor(Monitor::It), 50000.0),
            (RawChannel::Monitor(Monitor::If), 1200.0),
            (RawChannel::Monitor(Monitor::Ir), 48000.0),
            (RawChannel::Monitor(Monitor::Im), 900.0),
            (RawChannel::SddSum, 4800.0),
            (RawChannel::SddElement(1), 1000.0),
            (RawChannel::SddElement(4), 1300.0),
        ] {
            let t = load_trace(&p, GraphItem::Raw(ch), 0, MonitorPolicy::AsRecorded)
                .unwrap_or_else(|e| panic!("{}: {e}", ch.label()));
            assert_eq!(t.y[0], want, "{}", ch.label());
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn raw_clamp_applies_to_ratios_but_never_to_a_bare_channel() {
        let p = tmp("scan.002");
        std::fs::write(&p, pls2_scan()).expect("write scan");

        // Row 2 has I₀ = 0: as recorded, ln(0/50200) is -inf.
        let recorded = load_trace(
            &p,
            GraphItem::Raw(RawChannel::Transmission),
            0,
            MonitorPolicy::AsRecorded,
        )
        .expect("transmission");
        assert!(recorded.y[2].is_infinite());

        let clamped = load_trace(
            &p,
            GraphItem::Raw(RawChannel::Transmission),
            0,
            MonitorPolicy::ClampNonPositive,
        )
        .expect("transmission clamped");
        assert!((clamped.y[2] - (1.0f64 / 50200.0).ln()).abs() < 1e-12);
        assert!(clamped.y[0].is_finite() && (clamped.y[0] - recorded.y[0]).abs() < 1e-12);

        // The point of a bare-channel view is to see what the detector wrote, so
        // the dropped point stays a zero.
        let bare = load_trace(
            &p,
            GraphItem::Raw(RawChannel::Monitor(Monitor::I0)),
            0,
            MonitorPolicy::ClampNonPositive,
        )
        .expect("i0");
        assert_eq!(bare.y[2], 0.0, "recorded zero is not floored");

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_missing_channel_names_itself_rather_than_plotting_nothing() {
        // A plain transmission scan has no SDD columns.
        let p = tmp("plain.000");
        std::fs::write(&p, "# energy i0 it\n8900 100 50\n8901 101 51\n").expect("write");
        let err = load_trace(
            &p,
            GraphItem::Raw(RawChannel::SddSum),
            0,
            MonitorPolicy::AsRecorded,
        )
        .err()
        .expect("no SDD column");
        assert!(err.contains("SDD all ch"), "err={err}");
        let _ = std::fs::remove_file(&p);
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

        let t = load_trace(&p, GraphItem::DatR, 2, MonitorPolicy::AsRecorded).expect("load r.dat");
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

        let t =
            load_trace(&p, GraphItem::DatR, 2, MonitorPolicy::AsRecorded).expect("load prov_r.dat");
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

        let t = load_trace(&p, GraphItem::ChiK, 2, MonitorPolicy::AsRecorded).expect("load k.chi");
        // k²·χ with χ ≡ 1 ⇒ y = k².
        assert!((t.y[0] - 1.0).abs() < 1e-9);
        assert!((t.y[1] - 4.0).abs() < 1e-9);
        assert!((t.y[2] - 16.0).abs() < 1e-9);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn bkg_items_select_their_files_and_load_the_value_untouched() {
        // e.bkg / k.bkg suffixes are exclusive of each other and of the χ items.
        assert!(GraphItem::BkgK.matches(Path::new("FeS2k.bkg")));
        assert!(GraphItem::BkgE.matches(Path::new("FeS2e.bkg")));
        assert!(!GraphItem::BkgK.matches(Path::new("FeS2e.bkg")));
        assert!(!GraphItem::ChiK.matches(Path::new("FeS2k.bkg")));

        // The background is stored ready-to-plot: no k-weighting, no derivative.
        let p = tmp("k.bkg");
        let k = [1.0, 2.0, 4.0];
        let bkg = [0.5, 0.4, 0.3];
        std::fs::write(&p, chik_string("# t\r\n", &k, &bkg)).expect("write k.bkg");
        let t = load_trace(&p, GraphItem::BkgK, 3, MonitorPolicy::AsRecorded).expect("load k.bkg");
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

        let t = load_trace(&p, GraphItem::XmuNorm, 0, MonitorPolicy::AsRecorded)
            .expect("load xmu norm");
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

        let t =
            load_trace(&p, GraphItem::XmuD1, 0, MonitorPolicy::AsRecorded).expect("load xmu f'");
        for d in &t.y {
            assert!((d - 3.0).abs() < 1e-9, "slope 3, got {d}");
        }
        let _ = std::fs::remove_file(&p);
    }
}
