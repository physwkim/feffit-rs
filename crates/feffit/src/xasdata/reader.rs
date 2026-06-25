//! Reader for whitespace/comma-delimited beamline column files.
//!
//! Handles the ASCII formats XAFS beamlines actually emit — XDI (`# Column.N:`
//! tagged), UWXAFS/EPICS step-scan dumps (`;`-commented), and plain two-column
//! `energy mu` files — with the same conventions as larch's `read_ascii`
//! (`larch/io/columnfile.py`):
//!
//! - A line is **data** when every token parses as a float; everything else is a
//!   header (or trailing footer) line.
//! - Column **labels** come from `# Column.N: <name> …` tags when present,
//!   otherwise from the last header line before the data block, split on
//!   whitespace; failing that, `col1, col2, …`.
//! - Comment characters are `#;%*!$` (a header line is one whose first
//!   non-space character is one of these, or any non-data line before the data).
//!
//! The GUI maps the parsed columns to roles (energy / I0 / It / fluorescence)
//! and builds `mu(E)` with [`crate::xasdata::xmu`].

use std::path::{Path, PathBuf};

/// Characters that begin a header/comment line (matches larch `COMMENTCHARS`).
pub const COMMENT_CHARS: &str = "#;%*!$";

/// A parsed column file: the numeric data plus the labels and header text.
#[derive(Clone, Debug, Default)]
pub struct ColumnFile {
    /// Source path, if read from a file.
    pub path: Option<PathBuf>,
    /// All header (comment / pre-data) lines, in file order.
    pub header: Vec<String>,
    /// One label per column (original case; matching is case-insensitive).
    pub labels: Vec<String>,
    /// Column-major data: `columns[c][r]` is row `r` of column `c`.
    pub columns: Vec<Vec<f64>>,
}

/// What went wrong reading a column file.
#[derive(Debug)]
pub enum ReadError {
    /// The file could not be read.
    Io(std::io::Error),
    /// No numeric data rows were found.
    NoData,
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Io(e) => write!(f, "I/O error: {e}"),
            ReadError::NoData => write!(f, "no numeric data rows found"),
        }
    }
}

impl std::error::Error for ReadError {}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        ReadError::Io(e)
    }
}

impl ColumnFile {
    /// Read and parse a column file from disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ReadError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let mut cf = Self::from_text(&text)?;
        cf.path = Some(path.to_path_buf());
        Ok(cf)
    }

    /// Parse a column file from an in-memory string.
    pub fn from_text(text: &str) -> Result<Self, ReadError> {
        let lines: Vec<&str> = text.lines().collect();

        // First pass: find the contiguous data block. A line is "data" when it
        // is non-empty and every token parses as a float. The block runs from
        // the first such line to the last consecutive such line (a non-float
        // line afterwards is a footer and ends the block).
        let mut first_data: Option<usize> = None;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            match parse_floats(line) {
                Some(vals) if !vals.is_empty() => {
                    if first_data.is_none() {
                        first_data = Some(i);
                    }
                    rows.push(vals);
                }
                _ => {
                    if first_data.is_some() {
                        break; // footer: data block has ended
                    }
                }
            }
        }
        let first_data = first_data.ok_or(ReadError::NoData)?;

        // Header lines are everything before the data block.
        let header: Vec<String> = lines[..first_data].iter().map(|s| s.to_string()).collect();

        // Column count is the width of the first data row; ragged rows are
        // truncated to it so a stray short/long trailing row can't corrupt the
        // column shape.
        let ncols = rows[0].len();
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(rows.len()); ncols];
        for row in &rows {
            if row.len() < ncols {
                continue; // skip a short row rather than pad with garbage
            }
            for (c, col) in columns.iter_mut().enumerate() {
                col.push(row[c]);
            }
        }

        let labels = guess_labels(&header, ncols);

        Ok(Self {
            path: None,
            header,
            labels,
            columns,
        })
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Number of data rows.
    pub fn nrows(&self) -> usize {
        self.columns.first().map_or(0, |c| c.len())
    }

    /// The data for column `index`, if it exists.
    pub fn column(&self, index: usize) -> Option<&[f64]> {
        self.columns.get(index).map(|c| c.as_slice())
    }

    /// Index of the first column whose label equals `name` (case-insensitive).
    pub fn label_index(&self, name: &str) -> Option<usize> {
        self.labels
            .iter()
            .position(|l| l.eq_ignore_ascii_case(name))
    }

    /// Best-guess role → column index assignments from the labels, for seeding
    /// the GUI's column chooser. Looks for energy, I0, transmission (It),
    /// fluorescence, reference, and a precomputed mu column. Absent roles are
    /// left as `None`.
    pub fn guess_roles(&self) -> RoleGuess {
        let find = |keys: &[&str]| -> Option<usize> {
            self.labels.iter().position(|l| {
                let l = l.to_ascii_lowercase();
                keys.iter().any(|k| l == *k)
            })
        };
        let find_contains = |keys: &[&str]| -> Option<usize> {
            self.labels.iter().position(|l| {
                let l = l.to_ascii_lowercase();
                keys.iter().any(|k| l.contains(k))
            })
        };
        RoleGuess {
            energy: find(&["energy", "e", "col1"]).or_else(|| find_contains(&["energy"])),
            i0: find(&["i0", "io"]).or_else(|| find_contains(&["i0"])),
            it: find(&["it", "itrans", "i1", "trans"])
                .or_else(|| find_contains(&["itrans", "trans"])),
            iflu: find(&["if", "iflu", "ifluor"])
                .or_else(|| find_contains(&["fluor", "fluo", "iff"])),
            iref: find(&["iref", "i2", "iref2"]).or_else(|| find_contains(&["iref"])),
            mu: find(&["mu", "xmu", "mutrans", "norm"])
                .or_else(|| find_contains(&["mutrans", "xmu"])),
        }
    }
}

/// Read a FEFF `chi.dat` (or any ≥2-column k/χ text file): the first numeric
/// column is `k`, the second is `χ`. Header / comment lines (`#`-prefixed FEFF
/// headers, including the `# k chi mag phase` column header) are skipped by the
/// same data-block detection [`ColumnFile`] uses. Errors with [`ReadError::NoData`]
/// when fewer than two numeric columns are present.
pub fn read_chi_dat(path: impl AsRef<Path>) -> Result<(Vec<f64>, Vec<f64>), ReadError> {
    let cf = ColumnFile::from_path(path)?;
    let k = cf.column(0).ok_or(ReadError::NoData)?.to_vec();
    let chi = cf.column(1).ok_or(ReadError::NoData)?.to_vec();
    Ok((k, chi))
}

/// Best-guess column roles (indices into [`ColumnFile::columns`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct RoleGuess {
    /// Photon-energy column.
    pub energy: Option<usize>,
    /// Incident-beam monitor I0.
    pub i0: Option<usize>,
    /// Transmitted-beam monitor It.
    pub it: Option<usize>,
    /// Fluorescence monitor If (single channel; multi-channel needs manual pick).
    pub iflu: Option<usize>,
    /// Reference-foil transmitted monitor Iref.
    pub iref: Option<usize>,
    /// A precomputed mu column, if the file already carries one.
    pub mu: Option<usize>,
}

/// Split a line into floats. Returns `None` if the line is empty, begins with a
/// comment character, or any token fails to parse — i.e. it is *not* a data row.
fn parse_floats(line: &str) -> Option<Vec<f64>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(c) = trimmed.chars().next()
        && COMMENT_CHARS.contains(c)
    {
        return None;
    }
    let mut out = Vec::new();
    for tok in trimmed
        .split([' ', '\t', ',', '\r'])
        .filter(|t| !t.is_empty())
    {
        out.push(tok.parse::<f64>().ok()?);
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Strip a single leading comment character (and following space) from a header
/// line, returning the remainder trimmed.
fn strip_comment(line: &str) -> &str {
    let t = line.trim_start();
    let t = t.strip_prefix(|c| COMMENT_CHARS.contains(c)).unwrap_or(t);
    t.trim()
}

/// Pick column labels from the header: XDI `Column.N:` tags first, then the last
/// non-empty header line split on whitespace, then `col1..colN`.
fn guess_labels(header: &[String], ncols: usize) -> Vec<String> {
    // 1. XDI-style `Column.N: <name> …` tags (name is the first token after the
    //    colon; an EPICS `|| PV` suffix and units are dropped).
    let mut tagged: Vec<Option<String>> = vec![None; ncols];
    let mut any_tag = false;
    for line in header {
        let body = strip_comment(line);
        if let Some(rest) = body.strip_prefix("Column.")
            && let Some((num, after)) = rest.split_once(':')
            && let Ok(n) = num.trim().parse::<usize>()
            && (1..=ncols).contains(&n)
        {
            let name = after.split("||").next().unwrap_or("").trim();
            if let Some(first) = name.split_whitespace().next() {
                tagged[n - 1] = Some(first.to_string());
                any_tag = true;
            }
        }
    }
    if any_tag && tagged.iter().all(|t| t.is_some()) {
        return tagged.into_iter().map(|t| t.unwrap()).collect();
    }

    // 2. Last non-empty header line, split on whitespace, if it has ncols tokens.
    if let Some(last) = header.iter().rev().find(|l| !strip_comment(l).is_empty()) {
        let toks: Vec<&str> = strip_comment(last).split_whitespace().collect();
        if toks.len() == ncols {
            return toks.into_iter().map(|s| s.to_string()).collect();
        }
    }

    // 3. Fallback.
    (1..=ncols).map(|i| format!("col{i}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_floats_classifies_lines() {
        assert_eq!(parse_floats("1.0 2.0 3.0"), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(parse_floats("  .5e1,\t-2 "), Some(vec![5.0, -2.0]));
        assert_eq!(parse_floats(".8786204E+04"), Some(vec![8786.204]));
        assert!(parse_floats("# energy mu").is_none());
        assert!(parse_floats("; a comment").is_none());
        assert!(parse_floats("").is_none());
        assert!(parse_floats("energy mu").is_none());
    }

    #[test]
    fn parses_semicolon_commented_file_with_last_line_labels() {
        let text = "; scan 1\n; I0 sensitivity = 5\n;  energy   i0   it\n\
                    100.0 10.0 5.0\n200.0 20.0 8.0\n";
        let cf = ColumnFile::from_text(text).unwrap();
        assert_eq!(cf.ncols(), 3);
        assert_eq!(cf.nrows(), 2);
        assert_eq!(cf.labels, vec!["energy", "i0", "it"]);
        assert_eq!(cf.column(0), Some([100.0, 200.0].as_slice()));
        assert_eq!(cf.label_index("I0"), Some(1));
    }

    #[test]
    fn xdi_column_tags_win_over_last_header_line() {
        let text = "# XDI/1.0\n# Column.1: energy eV\n# Column.2: i0\n# Column.3: itrans\n\
                    # something else entirely\n10.0 1.0 0.5\n20.0 2.0 0.9\n";
        let cf = ColumnFile::from_text(text).unwrap();
        assert_eq!(cf.labels, vec!["energy", "i0", "itrans"]);
    }

    #[test]
    fn multichannel_xdi_drops_pv_suffix() {
        let text = "# Column.1: Energy eV  ||  BL:En.VAL\n# Column.2: I0 counts || BL:S2\n\
                    # Column.3: mca1 counts || BL:M1\n# Column.4: mca2 counts || BL:M2\n\
                    1.0 100.0 3.0 4.0\n2.0 200.0 6.0 8.0\n";
        let cf = ColumnFile::from_text(text).unwrap();
        assert_eq!(cf.labels, vec!["Energy", "I0", "mca1", "mca2"]);
        let roles = cf.guess_roles();
        assert_eq!(roles.energy, Some(0));
        assert_eq!(roles.i0, Some(1));
    }

    #[test]
    fn fallback_labels_when_unguessable() {
        let text = "1 2 3\n4 5 6\n";
        let cf = ColumnFile::from_text(text).unwrap();
        assert_eq!(cf.labels, vec!["col1", "col2", "col3"]);
    }

    #[test]
    fn footer_after_data_is_ignored() {
        let text = "# e mu\n1.0 0.1\n2.0 0.2\n# end of scan\nnot data\n";
        let cf = ColumnFile::from_text(text).unwrap();
        assert_eq!(cf.nrows(), 2);
        assert_eq!(cf.ncols(), 2);
    }

    #[test]
    fn read_chi_dat_skips_feff_header_and_takes_k_chi() {
        // FEFF chi.dat shape: '#'-commented header (incl. the dashed rule and
        // the `# k chi mag phase` column header) then 4 numeric columns.
        let text = "# Some FEFF header\n# Mu=-0.6 kf=2.1\n\
                    #  -----------\n#       k          chi          mag           phase\n\
                        0.0500    2.705808E-01  2.719035E-01  1.472117E+00\n\
                        0.1000   -2.710386E-01  2.721822E-01  1.479092E+00\n";
        let cf = ColumnFile::from_text(text).unwrap();
        assert_eq!(cf.ncols(), 4);
        assert_eq!(cf.column(0), Some([0.05, 0.10].as_slice()));
        assert_eq!(cf.column(1), Some([0.2705808, -0.2710386].as_slice()));

        // read_chi_dat round-trips the same content from disk, taking cols 0/1.
        let path = std::env::temp_dir().join("xasdata_read_chi_dat_test.dat");
        std::fs::write(&path, text).unwrap();
        let (k, chi) = read_chi_dat(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(k, vec![0.05, 0.10]);
        assert_eq!(chi, vec![0.2705808, -0.2710386]);
    }
}
