//! Parser for `feffNNNN.dat` files, a faithful port of
//! `larch.xafs.feffdat.FeffDatFile._read`.
//!
//! The file has four regions, switched by marker lines:
//!   * line 1            -> title (cols 0..64) + version (cols 64..)
//!   * `header`          -> `Abs`/`Pot` potentials, `Gam_ch=`, `Mu=`
//!   * `path`  (after a `----` rule) -> nleg/degen/reff/... then path geometry
//!   * `arrays`(after the `k ... real[p]@#` header) -> the 7-column data block
//!
//! Only the `pha/amp/rep/lam/k` arrays and `reff/degen` feed the EXAFS
//! equation; the rest is metadata reproduced for reporting parity.

use std::fs;
use std::io;
use std::path::Path;

use crate::constants::ktoe;

#[derive(Debug, Clone, PartialEq)]
pub struct Potential {
    pub ipot: i64,
    pub iz: i64,
    pub rmt: f64,
    pub rnm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomAtom {
    pub label: String,
    pub iz: i64,
    pub ipot: i64,
    /// Atomic mass (amu), looked up from `iz` at parse time — larch sets this
    /// with `xraydb.atomic_mass(iz)`. `0.0` if `iz` is out of the table range.
    pub mass: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Parsed contents of one `feffNNNN.dat` file.
///
/// Mirrors the attributes of `larch.xafs.feffdat.FeffDatFile` that the EXAFS
/// calculation and reporting depend on.
#[derive(Debug, Clone, Default)]
pub struct FeffDatFile {
    pub filename: Option<String>,
    pub title: String,
    pub version: String,
    pub shell: String,
    pub absorber: Option<String>,

    pub nleg: usize,
    pub degen: f64,
    pub reff: f64,
    pub rnorman: f64,
    pub edge: f64,

    pub gam_ch: f64,
    pub exch: String,
    pub vmu: f64,
    pub vfermi: f64,
    pub vint: f64,
    pub rs_int: f64,

    pub potentials: Vec<Potential>,
    pub geom: Vec<GeomAtom>,

    // raw data columns
    pub k: Vec<f64>,
    pub real_phc: Vec<f64>,
    pub mag_feff: Vec<f64>,
    pub pha_feff: Vec<f64>,
    pub red_fact: Vec<f64>,
    pub lam: Vec<f64>,
    pub rep: Vec<f64>,

    // derived columns (see `_read`: pha = real_phc + pha_feff, amp = mag_feff * red_fact)
    pub pha: Vec<f64>,
    pub amp: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Header,
    Path,
    Arrays,
}

impl FeffDatFile {
    /// Read and parse a `feffNNNN.dat` file from disk.
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let p = path.as_ref();
        let text = fs::read_to_string(p)?;
        let mut f = Self::parse(&text);
        f.filename = Some(p.to_string_lossy().into_owned());
        Ok(f)
    }

    /// Largest wavenumber present in the tabulated grid.
    pub fn k_max(&self) -> f64 {
        self.k.last().copied().unwrap_or(0.0)
    }

    /// Reduced mass of the path (amu), the `rmass` FEFFDAT symbol. Matches
    /// larch's `FeffDatFile.rmass`: `1 / Σ 1/max(1, amass)` over path atoms.
    pub fn rmass(&self) -> f64 {
        let s: f64 = self.geom.iter().map(|a| 1.0 / a.mass.max(1.0)).sum();
        if s > 0.0 {
            1.0 / s
        } else {
            0.0
        }
    }

    /// Parse the textual contents of a `feffNNNN.dat` file.
    pub fn parse(text: &str) -> Self {
        let mut f = FeffDatFile::default();
        let mut mode = Mode::Header;
        let mut pcounter: usize = 0;
        let mut data: Vec<[f64; 7]> = Vec::new();

        for (idx, raw) in text.lines().enumerate() {
            // larch: line = line[:-1].strip(); drop a leading '#'; strip again.
            let trimmed = raw.trim();
            let line = trimmed.strip_prefix('#').map(str::trim).unwrap_or(trimmed);

            // line 1: title (cols 0..64) + version (cols 64..)
            if idx == 0 {
                let chars: Vec<char> = line.chars().collect();
                f.title = chars.iter().take(64).collect::<String>().trim().to_string();
                f.version = chars.iter().skip(64).collect::<String>().trim().to_string();
                continue;
            }

            // region transitions (checked before the per-mode dispatch)
            if line.starts_with('k') && line.ends_with("real[p]@#") {
                mode = Mode::Arrays;
                continue;
            } else if char_window(line, 2, 10).contains("----") {
                mode = Mode::Path;
                continue;
            }

            match mode {
                Mode::Header => parse_header_line(&mut f, line),
                Mode::Path => {
                    pcounter += 1;
                    parse_path_line(&mut f, line, pcounter);
                }
                Mode::Arrays => {
                    if let Some(row) = parse_array_row(line) {
                        data.push(row);
                    }
                }
            }
        }

        // transpose the data block into columns
        let n = data.len();
        let col = |j: usize| -> Vec<f64> { data.iter().map(|r| r[j]).collect() };
        f.k = col(0);
        f.real_phc = col(1);
        f.mag_feff = col(2);
        f.pha_feff = col(3);
        f.red_fact = col(4);
        f.lam = col(5);
        f.rep = col(6);
        f.pha = (0..n).map(|i| f.real_phc[i] + f.pha_feff[i]).collect();
        f.amp = (0..n).map(|i| f.mag_feff[i] * f.red_fact[i]).collect();
        f
    }
}

/// Python-style `s[start:stop]` over characters (saturating, never panics).
fn char_window(s: &str, start: usize, stop: usize) -> String {
    s.chars()
        .skip(start)
        .take(stop.saturating_sub(start))
        .collect()
}

/// True when `line` begins with `word` followed by a word boundary
/// (whitespace or end-of-line), matching the leading `^word\b` regexes.
fn starts_word(line: &str, word: &str) -> bool {
    match line.strip_prefix(word) {
        Some(rest) => rest
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_'),
        None => false,
    }
}

fn parse_header_line(f: &mut FeffDatFile, line: &str) {
    let is_abs = starts_word(line, "Abs");
    let is_pot = pot_with_index(line).is_some();

    if (is_abs || is_pot) && has_z_eq(line) {
        // larch: words = line.replace('=', ' ').split(); pop 'Abs'/'Pot'.
        let cleaned = line.replace('=', " ");
        let mut words: Vec<&str> = cleaned.split_whitespace().collect();
        words.remove(0); // 'Abs' or 'Pot'
        let mut ipot = 0i64;
        if is_pot {
            ipot = words.remove(0).parse().unwrap_or(0);
        }
        let iz = words.get(1).and_then(|w| w.parse().ok()).unwrap_or(0);
        let rmt = words.get(3).and_then(|w| w.parse().ok()).unwrap_or(0.0);
        let rnm = words.get(5).and_then(|w| w.parse().ok()).unwrap_or(0.0);
        if is_abs {
            if let Some(s) = words.get(6) {
                f.shell = (*s).to_string();
            }
        }
        f.potentials.push(Potential { ipot, iz, rmt, rnm });
    } else if starts_word(line, "Gam_ch") && line.contains('=') {
        // 'Gam_ch=<val> <exch...>'  (larch: split(None, 2) -> 3 tokens max)
        let cleaned = line.replace('=', " ");
        let toks: Vec<&str> = cleaned.split_whitespace().collect();
        if toks.len() >= 2 {
            f.gam_ch = toks[1].parse().unwrap_or(0.0);
            f.exch = toks[2..].join(" ");
        }
    } else if starts_word(line, "Mu") && line.contains('=') {
        // 'Mu=<vmu> kf=<kf> Vint=<vint> Rs_int=<rs>'
        let cleaned = line.replace('=', " ").replace("eV", " ");
        let w: Vec<&str> = cleaned.split_whitespace().collect();
        if w.len() >= 8 {
            f.vmu = w[1].parse().unwrap_or(0.0);
            f.vfermi = ktoe(w[3].parse().unwrap_or(0.0));
            f.vint = w[5].parse().unwrap_or(0.0);
            f.rs_int = w[7].parse().unwrap_or(0.0);
        }
    }
}

/// `^Pot\s+\d+` -> the parsed pot index.
fn pot_with_index(line: &str) -> Option<i64> {
    let rest = line.strip_prefix("Pot")?;
    if rest == line {
        return None;
    }
    let rest = rest.trim_start();
    if rest.len() == line.len() - 3 {
        return None; // no whitespace separated the digit -> not '^Pot\s+\d+'
    }
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Matches `\bZ\s*=` anywhere in the line (e.g. `Z=29` or `Z =29`).
fn has_z_eq(line: &str) -> bool {
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'Z' {
            // word boundary before Z
            let prev_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            if !prev_ok {
                continue;
            }
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'=' {
                return true;
            }
        }
    }
    false
}

fn parse_path_line(f: &mut FeffDatFile, line: &str, pcounter: usize) {
    if pcounter == 1 {
        // 'nleg deg reff rnrmav edge ...'
        let w: Vec<f64> = line
            .split_whitespace()
            .take(5)
            .filter_map(|s| s.parse().ok())
            .collect();
        if w.len() >= 5 {
            f.nleg = w[0] as usize;
            f.degen = w[1];
            f.reff = w[2];
            f.rnorman = w[3];
            f.edge = w[4];
        }
    } else if pcounter > 2 {
        // 'x y z ipot iz [label] ...'
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() < 5 {
            return;
        }
        let x = words[0].parse().unwrap_or(0.0);
        let y = words[1].parse().unwrap_or(0.0);
        let z = words[2].parse().unwrap_or(0.0);
        let ipot = words[3].parse().unwrap_or(0);
        let iz = words[4].parse().unwrap_or(0);
        // larch falls back to xraydb.atomic_symbol(iz); the example files always
        // carry an explicit label, so the table lookup is deferred to a later
        // milestone (only needed for label-less rows).
        let label = words
            .get(5)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Z{iz}"));
        if f.geom.is_empty() {
            f.absorber = Some(label.clone());
        }
        f.geom.push(GeomAtom {
            label,
            iz,
            ipot,
            mass: crate::mass::atomic_mass(iz).unwrap_or(0.0),
            x,
            y,
            z,
        });
    }
}

fn parse_array_row(line: &str) -> Option<[f64; 7]> {
    let mut row = [0.0f64; 7];
    let mut count = 0usize;
    for tok in line.split_whitespace() {
        let v: f64 = tok.parse().ok()?;
        if count < 7 {
            row[count] = v;
        }
        count += 1;
    }
    if count == 7 {
        Some(row)
    } else {
        None
    }
}
