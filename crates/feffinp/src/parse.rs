//! Parse a `feff.inp`'s `TITLE` / `EDGE` / `POTENTIALS` / `ATOMS` cards back
//! into structured data — for round-tripping and for the 3D site viewer.
//!
//! This is a *reader* for the cards XAFSView cares about, not a full FEFF input
//! validator: unrecognised cards (`SCF`, `EXCHANGE`, `RPATH`, …) are skipped,
//! and only the geometry-bearing cards are retained.

use crate::crystal::Potential;
use crate::element::z_to_symbol;
use crate::{Edge, Error};

/// One atom parsed from an `ATOMS` card row.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffAtom {
    pub xyz: [f64; 3],
    pub ipot: usize,
    pub tag: String,
    /// The listed distance column, if present.
    pub distance: Option<f64>,
}

/// The geometry-bearing contents of a `feff.inp`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FeffInp {
    pub title: Vec<String>,
    pub edge: Option<Edge>,
    pub potentials: Vec<Potential>,
    pub atoms: Vec<FeffAtom>,
}

/// Which data section the parser is currently reading rows from.
#[derive(Clone, Copy, PartialEq)]
enum Section {
    None,
    Potentials,
    Atoms,
}

/// True if `t` begins like a numeric data field (so the line is a data row, not
/// a new card keyword).
fn is_data_token(t: &str) -> bool {
    t.chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit() || c == '+' || c == '-' || c == '.')
}

impl FeffInp {
    /// The potential declaration for `ipot`, if present.
    pub fn potential(&self, ipot: usize) -> Option<&Potential> {
        self.potentials.iter().find(|p| p.ipot == ipot)
    }

    /// Parse `feff.inp` text. Comment lines (`*…`) and inline trailing `*…`
    /// comments are stripped; unrecognised cards end the current data section.
    pub fn parse(text: &str) -> Result<FeffInp, Error> {
        let mut out = FeffInp::default();
        let mut section = Section::None;

        for raw in text.lines() {
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.starts_with('*') {
                continue;
            }
            // Strip an inline trailing comment (`… * site_info`).
            let line = trimmed.split('*').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let first = tokens[0];

            // A data row inside POTENTIALS/ATOMS starts with a number.
            if matches!(section, Section::Potentials | Section::Atoms) && is_data_token(first) {
                match section {
                    Section::Potentials => {
                        if let Some(p) = parse_potential(&tokens) {
                            out.potentials.push(p);
                        }
                    }
                    Section::Atoms => {
                        if let Some(a) = parse_atom(&tokens)? {
                            out.atoms.push(a);
                        }
                    }
                    Section::None => unreachable!(),
                }
                continue;
            }

            // Otherwise the line is a card keyword.
            match first.to_ascii_uppercase().as_str() {
                "TITLE" => {
                    section = Section::None;
                    let rest = line
                        .split_once(char::is_whitespace)
                        .map(|x| x.1)
                        .unwrap_or("")
                        .trim();
                    if !rest.is_empty() {
                        out.title.push(rest.to_owned());
                    }
                }
                "EDGE" => {
                    section = Section::None;
                    if let Some(t) = tokens.get(1) {
                        out.edge = Edge::from_str_ci(t);
                    }
                }
                "HOLE" => {
                    section = Section::None;
                    // HOLE index: 1=K, 2=L1, 3=L2, 4=L3.
                    if let Some(n) = tokens.get(1).and_then(|t| t.parse::<u32>().ok()) {
                        out.edge = match n {
                            1 => Some(Edge::K),
                            2 => Some(Edge::L1),
                            3 => Some(Edge::L2),
                            4 => Some(Edge::L3),
                            _ => out.edge,
                        };
                    }
                }
                "POTENTIALS" => section = Section::Potentials,
                "ATOMS" => section = Section::Atoms,
                "END" => section = Section::None,
                _ => section = Section::None,
            }
        }
        Ok(out)
    }
}

/// Parse a `POTENTIALS` row: `ipot  Z  [tag]`.
fn parse_potential(tokens: &[&str]) -> Option<Potential> {
    let ipot = tokens.first()?.parse::<usize>().ok()?;
    let z = tokens.get(1)?.parse::<u32>().ok()?;
    let tag = tokens
        .get(2)
        .map(|s| (*s).to_owned())
        .or_else(|| z_to_symbol(z).map(str::to_owned))
        .unwrap_or_default();
    Some(Potential { ipot, z, tag })
}

/// Parse an `ATOMS` row: `x  y  z  ipot  [tag]  [distance]`.
fn parse_atom(tokens: &[&str]) -> Result<Option<FeffAtom>, Error> {
    if tokens.len() < 4 {
        return Ok(None);
    }
    let parse_f = |i: usize| -> Result<f64, Error> {
        tokens[i]
            .parse::<f64>()
            .map_err(|_| Error::Parse(format!("bad coordinate `{}`", tokens[i])))
    };
    let x = parse_f(0)?;
    let y = parse_f(1)?;
    let z = parse_f(2)?;
    let ipot = tokens[3]
        .parse::<usize>()
        .map_err(|_| Error::Parse(format!("bad ipot `{}`", tokens[3])))?;
    let tag = tokens.get(4).map(|s| (*s).to_owned()).unwrap_or_default();
    let distance = tokens.get(5).and_then(|s| s.parse::<f64>().ok());
    Ok(Some(FeffAtom {
        xyz: [x, y, z],
        ipot,
        tag,
        distance,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
 * a comment line
 TITLE copper test
 HOLE 1
 EDGE  K
 S02   1.0
 CONTROL 1 1 1 1 1 1
 RMAX  5.0

 POTENTIALS
 *  ipot  Z  tag
     0   29  Cu
     1   29  Cu

 ATOMS
 *   x        y        z     ipot tag    distance
    0.00000  0.00000  0.00000  0  Cu1    0.00000
    1.80500  1.80500  0.00000  1  Cu     2.55266  * Cu.1
   -1.80500 -1.80500  0.00000  1  Cu     2.55266
 END
 * trailing comment
";

    #[test]
    fn parses_sample() {
        let f = FeffInp::parse(SAMPLE).expect("parse");
        assert_eq!(f.title, vec!["copper test".to_owned()]);
        // EDGE K overrides HOLE 1 (both map to K here anyway).
        assert_eq!(f.edge, Some(Edge::K));
        assert_eq!(f.potentials.len(), 2);
        assert_eq!(f.potential(0).unwrap().z, 29);
        assert_eq!(f.atoms.len(), 3);
        assert_eq!(f.atoms[0].ipot, 0);
        assert_eq!(f.atoms[0].tag, "Cu1");
        assert_eq!(f.atoms[1].distance, Some(2.55266));
        assert!((f.atoms[1].xyz[0] - 1.805).abs() < 1e-9);
    }

    #[test]
    fn hole_maps_to_edge_when_no_edge_card() {
        let text = " HOLE 3\n ATOMS\n 0 0 0 0 Fe\n END\n";
        let f = FeffInp::parse(text).expect("parse");
        assert_eq!(f.edge, Some(Edge::L2));
        assert_eq!(f.atoms.len(), 1);
        assert_eq!(f.atoms[0].distance, None);
    }

    #[test]
    fn unknown_card_ends_section() {
        // An `RPATH` card between ATOMS rows must end the ATOMS section so the
        // `7.0` argument is not mistaken for an atom.
        let text = " ATOMS\n 0 0 0 0 Fe\n RPATH 7.0\n 1 0 0 1 O\n END\n";
        let f = FeffInp::parse(text).expect("parse");
        assert_eq!(f.atoms.len(), 1, "second row is after a card, not in ATOMS");
    }
}
