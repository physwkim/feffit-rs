//! Space-group symmetry expansion: asymmetric unit + space group → full cell.
//!
//! This lifts the "enter the full cell" restriction of [`Crystal::cluster`]. It
//! is the piece larch delegates to `pymatgen`; here it is done with the pure-Rust
//! [`crystallographic-group`](https://docs.rs/crystallographic-group) crate,
//! which provides the 230 space groups' symmetry operators (via Hall symbols).
//!
//! Gated behind the `spacegroup` cargo feature so the crate's core stays
//! dependency-free. Given a space-group number (1‥230, default *setting*) and a
//! list of asymmetric-unit [`Site`]s, [`expand_sites`] applies every symmetry
//! operation to every site, wraps coordinates into the unit cell, and
//! deduplicates — yielding the full cell to feed into [`Crystal::cluster`].

use crystallographic_group::HallSymbolNotation;
use crystallographic_group::database::{DEFAULT_SPACE_GROUP_SYMBOLS, LookUpSpaceGroup};

use crate::{Crystal, Error, Site};

/// Wrap a fractional coordinate into `[0, 1)`.
fn wrap01(x: f64) -> f64 {
    let v = x - x.floor();
    if v >= 1.0 { 0.0 } else { v }
}

/// True if two fractional positions coincide under periodic boundaries
/// (per-axis difference reduced into `(−½, ½]`, compared to a tolerance).
fn frac_close(a: [f64; 3], b: [f64; 3]) -> bool {
    a.iter().zip(b.iter()).all(|(&p, &q)| {
        let mut d = p - q;
        d -= d.round();
        d.abs() < 1.0e-4
    })
}

/// Expand `sites` (an asymmetric unit) into the full unit cell for international
/// space-group number `space_group` (1‥230), using that group's default
/// setting. Each output [`Site`] keeps its source element; coincident images are
/// removed.
///
/// Space group 1 (P1) has only the identity, so the input is returned unchanged
/// (modulo wrapping into the cell).
pub fn expand_sites(space_group: u32, sites: &[Site]) -> Result<Vec<Site>, Error> {
    if !(1..=230).contains(&space_group) {
        return Err(Error::SpaceGroup(space_group));
    }
    // Default-setting Hall symbol for this international number.
    let hall = DEFAULT_SPACE_GROUP_SYMBOLS
        .get_hall_symbol((space_group - 1) as usize)
        .ok_or(Error::SpaceGroup(space_group))?;
    let notation = HallSymbolNotation::try_from_str(hall)
        .map_err(|_| Error::Parse(format!("could not parse Hall symbol `{hall}`")))?;
    // Every symmetry operation (point-group ops × lattice centring translations).
    let ops = notation.general_positions().derive_full_sets();

    let mut out: Vec<Site> = Vec::new();
    for s in sites {
        let f = s.frac;
        for set in &ops {
            for op in set {
                let m = op.to_f64_mat();
                let np = [
                    wrap01(m[(0, 0)] * f[0] + m[(0, 1)] * f[1] + m[(0, 2)] * f[2] + m[(0, 3)]),
                    wrap01(m[(1, 0)] * f[0] + m[(1, 1)] * f[1] + m[(1, 2)] * f[2] + m[(1, 3)]),
                    wrap01(m[(2, 0)] * f[0] + m[(2, 1)] * f[1] + m[(2, 2)] * f[2] + m[(2, 3)]),
                ];
                if !out
                    .iter()
                    .any(|o| o.element == s.element && frac_close(o.frac, np))
                {
                    out.push(Site {
                        element: s.element.clone(),
                        frac: np,
                    });
                }
            }
        }
    }
    Ok(out)
}

impl Crystal {
    /// Return a copy of this crystal with its asymmetric-unit sites expanded to
    /// the full cell for international space-group `space_group` (1‥230). The
    /// lattice is unchanged. See [`expand_sites`].
    pub fn expand(&self, space_group: u32) -> Result<Crystal, Error> {
        Ok(Crystal {
            lattice: self.lattice,
            sites: expand_sites(space_group, &self.sites)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, Lattice};

    fn has_site(sites: &[Site], elem: &str, frac: [f64; 3]) -> bool {
        sites
            .iter()
            .any(|s| s.element == elem && frac_close(s.frac, frac))
    }

    #[test]
    fn p1_is_identity() {
        let s = vec![Site::new("Cu", 0.1, 0.2, 0.3)];
        let out = expand_sites(1, &s).expect("expand");
        assert_eq!(out.len(), 1);
        assert!(frac_close(out[0].frac, [0.1, 0.2, 0.3]));
    }

    #[test]
    fn fm3m_expands_single_atom_to_fcc() {
        // Fm-3m (No. 225): one atom at the origin → the four F-centred sites.
        let out = expand_sites(225, &[Site::new("Cu", 0.0, 0.0, 0.0)]).expect("expand");
        assert_eq!(out.len(), 4, "fcc has 4 atoms in the conventional cell");
        for frac in [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ] {
            assert!(has_site(&out, "Cu", frac), "missing fcc site {frac:?}");
        }
    }

    #[test]
    fn im3m_expands_single_atom_to_bcc() {
        // Im-3m (No. 229): body-centred → 2 atoms.
        let out = expand_sites(229, &[Site::new("Fe", 0.0, 0.0, 0.0)]).expect("expand");
        assert_eq!(out.len(), 2);
        assert!(has_site(&out, "Fe", [0.0, 0.0, 0.0]));
        assert!(has_site(&out, "Fe", [0.5, 0.5, 0.5]));
    }

    #[test]
    fn fm3m_rocksalt_two_sublattices() {
        // NaCl (Fm-3m): Na at 4a (0,0,0), Cl at 4b (½,½,½) → 4 Na + 4 Cl.
        let out = expand_sites(
            225,
            &[
                Site::new("Na", 0.0, 0.0, 0.0),
                Site::new("Cl", 0.5, 0.5, 0.5),
            ],
        )
        .expect("expand");
        assert_eq!(out.iter().filter(|s| s.element == "Na").count(), 4);
        assert_eq!(out.iter().filter(|s| s.element == "Cl").count(), 4);
    }

    #[test]
    fn expanded_fcc_cluster_matches_manual_full_cell() {
        // Expanding 1 asymmetric-unit Cu atom via Fm-3m and building the cluster
        // reproduces the canonical 12-neighbour first shell at a/√2.
        let asym = Crystal {
            lattice: Lattice::cubic(3.61),
            sites: vec![Site::new("Cu", 0.0, 0.0, 0.0)],
        };
        let full = asym.expand(225).expect("expand");
        assert_eq!(full.sites.len(), 4);
        let c = full.cluster(0, 3.0, Edge::K).expect("cluster");
        let first = c
            .atoms
            .iter()
            .filter(|a| a.ipot != 0 && a.distance < 2.6)
            .count();
        assert_eq!(first, 12, "fcc first shell is 12 neighbours");
    }

    #[test]
    fn out_of_range() {
        assert!(matches!(expand_sites(0, &[]), Err(Error::SpaceGroup(0))));
        assert!(matches!(
            expand_sites(231, &[]),
            Err(Error::SpaceGroup(231))
        ));
    }
}
