//! Build a FEFF scattering cluster from an explicit unit cell, and emit it as a
//! `feff.inp`.
//!
//! # Scope: no space-group expansion
//!
//! This builder takes the **full cell contents** (every atom in the
//! conventional cell) and replicates them over the lattice to gather every atom
//! within `cluster_size` of a chosen absorber. It does **not** apply
//! space-group symmetry operators to expand an asymmetric unit — larch delegates
//! that step to `pymatgen` (`SpacegroupAnalyzer`), which is not portable from
//! the larch source. So to reproduce, e.g., the 12-neighbour fcc Cu shell you
//! must enter all four face-centred sites, not the single asymmetric-unit atom.
//! See the crate-level docs for the deferred space-group work.

use crate::element::{symbol_to_z, z_to_symbol};
use crate::lattice::Lattice;
use crate::{Edge, Error};

/// One atom of the input cell, by fractional coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    /// Element symbol (e.g. `"Cu"`); resolved to `Z` for the `POTENTIALS` card.
    pub element: String,
    /// Fractional coordinates within the cell.
    pub frac: [f64; 3],
}

impl Site {
    /// A site of `element` at fractional `(x, y, z)`.
    pub fn new(element: impl Into<String>, x: f64, y: f64, z: f64) -> Self {
        Self {
            element: element.into(),
            frac: [x, y, z],
        }
    }
}

/// A crystal: a [`Lattice`] plus the full list of cell [`Site`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct Crystal {
    pub lattice: Lattice,
    pub sites: Vec<Site>,
}

/// One unique-potential declaration (a `POTENTIALS` row).
#[derive(Debug, Clone, PartialEq)]
pub struct Potential {
    pub ipot: usize,
    pub z: u32,
    pub tag: String,
}

/// One atom of the generated cluster, Cartesian (Å), absorber at the origin.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterAtom {
    pub element: String,
    pub xyz: [f64; 3],
    pub ipot: usize,
    pub tag: String,
    pub distance: f64,
}

/// A FEFF scattering cluster: the absorber-centred atom list plus the unique
/// potentials, ready to write as a `feff.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    pub title: String,
    /// Absorbing element symbol.
    pub absorber: String,
    pub edge: Edge,
    pub rmax: f64,
    pub potentials: Vec<Potential>,
    /// Atoms sorted by ascending distance (the absorber, `ipot 0`, is first).
    pub atoms: Vec<ClusterAtom>,
}

/// Canonicalise an element symbol to the periodic-table capitalisation
/// (`"fe"` → `"Fe"`); returns the input trimmed if it is not a known symbol.
fn canonical(symbol: &str) -> String {
    symbol_to_z(symbol)
        .and_then(z_to_symbol)
        .map(str::to_owned)
        .unwrap_or_else(|| symbol.trim().to_owned())
}

impl Crystal {
    /// Build the absorber-centred cluster of every atom within `cluster_size`
    /// (Å) of site `absorber_index`, for the given `edge`.
    ///
    /// Returns an error for an out-of-range absorber index, a non-positive
    /// cluster size, or an unrecognised element symbol.
    pub fn cluster(
        &self,
        absorber_index: usize,
        cluster_size: f64,
        edge: Edge,
    ) -> Result<Cluster, Error> {
        if cluster_size.is_nan() || cluster_size <= 0.0 {
            return Err(Error::BadClusterSize(cluster_size));
        }
        let abs_site = self
            .sites
            .get(absorber_index)
            .ok_or(Error::AbsorberIndex(absorber_index))?;
        // Every element must resolve to a Z (needed for POTENTIALS).
        for s in &self.sites {
            if symbol_to_z(&s.element).is_none() {
                return Err(Error::UnknownElement(s.element.clone()));
            }
        }

        let p0 = self.lattice.frac_to_cart(abs_site.frac);
        let [na, nb, nc] = self.repl_counts(cluster_size);

        let mut atoms: Vec<ClusterAtom> = Vec::new();
        for i in -na..=na {
            for j in -nb..=nb {
                for k in -nc..=nc {
                    for (sidx, s) in self.sites.iter().enumerate() {
                        // The absorber is this exact site in the origin cell; its
                        // images in other cells are ordinary neighbours.
                        if sidx == absorber_index && i == 0 && j == 0 && k == 0 {
                            continue;
                        }
                        let frac = [
                            s.frac[0] + i as f64,
                            s.frac[1] + j as f64,
                            s.frac[2] + k as f64,
                        ];
                        let cart = self.lattice.frac_to_cart(frac);
                        let rel = [cart[0] - p0[0], cart[1] - p0[1], cart[2] - p0[2]];
                        let d = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
                        if d <= cluster_size + 1e-8 {
                            atoms.push(ClusterAtom {
                                element: canonical(&s.element),
                                xyz: rel,
                                ipot: 0, // assigned below
                                tag: canonical(&s.element),
                                distance: d,
                            });
                        }
                    }
                }
            }
        }
        atoms.sort_by(|x, y| {
            x.distance
                .partial_cmp(&y.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ipot ordering (FEFF/Atoms convention): 1 = the absorber's own element
        // (if present among neighbours), then the remaining elements in order of
        // first appearance by distance.
        let abs_elem = canonical(&abs_site.element);
        let mut elem_order: Vec<String> = Vec::new();
        if atoms.iter().any(|a| a.element == abs_elem) {
            elem_order.push(abs_elem.clone());
        }
        for a in &atoms {
            if !elem_order.contains(&a.element) {
                elem_order.push(a.element.clone());
            }
        }
        for a in &mut atoms {
            // +1: ipot 0 is the absorber.
            a.ipot = elem_order.iter().position(|e| *e == a.element).unwrap() + 1;
        }

        let mut potentials = vec![Potential {
            ipot: 0,
            z: symbol_to_z(&abs_elem).unwrap(),
            tag: abs_elem.clone(),
        }];
        for (idx, elem) in elem_order.iter().enumerate() {
            potentials.push(Potential {
                ipot: idx + 1,
                z: symbol_to_z(elem).unwrap(),
                tag: elem.clone(),
            });
        }

        // The absorber itself, at the origin.
        let mut all = Vec::with_capacity(atoms.len() + 1);
        all.push(ClusterAtom {
            element: abs_elem.clone(),
            xyz: [0.0, 0.0, 0.0],
            ipot: 0,
            tag: abs_elem.clone(),
            distance: 0.0,
        });
        all.extend(atoms);

        Ok(Cluster {
            title: String::new(),
            absorber: abs_elem,
            edge,
            rmax: cluster_size,
            potentials,
            atoms: all,
        })
    }

    /// Per-axis replication counts that guarantee every atom within
    /// `cluster_size` of any cell point is generated: `⌈r / dₚ⌉ + 1`, where
    /// `dₚ` is the interplanar spacing (cell volume / opposite-face area).
    fn repl_counts(&self, cluster_size: f64) -> [i32; 3] {
        let [va, vb, vc] = self.lattice.vectors();
        let vol = self.lattice.volume().max(1e-12);
        let area = |u: [f64; 3], w: [f64; 3]| {
            let c = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ];
            (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
        };
        let count = |d_perp: f64| -> i32 {
            let d = if d_perp > 1e-12 { d_perp } else { 1e-12 };
            (cluster_size / d).ceil() as i32 + 1
        };
        [
            count(vol / area(vb, vc).max(1e-12)),
            count(vol / area(va, vc).max(1e-12)),
            count(vol / area(va, vb).max(1e-12)),
        ]
    }
}

impl Cluster {
    /// Render the cluster as a `feff.inp` (FEFF8L/FEFF10 syntax).
    pub fn to_feff_inp(&self) -> String {
        let mut s = String::new();
        for line in self.title.lines() {
            s.push_str(&format!(" TITLE {line}\n"));
        }
        if self.title.is_empty() {
            s.push_str(&format!(
                " TITLE {} {} cluster (XAFSView/atoms)\n",
                self.absorber,
                self.edge.as_str()
            ));
        }
        s.push('\n');
        s.push_str(&format!(" EDGE      {}\n", self.edge.as_str()));
        s.push_str(" S02       1.0\n");
        s.push_str(" CONTROL   1 1 1 1 1 1\n");
        s.push_str(" PRINT     1 0 0 0 0 0\n");
        s.push_str(&format!(" RMAX      {:.4}\n", self.rmax));
        s.push_str(" EXCHANGE  0\n");
        s.push('\n');

        s.push_str(" POTENTIALS\n");
        s.push_str(" *    ipot   Z   tag\n");
        for p in &self.potentials {
            s.push_str(&format!("    {:5}{:5}   {:<6}\n", p.ipot, p.z, p.tag));
        }
        s.push('\n');

        s.push_str(" ATOMS\n");
        s.push_str(" *      x          y          z      ipot  tag      distance\n");
        for a in &self.atoms {
            s.push_str(&format!(
                "{:11.5}{:11.5}{:11.5}{:5}   {:<6}{:11.5}\n",
                a.xyz[0], a.xyz[1], a.xyz[2], a.ipot, a.tag, a.distance
            ));
        }
        s.push_str(" END\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fcc Cu (a = 3.61): the full conventional cell is the four face-centred
    /// sites. Reproduces the canonical first shell (12 atoms at a/√2 = 2.5527 Å).
    fn fcc_cu() -> Crystal {
        Crystal {
            lattice: Lattice::cubic(3.61),
            sites: vec![
                Site::new("Cu", 0.0, 0.0, 0.0),
                Site::new("Cu", 0.5, 0.5, 0.0),
                Site::new("Cu", 0.5, 0.0, 0.5),
                Site::new("Cu", 0.0, 0.5, 0.5),
            ],
        }
    }

    #[test]
    fn fcc_first_shell() {
        let c = fcc_cu().cluster(0, 3.0, Edge::K).expect("cluster");
        // ipot 0 absorber + 12 first neighbours within 3.0 Å.
        let first: Vec<_> = c
            .atoms
            .iter()
            .filter(|a| a.ipot != 0 && a.distance < 2.6)
            .collect();
        assert_eq!(first.len(), 12, "fcc first shell is 12 neighbours");
        for a in &first {
            assert!(
                (a.distance - 3.61 / 2f64.sqrt()).abs() < 1e-6,
                "nn distance {} != a/√2",
                a.distance
            );
        }
        // Two potentials: ipot 0 (absorber Cu) and ipot 1 (Cu).
        assert_eq!(c.potentials.len(), 2);
        assert_eq!(
            c.potentials[0],
            Potential {
                ipot: 0,
                z: 29,
                tag: "Cu".into()
            }
        );
        assert_eq!(
            c.potentials[1],
            Potential {
                ipot: 1,
                z: 29,
                tag: "Cu".into()
            }
        );
        // Absorber is first, at the origin.
        assert_eq!(c.atoms[0].ipot, 0);
        assert_eq!(c.atoms[0].distance, 0.0);
    }

    /// Rocksalt FeO (a = 4.31): Fe at the corner/face centres, O at the edge/
    /// body centres. First shell is 6 O at a/2; second is 12 Fe at a/√2.
    fn rocksalt_feo() -> Crystal {
        let a = 4.3088;
        Crystal {
            lattice: Lattice::cubic(a),
            sites: vec![
                Site::new("Fe", 0.0, 0.0, 0.0),
                Site::new("Fe", 0.5, 0.5, 0.0),
                Site::new("Fe", 0.5, 0.0, 0.5),
                Site::new("Fe", 0.0, 0.5, 0.5),
                Site::new("O", 0.5, 0.0, 0.0),
                Site::new("O", 0.0, 0.5, 0.0),
                Site::new("O", 0.0, 0.0, 0.5),
                Site::new("O", 0.5, 0.5, 0.5),
            ],
        }
    }

    #[test]
    fn rocksalt_potentials_and_shells() {
        let a = 4.3088;
        let c = rocksalt_feo().cluster(0, 3.2, Edge::K).expect("cluster");
        // Potentials: 0=Fe(abs), 1=Fe, 2=O (absorber element first, then O).
        assert_eq!(
            c.potentials[0],
            Potential {
                ipot: 0,
                z: 26,
                tag: "Fe".into()
            }
        );
        assert_eq!(
            c.potentials[1],
            Potential {
                ipot: 1,
                z: 26,
                tag: "Fe".into()
            }
        );
        assert_eq!(
            c.potentials[2],
            Potential {
                ipot: 2,
                z: 8,
                tag: "O".into()
            }
        );
        // First shell: 6 O at a/2.
        let o1: Vec<_> = c.atoms.iter().filter(|x| x.element == "O").collect();
        let nearest_o = o1
            .iter()
            .filter(|x| (x.distance - a / 2.0).abs() < 1e-6)
            .count();
        assert_eq!(nearest_o, 6, "6 O at a/2");
        // Second shell: 12 Fe at a/√2.
        let fe2 = c
            .atoms
            .iter()
            .filter(|x| x.ipot == 1 && (x.distance - a / 2f64.sqrt()).abs() < 1e-6)
            .count();
        assert_eq!(fe2, 12, "12 Fe at a/√2");
    }

    #[test]
    fn feff_inp_round_trips_atom_count() {
        let c = fcc_cu().cluster(0, 4.0, Edge::K).expect("cluster");
        let text = c.to_feff_inp();
        assert!(text.contains("POTENTIALS"));
        assert!(text.contains("ATOMS"));
        assert!(text.contains(" EDGE      K"));
        // Parsing the emitted text recovers the same atom count and potentials.
        let parsed = crate::FeffInp::parse(&text).expect("parse");
        assert_eq!(parsed.atoms.len(), c.atoms.len());
        assert_eq!(parsed.potentials.len(), c.potentials.len());
    }

    #[test]
    fn errors() {
        let c = fcc_cu();
        assert!(matches!(
            c.cluster(99, 3.0, Edge::K),
            Err(Error::AbsorberIndex(99))
        ));
        assert!(matches!(
            c.cluster(0, 0.0, Edge::K),
            Err(Error::BadClusterSize(_))
        ));
        let bad = Crystal {
            lattice: Lattice::cubic(3.0),
            sites: vec![Site::new("Zz", 0.0, 0.0, 0.0)],
        };
        assert!(matches!(
            bad.cluster(0, 3.0, Edge::K),
            Err(Error::UnknownElement(_))
        ));
    }
}
