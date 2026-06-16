//! `feffinp` — the **input side** of the FEFF boundary: build a `feff.inp`
//! scattering cluster from a crystal cell, and parse a `feff.inp` back into its
//! atoms and potentials.
//!
//! This is the counterpart to [`feffdat`](https://docs.rs/feffdat), which parses
//! the `feffNNNN.dat` files FEFF *produces*. Together they bracket the external
//! FEFF run: `feffinp` writes the input, [`feffrun`](https://docs.rs/feffrun)
//! drives FEFF, `feffdat` reads the output.
//!
//! # What this crate does and does not do
//!
//! - [`Crystal::cluster`] takes the **full contents of a unit cell** (every
//!   atom, already symmetry-expanded) and gathers every atom within a cutoff
//!   radius of a chosen absorber, assigning FEFF unique potentials. It then
//!   emits a `feff.inp` via [`Cluster::to_feff_inp`].
//! - [`FeffInp::parse`] reads a `feff.inp`'s `POTENTIALS`/`ATOMS` cards back
//!   into structured data — for round-tripping and for the 3D site viewer.
//!
//! **Space-group expansion** (asymmetric unit + space group → full cell) is what
//! larch delegates to `pymatgen` (`SpacegroupAnalyzer`). Here it is provided by
//! `expand_sites` / `Crystal::expand` behind the optional `spacegroup`
//! cargo feature, which pulls in the pure-Rust
//! [`crystallographic-group`](https://docs.rs/crystallographic-group) crate for
//! the 230 space groups' symmetry operators. With the feature off, the crate's
//! core stays dependency-free and the caller must supply the already-expanded
//! cell to [`Crystal::cluster`]; enabling it raises the effective MSRV to that
//! of `crystallographic-group` and its dependency tree.

use std::fmt;

mod crystal;
mod element;
mod lattice;
mod parse;
#[cfg(feature = "spacegroup")]
mod spacegroup;

pub use crystal::{Cluster, ClusterAtom, Crystal, Potential, Site};
pub use element::{symbol_to_z, z_to_symbol};
pub use lattice::Lattice;
pub use parse::{FeffAtom, FeffInp};
#[cfg(feature = "spacegroup")]
pub use spacegroup::expand_sites;

/// An X-ray absorption edge selectable for the FEFF calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Edge {
    #[default]
    K,
    L1,
    L2,
    L3,
}

impl Edge {
    /// The FEFF `EDGE` card token (`"K"`, `"L1"`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Edge::K => "K",
            Edge::L1 => "L1",
            Edge::L2 => "L2",
            Edge::L3 => "L3",
        }
    }

    /// Parse a FEFF edge token (case-insensitive); `None` if unrecognised.
    pub fn from_str_ci(s: &str) -> Option<Edge> {
        match s.trim().to_ascii_uppercase().as_str() {
            "K" => Some(Edge::K),
            "L1" => Some(Edge::L1),
            "L2" => Some(Edge::L2),
            "L3" => Some(Edge::L3),
            _ => None,
        }
    }
}

/// Errors from building a cluster or parsing a `feff.inp`.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The absorber index is past the end of the site list.
    AbsorberIndex(usize),
    /// The cluster radius was not strictly positive.
    BadClusterSize(f64),
    /// An element symbol did not resolve to an atomic number.
    UnknownElement(String),
    /// A `feff.inp` could not be parsed (with a human-readable reason).
    Parse(String),
    /// A space-group number was not in the range 1‥230.
    SpaceGroup(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AbsorberIndex(i) => write!(f, "absorber index {i} is out of range"),
            Error::BadClusterSize(r) => write!(f, "cluster size must be > 0 (got {r})"),
            Error::UnknownElement(s) => write!(f, "unknown element symbol `{s}`"),
            Error::Parse(m) => write!(f, "feff.inp parse error: {m}"),
            Error::SpaceGroup(n) => write!(f, "space-group number {n} is out of range 1..=230"),
        }
    }
}

impl std::error::Error for Error {}
