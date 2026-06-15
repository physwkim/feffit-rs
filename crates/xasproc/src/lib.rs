//! XAS data reduction — a pure-Rust port of the `larch.xafs` preprocessing
//! chain that turns raw `mu(E)` into normalized spectra and `chi(k)`.
//!
//! This is the front-end that sits *upstream* of the `feffit` fitting core: it
//! covers edge-energy finding, pre-edge subtraction / normalization, AUTOBK
//! background removal, rebinning, and deconvolution. Each piece is verified
//! against `larch` on a real `mu(E)` dataset.

pub mod autobk;
pub mod e0;
pub mod mathutils;
pub mod preedge;

pub use autobk::{autobk, Autobk, AutobkParams};
pub use e0::{find_e0, find_energy_step};
pub use preedge::{pre_edge, PreEdge, PreEdgeParams};
