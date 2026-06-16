//! XAS data reduction — a pure-Rust port of the `larch.xafs` preprocessing
//! chain that turns raw `mu(E)` into normalized spectra and `chi(k)`.
//!
//! This is the front-end that sits *upstream* of the `feffit` fitting core: it
//! covers edge-energy finding, pre-edge subtraction / normalization, AUTOBK
//! background removal, rebinning, and deconvolution. Each piece is verified
//! against `larch` on a real `mu(E)` dataset.

pub mod autobk;
pub mod clean;
pub mod deconvolve;
pub mod diffkk;
pub mod e0;
pub mod lincombo;
pub mod mathutils;
pub mod mback;
pub mod pca;
pub mod preedge;
pub mod rebin;
pub mod special;
pub mod xanes;

pub use autobk::{Autobk, AutobkDelta, AutobkParams, autobk, autobk_delta_chi};
pub use clean::{
    RangeSide, deglitch_point_mask, deglitch_range_mask, removed_count, select, trim_mask,
};
pub use deconvolve::{ConvParams, DeconvForm, DeconvParams, xas_convolve, xas_deconvolve};
pub use diffkk::{DiffKK, diffkk};
pub use e0::{find_e0, find_energy_step};
pub use lincombo::{Lincombo, LincomboParams, groups2matrix, lincombo_fit};
pub use mback::{Edge, Mback, MbackNorm, MbackNormParams, MbackParams, mback, mback_norm};
pub use pca::{PcaFit, PcaModel, pca_fit, pca_train};
pub use preedge::{PreEdge, PreEdgeParams, pre_edge};
pub use rebin::{RebinMethod, RebinParams, Rebinned, rebin_xafs, sort_xafs};
pub use xanes::{arctan_step, centroid, peak, valley, x_at_y};
