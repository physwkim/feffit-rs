//! Pre-edge subtraction and normalization — port of `larch.xafs.pre_edge`
//! (`preedge` core + the `pre_edge` group wrapper).
//!
//! Steps: find `E0` (if not given), fit a line (or constant) to the pre-edge
//! region of `mu*E^nvict`, fit a polynomial to the post-edge region, take the
//! extrapolated jump at `E0` as the edge step, and normalize. The wrapper adds
//! the flattened spectrum and the first/second derivatives.

use crate::xasproc::e0::find_e0;
use crate::xasproc::mathutils::{
    gradient, index_nearest, index_of, polyfit, remove_dups, remove_nans2,
};

const TINY_ENERGY: f64 = 0.00050;
const MAX_NNORM: usize = 5;

/// Round half to even (Python `round` / `numpy.round` semantics), used by
/// larch's `5.0*round(.../5.0)` range snapping.
fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// Tunable inputs to [`pre_edge`]; `None` reproduces larch's auto-defaults.
#[derive(Debug, Clone)]
pub struct PreEdgeParams {
    pub e0: Option<f64>,
    pub step: Option<f64>,
    pub nnorm: Option<usize>,
    pub nvict: i32,
    pub npre: i32,
    pub pre1: Option<f64>,
    pub pre2: Option<f64>,
    pub norm1: Option<f64>,
    pub norm2: Option<f64>,
    pub make_flat: bool,
}

impl Default for PreEdgeParams {
    /// larch's `pre_edge` defaults: a linear pre-edge (`npre=1`) and a flattened
    /// post-edge (`make_flat=true`); `E0`/step/ranges/`nnorm` auto-chosen.
    ///
    /// `npre`/`make_flat` are deliberately *not* the all-zero/false values a
    /// `derive(Default)` would give — those produce a constant pre-edge and skip
    /// flattening (`flat == norm`), silently diverging from larch. This is the
    /// single canonical constructor so no call site can pick the non-larch one.
    fn default() -> Self {
        PreEdgeParams {
            e0: None,
            step: None,
            nnorm: None,
            nvict: 0,
            npre: 1,
            pre1: None,
            pre2: None,
            norm1: None,
            norm2: None,
            make_flat: true,
        }
    }
}

/// Output of [`pre_edge`], on the cleaned (sorted, deduped) energy grid.
#[derive(Debug, Clone)]
pub struct PreEdge {
    pub energy: Vec<f64>,
    pub e0: f64,
    pub edge_step: f64,
    pub norm: Vec<f64>,
    pub flat: Vec<f64>,
    pub pre_edge: Vec<f64>,
    pub post_edge: Vec<f64>,
    pub dmude: Vec<f64>,
    pub d2mude: Vec<f64>,
    pub norm_coefs: Vec<f64>,
    pub precoefs: [f64; 2],
    pub nvict: i32,
    pub nnorm: usize,
    pub norm1: f64,
    pub norm2: f64,
    pub pre1: f64,
    pub pre2: f64,
}

/// Result of the inner `preedge` (no flattening / derivatives).
pub(crate) struct PreedgeCore {
    pub(crate) energy: Vec<f64>,
    pub(crate) e0: f64,
    pub(crate) ie0: usize,
    pub(crate) edge_step: f64,
    pub(crate) norm: Vec<f64>,
    pub(crate) pre_edge: Vec<f64>,
    pub(crate) post_edge: Vec<f64>,
    pub(crate) norm_coefs: Vec<f64>,
    pub(crate) precoefs: [f64; 2],
    pub(crate) nvict: i32,
    pub(crate) nnorm: usize,
    pub(crate) norm1: f64,
    pub(crate) norm2: f64,
    pub(crate) pre1: f64,
    pub(crate) pre2: f64,
}

/// `larch.xafs.pre_edge.preedge`: the pure-numeric pre/post-edge fit.
pub(crate) fn preedge_core(energy_in: &[f64], mu_in: &[f64], p: &PreEdgeParams) -> PreedgeCore {
    let (energy, mu) = remove_nans2(energy_in, mu_in);
    let energy = remove_dups(&energy, TINY_ENERGY);
    let n = energy.len();
    assert!(n > 1, "energy array must have at least 2 points");

    // E0
    let mut e0 = p.e0.unwrap_or(f64::NAN);
    if e0.is_nan() || e0 < energy[1] || e0 > energy[n - 2] {
        e0 = find_e0(&energy, &mu);
    }
    let ie0 = index_nearest(&energy, e0);
    let e0 = energy[ie0];

    let emin = energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax = energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // pre-edge range
    let mut pre1 = p.pre1.unwrap_or_else(|| {
        if ie0 > 20 {
            5.0 * round_half_even((energy[1] - e0) / 5.0)
        } else {
            2.0 * round_half_even((energy[1] - e0) / 2.0)
        }
    });
    pre1 = pre1.max(emin - e0);
    let mut pre2 = p.pre2.unwrap_or(0.5 * pre1);
    if pre1 > pre2 {
        std::mem::swap(&mut pre1, &mut pre2);
    }

    let e_shift: Vec<f64> = energy.iter().map(|&e| e - e0).collect();
    let ipre1 = index_of(&e_shift, pre1);
    let ipre2 = index_of(&e_shift, pre2);
    let mut nvict = p.nvict;
    let mut npre = p.npre;
    if (ipre2 as i64 - ipre1 as i64) < 3 {
        nvict = 0;
        npre = 0;
    }

    // post-edge range
    let mut norm2 = p
        .norm2
        .unwrap_or_else(|| 5.0 * round_half_even((emax - e0) / 5.0));
    if norm2 < 0.0 {
        norm2 = emax - e0 - norm2;
    }
    norm2 = norm2.min(emax - e0);
    let mut norm1 = p
        .norm1
        .unwrap_or_else(|| 25.0_f64.min(5.0 * round_half_even(norm2 / 15.0)));
    if norm1 > norm2 {
        std::mem::swap(&mut norm1, &mut norm2);
    }
    norm1 = norm1.min(norm2 - 2.0);

    let mut nnorm = match p.nnorm {
        Some(v) => v,
        None if norm2 - norm1 < 30.0 => 0,
        None if norm2 - norm1 < 300.0 => 1,
        None => 2,
    };
    nnorm = nnorm.min(MAX_NNORM);

    // ---- pre-edge fit ----
    let mut p1 = index_of(&energy, pre1 + e0);
    let mut p2 = index_nearest(&energy, pre2 + e0);
    let mut precoefs = [0.0f64; 2];
    let pre_edge: Vec<f64> = if npre == 0 {
        if p2 == p1 {
            p2 += 1;
        }
        let mu_mean = mu[p1..p2].iter().sum::<f64>() / (p2 - p1) as f64;
        precoefs = [mu_mean, 0.0];
        vec![mu_mean; n]
    } else {
        if p2 - p1 < 2 {
            p2 = n.min(p1 + 2);
        }
        let omu: Vec<f64> = energy
            .iter()
            .zip(&mu)
            .map(|(&e, &m)| m * e.powi(nvict))
            .collect();
        let ex = &energy[p1..p2];
        let mx = &omu[p1..p2];
        let c = polyfit(ex, mx, 1);
        precoefs[0] = c[0];
        precoefs[1] = if c.len() > 1 { c[1] } else { 0.0 };
        energy
            .iter()
            .map(|&e| (precoefs[0] + e * precoefs[1]) * e.powi(-nvict))
            .collect()
    };

    // ---- normalization (post-edge) fit ----
    p1 = index_of(&energy, norm1 + e0);
    p2 = index_nearest(&energy, norm2 + e0);
    if p1 > n - 3 {
        p1 = n - 3;
    }
    if (p2 as i64 - p1 as i64) < 2 {
        p1 -= 2;
        nnorm = 0;
    } else if (p2 as i64 - p1 as i64) < 5 {
        nnorm = nnorm.min(1);
    }
    let presub: Vec<f64> = (p1..p2).map(|i| mu[i] - pre_edge[i]).collect();
    let coefs = polyfit(&energy[p1..p2], &presub, nnorm);

    let mut post_edge = pre_edge.clone();
    for (deg, &c) in coefs.iter().enumerate() {
        for (pe, &e) in post_edge.iter_mut().zip(&energy) {
            *pe += c * e.powi(deg as i32);
        }
    }
    let edge_step = p
        .step
        .unwrap_or(post_edge[ie0] - pre_edge[ie0])
        .abs()
        .max(1.0e-12);
    let norm: Vec<f64> = mu
        .iter()
        .zip(&pre_edge)
        .map(|(&m, &pe)| (m - pe) / edge_step)
        .collect();

    PreedgeCore {
        energy,
        e0,
        ie0,
        edge_step,
        norm,
        pre_edge,
        post_edge,
        norm_coefs: coefs,
        precoefs,
        nvict,
        nnorm,
        norm1,
        norm2,
        pre1,
        pre2,
    }
}

/// `larch.xafs.pre_edge.pre_edge`: full pre-edge subtraction + normalization,
/// adding the flattened spectrum and first/second derivatives.
pub fn pre_edge(energy_in: &[f64], mu_in: &[f64], p: &PreEdgeParams) -> PreEdge {
    let (mut energy, mut mu) = remove_nans2(energy_in, mu_in);
    // sort if out of order
    let mut order: Vec<usize> = (0..energy.len()).collect();
    order.sort_by(|&i, &j| {
        energy[i]
            .partial_cmp(&energy[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if order.iter().enumerate().any(|(i, &o)| i != o) {
        energy = order.iter().map(|&i| energy[i]).collect();
        mu = order.iter().map(|&i| mu[i]).collect();
    }
    energy = remove_dups(&energy, TINY_ENERGY);

    let core = preedge_core(&energy, &mu, p);
    let ie0 = core.ie0;

    // flattened spectrum: subtract (post_edge - pre_edge)/edge_step, pinned at ie0
    let mut flat = core.norm.clone();
    if p.make_flat {
        let flat_residue_ie0 = (core.post_edge[ie0] - core.pre_edge[ie0]) / core.edge_step;
        for (((fl, &nm), &po), &pr) in flat
            .iter_mut()
            .zip(&core.norm)
            .zip(&core.post_edge)
            .zip(&core.pre_edge)
        {
            let flat_residue = (po - pr) / core.edge_step;
            *fl = nm - flat_residue + flat_residue_ie0;
        }
        flat[..ie0].copy_from_slice(&core.norm[..ie0]);
    }

    let dmude = {
        let gn = gradient(&core.norm);
        let ge = gradient(&core.energy);
        gn.iter().zip(&ge).map(|(a, b)| a / b).collect::<Vec<_>>()
    };
    let d2mude = {
        let gd = gradient(&dmude);
        let ge = gradient(&core.energy);
        gd.iter().zip(&ge).map(|(a, b)| a / b).collect::<Vec<_>>()
    };

    PreEdge {
        energy: core.energy,
        e0: core.e0,
        edge_step: core.edge_step,
        norm: core.norm,
        flat,
        pre_edge: core.pre_edge,
        post_edge: core.post_edge,
        dmude,
        d2mude,
        norm_coefs: core.norm_coefs,
        precoefs: core.precoefs,
        nvict: core.nvict,
        nnorm: core.nnorm,
        norm1: core.norm1,
        norm2: core.norm2,
        pre1: core.pre1,
        pre2: core.pre2,
    }
}
