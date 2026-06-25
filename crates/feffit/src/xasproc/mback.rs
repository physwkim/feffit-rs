//! MBACK normalization — port of `larch.xafs.mback.mback` and
//! `larch.xafs.mback.mback_norm`.
//!
//! Both routines match measured `mu(E)` to the tabulated imaginary scattering
//! factor `f''(E)` (Chantler `f2`) by a least-squares fit, then derive an edge
//! step and normalized spectrum from the matched curve.
//!
//! The tabulated `f2` (and optional `f1`) are *inputs* here rather than looked
//! up internally: larch pulls them from `xraydb.f2_chantler`, and isolating the
//! table source keeps the algorithm bit-exact verifiable independent of which
//! atomic-data library supplies the factors. Feed the same `f2` array larch
//! would compute (`xraydb.f2_chantler(z, energy)` on the *deduplicated* energy
//! grid) to reproduce larch to round-off.
//!
//! Only the default `fit_erfc=False` MBACK is implemented: with the error
//! function amplitude `a` fixed at 0 the `erfc` term drops out entirely, so no
//! special function or emission-line energy is needed. `fit_erfc=True` would
//! additionally require an `erfc` and lmfit's bounded-parameter transform (the
//! `xi` width carries `min=0`), which the unbounded `lm` solve does not model.
//!
//! The fits use `lmfit.minimize(method='leastsq')` with every parameter varying
//! and none bounded, which reduces exactly to `scipy.optimize.leastsq` — the
//! `lm` crate reproduces that bit-for-bit.

use crate::lm::{LmConfig, lmdif};

use crate::xasproc::e0::find_e0;
use crate::xasproc::mathutils::{index_nearest, index_of, remove_dups};
use crate::xasproc::preedge::{PreEdgeParams, pre_edge, preedge_core};

/// smallest tolerated energy step, in eV (`larch` `TINY_ENERGY`).
const TINY_ENERGY: f64 = 0.00050;
/// `larch` `mback`'s `MAXORDER` cap on the normalization polynomial order.
const MAXORDER: usize = 6;

/// lmfit's `leastsq` tolerances as passed by `mback` / `mback_norm`
/// (`gtol=ftol=xtol=epsfcn=1e-5`), with lmfit's default `maxfev` (`2 *
/// max_nfev`, `max_nfev` defaulting to 100000) and `factor=100`.
fn mback_lmcfg() -> LmConfig {
    LmConfig {
        ftol: 1.0e-5,
        xtol: 1.0e-5,
        gtol: 1.0e-5,
        maxfev: 200_000,
        epsfcn: 1.0e-5,
        factor: 100.0,
    }
}

/// `numpy.linspace(start, stop, num)` with `endpoint=True`: the last sample is
/// pinned exactly to `stop`, and `num == 1` yields `[start]`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    v[num - 1] = stop;
    v
}

/// Tunable inputs to [`mback`]; defaults reproduce larch's `mback` defaults.
#[derive(Debug, Clone)]
pub struct MbackParams {
    /// edge energy (eV); `None` (or out of `[energy[1], energy[-2]]`) runs
    /// `find_e0`.
    pub e0: Option<f64>,
    /// low pre-edge range relative to `e0`; `None` uses `min(energy) - e0`.
    pub pre1: Option<f64>,
    /// high pre-edge range relative to `e0`. Default -50.
    pub pre2: f64,
    /// low post-edge range relative to `e0`. Default 100.
    pub norm1: f64,
    /// high post-edge range relative to `e0`; `None` uses `max(energy) - e0`.
    pub norm2: Option<f64>,
    /// order of the normalization polynomial (clamped to `[0, 6]`). Default 3.
    pub order: usize,
    /// use the Lee & Xiang residual extension. Default false.
    pub leexiang: bool,
}

impl Default for MbackParams {
    fn default() -> Self {
        MbackParams {
            e0: None,
            pre1: None,
            pre2: -50.0,
            norm1: 100.0,
            norm2: None,
            order: 3,
            leexiang: false,
        }
    }
}

/// Output of [`mback`], on the deduplicated energy grid.
#[derive(Debug, Clone)]
pub struct Mback {
    /// edge energy used (snapped to the grid).
    pub e0: f64,
    /// edge step `pre_f2.edge_step / s`.
    pub edge_step: f64,
    /// matched spectrum `s*mu - norm_function`.
    pub fpp: Vec<f64>,
    /// normalized spectrum `(s*mu - pre_f2.pre_edge) / pre_f2.edge_step`.
    pub norm: Vec<f64>,
    /// the normalization curve `a*erfc(...) + sum c_i*(E-e0)^i` (here `a=0`).
    pub norm_function: Vec<f64>,
    /// fitted data scale `s`.
    pub s: f64,
    /// fitted polynomial coefficients `c0 .. c_order`.
    pub coefs: Vec<f64>,
    /// echoed tabulated `f2`.
    pub f2: Vec<f64>,
    /// echoed tabulated `f1` (if supplied).
    pub f1: Option<Vec<f64>>,
}

/// `larch.xafs.mback.mback` (default `fit_erfc=False`).
///
/// `f2` is the tabulated `f''(E)` on the deduplicated `energy` grid (larch:
/// `xraydb.f2_chantler(z, energy)`); `f1` is the optional `f'(E)`, echoed back
/// untouched. `energy`/`mu`/`f2` must share the same length.
pub fn mback(
    energy_in: &[f64],
    mu_in: &[f64],
    f2: &[f64],
    f1: Option<&[f64]>,
    p: &MbackParams,
) -> Mback {
    let order = p.order.min(MAXORDER);

    let energy = remove_dups(energy_in, TINY_ENERGY);
    let n = energy.len();
    assert!(n > 1, "energy array must have at least 2 points");
    assert_eq!(mu_in.len(), n, "energy and mu length mismatch");
    assert_eq!(f2.len(), n, "energy and f2 length mismatch");
    let mu = mu_in;

    let emin = energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax = energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // resolve e0, then snap to the nearest grid point
    let mut e0 = p.e0.unwrap_or(f64::NAN);
    if e0.is_nan() || e0 < energy[1] || e0 > energy[n - 2] {
        e0 = find_e0(&energy, mu);
    }
    let ie0 = index_nearest(&energy, e0);
    let e0 = energy[ie0];

    // pre-/post-edge ranges (relative to e0), matching mback's own resolution
    let mut pre1 = p.pre1.unwrap_or(emin - e0);
    let mut pre2 = p.pre2;
    let mut norm1 = p.norm1;
    let mut norm2 = p.norm2.unwrap_or(emax - e0);
    if norm2 < 0.0 {
        norm2 = emax - e0 - norm2;
    }
    pre1 = pre1.max(emin - e0);
    norm2 = norm2.min(emax - e0);
    if pre1 > pre2 {
        std::mem::swap(&mut pre1, &mut pre2);
    }
    if norm1 > norm2 {
        std::mem::swap(&mut norm1, &mut norm2);
    }

    let p1 = index_of(&energy, pre1 + e0);
    let mut p2 = index_nearest(&energy, pre2 + e0);
    let n1 = index_nearest(&energy, norm1 + e0);
    let n2 = index_of(&energy, norm2 + e0);
    if (p2 as i64 - p1 as i64) < 2 {
        p2 = n.min(p1 + 2);
    }
    // larch quirk: the n2-n1 check writes p2 (not n2). Replicated for parity.
    if (n2 as i64 - n1 as i64) < 2 {
        p2 = n.min(p1 + 2);
    }

    // theta: 1 where the fit is evaluated, 0 elsewhere
    let mut theta = vec![0.0f64; n];
    let pre_hi = (p2 + 1).min(n);
    let nor_hi = (n2 + 1).min(n);
    for t in &mut theta[p1..pre_hi] {
        *t = 1.0;
    }
    for t in &mut theta[n1..nor_hi] {
        *t = 1.0;
    }

    // weights: sqrt of the included-point count in each region
    let mut weight = vec![1.0f64; n];
    let wpre = weight[p1..pre_hi].iter().sum::<f64>().sqrt();
    for w in &mut weight[p1..pre_hi] {
        *w = wpre;
    }
    let wnor = weight[n1..nor_hi].iter().sum::<f64>().sqrt();
    for w in &mut weight[n1..nor_hi] {
        *w = wnor;
    }

    // eoff = energy - e0, precomputed for the residual and norm_function
    let eoff: Vec<f64> = energy.iter().map(|&e| e - e0).collect();

    // residual over [s, c0, c1, ..., c_order]
    let resid = |v: &[f64]| -> Vec<f64> {
        let s = v[0];
        (0..n)
            .map(|j| {
                let mut norm = v[1]; // c0 (a*erfc term is 0 with a=0)
                let mut pw = eoff[j];
                for c in v.iter().skip(2) {
                    norm += c * pw;
                    pw *= eoff[j];
                }
                let mut func = (f2[j] + norm - s * mu[j]) * theta[j] / weight[j];
                if p.leexiang {
                    func = func / s * mu[j];
                }
                func
            })
            .collect()
    };

    let mut v0 = vec![0.0f64; order + 2];
    v0[0] = 1.0; // s
    let result = lmdif(resid, &v0, &mback_lmcfg());
    let s = result.x[0];
    let coefs: Vec<f64> = result.x[1..].to_vec();

    // norm_function = c0 + sum_{i=1..order} c_i * eoff^i  (a*erfc term = 0)
    let norm_function: Vec<f64> = (0..n)
        .map(|j| {
            let mut nf = coefs[0];
            let mut pw = eoff[j];
            for c in coefs.iter().skip(1) {
                nf += c * pw;
                pw *= eoff[j];
            }
            nf
        })
        .collect();

    let fpp: Vec<f64> = (0..n).map(|j| s * mu[j] - norm_function[j]).collect();

    // edge step / normalization from f2 + norm_function
    let f2_plus: Vec<f64> = (0..n).map(|j| f2[j] + norm_function[j]).collect();
    let pre_f2 = preedge_core(
        &energy,
        &f2_plus,
        &PreEdgeParams {
            e0: Some(e0),
            nnorm: Some(2),
            nvict: 0,
            npre: 1,
            pre1: Some(pre1),
            pre2: Some(pre2),
            norm1: Some(norm1),
            norm2: Some(norm2),
            ..Default::default()
        },
    );
    let edge_step = pre_f2.edge_step / s;
    let norm: Vec<f64> = (0..n)
        .map(|j| (s * mu[j] - pre_f2.pre_edge[j]) / pre_f2.edge_step)
        .collect();

    Mback {
        e0,
        edge_step,
        fpp,
        norm,
        norm_function,
        s,
        coefs,
        f2: f2.to_vec(),
        f1: f1.map(|a| a.to_vec()),
    }
}

/// Tunable inputs to [`mback_norm`]; `None` reproduces larch's auto-defaults.
#[derive(Debug, Clone, Default)]
pub struct MbackNormParams {
    /// edge energy (eV); `None` uses the `pre_edge` value.
    pub e0: Option<f64>,
    /// low pre-edge range relative to `e0`; `None` uses the `pre_edge` value.
    pub pre1: Option<f64>,
    /// high pre-edge range relative to `e0`; `None` uses the `pre_edge` value.
    pub pre2: Option<f64>,
    /// low post-edge range relative to `e0`; `None` uses the `pre_edge` value.
    pub norm1: Option<f64>,
    /// high post-edge range relative to `e0`; `None` uses the `pre_edge` value.
    pub norm2: Option<f64>,
    /// post-edge polynomial degree; `None` uses the `pre_edge` value.
    pub nnorm: Option<usize>,
    /// energy exponent for the pre-edge fit. Default 1.
    pub nvict: i32,
    /// next-higher absorption-edge energy (eV) for L2/L3 `norm2` capping
    /// (`xray_edge(z, 'L2'/'L1')`); unused for K edges.
    pub next_edge_energy: Option<f64>,
    /// the absorption edge label; only its leading 'l'/'l2'/'l3' matters.
    pub edge: Edge,
}

impl MbackNormParams {
    /// larch's defaults (`nvict=1`, K edge, all ranges auto).
    pub fn defaults() -> Self {
        MbackNormParams {
            nvict: 1,
            edge: Edge::K,
            ..Default::default()
        }
    }
}

/// Absorption edge label, distinguishing only the cases `mback_norm` branches on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Edge {
    /// K edge (default; no `norm2` capping).
    #[default]
    K,
    /// L3 edge — cap `norm2` below the L2 edge.
    L3,
    /// L2 edge — cap `norm2` below the L1 edge.
    L2,
    /// L1 edge — no capping.
    L1,
}

/// Output of [`mback_norm`], on the deduplicated energy grid.
#[derive(Debug, Clone)]
pub struct MbackNorm {
    /// normalized spectrum `mu_pre / edge_step_mback`.
    pub norm: Vec<f64>,
    /// MBACK edge step (`preedge(model).edge_step`).
    pub edge_step: f64,
    /// the `pre_edge` polynomial edge step (`edge_step_poly`).
    pub edge_step_poly: f64,
    /// matched spectrum `model + pre_edge`.
    pub mback_mu: Vec<f64>,
    /// the scaled/shifted `f2` model `(offset + slope*E + f2) * scale`.
    pub model: Vec<f64>,
    /// fitted slope.
    pub slope: f64,
    /// fitted offset.
    pub offset: f64,
    /// fitted scale.
    pub scale: f64,
    /// fit weights.
    pub weights: Vec<f64>,
}

/// `larch.xafs.mback.mback_norm`: simplified MBACK for normalization.
///
/// `f2` is the tabulated `f''(E)` on the deduplicated `energy` grid
/// (`xraydb.f2_chantler(z, energy)`). Runs `pre_edge` internally for the
/// polynomial normalization and edge ranges.
pub fn mback_norm(energy_in: &[f64], mu_in: &[f64], f2: &[f64], p: &MbackNormParams) -> MbackNorm {
    let energy = remove_dups(energy_in, TINY_ENERGY);
    let n = energy.len();
    assert!(n > 1, "energy array must have at least 2 points");
    assert_eq!(mu_in.len(), n, "energy and mu length mismatch");
    assert_eq!(f2.len(), n, "energy and f2 length mismatch");
    let mu = mu_in;

    // pre_edge for the polynomial normalization (larch passes nvict through,
    // everything else auto).
    let pe = pre_edge(
        &energy,
        mu,
        &PreEdgeParams {
            e0: p.e0,
            nnorm: p.nnorm,
            nvict: p.nvict,
            npre: 1,
            pre1: p.pre1,
            pre2: p.pre2,
            norm1: p.norm1,
            norm2: p.norm2,
            make_flat: true,
            ..Default::default()
        },
    );

    let e0 = p.e0.unwrap_or(pe.e0);
    let pre1 = p.pre1.unwrap_or(pe.pre1);
    let pre2 = p.pre2.unwrap_or(pe.pre2);
    let nvict = p.nvict; // larch default 1 (not None) → stays as given
    let norm1 = p.norm1.unwrap_or(pe.norm1);
    let mut norm2 = p.norm2.unwrap_or(pe.norm2);
    let nnorm = p.nnorm.unwrap_or(pe.nnorm);

    let mu_pre: Vec<f64> = (0..n).map(|j| mu[j] - pe.pre_edge[j]).collect();

    let emax = energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if norm2 < 0.0 {
        norm2 = emax - e0 - norm2;
    }
    // avoid L2 and higher edges
    match p.edge {
        Edge::L3 | Edge::L2 => {
            if let Some(en) = p.next_edge_energy {
                norm2 = norm2.min(en - e0);
            }
        }
        _ => {}
    }

    let ipre2 = index_of(&energy, e0 + pre2);
    let inor1 = index_of(&energy, e0 + norm1);
    let inor2 = (index_of(&energy, e0 + norm2) + 1).min(n);

    let mut weights = vec![1.0f64; n];
    for w in &mut weights[ipre2.min(n)..] {
        *w = 0.0;
    }
    let ramp = linspace(0.1, 1.0, inor2.saturating_sub(inor1));
    for (w, &r) in weights[inor1..inor2].iter_mut().zip(&ramp) {
        *w = r;
    }

    // fit (offset + slope*E + f2) * scale to mu_pre, weighted
    let resid = |v: &[f64]| -> Vec<f64> {
        let (slope, offset, scale) = (v[0], v[1], v[2]);
        (0..n)
            .map(|j| {
                let model = (offset + slope * energy[j] + f2[j]) * scale;
                weights[j] * (model - mu_pre[j])
            })
            .collect()
    };
    let v0 = vec![0.0, -f2[0], f2[n - 1]];
    let result = lmdif(resid, &v0, &mback_lmcfg());
    let (slope, offset, scale) = (result.x[0], result.x[1], result.x[2]);

    let model: Vec<f64> = (0..n)
        .map(|j| (offset + slope * energy[j] + f2[j]) * scale)
        .collect();
    let mback_mu: Vec<f64> = (0..n).map(|j| model[j] + pe.pre_edge[j]).collect();

    let pre_f2 = preedge_core(
        &energy,
        &model,
        &PreEdgeParams {
            e0: Some(e0),
            nnorm: Some(nnorm),
            nvict,
            npre: 1,
            pre1: Some(pre1),
            pre2: Some(pre2),
            norm1: Some(norm1),
            norm2: Some(norm2),
            ..Default::default()
        },
    );
    let step_new = pre_f2.edge_step;
    let norm: Vec<f64> = mu_pre.iter().map(|&m| m / step_new).collect();

    MbackNorm {
        norm,
        edge_step: step_new,
        edge_step_poly: pe.edge_step,
        mback_mu,
        model,
        slope,
        offset,
        scale,
        weights,
    }
}
