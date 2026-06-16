//! Reduction orchestration: run the `feffit-rs` engines on a [`XasGroup`] and
//! store every stage back into the group's fields.
//!
//! These are thin adapters over `xasproc`/`xafsft` — the math (and its parity
//! with larch) lives in those crates and is tested there. Here we only wire the
//! engine outputs into the [`XasGroup`] model the GUI and batch drivers share,
//! so a tab can call `normalize` → `autobk` → `xftf` and then plot the fields.

use xafsft::{Window, xftf};
use xasproc::{AutobkParams, PreEdgeParams, autobk, autobk_delta_chi, pre_edge};

use crate::group::XasGroup;

/// Pre-edge subtraction + edge-step normalization.
///
/// Fills `e0`, `edge_step`, `pre_edge`, `post_edge`, `norm`, `flat`, and
/// `dmude`. If `params.e0` is `None`, `pre_edge` finds the edge itself.
pub fn normalize(group: &mut XasGroup, params: &PreEdgeParams) {
    let out = pre_edge(&group.energy, &group.mu, params);
    group.e0 = Some(out.e0);
    group.edge_step = Some(out.edge_step);
    group.pre_edge = Some(out.pre_edge);
    group.post_edge = Some(out.post_edge);
    group.norm = Some(out.norm);
    group.flat = Some(out.flat);
    group.dmude = Some(out.dmude);
}

/// AUTOBK background removal.
///
/// Fills `bkg`, `k`, `chi`, and (when `err_sigma > 0`) the uncertainty bands
/// `delta_bkg` / `delta_chi`. Missing `ek0`/`edge_step` in `params` default to
/// the group's `e0`/`edge_step` (so call [`normalize`] first), and the values
/// AUTOBK settles on are written back to the group.
pub fn autobk_group(group: &mut XasGroup, params: &AutobkParams, err_sigma: f64) {
    let mut p = params.clone();
    if p.ek0.is_none() {
        p.ek0 = group.e0;
    }
    if p.edge_step.is_none() {
        p.edge_step = group.edge_step;
    }

    let out = autobk(&group.energy, &group.mu, &p);

    if err_sigma > 0.0 {
        if let Some(delta) = autobk_delta_chi(&out, err_sigma) {
            group.delta_bkg = Some(delta.delta_bkg);
            group.delta_chi = Some(delta.delta_chi);
        }
    } else {
        group.delta_bkg = None;
        group.delta_chi = None;
    }

    group.e0 = Some(out.ek0);
    group.edge_step = Some(out.edge_step);
    group.bkg = Some(out.bkg);
    group.k = Some(out.k);
    group.chi = Some(out.chi);
}

/// Parameters for the forward Fourier transform `chi(k) → chi(R)`.
#[derive(Clone, Debug)]
pub struct FtParams {
    /// Window lower bound, Å⁻¹.
    pub kmin: f64,
    /// Window upper bound, Å⁻¹.
    pub kmax: f64,
    /// k-weight applied before transforming.
    pub kweight: i32,
    /// Window taper width, Å⁻¹.
    pub dk: f64,
    /// Window function.
    pub window: Window,
    /// Maximum output R, Å.
    pub rmax_out: f64,
    /// FFT array size.
    pub nfft: usize,
    /// Output k step, Å⁻¹.
    pub kstep: f64,
}

impl Default for FtParams {
    fn default() -> Self {
        Self {
            kmin: 2.0,
            kmax: 12.0,
            kweight: 2,
            dk: 1.0,
            window: Window::Hanning,
            rmax_out: 10.0,
            nfft: 2048,
            kstep: 0.05,
        }
    }
}

/// Forward FT of `chi(k)` into `chi(R)`.
///
/// Fills `r`, `chir_mag`, `chir_re`, `chir_im`. Does nothing (returns `false`)
/// if the group has no `k`/`chi` yet — run [`autobk_group`] first.
pub fn xftf_group(group: &mut XasGroup, params: &FtParams) -> bool {
    let (Some(k), Some(chi)) = (group.k.as_ref(), group.chi.as_ref()) else {
        return false;
    };
    let out = xftf(
        k,
        chi,
        params.kmin,
        params.kmax,
        params.kweight,
        params.dk,
        None,
        params.window,
        params.rmax_out,
        params.nfft,
        Some(params.kstep),
    );
    group.r = Some(out.r);
    group.chir_mag = Some(out.chir_mag);
    group.chir_re = Some(out.chir_re);
    group.chir_im = Some(out.chir_im);
    true
}
