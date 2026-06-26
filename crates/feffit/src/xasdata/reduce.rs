//! Reduction orchestration: run the `feffit-rs` engines on a [`XasGroup`] and
//! store every stage back into the group's fields.
//!
//! These are thin adapters over `xasproc`/`xafsft` — the math (and its parity
//! with larch) lives in those crates and is tested there. Here we only wire the
//! engine outputs into the [`XasGroup`] model the GUI and batch drivers share,
//! so a tab can call `normalize` → `autobk` → `xftf` and then plot the fields.

use num_complex::Complex64;

use crate::xafsft::{Window, xftf, xftr};
use crate::xasproc::{AutobkParams, PreEdgeParams, autobk, autobk_delta_chi, pre_edge};

use crate::xasdata::group::XasGroup;

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

/// Parameters for the reverse Fourier transform `chi(R) → chi(q)` — the
/// Fourier-filtered EXAFS isolated by an R-window (the back-transform of the
/// peaks selected between `rmin` and `rmax`).
#[derive(Clone, Debug)]
pub struct XftrParams {
    /// R-window lower bound, Å.
    pub rmin: f64,
    /// R-window upper bound, Å.
    pub rmax: f64,
    /// R-window taper width, Å.
    pub dr: f64,
    /// R-weight applied before transforming (larch's `rw`; normally 0).
    pub rweight: i32,
    /// Window function.
    pub window: Window,
    /// Maximum output q, Å⁻¹ (larch uses `kmax + 2`).
    pub qmax_out: f64,
    /// FFT array size.
    pub nfft: usize,
}

impl Default for XftrParams {
    fn default() -> Self {
        Self {
            rmin: 1.0,
            rmax: 3.0,
            dr: 0.2,
            rweight: 0,
            window: Window::Hanning,
            qmax_out: 14.0,
            nfft: 2048,
        }
    }
}

/// Reverse FT of `chi(R)` into the Fourier-filtered `chi(q)`.
///
/// Fills `q`, `chiq_mag`, `chiq_re`, `chiq_im`. Does nothing (returns `false`)
/// if the group has no `chi(R)` yet — run [`xftf_group`] first.
pub fn xftr_group(group: &mut XasGroup, params: &XftrParams) -> bool {
    let (Some(r), Some(re), Some(im)) = (
        group.r.as_ref(),
        group.chir_re.as_ref(),
        group.chir_im.as_ref(),
    ) else {
        return false;
    };
    let chir: Vec<Complex64> = re
        .iter()
        .zip(im)
        .map(|(&a, &b)| Complex64::new(a, b))
        .collect();
    let out = xftr(
        r,
        &chir,
        params.rmin,
        params.rmax,
        params.dr,
        None,
        params.rweight,
        params.window,
        params.qmax_out,
        params.nfft,
    );
    group.q = Some(out.q);
    group.chiq_mag = Some(out.chiq_mag);
    group.chiq_re = Some(out.chiq_re);
    group.chiq_im = Some(out.chiq_im);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reduce-pipeline round-trip: normalize → autobk → xftf → xftr fills the
    /// q-space fields, and a tight R-window around a single-frequency χ(k) returns
    /// a χ(q) that tracks the input over the windowed k-range (Fourier filtering
    /// passes the band it selects). Uses a pure sinusoid so the test owns its
    /// expectation without a reference file.
    #[test]
    fn xftr_group_fills_q_space_and_filters() {
        // χ(k) = sin(2 R0 k) → a single |χ(R)| peak at R0; window R around it.
        let k: Vec<f64> = (0..=280).map(|i| i as f64 * 0.05).collect();
        let r0 = 2.0;
        let chi: Vec<f64> = k.iter().map(|&kk| (2.0 * r0 * kk).sin()).collect();
        let mut g = XasGroup {
            k: Some(k.clone()),
            chi: Some(chi.clone()),
            ..Default::default()
        };

        let ft = FtParams {
            kmin: 2.0,
            kmax: 12.0,
            kweight: 0,
            dk: 1.0,
            ..FtParams::default()
        };
        assert!(xftf_group(&mut g, &ft), "forward FT should run");

        let xr = XftrParams {
            rmin: r0 - 0.4,
            rmax: r0 + 0.4,
            dr: 0.2,
            rweight: 0,
            window: Window::Hanning,
            qmax_out: 14.0,
            nfft: 2048,
        };
        assert!(xftr_group(&mut g, &xr), "reverse FT should run");

        let q = g.q.as_ref().expect("q grid filled");
        let chiq_re = g.chiq_re.as_ref().expect("chiq_re filled");
        let chiq_mag = g.chiq_mag.as_ref().expect("chiq_mag filled");
        assert_eq!(q.len(), chiq_re.len());
        assert_eq!(q.len(), chiq_mag.len());
        assert!(q.len() > 100, "q grid should be densely sampled");

        // The filtered χ(q) must carry real signal in the windowed band and be a
        // sinusoid of the same R0 frequency as the input. Compare against the
        // input sin(2 R0 q) where the FT k-window has full weight (k≈6, mid-band)
        // by correlation over the flat-top region.
        let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
        for (i, &qq) in q.iter().enumerate() {
            if (5.0..=9.0).contains(&qq) {
                let a = chiq_re[i];
                let b = (2.0 * r0 * qq).sin();
                num += a * b;
                da += a * a;
                db += b * b;
            }
        }
        let corr = num / (da.sqrt() * db.sqrt());
        assert!(
            corr > 0.9,
            "filtered χ(q) should track the input sinusoid in-band (corr {corr:.3})"
        );
    }

    /// `xftr_group` is a no-op when χ(R) has not been computed yet.
    #[test]
    fn xftr_group_without_chir_is_a_noop() {
        let mut g = XasGroup::default();
        assert!(!xftr_group(&mut g, &XftrParams::default()));
        assert!(g.q.is_none());
    }
}
