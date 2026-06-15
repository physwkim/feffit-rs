//! AUTOBK background removal — port of `larch.xafs.autobk.autobk`.
//!
//! Fits a smooth spline background `mu_0(E)` such that the low-`R` content of
//! `chi(k) = (mu - mu_0)/edge_step` below `rbkg` is minimized. The spline knot
//! values are the fit parameters; the residual is the real/imag parts of the
//! windowed forward FT of `chi` truncated at `irbkg`, plus optional spline
//! clamps. This mirrors larch exactly: FITPACK `splrep(s=0)`/`splev` for the
//! spline (via `rusty-fitpack`), an interpolating spline for `chi` on the
//! output grid (larch's `UnivariateSpline(..., s=0)`), and MINPACK `lmdif`
//! (via the `lm` crate) for the least-squares solve.

use std::f64::consts::PI;

use lm::{lmdif, LmConfig};
use num_complex::Complex64;
use rusty_fitpack::{splev, splrep};
use xafsft::{ftwindow, xftf_fast, Window};

use crate::mathutils::{index_nearest, index_of, remove_dups};
use crate::preedge::{pre_edge, PreEdgeParams};

/// `larch.xafs.xafsutils.KTOE` — `1e20 * hbar^2 / (2 m_e e)`, taken at the
/// exact runtime value (the source comment 3.8099819442818976 is stale from an
/// older CODATA set; the live `scipy.constants` value is the one below).
const KTOE: f64 = 3.809_982_116_154_859_7;
/// `ETOK = 1/KTOE`, the eV→k^2 conversion factor.
const ETOK: f64 = 1.0 / KTOE;
/// smallest tolerated energy step, in eV (`larch` `TINY_ENERGY`).
const TINY_ENERGY: f64 = 0.00050;
/// `xftf_fast`'s default `kstep`. larch's residual calls `xftf_fast(...,
/// nfft=nfft)` without forwarding `kstep`, so the FT scaling always uses 0.05
/// regardless of the autobk output `kstep`. Replicated here for parity.
const FT_KSTEP: f64 = 0.05;

/// Tunable inputs to [`autobk`]; defaults reproduce larch's `autobk` defaults.
#[derive(Debug, Clone)]
pub struct AutobkParams {
    /// distance (Ang) in chi(R) above which signal is ignored. Default 1.
    pub rbkg: f64,
    /// number of spline knots; `None` auto-determines from `rbkg`.
    pub nknots: Option<usize>,
    /// edge energy (eV); `None` runs `pre_edge` to find it.
    pub ek0: Option<f64>,
    /// edge step; `None` runs `pre_edge` to find it.
    pub edge_step: Option<f64>,
    /// minimum k. Default 0.
    pub kmin: f64,
    /// maximum k; `None` (or negative) uses the full data range.
    pub kmax: Option<f64>,
    /// k weight for the FFT. Default 1.
    pub kweight: i32,
    /// FFT window parameter. Default 0.1.
    pub dk: f64,
    /// FFT window function. Default Hanning.
    pub win: Window,
    /// FFT array size. Default 2048.
    pub nfft: usize,
    /// output k step. Default 0.05.
    pub kstep: f64,
    /// number of end-point clamps. Default 3.
    pub nclamp: usize,
    /// low-energy clamp weight. Default 0.
    pub clamp_lo: f64,
    /// high-energy clamp weight. Default 1.
    pub clamp_hi: f64,
}

impl Default for AutobkParams {
    fn default() -> Self {
        AutobkParams {
            rbkg: 1.0,
            nknots: None,
            ek0: None,
            edge_step: None,
            kmin: 0.0,
            kmax: None,
            kweight: 1,
            dk: 0.1,
            win: Window::Hanning,
            nfft: 2048,
            kstep: 0.05,
            nclamp: 3,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
        }
    }
}

/// Output of [`autobk`], on the deduplicated energy grid.
#[derive(Debug, Clone)]
pub struct Autobk {
    /// background `mu_0(E)` over the full energy grid.
    pub bkg: Vec<f64>,
    /// `(mu - bkg)/edge_step` over the full energy grid.
    pub chie: Vec<f64>,
    /// output k grid.
    pub k: Vec<f64>,
    /// `chi(k)/edge_step` on the output grid.
    pub chi: Vec<f64>,
    /// edge energy used.
    pub ek0: f64,
    /// rbkg used (raised to `2*rgrid` if smaller).
    pub rbkg: f64,
    /// edge step used.
    pub edge_step: f64,
    /// initial (pre-fit) background over the full energy grid.
    pub init_bkg: Vec<f64>,
    /// initial (pre-fit) `chi(k)/edge_step` on the output grid.
    pub init_chi: Vec<f64>,
    /// final spline coefficients.
    pub coefs: Vec<f64>,
    /// number of spline knot parameters.
    pub nspl: usize,
    /// FT residual cutoff index.
    pub irbkg: usize,
    /// index of `ek0` in the energy grid.
    pub iek0: usize,
    /// upper energy index used for the spline fit.
    pub iemax: usize,
    /// minimum k of the fit window.
    pub kmin: f64,
    /// maximum k of the fit window.
    pub kmax: f64,
}

/// `larch.xafs.autobk.spline_eval`: evaluate `bkg = splev(kraw)` and
/// `chi = UnivariateSpline(kraw, mu-bkg, s=0)(kout)`.
fn spline_eval(
    kraw: &[f64],
    mu: &[f64],
    knots: &[f64],
    coefs: &[f64],
    order: usize,
    kout: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let bkg = splev(knots.to_vec(), coefs.to_vec(), order, kraw.to_vec(), 0);
    let resid: Vec<f64> = mu.iter().zip(&bkg).map(|(m, b)| m - b).collect();
    // larch's UnivariateSpline(kraw, mu-bkg, s=0) is the FITPACK interpolating
    // (s=0) cubic spline; splrep with default args reproduces it.
    let (t2, c2, k2) = splrep(
        kraw.to_vec(),
        resid,
        None,
        None,
        None,
        Some(order),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let chi = splev(t2, c2, k2, kout.to_vec(), 0);
    (bkg, chi)
}

/// Build the least-squares residual for a trial coefficient vector, matching
/// larch's `_resid`: realimag of the windowed FT head plus the spline clamps.
#[allow(clippy::too_many_arguments)]
fn resid(
    vcoefs: &[f64],
    ncoef: usize,
    kraw: &[f64],
    mu: &[f64],
    knots: &[f64],
    order: usize,
    kout: &[f64],
    ftwin: &[f64],
    nfft: usize,
    irbkg: usize,
    nclamp: usize,
    clamp_lo: f64,
    clamp_hi: f64,
) -> Vec<f64> {
    let nspl = vcoefs.len();
    let mut coefs = vec![vcoefs[nspl - 1]; ncoef];
    coefs[..nspl].copy_from_slice(vcoefs);

    let (_bkg, chi) = spline_eval(kraw, mu, knots, &coefs, order, kout);

    let windowed: Vec<Complex64> = chi
        .iter()
        .zip(ftwin)
        .map(|(c, w)| Complex64::new(c * w, 0.0))
        .collect();
    let ft = xftf_fast(&windowed, nfft, FT_KSTEP);

    let mut out = Vec::with_capacity(2 * irbkg + 2 * nclamp);
    for c in ft.iter().take(irbkg) {
        out.push(c.re);
        out.push(c.im);
    }
    if nclamp == 0 {
        return out;
    }
    let mean_sq = out.iter().map(|v| v * v).sum::<f64>() / out.len() as f64;
    let scale = 0.1 + 10.0 * mean_sq;
    let nch = chi.len();
    let nc = nclamp.min(nch);
    for &c in chi.iter().take(nc) {
        out.push(clamp_lo.abs() * scale * c);
    }
    for &c in chi.iter().skip(nch - nc) {
        out.push(clamp_hi.abs() * scale * c);
    }
    out
}

/// `larch.xafs.autobk.autobk`: remove the XAFS background, returning the
/// background `mu_0(E)`, the output k grid, and `chi(k)`.
pub fn autobk(energy_in: &[f64], mu_in: &[f64], p: &AutobkParams) -> Autobk {
    assert_eq!(
        energy_in.len(),
        mu_in.len(),
        "energy and mu length mismatch"
    );
    let energy = remove_dups(energy_in, TINY_ENERGY);
    let mu = mu_in.to_vec();
    let n = energy.len();
    assert!(n > 2, "need at least 3 data points");

    let emin = energy.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax = energy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // resolve ek0 and edge_step, running pre_edge if needed
    let mut ek0 = p.ek0.filter(|&e| e >= emin && e <= emax);
    let mut edge_step = p.edge_step;
    if ek0.is_none() || edge_step.is_none() {
        let pe = pre_edge(&energy, &mu, &PreEdgeParams::defaults());
        if ek0.is_none() {
            ek0 = Some(pe.e0);
        }
        if edge_step.is_none() {
            edge_step = Some(pe.edge_step);
        }
    }
    let ek0 = ek0.expect("ek0 could not be determined");
    let edge_step = edge_step.expect("edge_step could not be determined");

    let kstep = p.kstep;
    let nfft = p.nfft;
    let kmin = p.kmin;

    let iek0 = index_of(&energy, ek0);
    let rgrid = PI / (kstep * nfft as f64);
    let rbkg = p.rbkg.max(2.0 * rgrid); // larch raises rbkg, leaves rgrid as is

    // ungridded k (kraw)
    let kraw: Vec<f64> = energy[iek0..]
        .iter()
        .map(|&e| {
            let d = e - ek0;
            d.signum() * (ETOK * d.abs()).sqrt()
        })
        .collect();
    let kraw_max = kraw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let kmax = match p.kmax {
        None => kraw_max,
        Some(v) if v < 0.0 => kraw_max,
        Some(v) => 0.0_f64.max(kraw_max.min(v)),
    };

    // gridded output k
    let nkout = (1.01 + kmax / kstep) as usize;
    let kout: Vec<f64> = (0..nkout).map(|i| kstep * i as f64).collect();

    let iemax = n.min(2 + index_of(&energy, ek0 + kmax * kmax / ETOK)) - 1;

    // FT window times k-weighting
    let win = ftwindow(&kout, Some(kmin), Some(kmax), p.dk, Some(p.dk), p.win);
    let ftwin: Vec<f64> = kout
        .iter()
        .zip(&win)
        .map(|(&k, &w)| k.powi(p.kweight) * w)
        .collect();

    // number of spline knots and FT cutoff (irbkg uses the un-clamped nspl)
    let mut nspl = 1 + (2.0 * rbkg * (kmax - kmin) / PI) as usize;
    let irbkg = (1.0 + (nspl as f64 - 1.0) * PI / (2.0 * rgrid * (kmax - kmin))) as usize;
    if let Some(nk) = p.nknots {
        nspl = nk;
    }
    nspl = nspl.clamp(5, 128);

    // initial spline knot positions and y-values
    let mut spl_k = vec![0.0; nspl];
    let mut spl_y = vec![0.0; nspl];
    let nkraw = kraw.len();
    for i in 0..nspl {
        let q = kmin + i as f64 * (kmax - kmin) / (nspl as f64 - 1.0);
        let ik = index_nearest(&kraw, q);
        let i1 = (ik + 5).min(nkraw - 1);
        let i2 = ik.saturating_sub(5);
        spl_k[i] = kraw[ik];
        spl_y[i] = (2.0 * mu[ik + iek0] + mu[i1 + iek0] + mu[i2 + iek0]) / 4.0;
    }

    let order = 3;
    let (knots, mut coefs, _k) = splrep(
        spl_k.clone(),
        spl_y.clone(),
        None,
        None,
        None,
        Some(order),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    // pad trailing coefs with the last meaningful one (larch coefs[nspl:]=coefs[nspl-1])
    let last = coefs[nspl - 1];
    for c in coefs.iter_mut().skip(nspl) {
        *c = last;
    }
    let ncoefs = coefs.len();

    let kraw_fit: Vec<f64> = kraw[..(iemax - iek0 + 1)].to_vec();
    let mu_fit: Vec<f64> = mu[iek0..=iemax].to_vec();

    let (initbkg, initchi) = spline_eval(&kraw_fit, &mu_fit, &knots, &coefs, order, &kout);

    // least-squares fit over the nspl knot values
    let vcoefs: Vec<f64> = coefs[..nspl].to_vec();
    let knots_r = knots.clone();
    let kout_r = kout.clone();
    let ftwin_r = ftwin.clone();
    let fcn = |v: &[f64]| -> Vec<f64> {
        resid(
            v, ncoefs, &kraw_fit, &mu_fit, &knots_r, order, &kout_r, &ftwin_r, nfft, irbkg,
            p.nclamp, p.clamp_lo, p.clamp_hi,
        )
    };
    let cfg = LmConfig {
        ftol: 1.0e-6,
        xtol: 1.0e-6,
        gtol: 0.0,
        maxfev: 2000 * (ncoefs as i32 + 1),
        epsfcn: 1.0e-6,
        factor: 100.0,
    };
    let result = lmdif(fcn, &vcoefs, &cfg);
    let best = result.x;

    // assemble final coefficients (larch final_coefs[:nspl]=best; [nspl:]=best[-1])
    let mut final_coefs = coefs.clone();
    final_coefs[..nspl].copy_from_slice(&best);
    let best_last = best[nspl - 1];
    for c in final_coefs.iter_mut().skip(nspl) {
        *c = best_last;
    }

    let (bkg, chi) = spline_eval(&kraw_fit, &mu_fit, &knots, &final_coefs, order, &kout);

    // background over the full energy grid
    let mut obkg = mu.clone();
    obkg[iek0..iek0 + bkg.len()].copy_from_slice(&bkg);

    let mut init_bkg = mu.clone();
    init_bkg[iek0..iek0 + initbkg.len()].copy_from_slice(&initbkg);

    let chie: Vec<f64> = mu
        .iter()
        .zip(&obkg)
        .map(|(&m, &b)| (m - b) / edge_step)
        .collect();
    let chi_out: Vec<f64> = chi.iter().map(|&c| c / edge_step).collect();
    let init_chi: Vec<f64> = initchi.iter().map(|&c| c / edge_step).collect();

    Autobk {
        bkg: obkg,
        chie,
        k: kout,
        chi: chi_out,
        ek0,
        rbkg,
        edge_step,
        init_bkg,
        init_chi,
        coefs: final_coefs,
        nspl,
        irbkg,
        iek0,
        iemax,
        kmin,
        kmax,
    }
}
