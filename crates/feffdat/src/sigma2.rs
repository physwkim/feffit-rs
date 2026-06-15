//! Debye-Waller factor (σ²) models for a Feff path: the Einstein model
//! ([`sigma2_eins`]) and the correlated-Debye model ([`sigma2_debye`]).
//!
//! Ported from `larch/xafs/sigma2_models.py` — `sigma2_eins` from the closed
//! form there, and `sigma2_debye` from `sigma2_correldebye_py` (itself a port of
//! Feff6 `sigms.f`, © 1993 University of Washington; J. Rehr, S. Zabinsky,
//! M. Newville). larch's production `sigma2_debye` calls the Feff6 C library;
//! this is the equivalent pure-Rust math, verified against the pure-Python
//! `sigma2_correldebye_py` reference.
//!
//! The GNXAS distribution model (`gnxas`) is a separate g(r) feature (and needs
//! the Γ function); it is not ported here.

use crate::parser::GeomAtom;

/// `1e20 · ħ²/(2 k_B · amu)` in Å²·K·amu — larch's `EINS_FACTOR`, computed from
/// `scipy.constants` (`hbar`, `k`, `atomic_mass`).
pub const EINS_FACTOR: f64 = 24.254367090467216;

/// σ² (Å²) for a Feff path under the Einstein model at temperature `t` (K) and
/// Einstein temperature `theta` (K). Port of `sigma2_models.sigma2_eins`.
///
/// Note: the reduced mass here floors each atomic mass at `0.1` amu — distinct
/// from [`crate::FeffDatFile::rmass`], which floors at `1.0` — so this does not
/// reuse the `rmass` symbol.
pub fn sigma2_eins(t: f64, theta: f64, geom: &[GeomAtom]) -> f64 {
    let theta = theta.max(1.0e-5);
    let t = t.max(1.0e-5);
    let mut rmass: f64 = geom.iter().map(|a| 1.0 / a.mass.max(0.1)).sum();
    rmass = 1.0 / rmass.max(1.0e-12);
    EINS_FACTOR / (theta * rmass * (theta / (2.0 * t)).tanh())
}

/// σ² (Å²) for a Feff path under the correlated-Debye model at temperature `t`
/// (K) and Debye temperature `theta` (K). Port of
/// `sigma2_models.sigma2_correldebye_py` (Feff6 `sigms.f`). `rnorman` is the
/// path's average Norman radius.
pub fn sigma2_debye(t: f64, theta: f64, rnorman: f64, geom: &[GeomAtom]) -> f64 {
    let thetad = theta.max(1.0e-5);
    let tempk = t.max(1.0e-5);
    let n = geom.len();
    if n == 0 {
        return 0.0;
    }
    let x: Vec<f64> = geom.iter().map(|a| a.x).collect();
    let y: Vec<f64> = geom.iter().map(|a| a.y).collect();
    let z: Vec<f64> = geom.iter().map(|a| a.z).collect();
    let m: Vec<f64> = geom.iter().map(|a| a.mass).collect();

    let mut sig2 = 0.0;
    for i0 in 0..n {
        let i1 = (i0 + 1) % n;
        for j0 in i0..n {
            let j1 = (j0 + 1) % n;
            let ri0j0 = dist(x[i0], y[i0], z[i0], x[j0], y[j0], z[j0]);
            let ri1j1 = dist(x[i1], y[i1], z[i1], x[j1], y[j1], z[j1]);
            let ri0j1 = dist(x[i0], y[i0], z[i0], x[j1], y[j1], z[j1]);
            let ri1j0 = dist(x[i1], y[i1], z[i1], x[j0], y[j0], z[j0]);
            let ri0i1 = dist(x[i0], y[i0], z[i0], x[i1], y[i1], z[i1]);
            let rj0j1 = dist(x[j0], y[j0], z[j0], x[j1], y[j1], z[j1]);
            let ridotj = (x[i0] - x[i1]) * (x[j0] - x[j1])
                + (y[i0] - y[i1]) * (y[j0] - y[j1])
                + (z[i0] - z[i1]) * (z[j0] - z[j1]);

            let ci0j0 = corrfn(ri0j0, thetad, tempk, m[i0], m[j0], rnorman);
            let ci1j1 = corrfn(ri1j1, thetad, tempk, m[i1], m[j1], rnorman);
            let ci0j1 = corrfn(ri0j1, thetad, tempk, m[i0], m[j1], rnorman);
            let ci1j0 = corrfn(ri1j0, thetad, tempk, m[i1], m[j0], rnorman);

            let mut sig2ij = ridotj * (ci0j0 + ci1j1 - ci0j1 - ci1j0) / (ri0i1 * rj0j1);
            // do not double-count the i == j term
            if j0 == i0 {
                sig2ij /= 2.0;
            }
            sig2 += sig2ij;
        }
    }
    sig2 / 2.0
}

/// Euclidean distance between two Cartesian points (Feff6 `dist`).
fn dist(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> f64 {
    ((x0 - x1).powi(2) + (y0 - y1).powi(2) + (z0 - z1).powi(2)).sqrt()
}

/// Debye correlation function `c(ri, rj) = <xi xj>` (Feff6 `sigms.f` `corrfn`).
/// `conh`/`conr` are kept at the Feff6 constants for backward compatibility.
fn corrfn(rij: f64, theta: f64, tk: f64, am1: f64, am2: f64, rs: f64) -> f64 {
    const CONH: f64 = 72.7630804732553;
    const CONR: f64 = 4.5693349700844;
    let rx = CONR * rij / rs;
    let tx = theta / tk;
    let rmass = theta * (am1 * am2).sqrt();
    CONH * debint(rx, tx) / rmass
}

/// `debfun = (sin(w·rx)/rx) · coth(w·tx/2)` (Feff6 `debfun`).
fn debfun(w: f64, rx: f64, tx: f64) -> f64 {
    const WMIN: f64 = 1.0e-20;
    const ARGMAX: f64 = 50.0;
    let mut result = 2.0 / tx; // allow t == 0 without bombing
    if w > WMIN {
        result = if rx > 0.0 { (w * rx).sin() / rx } else { w };
        let emwt = (-(w * tx).min(ARGMAX)).exp();
        result *= (1.0 + emwt) / (1.0 - emwt);
    }
    result
}

/// `∫₀¹ debfun dz` by trapezoidal rule with binary refinement / Romberg first
/// term (Feff6 `debint`; J. Rehr, 10 Feb 92).
fn debint(rx: f64, tx: f64) -> f64 {
    const MAXITER: usize = 12;
    const TOL: f64 = 1.0e-9;
    let mut itn = 1usize;
    let mut step = 1.0;
    let mut result = 0.0;
    let mut bo = (debfun(0.0, rx, tx) + debfun(1.0, rx, tx)) / 2.0;
    let mut bn = bo;
    for _ in 0..MAXITER {
        step /= 2.0;
        let mut sum = 0.0;
        for i in 0..itn {
            sum += debfun(step * (2 * i + 1) as f64, rx, tx);
        }
        itn *= 2;
        let bnp1 = step * sum + bn / 2.0;
        // cancel leading error term: b = (4·b_{n+1} − b_n)/3
        result = (4.0 * bnp1 - bn) / 3.0;
        if ((result - bo) / result).abs() < TOL {
            break;
        }
        bn = bnp1;
        bo = result;
    }
    result
}
