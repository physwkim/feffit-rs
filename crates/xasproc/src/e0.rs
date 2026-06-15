//! Edge-energy (`E0`) finding — port of `larch.xafs.pre_edge.find_e0` and its
//! internal `_finde0`. `E0` is the point of maximum derivative of `mu(E)`, with
//! larch's guards against pre-edge glitches: a coarse pass, then a smoothed
//! refinement over a window around the coarse estimate.

use crate::mathutils::{
    dmude, find_energy_step as fes, remove_dups, remove_nans2, smooth, SmoothForm,
};

/// Smallest tolerated energy step, in eV (`larch.xafs.xafsutils.TINY_ENERGY`).
const TINY_ENERGY: f64 = 0.00050;

/// Re-export of [`crate::mathutils::find_energy_step`] with larch's defaults
/// (`frac_ignore=0.01`, `nave=10`).
pub fn find_energy_step(energy: &[f64]) -> f64 {
    fes(energy, 0.01, 10)
}

/// Keep only indices that are "in order" the way larch does
/// (`np.where(np.diff(np.argsort(en))==1)[0]`); for a strictly increasing array
/// this returns `0..n-1` (dropping the final point).
fn ordered_indices(en: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..en.len()).collect();
    idx.sort_by(|&i, &j| {
        en[i]
            .partial_cmp(&en[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (0..idx.len().saturating_sub(1))
        .filter(|&i| idx[i + 1] as i64 - idx[i] as i64 == 1)
        .collect()
}

/// Internal `_finde0`: returns `(e0_value, e0_index, estep)` on the array after
/// dedup/ordering. `estep = None` triggers [`find_energy_step`].
fn finde0(
    energy: &[f64],
    mu_input: &[f64],
    estep: Option<f64>,
    use_smooth: bool,
) -> (f64, usize, f64) {
    let en_dd = remove_dups(energy, TINY_ENERGY);
    let ordered = ordered_indices(&en_dd);
    let en: Vec<f64> = ordered.iter().map(|&i| en_dd[i]).collect();
    let mu: Vec<f64> = ordered.iter().map(|&i| mu_input[i]).collect();
    let n = en.len();
    let estep = estep.unwrap_or_else(|| find_energy_step(&en));

    let nmin = (n as f64 * 0.02) as usize;
    let nmin = nmin.max(3);

    let mut dmu = if use_smooth {
        let raw = dmude(&mu, &en);
        smooth(&en, &raw, estep, estep, 5, SmoothForm::Lorentzian)
    } else {
        dmude(&mu, &en)
    };
    for d in dmu.iter_mut() {
        if !d.is_finite() {
            *d = -1.0;
        }
    }
    // normalize over the interior [nmin : n-nmin]
    if n <= 2 * nmin {
        return (en[0], 0, estep);
    }
    let interior = &dmu[nmin..n - nmin];
    let dm_min = interior.iter().cloned().fold(f64::INFINITY, f64::min);
    let dm_max = interior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dm_ptp = (dm_max - dm_min).max(1.0e-10);
    for d in dmu.iter_mut() {
        *d = (*d - dm_min) / dm_ptp;
    }

    let mut dhigh = if n > 20 { 0.60 } else { 0.30 };
    let mut high: Vec<usize> = (0..n).filter(|&i| dmu[i] > dhigh).collect();
    if high.len() < 3 {
        for _ in 0..2 {
            if high.len() > 3 {
                break;
            }
            dhigh *= 0.5;
            high = (0..n).filter(|&i| dmu[i] > dhigh).collect();
        }
    }
    if high.len() < 3 {
        high = (0..n).filter(|&i| dmu[i].is_finite()).collect();
    }
    let mut is_high = vec![false; n];
    for &i in &high {
        is_high[i] = true;
    }

    let mut imax = 0usize;
    let mut dmax = 0.0f64;
    for &i in &high {
        if i < nmin || i > n - nmin {
            continue;
        }
        if dmu[i] > dmax && i + 1 < n && is_high[i + 1] && i >= 1 && is_high[i - 1] {
            imax = i;
            dmax = dmu[i];
        }
    }
    (en[imax], imax, estep)
}

/// `larch.xafs.pre_edge.find_e0`: the edge energy `E0` of `mu(E)`.
///
/// A coarse unsmoothed pass locates the maximum-derivative point; a second pass
/// over a ±75-point window (or the tail, if the coarse estimate falls in the
/// first 5% of points) with derivative smoothing refines it.
pub fn find_e0(energy: &[f64], mu: &[f64]) -> f64 {
    let (energy, mu) = remove_nans2(energy, mu);
    let n = energy.len();

    let (mut e1, ie0, estep1) = finde0(&energy, &mu, None, false);
    let mut istart = (ie0 as i64 - 75).max(3) as usize;
    let mut istop = (ie0 + 75).min(n - 3);
    if (ie0 as f64) < 0.05 * n as f64 {
        e1 = energy.iter().sum::<f64>() / n as f64;
        istart = (ie0 as i64 - 20).max(3) as usize;
        istop = n - 3;
    }

    let estep = 0.5 * (estep1.clamp(0.01, 1.0) + (e1 / 25000.0).clamp(0.01, 1.0));
    let (mut e0, ix, _ex) = finde0(
        &energy[istart..istop],
        &mu[istart..istop],
        Some(estep),
        true,
    );
    if ix < 1 {
        e0 = energy[istart + 2];
    }
    e0
}
