//! Spectrum-editing orchestration: apply a deglitch/trim keep-mask or a
//! smoothing pass to a whole [`XasGroup`], filtering every co-indexed array and
//! clearing the now-stale derived reduction stages.
//!
//! The mask math lives in [`crate::xasproc::clean`] (parity with larch's interactive
//! deglitch); this layer maps it onto the group's parallel columns — `energy`,
//! `mu`, and whichever of `i0`/`it`/`iflu`/`iref` were kept — so a single edit
//! keeps every raw array the same length, then resets derived results via
//! [`XasGroup::clear_derived`].

use crate::xasproc::clean::{
    deglitch_point_mask, deglitch_range_mask, removed_count, select, trim_mask,
};
use crate::xasproc::mathutils::smooth;

use crate::xasdata::group::XasGroup;

pub use crate::xasproc::clean::RangeSide;
pub use crate::xasproc::mathutils::SmoothForm;

/// Filter every co-indexed array of `group` by `mask` (keeping the `true`
/// entries), then clear the derived reduction stages. Returns the number of
/// points removed; a no-op (`mask` keeps everything) leaves `group` untouched.
fn apply_keep_mask(group: &mut XasGroup, mask: &[bool]) -> usize {
    let removed = removed_count(mask);
    if removed == 0 {
        return 0;
    }
    group.energy = select(mask, &group.energy);
    group.mu = select(mask, &group.mu);
    for col in [
        &mut group.i0,
        &mut group.it,
        &mut group.iflu,
        &mut group.iref,
    ] {
        if let Some(v) = col.as_mut() {
            *v = select(mask, v);
        }
    }
    group.clear_derived();
    removed
}

/// Remove the single point whose energy is nearest `e` (larch deglitch "Remove
/// point"). Returns the number of points removed (0 or 1).
pub fn deglitch_point(group: &mut XasGroup, e: f64) -> usize {
    let mask = deglitch_point_mask(&group.energy, e);
    apply_keep_mask(group, &mask)
}

/// Remove a range of points relative to `e1` (and `e2` for
/// [`RangeSide::Between`]), matching larch deglitch "Remove range". Returns the
/// number of points removed.
pub fn deglitch_range(group: &mut XasGroup, side: RangeSide, e1: f64, e2: f64) -> usize {
    let mask = deglitch_range_mask(&group.energy, side, e1, e2);
    apply_keep_mask(group, &mask)
}

/// Trim the spectrum to the inclusive energy window `[lo, hi]` (XAFSView "Edit
/// XMU" trim). Returns the number of points removed.
pub fn trim(group: &mut XasGroup, lo: f64, hi: f64) -> usize {
    let mask = trim_mask(&group.energy, lo, hi);
    apply_keep_mask(group, &mask)
}

/// Smooth `mu(E)` in place with larch's convolution smoother
/// ([`crate::xasproc::mathutils::smooth`]), then clear the derived stages so reduction
/// re-runs on the smoothed spectrum. `xstep` defaults to the minimum successive
/// energy step and `npad` to 5, matching `larch.math.smooth`. Returns `false`
/// (leaving `group` untouched) when the grid has fewer than two points or is not
/// strictly increasing.
pub fn smooth_mu(group: &mut XasGroup, sigma: f64, form: SmoothForm) -> bool {
    if group.energy.len() < 2 {
        return false;
    }
    let min_step = group
        .energy
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    if min_step < 1e-12 {
        return false;
    }
    group.mu = smooth(&group.energy, &group.mu, sigma, min_step, 5, form);
    group.clear_derived();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group() -> XasGroup {
        let energy: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mu: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let mut g = XasGroup::from_mu("g", energy, mu);
        // co-indexed source columns and a stale derived field to verify wiping.
        g.i0 = Some((0..10).map(|i| 100.0 + i as f64).collect());
        g.it = Some((0..10).map(|i| 50.0 + i as f64).collect());
        g.e0 = Some(5.0);
        g.norm = Some(vec![0.0; 10]);
        g
    }

    #[test]
    fn deglitch_point_filters_all_columns_and_clears_derived() {
        let mut g = group();
        let removed = deglitch_point(&mut g, 4.2);
        assert_eq!(removed, 1);
        assert_eq!(g.energy.len(), 9);
        assert_eq!(g.mu.len(), 9);
        assert_eq!(g.i0.as_ref().unwrap().len(), 9);
        assert_eq!(g.it.as_ref().unwrap().len(), 9);
        assert!(!g.energy.contains(&4.0), "dropped energy must be gone");
        assert!(
            !g.i0.as_ref().unwrap().contains(&104.0),
            "the co-indexed i0 sample at index 4 must be dropped too"
        );
        assert_eq!(g.e0, None, "derived e0 must be cleared");
        assert_eq!(g.norm, None, "derived norm must be cleared");
    }

    #[test]
    fn trim_keeps_only_the_window() {
        let mut g = group();
        let removed = trim(&mut g, 3.0, 6.0);
        assert_eq!(removed, 6); // indices 0,1,2,7,8,9 dropped
        assert_eq!(g.energy, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(g.i0.as_ref().unwrap(), &vec![103.0, 104.0, 105.0, 106.0]);
    }

    #[test]
    fn noop_mask_leaves_group_and_derived_untouched() {
        let mut g = group();
        let removed = trim(&mut g, -5.0, 100.0); // window covers everything
        assert_eq!(removed, 0);
        assert_eq!(g.energy.len(), 10);
        assert_eq!(g.e0, Some(5.0), "no-op must not wipe derived results");
        assert_eq!(g.norm, Some(vec![0.0; 10]));
    }

    #[test]
    fn smooth_preserves_length_and_clears_derived() {
        let mut g = group();
        let ran = smooth_mu(&mut g, 0.5, SmoothForm::Gaussian);
        assert!(ran);
        assert_eq!(g.mu.len(), 10, "smooth interpolates back onto energy");
        assert_eq!(g.e0, None);
        assert_eq!(g.norm, None);
    }

    #[test]
    fn smooth_bails_on_too_short_grid() {
        let mut g = XasGroup::from_mu("s", vec![1.0], vec![0.5]);
        assert!(!smooth_mu(&mut g, 1.0, SmoothForm::Lorentzian));
    }
}
