//! Data-cleanup primitives — deglitch (point/range removal) and trim — matching
//! larch's interactive deglitch mask logic (`larch/wxxas/datatasks.py`).
//!
//! Each builder returns a *keep-mask* (`true` = keep) parallel to the input
//! `energy` grid; [`select`] applies a mask to any co-indexed array. The mask
//! shape mirrors larch exactly: a deglitch marks points to drop by index, and
//! the caller filters every co-indexed column (`mu` and the raw monitors) by the
//! same mask. Smoothing lives in [`crate::mathutils::smooth`]; this module is
//! only the point-dropping side of cleanup.

use crate::mathutils::index_nearest;

/// Which side of a reference energy a range deglitch removes, matching larch's
/// "above" / "below" / "between" choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeSide {
    /// Remove the point nearest `e1` and everything above it (`[i1, end]`).
    Above,
    /// Remove everything below the point nearest `e1` (`[0, i1)`; `i1` kept).
    Below,
    /// Remove the half-open span between the points nearest `e1` and `e2`.
    Between,
}

/// Keep-mask that drops the single point whose energy is nearest `e`
/// (larch `on_remove(opt='x')`: `mask[index_nearest(x, e)] = False`).
pub fn deglitch_point_mask(energy: &[f64], e: f64) -> Vec<bool> {
    let mut mask = vec![true; energy.len()];
    if !energy.is_empty() {
        mask[index_nearest(energy, e)] = false;
    }
    mask
}

/// Keep-mask for a range removal, matching larch `on_remove(opt='range')`:
/// `i1 = index_nearest(energy, e1)` and, for [`RangeSide::Between`],
/// `i2 = index_nearest(energy, e2)`.
///
/// - [`RangeSide::Above`] drops `[i1, end]`   (Python `mask[i1:None] = False`),
/// - [`RangeSide::Below`] drops `[0, i1)`     (Python `mask[None:i1] = False`, so `i1` is kept),
/// - [`RangeSide::Between`] drops `[lo, hi)` with `(lo, hi) = sort(i1, i2)`
///   (Python `mask[i1:i2] = False`).
///
/// `e2` is consulted only for [`RangeSide::Between`].
pub fn deglitch_range_mask(energy: &[f64], side: RangeSide, e1: f64, e2: f64) -> Vec<bool> {
    let n = energy.len();
    let mut mask = vec![true; n];
    if n == 0 {
        return mask;
    }
    let i1 = index_nearest(energy, e1);
    match side {
        RangeSide::Above => mask[i1..].fill(false),
        RangeSide::Below => mask[..i1].fill(false),
        RangeSide::Between => {
            let i2 = index_nearest(energy, e2);
            let (lo, hi) = if i1 <= i2 { (i1, i2) } else { (i2, i1) };
            mask[lo..hi].fill(false);
        }
    }
    mask
}

/// Keep-mask trimming to the inclusive energy window `[lo, hi]` (XAFSView "Edit
/// XMU" trim): points with `lo <= energy <= hi` are kept. Bounds are sorted, so
/// the caller may pass them in either order.
pub fn trim_mask(energy: &[f64], lo: f64, hi: f64) -> Vec<bool> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    energy.iter().map(|&e| e >= lo && e <= hi).collect()
}

/// Number of points a keep-mask discards (its `false` entries).
pub fn removed_count(mask: &[bool]) -> usize {
    mask.iter().filter(|&&keep| !keep).count()
}

/// Select the kept (`mask == true`) elements of `arr`, which must be co-indexed
/// with `mask`.
pub fn select(mask: &[bool], arr: &[f64]) -> Vec<f64> {
    debug_assert_eq!(mask.len(), arr.len(), "mask and array must be co-indexed");
    arr.iter()
        .zip(mask)
        .filter_map(|(&v, &keep)| keep.then_some(v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 0..=9 eV grid (index == energy) so index_nearest is unambiguous.
    fn grid() -> Vec<f64> {
        (0..10).map(|i| i as f64).collect()
    }

    #[test]
    fn point_drops_only_nearest() {
        let e = grid();
        let mask = deglitch_point_mask(&e, 4.2);
        assert_eq!(removed_count(&mask), 1);
        assert!(!mask[4], "point nearest 4.2 (index 4) must be dropped");
        for (i, &keep) in mask.iter().enumerate() {
            if i != 4 {
                assert!(keep, "index {i} must be kept");
            }
        }
    }

    #[test]
    fn above_drops_i1_inclusive_to_end() {
        let e = grid();
        let mask = deglitch_range_mask(&e, RangeSide::Above, 6.0, 0.0);
        // Python mask[6:None] = False -> indices 6..=9 dropped, 0..=5 kept.
        for (i, &keep) in mask.iter().enumerate() {
            assert_eq!(keep, i < 6, "above@6 index {i}");
        }
    }

    #[test]
    fn below_drops_up_to_i1_exclusive() {
        let e = grid();
        let mask = deglitch_range_mask(&e, RangeSide::Below, 3.0, 0.0);
        // Python mask[None:3] = False -> indices 0..=2 dropped, 3 kept.
        for (i, &keep) in mask.iter().enumerate() {
            assert_eq!(keep, i >= 3, "below@3 index {i}");
        }
    }

    #[test]
    fn between_drops_half_open_and_is_order_independent() {
        let e = grid();
        let fwd = deglitch_range_mask(&e, RangeSide::Between, 2.0, 5.0);
        let rev = deglitch_range_mask(&e, RangeSide::Between, 5.0, 2.0);
        assert_eq!(fwd, rev, "between must sort its bounds");
        // Python mask[2:5] = False -> indices 2,3,4 dropped; 5 kept.
        for (i, &keep) in fwd.iter().enumerate() {
            assert_eq!(keep, !(2..5).contains(&i), "between@[2,5) index {i}");
        }
    }

    #[test]
    fn trim_keeps_inclusive_window() {
        let e = grid();
        let mask = trim_mask(&e, 3.0, 7.0);
        for (i, &keep) in mask.iter().enumerate() {
            assert_eq!(keep, (3..=7).contains(&i), "trim[3,7] index {i}");
        }
        // Reversed bounds give the same window.
        assert_eq!(mask, trim_mask(&e, 7.0, 3.0));
    }

    #[test]
    fn select_keeps_co_indexed_values() {
        let e = grid();
        let mask = deglitch_range_mask(&e, RangeSide::Between, 2.0, 5.0);
        let kept = select(&mask, &e);
        assert_eq!(kept, vec![0.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(kept.len(), e.len() - removed_count(&mask));
    }

    #[test]
    fn empty_grid_is_noop() {
        assert!(deglitch_point_mask(&[], 1.0).is_empty());
        assert!(deglitch_range_mask(&[], RangeSide::Above, 1.0, 2.0).is_empty());
        assert!(trim_mask(&[], 1.0, 2.0).is_empty());
    }
}
