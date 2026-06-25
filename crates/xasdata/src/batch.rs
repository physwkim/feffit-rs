//! Headless batch drivers shared by the GUI's *Multiple_data* menu and the
//! *Plot Data* window.
//!
//! These functions own no UI: they take plain groups / columns / curves and the
//! same engine parameter structs a single-spectrum reduction uses, so the batch
//! UI is a thin wrapper and the logic is unit-testable without a window.
//!
//! - [`make_xmu_batch`] — build `mu(E)` for many files under one column mapping.
//! - [`reduce_all`] — normalize → AUTOBK → FT a whole set with shared settings.
//! - [`average_curves`] — the mean of several curves on a common grid.
//! - [`peak_in_range`] — the maximum of a curve within an x window.

use rayon::prelude::*;

use crate::group::XasGroup;
use crate::reader::ColumnFile;
use crate::reduce::{FtParams, autobk_group, normalize, xftf_group};
use crate::xmu::{MuSpec, XmuError, build_mu};
use xasproc::{AutobkParams, PreEdgeParams};

/// Build one [`XasGroup`] per file using a single shared [`MuSpec`] (one column
/// mapping applied across files of the same layout — XAFSView's batch make-xmu).
///
/// Returns one result per input file in order; a file whose columns don't match
/// the spec yields `Err(XmuError)` without aborting the rest. Each built group is
/// labelled from the file stem and remembers its source path.
pub fn make_xmu_batch(files: &[ColumnFile], spec: &MuSpec) -> Vec<Result<XasGroup, XmuError>> {
    // Each file is independent; `par_iter().collect()` preserves input order, so
    // the caller's `files[i]` ↔ `result[i]` pairing (used for output numbering)
    // still holds.
    files
        .par_iter()
        .map(|cf| {
            let (energy, mu) = build_mu(cf, spec)?;
            let label = cf
                .path
                .as_ref()
                .and_then(|p| p.file_stem())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "group".to_owned());
            let mut g = XasGroup::from_mu(label, energy, mu);
            g.filename = cf.path.clone();
            Ok(g)
        })
        .collect()
}

/// Run normalize → AUTOBK → forward-FT on every group that has a raw spectrum,
/// with one shared set of parameters. Groups with an empty `mu` are skipped.
/// Returns the number of groups reduced.
pub fn reduce_all(
    groups: &mut [XasGroup],
    pre: &PreEdgeParams,
    bk: &AutobkParams,
    ft: &FtParams,
    err_sigma: f64,
) -> usize {
    // Every group reduces independently (each call mutates only its own group and
    // builds its own FFT planner — no shared state), so fan the slow
    // normalize → AUTOBK → FT chain across cores. `count()` drives the `map`
    // side effects and returns the number of groups actually reduced.
    groups
        .par_iter_mut()
        .filter(|g| !g.mu.is_empty())
        .map(|g| {
            normalize(g, pre);
            autobk_group(g, bk, err_sigma);
            xftf_group(g, ft);
        })
        .count()
}

/// The point-wise mean of several `(x, y)` curves on a common grid.
///
/// The shared grid is the first curve's x samples restricted to the overlap
/// `[max of all x-mins, min of all x-maxes]`; every curve is linearly
/// interpolated onto it (XAFSView / larch `merge_groups` semantics) and averaged.
/// Returns `None` if there are no curves, any curve is mis-shaped or shorter than
/// two points, the x ranges don't overlap, or the overlap grid is degenerate.
pub fn average_curves(curves: &[(&[f64], &[f64])]) -> Option<(Vec<f64>, Vec<f64>)> {
    if curves.is_empty() {
        return None;
    }
    for (x, y) in curves {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }
    }
    let lo = curves
        .iter()
        .map(|(x, _)| x[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let hi = curves
        .iter()
        .map(|(x, _)| *x.last().unwrap())
        .fold(f64::INFINITY, f64::min);
    // Require a strictly positive overlap; `partial_cmp` also rejects NaN bounds
    // (an `hi <= lo` test would let a NaN slip through).
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let grid: Vec<f64> = curves[0]
        .0
        .iter()
        .copied()
        .filter(|&v| v >= lo && v <= hi)
        .collect();
    if grid.len() < 2 {
        return None;
    }
    let mut sum = vec![0.0; grid.len()];
    for (x, y) in curves {
        let yi = xasproc::mathutils::interp_linear(&grid, x, y);
        for (s, v) in sum.iter_mut().zip(yi) {
            *s += v;
        }
    }
    let n = curves.len() as f64;
    for s in &mut sum {
        *s /= n;
    }
    Some((grid, sum))
}

/// Resample several `(x, y)` curves onto one uniform grid of `npts` points over
/// the requested `[xmin, xmax]`, clamped to the curves' common overlap.
///
/// Each curve is linearly interpolated onto the shared grid (endpoints held
/// outside its range; see [`interp_linear`](xasproc::mathutils::interp_linear)),
/// giving a matrix whose rows are directly comparable — the common-grid step LCF
/// and PCA need before fitting. Returns `(grid, rows)` with `rows[i]` curve `i`
/// on `grid`, or `None` if there are no curves, `npts < 2`, the overlap is empty
/// (also rejecting NaN bounds), or any curve is mis-shaped / shorter than two.
pub fn resample_matrix(
    curves: &[(&[f64], &[f64])],
    xmin: f64,
    xmax: f64,
    npts: usize,
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    if curves.is_empty() || npts < 2 {
        return None;
    }
    for (x, y) in curves {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }
    }
    // Largest lower bound and smallest upper bound, also bounded by the request.
    let lo = curves.iter().map(|(x, _)| x[0]).fold(xmin, f64::max);
    let hi = curves
        .iter()
        .map(|(x, _)| *x.last().unwrap())
        .fold(xmax, f64::min);
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let step = (hi - lo) / (npts - 1) as f64;
    let grid: Vec<f64> = (0..npts).map(|i| lo + step * i as f64).collect();
    let rows = curves
        .iter()
        .map(|(x, y)| xasproc::mathutils::interp_linear(&grid, x, y))
        .collect();
    Some((grid, rows))
}

/// The maximum `(x, y)` of a curve within the inclusive window `[lo, hi]`.
///
/// `lo`/`hi` may be given in either order. On ties the first (lowest-index)
/// maximum wins. Returns `None` when no sample falls inside the window.
pub fn peak_in_range(x: &[f64], y: &[f64], lo: f64, hi: f64) -> Option<(f64, f64)> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let mut best: Option<(f64, f64)> = None;
    for (&xi, &yi) in x.iter().zip(y) {
        if xi < lo || xi > hi {
            continue;
        }
        match best {
            Some((_, by)) if yi <= by => {}
            _ => best = Some((xi, yi)),
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::ColumnFile;
    use crate::xmu::MuSpec;

    /// Two `mu(E)` columns under one Raw spec → two labelled groups; a file
    /// missing the column errors without taking down the batch.
    #[test]
    fn make_xmu_batch_applies_one_spec() {
        let a = ColumnFile::from_text("# e mu\n1 0.5\n2 1.5\n").unwrap();
        let b = ColumnFile::from_text("# e mu\n1 0.7\n2 1.1\n").unwrap();
        let one_col = ColumnFile::from_text("# e\n1\n2\n").unwrap();
        let spec = MuSpec::Raw { energy: 0, mu: 1 };
        let out = make_xmu_batch(&[a, b, one_col], &spec);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].as_ref().unwrap().mu, vec![0.5, 1.5]);
        assert_eq!(out[1].as_ref().unwrap().mu, vec![0.7, 1.1]);
        assert_eq!(out[2].as_ref().unwrap_err(), &XmuError::BadColumn(1));
    }

    #[test]
    fn reduce_all_populates_k_chi_r() {
        let cf = ColumnFile::from_text(include_str!("../tests/data/cu.xmu")).unwrap();
        let (energy, mu) = build_mu(&cf, &MuSpec::Raw { energy: 0, mu: 1 }).unwrap();
        let mut groups = vec![
            XasGroup::from_mu("a", energy.clone(), mu.clone()),
            XasGroup::from_mu("b", energy, mu),
            XasGroup::default(), // empty: must be skipped
        ];
        let n = reduce_all(
            &mut groups,
            &PreEdgeParams::default(),
            &AutobkParams::default(),
            &FtParams::default(),
            0.0,
        );
        assert_eq!(n, 2);
        for g in &groups[..2] {
            assert!(g.norm.as_ref().is_some_and(|v| !v.is_empty()));
            assert!(g.k.as_ref().is_some_and(|v| !v.is_empty()));
            assert!(g.chi.as_ref().is_some_and(|v| !v.is_empty()));
            assert!(g.r.as_ref().is_some_and(|v| !v.is_empty()));
        }
        assert!(groups[2].k.is_none());
    }

    #[test]
    fn average_of_identical_curves_is_the_curve() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let (gx, gy) = average_curves(&[(&x, &y), (&x, &y)]).unwrap();
        assert_eq!(gx, x.to_vec());
        for (a, b) in gy.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn average_interpolates_onto_overlap() {
        // Curve B is offset in x and on a coarser grid; the mean is taken on the
        // overlap [1, 3] using curve A's samples there.
        let xa = [0.0, 1.0, 2.0, 3.0];
        let ya = [0.0, 10.0, 20.0, 30.0];
        let xb = [1.0, 3.0];
        let yb = [12.0, 32.0]; // = 10*x + 2 at x=1,3 → interp gives 22 at x=2
        let (gx, gy) = average_curves(&[(&xa, &ya), (&xb, &yb)]).unwrap();
        assert_eq!(gx, vec![1.0, 2.0, 3.0]);
        // mean at x=1: (10+12)/2=11; x=2: (20+22)/2=21; x=3: (30+32)/2=31
        let want = [11.0, 21.0, 31.0];
        for (a, b) in gy.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-12, "got {gy:?}");
        }
    }

    #[test]
    fn average_rejects_disjoint_and_malformed() {
        assert!(average_curves(&[]).is_none());
        let x = [0.0, 1.0];
        let short = [0.0];
        assert!(average_curves(&[(&x[..], &short[..])]).is_none()); // mis-shaped
        let xa = [0.0, 1.0];
        let xb = [5.0, 6.0];
        let y = [1.0, 2.0];
        assert!(average_curves(&[(&xa, &y), (&xb, &y)]).is_none()); // disjoint
    }

    #[test]
    fn resample_matrix_clamps_to_overlap_and_interpolates() {
        let xa = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ya = [0.0, 10.0, 20.0, 30.0, 40.0]; // y = 10x
        let xb = [1.0, 2.0, 3.0];
        let yb = [100.0, 200.0, 300.0]; // y = 100x
        // Request [0,4] but the overlap clamps the grid to [1,3].
        let (grid, rows) = resample_matrix(&[(&xa, &ya), (&xb, &yb)], 0.0, 4.0, 3).unwrap();
        assert_eq!(grid, vec![1.0, 2.0, 3.0]);
        assert_eq!(rows.len(), 2);
        assert!((rows[0][1] - 20.0).abs() < 1e-12);
        assert!((rows[1][1] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn resample_matrix_rejects_bad() {
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        assert!(resample_matrix(&[(&x, &y)], 0.0, 1.0, 1).is_none()); // npts < 2
        let xb = [5.0, 6.0];
        assert!(resample_matrix(&[(&x, &y), (&xb, &y)], 0.0, 6.0, 4).is_none()); // disjoint
    }

    #[test]
    fn peak_in_range_finds_max_first_on_ties() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 5.0, 9.0, 9.0, 1.0];
        assert_eq!(peak_in_range(&x, &y, 0.0, 4.0), Some((2.0, 9.0)));
        // window excludes the global max → local max inside
        assert_eq!(peak_in_range(&x, &y, 3.5, 4.5), Some((4.0, 1.0)));
        // reversed bounds behave the same
        assert_eq!(peak_in_range(&x, &y, 4.0, 0.0), Some((2.0, 9.0)));
        // empty window
        assert_eq!(peak_in_range(&x, &y, 10.0, 11.0), None);
    }
}
