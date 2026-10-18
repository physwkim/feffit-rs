//! `pre_edge` at the lower boundary of the post-edge range: larch widens a range
//! of fewer than two points down to the two points below its end; where there
//! are none (range start below index 2) its slice is empty and `polyfit`
//! raises, and the port returns `PreEdgeError::EmptyNormRange` instead of
//! underflowing the index.

use feffit::xasproc::{PreEdgeError, PreEdgeParams, pre_edge};

/// 60 points, 2 eV apart, with a step at 7020 eV.
fn step_spectrum() -> (Vec<f64>, Vec<f64>) {
    let energy: Vec<f64> = (0..60).map(|i| 7000.0 + 2.0 * i as f64).collect();
    let mu = energy
        .iter()
        .map(|&e| {
            if e < 7020.0 {
                0.1 + 1e-5 * (e - 7000.0)
            } else {
                1.0 + 1e-4 * (7100.0 - e)
            }
        })
        .collect();
    (energy, mu)
}

fn params(e0: f64, norm1: f64, norm2: f64) -> PreEdgeParams {
    PreEdgeParams {
        e0: Some(e0),
        norm1: Some(norm1),
        norm2: Some(norm2),
        ..Default::default()
    }
}

#[test]
fn norm_range_starting_below_the_third_point_is_an_error() {
    let (energy, mu) = step_spectrum();
    // p1 = 0 (range start below the first point), p2 = 1 (e0 itself).
    let err = pre_edge(&energy, &mu, &params(energy[1], -5.0, 0.0)).unwrap_err();
    assert!(
        matches!(err, PreEdgeError::EmptyNormRange { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("norm1 = -5.0 eV"), "{err}");
    // p1 = 1, p2 = 2: one point below the range end, two needed.
    let err = pre_edge(&energy, &mu, &params(energy[2], -2.0, 0.0)).unwrap_err();
    assert!(
        matches!(err, PreEdgeError::EmptyNormRange { .. }),
        "{err:?}"
    );
}

#[test]
fn norm_range_starting_at_the_third_point_widens_like_larch() {
    let (energy, mu) = step_spectrum();
    // p1 = 2, p2 = 3: larch fits a constant to [p1 - 2, p2) = the first three
    // points.
    let pe = pre_edge(&energy, &mu, &params(energy[3], -2.0, 0.0)).unwrap();
    assert_eq!(pe.nnorm, 0);
    assert_eq!(pe.norm1, -2.0);
    assert!(pe.norm.iter().all(|v| v.is_finite()));
}
