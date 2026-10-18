//! Fits whose data end below the transform's k-range (an AUTOBK kmax under
//! the Feffit kmax). The residual must stop at the data's last k, as larch's
//! `[iqmin:iqmax]` slice of `_chi` does, instead of indexing past it; a range
//! holding no data at all, or fewer points than variables, is an error rather
//! than an optimizer panic.

use feffit::feffdat::{FeffPath, Interp, KGrid, PathParams, ff2chi};
use feffit::params::Parameters;
use feffit::transform::FitSpace;
use feffit::xafsft::Window;
use feffit::{DataSet, FitDataSet, PathSpec, Spec, Transform, feffit};

const PATH: &str = "tests/data/feff0001.dat";

/// Synthetic first-shell χ(k) from `feff0001.dat`, cut at `kmax_data`.
fn short_data(kmax_data: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let probe = FeffPath::from_path(PATH).unwrap();
    let degen = probe.feffdat.degen;
    let mut truth = probe.with_params(PathParams {
        s02: 0.9,
        sigma2: 0.005,
        ..PathParams::defaults(degen)
    });
    let (k, chi) = ff2chi(
        std::slice::from_mut(&mut truth),
        &KGrid::default_step(),
        Interp::Cubic,
    );
    let n = k.iter().position(|&kk| kk > kmax_data).unwrap();
    (k[..n].to_vec(), chi[..n].to_vec(), degen)
}

fn transform(kmin: f64, kmax: f64, space: FitSpace) -> Transform {
    Transform::new(
        kmin,
        kmax,
        vec![2],
        1.0,
        None,
        Window::Hanning,
        2048,
        0.05,
        1.0,
        4.0,
        0.0,
        None,
        Window::Hanning,
        0.0,
        space,
    )
}

fn k_transform(kmin: f64, kmax: f64) -> Transform {
    transform(kmin, kmax, FitSpace::K)
}

/// One first-shell path with `amp`, `sig2` (and `de0` when asked) varied.
fn fit_dataset(
    data_k: Vec<f64>,
    data_chi: Vec<f64>,
    degen: f64,
    trans: Transform,
    with_e0: bool,
) -> FitDataSet {
    let path = FeffPath::from_path(PATH).unwrap();
    let mut spec = PathSpec::defaults(degen);
    spec.s02 = Spec::Expr("amp".into());
    spec.sigma2 = Spec::Expr("sig2".into());
    if with_e0 {
        spec.e0 = Spec::Expr("de0".into());
    }
    FitDataSet {
        dataset: DataSet::new(data_k, data_chi, vec![path], trans),
        specs: vec![spec],
        epsilon_k: None,
    }
}

#[test]
fn k_space_residual_stops_at_the_data_kmax() {
    let (data_k, data_chi, _) = short_data(10.0);
    let path = FeffPath::from_path(PATH).unwrap();
    let mut ds = DataSet::new(data_k, data_chi, vec![path], k_transform(2.0, 17.0));
    ds.prepare_fit(None);

    // The model grid ends at the data: int(1.01 + 10/0.05) = 201 points; the
    // fit window starts at int(0.01 + 2/0.05) = 40.
    assert_eq!(ds.model_k().len(), 201);
    let r = ds.residual(true);
    assert_eq!(r.len(), 201 - 40, "one k-weight, data-limited window");
    assert!(r.iter().all(|v| v.is_finite()));
}

#[test]
fn k_space_fit_runs_with_data_below_kmax() {
    let (data_k, data_chi, degen) = short_data(10.0);
    let mut fds = vec![fit_dataset(
        data_k,
        data_chi,
        degen,
        k_transform(2.0, 17.0),
        false,
    )];
    let mut p = Parameters::new();
    p.add_var("amp", 1.0);
    p.add_var("sig2", 0.003);

    let res = feffit(&mut p, &mut fds).expect("feffit");
    let amp = res.best.iter().find(|b| b.name == "amp").unwrap().value;
    assert!((amp - 0.9).abs() < 0.02, "amp {amp} vs 0.9");
}

/// Data to k = 3 with the fit starting at k = 4: no data point lies in the
/// fit range, in any fit space. Before the check this was an optimizer panic
/// (k) or a result with a NaN χ² (R, q).
#[test]
fn fit_range_without_data_is_an_error_in_every_space() {
    for space in [FitSpace::K, FitSpace::R, FitSpace::Q] {
        let (data_k, data_chi, degen) = short_data(3.0);
        let mut fds = vec![fit_dataset(
            data_k,
            data_chi,
            degen,
            transform(4.0, 17.0, space),
            false,
        )];
        let mut p = Parameters::new();
        p.add_var("amp", 1.0);
        p.add_var("sig2", 0.003);

        let Err(err) = feffit(&mut p, &mut fds) else {
            panic!("{space:?}: fit ran");
        };
        let msg = err.to_string();
        assert!(msg.contains("no data"), "{space:?}: {msg}");
    }
}

/// Two data points in the fit range (k = 4.00, 4.05) for three variables:
/// MINPACK refuses the problem, which must surface as an error, not as a
/// result pinned at the start values.
#[test]
fn fewer_points_than_variables_is_an_error() {
    let (data_k, data_chi, degen) = short_data(4.06);
    let mut fds = vec![fit_dataset(
        data_k,
        data_chi,
        degen,
        k_transform(4.0, 17.0),
        true,
    )];
    let mut p = Parameters::new();
    p.add_var("amp", 1.0);
    p.add_var("sig2", 0.003);
    p.add_var("de0", 0.0);

    let Err(err) = feffit(&mut p, &mut fds) else {
        panic!("fit ran");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("2 residual points") && msg.contains("3 variables"),
        "{msg}"
    );
}
