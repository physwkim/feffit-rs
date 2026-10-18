//! A k-space fit whose data end below the transform's `kmax` (an AUTOBK kmax
//! under the Feffit kmax). The residual must stop at the data's last k, as
//! larch's `[iqmin:iqmax]` slice of `_chi` does, instead of indexing past it.

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

fn k_transform(kmin: f64, kmax: f64) -> Transform {
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
        FitSpace::K,
    )
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
    let path = FeffPath::from_path(PATH).unwrap();
    let mut spec = PathSpec::defaults(degen);
    spec.s02 = Spec::Expr("amp".into());
    spec.sigma2 = Spec::Expr("sig2".into());
    let dataset = DataSet::new(data_k, data_chi, vec![path], k_transform(2.0, 17.0));
    let mut fds = vec![FitDataSet {
        dataset,
        specs: vec![spec],
        epsilon_k: None,
    }];
    let mut p = Parameters::new();
    p.add_var("amp", 1.0);
    p.add_var("sig2", 0.003);

    let res = feffit(&mut p, &mut fds).expect("feffit");
    let amp = res.best.iter().find(|b| b.name == "amp").unwrap().value;
    assert!((amp - 0.9).abs() < 0.02, "amp {amp} vs 0.9");
}
