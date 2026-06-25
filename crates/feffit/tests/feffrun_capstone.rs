//! End-to-end capstone: FEFF8L (subprocess) → `feffdat` parse → `feffit` fit.
//!
//! Drives the native FEFF8L path generator (via `feffrun`) on a real Cu
//! `feff.inp`, takes the first-shell `feff0001.dat` it produces, builds a
//! synthetic χ(k) from that path at *known* parameters (s02 = 0.9,
//! σ² = 0.005 Å²), then runs `feffit` to recover them. This is the only test
//! that exercises the whole pipeline — subprocess FEFF, the `feffdat` parser,
//! and the `feffit` minimiser — in a single flow; the individual stages are
//! covered bit-exactly elsewhere.
//!
//! Needs the `feff8l_*` executables (point `FEFF8L_DIR` at them or add them to
//! PATH). When they are absent the test prints a SKIP notice and returns rather
//! than failing, so it is inert on machines without FEFF built.

use std::path::PathBuf;

use feffit::feffdat::{FeffPath, Interp, KGrid, PathParams, ff2chi};
use feffit::params::Parameters;
use feffit::transform::FitSpace;
use feffit::xafsft::Window;
use feffit::{DataSet, FitDataSet, PathSpec, Spec, Transform, feffit};

/// The Copper fcc `feff.inp` shared with the `feffrun` integration test
/// (workspace-relative; this test only runs with the full workspace present).
fn cu_feff_inp() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/feff.inp");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// A FEFF8L runner if the executables are reachable, else `None`. Mirrors the
/// discovery in `feffrun`'s own integration test (FEFF8L_DIR, then PATH).
fn runner() -> Option<feffit::feffrun::Feff8l> {
    if let Some(dir) = std::env::var_os(feffit::feffrun::BIN_DIR_ENV)
        && PathBuf::from(&dir).join("feff8l_pot").is_file()
    {
        return Some(feffit::feffrun::Feff8l::with_bin_dir(dir));
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for d in std::env::split_paths(&paths) {
            if d.join("feff8l_pot").is_file() {
                return Some(feffit::feffrun::Feff8l::new());
            }
        }
    }
    None
}

#[test]
fn feffrun_to_feffit_recovers_known_path_params() {
    let Some(runner) = runner() else {
        eprintln!(
            "SKIP feffrun_to_feffit_recovers_known_path_params: \
             feff8l_* not found (set {} or add to PATH)",
            feffit::feffrun::BIN_DIR_ENV
        );
        return;
    };

    // 1. Generate feffNNNN.dat from the Cu feff.inp.
    let workdir = std::env::temp_dir().join(format!("feffit-capstone-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&workdir);
    let out = runner
        .run(&cu_feff_inp(), &workdir)
        .expect("FEFF8L pipeline failed");
    assert!(
        out.dat_files.len() >= 10,
        "expected many feffNNNN.dat, got {}",
        out.dat_files.len()
    );
    let first = workdir.join("feff0001.dat");
    assert!(first.is_file(), "feff0001.dat was not generated");

    // Sanity: Cu fcc first shell — single-scattering, 12-fold, reff ≈ 2.55 Å.
    let probe = FeffPath::from_path(&first).unwrap();
    assert_eq!(
        probe.feffdat.nleg, 2,
        "first path should be single-scattering"
    );
    assert!(
        (probe.feffdat.reff - 2.55).abs() < 0.05,
        "reff {} not ~2.55 Å",
        probe.feffdat.reff
    );
    let degen = probe.feffdat.degen;

    // 2. Synthetic data: χ(k) of this path at KNOWN parameters.
    const TRUE_S02: f64 = 0.9;
    const TRUE_SIG2: f64 = 0.005;
    let mut truth = FeffPath::from_path(&first)
        .unwrap()
        .with_params(PathParams {
            s02: TRUE_S02,
            sigma2: TRUE_SIG2,
            ..PathParams::defaults(degen)
        });
    let (data_k, data_chi) = ff2chi(
        std::slice::from_mut(&mut truth),
        &KGrid::default_step(),
        Interp::Cubic,
    );

    // 3. Fit: same path, s02 = `amp`, σ² = `sig2`, starting away from truth.
    let fit_path = FeffPath::from_path(&first).unwrap();
    let mut spec = PathSpec::defaults(degen);
    spec.s02 = Spec::Expr("amp".into());
    spec.sigma2 = Spec::Expr("sig2".into());

    let transform = Transform::new(
        2.0,             // kmin
        17.0,            // kmax
        vec![2],         // kweight
        1.0,             // dk
        None,            // dk2
        Window::Hanning, // k window
        2048,            // nfft
        0.05,            // kstep
        1.0,             // rmin
        4.0,             // rmax
        0.0,             // dr
        None,            // dr2
        Window::Hanning, // R window
        0.0,             // rbkg (unused: fit in k-space)
        FitSpace::K,     // fit space
    );

    let dataset = DataSet::new(data_k, data_chi, vec![fit_path], transform);
    let mut fds = vec![FitDataSet {
        dataset,
        specs: vec![spec],
        epsilon_k: None,
    }];

    let mut p = Parameters::new();
    p.add_var("amp", 1.0);
    p.add_var("sig2", 0.003);

    let res = feffit(&mut p, &mut fds).expect("feffit");
    assert!(
        res.info >= 1 && res.info <= 4,
        "fit did not converge: info={}",
        res.info
    );

    let amp = res.best.iter().find(|b| b.name == "amp").unwrap().value;
    let sig2 = res.best.iter().find(|b| b.name == "sig2").unwrap().value;
    eprintln!(
        "capstone: recovered amp={amp:.6} (true {TRUE_S02}), \
         sig2={sig2:.6} (true {TRUE_SIG2}), nfev={}",
        res.nfev
    );

    // Self-consistency: the model generated the data, so recovery is tight.
    let rel = |g: f64, w: f64| (g - w).abs() / w.abs();
    assert!(rel(amp, TRUE_S02) < 1e-2, "amp {amp} vs {TRUE_S02}");
    assert!(rel(sig2, TRUE_SIG2) < 1e-2, "sig2 {sig2} vs {TRUE_SIG2}");

    let _ = std::fs::remove_dir_all(&workdir);
}
