//! Integration test: drive the native FEFF8L pipeline on a real Cu `feff.inp`
//! and parse the generated `feff0001.dat` with `feffdat`.
//!
//! This needs the `feff8l_*` executables. Point `FEFF8L_DIR` at the directory
//! holding them (e.g. `feff85exafs/local_install/bin`), or put them on `PATH`.
//! When they are not available the test prints a SKIP notice and returns rather
//! than failing, so it is inert on machines without FEFF built.

use std::path::PathBuf;

/// The real Copper `feff.inp` (fcc, a = 3.61 Å), embedded at compile time.
const CU_FEFF_INP: &str = include_str!("data/feff.inp");

/// A runner if the executables are reachable, else `None` (with the reason).
fn runner() -> Option<feffrun::Feff8l> {
    // Explicit FEFF8L_DIR wins.
    if let Some(dir) = std::env::var_os(feffrun::BIN_DIR_ENV)
        && PathBuf::from(&dir).join("feff8l_pot").is_file()
    {
        return Some(feffrun::Feff8l::with_bin_dir(dir));
    }
    // Otherwise look for it on PATH.
    if let Some(paths) = std::env::var_os("PATH") {
        for d in std::env::split_paths(&paths) {
            if d.join("feff8l_pot").is_file() {
                return Some(feffrun::Feff8l::new());
            }
        }
    }
    None
}

#[test]
fn feff8l_pipeline_generates_and_parses_cu_paths() {
    let Some(runner) = runner() else {
        eprintln!(
            "SKIP feff8l_pipeline_generates_and_parses_cu_paths: \
             feff8l_* not found (set {} or add to PATH)",
            feffrun::BIN_DIR_ENV
        );
        return;
    };

    let workdir = std::env::temp_dir().join(format!("feffrun-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&workdir);

    let out = runner
        .run(CU_FEFF_INP, &workdir)
        .expect("FEFF8L pipeline failed");

    // Copper fcc generates many paths within RPATH = 5.2 Å.
    assert!(
        out.dat_files.len() >= 10,
        "expected many feffNNNN.dat, got {}",
        out.dat_files.len()
    );

    let first = workdir.join("feff0001.dat");
    assert!(first.is_file(), "feff0001.dat was not generated");
    assert_eq!(
        out.dat_files[0], first,
        "first dat file should be feff0001.dat"
    );

    // Parse it with the ported feffdat reader and check the first-shell path.
    let dat = feffdat::FeffDatFile::from_path(&first).expect("parse feff0001.dat");
    assert_eq!(
        dat.nleg, 2,
        "first path is a single-scattering (2-leg) path"
    );
    assert!(
        (dat.reff - 2.5527).abs() < 2e-3,
        "Cu first-shell reff ≈ 2.5527 Å, got {}",
        dat.reff
    );
    assert!(
        (dat.degen - 12.0).abs() < 1e-9,
        "Cu fcc first shell has 12 neighbours, got {}",
        dat.degen
    );
    // k grid is populated and the amplitude is non-trivial.
    assert!(!dat.k.is_empty(), "empty k grid");
    assert!(
        dat.amp.iter().any(|&a| a > 0.0),
        "first-shell amplitude is all zero"
    );

    eprintln!(
        "RAN feff8l pipeline: {} feffNNNN.dat, feff0001 reff={:.4} nleg={} degen={}",
        out.dat_files.len(),
        dat.reff,
        dat.nleg,
        dat.degen
    );

    std::fs::remove_dir_all(&workdir).ok();
}

/// The same Cu pipeline driven through the in-process FEFF10 backend, selected
/// via [`feffrun::Backend::Feff10`]. Only built with the `feff10` feature; it
/// needs no external executables (FEFF10 Fortran is compiled into the binary).
#[cfg(feature = "feff10")]
#[test]
fn feff10_pipeline_generates_and_parses_cu_paths() {
    let workdir = std::env::temp_dir().join(format!("feffrun-it-feff10-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&workdir);

    let out = feffrun::Backend::Feff10
        .run(CU_FEFF_INP, &workdir)
        .expect("FEFF10 pipeline failed");

    // Copper fcc generates many paths within RPATH = 5.2 Å.
    assert!(
        out.dat_files.len() >= 10,
        "expected many feffNNNN.dat, got {}",
        out.dat_files.len()
    );

    let first = workdir.join("feff0001.dat");
    assert!(first.is_file(), "feff0001.dat was not generated");
    assert_eq!(
        out.dat_files[0], first,
        "first dat file should be feff0001.dat"
    );

    // Parse it with the ported feffdat reader and check the first-shell path.
    let dat = feffdat::FeffDatFile::from_path(&first).expect("parse feff0001.dat");
    assert_eq!(
        dat.nleg, 2,
        "first path is a single-scattering (2-leg) path"
    );
    assert!(
        (dat.reff - 2.5527).abs() < 2e-3,
        "Cu first-shell reff ≈ 2.5527 Å, got {}",
        dat.reff
    );
    assert!(
        (dat.degen - 12.0).abs() < 1e-9,
        "Cu fcc first shell has 12 neighbours, got {}",
        dat.degen
    );
    // k grid is populated and the amplitude is non-trivial.
    assert!(!dat.k.is_empty(), "empty k grid");
    assert!(
        dat.amp.iter().any(|&a| a > 0.0),
        "first-shell amplitude is all zero"
    );

    eprintln!(
        "RAN feff10 pipeline: {} feffNNNN.dat, feff0001 reff={:.4} nleg={} degen={}",
        out.dat_files.len(),
        dat.reff,
        dat.nleg,
        dat.degen
    );

    std::fs::remove_dir_all(&workdir).ok();
}
