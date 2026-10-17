//! FEFF10 end to end: the real Cu `feff.inp` through the in-binary FEFF10
//! pipeline, in a test binary that installs the stage-worker hook the way every
//! host executable must.
//!
//! `harness = false` (see `Cargo.toml`): FEFF10 runs each stage in a fresh
//! process, and where it cannot fork — Windows, fork-unsafe macOS hosts — it
//! re-executes the current executable with the worker environment set. That
//! only works if `main` recognises the worker role before doing anything else,
//! so this binary's `main` is ours and `feffit::feffrun::init_worker()` is its
//! first statement; libtest-mimic then supplies the usual test protocol.
//!
//! Two cases, one per isolation mode the hook has to serve:
//! - `Auto` through [`feffit::feffrun::Backend::Feff10`] — the shipped path
//!   (fork on Linux/macOS, worker on Windows);
//! - `Worker` forced through the `feff10` pipeline — exercises the re-exec of
//!   this binary on every platform, so a broken hook shows up on the Linux
//!   gate and not only on a Windows machine.

use libtest_mimic::{Arguments, Trial};

/// The real Copper `feff.inp` (fcc, a = 3.61 Å), embedded at compile time.
const CU_FEFF_INP: &str = include_str!("data/feff.inp");

fn main() {
    // In a worker process this runs one FEFF stage and exits.
    feffit::feffrun::init_worker();

    let args = Arguments::from_args();
    let tests = vec![
        Trial::test("feff10_pipeline_generates_and_parses_cu_paths", || {
            feff10_pipeline_generates_and_parses_cu_paths();
            Ok(())
        }),
        Trial::test("feff10_worker_isolation_generates_cu_paths", || {
            feff10_worker_isolation_generates_cu_paths();
            Ok(())
        }),
    ];
    libtest_mimic::run(&args, tests).exit();
}

/// A fresh, empty working directory under the system temp dir.
fn workdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("feffrun-it-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create workdir");
    dir
}

/// The Cu pipeline through [`feffit::feffrun::Backend::Feff10`] with the
/// default (`Auto`) isolation, then `feff0001.dat` parsed with the ported
/// feffdat reader and its first-shell path checked.
fn feff10_pipeline_generates_and_parses_cu_paths() {
    let workdir = workdir("feff10");

    let out = feffit::feffrun::Backend::Feff10
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

    let dat = feffit::feffdat::FeffDatFile::from_path(&first).expect("parse feff0001.dat");
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

/// The same Cu pipeline with `Worker` isolation forced: every stage is a
/// re-execution of this test binary, which `init_worker()` at the top of
/// `main` turns into that single stage. Goes to the `feff10` pipeline directly
/// because [`feffit::feffrun::Feff10`] has no isolation knob (it does not need
/// one: `Auto` already picks the worker wherever a fork is not an option).
fn feff10_worker_isolation_generates_cu_paths() {
    use feff10::{FeffConfigBuilder, FeffInput, FeffPipeline, config::StageIsolation};

    let workdir = workdir("feff10-worker");
    let input = FeffInput::parse(CU_FEFF_INP).expect("parse the Cu feff.inp");
    let config = FeffConfigBuilder::new()
        .work_dir(&workdir)
        .input(input)
        .stage_isolation(StageIsolation::Worker)
        .build()
        .expect("FEFF10 config");
    FeffPipeline::new(config)
        .run()
        .expect("FEFF10 pipeline with worker isolation failed");

    // The Cu input's PRINT card asks for feffNNNN.dat (ff2chi field 3), so the
    // raw pipeline writes them just like the runner does.
    let n_dat = std::fs::read_dir(&workdir)
        .expect("read workdir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("feff") && name.ends_with(".dat") && name.len() == 12
        })
        .count();
    assert!(n_dat >= 10, "expected many feffNNNN.dat, got {n_dat}");
    assert!(
        workdir.join("feff0001.dat").is_file(),
        "feff0001.dat was not generated"
    );

    eprintln!("RAN feff10 pipeline (worker isolation): {n_dat} feffNNNN.dat");

    std::fs::remove_dir_all(&workdir).ok();
}
