//! The `xafsview` executable must double as a FEFF10 stage worker.
//!
//! FEFF10 runs every stage in a fresh process. Where it cannot fork — Windows,
//! and macOS once AppKit is initialized — it re-executes the running executable
//! with `FEFF10_WORKER_STAGE` / `FEFF10_WORKER_DIR` set and expects that
//! process to run the one stage and exit. `main` arranges this by calling
//! `feffit::feffrun::init_worker()` before anything else; without it the GUI
//! would start instead, and every FEFF10 run on those hosts fails before its
//! first stage. Spawn the built binary the way feff10 does and check that it
//! behaves as the `rdinp` stage.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The Cu `feff.inp` in the normalized form feff10 writes for its stages (the
/// stage reads that file, not the user's original).
const CU_FEFF_INP: &str = include_str!("data/feff.inp");

#[test]
fn xafsview_binary_runs_as_feff10_stage_worker() {
    let dir = std::env::temp_dir().join(format!("xafsview-feff10-worker-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create workdir");
    std::fs::write(dir.join("feff.inp"), CU_FEFF_INP).expect("write feff.inp");

    // The worker environment feff10 sets (`feff10::worker::{ENV_STAGE, ENV_DIR}`).
    let mut child = Command::new(env!("CARGO_BIN_EXE_xafsview"))
        .env("FEFF10_WORKER_STAGE", "rdinp")
        .env("FEFF10_WORKER_DIR", &dir)
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn xafsview");

    // A worker is done in well under a second. A binary that missed the hook
    // would sit in (or fail to open) the GUI instead, so do not wait forever.
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait for xafsview") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "xafsview did not exit as a stage worker within 120 s: is init_worker() first in main?"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(status.success(), "rdinp worker exited with {status}");

    // rdinp's job is to split feff.inp into the per-module inputs.
    for name in ["geom.dat", "pot.inp", "global.inp"] {
        assert!(dir.join(name).is_file(), "rdinp did not write {name}");
    }

    std::fs::remove_dir_all(&dir).ok();
}
