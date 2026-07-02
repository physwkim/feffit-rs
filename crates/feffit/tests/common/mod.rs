//! Shared FEFF8L discovery for the `feffrun` integration tests
//! (`feffrun_run`, `feffrun_capstone`). Kept in one place so the executable-suffix
//! handling can't drift between the two — it did once: a bare `feff8l_pot` probe
//! never matched `feff8l_pot.exe`, so the `FEFF8L_DIR` set by Windows CI failed to
//! resolve and the tests silently self-skipped there.

use std::path::{Path, PathBuf};

use feffit::feffrun::{BIN_DIR_ENV, Feff8l};

/// A FEFF8L runner if the `feff8l_*` executables are reachable, else `None`
/// (the caller prints a SKIP notice and returns).
///
/// Resolution: an explicit `FEFF8L_DIR`, then `PATH`. The on-disk file carries
/// the platform executable suffix — `feff8l_pot.exe` on Windows, `feff8l_pot`
/// elsewhere — so probes append [`std::env::consts::EXE_SUFFIX`]. Without it the
/// `FEFF8L_DIR` a Windows runner sets never resolves and the test self-skips.
pub fn runner() -> Option<Feff8l> {
    let probe = format!("feff8l_pot{}", std::env::consts::EXE_SUFFIX);
    let has_exe = |dir: &Path| dir.join(&probe).is_file();

    // Explicit FEFF8L_DIR wins.
    if let Some(dir) = std::env::var_os(BIN_DIR_ENV)
        && has_exe(&PathBuf::from(&dir))
    {
        return Some(Feff8l::with_bin_dir(dir));
    }
    // Otherwise look for it on PATH.
    if let Some(paths) = std::env::var_os("PATH") {
        for d in std::env::split_paths(&paths) {
            if has_exe(&d) {
                return Some(Feff8l::new());
            }
        }
    }
    None
}
