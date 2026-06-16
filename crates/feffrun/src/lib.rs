//! `feffrun` — drive the FEFF8L path generator (subprocess) to turn a
//! `feff.inp` into `feffNNNN.dat` files.
//!
//! feffit-rs ports the EXAFS *math* to Rust (`feffdat` → `feffit`) but keeps
//! FEFF path *generation* as the original Fortran. This crate is that boundary:
//! it runs FEFF8L (from the `feff85exafs` project) as a subprocess pipeline —
//! the same six modules larch's `feffrunner` runs, in order:
//!
//! ```text
//! feff8l_rdinp → feff8l_pot → feff8l_xsph → feff8l_pathfinder
//!     → feff8l_genfmt → feff8l_ff2x
//! ```
//!
//! The pipeline writes `feffNNNN.dat` into a working directory; parse them with
//! `feffdat::FeffDatFile` (a dependency of the consumer, not of this crate).
//!
//! Subprocess rather than FFI is deliberate: the boundary stays at the
//! well-defined `feffNNNN.dat` file interface that `feffdat` already parses
//! bit-for-bit, and each module runs as its own process — so the native
//! executable's architecture is independent of this crate's (an arm64 build
//! drives the arm64 `feff8l_*` with no in-process arch coupling). The
//! `libfeff8lpath`/`libpotph` shared libraries are an alternative, per-path FFI
//! route not taken here.
//!
//! # Backends
//!
//! Two interchangeable backends produce the same `feffNNNN.dat` interface:
//!
//! - [`Feff8l`] — the FEFF8L subprocess pipeline described above; always
//!   available, needs the external `feff8l_*` executables.
//! - [`Feff10`] — the in-process FEFF10 pipeline from the `feff10` crate,
//!   gated behind the **`feff10`** cargo feature, which is **on by default**.
//!   It runs the FEFF10 Fortran in-process (one forked process per stage), so
//!   it needs no external executables. Build with `--no-default-features` to
//!   drop it and keep only [`Feff8l`].
//!
//! Pick one directly, or select at run time with [`Backend`] — whose
//! [`Default`] is [`Backend::Feff10`] in the default build. Both return a
//! [`RunOutput`] and write `feffNNNN.dat` into the working directory.
//!
//! # Locating the executables
//!
//! [`Feff8l`] resolves each `feff8l_*` in this order: an explicit directory from
//! [`Feff8l::with_bin_dir`], then the [`BIN_DIR_ENV`] (`FEFF8L_DIR`) environment
//! variable, then `PATH` (only when neither of the former is configured).
//!
//! # Scope
//!
//! The full six-module pipeline is always run, i.e. the equivalent of a feff.inp
//! `CONTROL 1 1 1 1 1 1`. Honouring a partial `CONTROL` card (skipping modules
//! to reuse prior outputs) is not implemented.

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The six FEFF8L modules, in pipeline order.
pub const MODULES: [&str; 6] = [
    "feff8l_rdinp",
    "feff8l_pot",
    "feff8l_xsph",
    "feff8l_pathfinder",
    "feff8l_genfmt",
    "feff8l_ff2x",
];

/// Environment variable naming the directory that holds the `feff8l_*`
/// executables. Consulted when no explicit bin directory is configured.
pub const BIN_DIR_ENV: &str = "FEFF8L_DIR";

/// A FEFF8L runner: knows where the `feff8l_*` executables live.
#[derive(Debug, Clone, Default)]
pub struct Feff8l {
    bin_dir: Option<PathBuf>,
}

/// Result of a successful pipeline run, from either backend.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// The directory the pipeline ran in (holds `feff.inp` and all outputs).
    pub workdir: PathBuf,
    /// `feffNNNN.dat` paths, sorted by file name.
    pub dat_files: Vec<PathBuf>,
}

/// Why a [`Feff8l`] run failed.
#[derive(Debug)]
pub enum FeffError {
    /// A module executable could not be found in any configured location.
    ExeNotFound {
        module: String,
        searched: Vec<PathBuf>,
    },
    /// The working directory does not contain a `feff.inp`.
    NoFeffInp(PathBuf),
    /// A module ran but exited non-zero.
    Module {
        module: String,
        code: Option<i32>,
        stderr: String,
    },
    /// The pipeline finished but produced no `feffNNNN.dat`.
    NoOutput(PathBuf),
    /// A non-FEFF8L backend pipeline (e.g. FEFF10) failed.
    #[cfg(feature = "feff10")]
    Backend {
        /// Backend name, e.g. `"feff10"`.
        backend: &'static str,
        /// The backend's own error message.
        message: String,
    },
    /// An I/O error while spawning a module or handling the working directory.
    Io {
        action: String,
        source: std::io::Error,
    },
}

impl fmt::Display for FeffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeffError::ExeNotFound { module, searched } => {
                write!(f, "FEFF8L executable `{module}` not found; looked in")?;
                for p in searched {
                    write!(f, " {}", p.display())?;
                }
                Ok(())
            }
            FeffError::NoFeffInp(p) => write!(f, "no feff.inp at {}", p.display()),
            FeffError::Module {
                module,
                code,
                stderr,
            } => {
                let code = code.map_or_else(|| "signal".to_string(), |c| c.to_string());
                write!(f, "module `{module}` exited {code}: {}", stderr.trim())
            }
            FeffError::NoOutput(p) => {
                write!(f, "pipeline produced no feffNNNN.dat in {}", p.display())
            }
            #[cfg(feature = "feff10")]
            FeffError::Backend { backend, message } => {
                write!(f, "{backend} pipeline failed: {message}")
            }
            FeffError::Io { action, source } => write!(f, "{action}: {source}"),
        }
    }
}

impl std::error::Error for FeffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FeffError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl Feff8l {
    /// A runner that resolves executables from [`BIN_DIR_ENV`] then `PATH`.
    pub fn new() -> Self {
        Self::default()
    }

    /// A runner that resolves executables from `dir` (falling back to neither
    /// the environment nor `PATH`).
    pub fn with_bin_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            bin_dir: Some(dir.into()),
        }
    }

    /// Build the [`Command`] for one module, resolving its executable path.
    fn command_for(&self, module: &str) -> Result<Command, FeffError> {
        // Configured directories, in precedence order. If any is set, the
        // executable must live in one of them.
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(d) = &self.bin_dir {
            dirs.push(d.clone());
        } else if let Some(d) = std::env::var_os(BIN_DIR_ENV) {
            dirs.push(PathBuf::from(d));
        }

        if dirs.is_empty() {
            // Nothing configured: hand a bare name to the OS and let it search
            // PATH at spawn time.
            return Ok(Command::new(module));
        }
        for d in &dirs {
            let p = d.join(module);
            if p.is_file() {
                return Ok(Command::new(p));
            }
        }
        Err(FeffError::ExeNotFound {
            module: module.to_string(),
            searched: dirs.iter().map(|d| d.join(module)).collect(),
        })
    }

    /// Run the full FEFF8L pipeline in `workdir`, which must already contain a
    /// `feff.inp`. Returns the generated `feffNNNN.dat` paths.
    pub fn run_in(&self, workdir: &Path) -> Result<RunOutput, FeffError> {
        let inp = workdir.join("feff.inp");
        if !inp.is_file() {
            return Err(FeffError::NoFeffInp(inp));
        }

        for module in MODULES {
            let mut cmd = self.command_for(module)?;
            cmd.current_dir(workdir);
            let output = cmd.output().map_err(|e| FeffError::Io {
                action: format!("spawn {module}"),
                source: e,
            })?;
            if !output.status.success() {
                return Err(FeffError::Module {
                    module: module.to_string(),
                    code: output.status.code(),
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                });
            }
        }

        let dat_files = collect_dat_files(workdir)?;
        Ok(RunOutput {
            workdir: workdir.to_path_buf(),
            dat_files,
        })
    }

    /// Write `feff_inp` into `workdir/feff.inp` (creating `workdir` if needed),
    /// then run the pipeline there.
    pub fn run(&self, feff_inp: &str, workdir: &Path) -> Result<RunOutput, FeffError> {
        std::fs::create_dir_all(workdir).map_err(|e| FeffError::Io {
            action: format!("create {}", workdir.display()),
            source: e,
        })?;
        std::fs::write(workdir.join("feff.inp"), feff_inp).map_err(|e| FeffError::Io {
            action: "write feff.inp".to_string(),
            source: e,
        })?;
        self.run_in(workdir)
    }
}

/// An in-process FEFF10 runner backed by the `feff10` crate.
///
/// Available only with the **`feff10`** cargo feature. Unlike [`Feff8l`],
/// FEFF10 needs no external executables: the `feff10` crate compiles the FEFF10
/// Fortran into this binary and runs each stage in a forked child, writing the
/// same `feffNNNN.dat` files into the working directory.
#[cfg(feature = "feff10")]
#[derive(Debug, Clone, Default)]
pub struct Feff10 {
    _priv: (),
}

#[cfg(feature = "feff10")]
impl Feff10 {
    /// A FEFF10 runner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the full FEFF10 pipeline in `workdir`, which must already contain a
    /// `feff.inp`. Returns the generated `feffNNNN.dat` paths.
    pub fn run_in(&self, workdir: &Path) -> Result<RunOutput, FeffError> {
        let inp = workdir.join("feff.inp");
        if !inp.is_file() {
            return Err(FeffError::NoFeffInp(inp));
        }
        let content = std::fs::read_to_string(&inp).map_err(|e| FeffError::Io {
            action: format!("read {}", inp.display()),
            source: e,
        })?;
        feff10::run_str(&content, workdir).map_err(|e| FeffError::Backend {
            backend: "feff10",
            message: e.to_string(),
        })?;
        let dat_files = collect_dat_files(workdir)?;
        Ok(RunOutput {
            workdir: workdir.to_path_buf(),
            dat_files,
        })
    }

    /// Write `feff_inp` into `workdir/feff.inp` (creating `workdir` if needed),
    /// then run the pipeline there.
    pub fn run(&self, feff_inp: &str, workdir: &Path) -> Result<RunOutput, FeffError> {
        std::fs::create_dir_all(workdir).map_err(|e| FeffError::Io {
            action: format!("create {}", workdir.display()),
            source: e,
        })?;
        std::fs::write(workdir.join("feff.inp"), feff_inp).map_err(|e| FeffError::Io {
            action: "write feff.inp".to_string(),
            source: e,
        })?;
        self.run_in(workdir)
    }
}

/// Which path-generator backend to drive.
///
/// A runtime selector over the available backends. [`Backend::Feff10`] exists
/// only with the `feff10` cargo feature (on by default). [`Default`] is
/// [`Backend::Feff10`] in the default build, falling back to [`Backend::Feff8l`]
/// when the feature is disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// The external FEFF8L subprocess pipeline ([`Feff8l`]). Always available;
    /// resolves executables from [`BIN_DIR_ENV`] then `PATH`.
    #[cfg_attr(not(feature = "feff10"), default)]
    Feff8l,
    /// The in-process FEFF10 pipeline ([`Feff10`]); the default backend.
    #[cfg(feature = "feff10")]
    #[default]
    Feff10,
}

impl Backend {
    /// Write `feff_inp` into `workdir/feff.inp` and run it with this backend,
    /// using each backend's default configuration.
    pub fn run(self, feff_inp: &str, workdir: &Path) -> Result<RunOutput, FeffError> {
        match self {
            Backend::Feff8l => Feff8l::new().run(feff_inp, workdir),
            #[cfg(feature = "feff10")]
            Backend::Feff10 => Feff10::new().run(feff_inp, workdir),
        }
    }
}

/// Collect and sort the `feffNNNN.dat` files a pipeline left in `workdir`.
fn collect_dat_files(workdir: &Path) -> Result<Vec<PathBuf>, FeffError> {
    let mut dat_files: Vec<PathBuf> = std::fs::read_dir(workdir)
        .map_err(|e| FeffError::Io {
            action: format!("read_dir {}", workdir.display()),
            source: e,
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(is_feff_dat)
        })
        .collect();
    dat_files.sort();
    if dat_files.is_empty() {
        return Err(FeffError::NoOutput(workdir.to_path_buf()));
    }
    Ok(dat_files)
}

/// Does `name` match `feffNNNN.dat` with one or more digits?
fn is_feff_dat(name: &str) -> bool {
    match name
        .strip_prefix("feff")
        .and_then(|s| s.strip_suffix(".dat"))
    {
        Some(mid) => !mid.is_empty() && mid.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::is_feff_dat;

    #[test]
    fn feff_dat_name_matching() {
        assert!(is_feff_dat("feff0001.dat"));
        assert!(is_feff_dat("feff1.dat"));
        assert!(is_feff_dat("feff9999.dat"));
        assert!(!is_feff_dat("feff.dat")); // no digits
        assert!(!is_feff_dat("feffNNNN.dat")); // non-digits
        assert!(!is_feff_dat("chi.dat"));
        assert!(!is_feff_dat("feff0001.txt"));
        assert!(!is_feff_dat("xfeff0001.dat")); // wrong prefix
    }
}
