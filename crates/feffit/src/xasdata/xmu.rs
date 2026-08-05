//! Build `mu(E)` from the columns of a [`ColumnFile`].
//!
//! Mirrors XAFSView's "Calc XMU": pick the energy column and the monitor
//! channels, choose the measurement mode, and compute the absorption
//! coefficient. The modes are kept as distinct, explicitly-named variants so a
//! channel can never silently play two roles:
//!
//! - **Transmission**: `mu = ln(I0 / It)`
//! - **Fluorescence**: `mu = (Σ signal channels) / I0`  (the 13-element-array
//!   sum drops dead channels simply by not listing them)
//! - **Reference**: `mu = ln(It / Iref)`  (a reference foil after the sample,
//!   used for energy alignment)
//! - **Raw**: use a precomputed `mu` column directly (e.g. an XDI `mutrans`)
//!
//! The arithmetic matches the convention used throughout XAFS (and reproduces
//! the `mutrans` column of an XDI file to round-off — see the crate tests).

use std::borrow::Cow;

use crate::xasdata::reader::ColumnFile;

/// The floor a monitor channel is raised to under
/// [`MonitorPolicy::ClampNonPositive`].
pub const MONITOR_FLOOR: f64 = 1.0;

/// How a monitor channel's recorded counts are treated before they enter a
/// ratio or a logarithm.
///
/// An ion chamber reading zero or negative — a dropped point, a baseline
/// over-subtraction, a detector glitch — makes `ln(I0/It)` non-finite and takes
/// the whole scan out of the plot's autoscale. Beamline viewers routinely floor
/// such counts at 1 so the rest of the scan stays visible. That is a deliberate
/// edit of the recorded data, so it is opt-in and named at every call site
/// rather than passed as a bare `bool`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MonitorPolicy {
    /// Use the counts as recorded. Non-positive values propagate as `NaN`/`inf`.
    #[default]
    AsRecorded,
    /// Raise non-positive counts to [`MONITOR_FLOOR`] before any ratio or log.
    ClampNonPositive,
}

impl MonitorPolicy {
    /// The counts to use for `channel` under this policy. Borrows under
    /// [`Self::AsRecorded`], so the default path does not allocate.
    pub fn apply(self, channel: &[f64]) -> Cow<'_, [f64]> {
        match self {
            MonitorPolicy::AsRecorded => Cow::Borrowed(channel),
            MonitorPolicy::ClampNonPositive => Cow::Owned(
                channel
                    .iter()
                    .map(|&v| if v <= 0.0 { MONITOR_FLOOR } else { v })
                    .collect(),
            ),
        }
    }
}

/// How to turn the monitor columns into `mu(E)`.
#[derive(Clone, Debug)]
pub enum MuSpec {
    /// Transmission: `mu = ln(i0 / it)`.
    Transmission {
        /// Energy column index.
        energy: usize,
        /// Incident-monitor (I0) column index.
        i0: usize,
        /// Transmitted-monitor (It) column index.
        it: usize,
    },
    /// Fluorescence: `mu = sum(channels) / i0`. List only the good channels.
    Fluorescence {
        /// Energy column index.
        energy: usize,
        /// Incident-monitor (I0) column index.
        i0: usize,
        /// Fluorescence signal column indices to sum (e.g. the live array pixels).
        channels: Vec<usize>,
    },
    /// Reference channel: `mu = ln(it / iref)`.
    Reference {
        /// Energy column index.
        energy: usize,
        /// Transmitted-monitor (It) column index — the incident beam on the foil.
        it: usize,
        /// Reference-foil transmitted-monitor (Iref) column index.
        iref: usize,
    },
    /// Use a precomputed `mu` column directly.
    Raw {
        /// Energy column index.
        energy: usize,
        /// Precomputed-mu column index.
        mu: usize,
    },
}

/// Why a `mu` build failed.
#[derive(Debug, PartialEq, Eq)]
pub enum XmuError {
    /// A referenced column index does not exist in the file.
    BadColumn(usize),
    /// No fluorescence channels were given.
    NoChannels,
}

impl std::fmt::Display for XmuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XmuError::BadColumn(i) => write!(f, "column index {i} is out of range"),
            XmuError::NoChannels => write!(f, "no fluorescence channels selected"),
        }
    }
}

impl std::error::Error for XmuError {}

/// Fetch a column or fail with [`XmuError::BadColumn`].
fn col(cf: &ColumnFile, i: usize) -> Result<&[f64], XmuError> {
    cf.column(i).ok_or(XmuError::BadColumn(i))
}

impl MuSpec {
    /// The energy column index this spec reads.
    pub fn energy_col(&self) -> usize {
        match *self {
            MuSpec::Transmission { energy, .. }
            | MuSpec::Fluorescence { energy, .. }
            | MuSpec::Reference { energy, .. }
            | MuSpec::Raw { energy, .. } => energy,
        }
    }
}

/// Build `(energy, mu)` from `cf` per `spec`, reading the monitor channels under
/// `monitors`.
///
/// Under [`MonitorPolicy::AsRecorded`] non-positive inputs to a logarithm or a
/// zero denominator yield non-finite values (`NaN`/`inf`) for those points
/// rather than an error, matching the numpy-based reference; clean XAS data is
/// strictly positive. [`MonitorPolicy::ClampNonPositive`] floors those counts
/// instead — see [`MonitorPolicy`].
///
/// The policy applies to every channel the formula divides by or takes the log
/// of, including the summed fluorescence signal; [`MuSpec::Raw`] reads a
/// precomputed `mu` rather than monitor counts and is unaffected.
pub fn build_mu(
    cf: &ColumnFile,
    spec: &MuSpec,
    monitors: MonitorPolicy,
) -> Result<(Vec<f64>, Vec<f64>), XmuError> {
    let energy = col(cf, spec.energy_col())?.to_vec();
    let mu = match spec {
        MuSpec::Transmission { i0, it, .. } => {
            let i0 = monitors.apply(col(cf, *i0)?);
            let it = monitors.apply(col(cf, *it)?);
            transmission_mu(&i0, &it)
        }
        MuSpec::Fluorescence { i0, channels, .. } => {
            if channels.is_empty() {
                return Err(XmuError::NoChannels);
            }
            let i0 = monitors.apply(col(cf, *i0)?);
            let mut chans: Vec<&[f64]> = Vec::with_capacity(channels.len());
            for &c in channels {
                chans.push(col(cf, c)?);
            }
            let signal = monitors.apply(&sum_channels(&chans)).into_owned();
            fluorescence_mu(&signal, &i0)
        }
        MuSpec::Reference { it, iref, .. } => {
            let it = monitors.apply(col(cf, *it)?);
            let iref = monitors.apply(col(cf, *iref)?);
            transmission_mu(&it, &iref)
        }
        MuSpec::Raw { mu, .. } => col(cf, *mu)?.to_vec(),
    };
    Ok((energy, mu))
}

/// `mu = ln(num / den)` element-wise (transmission / reference).
pub fn transmission_mu(num: &[f64], den: &[f64]) -> Vec<f64> {
    num.iter().zip(den).map(|(&n, &d)| (n / d).ln()).collect()
}

/// `mu = signal / i0` element-wise (fluorescence).
pub fn fluorescence_mu(signal: &[f64], i0: &[f64]) -> Vec<f64> {
    signal.iter().zip(i0).map(|(&s, &d)| s / d).collect()
}

/// Element-wise sum of several equal-length channels (multi-element detector).
/// Shorter channels stop contributing past their end; an empty list sums to an
/// empty vector.
pub fn sum_channels(channels: &[&[f64]]) -> Vec<f64> {
    let n = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut out = vec![0.0; n];
    for ch in channels {
        for (o, &v) in out.iter_mut().zip(ch.iter()) {
            *o += v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xasdata::reader::ColumnFile;

    fn cf3() -> ColumnFile {
        // energy, i0, it
        ColumnFile::from_text("# energy i0 it\n100 10 5\n200 20 8\n").unwrap()
    }

    #[test]
    fn transmission_matches_ln_ratio() {
        let cf = cf3();
        let spec = MuSpec::Transmission {
            energy: 0,
            i0: 1,
            it: 2,
        };
        let (e, mu) = build_mu(&cf, &spec, MonitorPolicy::AsRecorded).unwrap();
        assert_eq!(e, vec![100.0, 200.0]);
        assert!((mu[0] - (10.0f64 / 5.0).ln()).abs() < 1e-15);
        assert!((mu[1] - (20.0f64 / 8.0).ln()).abs() < 1e-15);
    }

    #[test]
    fn fluorescence_sums_channels_over_i0() {
        // energy, i0, ch1, ch2
        let cf = ColumnFile::from_text("# e i0 c1 c2\n1 100 3 4\n2 200 6 8\n").unwrap();
        let spec = MuSpec::Fluorescence {
            energy: 0,
            i0: 1,
            channels: vec![2, 3],
        };
        let (_e, mu) = build_mu(&cf, &spec, MonitorPolicy::AsRecorded).unwrap();
        assert!((mu[0] - 7.0 / 100.0).abs() < 1e-15);
        assert!((mu[1] - 14.0 / 200.0).abs() < 1e-15);
    }

    #[test]
    fn as_recorded_lets_a_non_positive_monitor_go_non_finite() {
        // energy, i0, it — the second row's It is a dropped point.
        let cf = ColumnFile::from_text("# energy i0 it\n100 10 5\n200 20 0\n").unwrap();
        let spec = MuSpec::Transmission {
            energy: 0,
            i0: 1,
            it: 2,
        };
        let (_e, mu) = build_mu(&cf, &spec, MonitorPolicy::AsRecorded).unwrap();
        assert!((mu[0] - (10.0f64 / 5.0).ln()).abs() < 1e-15);
        assert!(mu[1].is_infinite(), "ln(20/0) is +inf, got {}", mu[1]);
    }

    #[test]
    fn clamp_floors_the_monitor_and_leaves_the_good_point_alone() {
        let cf = ColumnFile::from_text("# energy i0 it\n100 10 5\n200 20 0\n").unwrap();
        let spec = MuSpec::Transmission {
            energy: 0,
            i0: 1,
            it: 2,
        };
        let (_e, mu) = build_mu(&cf, &spec, MonitorPolicy::ClampNonPositive).unwrap();
        assert!(
            (mu[0] - (10.0f64 / 5.0).ln()).abs() < 1e-15,
            "unclamped row"
        );
        assert!((mu[1] - (20.0f64 / MONITOR_FLOOR).ln()).abs() < 1e-15);
    }

    #[test]
    fn clamp_reaches_a_negative_denominator_and_the_fluorescence_signal() {
        // A negative I0 (over-subtracted baseline) and a negative signal sum.
        let cf = ColumnFile::from_text("# e i0 c1 c2\n1 -4 -3 -2\n2 200 6 8\n").unwrap();
        let spec = MuSpec::Fluorescence {
            energy: 0,
            i0: 1,
            channels: vec![2, 3],
        };
        let recorded = build_mu(&cf, &spec, MonitorPolicy::AsRecorded).unwrap().1;
        assert!((recorded[0] - (-5.0 / -4.0)).abs() < 1e-15);
        let clamped = build_mu(&cf, &spec, MonitorPolicy::ClampNonPositive)
            .unwrap()
            .1;
        assert!((clamped[0] - (MONITOR_FLOOR / MONITOR_FLOOR)).abs() < 1e-15);
        assert!(
            (clamped[1] - 14.0 / 200.0).abs() < 1e-15,
            "good row untouched"
        );
    }

    #[test]
    fn raw_ignores_the_monitor_policy() {
        // Raw reads a precomputed mu, which is signed: clamping it would be wrong.
        let cf = ColumnFile::from_text("# e mu\n1 -0.5\n2 1.5\n").unwrap();
        let spec = MuSpec::Raw { energy: 0, mu: 1 };
        let clamped = build_mu(&cf, &spec, MonitorPolicy::ClampNonPositive)
            .unwrap()
            .1;
        assert_eq!(clamped, vec![-0.5, 1.5]);
    }

    #[test]
    fn raw_passes_column_through() {
        let cf = ColumnFile::from_text("# e mu\n1 0.5\n2 1.5\n").unwrap();
        let spec = MuSpec::Raw { energy: 0, mu: 1 };
        let (_e, mu) = build_mu(&cf, &spec, MonitorPolicy::AsRecorded).unwrap();
        assert_eq!(mu, vec![0.5, 1.5]);
    }

    #[test]
    fn bad_column_errors() {
        let cf = cf3();
        let spec = MuSpec::Transmission {
            energy: 0,
            i0: 1,
            it: 9,
        };
        assert_eq!(
            build_mu(&cf, &spec, MonitorPolicy::AsRecorded),
            Err(XmuError::BadColumn(9))
        );
    }

    #[test]
    fn empty_channels_error() {
        let cf = cf3();
        let spec = MuSpec::Fluorescence {
            energy: 0,
            i0: 1,
            channels: vec![],
        };
        assert_eq!(
            build_mu(&cf, &spec, MonitorPolicy::AsRecorded),
            Err(XmuError::NoChannels)
        );
    }

    #[test]
    fn sum_channels_handles_ragged() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0];
        assert_eq!(sum_channels(&[&a, &b]), vec![11.0, 22.0, 3.0]);
        assert_eq!(sum_channels(&[]), Vec::<f64>::new());
    }
}
