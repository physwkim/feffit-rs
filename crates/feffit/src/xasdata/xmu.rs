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

use crate::xasdata::reader::ColumnFile;

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

/// Build `(energy, mu)` from `cf` per `spec`.
///
/// Non-positive inputs to a logarithm or a zero denominator yield non-finite
/// values (`NaN`/`inf`) for those points rather than an error, matching the
/// numpy-based reference; clean XAS data is strictly positive.
pub fn build_mu(cf: &ColumnFile, spec: &MuSpec) -> Result<(Vec<f64>, Vec<f64>), XmuError> {
    let energy = col(cf, spec.energy_col())?.to_vec();
    let mu = match spec {
        MuSpec::Transmission { i0, it, .. } => {
            let i0 = col(cf, *i0)?;
            let it = col(cf, *it)?;
            transmission_mu(i0, it)
        }
        MuSpec::Fluorescence { i0, channels, .. } => {
            if channels.is_empty() {
                return Err(XmuError::NoChannels);
            }
            let i0 = col(cf, *i0)?;
            let mut chans: Vec<&[f64]> = Vec::with_capacity(channels.len());
            for &c in channels {
                chans.push(col(cf, c)?);
            }
            let signal = sum_channels(&chans);
            fluorescence_mu(&signal, i0)
        }
        MuSpec::Reference { it, iref, .. } => {
            let it = col(cf, *it)?;
            let iref = col(cf, *iref)?;
            transmission_mu(it, iref)
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
        let (e, mu) = build_mu(&cf, &spec).unwrap();
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
        let (_e, mu) = build_mu(&cf, &spec).unwrap();
        assert!((mu[0] - 7.0 / 100.0).abs() < 1e-15);
        assert!((mu[1] - 14.0 / 200.0).abs() < 1e-15);
    }

    #[test]
    fn raw_passes_column_through() {
        let cf = ColumnFile::from_text("# e mu\n1 0.5\n2 1.5\n").unwrap();
        let spec = MuSpec::Raw { energy: 0, mu: 1 };
        let (_e, mu) = build_mu(&cf, &spec).unwrap();
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
        assert_eq!(build_mu(&cf, &spec), Err(XmuError::BadColumn(9)));
    }

    #[test]
    fn empty_channels_error() {
        let cf = cf3();
        let spec = MuSpec::Fluorescence {
            energy: 0,
            i0: 1,
            channels: vec![],
        };
        assert_eq!(build_mu(&cf, &spec), Err(XmuError::NoChannels));
    }

    #[test]
    fn sum_channels_handles_ragged() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0];
        assert_eq!(sum_channels(&[&a, &b]), vec![11.0, 22.0, 3.0]);
        assert_eq!(sum_channels(&[]), Vec::<f64>::new());
    }
}
