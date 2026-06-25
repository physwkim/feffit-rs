//! Integration test for the beamline reader + μ(E) builder against a real XDI
//! file (`cu_metal_rt.xdi`, from the xraylarch examples). The file carries its
//! own precomputed `mutrans` column, so building transmission `mu` from `i0`/
//! `itrans` is checked directly against it — a self-contained parity test.

use feffit::xasdata::{ColumnFile, MuSpec, build_mu};

const XDI: &str = include_str!("data/cu_metal_rt.xdi");

#[test]
fn parses_xdi_columns_and_labels() {
    let cf = ColumnFile::from_text(XDI).expect("parse XDI");
    assert_eq!(cf.ncols(), 4);
    assert_eq!(cf.labels, vec!["energy", "i0", "itrans", "mutrans"]);
    assert!(
        cf.nrows() > 400,
        "expected the full scan, got {}",
        cf.nrows()
    );

    // First data row, verbatim from the file.
    assert_eq!(cf.column(0).unwrap()[0], 8779.0);
    assert_eq!(cf.column(1).unwrap()[0], 149013.7);

    let roles = cf.guess_roles();
    assert_eq!(roles.energy, Some(0));
    assert_eq!(roles.i0, Some(1));
    assert_eq!(roles.it, Some(2));
    assert_eq!(
        roles.mu,
        Some(3),
        "mutrans should be detected as a mu column"
    );
}

#[test]
fn transmission_mu_reproduces_file_mutrans_column() {
    let cf = ColumnFile::from_text(XDI).expect("parse XDI");
    let spec = MuSpec::Transmission {
        energy: 0,
        i0: 1,
        it: 2,
    };
    let (energy, mu) = build_mu(&cf, &spec).expect("build mu");

    let mutrans = cf.column(3).unwrap();
    assert_eq!(energy.len(), mu.len());
    assert_eq!(mu.len(), mutrans.len());

    // ln(i0/it) must equal the file's own mutrans to round-off across all rows.
    let max_err = mu
        .iter()
        .zip(mutrans)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-6,
        "transmission mu disagrees with file mutrans, max |Δ| = {max_err:e}"
    );
}

#[test]
fn raw_mode_returns_mutrans_unchanged() {
    let cf = ColumnFile::from_text(XDI).expect("parse XDI");
    let (_e, mu) = build_mu(&cf, &MuSpec::Raw { energy: 0, mu: 3 }).expect("raw mu");
    assert_eq!(mu, cf.column(3).unwrap().to_vec());
}
