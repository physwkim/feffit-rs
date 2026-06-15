//! Parity for the refined-background cubic B-spline pieces against FITPACK.
//!
//! `splev` is verified against `scipy.interpolate.splev` (which *is* FITPACK
//! `splev`) on the real `refine_bkg` knot vector with arbitrary coefficients,
//! evaluated on a model-k-style grid that extends below `kmin` and above `kmax`
//! so the FITPACK boundary-polynomial extrapolation is exercised. The knot
//! vector itself (`bkg_knots`) is verified against the knots `scipy.splrep`
//! returns (embedded in the same reference).

use std::collections::HashMap;
use std::path::PathBuf;

use feffit::bkg::{bkg_knots, splev};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// Read the `#begin <name> … #end` float blocks from a reference file.
fn load_blocks(name: &str) -> HashMap<String, Vec<f64>> {
    let text = std::fs::read_to_string(data_dir().join(name)).unwrap();
    let mut blocks: HashMap<String, Vec<f64>> = HashMap::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        if let Some(n) = line.strip_prefix("#begin ") {
            cur = Some(n.to_string());
            blocks.insert(n.to_string(), Vec::new());
        } else if line == "#end" {
            cur = None;
        } else if line.starts_with('#') {
            continue;
        } else if let Some(ref n) = cur {
            blocks.get_mut(n).unwrap().push(line.parse().unwrap());
        }
    }
    blocks
}

#[test]
fn splev_matches_fitpack() {
    let b = load_blocks("ref_splev.txt");

    for nspline in [5usize, 9, 13] {
        let knots = &b[&format!("knots_{nspline}")];
        let coefs = &b[&format!("coefs_{nspline}")];
        let x = &b[&format!("x_{nspline}")];
        let want = &b[&format!("y_{nspline}")];

        // closed-form knot vector matches scipy.splrep's knots exactly
        let kn = bkg_knots(3.0, 15.0, nspline);
        assert_eq!(kn.len(), knots.len(), "knot count nspline={nspline}");
        let kmaxd = kn
            .iter()
            .zip(knots)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        assert_eq!(
            kmaxd, 0.0,
            "knots nspline={nspline} differ (max {kmaxd:.3e})"
        );

        // splev matches FITPACK to round-off across the whole grid (incl.
        // extrapolation below kmin and above kmax)
        let got = splev(knots, coefs, 3, x);
        assert_eq!(got.len(), want.len(), "splev length nspline={nspline}");
        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        let maxd = got
            .iter()
            .zip(want)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "nspline={nspline}: splev peak={peak:.4e} maxd={maxd:.3e} rel={:.3e}",
            maxd / peak
        );
        assert!(
            maxd / peak < 1e-12,
            "splev nspline={nspline}: rel {:.3e}",
            maxd / peak
        );
    }
}
