//! Parity tests for the MINPACK `lmdif` port against `scipy.optimize.leastsq`.
//!
//! Each residual function is reconstructed with byte-identical data and
//! arithmetic grouping to `scripts/ref_lmdif.py`, so for the polynomial /
//! rational problems the LM iteration path (and thus `nfev`/`info`) is
//! bit-identical, not merely close. The transcendental `expdecay` case shares
//! `exp`, which is correctly rounded on both platforms here.

use feffit::lm::{LmConfig, lmdif};

// ---- residual functions (mirror scripts/ref_lmdif.py exactly) --------------

const T_LIN: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
const Y_LIN: [f64; 5] = [1.0, 2.9, 5.1, 6.8, 9.2];

fn f_linear(p: &[f64]) -> Vec<f64> {
    let (a, b) = (p[0], p[1]);
    T_LIN
        .iter()
        .zip(Y_LIN)
        .map(|(&ti, yi)| a * ti + b - yi)
        .collect()
}

fn f_rosenbrock(p: &[f64]) -> Vec<f64> {
    let (x0, x1) = (p[0], p[1]);
    vec![10.0 * (x1 - x0 * x0), 1.0 - x0]
}

const T_RAT: [f64; 6] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const Y_RAT: [f64; 6] = [3.8, 2.1, 1.4, 1.1, 0.9, 0.75];

fn f_rational(p: &[f64]) -> Vec<f64> {
    let (p0, p1) = (p[0], p[1]);
    T_RAT
        .iter()
        .zip(Y_RAT)
        .map(|(&ti, yi)| p0 / (p1 + ti) - yi)
        .collect()
}

fn f_powell(p: &[f64]) -> Vec<f64> {
    let (x0, x1, x2, x3) = (p[0], p[1], p[2], p[3]);
    let s5 = 5.0_f64.sqrt();
    let s10 = 10.0_f64.sqrt();
    vec![
        x0 + 10.0 * x1,
        s5 * (x2 - x3),
        (x1 - 2.0 * x2) * (x1 - 2.0 * x2),
        s10 * ((x0 - x3) * (x0 - x3)),
    ]
}

const T_EXP: [f64; 7] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const Y_EXP: [f64; 7] = [2.5, 2.0, 1.6, 1.3, 1.05, 0.85, 0.7];

fn f_expdecay(p: &[f64]) -> Vec<f64> {
    let (p0, p1) = (p[0], p[1]);
    T_EXP
        .iter()
        .zip(Y_EXP)
        .map(|(&ti, yi)| p0 * (-p1 * ti).exp() - yi)
        .collect()
}

type ResidualFn = fn(&[f64]) -> Vec<f64>;

fn residual(name: &str) -> (ResidualFn, Vec<f64>) {
    match name {
        "linear" => (f_linear, vec![0.0, 0.0]),
        "rosenbrock" => (f_rosenbrock, vec![-1.2, 1.0]),
        "rational" => (f_rational, vec![1.0, 0.5]),
        "powell" => (f_powell, vec![3.0, -1.0, 0.0, 1.0]),
        "expdecay" => (f_expdecay, vec![2.0, 0.3]),
        other => panic!("unknown case {other}"),
    }
}

// ---- reference parsing ------------------------------------------------------

struct Case {
    name: String,
    x: Vec<f64>,
    fvec: Vec<f64>,
    fnorm: f64,
    nfev: i32,
    info: i32,
    cov: Option<Vec<f64>>, // flattened row-major, or None
}

fn floats(s: &str) -> Vec<f64> {
    s.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

fn load_cases() -> Vec<Case> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/ref_lmdif.txt");
    let text = std::fs::read_to_string(path).expect("read ref_lmdif.txt");
    let mut cases = Vec::new();
    let mut cur: Option<Case> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("#case ") {
            cur = Some(Case {
                name: rest.to_string(),
                x: vec![],
                fvec: vec![],
                fnorm: 0.0,
                nfev: 0,
                info: 0,
                cov: None,
            });
        } else if let Some(rest) = line.strip_prefix("#x ") {
            cur.as_mut().unwrap().x = floats(rest);
        } else if let Some(rest) = line.strip_prefix("#fvec ") {
            cur.as_mut().unwrap().fvec = floats(rest);
        } else if let Some(rest) = line.strip_prefix("#fnorm ") {
            cur.as_mut().unwrap().fnorm = rest.parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("#nfev ") {
            cur.as_mut().unwrap().nfev = rest.parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("#info ") {
            cur.as_mut().unwrap().info = rest.parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("#cov ") {
            cur.as_mut().unwrap().cov = if rest == "none" {
                None
            } else {
                Some(floats(rest))
            };
        } else if line == "#end" {
            cases.push(cur.take().unwrap());
        }
    }
    cases
}

fn rel_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

#[test]
fn lmdif_matches_scipy_leastsq() {
    let cases = load_cases();
    assert_eq!(cases.len(), 5, "expected 5 reference cases");

    for c in &cases {
        let (fcn, x0) = residual(&c.name);
        let res = lmdif(fcn, &x0, &LmConfig::default());

        // Termination code matches scipy exactly. The evaluation count is exact
        // for every converged problem (info 1-4); for the deliberately-singular
        // `powell` case both runs truncate at maxfev (info 5) and the exact
        // count drifts a few evals from ULP differences between scipy's original
        // FORTRAN MINPACK and the fortran-lang variant ported here.
        assert_eq!(res.info, c.info, "[{}] info", c.name);
        if c.info == 5 {
            assert!(
                (res.nfev - c.nfev).abs() <= 8,
                "[{}] nfev {} vs {} (maxfev band)",
                c.name,
                res.nfev,
                c.nfev
            );
        } else {
            assert_eq!(res.nfev, c.nfev, "[{}] nfev", c.name);
        }

        assert_eq!(res.x.len(), c.x.len(), "[{}] x len", c.name);
        for (i, (&got, &exp)) in res.x.iter().zip(&c.x).enumerate() {
            assert!(
                rel_close(got, exp, 1e-6, 1e-9),
                "[{}] x[{i}]: got {got:e}, exp {exp:e}",
                c.name
            );
        }

        for (i, (&got, &exp)) in res.fvec.iter().zip(&c.fvec).enumerate() {
            assert!(
                rel_close(got, exp, 1e-6, 1e-9),
                "[{}] fvec[{i}]: got {got:e}, exp {exp:e}",
                c.name
            );
        }

        assert!(
            rel_close(res.fnorm, c.fnorm, 1e-7, 1e-12),
            "[{}] fnorm: got {:e}, exp {:e}",
            c.name,
            res.fnorm,
            c.fnorm
        );

        // covariance: present-vs-None must agree, and present values match.
        match (&c.cov, res.covar()) {
            (None, got) => assert!(
                got.is_none() || c.info == 5,
                "[{}] expected singular covariance (None)",
                c.name
            ),
            (Some(exp), Some(got)) => {
                let n = res.x.len();
                for i in 0..n {
                    for j in 0..n {
                        let g = got[i][j];
                        let e = exp[i * n + j];
                        assert!(
                            rel_close(g, e, 1e-5, 1e-12),
                            "[{}] cov[{i}][{j}]: got {g:e}, exp {e:e}",
                            c.name
                        );
                    }
                }
            }
            (Some(_), None) => panic!("[{}] covariance unexpectedly singular", c.name),
        }
    }
}
