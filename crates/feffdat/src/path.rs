//! The EXAFS path sum, a port of `larch.xafs.feffdat.FeffPathGroup._calc_chi`,
//! `path2chi`, and `ff2chi`.

use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::constants::{ETOK, SMALL_ENERGY};
use crate::interp::{interp_linear, CubicSpline, Interp};
use crate::parser::FeffDatFile;

/// The seven adjustable path parameters (`degen` plus the six refined ones).
///
/// Defaults match `init_path_params`: `degen` from the feff.dat file,
/// `s02 = 1`, everything else `0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathParams {
    pub degen: f64,
    pub s02: f64,
    pub e0: f64,
    pub ei: f64,
    pub deltar: f64,
    pub sigma2: f64,
    pub third: f64,
    pub fourth: f64,
}

impl PathParams {
    /// larch defaults, with `degen` taken from the file's tabulated degeneracy.
    pub fn defaults(file_degen: f64) -> Self {
        PathParams {
            degen: file_degen,
            s02: 1.0,
            e0: 0.0,
            ei: 0.0,
            deltar: 0.0,
            sigma2: 0.0,
            third: 0.0,
            fourth: 0.0,
        }
    }
}

/// How to build the output wavenumber grid.
#[derive(Debug, Clone)]
pub enum KGrid {
    /// Use an explicit k array verbatim.
    Explicit(Vec<f64>),
    /// `k = kstep * arange(int(1.01 + kmax/kstep))`, with
    /// `kmax = min(max(feff.k), kmax_cap.unwrap_or(30))`.
    Step { kstep: f64, kmax_cap: Option<f64> },
}

impl KGrid {
    /// The `ff2chi` / `path2chi` default: `kstep = 0.05`, `kmax` capped by the file.
    pub fn default_step() -> Self {
        KGrid::Step {
            kstep: 0.05,
            kmax_cap: None,
        }
    }
}

/// A Feff path: the parsed file plus its path parameters and cached splines.
#[derive(Debug, Clone)]
pub struct FeffPath {
    pub feffdat: FeffDatFile,
    pub params: PathParams,
    pub use_path: bool,

    spline_pha: CubicSpline,
    spline_amp: CubicSpline,
    spline_rep: CubicSpline,
    spline_lam: CubicSpline,

    /// Output wavenumber grid from the last `calc_chi`.
    pub k: Vec<f64>,
    /// Imag part of the complex path sum — i.e. chi(k).
    pub chi: Vec<f64>,
    /// `-real` part of the complex path sum.
    pub chi_imag: Vec<f64>,
}

impl FeffPath {
    /// Read a feff.dat file and wrap it as a path with default parameters.
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let feffdat = FeffDatFile::from_path(path)?;
        Ok(Self::new(feffdat))
    }

    /// Build a path from an already-parsed file, with default parameters.
    pub fn new(feffdat: FeffDatFile) -> Self {
        let params = PathParams::defaults(feffdat.degen);
        // pre-calculate spline coefficients (larch: create_spline_coefs)
        let spline_pha = CubicSpline::new(&feffdat.k, &feffdat.pha);
        let spline_amp = CubicSpline::new(&feffdat.k, &feffdat.amp);
        let spline_rep = CubicSpline::new(&feffdat.k, &feffdat.rep);
        let spline_lam = CubicSpline::new(&feffdat.k, &feffdat.lam);
        FeffPath {
            feffdat,
            params,
            use_path: true,
            spline_pha,
            spline_amp,
            spline_rep,
            spline_lam,
            k: Vec::new(),
            chi: Vec::new(),
            chi_imag: Vec::new(),
        }
    }

    /// Replace the path parameters (builder-style).
    pub fn with_params(mut self, params: PathParams) -> Self {
        self.params = params;
        self
    }

    fn build_k(&self, grid: &KGrid) -> Vec<f64> {
        match grid {
            KGrid::Explicit(v) => v.clone(),
            KGrid::Step { kstep, kmax_cap } => {
                let kmax = kmax_cap.unwrap_or(30.0).min(self.feffdat.k_max());
                let n = (1.01 + kmax / kstep).trunc() as usize;
                (0..n).map(|i| kstep * i as f64).collect()
            }
        }
    }

    /// Compute chi(k), writing `self.k`, `self.chi`, `self.chi_imag`.
    /// Mirrors `_calc_chi` exactly (the EXAFS equation).
    pub fn calc_chi(&mut self, grid: &KGrid, interp: Interp) {
        let fdat = &self.feffdat;
        if fdat.reff < 0.05 {
            // larch prints and returns without writing chi
            return;
        }
        let k = self.build_k(grid);
        let n = k.len();

        if !self.use_path {
            self.k = k;
            self.chi = vec![0.0; n];
            self.chi_imag = vec![0.0; n];
            return;
        }

        let reff = fdat.reff;
        let p = self.params;

        // e0-shifted energy, with the |e0|~=0 guard.
        let mut en: Vec<f64> = k.iter().map(|kk| kk * kk - p.e0 * ETOK).collect();
        let min_abs = en.iter().fold(f64::INFINITY, |acc, &e| acc.min(e.abs()));
        if min_abs < SMALL_ENERGY {
            for e in en.iter_mut() {
                if e.abs() < 1.5 * SMALL_ENERGY {
                    *e = SMALL_ENERGY;
                }
            }
        }
        // e0-shifted wavenumber q = sign(en) * sqrt(|en|)  (numpy sign: sign(0)=0)
        let q: Vec<f64> = en.iter().map(|&e| npsign(e) * e.abs().sqrt()).collect();

        let mut cchi = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            let qi = q[i];
            let (pha, amp, rep, lam) = match interp {
                Interp::Linear => (
                    interp_linear(qi, &fdat.k, &fdat.pha),
                    interp_linear(qi, &fdat.k, &fdat.amp),
                    interp_linear(qi, &fdat.k, &fdat.rep),
                    interp_linear(qi, &fdat.k, &fdat.lam),
                ),
                Interp::Cubic => (
                    self.spline_pha.eval(qi),
                    self.spline_amp.eval(qi),
                    self.spline_rep.eval(qi),
                    self.spline_lam.eval(qi),
                ),
            };

            // p = complex wavenumber; pp = p^2
            let base = Complex64::new(rep, 1.0 / lam);
            let pp = base * base + Complex64::new(0.0, p.ei * ETOK);
            let pcx = pp.sqrt();

            // the xafs equation (exponent assembled term by term)
            let term_a = Complex64::new(-2.0 * reff * pcx.im, 0.0);
            let term_b = -2.0 * pp * (Complex64::new(p.sigma2, 0.0) - pp * (p.fourth / 3.0));
            let inner_p =
                Complex64::new(p.deltar - 2.0 * p.sigma2 / reff, 0.0) - pp * (2.0 * p.third / 3.0);
            let imag_arg = Complex64::new(2.0 * qi * reff + pha, 0.0) + 2.0 * pcx * inner_p;
            let expo = term_a + term_b + Complex64::new(0.0, 1.0) * imag_arg;

            let denom = qi * (reff + p.deltar) * (reff + p.deltar);
            let scale = p.degen * p.s02 * amp / denom;
            cchi[i] = expo.exp() * scale;
        }

        // fix the k=0 singularity by linear extrapolation
        if n >= 3 {
            cchi[0] = 2.0 * cchi[1] - cchi[2];
        }

        self.k = k;
        self.chi = cchi.iter().map(|c| c.im).collect();
        self.chi_imag = cchi.iter().map(|c| -c.re).collect();
    }
}

/// numpy `sign`: `sign(0) == 0` (Rust `f64::signum` returns ±1 for zero).
#[inline]
fn npsign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Compute chi(k) for a single path (port of `path2chi`).
pub fn path2chi(path: &mut FeffPath, grid: &KGrid, interp: Interp) {
    path.calc_chi(grid, interp);
}

/// Sum chi(k) over a list of paths on a common grid (port of `ff2chi`).
/// Returns `(k, chi)`.
pub fn ff2chi(paths: &mut [FeffPath], grid: &KGrid, interp: Interp) -> (Vec<f64>, Vec<f64>) {
    if paths.is_empty() {
        // larch returns a default 0..20 / 401-point zero array
        let k: Vec<f64> = (0..401).map(|i| 20.0 * i as f64 / 400.0).collect();
        let chi = vec![0.0; 401];
        return (k, chi);
    }
    for path in paths.iter_mut() {
        path.calc_chi(grid, interp);
    }
    let k = paths[0].k.clone();
    let len = k.len();
    let mut out = vec![0.0; len];
    for path in paths.iter() {
        assert_eq!(
            path.chi.len(),
            len,
            "paths produced different-length k grids; cannot sum (matches larch broadcasting failure)"
        );
        for (o, c) in out.iter_mut().zip(&path.chi) {
            *o += c;
        }
    }
    (k, out)
}
