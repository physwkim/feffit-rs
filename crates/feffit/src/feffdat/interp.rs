//! Interpolators used to resample the `feff.dat` arrays onto the e0-shifted
//! wavenumber grid `q`.
//!
//! larch's `_calc_chi` supports two modes:
//!   * `interp='lin'`    -> `numpy.interp` (linear, endpoint-clamped)
//!   * `interp='cubic'`  -> `scipy.interpolate.UnivariateSpline(k, y, s=0)`  \[default\]
//!
//! [`interp_linear`] reproduces `numpy.interp` exactly. [`CubicSpline`] is a
//! not-a-knot cubic interpolant; FITPACK's `s=0` cubic interpolation *is* the
//! not-a-knot spline, and the resulting chi(k) matches scipy
//! `UnivariateSpline(s=0)` to ~5e-14 (in-range and in the extrapolation
//! region) — see `tests/parity.rs::chi_cubic_*`.

/// Selects the resampling scheme, mirroring larch's `interp=` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    /// `numpy.interp` — linear with endpoint clamping.
    Linear,
    /// not-a-knot cubic spline (matches scipy `UnivariateSpline(s=0)`).
    Cubic,
}

/// Exact reproduction of `numpy.interp(xq, xp, fp)`:
/// linear interpolation, clamped to `fp[0]` / `fp[last]` outside `[xp[0], xp[last]]`.
/// `xp` must be sorted ascending (the feff.dat k grid is).
pub fn interp_linear(xq: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    debug_assert_eq!(n, fp.len());
    if n == 0 {
        return f64::NAN;
    }
    if xq <= xp[0] {
        return fp[0];
    }
    if xq >= xp[n - 1] {
        return fp[n - 1];
    }
    // first index with xp[i] >= xq
    let i = match xp.binary_search_by(|v| v.partial_cmp(&xq).unwrap()) {
        Ok(i) => return fp[i],
        Err(i) => i,
    };
    let (x0, x1) = (xp[i - 1], xp[i]);
    let (y0, y1) = (fp[i - 1], fp[i]);
    y0 + (y1 - y0) * (xq - x0) / (x1 - x0)
}

/// Precomputed not-a-knot cubic spline over a strictly increasing `x` grid.
///
/// On segment `i` (over `[x[i], x[i+1]]`) the value is
/// `a[i] + b[i]*dx + c[i]*dx^2 + d[i]*dx^3` with `dx = t - x[i]`. Evaluation
/// outside the data range extrapolates with the nearest boundary segment
/// (scipy `ext=0`).
#[derive(Debug, Clone)]
pub struct CubicSpline {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline {
    /// Build the spline from sample points. Requires `x.len() == y.len()`.
    /// For fewer than 4 points it falls back to a natural spline / linear /
    /// constant fit (the example grids always have many points).
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        assert_eq!(x.len(), y.len(), "x and y length mismatch");
        let n = x.len();
        assert!(n >= 2, "need at least 2 points");

        let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();

        // second derivatives m[0..n]
        let m = if n < 4 {
            natural_second_derivs(&h, y, n)
        } else {
            not_a_knot_second_derivs(&h, y, n)
        };

        let mut a = vec![0.0; n - 1];
        let mut b = vec![0.0; n - 1];
        let mut c = vec![0.0; n - 1];
        let mut d = vec![0.0; n - 1];
        for i in 0..n - 1 {
            a[i] = y[i];
            b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0;
            c[i] = m[i] / 2.0;
            d[i] = (m[i + 1] - m[i]) / (6.0 * h[i]);
        }
        CubicSpline {
            x: x.to_vec(),
            a,
            b,
            c,
            d,
        }
    }

    /// Evaluate the spline at `t` (extrapolating beyond the data range).
    pub fn eval(&self, t: f64) -> f64 {
        let nseg = self.a.len();
        // locate segment: largest i with x[i] <= t, clamped to [0, nseg-1]
        let i = match self.x.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
            Ok(i) => i.min(nseg - 1),
            Err(0) => 0,
            Err(pos) => (pos - 1).min(nseg - 1),
        };
        let dx = t - self.x[i];
        ((self.d[i] * dx + self.c[i]) * dx + self.b[i]) * dx + self.a[i]
    }
}

/// Natural-spline second derivatives (m[0] = m[n-1] = 0). Used as the small-n
/// fallback only.
fn natural_second_derivs(h: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n];
    if n < 3 {
        return m; // single segment -> straight line, m all zero
    }
    let mut sub = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    diag[0] = 1.0;
    diag[n - 1] = 1.0;
    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }
    thomas(&sub, &diag, &sup, &rhs, &mut m);
    m
}

/// Not-a-knot second derivatives: enforces a continuous third derivative across
/// the first and last interior knots (scipy/FITPACK `s=0` boundary behaviour).
fn not_a_knot_second_derivs(h: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    // Interior system in m[1..n-1] after eliminating m[0], m[n-1] via:
    //   m[0]   = (1 + h0/h1) m1 - (h0/h1) m2
    //   m[n-1] = (1 + h_{n-2}/h_{n-3}) m_{n-2} - (h_{n-2}/h_{n-3}) m_{n-3}
    let nint = n - 2; // unknowns m[1..=n-2]
    let mut sub = vec![0.0; nint];
    let mut diag = vec![0.0; nint];
    let mut sup = vec![0.0; nint];
    let mut rhs = vec![0.0; nint];

    let slope = |i: usize| (y[i + 1] - y[i]) / h[i];
    for k in 0..nint {
        let i = k + 1; // global index of the unknown m[i]
        let lo = h[i - 1];
        let hi = h[i];
        sub[k] = lo;
        diag[k] = 2.0 * (lo + hi);
        sup[k] = hi;
        rhs[k] = 6.0 * (slope(i) - slope(i - 1));
    }

    // Fold the eliminated m[0] into the first row (i=1).
    let r = h[0] / h[1];
    // m0 = (1+r) m1 - r m2 ; original first-row coeff on m0 is h[0].
    diag[0] += h[0] * (1.0 + r); // contribution to m1
    if nint >= 2 {
        sup[0] += -h[0] * r; // contribution to m2
    }
    // sub[0] (coeff on m0) is dropped — m0 is no longer an unknown.
    sub[0] = 0.0;

    // Fold the eliminated m[n-1] into the last row (i=n-2).
    let rl = h[n - 2] / h[n - 3];
    let last = nint - 1;
    diag[last] += h[n - 2] * (1.0 + rl); // contribution to m_{n-2}
    if nint >= 2 {
        sub[last] += -h[n - 2] * rl; // contribution to m_{n-3}
    }
    sup[last] = 0.0;

    let mut mint = vec![0.0; nint];
    thomas(&sub, &diag, &sup, &rhs, &mut mint);

    let mut m = vec![0.0; n];
    m[1..n - 1].copy_from_slice(&mint);
    m[0] = (1.0 + r) * m[1] - r * m[2];
    m[n - 1] = (1.0 + rl) * m[n - 2] - rl * m[n - 3];
    m
}

/// Thomas algorithm for a tridiagonal system. `sub[0]` and `sup[last]` are
/// ignored. Solves into `out`.
fn thomas(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64], out: &mut [f64]) {
    let n = diag.len();
    if n == 0 {
        return;
    }
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = sup[0] / diag[0];
    dp[0] = rhs[0] / diag[0];
    for i in 1..n {
        let denom = diag[i] - sub[i] * cp[i - 1];
        cp[i] = sup[i] / denom;
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / denom;
    }
    out[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = dp[i] - cp[i] * out[i + 1];
    }
}
