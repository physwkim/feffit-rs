//! Refined-background cubic B-spline, the FITPACK pieces larch's `refine_bkg`
//! relies on.
//!
//! larch refines a background as a cubic B-spline: a knot vector built once from
//! `splrep(linspace(kmin, kmax, nspline), …, k=3)`, then evaluated every
//! iteration with `splev(model.k, [knots, coefs, 3])` where `coefs` are the
//! `bkg00..bkgNN` fit variables. larch keeps only the *knots* from `splrep`
//! (the coefficients are the fit variables), and that knot vector has a closed
//! form for the evenly-spaced abscissae larch uses — so the only numeric port
//! needed is `splev` (FITPACK `splev.f` + `fpbspl.f`), the de Boor B-spline
//! evaluation, reproduced here exactly (including the boundary-polynomial
//! extrapolation FITPACK uses for `k < kmin`).

/// Number of background spline coefficients (larch `prepare_fit`):
/// `1 + round(2*rbkg*(kmax-kmin)/π)`. `round` is half-to-even (numpy/Python 3).
pub fn nspline(rbkg: f64, kmin: f64, kmax: f64) -> usize {
    let x = 2.0 * rbkg * (kmax - kmin) / std::f64::consts::PI;
    1 + round_half_even(x) as usize
}

/// Round half to even (`numpy.round`/Python 3 `round`) for `x >= 0`.
fn round_half_even(x: f64) -> f64 {
    let lower = x.floor();
    let frac = x - lower;
    if frac < 0.5 {
        lower
    } else if frac > 0.5 {
        lower + 1.0
    } else if (lower as i64) % 2 == 0 {
        lower
    } else {
        lower + 1.0
    }
}

/// The cubic (k=3) B-spline knot vector larch's `refine_bkg` uses, in closed
/// form: `splrep(linspace(kmin, kmax, nspline), …, k=3)` places `nspline+4`
/// knots — the endpoints repeated `k+1=4` times plus the interior abscissae
/// `linspace(kmin, kmax, nspline)[2 : nspline-2]`. Verified bit-exact against
/// `scipy.interpolate.splrep` for `nspline` in 5..40.
pub fn bkg_knots(kmin: f64, kmax: f64, nspline: usize) -> Vec<f64> {
    // numpy.linspace evaluation order: precompute step = (stop-start)/(num-1),
    // then start + i*step (NOT (stop-start)*i/(num-1)), so the interior knots
    // match scipy.splrep's knots to the last bit even when step is inexact.
    let step = (kmax - kmin) / (nspline - 1) as f64;
    let kk: Vec<f64> = (0..nspline).map(|i| kmin + i as f64 * step).collect();
    let mut t = Vec::with_capacity(nspline + 4);
    for _ in 0..4 {
        t.push(kmin);
    }
    // interior abscissae kk[2 .. nspline-2]
    t.extend_from_slice(&kk[2..nspline - 2]);
    for _ in 0..4 {
        t.push(kmax);
    }
    t
}

/// The `k+1` non-zero B-splines of degree `k` at `x` for knot interval `l`
/// (1-based, `t(l) <= x < t(l+1)`), written into `h[0..=k]`. Port of FITPACK
/// `fpbspl.f`; the recurrence is polynomial, so it extrapolates correctly when
/// `x` lies outside the interval.
fn fpbspl(t: &[f64], k: usize, x: f64, l: usize, h: &mut [f64]) {
    // 1-based knot access: tt(i) == t[i-1].
    let tt = |i: usize| t[i - 1];
    let mut hh = [0.0f64; 19]; // k is small (3); generous fixed buffer
    h[0] = 1.0;
    for j in 1..=k {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for i in 1..=j {
            let li = l + i;
            let lj = li - j;
            if tt(li) == tt(lj) {
                h[i] = 0.0;
            } else {
                let f = hh[i - 1] / (tt(li) - tt(lj));
                h[i - 1] += f * (tt(li) - x);
                h[i] = f * (x - tt(lj));
            }
        }
    }
}

/// Evaluate the spline `(t, c, k)` at each `x` (FITPACK `splev.f`, `ext=0`:
/// extrapolate outside `[t[k], t[n-k-1]]` via the boundary polynomial). `c` is
/// the coefficient vector (length `>= n-k-1`); only the first `n-k-1` entries
/// are used, matching larch passing the `nspline` background coefficients.
pub fn splev(t: &[f64], c: &[f64], k: usize, x: &[f64]) -> Vec<f64> {
    let n = t.len();
    let k1 = k + 1;
    let nk1 = n - k1; // last valid 1-based interval index
    let tt = |i: usize| t[i - 1];
    let cc = |i: usize| c[i - 1];

    let mut h = [0.0f64; 20];
    let mut out = Vec::with_capacity(x.len());
    for &arg in x {
        // find the knot interval l in [k1, nk1] (FITPACK's forward search;
        // a fresh per-point search gives the same l for ascending x and is
        // also correct for unordered x).
        let mut l = k1;
        while l < nk1 && arg >= tt(l + 1) {
            l += 1;
        }
        fpbspl(t, k, arg, l, &mut h);
        // sp = Σ_{j=1..k1} c(l-k1+j) * h(j)
        let mut sp = 0.0;
        let mut ll = l - k1;
        for j in 1..=k1 {
            ll += 1;
            sp += cc(ll) * h[j - 1];
        }
        out.push(sp);
    }
    out
}
