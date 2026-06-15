//! Numerical primitives shared across the XAS preprocessing ports, matching
//! `larch.math.utils` (and the bits of numpy/scipy it leans on) closely enough
//! to reproduce larch's results to floating-point round-off.
//!
//! Ported here: [`index_of`], [`index_nearest`], [`gradient`] (numpy
//! `np.gradient`, unit spacing), [`remove_dups`], [`remove_nans2`],
//! [`find_energy_step`], [`polyfit`] (numpy `Polynomial.fit` ordering),
//! [`interp_linear`] (numpy `np.interp`, edge-clamped), and [`smooth`]
//! (Lorentzian/Gaussian convolution smoothing).

/// `larch.xafs.xafsutils.KTOE` — `1e20 * hbar^2 / (2 m_e e)`, taken at the
/// exact live `scipy.constants` value (the larch source comment
/// `3.8099819442818976` is stale from an older CODATA set). Single source of
/// truth for the eV↔k conversion across the XAS ports.
pub const KTOE: f64 = 3.809_982_116_154_859_7;
/// `ETOK = 1/KTOE`, the eV→k^2 conversion factor.
pub const ETOK: f64 = 1.0 / KTOE;

/// `larch.xafs.xafsutils.etok`: photo-electron energy (eV) → wavenumber
/// (`sqrt(E·ETOK)`); returns 0 for negative energy, matching larch's guard.
pub fn etok(energy: f64) -> f64 {
    if energy < 0.0 {
        0.0
    } else {
        (energy * ETOK).sqrt()
    }
}

/// `larch.xafs.xafsutils.ktoe`: wavenumber → photo-electron energy (`k²·KTOE`).
pub fn ktoe(k: f64) -> f64 {
    k * k * KTOE
}

/// `larch.math.index_of`: index of `array` *at or below* `value`; `0` if
/// `value < min(array)`. Works on unsorted arrays (max index satisfying `<=`).
pub fn index_of(array: &[f64], value: f64) -> usize {
    if array.is_empty() {
        return 0;
    }
    let amin = array.iter().cloned().fold(f64::INFINITY, f64::min);
    if value < amin {
        return 0;
    }
    let mut idx = 0;
    for (i, &a) in array.iter().enumerate() {
        if a <= value {
            idx = i;
        }
    }
    idx
}

/// `larch.math.index_nearest`: index of `array` nearest `value` (first on ties,
/// matching `np.abs(array-value).argmin()`).
pub fn index_nearest(array: &[f64], value: f64) -> usize {
    let mut best = 0;
    let mut bestd = f64::INFINITY;
    for (i, &a) in array.iter().enumerate() {
        let d = (a - value).abs();
        if d < bestd {
            bestd = d;
            best = i;
        }
    }
    best
}

/// `np.gradient(f)` with unit spacing and `edge_order=1`: central differences
/// in the interior, one-sided at the two ends.
pub fn gradient(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut g = vec![0.0; n];
    if n < 2 {
        return g;
    }
    g[0] = f[1] - f[0];
    g[n - 1] = f[n - 1] - f[n - 2];
    for i in 1..n - 1 {
        g[i] = (f[i + 1] - f[i - 1]) / 2.0;
    }
    g
}

/// `d(mu)/d(energy)` the way larch computes it: `np.gradient(mu)/np.gradient(energy)`
/// (the unit-spacing index gradients divide out, leaving the chain-rule derivative).
pub fn dmude(mu: &[f64], energy: &[f64]) -> Vec<f64> {
    let gm = gradient(mu);
    let ge = gradient(energy);
    gm.iter().zip(&ge).map(|(a, b)| a / b).collect()
}

/// `larch.math.remove_dups`: nudge repeated/too-close successive values of a
/// (nominally increasing) array up by `tiny` so it is strictly increasing.
/// Returns the input unchanged when the minimum step already exceeds `10*tiny`.
pub fn remove_dups(arr: &[f64], tiny: f64) -> Vec<f64> {
    let n = arr.len();
    if n <= 1 {
        return arr.to_vec();
    }
    let min_step = (1..n)
        .map(|i| arr[i] - arr[i - 1])
        .fold(f64::INFINITY, f64::min);
    if min_step > 10.0 * tiny {
        return arr.to_vec();
    }
    let mut add = vec![0.0; n];
    let mut previous_val = f64::NAN;
    let mut previous_add = 0.0;
    for i in 1..n {
        if !arr[i - 1].is_nan() {
            previous_val = arr[i - 1];
            previous_add = add[i - 1];
        }
        let val = arr[i];
        if val.is_nan() || previous_val.is_nan() {
            continue;
        }
        if (val - previous_val).abs() < tiny {
            add[i] = previous_add + tiny;
        }
    }
    arr.iter().zip(&add).map(|(a, d)| a + d).collect()
}

/// `larch.math.remove_nans2`: drop every index where either `a` or `b` is
/// non-finite, returning the two filtered arrays.
pub fn remove_nans2(a: &[f64], b: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut ao = Vec::with_capacity(a.len());
    let mut bo = Vec::with_capacity(b.len());
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            ao.push(x);
            bo.push(y);
        }
    }
    (ao, bo)
}

/// `argsort` (ascending, stable). For strictly-increasing input this is the
/// identity, matching numpy's `argsort` on distinct values.
fn argsort(a: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// `larch.xafs.pre_edge.find_energy_step`: robust energy step, ignoring the
/// smallest `frac_ignore` fraction of steps and averaging the next `nave`.
pub fn find_energy_step(energy: &[f64], frac_ignore: f64, nave: usize) -> f64 {
    let nskip = (frac_ignore * energy.len() as f64) as usize;
    // positions where the argsort order advances by exactly 1 (locally in order)
    let order = argsort(energy);
    let e_ordered: Vec<usize> = (0..order.len().saturating_sub(1))
        .filter(|&i| order[i + 1] as i64 - order[i] as i64 == 1)
        .collect();
    let evals: Vec<f64> = e_ordered.iter().map(|&i| energy[i]).collect();
    // evals[nskip : len-nskip]
    let lo = nskip;
    let hi = evals.len().saturating_sub(nskip);
    let trimmed = &evals[lo..hi];
    let mut ediff: Vec<f64> = (1..trimmed.len())
        .map(|i| trimmed[i] - trimmed[i - 1])
        .collect();
    // mean of ediff sorted-ascending over [nskip : nskip+nave]
    ediff.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let s = nskip.min(ediff.len());
    let e = (nskip + nave).min(ediff.len());
    let win = &ediff[s..e];
    win.iter().sum::<f64>() / win.len() as f64
}

/// Solve a small dense linear system `A x = b` in place by Gaussian elimination
/// with partial pivoting (`A` is `n x n` row-major). Used by [`polyfit`] and the
/// Savitzky–Golay normal equations.
pub(crate) fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        // partial pivot
        let mut piv = col;
        let mut best = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > best {
                best = row[col].abs();
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let pivot_row = a[col].clone();
        let pivot_b = b[col];
        let d = pivot_row[col];
        for (row, bv) in a.iter_mut().zip(b.iter_mut()).skip(col + 1) {
            let factor = row[col] / d;
            if factor != 0.0 {
                for (c, pv) in pivot_row.iter().enumerate().skip(col) {
                    row[c] -= factor * pv;
                }
                *bv -= factor * pivot_b;
            }
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let dot: f64 = a[col][col + 1..]
            .iter()
            .zip(&x[col + 1..])
            .map(|(aij, xj)| aij * xj)
            .sum();
        x[col] = (b[col] - dot) / a[col][col];
    }
    x
}

/// `larch.math.polyfit`: least-squares polynomial fit returning monomial
/// coefficients **lowest-degree first** (matching `numpy.polynomial.Polynomial.fit`
/// after `.convert()`). The fit is done on the `[-1, 1]`-mapped domain (numpy's
/// default) for conditioning, then converted back to the unscaled basis.
pub fn polyfit(x: &[f64], y: &[f64], deg: usize) -> Vec<f64> {
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // map x -> u in [-1, 1]:  u = scl_a * x + scl_b
    let (scl_a, scl_b) = if xmax > xmin {
        (2.0 / (xmax - xmin), -(xmax + xmin) / (xmax - xmin))
    } else {
        (1.0, 0.0)
    };
    let nc = deg + 1;
    // Vandermonde in u, then normal equations (V^T V) c = V^T y.
    let us: Vec<f64> = x.iter().map(|&xi| scl_a * xi + scl_b).collect();
    let mut vtv = vec![vec![0.0; nc]; nc];
    let mut vty = vec![0.0; nc];
    for (k, &u) in us.iter().enumerate() {
        let mut pows = vec![1.0; nc];
        for j in 1..nc {
            pows[j] = pows[j - 1] * u;
        }
        for i in 0..nc {
            vty[i] += pows[i] * y[k];
            for j in 0..nc {
                vtv[i][j] += pows[i] * pows[j];
            }
        }
    }
    let cu = solve_linear(vtv, vty); // coefficients in u, low->high
                                     // convert poly in u = a*x + b back to poly in x:  sum_j cu[j] (a x + b)^j
    let mut cx = vec![0.0; nc];
    for (j, &cj) in cu.iter().enumerate() {
        // expand (a x + b)^j via binomial: sum_{m=0..j} C(j,m) a^m b^(j-m) x^m
        let mut comb = 1.0; // C(j, 0)
        for (m, cxm) in cx.iter_mut().take(j + 1).enumerate() {
            let term = comb * scl_a.powi(m as i32) * scl_b.powi((j - m) as i32);
            *cxm += cj * term;
            // C(j, m+1) = C(j, m) * (j - m) / (m + 1)
            comb = comb * (j - m) as f64 / (m + 1) as f64;
        }
    }
    cx
}

/// `np.interp(xnew, x, y)`: piecewise-linear interpolation with the endpoints
/// held constant outside `[x[0], x[-1]]` (no extrapolation). `x` must be
/// increasing.
pub fn interp_linear(xnew: &[f64], x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    xnew.iter()
        .map(|&xv| {
            if xv <= x[0] {
                return y[0];
            }
            if xv >= x[n - 1] {
                return y[n - 1];
            }
            // binary search for the interval
            let mut lo = 0usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if x[mid] <= xv {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let t = (xv - x[lo]) / (x[lo + 1] - x[lo]);
            y[lo] + t * (y[lo + 1] - y[lo])
        })
        .collect()
}

/// Full discrete convolution `a * v` (numpy `np.convolve(a, v)`), length
/// `a.len() + v.len() - 1`.
pub fn convolve_full(a: &[f64], v: &[f64]) -> Vec<f64> {
    let n = a.len() + v.len() - 1;
    let mut out = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            out[i + j] += ai * vj;
        }
    }
    out
}

/// `np.convolve(a, v, mode='valid')`: the central part where the windows fully
/// overlap, length `max(M,N) - min(M,N) + 1`.
pub fn convolve_valid(a: &[f64], v: &[f64]) -> Vec<f64> {
    let full = convolve_full(a, v);
    let (m, n) = (a.len(), v.len());
    let minlen = m.min(n);
    let maxlen = m.max(n);
    full[(minlen - 1)..maxlen].to_vec()
}

/// Convolution-window form for [`smooth`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothForm {
    Lorentzian,
    Gaussian,
}

/// `larch.math.smooth`: smooth `y(x)` by convolving with a Lorentzian or
/// Gaussian of width `sigma` (in x-units), on a uniform grid of step `xstep`
/// with `npad` padding points, then interpolate back onto the input `x`.
/// The window is sum-normalized, so the line-shape prefactors cancel.
pub fn smooth(
    x: &[f64],
    y: &[f64],
    sigma: f64,
    xstep: f64,
    npad: usize,
    form: SmoothForm,
) -> Vec<f64> {
    let xmin0 = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax0 = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // xmin/xmax snapped to the xstep grid (Python int() truncates toward zero)
    let xmin = xstep * ((xmin0 - npad as f64 * xstep) / xstep).trunc();
    let xmax = xstep * ((xmax0 + npad as f64 * xstep) / xstep).trunc();
    let npts1 = 1 + ((xmax - xmin + xstep * 0.1).abs() / xstep) as usize;
    let npts = npts1.min(50 * x.len());
    // x0 = linspace(xmin, xmax, npts)
    let x0: Vec<f64> = (0..npts)
        .map(|i| {
            if npts == 1 {
                xmin
            } else {
                xmin + (xmax - xmin) * i as f64 / (npts - 1) as f64
            }
        })
        .collect();
    let y0 = interp_linear(&x0, x, y);

    let sig = sigma / xstep;
    // window over wx = 0..2*npts, centered at npts
    let nwin = 2 * npts;
    let win: Vec<f64> = (0..nwin)
        .map(|i| {
            let d = i as f64 - npts as f64;
            match form {
                SmoothForm::Lorentzian => sig / (d * d + sig * sig),
                SmoothForm::Gaussian => (-(d * d) / (2.0 * sig * sig)).exp(),
            }
        })
        .collect();
    let wsum: f64 = win.iter().sum();
    let winn: Vec<f64> = win.iter().map(|w| w / wsum).collect();

    // y1 = concat(y0[npts:0:-1], y0, y0[-1:-npts-1:-1])
    let mut y1 = Vec::with_capacity(3 * npts);
    for i in (1..npts).rev() {
        y1.push(y0[i]); // y0[npts-1 .. 1]
    }
    y1.extend_from_slice(&y0); // y0[0 .. npts-1]
    for i in (0..npts).rev() {
        y1.push(y0[i]); // y0[npts-1 .. 0]
    }

    let mut y2 = convolve_valid(&winn, &y1);
    if y2.len() > x0.len() {
        let nex = (y2.len() - x0.len()) / 2;
        y2 = y2[nex..nex + x0.len()].to_vec();
    }
    // interp back to x (in-range -> linear)
    interp_linear(x, &x0, &y2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_unit_spacing() {
        let f = [1.0, 4.0, 9.0, 16.0]; // x^2 at 1,2,3,4
        let g = gradient(&f);
        assert_eq!(g[0], 3.0); // 4-1
        assert_eq!(g[1], 4.0); // (9-1)/2
        assert_eq!(g[2], 6.0); // (16-4)/2
        assert_eq!(g[3], 7.0); // 16-9
    }

    #[test]
    fn polyfit_recovers_line() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
        let c = polyfit(&x, &y, 1);
        assert!((c[0] - 2.0).abs() < 1e-10, "c0={}", c[0]);
        assert!((c[1] - 3.0).abs() < 1e-10, "c1={}", c[1]);
    }

    #[test]
    fn polyfit_recovers_quadratic_shifted() {
        // large x to exercise the domain mapping
        let x: Vec<f64> = (0..20).map(|i| 8000.0 + 5.0 * i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 1.5 - 2e-3 * xi + 4e-7 * xi * xi)
            .collect();
        let c = polyfit(&x, &y, 2);
        assert!((c[0] - 1.5).abs() < 1e-6, "c0={}", c[0]);
        assert!((c[1] + 2e-3).abs() < 1e-9, "c1={}", c[1]);
        assert!((c[2] - 4e-7).abs() < 1e-13, "c2={}", c[2]);
    }

    #[test]
    fn interp_linear_clamps_edges() {
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 10.0, 20.0];
        assert_eq!(interp_linear(&[-1.0], &x, &y)[0], 0.0);
        assert_eq!(interp_linear(&[3.0], &x, &y)[0], 20.0);
        assert_eq!(interp_linear(&[0.5], &x, &y)[0], 5.0);
    }

    #[test]
    fn remove_dups_nudges_repeats() {
        let x = [1.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        let out = remove_dups(&x, 1e-6);
        assert_eq!(out[2], 3.0);
        assert!((out[3] - 3.000001).abs() < 1e-12);
        assert!((out[4] - 3.000002).abs() < 1e-12);
        assert_eq!(out[5], 4.0);
    }
}
