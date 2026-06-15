//! A faithful port of MINPACK's `lmdif` (Levenberg-Marquardt least squares with
//! a forward-difference Jacobian) and its support routines `enorm`, `fdjac2`,
//! `qrfac`, `qrsolv`, `lmpar`, ported line-by-line from the modernized
//! `fortran-lang/minpack` (`src/minpack.f90`, MIT). This is the same algorithm
//! `scipy.optimize.leastsq` wraps, so results match scipy to rounding.
//!
//! Storage: matrices are column-major `Vec<Vec<f64>>` (`a[col][row]`), and all
//! indices are 0-based; the Fortran 1-based loops are translated accordingly.
//!
//! These routines are verbatim ports: the index arithmetic is load-bearing
//! (column-major `a[col][row]`, cross-indexed `a[j][i]*a[k][i]`, triangular
//! bounds tied to the outer loop, the transpose copy `r[j][i]=r[i][j]`) and the
//! signatures mirror MINPACK's, so `needless_range_loop` / `too_many_arguments`
//! are suppressed to keep the line-by-line correspondence with the source.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

/// machine precision, `dpmpar(1) = epsilon(1.0_wp)`.
const EPSMCH: f64 = f64::EPSILON;
/// smallest positive magnitude, `dpmpar(2) = tiny(1.0_wp)`.
const DWARF: f64 = f64::MIN_POSITIVE;

/// MINPACK `enorm`: Euclidean norm computed to avoid overflow/underflow.
pub fn enorm(x: &[f64]) -> f64 {
    const RDWARF: f64 = 3.834e-20;
    const RGIANT: f64 = 1.304e19;
    let n = x.len();
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut x1max = 0.0;
    let mut x3max = 0.0;
    let agiant = RGIANT / n as f64;
    for &xi in x {
        let xabs = xi.abs();
        if xabs > RDWARF && xabs < agiant {
            // intermediate components
            s2 += xabs * xabs;
        } else if xabs <= RDWARF {
            // small components
            if xabs <= x3max {
                if xabs != 0.0 {
                    s3 += (xabs / x3max).powi(2);
                }
            } else {
                s3 = 1.0 + s3 * (x3max / xabs).powi(2);
                x3max = xabs;
            }
        } else if xabs <= x1max {
            // large components
            s1 += (xabs / x1max).powi(2);
        } else {
            s1 = 1.0 + s1 * (x1max / xabs).powi(2);
            x1max = xabs;
        }
    }
    if s1 != 0.0 {
        x1max * (s1 + (s2 / x1max) / x1max).sqrt()
    } else if s2 == 0.0 {
        x3max * s3.sqrt()
    } else if s2 >= x3max {
        (s2 * (1.0 + (x3max / s2) * (x3max * s3))).sqrt()
    } else {
        (x3max * ((s2 / x3max) + (x3max * s3))).sqrt()
    }
}

/// MINPACK `fdjac2`: forward-difference Jacobian. `fjac[j]` (column `j`) receives
/// `(fcn(x + h e_j) - fvec) / h`. Returns the number of `fcn` calls made (`n`).
fn fdjac2<F>(
    fcn: &mut F,
    m: usize,
    n: usize,
    x: &mut [f64],
    fvec: &[f64],
    fjac: &mut [Vec<f64>],
    epsfcn: f64,
) -> i32
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let eps = epsfcn.max(EPSMCH).sqrt();
    for j in 0..n {
        let temp = x[j];
        let mut h = eps * temp.abs();
        if h == 0.0 {
            h = eps;
        }
        x[j] = temp + h;
        let wa = fcn(x);
        x[j] = temp;
        for i in 0..m {
            fjac[j][i] = (wa[i] - fvec[i]) / h;
        }
    }
    n as i32
}

/// MINPACK `qrfac` with column pivoting (`pivot = true`). Reduces `a` (m×n,
/// column-major) to R via Householder transforms in place; fills `ipvt`,
/// `rdiag` (R's diagonal), `acnorm` (input column norms).
fn qrfac(
    m: usize,
    n: usize,
    a: &mut [Vec<f64>],
    ipvt: &mut [usize],
    rdiag: &mut [f64],
    acnorm: &mut [f64],
    wa: &mut [f64],
) {
    const P05: f64 = 5.0e-2;
    for j in 0..n {
        acnorm[j] = enorm(&a[j][0..m]);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
        ipvt[j] = j;
    }
    let minmn = m.min(n);
    for j in 0..minmn {
        // bring the column of largest norm into the pivot position
        let mut kmax = j;
        for k in j..n {
            if rdiag[k] > rdiag[kmax] {
                kmax = k;
            }
        }
        if kmax != j {
            a.swap(j, kmax);
            rdiag[kmax] = rdiag[j];
            wa[kmax] = wa[j];
            ipvt.swap(j, kmax);
        }

        // Householder transform to reduce column j to a multiple of e_j
        let mut ajnorm = enorm(&a[j][j..m]);
        if ajnorm != 0.0 {
            if a[j][j] < 0.0 {
                ajnorm = -ajnorm;
            }
            for i in j..m {
                a[j][i] /= ajnorm;
            }
            a[j][j] += 1.0;

            // apply to the remaining columns and update the norms
            for k in (j + 1)..n {
                let mut sum = 0.0;
                for i in j..m {
                    sum += a[j][i] * a[k][i];
                }
                let temp = sum / a[j][j];
                for i in j..m {
                    a[k][i] -= temp * a[j][i];
                }
                if rdiag[k] != 0.0 {
                    let t = a[k][j] / rdiag[k];
                    rdiag[k] *= (0.0_f64).max(1.0 - t * t).sqrt();
                    if P05 * (rdiag[k] / wa[k]).powi(2) <= EPSMCH {
                        rdiag[k] = enorm(&a[k][(j + 1)..m]);
                        wa[k] = rdiag[k];
                    }
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
}

/// MINPACK `qrsolv`: solve `a*x = b, d*x = 0` in least squares given the QR
/// factorization. `r` holds R on input; its strict lower triangle is overwritten
/// with S (transposed). `sdiag` receives S's diagonal.
fn qrsolv(
    n: usize,
    r: &mut [Vec<f64>],
    ipvt: &[usize],
    diag: &[f64],
    qtb: &[f64],
    x: &mut [f64],
    sdiag: &mut [f64],
    wa: &mut [f64],
) {
    const P5: f64 = 0.5;
    const P25: f64 = 0.25;

    // copy R into the strict lower triangle, save R's diagonal in x, qtb in wa
    for j in 0..n {
        for i in j..n {
            r[j][i] = r[i][j];
        }
        x[j] = r[j][j];
        wa[j] = qtb[j];
    }

    // eliminate the diagonal matrix d with Givens rotations
    for j in 0..n {
        let l = ipvt[j];
        if diag[l] != 0.0 {
            for k in j..n {
                sdiag[k] = 0.0;
            }
            sdiag[j] = diag[l];

            let mut qtbpj = 0.0;
            for k in j..n {
                if sdiag[k] != 0.0 {
                    let (cos, sin);
                    if r[k][k].abs() >= sdiag[k].abs() {
                        let tan = sdiag[k] / r[k][k];
                        cos = P5 / (P25 + P25 * tan * tan).sqrt();
                        sin = cos * tan;
                    } else {
                        let cotan = r[k][k] / sdiag[k];
                        sin = P5 / (P25 + P25 * cotan * cotan).sqrt();
                        cos = sin * cotan;
                    }
                    r[k][k] = cos * r[k][k] + sin * sdiag[k];
                    let temp = cos * wa[k] + sin * qtbpj;
                    qtbpj = -sin * wa[k] + cos * qtbpj;
                    wa[k] = temp;
                    for i in (k + 1)..n {
                        let t = cos * r[k][i] + sin * sdiag[i];
                        sdiag[i] = -sin * r[k][i] + cos * sdiag[i];
                        r[k][i] = t;
                    }
                }
            }
        }
        sdiag[j] = r[j][j];
        r[j][j] = x[j];
    }

    // solve the triangular system for z; least-squares if singular
    let mut nsing = n;
    for j in 0..n {
        if sdiag[j] == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa[j] = 0.0;
        }
    }
    if nsing >= 1 {
        for k in 0..nsing {
            let j = nsing - 1 - k;
            let mut sum = 0.0;
            for i in (j + 1)..nsing {
                sum += r[j][i] * wa[i];
            }
            wa[j] = (wa[j] - sum) / sdiag[j];
        }
    }

    // permute z back to x
    for j in 0..n {
        x[ipvt[j]] = wa[j];
    }
}

/// MINPACK `lmpar`: determine the Levenberg-Marquardt parameter `par` and the
/// corresponding solution `x` of the trust-region subproblem.
fn lmpar(
    n: usize,
    r: &mut [Vec<f64>],
    ipvt: &[usize],
    diag: &[f64],
    qtb: &[f64],
    delta: f64,
    par: &mut f64,
    x: &mut [f64],
    sdiag: &mut [f64],
    wa1: &mut [f64],
    wa2: &mut [f64],
) {
    const P1: f64 = 0.1;
    const P001: f64 = 1.0e-3;

    // Gauss-Newton direction (least squares if rank deficient)
    let mut nsing = n;
    for j in 0..n {
        wa1[j] = qtb[j];
        if r[j][j] == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa1[j] = 0.0;
        }
    }
    if nsing >= 1 {
        for k in 0..nsing {
            let j = nsing - 1 - k;
            wa1[j] /= r[j][j];
            let temp = wa1[j];
            for i in 0..j {
                wa1[i] -= r[j][i] * temp;
            }
        }
    }
    for j in 0..n {
        x[ipvt[j]] = wa1[j];
    }

    // test the Gauss-Newton direction
    let mut iter = 0;
    for j in 0..n {
        wa2[j] = diag[j] * x[j];
    }
    let mut dxnorm = enorm(&wa2[0..n]);
    let mut fp = dxnorm - delta;
    if fp <= P1 * delta {
        if iter == 0 {
            *par = 0.0;
        }
        return;
    }

    // lower bound parl for the zero of the function
    let mut parl = 0.0;
    if nsing >= n {
        for j in 0..n {
            let l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..j {
                sum += r[j][i] * wa1[i];
            }
            wa1[j] = (wa1[j] - sum) / r[j][j];
        }
        let temp = enorm(&wa1[0..n]);
        parl = ((fp / delta) / temp) / temp;
    }

    // upper bound paru
    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..=j {
            sum += r[j][i] * qtb[i];
        }
        let l = ipvt[j];
        wa1[j] = sum / diag[l];
    }
    let gnorm = enorm(&wa1[0..n]);
    let mut paru = gnorm / delta;
    if paru == 0.0 {
        paru = DWARF / delta.min(P1);
    }

    // clamp the initial par to [parl, paru]
    *par = par.max(parl);
    *par = par.min(paru);
    if *par == 0.0 {
        *par = gnorm / dxnorm;
    }

    loop {
        iter += 1;

        if *par == 0.0 {
            *par = DWARF.max(P001 * paru);
        }
        let temp = par.sqrt();
        for j in 0..n {
            wa1[j] = temp * diag[j];
        }
        qrsolv(n, r, ipvt, wa1, qtb, x, sdiag, wa2);
        for j in 0..n {
            wa2[j] = diag[j] * x[j];
        }
        dxnorm = enorm(&wa2[0..n]);
        let temp_fp = fp;
        fp = dxnorm - delta;

        // accept par if fp is small, or the exceptional parl/iter cases.
        // NOTE Fortran `.and.` binds tighter than `.or.`:
        //   |fp|<=0.1*delta  OR  (parl==0 AND fp<=temp AND temp<0)  OR  iter==10
        if fp.abs() <= P1 * delta || (parl == 0.0 && fp <= temp_fp && temp_fp < 0.0) || iter == 10 {
            return;
        }

        // Newton correction
        for j in 0..n {
            let l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for j in 0..n {
            wa1[j] /= sdiag[j];
            let t = wa1[j];
            for i in (j + 1)..n {
                wa1[i] -= r[j][i] * t;
            }
        }
        let temp = enorm(&wa1[0..n]);
        let parc = ((fp / delta) / temp) / temp;

        if fp > 0.0 {
            parl = parl.max(*par);
        }
        if fp < 0.0 {
            paru = paru.min(*par);
        }
        *par = parl.max(*par + parc);
    }
}

/// Configuration for [`lmdif`] (mirrors the MINPACK/scipy controls).
#[derive(Debug, Clone)]
pub struct LmConfig {
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    /// Maximum `fcn` evaluations; if `<= 0`, defaults to `200*(n+1)`.
    pub maxfev: i32,
    pub epsfcn: f64,
    pub factor: f64,
}

impl Default for LmConfig {
    /// `scipy.optimize.leastsq` defaults.
    fn default() -> Self {
        LmConfig {
            ftol: 1.49012e-8,
            xtol: 1.49012e-8,
            gtol: 0.0,
            maxfev: 0,
            epsfcn: EPSMCH,
            factor: 100.0,
        }
    }
}

/// Result of an [`lmdif`] fit.
#[derive(Debug, Clone)]
pub struct LmResult {
    /// final parameter estimate.
    pub x: Vec<f64>,
    /// residuals at `x`.
    pub fvec: Vec<f64>,
    /// `enorm(fvec)`.
    pub fnorm: f64,
    /// number of `fcn` evaluations.
    pub nfev: i32,
    /// MINPACK `info` termination code (1–4 are success).
    pub info: i32,
    /// permutation: `ipvt[j]` is the original column in pivot position `j`.
    pub ipvt: Vec<usize>,
    /// R from the final QR (upper triangle of `fjac[col][row]`).
    pub fjac: Vec<Vec<f64>>,
}

impl LmResult {
    /// The (unscaled) covariance `(JᵀJ)⁻¹`, computed from the final R and the
    /// pivot exactly as `scipy.optimize.leastsq` does. `None` if R is singular.
    pub fn covar(&self) -> Option<Vec<Vec<f64>>> {
        let n = self.x.len();
        // dense upper-triangular R (n×n): R[i][j] = fjac[j][i] for i <= j
        let mut rmat = vec![vec![0.0; n]; n];
        for j in 0..n {
            for i in 0..=j {
                rmat[i][j] = self.fjac[j][i];
            }
            if rmat[j][j] == 0.0 {
                return None;
            }
        }
        // invert upper-triangular R by back-substitution -> rinv (upper-tri)
        let mut rinv = vec![vec![0.0; n]; n];
        for i in (0..n).rev() {
            rinv[i][i] = 1.0 / rmat[i][i];
            for j in (i + 1)..n {
                let mut s = 0.0;
                for k in (i + 1)..=j {
                    s += rmat[i][k] * rinv[k][j];
                }
                rinv[i][j] = -s / rmat[i][i];
            }
        }
        // M = rinv * rinvᵀ  = (RᵀR)⁻¹
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += rinv[i][k] * rinv[j][k];
                }
                m[i][j] = s;
            }
        }
        // permute back: cov[ipvt[i]][ipvt[j]] = M[i][j]
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[self.ipvt[i]][self.ipvt[j]] = m[i][j];
            }
        }
        Some(cov)
    }
}

/// Minimize the sum of squares of `m` residuals in `n` variables (MINPACK
/// `lmdif`). `fcn(x)` returns the length-`m` residual vector; `x0` is the
/// starting point. Returns the fit result (see [`LmResult`]).
pub fn lmdif<F>(mut fcn: F, x0: &[f64], cfg: &LmConfig) -> LmResult
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    const P1: f64 = 0.1;
    const P5: f64 = 0.5;
    const P25: f64 = 0.25;
    const P75: f64 = 0.75;
    const P0001: f64 = 1.0e-4;

    let n = x0.len();
    let mut x = x0.to_vec();
    let maxfev = if cfg.maxfev > 0 {
        cfg.maxfev
    } else {
        200 * (n as i32 + 1)
    };

    let mut fvec = fcn(&x);
    let m = fvec.len();
    let mut nfev = 1;

    let mut fjac = vec![vec![0.0; m]; n]; // n columns of length m
    let mut ipvt = vec![0usize; n];
    let mut qtf = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut wa1 = vec![0.0; n];
    let mut wa2 = vec![0.0; n];
    let mut wa3 = vec![0.0; n];
    let mut wa4 = vec![0.0; m];
    let mut wa5 = vec![0.0; n]; // lmpar's wa2 scratch (Fortran reuses wa4(1:n))

    let mut info = 0i32;
    let mut fnorm = enorm(&fvec);
    let mut par = 0.0;
    let mut iter = 1;
    let mut delta = 0.0;
    let mut xnorm = 0.0;

    'main: {
        // input checks (mode is always 1 here)
        if !(n > 0
            && m >= n
            && cfg.ftol >= 0.0
            && cfg.xtol >= 0.0
            && cfg.gtol >= 0.0
            && maxfev > 0
            && cfg.factor > 0.0)
        {
            break 'main;
        }

        loop {
            // Jacobian
            nfev += fdjac2(&mut fcn, m, n, &mut x, &fvec, &mut fjac, cfg.epsfcn);

            // QR factorization (rdiag -> wa1, acnorm -> wa2, wa -> wa3)
            qrfac(m, n, &mut fjac, &mut ipvt, &mut wa1, &mut wa2, &mut wa3);

            if iter == 1 {
                for j in 0..n {
                    diag[j] = wa2[j];
                    if wa2[j] == 0.0 {
                        diag[j] = 1.0;
                    }
                }
                for j in 0..n {
                    wa3[j] = diag[j] * x[j];
                }
                xnorm = enorm(&wa3[0..n]);
                delta = cfg.factor * xnorm;
                if delta == 0.0 {
                    delta = cfg.factor;
                }
            }

            // form Qᵀ*fvec, store first n components in qtf
            wa4.copy_from_slice(&fvec);
            for j in 0..n {
                if fjac[j][j] != 0.0 {
                    let mut sum = 0.0;
                    for i in j..m {
                        sum += fjac[j][i] * wa4[i];
                    }
                    let temp = -sum / fjac[j][j];
                    for i in j..m {
                        wa4[i] += fjac[j][i] * temp;
                    }
                }
                fjac[j][j] = wa1[j];
                qtf[j] = wa4[j];
            }

            // norm of the scaled gradient
            let mut gnorm: f64 = 0.0;
            if fnorm != 0.0 {
                for j in 0..n {
                    let l = ipvt[j];
                    if wa2[l] != 0.0 {
                        let mut sum = 0.0;
                        for i in 0..=j {
                            sum += fjac[j][i] * (qtf[i] / fnorm);
                        }
                        gnorm = gnorm.max((sum / wa2[l]).abs());
                    }
                }
            }
            if gnorm <= cfg.gtol {
                info = 4;
            }
            if info != 0 {
                break 'main;
            }

            // rescale
            for j in 0..n {
                diag[j] = diag[j].max(wa2[j]);
            }

            'inner: loop {
                // Levenberg-Marquardt parameter; the step lands in wa1, sdiag in
                // wa2, and wa3/wa5 are pure scratch (Fortran: wa1..wa4 of lmdif).
                lmpar(
                    n, &mut fjac, &ipvt, &diag, &qtf, delta, &mut par, &mut wa1, &mut wa2,
                    &mut wa3, &mut wa5,
                );

                // direction p = -wa1, and x + p
                for j in 0..n {
                    wa1[j] = -wa1[j];
                    wa2[j] = x[j] + wa1[j];
                    wa3[j] = diag[j] * wa1[j];
                }
                let pnorm = enorm(&wa3[0..n]);

                if iter == 1 {
                    delta = delta.min(pnorm);
                }

                // evaluate at x + p
                let fwa = fcn(&wa2[0..n]);
                wa4.copy_from_slice(&fwa);
                nfev += 1;
                let fnorm1 = enorm(&wa4);

                // scaled actual reduction
                let mut actred = -1.0;
                if P1 * fnorm1 < fnorm {
                    actred = 1.0 - (fnorm1 / fnorm).powi(2);
                }

                // scaled predicted reduction and directional derivative
                for j in 0..n {
                    wa3[j] = 0.0;
                    let l = ipvt[j];
                    let temp = wa1[l];
                    for i in 0..=j {
                        wa3[i] += fjac[j][i] * temp;
                    }
                }
                let temp1 = enorm(&wa3[0..n]) / fnorm;
                let temp2 = (par.sqrt() * pnorm) / fnorm;
                let prered = temp1 * temp1 + temp2 * temp2 / P5;
                let dirder = -(temp1 * temp1 + temp2 * temp2);

                // ratio of actual to predicted reduction
                let mut ratio = 0.0;
                if prered != 0.0 {
                    ratio = actred / prered;
                }

                // update the step bound
                if ratio <= P25 {
                    let mut temp = if actred >= 0.0 {
                        P5
                    } else {
                        P5 * dirder / (dirder + P5 * actred)
                    };
                    if P1 * fnorm1 >= fnorm || temp < P1 {
                        temp = P1;
                    }
                    delta = temp * delta.min(pnorm / P1);
                    par /= temp;
                } else if par == 0.0 || ratio >= P75 {
                    delta = pnorm / P5;
                    par *= P5;
                }

                // successful iteration: update x, fvec, norms
                if ratio >= P0001 {
                    for j in 0..n {
                        x[j] = wa2[j];
                        wa2[j] = diag[j] * x[j];
                    }
                    fvec.copy_from_slice(&wa4);
                    xnorm = enorm(&wa2[0..n]);
                    fnorm = fnorm1;
                    iter += 1;
                }

                // convergence tests
                if actred.abs() <= cfg.ftol && prered <= cfg.ftol && P5 * ratio <= 1.0 {
                    info = 1;
                }
                if delta <= cfg.xtol * xnorm {
                    info = 2;
                }
                if actred.abs() <= cfg.ftol && prered <= cfg.ftol && P5 * ratio <= 1.0 && info == 2
                {
                    info = 3;
                }
                if info != 0 {
                    break 'main;
                }

                // termination and stringent-tolerance tests
                if nfev >= maxfev {
                    info = 5;
                }
                if actred.abs() <= EPSMCH && prered <= EPSMCH && P5 * ratio <= 1.0 {
                    info = 6;
                }
                if delta <= EPSMCH * xnorm {
                    info = 7;
                }
                if gnorm <= EPSMCH {
                    info = 8;
                }
                if info != 0 {
                    break 'main;
                }

                if ratio >= P0001 {
                    break 'inner;
                }
            }
            // exited inner on a successful step -> recompute the Jacobian
        }
    }

    fnorm = enorm(&fvec);
    LmResult {
        x,
        fvec,
        fnorm,
        nfev,
        info,
        ipvt,
        fjac,
    }
}
