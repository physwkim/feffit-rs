//! Gamma function `Γ(x)`.
//!
//! Direct port of the Cephes `Gamma` routine (public domain) — the same code
//! `scipy.special.gamma` calls — so `gnxas`'s `Γ(q)` factor matches larch to
//! floating-point round-off. The coefficient tables are the Cephes constants
//! (`Cephes Math Library Release 2.2`, Stephen L. Moshier), each written as the
//! shortest decimal that round-trips to the same `f64` as the original literal
//! (verified bit-for-bit against `scipy.special.gamma`).
//!
//! Only `Gamma` is ported here (the `lgam`/log-gamma branch of the Cephes file
//! is unused by `gnxas`).

use std::f64::consts::PI;

// Rational-function coefficients for Γ on the interval (2, 3).
const GAMMA_P: [f64; 7] = [
    1.601_195_224_767_518_5E-4,
    1.191_351_470_065_863_8E-3,
    1.042_137_975_617_615_8E-2,
    4.763_678_004_571_372E-2,
    2.074_482_276_484_359_8E-1,
    4.942_148_268_014_971E-1,
    1.0,
];

const GAMMA_Q: [f64; 8] = [
    -2.315_818_733_241_201_4E-5,
    5.396_055_804_933_034E-4,
    -4.456_419_138_517_973E-3,
    1.181_397_852_220_604_3E-2,
    3.582_363_986_054_986_5E-2,
    -2.345_917_957_182_433_5E-1,
    7.143_049_170_302_73E-2,
    1.0,
];

// Stirling's-formula expansion coefficients (valid for 33 <= x <= 172).
const GAMMA_STIR: [f64; 5] = [
    7.873_113_957_930_937E-4,
    -2.295_499_616_133_781_3E-4,
    -2.681_326_178_057_812_4E-3,
    3.472_222_216_054_586_6E-3,
    8.333_333_333_334_822E-2,
];

const MAXGAM: f64 = 171.624_376_956_302_7;
const MAXSTIR: f64 = 143.01608;
/// `sqrt(2*pi)`.
const SQTPI: f64 = 2.506_628_274_631_000_7;
/// Euler–Mascheroni constant (used near `x == 0`).
const EULER: f64 = 0.5772156649015329;

/// Cephes `polevl`: evaluate a polynomial by Horner's rule. `coef[0]` is the
/// highest-degree coefficient; the polynomial has degree `coef.len() - 1`.
fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// Γ computed by Stirling's formula (Cephes `stirf`), valid for `33 <= x <= 172`.
fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let w = 1.0 / x;
    let w = 1.0 + w * polevl(w, &GAMMA_STIR);
    let mut y = x.exp();
    if x > MAXSTIR {
        // Avoid overflow in powf.
        let v = x.powf(0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = x.powf(x - 0.5) / y;
    }
    SQTPI * y * w
}

/// Gamma function `Γ(x)`, correctly signed (Cephes `Gamma`, parity with
/// `scipy.special.gamma`).
pub fn gamma(mut x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let q = x.abs();

    if q > 33.0 {
        let mut sgngam = 1.0;
        let z = if x < 0.0 {
            let mut p = q.floor();
            if p == q {
                // Pole at a negative integer.
                return f64::INFINITY;
            }
            let i = p as i64;
            if i & 1 == 0 {
                sgngam = -1.0;
            }
            let mut z = q - p;
            if z > 0.5 {
                p += 1.0;
                z = q - p;
            }
            z = q * (PI * z).sin();
            if z == 0.0 {
                return sgngam * f64::INFINITY;
            }
            z = z.abs();
            PI / (z * stirf(q))
        } else {
            stirf(x)
        };
        return sgngam * z;
    }

    let mut z = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    while x < 0.0 {
        if x > -1.0E-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    while x < 2.0 {
        if x < 1.0E-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }

    if x == 2.0 {
        return z;
    }

    x -= 2.0;
    let p = polevl(x, &GAMMA_P);
    let q = polevl(x, &GAMMA_Q);
    z * p / q
}

/// The Cephes `small:` branch (argument very close to 0).
fn small(x: f64, z: f64) -> f64 {
    if x == 0.0 {
        f64::INFINITY
    } else {
        z / ((1.0 + EULER * x) * x)
    }
}

#[cfg(test)]
mod tests {
    use super::gamma;

    #[test]
    fn gamma_known_values() {
        // Γ(n) = (n-1)!
        assert!((gamma(1.0) - 1.0).abs() < 1e-14);
        assert!((gamma(2.0) - 1.0).abs() < 1e-14);
        assert!((gamma(3.0) - 2.0).abs() < 1e-14);
        assert!((gamma(4.0) - 6.0).abs() < 1e-13);
        assert!((gamma(5.0) - 24.0).abs() < 1e-12);
        // Γ(1/2) = sqrt(pi)
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-14);
    }
}
