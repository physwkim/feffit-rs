//! GNXAS amplitude helper `gnxas(r0, sigma, beta)`.
//!
//! A path-expression helper (registered like `sigma2_eins`/`sigma2_debye`) that
//! returns the GNXAS amplitude for an asymmetric pair distribution `g(r)`
//! parametrised by `(r0, sigma, beta)`, evaluated at the path's `reff`. Port of
//! larch's `gnxas` (`larch/xafs/sigma2_models.py`):
//!
//! ```text
//! q     = 4 / beta^2
//! alpha = q * (1 + (reff - r0) * beta / (2*sigma))   (= q + 2*(reff-r0)/(beta*sigma))
//! out   = max(0, 2 * exp(-alpha) * alpha^(q-1) / (sigma * |beta| * Γ(q)))
//! ```
//!
//! Verified against larch's **module-level** `gnxas(r0, sigma, beta, path)`,
//! which is the documented, working implementation. NOTE: larch's
//! asteval-injected copy (the form a feffit path expression would actually call)
//! is broken in current larch — its debug `print('> ', reff, ...)` references an
//! undefined name and raises, so a `gnxas(...)` path expression evaluates to
//! `None`. The two implementations are otherwise numerically identical (verified
//! bit-for-bit), so this port reproduces the documented-correct behaviour and
//! there is no stock-larch end-to-end feffit reference to match.

use crate::feffdat::gamma::gamma;

/// GNXAS amplitude for `(r0, sigma, beta)` at a path of radius `reff`.
pub fn gnxas(r0: f64, sigma: f64, beta: f64, reff: f64) -> f64 {
    // larch clamps a near-zero beta to avoid division by zero.
    let beta = if beta.abs() < 1.0e-15 { 1.0e-15 } else { beta };
    let q = 4.0 / (beta * beta);
    let x = (reff - r0) * beta / (2.0 * sigma);
    let alpha = q * (1.0 + x);
    // larch wraps the power in try/except, falling back to amp = 0.0; a
    // non-finite result (overflow, or a negative base raised to a non-integer
    // power) maps to 0.0 the same way.
    let amp = {
        let a = (-alpha).exp() * alpha.powf(q - 1.0);
        if a.is_finite() { a } else { 0.0 }
    };
    let out = 2.0 * amp / (sigma * beta.abs() * gamma(q));
    out.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::gnxas;

    #[test]
    fn gnxas_reference_point() {
        // Matches larch's module-level gnxas(2.5, 0.05, 0.3, path) with
        // path._feffdat.reff = 2.55 → 4.393525791371315 (verified in larch).
        let v = gnxas(2.5, 0.05, 0.3, 2.55);
        assert!((v - 4.393525791371315).abs() < 1e-12, "got {v}");
    }

    #[test]
    fn gnxas_nonnegative() {
        // The result is clamped at 0; a far-out r0 drives the amplitude to 0.
        assert!(gnxas(10.0, 0.05, 0.3, 2.55) >= 0.0);
    }
}
