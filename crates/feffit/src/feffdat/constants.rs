//! Physical constants and conversions, kept bit-identical to
//! `larch/xafs/xafsutils.py`.
//!
//! larch derives `KTOE = 1e20 * hbar^2 / (2 * m_e * e)` from `scipy.constants`,
//! which evaluates to the literal below, and then `ETOK = 1.0 / KTOE`. We
//! hardcode the same literal and perform the same single division so the
//! wavenumber <-> energy conversion matches larch to the last bit.

/// `k^2 * KTOE == energy(eV)`  (larch `xafsutils.KTOE`).
pub const KTOE: f64 = 3.8099819442818976;

/// `energy(eV) * ETOK == k^2`  (larch `xafsutils.ETOK = 1.0 / KTOE`).
pub const ETOK: f64 = 1.0 / KTOE;

/// Energy floor used in `_calc_chi` to step around the k=0 singularity
/// (`larch.xafs.feffdat.SMALL_ENERGY`).
pub const SMALL_ENERGY: f64 = 1.0e-6;

/// photo-electron wavenumber -> energy (eV).
#[inline]
pub fn ktoe(k: f64) -> f64 {
    k * k * KTOE
}

/// photo-electron energy (eV) -> wavenumber.
#[inline]
pub fn etok(energy: f64) -> f64 {
    (energy * ETOK).sqrt()
}
