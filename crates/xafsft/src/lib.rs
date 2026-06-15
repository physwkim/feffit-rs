//! `xafsft` — a pure-Rust port of `xraylarch`'s `larch/xafs/xafsft.py`.
//!
//! XAFS Fourier transforms: forward `xftf` (chi(k) -> chi(R)), reverse `xftr`
//! (chi(R) -> chi(q)), the `*_fast` inner transforms, `xftf_prep`, and the FT
//! windows (`ftwindow`). The Kaiser-Bessel window uses a Cephes `I0` port for
//! parity with `scipy.special.i0`; FFTs use `rustfft` (agreeing with larch's
//! `scipy.fftpack` to FFT round-off).

pub mod bessel;
pub mod transform;
pub mod window;

pub use bessel::i0;
pub use transform::{xftf, xftf_fast, xftf_prep, xftr, xftr_fast, XftfOut, XftrOut};
pub use window::{ftwindow, Window};
