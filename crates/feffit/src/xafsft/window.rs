//! Fourier-transform windows, a port of `larch.xafs.xafsft.ftwindow`.

use std::f64::consts::PI;

use crate::xafsft::bessel::i0;

/// FT window type. Parsed from a name by its first three lowercased letters,
/// matching larch's `FT_WINDOWS_SHORT` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// `hanning` — cosine-squared taper (also the `fha` flat-hanning variant).
    Hanning,
    /// flat-top hanning (`fha`).
    Fha,
    /// `parzen` — linear taper.
    Parzen,
    /// `welch` — quadratic taper.
    Welch,
    /// `kaiser` — Kaiser-Bessel (the "better" `(I0-1)/(I0(dx)-1)` form).
    Kaiser,
    /// `bes` — ifeffit-1.0 Kaiser-Bessel form.
    Bes,
    /// `sine` — sine taper.
    Sine,
    /// `gaussian` — normal-function window.
    Gaussian,
}

impl Window {
    /// Parse a window name (uses the first three lowercased characters).
    pub fn from_name(name: &str) -> Option<Window> {
        let nam: String = name.trim().to_lowercase().chars().take(3).collect();
        Some(match nam.as_str() {
            "han" => Window::Hanning,
            "fha" => Window::Fha,
            "par" => Window::Parzen,
            "wel" => Window::Welch,
            "kai" => Window::Kaiser,
            "bes" => Window::Bes,
            "sin" => Window::Sine,
            "gau" => Window::Gaussian,
            _ => return None,
        })
    }
}

/// Build a Fourier-transform window over `x` (port of `ftwindow`).
///
/// `xmin`/`xmax` default to `min(x)`/`max(x)` when `None`; `dx2` defaults to `dx`.
pub fn ftwindow(
    x: &[f64],
    xmin: Option<f64>,
    xmax: Option<f64>,
    dx: f64,
    dx2: Option<f64>,
    window: Window,
) -> Vec<f64> {
    let n = x.len();
    let mut fwin = vec![0.0; n];
    if n == 0 {
        return fwin;
    }

    let mut dx1 = dx;
    let mut dx2 = dx2.unwrap_or(dx);
    let xmin = xmin.unwrap_or_else(|| min_of(x));
    let xmax = xmax.unwrap_or_else(|| max_of(x));

    let mut xstep = (x[n - 1] - x[0]) / (n as f64 - 1.0);
    if xstep < 0.0 || xstep.is_nan() {
        xstep = 1.0e-3;
    }
    let xeps = 1.0e-4 * xstep;

    let mut x1 = min_of(x).max(xmin - dx1 / 2.0);
    let mut x2 = xmin + dx1 / 2.0 + xeps;
    let mut x3 = xmax - dx2 / 2.0 - xeps;
    let mut x4 = max_of(x).min(xmax + dx2 / 2.0);

    match window {
        Window::Fha => {
            if dx1 < 0.0 {
                dx1 = 0.0;
            }
            if dx2 > 1.0 {
                dx2 = 1.0;
            }
            x2 = x1 + xeps + dx1 * (xmax - xmin) / 2.0;
            x3 = x4 - xeps - dx2 * (xmax - xmin) / 2.0;
        }
        Window::Gaussian => {
            dx1 = dx1.max(xeps);
        }
        _ => {}
    }

    let asint = |val: f64| -> i64 { ((val + xeps) / xstep) as i64 };
    let (mut i1, mut i2, mut i3, mut i4) = (asint(x1), asint(x2), asint(x3), asint(x4));
    i1 = i1.max(0);
    i2 = i2.max(0);
    i3 = i3.min(n as i64 - 1);
    i4 = i4.min(n as i64 - 1);
    if i2 == i1 {
        i1 = (i2 - 1).max(0);
    }
    if i4 == i3 {
        i3 = (i4 - 1).max(i2);
    }
    let (i1, i2, i3, i4) = (i1 as usize, i2 as usize, i3 as usize, i4 as usize);
    x1 = x[i1];
    x2 = x[i2];
    x3 = x[i3];
    x4 = x[i4];
    if x1 == x2 {
        x2 += xeps;
    }
    if x3 == x4 {
        x4 += xeps;
    }

    // initial flat top
    if i3 > i2 {
        for v in fwin.iter_mut().take(i3).skip(i2) {
            *v = 1.0;
        }
    }

    match window {
        Window::Hanning | Window::Fha => {
            for i in i1..=i2.min(n - 1) {
                fwin[i] = (((PI / 2.0) * (x[i] - x1) / (x2 - x1)).sin()).powi(2);
            }
            for i in i3..=i4.min(n - 1) {
                fwin[i] = (((PI / 2.0) * (x[i] - x3) / (x4 - x3)).cos()).powi(2);
            }
        }
        Window::Parzen => {
            for i in i1..=i2.min(n - 1) {
                fwin[i] = (x[i] - x1) / (x2 - x1);
            }
            for i in i3..=i4.min(n - 1) {
                fwin[i] = 1.0 - (x[i] - x3) / (x4 - x3);
            }
        }
        Window::Welch => {
            for i in i1..=i2.min(n - 1) {
                fwin[i] = 1.0 - ((x[i] - x2) / (x2 - x1)).powi(2);
            }
            for i in i3..=i4.min(n - 1) {
                fwin[i] = 1.0 - ((x[i] - x3) / (x4 - x3)).powi(2);
            }
        }
        Window::Kaiser | Window::Bes => {
            let cen = (x4 + x1) / 2.0;
            let wid = (x4 - x1) / 2.0;
            let arg = |xi: f64| {
                let a = 1.0 - (xi - cen).powi(2) / (wid * wid);
                if a < 0.0 { 0.0 } else { a }
            };
            if window == Window::Bes {
                let denom = i0(dx);
                for (i, &xi) in x.iter().enumerate() {
                    fwin[i] = i0(dx * arg(xi).sqrt()) / denom;
                }
                for (i, &xi) in x.iter().enumerate() {
                    if xi <= x1 || xi >= x4 {
                        fwin[i] = 0.0;
                    }
                }
            } else {
                let scale = (i0(dx) - 1.0).max(1.0e-10);
                for (i, &xi) in x.iter().enumerate() {
                    fwin[i] = (i0(dx * arg(xi).sqrt()) - 1.0) / scale;
                }
            }
        }
        Window::Sine => {
            for i in i1..=i4.min(n - 1) {
                fwin[i] = (PI * (x4 - x[i]) / (x4 - x1)).sin();
            }
        }
        Window::Gaussian => {
            let cen = (x4 + x1) / 2.0;
            for (i, &xi) in x.iter().enumerate() {
                fwin[i] = (-((xi - cen).powi(2)) / (2.0 * dx1 * dx1)).exp();
            }
        }
    }
    fwin
}

fn min_of(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::INFINITY, f64::min)
}
fn max_of(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
