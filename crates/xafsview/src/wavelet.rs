//! Morlet continuous wavelet transform of EXAFS `χ(k)` — the wavelet view.
//!
//! Uses the pure-Rust [`fcwt`] crate (rustfft backend) to compute a 2-D
//! `|W(k,R)|` magnitude map, which localizes EXAFS contributions jointly in `k`
//! and `R` (single-scattering shells appear as blobs at their `R` and their
//! dominant `k` range). [`morlet_cwt`] is the pure transform (unit-tested);
//! [`WaveletWindow`] renders the heatmap.
//!
//! **R calibration.** `fcwt` analyzes at frequency `f` (cycles per unit-`k`),
//! and its Morlet carrier has period `scale = fs/f` samples, so it matches a
//! signal of angular frequency `2π·f` per unit-`k`. EXAFS `χ(k)` oscillates as
//! `sin(2kR + φ)` — angular frequency `2R` — so the analyzing frequency that
//! responds to a shell at `R` is `f = R/π`, i.e. **`R = π·f`**. The sampling
//! rate is `fs = 1/kstep` (it must be an integer; the autobk `kstep = 0.05`
//! grid gives `fs = 20`).

use eframe::egui;
use egui::Color32;
use fcwt::{FastCwt, LinFreqs, MorletWavelet};
use std::f64::consts::PI;

/// A Morlet wavelet transform of `χ(k)`: a 2-D `|W(k,R)|` magnitude grid.
pub struct WaveletTransform {
    /// k axis (Å⁻¹), one entry per column.
    pub k: Vec<f64>,
    /// R axis (Å), one entry per row; row 0 is `rmax`, the last row `rmin`.
    pub r: Vec<f64>,
    /// `magnitude[row][col]` = `|W|` at `R = r[row]`, `k = k[col]`.
    pub magnitude: Vec<Vec<f64>>,
    /// Largest magnitude in the grid (for color scaling).
    pub max: f64,
}

impl WaveletTransform {
    /// Serialize the `|W(k,R)|` grid as a labelled CSV matrix: a header row of
    /// the `k` values, then one row per `R` (in stored top-down order — row 0 is
    /// `rmax`), each prefixed with its `R` value. Loads directly as a labelled
    /// matrix in numpy/pandas/Excel:
    ///
    /// ```text
    /// R\k,k0,k1,…,k_{n-1}
    /// r0,|W|,…
    /// r1,|W|,…
    /// ```
    ///
    /// Axis values use 4 decimals; magnitudes use scientific notation so the
    /// map's full dynamic range survives the round-trip.
    pub fn to_csv(&self) -> String {
        let mut s = String::from("R\\k");
        for k in &self.k {
            s.push_str(&format!(",{k:.4}"));
        }
        s.push('\n');
        for (row, mags) in self.magnitude.iter().enumerate() {
            s.push_str(&format!("{:.4}", self.r[row]));
            for m in mags {
                s.push_str(&format!(",{m:.6e}"));
            }
            s.push('\n');
        }
        s
    }
}

/// Compute the Morlet continuous wavelet transform of k-weighted `χ(k)`.
///
/// `k` is a uniform grid of step `kstep`; `chi` is co-indexed. `p` carries the
/// k-weighting, R range, scale count, and Morlet σ (see [`WaveletParams`]).
/// Returns `None` when the inputs are too short, mis-shaped, or the requested
/// `rmax` exceeds the Nyquist limit for this `kstep` (`rmax/π > fs/2`).
pub fn morlet_cwt(
    k: &[f64],
    chi: &[f64],
    kstep: f64,
    p: &WaveletParams,
) -> Option<WaveletTransform> {
    let WaveletParams {
        kweight,
        rmin,
        rmax,
        nr,
        bandwidth,
    } = *p;
    let n = k.len();
    if n < 8 || chi.len() != n || kstep <= 0.0 || nr == 0 || rmax <= rmin || rmin <= 0.0 {
        return None;
    }
    let fs = (1.0 / kstep).round() as usize;
    if fs < 2 {
        return None;
    }
    // R = π·f  ⇒  f = R/π. fcwt requires end_freq ≤ Nyquist (fs/2).
    let fmin = (rmin / PI) as f32;
    let fmax = (rmax / PI) as f32;
    if fmax <= 0.0 || fmax > (fs / 2) as f32 || fmin >= fmax {
        return None;
    }

    // fcwt's FFT path requires a power-of-two length, so zero-pad the k-weighted
    // signal to `npad` (this is what larch's `cauchy_wavelet` does). The padding
    // sits past `k[n-1]`; its right-edge artifact lands in columns we truncate
    // off below, so the k-supported region is unaffected.
    let npad = n.next_power_of_two();
    let mut signal = vec![0.0f32; npad];
    for (slot, (&kk, &c)) in signal.iter_mut().zip(k.iter().zip(chi)) {
        *slot = (c * kk.powi(kweight)) as f32;
    }

    let wavelet = MorletWavelet::new(bandwidth as f32);
    let scales = LinFreqs::new(&wavelet, fs, fmin, fmax, nr);
    let mut cwt = FastCwt::new(wavelet, scales, true);
    let result = cwt.cwt(&signal);
    let rows = result.rows();
    if rows.is_empty() {
        return None;
    }

    // R axis. LinFreqs samples freq(row) = fmin + (df/nr)·(nr-1-row), so row 0 is
    // the highest freq (largest R) and the last row is `fmin` (smallest R).
    let df = fmax - fmin;
    let r: Vec<f64> = (0..rows.len())
        .map(|row| {
            let f = fmin + (df / nr as f32) * (nr as f32 - 1.0 - row as f32);
            PI * f as f64
        })
        .collect();

    // Drop the zero-pad columns (keep only the `n` real-k samples).
    let mut magnitude = Vec::with_capacity(rows.len());
    let mut max = 0.0_f64;
    for row in rows {
        let m: Vec<f64> = row
            .iter()
            .take(n)
            .map(|c| {
                let v = (c.re as f64).hypot(c.im as f64);
                if v > max {
                    max = v;
                }
                v
            })
            .collect();
        magnitude.push(m);
    }

    Some(WaveletTransform {
        k: k.to_vec(),
        r,
        magnitude,
        max,
    })
}

/// What the wavelet window needs the app to do this frame.
pub enum WaveletAction {
    /// Recompute the transform from the current group's `χ(k)`.
    Compute,
    /// Save the last `|W(k,R)|` result to a CSV file (the app runs the save
    /// dialog and writes [`WaveletWindow::result_csv`]).
    ExportCsv,
}

/// The floating wavelet-transform window: parameters, a Compute button, and the
/// `|W(k,R)|` heatmap of the last result.
pub struct WaveletWindow {
    /// Whether the window is shown.
    pub open: bool,
    kweight: i32,
    rmin: f64,
    rmax: f64,
    nr: usize,
    bandwidth: f64,
    wt: Option<WaveletTransform>,
    texture: Option<egui::TextureHandle>,
    info: String,
}

impl Default for WaveletWindow {
    fn default() -> Self {
        Self {
            open: false,
            kweight: 2,
            rmin: 0.5,
            rmax: 6.0,
            nr: 256,
            bandwidth: 1.0,
            wt: None,
            texture: None,
            info: String::new(),
        }
    }
}

impl WaveletWindow {
    /// The transform parameters the app should use to compute the next result.
    pub fn params(&self) -> WaveletParams {
        WaveletParams {
            kweight: self.kweight,
            rmin: self.rmin,
            rmax: self.rmax,
            nr: self.nr,
            bandwidth: self.bandwidth,
        }
    }

    /// The last result serialized as CSV (see [`WaveletTransform::to_csv`]), or
    /// `None` when nothing has been computed yet. The app calls this to fill the
    /// Export-CSV save dialog.
    pub fn result_csv(&self) -> Option<String> {
        self.wt.as_ref().map(WaveletTransform::to_csv)
    }

    /// Store a freshly computed transform (or `None` if it could not be
    /// computed), invalidating the cached texture so it rebuilds on next show.
    pub fn set_result(&mut self, wt: Option<WaveletTransform>, info: String) {
        self.wt = wt;
        self.texture = None;
        self.info = info;
    }

    /// Render the window. Returns [`WaveletAction::Compute`] when the user asks
    /// for a (re)compute. `has_chi` enables the Compute button.
    pub fn show(&mut self, ctx: &egui::Context, has_chi: bool) -> Option<WaveletAction> {
        // (Re)build the heatmap texture from the current result if needed.
        if self.texture.is_none()
            && let Some(wt) = &self.wt
        {
            let image = to_color_image(wt);
            self.texture = Some(ctx.load_texture("wavelet", image, egui::TextureOptions::LINEAR));
        }

        let mut action = None;
        let mut open = self.open;
        crate::window::detached(
            ctx,
            "wavelet",
            "Wavelet |W(k,R)|",
            &mut open,
            [560.0, 560.0],
            |ui| {
                ui.horizontal(|ui| {
                    ui.label("k-weight");
                    ui.add(egui::DragValue::new(&mut self.kweight).range(0..=3));
                    ui.label("σ");
                    ui.add(
                        egui::DragValue::new(&mut self.bandwidth)
                            .speed(0.1)
                            .range(0.2..=8.0),
                    );
                });
                ui.add(egui::Slider::new(&mut self.rmin, 0.1..=4.0).text("R min (Å)"));
                ui.add(egui::Slider::new(&mut self.rmax, 1.0..=10.0).text("R max (Å)"));
                ui.add(egui::Slider::new(&mut self.nr, 32..=512).text("R samples"));

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(has_chi, egui::Button::new("Compute"))
                        .clicked()
                    {
                        action = Some(WaveletAction::Compute);
                    }
                    if ui
                        .add_enabled(self.wt.is_some(), egui::Button::new("Export CSV…"))
                        .on_hover_text("Save |W(k,R)| as a CSV matrix (k columns, R rows)")
                        .clicked()
                    {
                        action = Some(WaveletAction::ExportCsv);
                    }
                });
                if !self.info.is_empty() {
                    ui.weak(&self.info);
                }
                ui.separator();

                match (&self.wt, &self.texture) {
                    (Some(wt), Some(tex)) => {
                        ui.weak(format!(
                            "k: {:.1}…{:.1} Å⁻¹   (R increases upward: {:.1}…{:.1} Å)",
                            wt.k.first().copied().unwrap_or(0.0),
                            wt.k.last().copied().unwrap_or(0.0),
                            wt.r.last().copied().unwrap_or(0.0),
                            wt.r.first().copied().unwrap_or(0.0),
                        ));
                        let w = ui.available_width().clamp(200.0, 700.0);
                        ui.add(
                            egui::Image::new(tex)
                                .fit_to_exact_size(egui::vec2(w, 320.0))
                                .maintain_aspect_ratio(false),
                        );
                    }
                    _ => {
                        ui.weak("Run AUTOBK to get χ(k), then Compute.");
                    }
                }
            },
        );
        self.open = open;
        action
    }
}

/// The transform parameters chosen in the window (the app pairs these with the
/// current group's `k`/`χ`).
#[derive(Clone, Copy)]
pub struct WaveletParams {
    pub kweight: i32,
    pub rmin: f64,
    pub rmax: f64,
    pub nr: usize,
    pub bandwidth: f64,
}

/// Render a [`WaveletTransform`] to an RGBA image: column = k, row = R (row 0 =
/// `rmax` at the top), color = a plasma map of `|W|/max`.
fn to_color_image(wt: &WaveletTransform) -> egui::ColorImage {
    let h = wt.magnitude.len();
    let w = wt.magnitude.first().map_or(0, |r| r.len());
    let inv_max = if wt.max > 0.0 { 1.0 / wt.max } else { 0.0 };
    let mut pixels = Vec::with_capacity(w * h);
    for row in &wt.magnitude {
        for &v in row {
            pixels.push(plasma(v * inv_max));
        }
    }
    egui::ColorImage::new([w, h], pixels)
}

/// A compact plasma-style colormap for `t ∈ [0, 1]`.
fn plasma(t: f64) -> Color32 {
    const STOPS: [(f64, [u8; 3]); 5] = [
        (0.00, [13, 8, 135]),
        (0.25, [126, 3, 168]),
        (0.50, [204, 71, 120]),
        (0.75, [248, 149, 64]),
        (1.00, [240, 249, 33]),
    ];
    let t = t.clamp(0.0, 1.0);
    for pair in STOPS.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * f).round() as u8;
            return Color32::from_rgb(lerp(c0[0], c1[0]), lerp(c0[1], c1[1]), lerp(c0[2], c1[2]));
        }
    }
    let last = STOPS[STOPS.len() - 1].1;
    Color32::from_rgb(last[0], last[1], last[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure `sin(2·R0·k)` tone must produce a wavelet ridge at `R ≈ R0`,
    /// validating the `R = π·f` calibration end-to-end through fcwt.
    #[test]
    fn ridge_lands_at_expected_r() {
        let kstep = 0.05;
        let r0 = 2.5;
        let k: Vec<f64> = (0..360).map(|i| i as f64 * kstep).collect();
        let chi: Vec<f64> = k.iter().map(|&kk| (2.0 * r0 * kk).sin()).collect();

        let p = WaveletParams {
            kweight: 0,
            rmin: 1.0,
            rmax: 4.0,
            nr: 240,
            bandwidth: 2.0,
        };
        let wt = morlet_cwt(&k, &chi, kstep, &p).expect("cwt");
        assert_eq!(wt.magnitude.len(), 240);
        assert_eq!(wt.magnitude[0].len(), k.len());

        // Row with the largest total magnitude over the well-supported middle.
        let lo = k.len() / 4;
        let hi = 3 * k.len() / 4;
        let mut best_row = 0;
        let mut best_sum = -1.0_f64;
        for (row, m) in wt.magnitude.iter().enumerate() {
            let s: f64 = m[lo..hi].iter().sum();
            if s > best_sum {
                best_sum = s;
                best_row = row;
            }
        }
        let r_peak = wt.r[best_row];
        assert!(
            (r_peak - r0).abs() < 0.35,
            "wavelet ridge should sit at R≈{r0}, got {r_peak:.3}"
        );
    }

    #[test]
    fn rejects_bad_shapes_and_nyquist() {
        let k: Vec<f64> = (0..40).map(|i| i as f64 * 0.05).collect();
        let chi = vec![0.0; 40];
        let base = WaveletParams {
            kweight: 0,
            rmin: 1.0,
            rmax: 4.0,
            nr: 64,
            bandwidth: 1.0,
        };
        // mismatched lengths
        assert!(morlet_cwt(&k, &chi[..30], 0.05, &base).is_none());
        // rmin >= rmax
        assert!(morlet_cwt(&k, &chi, 0.05, &WaveletParams { rmin: 4.0, ..base }).is_none());
        // rmax beyond Nyquist: fs=20 → Nyq f=10 → R=π·10≈31.4; ask R=40
        assert!(morlet_cwt(&k, &chi, 0.05, &WaveletParams { rmax: 40.0, ..base }).is_none());
    }

    #[test]
    fn to_csv_labels_k_header_and_r_rows() {
        let wt = WaveletTransform {
            k: vec![0.0, 0.05],
            r: vec![3.0, 1.0],
            magnitude: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            max: 4.0,
        };
        let csv = wt.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        // Header + one row per R.
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "R\\k,0.0000,0.0500");
        // Each data row: its R value, then one |W| per k column.
        assert!(lines[1].starts_with("3.0000,"));
        assert!(lines[2].starts_with("1.0000,"));
        assert_eq!(lines[1].split(',').count(), 3);
        assert_eq!(lines[2].split(',').count(), 3);
    }

    #[test]
    fn plasma_is_monotone_endpoints() {
        assert_eq!(plasma(0.0), Color32::from_rgb(13, 8, 135));
        assert_eq!(plasma(1.0), Color32::from_rgb(240, 249, 33));
        // clamps out-of-range
        assert_eq!(plasma(-1.0), Color32::from_rgb(13, 8, 135));
        assert_eq!(plasma(2.0), Color32::from_rgb(240, 249, 33));
    }
}
