//! Glyph-fallback font for the whole GUI.
//!
//! egui 0.34 ships Ubuntu-Light as its proportional UI font, which lacks the
//! superscript/subscript modifier letters and a few math symbols the XAFS
//! labels use — `⁻` (U+207B superscript minus), `ʷ` (U+02B7), `ᵣ` (U+1D63) —
//! so they render as the missing-glyph box. siplot draws its axis labels through
//! the same egui `Context` fonts, so it inherits the gap on `k (Å⁻¹)`, `kʷ·χ(k)`,
//! `χ²ᵣ`, etc.
//!
//! Installing a broad-coverage font (Noto Sans, SIL OFL 1.1) as the *last*
//! fallback in every font family fills only the glyphs the primary fonts lack —
//! the normal Latin text keeps Ubuntu-Light's look — and because the fallback is
//! registered on the shared `Context`, it reaches siplot's labels too. One call
//! at startup closes the whole missing-glyph family for both the egui UI and the
//! plots.

use std::sync::Arc;

use eframe::egui;

/// Noto Sans Regular (v2.007, SIL OFL 1.1 — see `assets/fonts/OFL.txt`),
/// vendored unmodified. Covers the superscript/subscript and Greek glyphs the
/// default UI font misses.
static NOTO_SANS: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");

/// Register Noto Sans as the last glyph fallback for every font family on `ctx`,
/// so any glyph the default fonts lack (superscript minus, modifier letters,
/// Greek) is filled — in the egui UI *and* in siplot's axis labels, which paint
/// through this same context.
pub fn install_fallback(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "noto_sans".to_owned(),
        Arc::new(egui::FontData::from_static(NOTO_SANS)),
    );
    // Append (not prepend): a fallback at the end is consulted only for glyphs
    // the family's earlier fonts cannot render, so the primary UI font is
    // unchanged and Noto Sans only fills the gaps.
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("noto_sans".to_owned());
    }
    ctx.set_fonts(fonts);
}
