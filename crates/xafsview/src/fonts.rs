//! Glyph-fallback fonts for the whole GUI.
//!
//! egui ships Ubuntu-Light as its proportional UI font, which lacks the
//! superscript/subscript modifier letters and a few math symbols the XAFS
//! labels use — `⁻` (U+207B superscript minus), `ʷ` (U+02B7), `ᵣ` (U+1D63) —
//! so they render as the missing-glyph box. rsplot draws its axis labels through
//! the same egui `Context` fonts, so it inherits the gap on `k (Å⁻¹)`, `kʷ·χ(k)`,
//! `χ²ᵣ`, etc.
//!
//! Noto Sans fills those, but misses the arrows and box-drawing glyphs the UI
//! also uses — `→` (U+2192) in the Atoms/Feff headers and status lines, `⇒`
//! (U+21D2) in the k↔E converter, and rsplot's ROI line-style labels `─ ╌ ┈`
//! and compare-toolbar `═` — so DejaVu Sans follows it in the chain for the
//! symbol blocks.
//!
//! Installing broad-coverage fonts as the *last* fallbacks in every font family
//! fills only the glyphs the primary fonts lack — the normal Latin text keeps
//! Ubuntu-Light's look — and because the fallbacks are registered on the shared
//! `Context`, they reach rsplot's labels too. One call at startup closes the
//! whole missing-glyph family for both the egui UI and the plots.

use std::sync::Arc;

use eframe::egui;

/// Noto Sans Regular (v2.007, SIL OFL 1.1 — see `assets/fonts/OFL.txt`),
/// vendored unmodified. Covers the superscript/subscript and Greek glyphs the
/// default UI font misses.
static NOTO_SANS: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");

/// DejaVu Sans (v2.37, Bitstream Vera license — see
/// `assets/fonts/LICENSE-DejaVu.txt`), vendored unmodified. Covers the arrow,
/// math-operator, box-drawing, and geometric-shape blocks that both Ubuntu-Light
/// and Noto Sans lack.
static DEJAVU_SANS: &[u8] = include_bytes!("../assets/fonts/DejaVuSans.ttf");

/// The egui font set plus the vendored fallbacks, appended (not prepended) to
/// every family: a fallback at the end is consulted only for glyphs the
/// family's earlier fonts cannot render, so the primary UI font is unchanged
/// and the fallbacks only fill the gaps.
fn font_definitions() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "noto_sans".to_owned(),
        Arc::new(egui::FontData::from_static(NOTO_SANS)),
    );
    fonts.font_data.insert(
        "dejavu_sans".to_owned(),
        Arc::new(egui::FontData::from_static(DEJAVU_SANS)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        let list = fonts.families.entry(family).or_default();
        list.push("noto_sans".to_owned());
        list.push("dejavu_sans".to_owned());
    }
    fonts
}

/// Register the fallback fonts on `ctx`, so any glyph the default fonts lack
/// (superscript minus, modifier letters, Greek, arrows, box drawing) is filled
/// — in the egui UI *and* in rsplot's axis labels, which paint through this
/// same context.
pub fn install_fallback(ctx: &egui::Context) {
    ctx.set_fonts(font_definitions());
}

#[cfg(test)]
mod tests {
    use skrifa::MetadataProvider;

    use super::*;

    /// Every non-ASCII glyph a user-visible string in xafsview or the embedded
    /// rsplot widgets uses must be covered by the installed font chain —
    /// swapping or dropping a fallback font must fail here, not render tofu.
    #[test]
    fn ui_glyphs_covered_by_font_chain() {
        let defs = font_definitions();
        let family = &defs.families[&egui::FontFamily::Proportional];
        let faces: Vec<(String, skrifa::FontRef)> = family
            .iter()
            .map(|name| {
                let data = &defs.font_data[name].font;
                (
                    name.clone(),
                    skrifa::FontRef::new(data).expect("font data parses"),
                )
            })
            .collect();

        // Science notation (the Noto Sans additions) and the symbols in UI
        // labels and status lines (the DejaVu Sans additions).
        let needed = "μχσΔθÅ°±²³⁻ʷᵣ…·½—→⇒─╌┈═";
        let missing: Vec<char> = needed
            .chars()
            .filter(|&c| !faces.iter().any(|(_, f)| f.charmap().map(c).is_some()))
            .collect();
        assert!(
            missing.is_empty(),
            "UI glyphs not covered by any font in the chain: {missing:?}"
        );
    }
}
