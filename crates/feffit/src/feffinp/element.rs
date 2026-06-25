//! Element symbol ⇄ atomic-number lookup.
//!
//! The cluster builder needs `Z` only for the `POTENTIALS` card, and the
//! `atoms` crate must stay self-contained (workspace MSRV 1.89, no `xraydb`
//! dependency — `xraydb` is the GUI's MSRV-1.92, headless-unfriendly atomic
//! database). Atomic numbers are universal physical constants, so a small
//! built-in table is the right call here rather than a cross-crate dependency.
//! The GUI keeps its own display-oriented periodic table (validated against
//! `xraydb`); this table is the library-layer counterpart, self-validated by
//! the `z_symbol_round_trips` test.

/// Element symbols indexed by `Z − 1` (so `ELEMENTS[0] == "H"`). Z 1‥118.
const ELEMENTS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];

/// The symbol for atomic number `z` (1‥118), or `None` if out of range.
pub fn z_to_symbol(z: u32) -> Option<&'static str> {
    let z = usize::try_from(z).ok()?;
    z.checked_sub(1).and_then(|i| ELEMENTS.get(i).copied())
}

/// The atomic number for an element `symbol`, case-insensitively (so `"fe"`,
/// `"Fe"`, and `"FE"` all resolve to 26). Returns `None` for an unknown symbol.
pub fn symbol_to_z(symbol: &str) -> Option<u32> {
    let s = symbol.trim();
    ELEMENTS
        .iter()
        .position(|e| e.eq_ignore_ascii_case(s))
        .map(|i| i as u32 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_symbol_round_trips() {
        for z in 1..=118u32 {
            let sym = z_to_symbol(z).expect("symbol in range");
            assert_eq!(symbol_to_z(sym), Some(z), "round-trip Z={z}");
        }
    }

    #[test]
    fn known_anchors() {
        assert_eq!(symbol_to_z("H"), Some(1));
        assert_eq!(symbol_to_z("Cu"), Some(29));
        assert_eq!(symbol_to_z("Fe"), Some(26));
        assert_eq!(symbol_to_z("Pb"), Some(82));
        assert_eq!(symbol_to_z("Og"), Some(118));
        assert_eq!(z_to_symbol(8), Some("O"));
        assert_eq!(z_to_symbol(0), None);
        assert_eq!(z_to_symbol(119), None);
        assert_eq!(symbol_to_z("Xx"), None);
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(symbol_to_z("fe"), Some(26));
        assert_eq!(symbol_to_z("FE"), Some(26));
        assert_eq!(symbol_to_z("  cu  "), Some(29));
    }
}
