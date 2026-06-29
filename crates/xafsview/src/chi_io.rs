//! Serializers for the column-text data files XAFSView writes (`.chi`, `.dat`,
//! `.fit`, `.bkg`): a `# label` line, a `# column-names` line, then numeric
//! columns in the UWXAFS feffit/AUTOBK layout — Fortran `E13.7` numbers and CRLF
//! line endings, matching the reference files byte-for-byte in *format* (the
//! values themselves differ slightly because our FT engine is not the UWXAFS
//! one). Kept in one place so the on-disk single-fit writers (`app.rs`) and the
//! batch `(filename, content)` builder (`feffit_batch.rs`) emit byte-identical
//! files instead of drifting two copies of the format.

/// Format `v` in Fortran `E13.7` form — `0.dddddddE±ee`, a normalized mantissa
/// in `[0.1, 1.0)` with 7 digits and a 2-digit signed exponent, exactly 13 chars
/// wide. Negative values drop the leading zero (`-.dddddddE±ee`) so the field
/// stays 13 wide, which is what Fortran's `E13.7` edit descriptor does and what
/// the reference UWXAFS files contain. Non-finite / zero render as the canonical
/// `0.0000000E+00` (our column data is always finite, so this only fixes the
/// shape rather than emitting `inf`/`NaN`).
fn fortran_e13_7(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return "0.0000000E+00".to_string();
    }
    let neg = v < 0.0;
    let a = v.abs();
    let mut exp = a.log10().floor() as i32 + 1;
    let mut digits = (a / 10f64.powi(exp) * 1.0e7).round() as i64;
    // Rounding can carry the mantissa up to 1.0 (10_000_000 = eight digits);
    // renormalize back to a leading 0.1 and bump the exponent.
    if digits >= 10_000_000 {
        digits = 1_000_000;
        exp += 1;
    }
    let lead = if neg { '-' } else { '0' };
    let esign = if exp < 0 { '-' } else { '+' };
    format!("{lead}.{digits:07}E{esign}{:02}", exp.abs())
}

/// Append `line` followed by a CRLF — the UWXAFS files use DOS line endings on
/// every line (header and data alike).
fn push_crlf(s: &mut String, line: &str) {
    s.push_str(line);
    s.push_str("\r\n");
}

/// Two-column k-space body — `k`, `χ(k)` — in the UWXAFS `.chi`/`.dat`/`.fit`
/// layout: a `2x,E13.7,3x,E13.7` row (column widths 15 then 16) under the fixed
/// `#     k              chi(k)` header, CRLF throughout.
pub(crate) fn chik_string(label: &str, k: &[f64], chi: &[f64]) -> String {
    let mut s = String::with_capacity(k.len() * 34 + 96);
    push_crlf(&mut s, &format!("# {label}"));
    push_crlf(&mut s, "#     k              chi(k)          ");
    for (&kk, &cc) in k.iter().zip(chi) {
        push_crlf(
            &mut s,
            &format!("{:>15}{:>16}", fortran_e13_7(kk), fortran_e13_7(cc)),
        );
    }
    s
}

/// Two-column `x`, `y` body with a caller-supplied column header. Used for the
/// AUTOBK background files the original XAFSView writes: `e.bkg` (energy, μ₀) and
/// `k.bkg` (k, μ₀−μ). Same Fortran `E13.7` / CRLF numeric layout as the
/// `.chi`/`.dat`/`.fit` k-space files (`2x,E13.7,3x,E13.7`).
pub(crate) fn xy_string(label: &str, columns: &str, x: &[f64], y: &[f64]) -> String {
    let mut s = String::with_capacity(x.len() * 34 + 96);
    push_crlf(&mut s, &format!("# {label}"));
    push_crlf(&mut s, &format!("# {columns}"));
    for (&xx, &yy) in x.iter().zip(y) {
        push_crlf(
            &mut s,
            &format!("{:>15}{:>16}", fortran_e13_7(xx), fortran_e13_7(yy)),
        );
    }
    s
}

/// Five-column R/q-space body — `axis`, `real`, `imag`, `ampl`, `phase` — in the
/// UWXAFS `.chi`/`.dat`/`.fit` layout (`5E15.7`, CRLF). `sym` names the space:
/// `"r"` for R-space (`r` column, `chi(r)` titles) and `"k"` for q-space (the q
/// transform lives on a k-grid, so its axis column is named `k` with `chi(k)`
/// titles) — in both the column name and the bracketed symbol are the same
/// letter, matching the reference. `ampl` is the supplied magnitude (≡ |re+i·im|)
/// and `phase` is derived as `atan2(im, re)`.
pub(crate) fn complex5_string(
    label: &str,
    sym: &str,
    x: &[f64],
    mag: &[f64],
    re: &[f64],
    im: &[f64],
) -> String {
    let mut s = String::with_capacity(x.len() * 80 + 128);
    push_crlf(&mut s, &format!("# {label}"));
    push_crlf(
        &mut s,
        &format!(
            "#     {sym}          real[chi({sym})]   imag[chi({sym})]   ampl[chi({sym})]   phase[chi({sym})] "
        ),
    );
    for (i, &xx) in x.iter().enumerate() {
        let m = mag.get(i).copied().unwrap_or(0.0);
        let re_i = re.get(i).copied().unwrap_or(0.0);
        let im_i = im.get(i).copied().unwrap_or(0.0);
        let ph = im_i.atan2(re_i);
        push_crlf(
            &mut s,
            &format!(
                "{:>15}{:>15}{:>15}{:>15}{:>15}",
                fortran_e13_7(xx),
                fortran_e13_7(re_i),
                fortran_e13_7(im_i),
                fortran_e13_7(m),
                fortran_e13_7(ph),
            ),
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_data_row(s: &str) -> &str {
        s.lines().find(|l| !l.starts_with('#')).expect("a data row")
    }

    #[test]
    fn fortran_e13_7_matches_the_reference_field() {
        // Values lifted from the reference UWXAFS files; each renders to exactly
        // 13 characters with the leading zero dropped on negatives.
        assert_eq!(fortran_e13_7(0.05), "0.5000000E-01");
        assert_eq!(fortran_e13_7(4.961208), "0.4961208E+01");
        assert_eq!(fortran_e13_7(-0.02182338), "-.2182338E-01");
        assert_eq!(fortran_e13_7(6339.0), "0.6339000E+04");
        assert_eq!(fortran_e13_7(0.0), "0.0000000E+00");
        for v in [0.05, -7.549715, 1234.5, -0.000123, 1.0] {
            assert_eq!(fortran_e13_7(v).len(), 13, "E13.7 is 13 wide for {v}");
        }
    }

    #[test]
    fn chik_string_has_two_columns_crlf_and_31_char_rows() {
        let s = chik_string("t", &[0.05, 2.0], &[4.961208, 0.2]);
        assert_eq!(first_data_row(&s).split_whitespace().count(), 2);
        // Every line is CRLF-terminated.
        assert!(s.lines().count() >= 3 && s.ends_with("\r\n"));
        // The k-space data row is `2x,E13.7,3x,E13.7` = 31 chars before the CR.
        let row = s.split("\r\n").find(|l| l.starts_with("  ")).unwrap();
        assert_eq!(row.len(), 31, "row=[{row}]");
        assert_eq!(row, "  0.5000000E-01   0.4961208E+01");
    }

    #[test]
    fn xy_string_has_two_columns_and_the_given_header() {
        let s = xy_string("t", " energy            bkg", &[1.0, 2.0], &[0.1, 0.2]);
        assert!(s.contains("energy"), "column header expected: {s}");
        assert_eq!(first_data_row(&s).split_whitespace().count(), 2);
        assert!(s.ends_with("\r\n"));
    }

    #[test]
    fn complex5_string_has_five_columns_names_the_axis_and_derives_phase() {
        // q-space: axis column named `k`, titles `chi(k)`; phase = atan2(im, re).
        let s = complex5_string("t", "k", &[1.0], &[0.5], &[0.3], &[0.4]);
        assert!(s.contains("ampl[chi(k)]"), "q-axis header expected: {s}");
        let row = first_data_row(&s);
        assert_eq!(row.split_whitespace().count(), 5);
        // 5E15.7 ⇒ 75 chars before the CR.
        let row = s.split("\r\n").find(|l| l.starts_with("  ")).unwrap();
        assert_eq!(row.len(), 75, "row=[{row}]");
        // Columns are axis, real, imag, ampl, phase: phase = atan2(0.4, 0.3).
        let cols: Vec<&str> = row.split_whitespace().collect();
        assert_eq!(cols[1], "0.3000000E+00", "real");
        assert_eq!(cols[2], "0.4000000E+00", "imag");
        assert_eq!(cols[3], "0.5000000E+00", "ampl");
        let phase: f64 = cols[4].parse().expect("phase parses");
        assert!((phase - 0.4_f64.atan2(0.3)).abs() < 1e-6, "phase={phase}");
    }
}
