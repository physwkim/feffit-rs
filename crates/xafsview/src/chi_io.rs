//! Serializers for the column-text data files XAFSView writes (`.chi`, `.dat`,
//! `.fit`): a `# label` line, a `# column-names` line, then fixed/scientific
//! numeric columns. Kept in one place so the on-disk single-fit writers
//! (`app.rs`) and the batch `(filename, content)` builder (`feffit_batch.rs`)
//! emit byte-identical files instead of drifting two copies of the format.

use std::fmt::Write as _;

/// Two-column `k`, `χ(k)` body — the k-space `.chi`/`.dat`/`.fit` format.
pub(crate) fn chik_string(label: &str, k: &[f64], chi: &[f64]) -> String {
    let mut s = String::with_capacity(k.len() * 32 + 64);
    let _ = writeln!(s, "# {label}");
    let _ = writeln!(s, "#  k                chi");
    for (&kk, &cc) in k.iter().zip(chi) {
        let _ = writeln!(s, "{kk:12.6}  {cc:18.10}");
    }
    s
}

/// Four-column `axis`, `|chi|`, `re`, `im` body — the R/q-space
/// `.chi`/`.dat`/`.fit` format. `axis` names the first column (`R` or `q`) so
/// r- and q-space files are labelled correctly.
pub(crate) fn complex4_string(
    label: &str,
    axis: &str,
    x: &[f64],
    mag: &[f64],
    re: &[f64],
    im: &[f64],
) -> String {
    let mut s = String::with_capacity(x.len() * 60 + 64);
    let _ = writeln!(s, "# {label}");
    let _ = writeln!(
        s,
        "#  {axis:<12}  |chi({axis})|        re                im"
    );
    for (i, &xx) in x.iter().enumerate() {
        let m = mag.get(i).copied().unwrap_or(0.0);
        let re_i = re.get(i).copied().unwrap_or(0.0);
        let im_i = im.get(i).copied().unwrap_or(0.0);
        let _ = writeln!(s, "{xx:12.6}  {m:16.8e}  {re_i:16.8e}  {im_i:16.8e}");
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
    fn chik_string_has_two_data_columns() {
        let s = chik_string("t", &[1.0, 2.0], &[0.1, 0.2]);
        assert_eq!(first_data_row(&s).split_whitespace().count(), 2);
    }

    #[test]
    fn complex4_string_has_four_columns_and_names_the_axis() {
        let s = complex4_string("t", "q", &[1.0], &[0.5], &[0.3], &[0.4]);
        assert!(s.contains("|chi(q)|"), "q-axis header expected: {s}");
        assert_eq!(first_data_row(&s).split_whitespace().count(), 4);
    }
}
