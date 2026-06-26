//! Reading text files that may not be UTF-8.
//!
//! Beamline raw files from Korean facilities (e.g. PLS-II) carry CP949/EUC-KR
//! text in their headers — sample names, the `Description :` note — which a plain
//! `std::fs::read_to_string` rejects with a UTF-8 error, so the *whole* file fails
//! to load even though the numeric data is ASCII. The same applies to a
//! `feff.inp` / `feffNNNN.dat` whose TITLE was typed in Korean.
//!
//! [`read_to_string_lenient`] is the single reader the data/FEFF text loaders go
//! through so this decoding rule lives in one place.

use std::io;
use std::path::Path;

/// Read a text file, decoding it as UTF-8 when the bytes are valid UTF-8 and
/// otherwise falling back to EUC-KR (CP949) — the standard Korean-Windows
/// encoding. The fallback decode never fails (any undecodable byte becomes
/// U+FFFD), so a readable file always yields a `String`; only a genuine I/O error
/// propagates. Valid-UTF-8 files are unaffected: they are decoded as UTF-8, never
/// reinterpreted as EUC-KR.
pub fn read_to_string_lenient(path: &Path) -> io::Result<String> {
    let bytes = std::fs::read(path)?;
    match std::str::from_utf8(&bytes) {
        Ok(s) => Ok(s.to_owned()),
        Err(_) => Ok(encoding_rs::EUC_KR.decode(&bytes).0.into_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("feffit_textio_{}_{}.txt", tag, std::process::id()))
    }

    #[test]
    fn falls_back_to_euc_kr_for_non_utf8_korean() {
        // A Korean-Windows file: CP949 bytes that are NOT valid UTF-8.
        let (bytes, _, _) = encoding_rs::EUC_KR.encode("한글 시료명");
        assert!(
            std::str::from_utf8(&bytes).is_err(),
            "fixture must be non-UTF-8 to exercise the fallback"
        );
        let path = temp_path("euckr");
        std::fs::write(&path, &bytes).expect("write");

        let text = read_to_string_lenient(&path).expect("non-UTF-8 file must still read");
        assert_eq!(text, "한글 시료명");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reads_valid_utf8_unchanged() {
        let path = temp_path("utf8");
        let content = "한글 UTF-8 ✓\n7000 0.1\n";
        std::fs::write(&path, content).expect("write");

        let text = read_to_string_lenient(&path).expect("read");
        assert_eq!(
            text, content,
            "valid UTF-8 must not be reinterpreted as EUC-KR"
        );

        let _ = std::fs::remove_file(&path);
    }
}
