//! The unified "Open data" loader for the Autobk flow.
//!
//! One in-app window that replaces both the native OS file dialog and the old
//! "Make μ(E) from files" picker. It is anchored on the Folders Data folder and
//! offers the shared [`FileBrowser`] — Up / sub-folder / Browse navigation, a
//! two-pane transfer list (stage raw scan files on the right with `=>`, pull a
//! wrong pick back with `<=`, Sort / Clear all, Delete) — plus, hosted by the app
//! around this browser, the reading-format chooser
//! ([`crate::import::ImportState`]) and an **Autobk Start** button that builds
//! μ(E) for every staged file and reduces them.
//!
//! It lists **raw scan files only**: the `.xmu`/`.chi`/… outputs this app writes
//! back are hidden by [`is_output_file`], and a file whose content does not read
//! as a scan table — an image, a window dump, a JSON manifest — is hidden by
//! [`looks_like_scan`]. A precomputed-μ file is loaded through the same panel
//! using the reading-format Mode "Raw μ". The reading-format mapping applies to
//! *every* staged file at build time, so the app previews the
//! [`representative`](OpenDataBrowser::representative) (the first staged file).

use std::io::Read;
use std::path::{Path, PathBuf};

use eframe::egui;

use crate::file_browser::FileBrowser;
use crate::widgets::file_name_of;

/// The Open-data loader's browse + staging state. The reading-format chooser and
/// the Autobk Start button live on the app (they need
/// [`crate::import::ImportState`] and the build path); this owns only the
/// browser and the raw-scan filter it is listed under.
pub struct OpenDataBrowser {
    /// Whether the loader window is shown.
    pub open: bool,
    /// Folder navigation + the two-pane selection.
    browser: FileBrowser,
}

impl Default for OpenDataBrowser {
    fn default() -> Self {
        Self {
            open: false,
            browser: FileBrowser::new("od"),
        }
    }
}

impl OpenDataBrowser {
    /// Open the loader anchored on `dir` (the Folders Data folder), seeding the
    /// Selected pane with `loaded` — the source files already loaded as groups.
    /// The loader thus shows what is currently loaded (not an empty pane) and
    /// Autobk Start rebuilds only the newly-added files, leaving existing groups
    /// alone. The session's groups are the single source of truth, so re-anchoring
    /// on every open keeps the pane in sync. Re-anchors the folder on every open;
    /// falls back to the working directory when none is set so the window is never
    /// an empty dead-end.
    pub fn open_on(&mut self, dir: Option<PathBuf>, loaded: Vec<PathBuf>) {
        self.open = true;
        self.browser.set_dir(dir);
        self.browser.ensure_dir();
        self.browser.set_staged(loaded);
        self.browser.refresh(raw_scans);
    }

    /// The representative file for the reading-format panel: the first staged
    /// file. All staged files share one column mapping at build time, so the app
    /// previews and seeds columns from this one.
    pub fn representative(&self) -> Option<&PathBuf> {
        self.browser.staged().first()
    }

    /// Whether anything is staged (gates the Autobk Start button).
    pub fn has_staged(&self) -> bool {
        self.browser.has_staged()
    }

    /// Take the staged paths, leaving the set empty (called on Autobk Start).
    pub fn take_staged(&mut self) -> Vec<PathBuf> {
        self.browser.take_staged()
    }

    /// Render the nav bar, sub-folders, and the two-pane staging list. Mutates the
    /// staged set; the caller reads [`representative`](Self::representative) /
    /// [`has_staged`](Self::has_staged) / [`take_staged`](Self::take_staged).
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        if self
            .browser
            .nav_bar(ui, "(no Data folder — set one in the Folders tab)")
        {
            self.browser.refresh(raw_scans);
        }
        ui.separator();
        FileBrowser::pane_headers(ui);
        let _ = self.browser.panes(ui, 180.0, raw_scans);
    }

    /// Render the browser's delete confirmation. Called from the app's window so
    /// the modal is not clipped by the loader's panel.
    pub fn confirm_delete(&mut self, ctx: &egui::Context) {
        let _ = self.browser.confirm_delete(ctx, raw_scans);
    }
}

/// The Open-data listing filter — raw scan files only: not one of this app's
/// own outputs, and content that plausibly holds a scan table. Shared with the
/// Plot Data window's raw-channel items ([`crate::plot_files::GraphItem`]),
/// which list by the same rule.
pub(crate) fn raw_scans(path: &Path) -> bool {
    !is_output_file(&file_name_of(path)) && looks_like_scan(path)
}

/// True when the file's first few KB read as text containing at least one
/// numeric data row — judged by [`feffit::xasdata::is_data_row`], the reader's
/// own tokenization, so this rule cannot drift from what import accepts.
/// Binary files (images, window dumps) and structured text without bare
/// numeric rows (JSON manifests) can never build μ(E), so the loader does not
/// list them. The 8 KB window is generous: beamline headers run a dozen lines.
/// Non-UTF-8 bytes are decoded lossily rather than rejected — CP949 Korean
/// headers are text, not binary.
fn looks_like_scan(path: &Path) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 8192];
    let mut n = 0;
    loop {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(_) => return false,
        }
        if n == buf.len() {
            break;
        }
    }
    if buf[..n].contains(&0) {
        return false;
    }
    let text = String::from_utf8_lossy(&buf[..n]);
    let mut lines: Vec<&str> = text.lines().collect();
    // A full window can cut the final line mid-row; judging a partial line
    // (e.g. a JSON "6539.0," losing its comma) would misclassify.
    if n == buf.len() {
        lines.pop();
    }
    lines.into_iter().any(feffit::xasdata::is_data_row)
}

/// True for the `.xmu` / `.chi` / `.dat` / `.fit` / `.bkg` files this app writes
/// back into the data folder, so the loader lists only raw scan files.
pub(crate) fn is_output_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [".xmu", ".chi", ".dat", ".fit", ".bkg"]
        .iter()
        .any(|ext| lower.ends_with(ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_output_file_hides_our_outputs_keeps_raw_scans() {
        // Our written outputs are hidden.
        assert!(is_output_file("sample.xmu"));
        assert!(is_output_file("samplek.chi"));
        assert!(is_output_file("RDF.dat"));
        assert!(is_output_file("fit.FIT"));
        assert!(is_output_file("samplee.bkg"));
        // Raw scan files (numeric extensions) are kept.
        assert!(!is_output_file("PGT_Mn_S2_NCM50_W_Insitu.000"));
        assert!(!is_output_file("scan.001"));
        assert!(!is_output_file("data.txt"));
    }

    #[test]
    fn looks_like_scan_keeps_tables_drops_binaries_and_manifests() {
        let d = std::env::temp_dir().join(format!("xafsview_od_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("mkdir");

        let scan = d.join("scan.000");
        std::fs::write(
            &scan,
            "1 Data were taken at HFXAFS\n2 Description :\n6539.0 100000 50000\n",
        )
        .expect("w");
        assert!(raw_scans(&scan));

        // EUC-KR header bytes are text, not binary — the lossy decode keeps
        // the numeric rows below them visible.
        let kr = d.join("korean.000");
        let mut bytes = vec![0xC7, 0xD1, 0xB1, 0xDB, b'\n'];
        bytes.extend_from_slice(b"6539.0 100.0\n");
        std::fs::write(&kr, &bytes).expect("w");
        assert!(raw_scans(&kr));

        let json = d.join("batchload.json");
        std::fs::write(
            &json,
            "{\n  \"files\": [\"a.000\"],\n  \"n\": [6539.0, 1.0]\n}\n",
        )
        .expect("w");
        assert!(!raw_scans(&json), "no bare numeric row");

        let png = d.join("shot.png");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").expect("w");
        assert!(!raw_scans(&png), "NUL bytes mark it binary");

        let xmu = d.join("scan.xmu");
        std::fs::write(&xmu, "6539.0 1.0\n").expect("w");
        assert!(
            !raw_scans(&xmu),
            "our own output stays hidden even as a table"
        );

        let _ = std::fs::remove_dir_all(&d);
    }
}
