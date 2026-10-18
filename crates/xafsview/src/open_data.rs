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

    /// MCP `open_data`: the browser half — Browse to `dir`, stage / unstage
    /// `stage` / `unstage` (absolute paths, or names in the browsed folder),
    /// Clear all. A staged file must exist and pass the raw-scan listing rule,
    /// as it would have to appear in Available Data to be staged by hand.
    pub fn apply_mcp(
        &mut self,
        dir: Option<PathBuf>,
        stage: &[String],
        unstage: &[String],
        clear: bool,
    ) -> Result<(), String> {
        if let Some(d) = dir {
            if !d.is_dir() {
                return Err(format!("{} is not a folder", d.display()));
            }
            self.browser.set_dir(Some(d));
            self.browser.refresh(raw_scans);
        }
        if clear {
            self.browser.clear_staged(raw_scans);
        }
        if !unstage.is_empty() {
            let paths: Vec<PathBuf> = unstage.iter().map(|f| self.resolve(f)).collect();
            self.browser.unstage(&paths, raw_scans);
        }
        if !stage.is_empty() {
            let mut paths = Vec::with_capacity(stage.len());
            for f in stage {
                let p = self.resolve(f);
                if !p.is_file() {
                    return Err(format!("{} is not a file", p.display()));
                }
                if !raw_scans(&p) {
                    return Err(format!(
                        "{} is not a raw scan file (this app's .xmu/.chi/.dat/.fit/.bkg outputs and non-table files are not listed)",
                        p.display()
                    ));
                }
                paths.push(p);
            }
            self.browser.stage(paths, raw_scans);
        }
        Ok(())
    }

    /// A file argument as a path: absolute as given, a bare name against the
    /// browsed folder.
    fn resolve(&self, f: &str) -> PathBuf {
        let p = crate::mcp::native_path(f);
        match (p.is_absolute(), self.browser.dir()) {
            (false, Some(dir)) => dir.join(p),
            _ => p,
        }
    }

    /// The browser as text: the folder, its sub-folders, the Available Data
    /// names and the Selected Data paths.
    pub fn state_text(&self) -> String {
        let names = |paths: &[PathBuf]| -> String {
            if paths.is_empty() {
                "(none)".to_owned()
            } else {
                paths
                    .iter()
                    .map(|p| file_name_of(p))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        let staged = if self.browser.staged().is_empty() {
            "(none)".to_owned()
        } else {
            self.browser
                .staged()
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "folder: {}\nsub-folders: {}\navailable: {}\nstaged: {staged}",
            self.browser
                .dir()
                .map_or_else(|| "(none)".to_owned(), |d| d.display().to_string()),
            names(self.browser.subdirs()),
            names(self.browser.available())
        )
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
    fn mcp_stages_raw_scans_by_name_and_rejects_outputs() {
        let d = std::env::temp_dir().join(format!("xafsview_od_mcp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub")).expect("mkdir");
        std::fs::write(d.join("scan.000"), "# h\n6539.0 100 50\n").expect("w");
        std::fs::write(d.join("scan.xmu"), "6539.0 1.0\n").expect("w");

        let mut od = OpenDataBrowser::default();
        od.apply_mcp(Some(d.clone()), &["scan.000".into()], &[], false)
            .unwrap();
        assert_eq!(od.representative(), Some(&d.join("scan.000")));
        let text = od.state_text();
        assert!(text.contains("sub-folders: sub"), "{text}");
        assert!(text.contains("available: (none)"), "{text}");
        assert!(
            text.contains("staged: ") && text.contains("scan.000"),
            "{text}"
        );

        let err = od
            .apply_mcp(None, &["scan.xmu".into()], &[], false)
            .unwrap_err();
        assert!(err.contains("not a raw scan file"), "{err}");
        let err = od
            .apply_mcp(None, &["missing.000".into()], &[], false)
            .unwrap_err();
        assert!(err.contains("not a file"), "{err}");

        od.apply_mcp(None, &[], &["scan.000".into()], false)
            .unwrap();
        assert!(!od.has_staged());
        assert!(od.state_text().contains("available: scan.000"));
        assert!(
            od.apply_mcp(Some(d.join("nope")), &[], &[], false)
                .unwrap_err()
                .contains("not a folder")
        );
        let _ = std::fs::remove_dir_all(&d);
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
