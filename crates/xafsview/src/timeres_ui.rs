//! The **Extract XAS measured time** window (그림 1-3-10, Tools menu): tabulate
//! the measurement start/finish times of a time-resolved (quick-XAFS) file
//! series, then save the table as a single file.
//!
//! This is the UI skeleton: pick the file series, list one row per file with its
//! start and finish time (hours from the first scan), and "Save as a single
//! file". The per-file timestamp extraction is a stub — see
//! [`extract_measured_time`] — pending the beamline data-file header format.

use std::fmt::Write as _;
use std::path::Path;

use eframe::egui;

/// One time-resolved file's table row.
struct TimeRow {
    /// The file's display name (its file name).
    name: String,
    /// Absolute measurement `(start, finish)` timestamps in hours, parsed from
    /// the file header — `None` until [`extract_measured_time`] is implemented.
    abs: Option<(f64, f64)>,
}

/// **Extract XAS measured time** (그림 1-3-10): a table of one row per file in a
/// time-resolved series — `File name | starting time (hr) | finish time (hr)` —
/// rebased so the first scan starts at `0`, with "Save as a single file". The
/// timestamps come from each file's header; that parser is not yet implemented
/// (see [`extract_measured_time`]), so the time columns read `—` for now.
#[derive(Default)]
pub struct TimeResolvedWindow {
    /// Whether the window is shown.
    pub open: bool,
    /// The picked files, in pick order.
    rows: Vec<TimeRow>,
    /// A one-line status/result message.
    status: String,
}

impl TimeResolvedWindow {
    /// Render the window.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        // `keep_open` tracks the OS window close button, which closes the window.
        let mut keep_open = true;
        crate::window::detached(
            ctx,
            "time_resolved",
            "Extract XAS measured time",
            &mut keep_open,
            [500.0, 380.0],
            |ui| self.body(ui),
        );
        if !keep_open {
            self.open = false;
        }
    }

    /// The window contents: pick/clear controls, the time table, and save/exit.
    fn body(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button("Pick files…")
                .on_hover_text("Choose the time-resolved data-file series")
                .clicked()
            {
                self.pick_files();
            }
            if !self.rows.is_empty() && ui.button("Clear").clicked() {
                self.rows.clear();
                self.status.clear();
            }
        });
        ui.separator();

        if self.rows.is_empty() {
            ui.weak(
                "No files. Pick a time-resolved file series to tabulate its measurement times.",
            );
        } else {
            let displayed = self.displayed();
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("time_resolved_table")
                    .striped(true)
                    .num_columns(3)
                    .show(ui, |ui| {
                        ui.strong("File name");
                        ui.strong("starting time (hr)");
                        ui.strong("finish time (hr)");
                        ui.end_row();
                        for (name, t) in &displayed {
                            ui.monospace(name);
                            match t {
                                Some((start, finish)) => {
                                    ui.monospace(format!("{start:.6}"));
                                    ui.monospace(format!("{finish:.6}"));
                                }
                                None => {
                                    ui.weak("—");
                                    ui.weak("—");
                                }
                            }
                            ui.end_row();
                        }
                    });
            });
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.rows.is_empty(),
                    egui::Button::new("Save as a single file"),
                )
                .clicked()
            {
                self.save_single_file();
            }
        });
        if !self.status.is_empty() {
            ui.weak(&self.status);
        }
    }

    /// Replace the table with the picked file series, extracting each file's
    /// measurement timestamps (a no-op stub for now).
    fn pick_files(&mut self) {
        let Some(paths) = rfd::FileDialog::new().pick_files() else {
            return;
        };
        self.rows = paths
            .iter()
            .map(|p| TimeRow {
                name: p
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string()),
                abs: extract_measured_time(p),
            })
            .collect();
        let parsed = self.rows.iter().filter(|r| r.abs.is_some()).count();
        self.status = if parsed == 0 {
            format!(
                "{} file(s) listed. Measurement-time extraction is not yet implemented \
                 (data-file header format pending).",
                self.rows.len()
            )
        } else {
            format!(
                "{} file(s) listed, {parsed} with extracted times.",
                self.rows.len()
            )
        };
    }

    /// The rows with their times rebased to hours from the first scan that has a
    /// parsed timestamp (the original's first row starts at `0.000000`). Rows
    /// without a parsed timestamp carry `None`.
    fn displayed(&self) -> Vec<(String, Option<(f64, f64)>)> {
        let t0 = self.rows.iter().find_map(|r| r.abs.map(|(start, _)| start));
        self.rows
            .iter()
            .map(|r| {
                let rel = match (r.abs, t0) {
                    (Some((start, finish)), Some(t0)) => Some((start - t0, finish - t0)),
                    _ => None,
                };
                (r.name.clone(), rel)
            })
            .collect()
    }

    /// Write the table to a single user-chosen text file.
    fn save_single_file(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("measured_time.txt")
            .save_file()
        else {
            return;
        };
        let mut s = String::new();
        s.push_str("# Extract XAS measured time — hours from the first scan\n");
        let _ = writeln!(s, "# {:<24}{:>18}{:>18}", "file", "start_hr", "finish_hr");
        for (name, t) in self.displayed() {
            match t {
                Some((start, finish)) => {
                    let _ = writeln!(s, "  {name:<24}{start:>18.6}{finish:>18.6}");
                }
                None => {
                    let _ = writeln!(s, "  {name:<24}{:>18}{:>18}", "n/a", "n/a");
                }
            }
        }
        match std::fs::write(&path, s) {
            Ok(()) => {
                self.status = format!("Saved {} row(s) to {}.", self.rows.len(), path.display());
            }
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }
}

/// Parse the absolute measurement `(start, finish)` timestamps, in hours, from a
/// time-resolved data file's header.
///
/// **Stub** — always returns `None`. The original extracts these from the
/// timestamp the beamline DAQ writes into each file's header; a faithful port
/// needs that header format (PLS/POSTECH), which is not yet confirmed. Once the
/// format is known, parse the absolute start/finish here and the window's
/// [`displayed`](TimeResolvedWindow::displayed) rebasing + table + save fill in
/// without further UI changes.
fn extract_measured_time(_path: &Path) -> Option<(f64, f64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(rows: Vec<TimeRow>) -> TimeResolvedWindow {
        TimeResolvedWindow {
            open: true,
            rows,
            status: String::new(),
        }
    }

    #[test]
    fn displayed_rebases_to_the_first_parsed_scan() {
        // Times rebase so the earliest parsed start is 0; a None row stays None.
        let w = win(vec![
            TimeRow {
                name: "a.000".into(),
                abs: Some((10.0, 10.5)),
            },
            TimeRow {
                name: "b.001".into(),
                abs: None,
            },
            TimeRow {
                name: "c.002".into(),
                abs: Some((11.0, 11.5)),
            },
        ]);
        let d = w.displayed();
        assert_eq!(d[0].1, Some((0.0, 0.5)));
        assert_eq!(d[1].1, None);
        assert_eq!(d[2].1, Some((1.0, 1.5)));
    }

    #[test]
    fn displayed_all_none_when_no_timestamps_parsed() {
        // With the stub parser (no header support yet), every row is None and the
        // table shows no times.
        let w = win(vec![
            TimeRow {
                name: "a".into(),
                abs: extract_measured_time(Path::new("a")),
            },
            TimeRow {
                name: "b".into(),
                abs: extract_measured_time(Path::new("b")),
            },
        ]);
        assert!(w.displayed().iter().all(|(_, t)| t.is_none()));
    }
}
