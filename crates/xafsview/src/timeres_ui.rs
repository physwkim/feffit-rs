//! The **Extract XAS measured time** window (그림 1-3-10, Tools menu): tabulate
//! the measurement start/finish times of a time-resolved (quick-XAFS) file
//! series, then save the table as a single file.
//!
//! Pick the file series, list one row per file with its start and finish time
//! (hours from the first scan), and "Save as a single file". The timestamps
//! come from the `Date and Time : <date> <start> ~ <finish>` field the PLS-II
//! 10C DAQ writes into every file's first header line — see
//! [`extract_measured_time`].

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::Path;

use eframe::egui;

/// One time-resolved file's table row.
struct TimeRow {
    /// The file's display name (its file name).
    name: String,
    /// Absolute measurement `(start, finish)` timestamps in hours, parsed from
    /// the file header ([`extract_measured_time`]); `None` when the file has no
    /// readable `Date and Time` field.
    abs: Option<(f64, f64)>,
}

/// **Extract XAS measured time** (그림 1-3-10): a table of one row per file in a
/// time-resolved series — `File name | starting time (hr) | finish time (hr)` —
/// rebased so the first scan starts at `0`, with "Save as a single file". The
/// timestamps come from each file's header (see [`extract_measured_time`]); a
/// file without one shows `—`.
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
        // The save action is pinned under the (growing) table; read before the
        // closures since the pinned row renders first and cannot borrow `self`
        // while the body holds it.
        let has_rows = !self.rows.is_empty();
        let status = self.status.clone();
        let mut do_save = false;
        crate::window::detached_with_action(
            ctx,
            "time_resolved",
            "Extract XAS measured time",
            &mut keep_open,
            [500.0, 380.0],
            |ui| self.body(ui),
            |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(has_rows, egui::Button::new("Save as a single file"))
                        .clicked()
                    {
                        do_save = true;
                    }
                    if !status.is_empty() {
                        ui.weak(&status);
                    }
                });
            },
        );
        if do_save {
            self.save_single_file();
        }
        if !keep_open {
            self.open = false;
        }
    }

    /// The window contents: pick/clear controls and the time table (the save
    /// action is the window's pinned action row).
    fn body(&mut self, ui: &mut egui::Ui) {
        ui.label(
            "Start / finish times from each file's \"Date and Time\" header, in \
             hours from the first scan.",
        );
        ui.separator();
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
    }

    /// Replace the table with the picked file series, extracting each file's
    /// measurement timestamps from its header.
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
        let n = self.rows.len();
        self.status = if parsed == n {
            format!("{n} file(s) listed.")
        } else {
            format!(
                "{n} file(s) listed; {} without a readable \"Date and Time\" header (shown as —).",
                n - parsed
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
/// time-resolved data file's header: the `Date and Time : 2026-03-11 15:15:24 ~
/// 15:32:00` field of the PLS-II 10C first header line (tab-separated
/// `key : value` fields). `None` when the file cannot be read or carries no
/// such field. Only the file's first bytes are read — the field sits in the
/// first line and a fly-scan file runs to megabytes.
fn extract_measured_time(path: &Path) -> Option<(f64, f64)> {
    const HEADER_BYTES: u64 = 8192;
    let mut head = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(HEADER_BYTES)
        .read_to_end(&mut head)
        .ok()?;
    parse_date_and_time(&String::from_utf8_lossy(&head))
}

/// The `(start, finish)` of a header's `Date and Time : <date> <start> ~
/// <finish>` field as absolute hours (days since 1970-01-01 × 24 + clock).
/// `<finish>` is a bare clock on the same day, or a full `<date> <clock>`; a
/// bare finish earlier than the start is a scan that crossed midnight.
fn parse_date_and_time(header: &str) -> Option<(f64, f64)> {
    let after_key = header.split_once("Date and Time")?.1;
    let value = after_key.split_once(':')?.1;
    // The field ends at the next tab-separated header field or the line end.
    let field = value.split(['\t', '\r', '\n']).next()?.trim();
    let (start_s, finish_s) = field.split_once('~')?;
    let (date, clock) = start_s.trim().split_once(' ')?;
    let day = civil_days(date.trim())?;
    let start = day as f64 * 24.0 + clock_hours(clock.trim())?;
    let finish_s = finish_s.trim();
    let finish = match finish_s.split_once(' ') {
        Some((d2, c2)) => civil_days(d2.trim())? as f64 * 24.0 + clock_hours(c2.trim())?,
        None => {
            let same_day = day as f64 * 24.0 + clock_hours(finish_s)?;
            if same_day < start {
                same_day + 24.0
            } else {
                same_day
            }
        }
    };
    Some((start, finish))
}

/// `HH:MM:SS` (or `HH:MM`) as hours.
fn clock_hours(clock: &str) -> Option<f64> {
    let mut parts = clock.split(':');
    let h: f64 = parts.next()?.trim().parse().ok()?;
    let m: f64 = parts.next()?.trim().parse().ok()?;
    let s: f64 = match parts.next() {
        Some(s) => s.trim().parse().ok()?,
        None => 0.0,
    };
    if parts.next().is_some() || !(0.0..24.0).contains(&h) || !(0.0..60.0).contains(&m) {
        return None;
    }
    Some(h + m / 60.0 + s / 3600.0)
}

/// Days from 1970-01-01 to a `YYYY-MM-DD` date (proleptic Gregorian; the
/// days-from-civil algorithm), so dates subtract as whole days.
fn civil_days(date: &str) -> Option<i64> {
    let mut parts = date.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let d: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = (m + 9) % 12; // March = 0
    let doy = (153 * u64::from(mp) + 2) / 5 + u64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe as i64 - 719_468)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first header line of a 10C step-scan file, verbatim.
    const HEADER_10C: &str = "Data were taken at HFXAFS in Pohang Light Source(PLS-II) by : \
        bl10c\tNumber of Points : 459\tScanning Mode : Step Scan (Transmission)\t\
        Date and Time : 2026-03-11 15:15:24 ~ 15:32:00\tEnergy Origin(Eo) : 6539.000 (eV)\t\
        Mono Offset(deg) : -0.006681\tScanning Time : 16.599 (min)\t\n\
        Slit size (normal) : 0.500 x 5.000\n";

    #[test]
    fn parses_the_10c_date_and_time_field() {
        let (start, finish) = parse_date_and_time(HEADER_10C).expect("field present");
        // 2026-03-11 is day 20523 since the epoch; 15:15:24 and 15:32:00.
        assert_eq!(civil_days("2026-03-11"), Some(20_523));
        assert!((start - (20_523.0 * 24.0 + 15.0 + 15.0 / 60.0 + 24.0 / 3600.0)).abs() < 1e-9);
        assert!((finish - (20_523.0 * 24.0 + 15.0 + 32.0 / 60.0)).abs() < 1e-9);
        // 16 min 36 s of scanning, as the header's own "Scanning Time : 16.599".
        assert!(((finish - start) * 60.0 - 16.6).abs() < 1e-6);
    }

    #[test]
    fn finish_clock_before_start_crossed_midnight_and_a_dated_finish_is_taken_as_is() {
        let (s, f) =
            parse_date_and_time("Date and Time : 2026-03-11 23:50:00 ~ 00:10:00\t").unwrap();
        assert!(((f - s) * 60.0 - 20.0).abs() < 1e-6);
        let (s, f) =
            parse_date_and_time("Date and Time : 2026-03-11 23:50:00 ~ 2026-03-12 00:10:00\n")
                .unwrap();
        assert!(((f - s) * 60.0 - 20.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_headers_without_a_usable_field() {
        assert_eq!(parse_date_and_time("Number of Points : 459\n"), None);
        assert_eq!(
            parse_date_and_time("Date and Time : \tEnergy Origin : 1\n"),
            None
        );
        assert_eq!(
            parse_date_and_time("Date and Time : 2026-13-01 10:00:00 ~ 11:00:00"),
            None
        );
        assert_eq!(clock_hours("25:00:00"), None);
    }

    #[test]
    fn extract_reads_the_file_head() {
        let dir = std::env::temp_dir().join(format!("xafsview_timeres_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p = dir.join("scan.000");
        std::fs::write(&p, HEADER_10C).expect("write");
        let (s, f) = extract_measured_time(&p).expect("parsed");
        assert!(((f - s) * 60.0 - 16.6).abs() < 1e-6);
        assert_eq!(extract_measured_time(&dir.join("missing.001")), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

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
        // Unreadable files parse to None, and the table then shows no times.
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
