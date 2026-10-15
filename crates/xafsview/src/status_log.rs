//! The session status log: the one-line status strip text plus a cumulative
//! message log behind it.
//!
//! The strip shows only the *latest* message, so any earlier failure is
//! overwritten by the next status update, and batch failures used to be
//! aggregated into bare counters ("build errors 2") with the per-file reasons
//! discarded. This module is the single owner of both surfaces: every
//! user-visible outcome goes through [`StatusLog::info`] / [`StatusLog::error`]
//! (which set the strip line *and* append to the log), and batch loops add the
//! per-file reasons with [`StatusLog::detail_error`] (append-only, so the
//! summary line stays readable). The "Log…" button on the strip opens the
//! detached window with the whole session's messages.

use eframe::egui;

/// Severity of one log entry — errors render highlighted in the window.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Error,
}

/// One logged message.
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
}

/// Keep the log bounded: a beamline session can emit thousands of per-file
/// messages; the oldest fall off past this many entries.
const MAX_ENTRIES: usize = 1000;

/// The status strip line + the cumulative session log (see the module docs).
pub struct StatusLog {
    /// The one-line text the status strip shows (the latest summary message).
    line: String,
    /// Every message of the session, oldest first, capped at [`MAX_ENTRIES`].
    entries: Vec<LogEntry>,
    /// Whether the detached Log window is open.
    pub open: bool,
}

impl Default for StatusLog {
    fn default() -> Self {
        Self {
            line: "Ready.".to_owned(),
            entries: Vec::new(),
            open: false,
        }
    }
}

impl StatusLog {
    /// Set the strip line and log the message at info level.
    pub fn info(&mut self, message: impl Into<String>) {
        self.set(LogLevel::Info, message.into());
    }

    /// Set the strip line and log the message as an error.
    pub fn error(&mut self, message: impl Into<String>) {
        self.set(LogLevel::Error, message.into());
    }

    /// Log a per-item failure reason *without* replacing the strip line — batch
    /// loops call this once per failed file and then set one summary line.
    pub fn detail_error(&mut self, message: impl Into<String>) {
        self.push(LogLevel::Error, message.into());
    }

    /// The text the status strip shows.
    pub fn line(&self) -> &str {
        &self.line
    }

    /// How many error entries the session has accumulated (shown on the
    /// strip's Log button so failures stay visible after the line moves on).
    pub fn error_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.level == LogLevel::Error)
            .count()
    }

    fn set(&mut self, level: LogLevel, message: String) {
        self.line.clone_from(&message);
        self.push(level, message);
    }

    fn push(&mut self, level: LogLevel, message: String) {
        self.entries.push(LogEntry { level, message });
        if self.entries.len() > MAX_ENTRIES {
            let excess = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(..excess);
        }
    }

    /// The detached Log window: the session's messages oldest-first (errors
    /// highlighted), pinned to the newest, with a Clear button.
    pub fn show_window(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        let mut keep_open = true;
        // Own panel layout: the message list fills the window (`auto_shrink`
        // off, pinned to the newest), which a scrolled `detached` body cannot
        // host, so the header goes in a top panel of its own.
        crate::window::detached_panels(
            ctx,
            "status_log",
            "Log",
            &mut keep_open,
            [560.0, 420.0],
            |ui| {
                egui::Panel::top("status_log_header").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "{} message(s), {} error(s)",
                            self.entries.len(),
                            self.error_count()
                        ));
                        if ui.button("Clear").clicked() {
                            self.entries.clear();
                        }
                    });
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    let err_color = ui.visuals().error_fg_color;
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for e in &self.entries {
                                match e.level {
                                    LogLevel::Error => {
                                        ui.colored_label(err_color, &e.message);
                                    }
                                    LogLevel::Info => {
                                        ui.label(&e.message);
                                    }
                                }
                            }
                            if self.entries.is_empty() {
                                ui.weak("No messages yet.");
                            }
                        });
                });
            },
        );
        self.open = self.open && keep_open;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_and_error_set_the_line_and_accumulate() {
        let mut log = StatusLog::default();
        assert_eq!(log.line(), "Ready.");
        log.info("loaded 3 files");
        assert_eq!(log.line(), "loaded 3 files");
        log.error("Open failed: nope");
        assert_eq!(log.line(), "Open failed: nope");
        // Both messages are retained even though the line moved on.
        assert_eq!(log.entries.len(), 2);
        assert_eq!(log.error_count(), 1);
    }

    #[test]
    fn detail_error_logs_without_touching_the_line() {
        let mut log = StatusLog::default();
        log.detail_error("a.dat: unreadable");
        log.detail_error("b.dat: no columns");
        log.info("Batch: built 1, 2 failed");
        assert_eq!(log.line(), "Batch: built 1, 2 failed");
        assert_eq!(log.entries.len(), 3, "details + summary all logged");
        assert_eq!(log.error_count(), 2);
    }

    #[test]
    fn log_is_capped_at_max_entries() {
        let mut log = StatusLog::default();
        for i in 0..(MAX_ENTRIES + 25) {
            log.detail_error(format!("f{i}: err"));
        }
        assert_eq!(log.entries.len(), MAX_ENTRIES);
        // The oldest fell off; the newest survived.
        assert_eq!(log.entries.last().unwrap().message, "f1024: err");
        assert!(log.entries.first().unwrap().message.contains("f25"));
    }
}
