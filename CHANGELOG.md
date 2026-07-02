# Changelog

All notable changes to **XAFSView** (the `xafsview` application) are recorded
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the release version tracks the `xafsview` binary. The `feffit` library crate is
versioned independently.

## [0.1.3] - 2026-07-02

### FEFF backend

- **The bundled FEFF8.5L (`feff8l`) external pipeline is now the default backend**
  in the Feff tab. The rpath-fixed `feff8l_*` executables are vendored for
  Linux, macOS, and Windows and resolved automatically next to the app (or via
  `FEFF8L_DIR`), so no separate FEFF install is needed.
- **The in-process FEFF10 backend is disabled in the Feff tab.** Its prebuilt is
  broken on Linux (writes no `feffNNNN.dat`) and Windows (Fortran `lrstat` abort)
  — upstream `Ameyanagi/feff10-rs#1`. The backend code stays in the tree; only
  the UI option is greyed out.

### Added

- Feffit: per-path fit parameters (σ²/N/cumulants), `guess`/`set` terminology,
  add standard variables by name, "Use fit as guess" with Undo.
- Feffit: import a UWXAFS `feffit.inp` ("Load inp"); Save result / Load result /
  Open log; the "View …" result-report viewers; a Fit-mode dropdown
  (fit / no fit / only FT); the K+Q graph item overlaying kʷ·χ(k) and χ(q).
- Feffit Run and the Feffit batch overlay every fit at once (matching AUTOBK's
  "Show all groups").
- Reduce pipeline: reverse FT filling χ(q) (`xftr_group`); an R-window band with
  a Fourier-filtered `*Q.CHI` view; a draggable FT k-window on the kʷ·χ(k) plot;
  a toggleable draggable k/R range band on the Feffit and Autobk graphs.
- Reduce: exposed the post-edge normalization order (Nnorm); manual pre/norm
  ranges now default to XAFSView's (-200/-50/150/400).
- Autobk: auto-run on file load and after a parameter edit; Start auto-runs
  Calc XMU when μ(E) isn't built yet; a "Show background" toggle; a stacking
  waterfall; a loaded-data list with select/remove in the sidebar.
- The original's nine Graph-type views; AUTOBK background files (`e.bkg`/`k.bkg`);
  "Sub base folder" that creates the five working folders.
- Plot Data pulled into its own detached window with a staging picker for
  "Make μ(E) from files"; the `*.result` and `*.bkg` file types; a normalized-μ
  "norm" item; click-a-legend-entry highlight; a wide-table "Save in single file".
- UWXAFS byte-format `.chi`/`.dat`/`.fit`/`.bkg` column files, each with a
  provenance header.
- Multi-file "Open New file" (builds every selected file); an editable
  header-skip override for the reading format.
- The loaded-group manager moved into a detached "Data" window; a shared
  checked-group selection so each tab's Run acts on the same groups.
- UI: graph click pins the nearest curve's name and data point; the selected /
  emphasized curve and its legend row are highlighted; Ctrl/Cmd+A selects all in
  the file lists; shift-range selection for the batch group list.
- A Settings menu with adjustable UI scale.

### Fixed

- Re-enabled eframe `x11`/`wayland` so the Linux build finds a winit backend.
- Bumped siplot 0.3 → 0.4.1 to stop momentum re-zooming a reset view.
- Detached pop-ups reopen after being closed (fresh `ViewportId`).
- Load raw files with CP949/EUC-KR (non-UTF-8) headers.
- Batch "Save χ data+fit" writes to the Feffit folder; the output folder is
  created on demand; `sample.NNN` scans stay distinct in batch outputs.
- feffit resolves `feff8l_*` executables with the platform exe suffix
  (`feff8l_pot.exe` on Windows) — in the library and in the integration-test
  helper, so the Windows CI gate genuinely runs the pipeline instead of
  self-skipping.
- Default the Fit-mode dropdown to "Only FT"; default the Plot Data pickers to
  the Data / Results folders.

### CI

- Added Windows and macOS pull-request gates and wired the vendored `feff8l`
  binaries into all three (Linux / Windows / macOS), so the default backend is
  exercised on every platform — each gate reproduces the Cu first shell
  (reff = 2.5527 Å).

## [0.1.2] - 2026-06-25

Earlier releases predate this changelog; see the GitHub release notes.

## [0.1.1] - 2026-06-25

See the GitHub release notes.

[0.1.3]: https://github.com/physwkim/feffit-rs/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/physwkim/feffit-rs/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/physwkim/feffit-rs/releases/tag/0.1.1
