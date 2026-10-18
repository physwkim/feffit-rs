# Changelog

All notable changes to **XAFSView** (the `xafsview` application) are recorded
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the release version tracks the `xafsview` binary. The `feffit` library crate is
versioned independently.

## [0.3.0] - 2026-10-18

### Added

- **MCP tools for every window and tab** — 30 new tools beside the 0.2.0
  set, each calling the function its button or menu item calls, so an agent
  can drive and check the whole GUI without synthetic mouse input:
  `open_window` / `close_window` / `set_theme`; `plot_data` and
  `feffit_graph` for the Plot Data window; `folders`, `data`, `open_data`,
  `theory`, `run_multi_autobk`, `set_path`, `feffit_file`, `feffit_batch`,
  `build_feff_inp`, `run_feff` and `plot_sites` for the tabs; `xanes`,
  `mback`, `lcf`, `pca`, `wavelet`, `edit_mu`, `time_resolved`,
  `periodic_table`, `ion_chamber`, `powder`, `ke_convert`, `log` and `about`
  for the Tools windows; and `graph`, which makes the mouse gestures on any
  1-D plot (zoom, box zoom, pan, click, band drag, legend click). `set_graph`
  / `feffit_graph` gain `show_range`, `insitu_map` gains `scan` / `x` for a
  map-cell click, and `edit_mu remove_point` falls back to the pinned pick.
- **`screenshot(window=…)`** captures an open pop-up window. eframe 0.36.2 is
  vendored with a one-hunk patch for it: its immediate viewports dropped the
  screenshot command.
- **Korean quick-start manual** with screenshots, `doc/manual-ko.md`, linked
  from the README.

### Changed

- **rsplot 0.6.1**: per-curve storage-buffer bindings (DX12/WARP read every
  draw's uniform block from the last one bound, so overlaid curves came out
  in one colour) and target-encoding colour fixes that brighten the Linux
  rendering to the Color32 values.
- **Solid scroll bars** in every theme: a height-capped list (the legend,
  results footers, Plot Data's lists, the import and browser lists) shows
  its bar whenever it overflows, as Qt does, instead of only on hover.
- Side panels keep their width: their controls sit in a both-axis scroll
  area, so a row that cannot wrap scrolls instead of widening the panel every
  frame; the Autobk panel opens at 460 px and the In-situ reference combo
  truncates to the room left.
- An MCP session no longer expires after five idle minutes.

### Fixed

- `feffit`: a k-space fit with data ending below the fit's kmax (an AUTOBK
  kmax under the Feffit one) panicked with index out of bounds; a k-range
  holding no data, or fewer points than variables, is now an error in every
  fit space instead of a panic or NaN statistics; `pre_edge` and
  `mback_norm` return an error on an empty normalization range instead of
  panicking.
- Run FEFF failed on Windows with a Config error: the FEFF10 stage-worker
  hook now runs first in `main`.
- MCP path arguments are respelled natively, so a forward-slash Windows path
  no longer travels into the session as `C:/…`.
- The Open-data window's Autobk Start writes the AUTOBK files (`k.chi`,
  `r.chi`, `e.bkg`, `k.bkg`) per reduced group, as the Autobk tab's Start
  does and the manual promises.
- With "auto pre/norm" on, the disabled Pre1/Pre2/Nor1/Nor2/Nnorm fields
  show the ranges the reduction used; leaving auto seeds the manual fields
  from them.
- Legend labels elide around their distinguishing core — the prefix and
  suffix all rows share are cut on token boundaries — and a lone fit's rows
  keep the label stem.
- Plot Data: a fit sent from the Feffit tab is drawn only on its own x axis
  and its item decides the axes of an otherwise empty window; each overlaid
  Feffit group gets two colours of its own; the canvas follows the dark/light
  theme.
- In-situ map: the profile axes are labelled from the map, not the last
  curve.
- An Edit μ(E) edit re-reduces the current group, so E0 and the k/R graphs
  stay live instead of blank until the next AUTOBK run.
- Open New file raises the loader window when it is already open behind the
  main window.
- An MCP screenshot of a pop-up could wait forever on a slow GPU (wgpu runs
  the readback callback only on a later GPU submit): the app keeps
  repainting while a shot is pending and fails one after five seconds.
- The `graph` tool refuses every non-graph window with one message.

## [0.2.0] - 2026-10-16

### Added

- **In-situ map window** (Tools › In-situ map, or the Autobk tab's new Analysis
  group), ported from the BL10C_XAFS PyMca app: a scan series as a 2-D map of
  RDF |χ(R)| or XANES μ(E) (norm or flat), built from the loaded groups or a
  legacy series `.dat`, with reference subtraction, click/drag profile
  extraction with a marker line, one-click **Re-reduce & build**, an
  OriginPro-ready map export, and **Fit peaks** — PyMca's Specfit "Simple Fit"
  on the profile (area Gaussians on a linear background, Mariscotti peak
  search, every FWHM tied to the highest peak's).
- **Embedded MCP server** (`--mcp [port]` or a Settings toggle; streamable
  HTTP on 127.0.0.1) for AI-agent control, with load_data (one file or a
  batch), set_params, run_autobk, run_fit (fit / nofit / only_ft),
  get_results, screenshot, add_path, select_groups, remove_groups,
  remove_paths, set_graph, set_tab, set_zoom, deglitch and insitu_map tools.
- **FEFF10** in-process backend enabled and made the **default** (feff10 ≥
  0.2.3); FEFF8L stays selectable. The runner normalises the PRINT card so
  FEFF10 always writes the path files.
- **Automatic glitch detection**: `xasdata::find_glitches` / `auto_deglitch`
  (7-point rolling median, MAD-sigma threshold); Edit μ(E) gains **Find**,
  **Remove** and a batch over the checked groups, plus a "use clicked" button
  that seeds the remove-point energy from the plot pick.
- **Plot Data**: a raw scan file type (intensity ratios and bare channels
  against energy, with an optional clamp of non-positive counts), **Add loaded
  groups** to overlay in-memory groups, and the shared file browser (Open data
  and Plot Data) with Up / sub-folder navigation and **Delete** behind a
  confirmation dialog. Legends past 15 entries start folded to an N-curves
  header.
- **Settings and working folders persist** across runs (font size, dark/light,
  the five folders, window geometry); the MCP toggle stays per-run.
- Numeric **axis-limits row** behind a Limits toolbar toggle on every plot; all
  eight FT windows in the window combos; a live **feff.inp preview** on the
  Atoms tab; "Go to Feffit" on the empty Fit report tab; shift-click range
  check and move up/down in the Data window; empty-state hints on curveless
  plots and the Feff editor.
- `xasdata`: column roles match parenthesised channel names (`Ch1 (I0)`,
  `Ch2 (IT)`), label lines split on tab or comma before whitespace, and an
  opt-in **clamp of non-positive monitor counts** (`MonitorPolicy`; a checkbox
  on the reading-format chooser, off by default).

### Changed

- GUI stack on **egui/eframe 0.36** and **rsplot 0.6** (wgpu 30); the
  `xafsview` binary needs Rust 1.95. Forms use rsplot's native form helpers at
  desktop density: group boxes span the panel, numeric fields share one width,
  primary buttons take the theme accent, and plots follow the dark/light
  toggle.
- Layout: status bar at the window bottom, About under a Help menu, the
  Feffit_txt tab renamed **Fit report** (MCP set_tab keeps the alias), the
  Atoms tab in two columns, Autobk Start and Feffit Run pinned below their
  scrolling forms, detached windows scroll with pinned action rows.
- The AUTOBK reduction runs on the background job; replots diff the declared
  curves against the retained ones (selecting 66 groups: 243 ms → 26 ms);
  kʷ·χ graphs autoscale y over the FT k-window; the graph title names the
  focused group.
- `feffit()` rejects varied parameters no path references (as UWXAFS feffit
  does) instead of returning NaN uncertainties.
- Fonts: DejaVu Sans and the first readable system CJK face join the fallback
  chain, so arrows, box-drawing glyphs and Hangul paths render.

### Fixed

- Undo restores the group its snapshot came from; the Autobk Output file,
  manual E0, Edit μ(E) energies and the XANES/MBACK/LCF/PCA inputs re-seed
  when the current or picked group changes; MBACK's edge guess prefers K, then
  L3, within 25 eV.
- KeConvert took a second square root in E→k; "deg of free" shows n_idp −
  nvarys; "Use fit as guess" adopts the per-path locals; the Feffit Results
  box names the fitted group; the Feffit batch runs over the session's checked
  set.
- Plot Data: peak markers no longer accumulate across rebuilds; the status
  line and peak table follow every trace-set change; the Feffit overlay
  honours G_kweight and Same height; only energy-overlapping groups overlay on
  energy axes; the view resets when the last curve is removed.
- Legend viewport sizing, overlay-text clipping, status-line truncation, combo
  popup height, detached-window footers, raw listings that require a scan
  table, series headers with a trailing tab, and the 10C "Date and Time"
  header in the measured-time window.

## [0.1.5] - 2026-07-06

### Changed

- Plotting migrated from `siplot` to **`rsplot` 0.5.0** (the renamed successor;
  identical API and pins).
- **Batch operations run on a background job** with a spinner + `done/total`
  progress in the status strip — Multiple AUTOBK, batch μ(E) (Open data →
  Autobk Start), Feffit Run, and the Feffit batch "Run all" no longer freeze
  the UI. One job at a time; results apply on the main thread when done, and
  the Feffit editor stays usable while a fit runs.
- **Session log**: the status strip keeps only the latest line, but every
  message now also lands in a cumulative Log window (button on the strip, with
  an error-count badge). Batch failures log each file's reason instead of a
  bare counter.
- Feffit paths are a selectable list (enable box, label, reff per row) instead
  of the one-at-a-time "Path index" spinner.
- Feffit results show a **stale badge** when any fit input (paths, variables,
  transform, functions, mode) changed since the shown fit.
- Clearer action labels: Atoms "Execute" → **Build feff.inp**, Feff "Execute" →
  **Run FEFF**, Feffit "Graph type" → **Component**, batch "Show all fits on
  the graph" → **Show all fits**.
- Every detached window carries the standard inner margin (content no longer
  hugs the OS-window edges).
- The Extract-XAS-measured-time window states up front that timestamp parsing
  is not implemented yet (times show "—").

## [0.1.4] - 2026-07-03

### Changed

- Autobk data loading is now one in-app **"Open data"** window: an Up / sub-folder
  browser anchored on the Data folder plus a two-pane staging list (`=>`/`<=`,
  Sort, Clear all) that lists raw scans only. A single **Autobk Start** builds
  μ(E) for every staged file and reduces them. Both "Open New file" and the Data
  window's "Add data files…" open it, replacing the separate "Make μ(E) from
  files" batch picker.
- The Open-data file preview shows the whole file (was capped at a few rows).
- Removed **Calc XMU** and the **Loading** mode; the flow is unified on Autobk
  Start.
- The Open-data Selected pane mirrors the currently loaded groups on open, so
  reopening shows the session; Autobk Start builds only the not-yet-loaded files.

### Fixed

- Autobk and Feffit plots legend each trace by its data / group (file) name, so
  the focused group matches the multi-group overlays instead of showing a bare
  quantity ("μ(E)", "kʷ·χ(k)", …) or "data".

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

[0.3.0]: https://github.com/physwkim/feffit-rs/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/physwkim/feffit-rs/compare/0.1.5...0.2.0
[0.1.5]: https://github.com/physwkim/feffit-rs/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/physwkim/feffit-rs/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/physwkim/feffit-rs/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/physwkim/feffit-rs/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/physwkim/feffit-rs/releases/tag/0.1.1
