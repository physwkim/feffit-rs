//! XAS data session model and (later) beamline file I/O for the `xafsview` GUI.
//!
//! This crate owns the *data* side of the application — independent of any GUI
//! toolkit so it stays lightweight and unit-testable:
//!
//! - [`XasGroup`] — one spectrum end-to-end (raw `mu(E)` → normalize → AUTOBK →
//!   FT), the unit every tab reads and writes.
//! - [`Session`] — the loaded groups, current selection, and working
//!   [`Folders`]; serializable so a whole session can be saved and reloaded.
//!
//! Beamline file reading and `mu(E)` building live in `reader`/`xmu`; the
//! headless [`batch`] drivers (make-xmu, multiple-AUTOBK, average, peak) back the
//! GUI's *Multiple_data* menu and *Plot Data* window without depending on any GUI
//! toolkit, so they stay unit-testable.

pub mod batch;
pub mod clean;
pub mod group;
pub mod reader;
pub mod reduce;
pub mod xmu;

pub use batch::{average_curves, make_xmu_batch, peak_in_range, reduce_all, resample_matrix};
pub use clean::{RangeSide, SmoothForm, deglitch_point, deglitch_range, smooth_mu, trim};
pub use group::{Folders, Session, XasGroup};
pub use reader::{ColumnFile, ReadError, RoleGuess, read_chi_dat};
pub use reduce::{FtParams, autobk_group, normalize, xftf_group};
pub use xmu::{MuSpec, XmuError, build_mu};

// Re-export the engine parameter/window types so downstream crates (the GUI,
// batch tools) can drive reduction through `xasdata` alone.
pub use xafsft::Window;
pub use xasproc::mathutils::{interp_cubic, interp_linear};
pub use xasproc::mback::{Edge, MbackNorm, MbackNormParams, mback_norm};
pub use xasproc::xanes::{arctan_step, centroid, peak, valley, x_at_y};
pub use xasproc::{
    AutobkParams, Lincombo, LincomboParams, PcaFit, PcaModel, PreEdgeParams, groups2matrix,
    lincombo_fit, pca_fit, pca_train,
};
