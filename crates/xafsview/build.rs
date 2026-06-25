//! Embed the Windows executable icon as a resource.
//!
//! On a Windows host build, compile `assets/icon.ico` into the `.exe` so Windows
//! Explorer and the taskbar show it as the file icon. This is the executable's
//! *file* icon, distinct from the runtime window icon set in `main.rs`.
//!
//! A no-op on every other host: the resource compile is toolchain-specific (it
//! needs the MSVC/GNU resource compiler) and only a Windows build produces a
//! `.exe` to carry the icon, so macOS/Linux builds are unaffected. The
//! `winresource` build-dependency is itself gated to a Windows host in
//! `Cargo.toml`, matching the `#[cfg(windows)]` guard below.

fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        if let Err(e) = res.compile() {
            // A cosmetic resource must not fail the whole build; warn instead.
            println!("cargo:warning=could not embed the Windows .exe icon: {e}");
        }
    }
}
