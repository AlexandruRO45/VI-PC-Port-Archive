//! The D3D12 Agility SDK: a D3D12 runtime (`D3D12Core.dll`) that a process
//! loads from `D3D12\` beside its executable instead of the system's own.
//!
//! Everything this crate emits is shader model 6.6, and the world needs mesh
//! shaders on top. Windows 10's own runtime stops at 6.5 whatever the GPU and
//! driver can do, so there the translated pipelines are refused. The system
//! loader (Windows 10 1909 and later) switches to the SDK's runtime when the
//! process's executable exports [`SDK_VERSION`] as `D3D12SDKVersion` and a
//! relative path as `D3D12SDKPath`, and that folder holds a runtime of that
//! version. `cargo xtask agility` fetches it and puts it beside the build
//! outputs; on Windows 11 the system runtime suffices and the folder may be
//! missing (the loader then keeps the system's).

/// `D3D12SDKVersion` of the runtime `cargo xtask agility` fetches.
pub const SDK_VERSION: u32 = 620;
/// The NuGet version (`Microsoft.Direct3D.D3D12`) that carries it.
pub const PACKAGE_VERSION: &str = "1.620.2";

/// Defines the two exports the system's D3D12 loader reads from the
/// executable. Invoke it once in every crate that builds an executable which
/// creates D3D12 devices (the `victory` binary; the unit-test harnesses that
/// do), and have that crate's `build.rs` pass `/EXPORT:D3D12SDKVersion,DATA`
/// and `/EXPORT:D3D12SDKPath,DATA` to the linker: Rust does not export
/// symbols from an executable by itself.
#[macro_export]
macro_rules! agility_exports {
    () => {
        #[allow(non_upper_case_globals)]
        #[unsafe(no_mangle)]
        #[used]
        pub static D3D12SDKVersion: u32 = $crate::agility::SDK_VERSION;
        /// A `const char *`, relative to the executable, ending in a separator.
        #[allow(non_upper_case_globals)]
        #[unsafe(no_mangle)]
        #[used]
        pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";
    };
}

/// Where the process's D3D12 runtime was loaded from, once a device exists:
/// `...\D3D12\D3D12Core.dll` with the SDK, the System32 one without.
#[cfg(windows)]
pub fn loaded_runtime() -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
    let name: Vec<u16> = "D3D12Core.dll\0".encode_utf16().collect();
    // SAFETY: a NUL-terminated name; the buffer length is passed with it.
    unsafe {
        let module = GetModuleHandleW(name.as_ptr());
        if module.is_null() {
            return None;
        }
        let mut buf = [0u16; 1024];
        let n = GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) as usize;
        (n > 0).then(|| std::ffi::OsString::from_wide(&buf[..n]).into())
    }
}

/// Whether [`loaded_runtime`] is the SDK's (not the system's) runtime.
#[cfg(windows)]
pub fn sdk_runtime_loaded() -> bool {
    loaded_runtime().is_some_and(|p| {
        p.parent()
            .and_then(|d| d.file_name())
            .is_some_and(|d| d.eq_ignore_ascii_case("D3D12"))
    })
}

#[cfg(test)]
crate::agility_exports!();
