//! Opts victory.exe into the D3D12 Agility SDK.
//!
//! Translated shaders use SM 6.6 directly indexed heaps (root signature flags
//! `0xC00`), which the D3D12 runtime inside Windows 10 refuses: the host root
//! signature is rejected and every draw is dropped. The Agility SDK's
//! `D3D12Core.dll` adds that support. A program selects it by exporting these
//! two symbols (see `build.rs`); the runtime then loads `D3D12\D3D12Core.dll`
//! from next to the executable. Without that file Windows falls back to its
//! own runtime, so nothing changes where it is absent (Windows 11 needs none).
//!
//! The version is the SDK release the bundled `D3D12Core.dll` comes from
//! (NuGet `Microsoft.Direct3D.D3D12` 1.619.x).

/// A `const char *` the runtime reads through the exported symbol.
#[repr(transparent)]
pub struct CStrPtr(*const u8);

// SAFETY: points at an immutable static string.
unsafe impl Sync for CStrPtr {}

#[unsafe(no_mangle)]
pub static D3D12SDKVersion: u32 = 619;

#[unsafe(no_mangle)]
pub static D3D12SDKPath: CStrPtr = CStrPtr(c".\\D3D12\\".as_ptr().cast());
