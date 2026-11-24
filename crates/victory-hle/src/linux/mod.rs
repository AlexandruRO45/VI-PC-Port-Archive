//! Our own implementation, on Linux, of the Windows DLLs that the host
//! serves on Windows ([`crate::HostPolicy`]): `kernel32`, `ntdll`, the UCRT,
//! `ws2_32` and the rest. See `docs/linux.md`.
//!
//! One file per DLL family and slice (`kernel32_sync.rs`, `ucrt_math.rs`),
//! each with a `module()` listed in [`modules`]. Every function is
//! `extern "win64"` and named exactly like the export; an export by
//! ordinal is listed under the name `#<ordinal>`.

use std::sync::LazyLock;

use crate::HleModule;

#[allow(dead_code)] // used by the slices as they land
pub(crate) mod handles;
mod kernel32_process;
mod kernel32_sync;
mod kernel32_system;
mod kernel32_thread;
mod kernel32_wait;

fn modules() -> Vec<HleModule> {
    vec![
        kernel32_wait::module(),
        kernel32_sync::module(),
        kernel32_process::module(),
        kernel32_system::module(),
        kernel32_thread::module(),
    ]
}

static MODULES: LazyLock<Vec<HleModule>> = LazyLock::new(modules);

/// Our Linux function for `dll!name`, if we have one.
pub(crate) fn lookup(dll: &str, name: &str) -> Option<usize> {
    MODULES
        .iter()
        .filter(|m| m.serves(dll))
        .flat_map(|m| m.exports.iter())
        .find(|e| e.name == name)
        .map(|e| e.address)
}

/// As [`lookup`], for an import by ordinal.
pub(crate) fn lookup_ordinal(dll: &str, ordinal: u16) -> Option<usize> {
    lookup(dll, &format!("#{ordinal}"))
}
