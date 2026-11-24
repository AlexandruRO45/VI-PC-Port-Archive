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
pub(crate) mod kernel32_file;
pub(crate) mod kernel32_heap;
mod kernel32_io;
mod kernel32_locale;
mod kernel32_memory;
mod kernel32_path;
mod kernel32_text;

fn modules() -> Vec<HleModule> {
    vec![
        kernel32_file::module(),
        kernel32_heap::module(),
        kernel32_io::module(),
        kernel32_locale::module(),
        kernel32_memory::module(),
        kernel32_path::module(),
        kernel32_text::module(),
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
