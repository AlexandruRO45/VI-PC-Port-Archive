//! State shared between the loader and the GPU layer about the title's
//! ray-tracing structure builder DLL (diagnostics).
//!
//! The DLL's rehydrate entry point writes nothing itself: it records the
//! builder's GPU stages into the command list it is given. The loader's
//! shim of that entry point (`victory-loader` `bvhcalls`) sets
//! [`REHYDRATING`] on the calling thread for the call's length, so the GPU
//! layer can tell the stages recorded by a rehydrate from those of a build.

use std::cell::Cell;

thread_local! {
    /// Whether this thread is inside a call to the builder DLL's rehydrate
    /// entry point.
    pub static REHYDRATING: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread is recording a rehydrate's stages.
pub fn rehydrating() -> bool {
    REHYDRATING.with(Cell::get)
}

/// Runs `f` with [`REHYDRATING`] set on this thread.
pub fn while_rehydrating<T>(f: impl FnOnce() -> T) -> T {
    let was = REHYDRATING.with(|r| r.replace(true));
    let out = f();
    REHYDRATING.with(|r| r.set(was));
    out
}
