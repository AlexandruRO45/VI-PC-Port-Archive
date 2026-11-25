//! `VICTORY_BVH_SKIP_REHYDRATE=1` (diagnostics): the title's calls to the
//! rehydrate entry point of the ray-tracing structure builder DLL it ships
//! are marked, so the GPU layer can drop the builder stages they record.
//!
//! That entry point writes nothing itself: it records builder stages into
//! the command list it is given, and those stages expand the structures in
//! place. With the builder on (`VICTORY_BVH_BUILDER=1`), their results
//! overwrite parts of the structures the title reads back on the CPU, and
//! the title crashes (`docs/fork-ray-tracing.md`). Dropping them, and only
//! them, keeps the title running with the builder's other stages on.
//!
//! The shim forwards every argument unchanged; it only sets
//! `victory_core::bvh::REHYDRATING` on the calling thread for the call.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Whether `VICTORY_BVH_SKIP_REHYDRATE=1`.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_SKIP_REHYDRATE").is_some_and(|v| v == "1"))
}

/// The DLL's own rehydrate entry point, behind our shim.
static REAL: AtomicUsize = AtomicUsize::new(0);

/// The shim for export `name` of the builder DLL whose own address is
/// `real`, or `real` itself for any other export.
pub(crate) fn shim(name: &str, real: usize) -> usize {
    if !name.contains("Rehydrate") {
        return real;
    }
    REAL.store(real, Ordering::Relaxed);
    tracing::info!("the builder DLL's rehydrate entry point is marked: the stages it records are dropped");
    rehydrate as extern "system" fn(usize, usize, u32, usize, usize) -> i32 as usize
}

extern "system" fn rehydrate(this: usize, list: usize, count: u32, descs: usize, extra: usize) -> i32 {
    // SAFETY: the DLL's own entry point, with the arguments it was given.
    let real: extern "system" fn(usize, usize, u32, usize, usize) -> i32 =
        unsafe { std::mem::transmute(REAL.load(Ordering::Relaxed)) };
    victory_core::bvh::while_rehydrating(|| real(this, list, count, descs, extra))
}
