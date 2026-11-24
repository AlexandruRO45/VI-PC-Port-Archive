//! Modules the driver's shader compiler faults on.
//!
//! `CreateComputePipelineState` and `CreateGraphicsPipelineState` sometimes end
//! the process inside `amdxc64.dll+0x85fd0f` (a read of `[rcx+8]`, `rcx = 0`),
//! and the module that causes it determines the fault, not the description and
//! not concurrency: one dump line naming `compute:a1266cc7391ae0d6` faults on
//! the first create a fresh process attempts, while thousands of others create
//! in the same process. Once such a module is known, the session should refuse
//! its draws rather than die at it, which is what this list is for: the
//! `victory poison` prober creates each module in a child process and records
//! the ones that killed it, and the runtime consults the list before a create.
//!
//! An entry is a statement about the DXIL this build emits, so the file carries
//! the build it was written for and the adapter it was written on, and a file
//! from another build is ignored rather than obeyed.

use std::collections::HashSet;
use std::sync::OnceLock;

/// (stage, hash) pairs the compiler faulted on, when a list was written for this
/// build and this adapter.
fn set() -> &'static Option<HashSet<(String, String)>> {
    static SET: OnceLock<Option<HashSet<(String, String)>>> = OnceLock::new();
    SET.get_or_init(|| {
        let dir = super::library::cache_dir()?;
        let path = dir.join(file_name());
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                // Named, not silent: a list in the wrong directory reads as a
                // run with no list at all, which is how a list once went
                // unnoticed while the scene kept dying.
                tracing::info!(
                    path = %path.display(),
                    error = %e,
                    "poisoned modules: no list here, every module is created"
                );
                return None;
            }
        };
        let build = build_key()?;
        let mut pairs = HashSet::new();
        let mut mine = false;
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("# build=") {
                mine = rest.trim() == build;
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let mut fields = line.split_whitespace();
            let (Some(stage), Some(hash)) = (fields.next(), fields.next()) else { continue };
            pairs.insert((stage.to_owned(), hash.to_owned()));
        }
        if !mine {
            tracing::info!(
                path = %path.display(),
                pairs = pairs.len(),
                build = %build,
                "poisoned modules: not this build's list, ignored"
            );
            return None;
        }
        tracing::info!(
            path = %path.display(),
            pairs = pairs.len(),
            "modules this build refuses to create"
        );
        Some(pairs)
    })
}

/// `poisoned-modules.txt`, or `poisoned-modules-warp.txt` on the software
/// adapter: a module one adapter's compiler faults on need not be one the
/// other refuses.
fn file_name() -> &'static str {
    if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
        "poisoned-modules-warp.txt"
    } else {
        "poisoned-modules.txt"
    }
}

/// The build a poison list belongs to: the first 16 hex of the executable's
/// own hash. The prober writes the same value, so a list from an older binary
/// is ignored instead of hiding modules a rebuild no longer faults on.
pub(crate) fn build_key() -> Option<String> {
    static KEY: OnceLock<Option<String>> = OnceLock::new();
    KEY.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let bytes = std::fs::read(exe).ok()?;
        Some(blake3::hash(&bytes).to_hex()[..16].to_owned())
    })
    .clone()
}

/// Whether this build's list refuses `stage`/`hash`.
pub(crate) fn poisoned(stage: &str, hash: &str) -> bool {
    set()
        .as_ref()
        .is_some_and(|s| s.contains(&(stage.to_owned(), hash.to_owned())))
}

/// Draws dropped because a module they needed is refused.
pub(crate) fn drops() -> usize {
    DROPS.load(std::sync::atomic::Ordering::Relaxed)
}

static DROPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Counts one refusal, logging the first of each module.
pub(crate) fn note_refusal(stage: &str, hash: &str) {
    DROPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.iter().any(|k| k == &format!("{stage}:{hash}")) {
        seen.push(format!("{stage}:{hash}"));
        tracing::warn!(
            stage,
            hash,
            "a module this build's compiler faults on: its pipelines are refused, not created"
        );
    }
}
