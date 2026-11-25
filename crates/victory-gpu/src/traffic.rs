//! Per-resource traffic of a frame, for the frame budget: which title memory a
//! texture refresh reads, or a copy brings up to date, and how much of it turned
//! out to have changed.
//!
//! With `VICTORY_TRAFFIC=1` (diagnostics builds), every refresh and copy notes
//! its address and its bytes, and every few seconds the report names the ten
//! that moved most, biggest first, with the share of their bytes that the
//! comparison found identical. An upload whose bytes are identical to the last
//! one is work a hash skip can remove; one that is mostly identical is a copy
//! that could upload only the pages that changed.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// What moved: a texture refresh reads its whole memory, a copy reads what the
/// write watch called stale.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    Texture,
    Copy,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Texture => "texture",
            Kind::Copy => "copy",
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Counts {
    /// The resource's own size, for the share.
    size: u64,
    /// Bytes read from title memory, and of those the bytes found changed.
    read: u64,
    changed: u64,
    /// Refreshes, so a resource read once is not confused with one read every frame.
    looks: u64,
}

static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static TOTALS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

fn on() -> bool {
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAFFIC").is_some_and(|v| v != "0"))
}

static SEEN: Mutex<Option<HashMap<(Kind, usize), Counts>>> = Mutex::new(None);

/// `read` bytes of `address` were brought up to date, `changed` of them really
/// different.
pub(crate) fn note(kind: Kind, address: usize, size: u64, read: u64, changed: u64) {
    if !on() {
        return;
    }
    TOTALS[0].fetch_add(read, Ordering::Relaxed);
    TOTALS[1].fetch_add(changed, Ordering::Relaxed);
    TOTALS[2].fetch_add(1, Ordering::Relaxed);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let counts = seen.get_or_insert_with(HashMap::new).entry((kind, address)).or_default();
    counts.size = size;
    counts.read += read;
    counts.changed += changed;
    counts.looks += 1;
}

/// Logs the window's traffic, biggest first, and starts a new window. `frames`:
/// the frames since the last call, to divide by.
pub(crate) fn report(frames: f64) {
    if !on() {
        return;
    }
    let frames = frames.max(1.0);
    let read = TOTALS[0].swap(0, Ordering::Relaxed) as f64 / frames;
    let changed = TOTALS[1].swap(0, Ordering::Relaxed) as f64 / frames;
    let looks = TOTALS[2].swap(0, Ordering::Relaxed) as f64 / frames;
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut map) = seen.take() else {
        return;
    };
    let mut rows: Vec<((Kind, usize), Counts)> = map.drain().collect();
    rows.sort_unstable_by(|a, b| b.1.read.cmp(&a.1.read));
    let top: Vec<String> = rows
        .iter()
        .take(10)
        .map(|((kind, at), c)| {
            let unchanged = c.read.saturating_sub(c.changed) as f64 * 100.0 / c.read.max(1) as f64;
            let kind = kind.name();
            let size = c.size as f64 / 1e6;
            let read = c.read as f64 / frames / 1e6;
            let changed = c.changed as f64 / frames / 1e6;
            let looks = c.looks;
            format!(
                "{kind} {at:#x} {size:.1} MB, read {read:.1} MB a frame, {changed:.1} MB of it                  changed, {unchanged:.0} % identical, {looks} reads"
            )
        })
        .collect();
    tracing::info!(
        "texture and copy traffic: {:.1} MB read a frame, {:.1} MB of it changed, in {looks:.0} looks; \
         the biggest first: {}",
        read / 1e6,
        changed / 1e6,
        top.join("; ")
    );
}
