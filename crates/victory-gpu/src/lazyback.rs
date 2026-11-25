//! Lazy copy-back (`VICTORY_LAZY_BACK`, off by default): pages of title
//! memory that only the card's shaders write are left on the card instead
//! of being merged into title memory at every landing. See
//! `docs/lazy-copyback.md`.
//!
//! Step 1 (`VICTORY_LAZY_BACK=1`): bookkeeping only. Every piece the
//! copy-back fetches is noted per 64 KiB page of title memory; a page that
//! nothing on the CPU's side touched for [`QUIET_FRAMES`] frames is a
//! *candidate* (it would be card-only). Our own reads and writes of title
//! memory through the kernel (`crate::host`) count as touches, except the
//! copy-back's own ([`merging`]). Nothing is skipped and nothing is
//! protected; the frame line says what step 2 would skip and how often a
//! candidate would have been materialized.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Pages of title memory as the copy-back cuts them.
const PAGE: usize = 64 << 10;

/// A page untouched on the CPU's side for this many frames is a candidate.
pub(crate) const QUIET_FRAMES: u64 = 60;

/// The step `VICTORY_LAZY_BACK` asks for; 0: off.
pub(crate) fn step() -> u8 {
    static STEP: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *STEP.get_or_init(|| {
        let step = std::env::var("VICTORY_LAZY_BACK").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0).min(1);
        if step > 0 {
            tracing::info!(step, "lazy copy-back: noting the pages only the card writes, nothing skipped yet (VICTORY_LAZY_BACK)");
        }
        step
    })
}

#[derive(Clone, Copy, Default)]
struct Page {
    /// The frame a fetch last brought it; 0: never.
    fetched: u64,
    /// The frame something on the CPU's side last touched it; 0: never.
    touched: u64,
}

impl Page {
    fn candidate(&self, frame: u64) -> bool {
        self.touched == 0 || frame >= self.touched + QUIET_FRAMES
    }
}

static PAGES: Mutex<Option<HashMap<usize, Page>>> = Mutex::new(None);

/// Since the last frame line: pieces fetched, those whose every page was a
/// candidate, their bytes; touches of candidates (each one a materialize
/// in step 2), touches of other noted pages.
static COUNT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

thread_local! {
    /// Set while this thread runs the copy-back's merge: its own reads and
    /// writes of title memory are not touches.
    static MERGING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` as the copy-back's merge ([`MERGING`]).
pub(crate) fn merging<R>(f: impl FnOnce() -> R) -> R {
    let was = MERGING.with(|m| m.replace(true));
    let r = f();
    MERGING.with(|m| m.set(was));
    r
}

/// The copy-back is about to fetch title memory `[from, to)` (a piece).
pub(crate) fn fetched(from: usize, to: usize) {
    if step() == 0 || from >= to {
        return;
    }
    let frame = crate::frame::current().max(1);
    let mut pages = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let pages = pages.get_or_insert_with(HashMap::new);
    let mut all = true;
    for page in from / PAGE..to.div_ceil(PAGE) {
        let p = pages.entry(page).or_default();
        all &= p.fetched != 0 && p.candidate(frame);
        p.fetched = frame;
    }
    COUNT[0].fetch_add(1, Relaxed);
    if all {
        COUNT[1].fetch_add(1, Relaxed);
        COUNT[2].fetch_add((to - from) as u64, Relaxed);
    }
}

/// Our own read or write of title memory `[at, at + len)` through the
/// kernel, outside the copy-back's merge: a touch of the noted pages there.
pub(crate) fn touched(at: usize, len: usize) {
    if step() == 0 || len == 0 || MERGING.with(std::cell::Cell::get) {
        return;
    }
    let frame = crate::frame::current().max(1);
    let mut pages = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pages) = pages.as_mut() else { return };
    for page in at / PAGE..(at + len).div_ceil(PAGE) {
        if let Some(p) = pages.get_mut(&page) {
            COUNT[if p.candidate(frame) { 3 } else { 4 }].fetch_add(1, Relaxed);
            p.touched = frame;
        }
    }
}

/// With the frame line.
pub(crate) fn line(frames: f64) {
    if step() == 0 {
        return;
    }
    let [pieces, skipped, bytes, candidates, others] = [0, 1, 2, 3, 4].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    let (noted, now) = {
        let frame = crate::frame::current().max(1);
        let pages = PAGES.lock().unwrap_or_else(|e| e.into_inner());
        pages.as_ref().map_or((0, 0), |p| (p.len(), p.values().filter(|p| p.fetched != 0 && p.candidate(frame)).count()))
    };
    tracing::info!(
        "lazy copy-back, noting only (VICTORY_LAZY_BACK=1), per frame: {pieces:.0} pieces fetched, {skipped:.0} of them all on \
         card-only candidates ({:.1} MB step 2 would not merge); our own touches of candidates {candidates:.1} (each a \
         materialize), of other noted pages {others:.1}; {noted} pages noted, {now} candidates now",
        bytes / (1 << 20) as f64
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule on one page, by its counters: fetched twice and untouched
    /// is a candidate; a touch of ours (not the merge's) takes it out.
    #[test]
    fn untouched_fetched_pages_are_candidates_and_a_touch_is_not_the_merges() {
        let page = Page::default();
        assert!(page.candidate(1), "never touched");
        let touched = Page { fetched: 10, touched: 10 };
        assert!(!touched.candidate(10 + QUIET_FRAMES - 1));
        assert!(touched.candidate(10 + QUIET_FRAMES));
        // The merge's own accesses are not touches.
        assert!(!MERGING.with(std::cell::Cell::get));
        let inside = merging(|| MERGING.with(std::cell::Cell::get));
        assert!(inside);
        assert!(!MERGING.with(std::cell::Cell::get));
    }
}
