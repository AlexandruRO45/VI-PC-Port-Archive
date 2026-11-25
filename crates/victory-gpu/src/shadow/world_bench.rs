//! A world frame's copies of title memory, replayed on a made-up heap:
//! the refreshes, snapshots and copy-back of 36 submissions, timed apart.
//! Its shape follows what world frames were measured to do (#400): one
//! 174 MB heap that every submission declares dozens of overlapping views
//! of as shader-written, refreshes asking for some 20 GB a frame of which
//! a few percent changed, and a few megabytes written by shaders. The
//! numbers are not the title's; they are the same for every build, so two
//! builds can be compared on one machine.
//!
//! It also checks what copy-back must keep (decision 0007, "Copies of title
//! memory"): bytes the GPU wrote reach title memory, and bytes the title
//! wrote while the GPU ran stay as the title left them.
//!
//! Two modes, as `Host::execute` runs them: `world_frame_copies` snapshots
//! each declared range before the submission (memory that is not tracked),
//! `world_frame_copies_marked` arms the ranges and copies back the pages
//! the shaders marked (`Shadow::writing` with `marked`, then
//! [`copy_back_marks`]). Both check the same things, with one exception
//! the marked mode prints instead of asserting: a GPU write on a page that
//! no declared range covers whole, which the title has written since its
//! copy last took it. Such a page is never taken (`Shadow::update` takes
//! whole pages only), so `Shadow::copy_back_marked` cannot tell the
//! shaders' bytes from stale ones and keeps the title's.
//!
//! `cargo test --release -p victory-gpu world_frame_copies -- --ignored --nocapture`

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::*;

/// The heap every submission declares views of.
const HEAP: usize = 174 << 20;
const SUBMISSIONS: usize = 36;
/// Ranges each submission declares as shader-written (and refreshes before).
const DECLARED: usize = 24;
/// The views of the heap those ranges are taken from (passes declare the
/// same ranges again and again).
const VIEWS: usize = 64;
/// Pages the title's threads write between two submissions.
const TITLE_PAGES: usize = 256;
/// Places in the declared ranges the GPU writes per submission.
const GPU_WRITES: usize = 4;
/// The first frame takes the whole heap and fills the pools; not counted.
const FRAMES: usize = 6;

/// Xorshift, seeded the same in every build.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A length between `lo` and `hi`, as often in each power of two.
    fn length(&mut self, lo: usize, hi: usize) -> usize {
        let (a, b) = (lo.ilog2(), hi.ilog2());
        let bits = a + self.below((b - a) as usize) as u32;
        (1 << bits) + self.below(1 << bits)
    }

    /// A range of the heap at `base`, starting on 256 bytes.
    fn range(&mut self, base: usize, lo: usize, hi: usize) -> (usize, usize) {
        let len = self.length(lo, hi).min(HEAP - 256);
        let at = self.below((HEAP - len) / 256) * 256;
        (base + at, len)
    }
}

#[derive(Default)]
struct Frame {
    refresh: Duration,
    refreshed: usize,
    asked: usize,
    snapshots: Duration,
    copy_back: Duration,
    written: usize,
    /// GPU writes on pages no declared range covered whole that did not
    /// reach title memory (marked mode).
    edge_lost: usize,
}

fn time<T>(total: &mut Duration, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let r = f();
    *total += started.elapsed();
    r
}

fn put(at: usize, value: u64) {
    // SAFETY: an aligned address inside an allocation of the test's.
    unsafe { (at as *mut u64).write(value) }
}

fn get(at: usize) -> u64 {
    // SAFETY: as above.
    unsafe { (at as *const u64).read() }
}

#[test]
#[ignore = "benchmark: needs a host D3D12 device and ~600 MB; run in release with --ignored --nocapture"]
fn world_frame_copies() {
    replay(false);
}

#[test]
#[ignore = "benchmark: needs a host D3D12 device and ~600 MB; run in release with --ignored --nocapture"]
fn world_frame_copies_marked() {
    replay(true);
}

/// Sets the written-pages bit of title address `at`, as a shader writing
/// through a view of `shadow` does.
fn mark(shadow: &Shadow, at: usize) {
    let (bit, _) = shadow
        .mark_entry(at as u64)
        .expect("a tracked copy marks its views' writes");
    let words = bitmap().expect("the written-pages bitmap").data as *mut u32;
    // SAFETY: a dword of the mapped bitmap (`mark_entry` gives bits of it);
    // nothing else runs on the GPU in this test.
    unsafe { *words.add(bit as usize / 32) |= 1 << (bit % 32) };
}

fn replay(marked: bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    if crate::host::host().is_none() {
        eprintln!("no host D3D12 device: skipped");
        return;
    }
    // Title memory, write-watched as the runtime reserves it.
    // SAFETY: a fresh allocation, freed at the end.
    let base = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            HEAP,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
        )
    } as usize;
    assert_ne!(base, 0);
    for at in (base..base + HEAP).step_by(8) {
        put(at, at as u64);
    }
    register(0x5eed_0174, base as u64, HEAP as u64);
    let shadow = find(base as u64, 1).expect("a host copy of the heap");
    assert!(!marked || shadow.marked(), "the heap's copy is not tracked");
    shadow.update(base, HEAP);

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let views: Vec<(usize, usize)> = (0..VIEWS)
        .map(|_| rng.range(base, 64 << 10, 64 << 20))
        .collect();
    let mut stamp = 1u64 << 63;
    let mut frames = Vec::new();
    for _ in 0..FRAMES {
        let mut f = Frame::default();
        for _ in 0..SUBMISSIONS {
            for _ in 0..TITLE_PAGES {
                stamp += 1;
                put(base + rng.below(HEAP / 8) * 8, stamp);
            }
            let declared: Vec<(usize, usize)> =
                (0..DECLARED).map(|_| views[rng.below(VIEWS)]).collect();
            // Each range's draws read some of the heap first; the range
            // itself is refreshed before it is declared, as `gpu_writes`
            // asks (the shaders start from the title's data).
            for &(address, len) in &declared {
                let (read, n) = rng.range(base, 1 << 20, 128 << 20);
                f.asked += n + len;
                f.refreshed += time(&mut f.refresh, || {
                    shadow.update(read, n) + shadow.update(address, len)
                });
                time(&mut f.snapshots, || shadow.writing(address, len, marked));
            }
            // The GPU writes into the declared ranges, while the title
            // writes the eight bytes after each place (same pages).
            let mut gpu = BTreeMap::new();
            let mut title = BTreeMap::new();
            for _ in 0..GPU_WRITES {
                let (address, len) = declared[rng.below(DECLARED)];
                let at = (address + rng.below(len - 16)) & !7;
                if gpu.contains_key(&(at + 8)) || title.contains_key(&at) {
                    continue;
                }
                stamp += 1;
                gpu.insert(at, stamp);
                put(shadow.data as usize + (at - base), stamp);
                if marked {
                    mark(&shadow, at);
                }
                stamp += 1;
                title.insert(at + 8, stamp);
                put(at + 8, stamp);
            }
            // After the submission: the marked pages, then the declared
            // ranges (which disarm the copy).
            if marked {
                time(&mut f.copy_back, || copy_back_marks(&[]));
            }
            let mut once = declared.clone();
            once.sort_unstable();
            once.dedup();
            for (address, len) in once {
                f.written += time(&mut f.copy_back, || shadow.copy_back(address, len));
            }
            for (&at, &value) in &gpu {
                let page = at & !(crate::watch::PAGE - 1);
                let whole = declared
                    .iter()
                    .any(|&(a, l)| a <= page && page + crate::watch::PAGE <= a + l);
                if marked && !whole && get(at) != value {
                    f.edge_lost += 1;
                    continue;
                }
                assert_eq!(
                    get(at),
                    value,
                    "a GPU write at heap offset {:#x} did not reach title memory",
                    at - base
                );
            }
            for (&at, &value) in &title {
                assert_eq!(
                    get(at),
                    value,
                    "the title's write at heap offset {:#x} was overwritten",
                    at - base
                );
            }
        }
        frames.push(f);
    }
    // SAFETY: our allocation; the copy keeps no pointer into it.
    unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mb = |n: usize| n as f64 / f64::from(1 << 20);
    println!(
        "mode: {}",
        if marked { "marked pages" } else { "snapshots" }
    );
    println!(
        "frame | refresh ms (MB copied of GB asked) | {} ms | copy-back ms ({}) | total ms",
        if marked { "arming" } else { "snapshots" },
        // `copy_back_marks` does not say what it wrote.
        if marked {
            "KB written by declared ranges"
        } else {
            "KB written"
        }
    );
    for (i, f) in frames.iter().enumerate().skip(1) {
        println!(
            "{i} | {:.1} ({:.0} of {:.1}) | {:.1} | {:.1} ({:.0}) | {:.1}",
            ms(f.refresh),
            mb(f.refreshed),
            mb(f.asked) / 1024.0,
            ms(f.snapshots),
            ms(f.copy_back),
            mb(f.written) * 1024.0,
            ms(f.refresh + f.snapshots + f.copy_back)
        );
    }
    let mut totals: Vec<f64> = frames[1..]
        .iter()
        .map(|f| ms(f.refresh + f.snapshots + f.copy_back))
        .collect();
    totals.sort_by(f64::total_cmp);
    println!("median total: {:.1} ms per frame", totals[totals.len() / 2]);
    if marked {
        let (lost, writes) = (
            frames.iter().map(|f| f.edge_lost).sum::<usize>(),
            FRAMES * SUBMISSIONS * GPU_WRITES,
        );
        println!(
            "GPU writes kept out of title memory on pages no declared range covers whole: {lost} (of up to {writes})"
        );
    }
}
