//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Time in the kernel's write-watch answers, in the view check of the
    /// guard, in protecting; bytes and time of copies of memory neither
    /// watched nor guarded (copied whole).
    KernelUs,
    ViewQueryUs,
    ProtectUs,
    WholeBytes,
    WholeUs,
    /// Snapshots of ranges shaders write (bytes, time); page faults while
    /// preparing submissions and while copying back.
    SnapshotBytes,
    SnapshotUs,
    FaultsPrep,
    FaultsBack,
    /// Pages marked changed because a write landed on another mapping of
    /// them.
    AliasMarks,
    /// Time in the guard handler; protection changes making pages read-only
    /// (calls, pages) and giving write access back (pages).
    HandlerUs,
    ProtectCalls,
    ProtectPages,
    OpenPages,
    /// Tracked pages written (distinct, per frame), and of those the ones
    /// written the frame before too.
    WrittenPages,
    WrittenAgain,
    /// In change checks: time asking the runtime for other mappings, and
    /// waiting for the guard lock.
    AliasUs,
    GuardWaitUs,
    /// Preparing submissions: refresh batches done (copies and change
    /// checks); time in each kind of step (a step that flushes the batch
    /// counts that too); texture uploads for heaps; state preludes.
    FlushUs,
    StepRefreshUs,
    StepCopyUs,
    StepWorkUs,
    StepGpuWriteUs,
    StepTextureUs,
    StepHeapUs,
    StepMemoryUs,
    UploadsUs,
    PreludeUs,
    /// Copying back: bytes of copies compared with their snapshots, and
    /// the time.
    CompareBytes,
    CompareUs,
    /// Copying back pages shaders marked as written through views: pages
    /// compared, bytes written back, the time; pages that kept the title's
    /// bytes (it wrote them, their bytes from before unknown); pages whose
    /// bytes the title's first write saved.
    MarkedPages,
    MarkedBytes,
    MarkedUs,
    MarkedKept,
    SavedPages,
}
const COUNTERS: usize = 57;

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;

struct Stats {
    count: [AtomicU64; COUNTERS],
    ranges: Mutex<Option<Ranges>>,
    snapshots: Mutex<Option<HashMap<(u64, u64), u64>>>,
    faults: AtomicU64,
    hard: AtomicU64,
}

impl Stats {
    const fn new() -> Self {
        Self {
            count: [const { AtomicU64::new(0) }; COUNTERS],
            ranges: Mutex::new(None),
            snapshots: Mutex::new(None),
            faults: AtomicU64::new(0),
            hard: AtomicU64::new(0),
        }
    }

    fn add(&self, c: C, n: u64) {
        self.count[c as usize].fetch_add(n, Relaxed);
    }

    fn range(&self, src: usize, len: usize, us: u64, kind: u32) {
        let mut map = self.ranges.lock().unwrap_or_else(|e| e.into_inner());
        let e = map
            .get_or_insert_with(HashMap::new)
            .entry((src, len))
            .or_insert((0, 0, 0, kind));
        e.0 += 1;
        e.1 += len as u64;
        e.2 += us;
    }

    fn snapshot(&self, start: u64, len: u64, bytes: u64) {
        *self
            .snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(HashMap::new)
            .entry((start, len))
            .or_default() += bytes;
    }

    fn log(&self, frames: f64) {
        let frames = if frames.is_finite() && frames > 0.0 {
            frames
        } else {
            1.0
        };
        let faults = page_faults();
        let faults = faults - self.faults.swap(faults, Relaxed).min(faults);
        let c: Vec<f64> = self
            .count
            .iter()
            .map(|a| a.swap(0, Relaxed) as f64 / frames)
            .collect();
        let mb = |v: f64| v / 1e6;
        let ms = |v: f64| v / 1e3;
        tracing::info!(
            "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
             {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
             {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
             unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
             changed, {:.1} MB hashed, {:.1} MB of them changed; watch time: kernel {:.1} ms, view queries {:.1} ms, protecting {:.1} ms; copied whole (not tracked) {:.1} MB in {:.1} ms; snapshots {:.1} MB in {:.1} ms; page faults {:.0} preparing, {:.0} copying back; {:.0} tracked pages marked through another mapping; \
             guards: {:.2} ms in the handler, {:.0} protect calls for {:.0} pages, {:.0} pages opened; \
             {:.0} tracked pages written ({:.0} of them the frame before too); change checks: {:.1} ms \
             asking for other mappings, {:.1} ms waiting for the guard lock; preparing: {:.1} ms in \
             refresh batches, steps (each with the batch it flushes): refresh {:.1} ms, copy {:.1} ms, \
             work {:.1} ms, GPU writes {:.1} ms, textures {:.1} ms, heaps {:.1} ms, title memory {:.1} ms; \
             heap texture uploads {:.1} ms, state preludes {:.1} ms; copying back compared {:.1} MB          in {:.1} ms; marked pages: {:.0} compared, {:.1} MB written back in {:.1} ms, {:.0} kept the title's, {:.0} saved; working set {:.0} MB, \
             {:.0} hard faults",
            c[C::Submissions as usize],
            c[C::Copies as usize],
            mb(c[C::CopyBytes as usize]),
            c[C::Refreshes as usize],
            mb(c[C::RefreshBytes as usize]),
            c[C::Skipped as usize],
            mb(c[C::SkippedBytes as usize]),
            c[C::Runs as usize],
            ms(c[C::RunUs as usize]),
            c[C::Queries as usize],
            ms(c[C::QueryUs as usize]),
            ms(c[C::MoveUs as usize]),
            mb(c[C::Unmapped as usize]),
            faults as f64 / frames,
            c[C::Watches as usize],
            ms(c[C::WatchUs as usize]),
            mb(c[C::Checked as usize]),
            mb(c[C::Unchanged as usize]),
            c[C::Fallbacks as usize],
            c[C::GuardFaults as usize],
            mb(c[C::TexWatched as usize]),
            mb(c[C::TexHashed as usize]),
            mb(c[C::TexChanged as usize]),
            ms(c[C::KernelUs as usize]),
            ms(c[C::ViewQueryUs as usize]),
            ms(c[C::ProtectUs as usize]),
            mb(c[C::WholeBytes as usize]),
            ms(c[C::WholeUs as usize]),
            mb(c[C::SnapshotBytes as usize]),
            ms(c[C::SnapshotUs as usize]),
            c[C::FaultsPrep as usize],
            c[C::FaultsBack as usize],
            c[C::AliasMarks as usize],
            ms(c[C::HandlerUs as usize]),
            c[C::ProtectCalls as usize],
            c[C::ProtectPages as usize],
            c[C::OpenPages as usize],
            c[C::WrittenPages as usize],
            c[C::WrittenAgain as usize],
            ms(c[C::AliasUs as usize]),
            ms(c[C::GuardWaitUs as usize]),
            ms(c[C::FlushUs as usize]),
            ms(c[C::StepRefreshUs as usize]),
            ms(c[C::StepCopyUs as usize]),
            ms(c[C::StepWorkUs as usize]),
            ms(c[C::StepGpuWriteUs as usize]),
            ms(c[C::StepTextureUs as usize]),
            ms(c[C::StepHeapUs as usize]),
            ms(c[C::StepMemoryUs as usize]),
            ms(c[C::UploadsUs as usize]),
            ms(c[C::PreludeUs as usize]),
            mb(c[C::CompareBytes as usize]),
            ms(c[C::CompareUs as usize]),
            c[C::MarkedPages as usize],
            mb(c[C::MarkedBytes as usize]),
            ms(c[C::MarkedUs as usize]),
            c[C::MarkedKept as usize],
            c[C::SavedPages as usize],
            working_set() as f64 / 1e6,
            {
                let hard = hard_faults();
                (hard - self.hard.swap(hard, Relaxed).min(hard)) as f64 / frames
            },
        );
        let held: Vec<String> = crate::watch::held_times()
            .iter()
            .map(|(line, total, longest)| {
                format!("watch.rs:{line} {:.1} ms (longest {longest:.1})", total / frames)
            })
            .collect();
        tracing::info!("  guard lock held per frame: {}", held.join(", "));
        let mut snapshots: Vec<((u64, u64), u64)> = self
            .snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .into_iter()
            .flatten()
            .collect();
        snapshots.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
        let snapshots: Vec<String> = snapshots
            .iter()
            .take(6)
            .map(|((start, len), bytes)| {
                format!("{start:#x}+{len:#x} {:.1} MB", *bytes as f64 / 1e6 / frames)
            })
            .collect();
        tracing::info!("  snapshots per frame by copy: {}", snapshots.join(", "));
        let ranges = self.ranges.lock().unwrap_or_else(|e| e.into_inner()).take();
        let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
        ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
        for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
            use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
            let kind = info.Type;
            tracing::info!(
                "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
                n as f64 / frames,
                bytes as f64 / 1e6 / frames,
                us as f64 / 1e3 / frames
            );
        }
    }
}

static STATS: Stats = Stats::new();

pub(crate) fn add(c: C, n: u64) {
    STATS.add(c, n);
}

fn parse_mode(value: Option<&str>) -> u8 {
    match value {
        Some("plain") => 1,
        Some("none") => 2,
        _ => 0,
    }
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("VICTORY_COPY_MODE").ok().as_deref()))
}

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    STATS.range(src, len, us, kind);
}

/// A snapshot of `bytes` of the copy of `[start, start + len)`.
pub(crate) fn snapshot(start: u64, len: u64, bytes: u64) {
    STATS.snapshot(start, len, bytes);
}

/// Our process's working set, bytes.
fn working_set() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.WorkingSetSize as u64
}

/// Our process's hard page faults so far (read from disk or the page
/// file), from the system's process list: `HardFaultCount` at +0x10 of
/// each entry, the process id at +0x50.
fn hard_faults() -> u64 {
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(class: u32, info: *mut u8, len: u32, ret: *mut u32) -> i32;
    }
    const PROCESSES: u32 = 5;
    let id = std::process::id() as usize;
    let mut buf = vec![0u8; 4 << 20];
    let mut ret = 0;
    // SAFETY: a buffer of that length.
    if unsafe { NtQuerySystemInformation(PROCESSES, buf.as_mut_ptr(), buf.len() as u32, &mut ret) } < 0 {
        return 0;
    }
    let mut at = 0usize;
    loop {
        let entry = &buf[at..];
        let next = u32::from_le_bytes(entry[0..4].try_into().unwrap_or_default()) as usize;
        let pid = usize::from_le_bytes(entry[0x50..0x58].try_into().unwrap_or_default());
        if pid == id {
            return u64::from(u32::from_le_bytes(entry[0x10..0x14].try_into().unwrap_or_default()));
        }
        if next == 0 || at + next + 0x58 > buf.len() {
            return 0;
        }
        at += next;
    }
}

pub(crate) fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    STATS.log(frames);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;
    use std::sync::Arc;

    struct LineCollector(Arc<Mutex<Vec<String>>>);

    struct MsgVisitor(String);

    impl tracing::field::Visit for MsgVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
            if field.name() == "message" {
                use fmt::Write;
                let _ = write!(self.0, "{value:?}");
            }
        }
    }

    impl tracing::Subscriber for LineCollector {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            let mut visitor = MsgVisitor(String::new());
            event.record(&mut visitor);
            self.0.lock().unwrap().push(visitor.0);
        }

        fn enter(&self, _: &tracing::span::Id) {}

        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[test]
    fn counter_enum_matches_array_length() {
        assert_eq!(C::SavedPages as usize + 1, COUNTERS);
        assert_eq!(STATS.count.len(), COUNTERS);
    }

    #[test]
    fn parse_mode_maps_known_and_unknown_values() {
        assert_eq!(parse_mode(None), 0);
        assert_eq!(parse_mode(Some("")), 0);
        assert_eq!(parse_mode(Some("plain")), 1);
        assert_eq!(parse_mode(Some("none")), 2);
        assert_eq!(parse_mode(Some("PLAIN")), 0);
        assert_eq!(parse_mode(Some("guarded")), 0);
    }

    #[test]
    fn range_and_snapshot_accumulate_and_log_guards_invalid_frames() {
        let stats = Stats::new();
        let buf = [0u8; 64];
        let base = buf.as_ptr() as usize;
        stats.range(base, 32, 10, 1);
        stats.range(base, 32, 25, 1);
        stats.range(base + 32, 16, 5, 2);
        {
            let guard = stats.ranges.lock().unwrap();
            let map = guard.as_ref().expect("ranges initialized");
            assert_eq!(map.get(&(base, 32)), Some(&(2, 64, 35, 1)));
            assert_eq!(map.get(&(base + 32, 16)), Some(&(1, 16, 5, 2)));
        }

        stats.snapshot(0x1000, 0x2000, 512);
        stats.snapshot(0x1000, 0x2000, 256);
        stats.snapshot(0x4000, 0x1000, 128);
        {
            let guard = stats.snapshots.lock().unwrap();
            let map = guard.as_ref().expect("snapshots initialized");
            assert_eq!(map.get(&(0x1000, 0x2000)), Some(&768));
            assert_eq!(map.get(&(0x4000, 0x1000)), Some(&128));
        }

        let lines = Arc::new(Mutex::new(Vec::new()));
        let subscriber = LineCollector(lines.clone());
        tracing::subscriber::with_default(subscriber, || {
            for bad in [0.0, -3.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                stats.add(C::Submissions, 4);
                stats.add(C::Copies, 8);
                stats.add(C::CopyBytes, 2_000_000);
                stats.add(C::SavedPages, 3);
                stats.range(base, 32, 100, 1);
                stats.snapshot(0x1000, 0x2000, 1_000_000);
                stats.log(bad);

                assert!(stats.count.iter().all(|c| c.load(Relaxed) == 0));
                assert!(stats.ranges.lock().unwrap().is_none());
                assert!(stats.snapshots.lock().unwrap().is_none());
            }
            log(0.0);
        });

        let captured = lines.lock().unwrap();
        assert!(!captured.is_empty());
        for line in captured.iter() {
            assert!(!line.contains("NaN"), "emitted NaN in line: {line}");
            assert!(!line.contains("inf"), "emitted inf in line: {line}");
        }
    }

    #[test]
    fn process_memory_queries_succeed() {
        assert!(working_set() > 0);
        assert!(page_faults() > 0);
        let _ = hard_faults();
    }
}
