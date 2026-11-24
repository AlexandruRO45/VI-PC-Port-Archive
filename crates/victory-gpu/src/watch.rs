//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::BTreeMap;
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
});

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
/// Local build: what was asked of the kernel in the current submission: ranges covered (their
/// journal is current), and the extents of private allocations (one asking covers them whole).
struct Cover {
    submission: u64,
    ranges: Vec<(usize, usize)>,
}
static COVER: Mutex<Cover> = Mutex::new(Cover { submission: 0, ranges: Vec::new() });
/// Allocations seen by base: (end, private).
static EXTENTS: Mutex<BTreeMap<usize, (usize, bool)>> = Mutex::new(BTreeMap::new());

/// `VICTORY_EXTENT_CACHE=0`: every asking queries the allocation of its range anew (a
/// `VirtualQuery`, which walks page tables to the region's end: ~2 ms in the title's big pools).
fn extent_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_EXTENT_CACHE").map_or(true, |v| v.trim() != "0"))
}

/// Memory at `[at, at + len)` was mapped or unmapped: the allocations there are unknown again.
fn forget_extents(at: usize, len: usize) {
    let end = at.saturating_add(len);
    EXTENTS.lock().unwrap_or_else(|e| e.into_inner()).retain(|&base, &mut (top, _)| !(base < end && top > at));
}

/// The allocation holding `at`: (base, end, private), cached by base; with the extent cache,
/// found without a system call when `at` lies inside one seen before.
fn allocation(at: usize) -> Option<(usize, usize, bool)> {
    use windows_sys::Win32::System::Memory::{MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if extent_cache()
        && let Some((&base, &(top, private))) =
            EXTENTS.lock().unwrap_or_else(|e| e.into_inner()).range(..=at).next_back()
        && at < top
    {
        return Some((base, top, private));
    }
    let n = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: queries only.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(at as *const _, &mut info, n) } == 0 {
        return None;
    }
    let base = info.AllocationBase as usize;
    if base == 0 {
        return None;
    }
    if let Some(&(top, private)) = EXTENTS.lock().unwrap_or_else(|e| e.into_inner()).get(&base) {
        return Some((base, top, private));
    }
    let private = info.Type == MEM_PRIVATE;
    let mut end = base;
    loop {
        // SAFETY: a query only.
        if unsafe { VirtualQuery(end as *const _, &mut info, n) } == 0 || info.AllocationBase as usize != base {
            break;
        }
        end = info.BaseAddress as usize + info.RegionSize;
    }
    EXTENTS.lock().unwrap_or_else(|e| e.into_inner()).insert(base, (end, private));
    Some((base, end, private))
}

/// Local build: [`writes_kernel`], asked once per submission for what it covers: a private
/// allocation is asked whole (one call instead of the ~1,300 a world frame made on its
/// pieces), and a range already covered in this submission is answered from the journal.
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let sub = crate::shadow::current_submission();
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let (a, b) = (first * PAGE, last * PAGE);
    {
        let mut c = COVER.lock().unwrap_or_else(|e| e.into_inner());
        if c.submission != sub {
            c.submission = sub;
            c.ranges.clear();
        }
        if c.ranges.iter().any(|&(x, y)| x <= a && b <= y) {
            let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            return Some((j.asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()));
        }
    }
    // a private allocation of at most 2 GiB is asked whole
    let (qa, qb) = match if known_view(a) { None } else { allocation(a) } {
        Some((base, top, true)) if top - base <= (2 << 30) && base <= a && b <= top => (base, top),
        _ => (a, b),
    };
    let (now, _) = match writes_kernel(qa, qb, guarded) {
        Some(r) => r,
        // (a cached extent gone stale: ask for the range itself)
        None if (qa, qb) != (a, b) => {
            forget_extents(qa, qb - qa);
            writes_kernel(a, b, guarded)?
        }
        None => return None,
    };
    COVER.lock().unwrap_or_else(|e| e.into_inner()).ranges.push((qa / PAGE * PAGE, qb.div_ceil(PAGE) * PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

fn writes_kernel(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // Local build: memory known to be guarded view memory is not write-watched: no asking.
    let view = guarded && known_view(first * PAGE);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = view || unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        guard(first * PAGE, last * PAGE)?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Per 64 KiB block of view memory, what we know of it (see [`Flags`]).
const PROTECTED: u8 = 1;
const CLEAN: u8 = 2;
const HOLE: u8 = 4;
/// Blocks per chunk of the flag table (256 MB of title memory).
const CHUNK: usize = 4096;

/// Local build: the block flags as a chunked flat table (they were three `BTreeSet`s: the
/// world's askings walked its 2,800-block heap with a few thousand tree lookups each).
#[derive(Default)]
struct Flags {
    chunks: std::collections::HashMap<usize, Box<[u8; CHUNK]>>,
}

impl Flags {
    fn get(&self, block: usize) -> u8 {
        self.chunks.get(&(block / CHUNK)).map_or(0, |c| c[block % CHUNK])
    }
    fn set(&mut self, block: usize, bits: u8) {
        self.chunk(block)[block % CHUNK] |= bits;
    }
    fn clear(&mut self, block: usize, bits: u8) {
        if let Some(c) = self.chunks.get_mut(&(block / CHUNK)) {
            c[block % CHUNK] &= !bits;
        }
    }
    fn chunk(&mut self, block: usize) -> &mut [u8; CHUNK] {
        self.chunks.entry(block / CHUNK).or_insert_with(|| Box::new([0; CHUNK]))
    }
    /// Calls `f(block, flags)` for each block of `[from, to)` with any of `mask` set (in order
    /// within a chunk; a range wider than the table walks the chunks that exist).
    fn each(&self, from: usize, to: usize, mask: u8, mut f: impl FnMut(usize, u8)) {
        if from >= to {
            return;
        }
        let (lo, hi) = (from / CHUNK, (to - 1) / CHUNK);
        let mut keys: Vec<usize> = if hi - lo < self.chunks.len() {
            (lo..=hi).filter(|k| self.chunks.contains_key(k)).collect()
        } else {
            self.chunks.keys().copied().filter(|k| (lo..=hi).contains(k)).collect()
        };
        keys.sort_unstable();
        for k in keys {
            let c = &self.chunks[&k];
            for block in (k * CHUNK).max(from)..((k + 1) * CHUNK).min(to) {
                let v = c[block % CHUNK];
                if v & mask != 0 {
                    f(block, v);
                }
            }
        }
    }
    /// Whether every block of `[from, to)` is clean or a hole; and how many are holes.
    fn settled(&self, from: usize, to: usize) -> Option<usize> {
        let mut at = from;
        let mut holes = 0;
        while at < to {
            let end = ((at / CHUNK + 1) * CHUNK).min(to);
            let c = self.chunks.get(&(at / CHUNK))?;
            for block in at..end {
                let v = c[block % CHUNK];
                if v & (CLEAN | HOLE) == 0 {
                    return None;
                }
                holes += usize::from(v & HOLE != 0);
            }
            at = end;
        }
        Some(holes)
    }
}

struct Guards {
    /// Blocks we made read-only (`PROTECTED`), of those the ones not written since a copy took
    /// them (`CLEAN`), and, local build, blocks found entirely unmapped (`HOLE`: holes of the
    /// tile heap, 132 MB of them). A copy reads holes as zeros, once; they are not asked about
    /// again until the runtime maps or unmaps memory there (`title_memory_remapped`). Before,
    /// every asking called them written, and each refresh read them anew (a failing read,
    /// region queries, zero fill).
    flags: Flags,
}

static GUARDS: Mutex<Option<Guards>> = Mutex::new(None);

#[cfg(test)]
static FORCE_SLOW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `VICTORY_FAST_GUARD=0` turns off the shortcuts that skip system calls and walks for memory
/// already known to be guarded (the flag table itself stays).
fn fast_guard() -> bool {
    #[cfg(test)]
    if FORCE_SLOW.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_FAST_GUARD").map_or(true, |v| v.trim() != "0"))
}

/// `VICTORY_HOLE_MEMO=0` turns the memory of holes off.
fn remember_holes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_HOLE_MEMO").map_or(true, |v| v.trim() != "0"))
}

/// Whether block `block` is entirely unmapped (not committed anywhere in it).
fn is_hole(block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let at = block * BLOCK;
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State != MEM_COMMIT && info.BaseAddress as usize + info.RegionSize >= at + BLOCK
}

struct GuardsLock(std::sync::MutexGuard<'static, Option<Guards>>);
impl std::ops::Deref for GuardsLock {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        self.0.as_ref().expect("made")
    }
}
impl std::ops::DerefMut for GuardsLock {
    fn deref_mut(&mut self) -> &mut Guards {
        self.0.as_mut().expect("made")
    }
}

fn guards() -> GuardsLock {
    let mut g = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(|| Guards { flags: Flags::default() });
    GuardsLock(g)
}

/// Whether the block at `at` is known to be guarded view memory (or a hole in one): then no
/// system call is needed to tell it from write-watched or plain memory.
fn known_view(at: usize) -> bool {
    fast_guard() && guards().flags.get(at / BLOCK) & (PROTECTED | HOLE) != 0
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if !known_view(start) {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.Type != MEM_MAPPED {
            return None;
        }
    }
    install();
    let mut g = guards();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Local build: all clean (or holes) is the usual answer for a heap nobody wrote: no walk.
    if fast_guard()
        && let Some(_holes) = g.flags.settled(first, last)
    {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::HoleSkips, _holes as u64);
        return Some(Vec::new());
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    for block in first..=last {
        let flags = if block < last { g.flags.get(block) } else { 0 };
        let hole = flags & HOLE != 0;
        #[cfg(feature = "diagnostics")]
        if hole {
            crate::copystat::add(crate::copystat::C::HoleSkips, 1);
        }
        let open = block < last && !hole && flags & CLEAN == 0;
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if flags & PROTECTED != 0 {
                // Written through the runtime, still read-only.
                g.flags.set(block, CLEAN);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        for block in from..to {
            g.flags.set(block, PROTECTED | CLEAN);
        }
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if remember_holes() && is_hole(block) {
            g.flags.set(block, HOLE);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut blocks: Vec<usize> = Vec::new();
    g.flags.each(start / BLOCK, end.div_ceil(BLOCK), PROTECTED, |block, _| blocks.push(block));
    for &block in &blocks {
        g.flags.clear(block, PROTECTED | CLEAN);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    crate::direct::remapped(at, len);
    crate::shadow::mirror_pages_remapped(at, len);
    forget_extents(at, len);
    let mut g = guards();
    let mut blocks: Vec<usize> = Vec::new();
    g.flags.each(at / BLOCK, at.saturating_add(len).div_ceil(BLOCK), PROTECTED | HOLE, |block, _| blocks.push(block));
    for block in blocks {
        g.flags.clear(block, PROTECTED | CLEAN | HOLE);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        crate::shadow::next_submission(); // (within one, a covered range is answered from the journal)
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        crate::shadow::next_submission();
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// Timing of the world's askings: 444 per frame over a 174 MB view heap nobody wrote.
    /// `cargo test -p victory-gpu --lib guarded_asking_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn guarded_asking_cost() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        const LEN: usize = 174 << 20;
        // SAFETY: a pagefile section mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, LEN as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as usize
        };
        assert_ne!(view, 0);
        let run = |slow: bool| {
            FORCE_SLOW.store(slow, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::Instant::now();
            for i in 0..444 {
                crate::shadow::next_submission();
                // pieces of the heap as the world asks them, and the whole of it
                let (a, b) = if i % 4 == 0 { (view, view + LEN) } else { (view + (i % 40) * PAGE * 8, view + (i % 40) * PAGE * 8 + 20 * PAGE * 8) };
                assert!(writes(a, b, true).is_some());
            }
            t.elapsed()
        };
        let first = run(true); // also guards everything
        let slow = run(true);
        let fast = run(false);
        eprintln!("guarded askings: first {first:?}; 444 slow walks {slow:?}; 444 with the shortcuts {fast:?}");
        title_memory_remapped(view, LEN);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}
