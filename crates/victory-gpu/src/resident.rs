//! Residents (the in-place design's Stage 2, `VICTORY_INPLACE=4`): a big
//! title buffer whose memory lies in an arena (`crate::arena`) is read and
//! written by shaders in a buffer in the card's own memory, `R`, kept in
//! step with the title's memory by the card alone, on the one host queue:
//!
//! - **CPU writes to `R`.** At submission the 64 KiB pages the title wrote
//!   (write watching, the guards: [`Resident::refresh`]) go up from the
//!   arena with `CopyBufferRegion`, in a list of ours run before the
//!   submission's lists ([`prepare`]). Nothing is copied by the CPU, nothing
//!   is staged, nothing is hashed. Whole pages are safe: everything earlier
//!   on the queue, each earlier submission's merge included, has run when
//!   the copy runs, so the arena holds the card's bytes and the title's.
//!   No range is held back because a flight may write it (the #1322 kind of
//!   bug: a page the title wrote always goes up).
//! - **GPU writes to the title.** Every resident has a twin `B` in the
//!   card's memory: what `R` held when the submission's shaders started
//!   (uploads write `R`, then `B` from `R`). After the submission's lists a
//!   compute pass of ours ([`MERGE`]) compares `R` with `B` per dword over
//!   the ranges declared written and stores each dword that differs into
//!   the arena and into `B`: a whole dword as a store, a dword the shaders
//!   changed in part with `InterlockedAnd` and `InterlockedOr` on the
//!   arena (probe P3: bytes the CPU writes meanwhile in the same dword are
//!   kept). It runs in the same submission, before the host fence's
//!   signal, so when the fence is reached the title's memory holds what the
//!   shaders wrote (writes of the card to the write-back pages of an arena
//!   are seen by the CPU once the fence completes). There is no fetch, no
//!   merge on the CPU, no snapshot and no copy-back for a resident.
//!
//! Lists of ours run in their own `ExecuteCommandLists`, before and after
//! the title's: buffers fall back to the common state between them, and no
//! list uses a buffer as copy source and destination without an explicit
//! transition (risk R8; an arena's buffer is only ever a copy source in the
//! upload and only unordered access in the merge).
//!
//! A buffer that does not lie in an arena (or the arena cannot be opened,
//! or the budget is spent) keeps today's copy in the card's memory
//! (`crate::shadow::cards`). Shader writes to residents are declared to the
//! flights as writes in place are (`host::in_place_writes`), so late wait
//! and the relay hold back what reads them until the flight has run.
//!
//! Residents are per address range: a title buffer inside a resident's
//! pages is that resident at an offset, whatever its size, and shares its
//! bytes. Where copies of the same title memory still differ (a buffer in
//! place directly in the arena inside a resident, a resident made before a
//! bigger one), what the card writes to one is merged into the arena and
//! its pages go up to the others before the next submission
//! ([`prepare`]).
//!
//! shortcut: as with copies in the card's memory, shaders are assumed to
//! write only the ranges declared written; a write outside them stays in
//! `R` until a declared range holds it, or a page the title wrote goes up
//! over it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::transition;

/// The unit that goes up: the console's page, write watching's unit.
const PAGE: u64 = crate::watch::PAGE as u64;

/// Whether title buffers that would get a copy in the card's memory are
/// residents where their memory lies in an arena (`VICTORY_INPLACE=4`; the
/// levels are cumulative, `crate::inplace::level`).
pub(crate) fn on() -> bool {
    crate::inplace::level() >= 4
}

/// A title range in the card's memory, fed from its arena.
pub(crate) struct Resident {
    /// Title memory `[start, start + len)`, on 64 KiB.
    start: u64,
    len: u64,
    /// `R`: what shaders read and write.
    pub(crate) buffer: ID3D12Resource,
    gpu: u64,
    /// `B`: `R`'s bytes as the submission's shaders found them.
    twin: ID3D12Resource,
    /// The arena's buffer, and `start`'s offset and host address in it.
    arena: ID3D12Resource,
    _heap: crate::arena::Heap,
    arena_offset: u64,
    title_gpu: u64,
    /// In a section's arena (view memory): the section.
    section: Option<usize>,
    /// Per page: the write-watch asking after which it last went up (0:
    /// never).
    copied: Mutex<Vec<u64>>,
    /// Offsets of `R` to go up before the next submission's lists, in the
    /// order noted, joined where they touch the last.
    dirty: Mutex<Vec<(u64, u64)>>,
}

// SAFETY: D3D12 objects are free-threaded; the rest is behind locks.
unsafe impl Send for Resident {}
// SAFETY: as above.
unsafe impl Sync for Resident {}

impl Drop for Resident {
    fn drop(&mut self) {
        BYTES.fetch_sub(2 * self.len, Ordering::Relaxed);
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Resident {
    /// `buffer` and `twin` (`len` bytes each, in the card's memory) for
    /// title memory `[start, start + len)` at `at` in its arena. Everything
    /// goes up before the first submission. The caller has counted the
    /// bytes in [`BYTES`].
    fn new(start: u64, len: u64, buffer: ID3D12Resource, twin: ID3D12Resource, at: crate::arena::Resolved) -> Resident {
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Resident {
            start,
            len,
            buffer,
            gpu,
            twin,
            arena: at.resource,
            _heap: at.heap,
            arena_offset: at.offset,
            title_gpu: at.gpu,
            section: at.section,
            copied: Mutex::new(vec![0; (len / PAGE) as usize]),
            dirty: Mutex::new(vec![(0, len)]),
        }
    }

    /// Whether it holds any of title memory `[at, end)`.
    pub(crate) fn overlaps(&self, at: u64, end: u64) -> bool {
        self.start < end && at < self.start + self.len
    }

    /// `R`'s host GPU address.
    pub(crate) fn gpu(&self) -> u64 {
        self.gpu
    }

    /// In a section's arena: the section, and the offset in it of the
    /// title memory at `start + bias`.
    pub(crate) fn section_at(&self, bias: u64) -> Option<(usize, u64)> {
        self.section.map(|s| (s, self.arena_offset + bias))
    }

    /// Whether it still shows the title's memory: view memory the title
    /// mapped anew is no longer the same run of its section.
    fn still_good(&self) -> bool {
        self.section.is_none_or(|s| crate::inplace::still_run(self.start, self.len, s, self.arena_offset))
    }

    /// At submission: the pages of title memory `[from, to)` (cut to the
    /// resident) the title wrote since they last went up are to go up
    /// before the submission's lists, whole. Whether write watching or the
    /// guards could say; if not, every page of the range goes up.
    pub(crate) fn refresh(&self, from: u64, to: u64) -> bool {
        let (from, to) = (from.max(self.start), to.min(self.start + self.len));
        if from >= to {
            return true;
        }
        let (first, last) = ((from - self.start) / PAGE, (to - self.start).div_ceil(PAGE));
        let Some((now, written)) = crate::watch::writes(from as usize, to as usize, true) else {
            self.mark(first * PAGE, last * PAGE);
            return false;
        };
        let base = self.start / PAGE;
        let mut pages = Vec::new();
        {
            let mut copied = lock(&self.copied);
            let mut writes = written.iter().peekable();
            for page in first..last {
                let write = match writes.peek() {
                    Some(&&(p, w)) if p as u64 == base + page => {
                        writes.next();
                        w
                    }
                    _ => 0,
                };
                let at = &mut copied[page as usize];
                if *at == 0 || write > *at {
                    *at = now;
                    pages.push(page);
                }
            }
        }
        for (a, b) in page_runs(&pages) {
            self.mark(a * PAGE, b * PAGE);
        }
        true
    }

    /// Offsets `[from, to)` of `R` go up before the next submission.
    fn mark(&self, from: u64, to: u64) {
        let (from, to) = (from.min(self.len), to.min(self.len));
        if from >= to {
            return;
        }
        let mut dirty = lock(&self.dirty);
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
    }

    /// What is to go up, taken: in order, joined.
    fn take(&self) -> Vec<(u64, u64)> {
        joined(std::mem::take(&mut *lock(&self.dirty)))
    }
}

/// Runs of consecutive page numbers (ascending) as `[first, end)`.
fn page_runs(pages: &[u64]) -> Vec<(u64, u64)> {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &page in pages {
        match runs.last_mut() {
            Some(last) if last.1 == page => last.1 += 1,
            _ => runs.push((page, page + 1)),
        }
    }
    runs
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Every resident alive (also those only a list of ours or a title list
/// still holds): what the title wrote goes up to each of them.
static ALL: Mutex<Vec<Weak<Resident>>> = Mutex::new(Vec::new());

/// Residents by their first title address, for title buffers in their
/// pages to share them.
static RANGES: Mutex<BTreeMap<u64, Weak<Resident>>> = Mutex::new(BTreeMap::new());

/// Bytes of residents alive, `R` and `B` together.
static BYTES: AtomicU64 = AtomicU64::new(0);

/// Since the last line: bytes that went up from arenas and their ranges;
/// bytes the merge compared and its dispatches; bytes that went up again
/// because another copy of the same memory was written; submissions that
/// had work for residents.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The most bytes of residents, `R` and `B` together
/// (`VICTORY_RESIDENT_MB`; by default 60% of the card's budget for this
/// process when first asked). Each resident is also made only while the
/// card's memory in use stays under 80% of that budget with it, as copies
/// in the card's memory are (`shadow::card_for`).
fn most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_RESIDENT_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|mb| mb << 20)
            .or_else(|| crate::host::host()?.video_memory().map(|[budget, ..]| budget / 10 * 6))
            .unwrap_or(4 << 30)
    })
}

/// A live resident that holds title memory `[from, to)`, if one does.
/// Residents of view memory the title mapped anew are forgotten here.
fn holder(from: u64, to: u64) -> Option<Arc<Resident>> {
    let mut ranges = lock(&RANGES);
    let mut gone = Vec::new();
    let found = ranges.range(..=from).rev().take(64).find_map(|(&start, weak)| {
        let Some(r) = weak.upgrade().filter(|r| r.still_good()) else {
            gone.push(start);
            return None;
        };
        (r.start + r.len >= to).then_some(r)
    });
    for start in gone {
        ranges.remove(&start);
    }
    found
}

/// What [`find`] gives a title buffer.
pub(crate) struct Found {
    pub(crate) resident: Arc<Resident>,
    /// The bytes from the resident's first byte to the buffer's.
    pub(crate) bias: u64,
    /// Residents the new one was made over and replaces: the caller lets
    /// go of the buffers in them, whose next ask finds the new one.
    pub(crate) replaced: Vec<Arc<Resident>>,
}

/// The resident title buffer `[start, start + len)` is in: the one that
/// holds its 64 KiB pages if there is one (any buffer); else, with `new`
/// (a buffer that would get a copy in the card's memory), one made for
/// its pages if they lie in an arena and the budget has room ([`make_over`]:
/// over the residents it overlaps too, if it can be). `None`: as before
/// (in place, or a copy).
pub(crate) fn find(start: u64, len: u64, new: bool) -> Option<Found> {
    if !on() {
        return None;
    }
    let from = start & !(PAGE - 1);
    let to = start.checked_add(len.max(1))?.checked_next_multiple_of(PAGE)?;
    if let Some(resident) = holder(from, to) {
        let bias = start - resident.start;
        return Some(Found { resident, bias, replaced: Vec::new() });
    }
    if !new {
        return None;
    }
    let host = crate::host::host()?;
    if !ready(&host.device) {
        return None;
    }
    let (resident, replaced) = make_over(from, to, most(), &mut crate::arena::resolve_waiting, &mut |bytes| {
        if let Some([budget, used, ..]) = host.video_memory()
            && used + 2 * bytes > budget / 10 * 8
        {
            return None;
        }
        host.uav_buffer(bytes)
    })?;
    tracing::info!(
        start = format_args!("{:#x}", resident.start),
        mb = resident.len >> 20,
        replaced = replaced.len(),
        "a title buffer is resident in the card's memory, fed from its arena (VICTORY_INPLACE=4)"
    );
    let bias = start - resident.start;
    Some(Found { resident, bias, replaced })
}

/// A resident for title memory `[from, to)` (on 64 KiB) that no resident
/// holds. Residents it overlaps become one with it: the new resident is
/// made over all of them, if that lies in one arena within the budget,
/// and takes their place in the registry (they stay alive, and in step,
/// for what holds them: [`cross_mark`]). Else it is made over `[from, to)`
/// alone. With the residents it replaces.
fn make_over(
    from: u64,
    to: u64,
    most: u64,
    resolve: &mut dyn FnMut(u64, u64) -> Option<crate::arena::Resolved>,
    buffer: &mut dyn FnMut(u64) -> Option<ID3D12Resource>,
) -> Option<(Arc<Resident>, Vec<Arc<Resident>>)> {
    let over: Vec<Arc<Resident>> =
        lock(&RANGES).range(..to).filter_map(|(_, w)| w.upgrade()).filter(|r| r.overlaps(from, to) && r.still_good()).collect();
    if !over.is_empty() {
        let (lo, hi) = over.iter().fold((from, to), |(a, b), r| (a.min(r.start), b.max(r.start + r.len)));
        if let Some(union) = make(lo, hi - lo, most, &mut *resolve, &mut *buffer) {
            lock(&RANGES).retain(|_, w| !over.iter().any(|r| std::ptr::eq(w.as_ptr(), Arc::as_ptr(r))));
            return Some((union, over));
        }
    }
    Some((make(from, to - from, most, resolve, buffer)?, Vec::new()))
}

/// A resident for title memory `[from, from + bytes)` (on 64 KiB), if the
/// residents' bytes stay within `most`: `resolve` says where it is in an
/// arena, `buffer` makes `R` and `B`. Registered for buffers in its
/// pages to share.
fn make(
    from: u64,
    bytes: u64,
    most: u64,
    resolve: &mut dyn FnMut(u64, u64) -> Option<crate::arena::Resolved>,
    buffer: &mut dyn FnMut(u64) -> Option<ID3D12Resource>,
) -> Option<Arc<Resident>> {
    // The merge counts in 32 bits from `R`'s first byte.
    if bytes >= crate::arena::PART {
        return None;
    }
    let at = resolve(from, bytes).filter(|r| r.left >= bytes)?;
    if BYTES.fetch_add(2 * bytes, Ordering::Relaxed) + 2 * bytes > most {
        BYTES.fetch_sub(2 * bytes, Ordering::Relaxed);
        return None;
    }
    let (Some(r), Some(b)) = (buffer(bytes), buffer(bytes)) else {
        BYTES.fetch_sub(2 * bytes, Ordering::Relaxed);
        return None;
    };
    crate::host::name(&r, || format!("resident {from:#x}+{bytes:#x}"));
    crate::host::name(&b, || format!("resident {from:#x}+{bytes:#x}, its twin"));
    let resident = Arc::new(Resident::new(from, bytes, r, b, at));
    lock(&ALL).push(Arc::downgrade(&resident));
    lock(&RANGES).insert(from, Arc::downgrade(&resident));
    Some(resident)
}

/// Title memory `[at, end)` is let go of in place (`shadow::in_place_let_go`:
/// its part retired, or the title maps pages anew): no new title buffer
/// takes a resident over it. With `remap`, residents of view memory stay
/// (they are looked at again when next asked for, [`holder`]). Whether a
/// resident was forgotten (the caller lets go of the buffers in it).
pub(crate) fn forget(at: u64, end: u64, remap: bool) -> bool {
    let mut forgot = false;
    lock(&RANGES).retain(|_, weak| {
        let keep = weak.upgrade().is_none_or(|r| !r.overlaps(at, end) || (remap && r.section.is_some()));
        forgot |= !keep;
        keep
    });
    forgot
}

/// The merge (one thread per 16 bytes of a declared range of `R`):
/// `from`/`to` are offsets in `R` on 4 bytes; `title` is the arena at `R`'s
/// first byte. A dword of `R` that differs from `B` goes into the title's
/// memory, and into `B`: whole if every byte differs, else only the bytes
/// that do, by two atomics, so that the other bytes of the dword stay what
/// they are in the title's memory (the CPU may write them meanwhile).
/// `counts` adds the dwords that differed and how many of them in part.
pub(crate) const MERGE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Range : register(b0) {
    uint from;
    uint to;
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer twin : register(u0);
RWByteAddressBuffer title : register(u1);
RWByteAddressBuffer counts : register(u2);
groupshared uint changed;
groupshared uint partial;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0u) {
        changed = 0u;
        partial = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint c = 0u;
    uint p = 0u;
    uint at = from + id.x * 16u;
    [unroll]
    for (uint k = 0u; k < 4u; k++) {
        uint o = at + k * 4u;
        if (o < to) {
            uint gpu = now.Load(o);
            uint diff = gpu ^ twin.Load(o);
            if (diff != 0u) {
                uint mask = ((diff & 0x000000FFu) != 0u ? 0x000000FFu : 0u)
                          | ((diff & 0x0000FF00u) != 0u ? 0x0000FF00u : 0u)
                          | ((diff & 0x00FF0000u) != 0u ? 0x00FF0000u : 0u)
                          | ((diff & 0xFF000000u) != 0u ? 0xFF000000u : 0u);
                if (mask == 0xFFFFFFFFu) {
                    title.Store(o, gpu);
                } else {
                    title.InterlockedAnd(o, gpu | ~mask);
                    title.InterlockedOr(o, gpu & mask);
                    p++;
                }
                twin.Store(o, gpu);
                c++;
            }
        }
    }
    if (c != 0u) InterlockedAdd(changed, c);
    if (p != 0u) InterlockedAdd(partial, p);
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0u && changed != 0u) {
        counts.InterlockedAdd(0, changed);
        counts.InterlockedAdd(4, partial);
    }
}
"#;

/// The bytes of `R` one dispatch of the merge covers at most (32,768
/// groups of 64 threads of 16 bytes; a dispatch takes 65,535 groups).
const MERGE_CHUNK: u64 = 32 << 20;

/// What goes up to one resident in a submission: offsets of `R`.
pub(crate) struct Upload<'a> {
    pub(crate) resident: &'a Resident,
    pub(crate) ranges: Vec<(u64, u64)>,
}

/// Records the upload: each range from the arena into `R`, then from `R`
/// into `B` (from `R`, not from the arena, so that a CPU write landing
/// between the two copies cannot make them differ). Returns the bytes.
///
/// # Safety
/// `list` is open; the residents are alive until it has run.
pub(crate) unsafe fn record_upload(list: &ID3D12GraphicsCommandList, uploads: &[Upload]) -> u64 {
    let uploads: Vec<&Upload> = uploads.iter().filter(|u| !u.ranges.is_empty()).collect();
    if uploads.is_empty() {
        return 0;
    }
    let mut bytes = 0;
    // SAFETY: the caller's open list; live buffers, the ranges inside them.
    unsafe {
        let to_dest: Vec<D3D12_RESOURCE_BARRIER> =
            uploads.iter().map(|u| transition(&u.resident.buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)).collect();
        list.ResourceBarrier(&to_dest);
        for u in &uploads {
            let r = u.resident;
            for &(a, b) in &u.ranges {
                list.CopyBufferRegion(&r.buffer, a, &r.arena, r.arena_offset + a, b - a);
                bytes += b - a;
            }
        }
        let to_source: Vec<D3D12_RESOURCE_BARRIER> =
            uploads.iter().map(|u| transition(&u.resident.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)).collect();
        list.ResourceBarrier(&to_source);
        for u in &uploads {
            for &(a, b) in &u.ranges {
                list.CopyBufferRegion(&u.resident.twin, a, &u.resident.buffer, a, b - a);
            }
        }
    }
    bytes
}

/// What the merge needs beside the residents: its pipeline and the
/// counters (in the card's memory, copied after each merge into memory the
/// CPU reads).
pub(crate) struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    counts: ID3D12Resource,
    seen: ID3D12Resource,
    seen_at: *const u32,
}

// SAFETY: D3D12 objects and a persistent mapping that is only read.
unsafe impl Send for Pass {}

impl Pass {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Pass> {
        let (root, pso) = crate::inplace_probes::compute(device, MERGE)?;
        let counts = committed(device, D3D12_HEAP_TYPE_DEFAULT, 256)?;
        let seen = committed(device, D3D12_HEAP_TYPE_CUSTOM, 256)?;
        let mut at = std::ptr::null_mut();
        // SAFETY: a CPU-visible buffer of ours, mapped for its life.
        unsafe { seen.Map(0, None, Some(&mut at))? };
        crate::host::name(&counts, || "residents: the merge's counters".into());
        Ok(Pass { root, pso, counts, seen, seen_at: at.cast() })
    }

    /// The counters as the last merge that ran left them: dwords that
    /// differed, of them those that differed in part (both wrap).
    pub(crate) fn counts(&self) -> [u32; 2] {
        // SAFETY: the mapping holds 256 bytes; the card writes them whole.
        unsafe { [self.seen_at.read_volatile(), self.seen_at.add(1).read_volatile()] }
    }
}

/// A buffer of ours of `size` bytes, unordered access allowed: in the
/// card's memory (`DEFAULT`), or in the PC's, write-back, for the CPU to
/// map (`CUSTOM`).
fn committed(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, size: u64) -> windows::core::Result<ID3D12Resource> {
    let heap = if kind == D3D12_HEAP_TYPE_CUSTOM {
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        }
    } else {
        D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() }
    };
    let mut out: Option<ID3D12Resource> = None;
    let desc = crate::inplace::buffer_desc(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    // SAFETY: plain creation.
    unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut out)? };
    out.ok_or_else(windows::core::Error::empty)
}

/// What the merge does for one resident in a submission: offsets of `R`,
/// in order, apart, on 4 bytes.
pub(crate) struct Merge<'a> {
    pub(crate) resident: &'a Resident,
    pub(crate) ranges: Vec<(u64, u64)>,
}

/// Records the merge of `merges`. Returns the bytes compared and the
/// dispatches.
///
/// # Safety
/// `list` is open and runs after the lists whose shader writes it merges,
/// in an `ExecuteCommandLists` of its own; the residents are alive until it
/// has run.
pub(crate) unsafe fn record_merge(list: &ID3D12GraphicsCommandList, pass: &Pass, merges: &[Merge]) -> (u64, u64) {
    let (mut bytes, mut dispatches) = (0, 0);
    // SAFETY: the caller's open list; live buffers; the merge's own root
    // signature; every offset lies in `R`, `B` and the arena.
    unsafe {
        list.ResourceBarrier(&[transition(&pass.counts, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        list.SetComputeRootUnorderedAccessView(4, pass.counts.GetGPUVirtualAddress());
        for m in merges {
            let r = m.resident;
            list.SetComputeRootShaderResourceView(1, r.gpu);
            list.SetComputeRootUnorderedAccessView(2, r.twin.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, r.title_gpu);
            for &(a, b) in &m.ranges {
                let mut at = a;
                while at < b {
                    let end = (at + MERGE_CHUNK).min(b);
                    let constants = [at as u32, end as u32];
                    list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                    list.Dispatch((end - at).div_ceil(1024) as u32, 1, 1);
                    dispatches += 1;
                    at = end;
                }
                bytes += b - a;
            }
        }
        list.ResourceBarrier(&[transition(&pass.counts, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&pass.seen, 0, &pass.counts, 0, 8);
    }
    (bytes, dispatches)
}

/// A list of ours with the residents its run needs alive, and the value
/// of the host's fence after which it may be used again.
struct Spare {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    after: u64,
    kept: Vec<Arc<Resident>>,
}

/// What the host keeps for residents.
struct Work {
    pass: Pass,
    /// Lists free once the host's fence passes their value.
    spare: Vec<Spare>,
    /// The lists of the submission being prepared ([`sent`] gives them
    /// their value).
    out: Vec<Spare>,
    /// The counters as the last line read them.
    read: [u32; 2],
}

// SAFETY: D3D12 objects, used under the lock.
unsafe impl Send for Work {}

static WORK: Mutex<Option<Work>> = Mutex::new(None);

/// Whether the merge could be made (at the first ask); a resident is made
/// only if it could.
fn ready(device: &ID3D12Device) -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| match Pass::new(device) {
        Ok(pass) => {
            *lock(&WORK) = Some(Work { pass, spare: Vec::new(), out: Vec::new(), read: [0; 2] });
            true
        }
        Err(e) => {
            tracing::warn!(error = %e, "no residents: the merge could not be made (VICTORY_INPLACE=4)");
            false
        }
    })
}

impl Work {
    /// A list of ours, open, whose last run is done (`completed` is the
    /// host fence's value), holding `kept` until its next run is done. A
    /// list that cannot be had ends the run: what went up or was written
    /// would be lost.
    fn take(&mut self, device: &ID3D12Device, completed: u64, kept: Vec<Arc<Resident>>) -> ID3D12GraphicsCommandList {
        for spare in &mut self.spare {
            if spare.after <= completed {
                spare.kept.clear();
            }
        }
        let spare = match self.spare.iter().position(|s| s.after <= completed) {
            Some(at) => Ok(self.spare.swap_remove(at)),
            // SAFETY: plain object creation.
            None => unsafe {
                (|| {
                    let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    Ok(Spare { allocator, list, after: 0, kept: Vec::new() })
                })()
            },
        };
        // SAFETY: the list's last run is done.
        let spare = spare.and_then(|s| unsafe { s.allocator.Reset().and_then(|()| s.list.Reset(&s.allocator, None)).map(|()| s) });
        match spare {
            Ok(spare) => {
                let list = spare.list.clone();
                self.out.push(Spare { kept, ..spare });
                list
            }
            Err(e) => crate::host::fail(format_args!("no list of ours for residents ({e}, VICTORY_INPLACE=4)")),
        }
    }
}

/// What [`prepare`] made for a submission: its lists, closed, to run
/// before and after the submission's lists, each in an
/// `ExecuteCommandLists` of its own.
#[derive(Default)]
pub(crate) struct Prepared {
    pub(crate) upload: Option<ID3D12CommandList>,
    pub(crate) merge: Option<ID3D12CommandList>,
}

/// At submission, after its refreshes (under the submission lock): what
/// went up to residents since the last submission is recorded to run
/// before its lists, and the merge of `declared` (title ranges its shaders
/// may write in residents) to run after them. `in_place` are the title
/// ranges its shaders may write in arenas directly (buffers and windows in
/// place); they and `declared` are marked to go up, before the next
/// submission, to every other resident over them. `completed` is the host
/// fence's value; [`sent`] follows with the value signalled after the
/// submission.
pub(crate) fn prepare(device: &ID3D12Device, completed: u64, declared: &[(Arc<Resident>, u64, u64)], in_place: &[(u64, u64)]) -> Prepared {
    let mut out = Prepared::default();
    let mut work = lock(&WORK);
    let Some(work) = work.as_mut() else {
        return out;
    };
    let live: Vec<Arc<Resident>> = {
        let mut all = lock(&ALL);
        all.retain(|w| w.strong_count() > 0);
        all.iter().filter_map(Weak::upgrade).collect()
    };
    if live.is_empty() {
        return out;
    }
    let mut ups: Vec<Upload> = Vec::new();
    let mut up_kept: Vec<Arc<Resident>> = Vec::new();
    for r in &live {
        let ranges = r.take();
        if !ranges.is_empty() {
            ups.push(Upload { resident: r, ranges });
            up_kept.push(r.clone());
        }
    }
    // Per resident, the declared ranges as offsets, joined, on 4 bytes
    // (dispatches declare the same ranges thousands of times).
    let mut merges: Vec<Merge> = Vec::new();
    let mut merge_kept: Vec<Arc<Resident>> = Vec::new();
    for (r, from, to) in declared {
        let (a, b) = (from.saturating_sub(r.start).min(r.len) & !3, to.saturating_sub(r.start).min(r.len).next_multiple_of(4));
        if a >= b {
            continue;
        }
        match merges.iter_mut().find(|m| std::ptr::eq(m.resident, &**r)) {
            Some(m) => m.ranges.push((a, b)),
            None => {
                merges.push(Merge { resident: r, ranges: vec![(a, b)] });
                merge_kept.push(r.clone());
            }
        }
    }
    for m in &mut merges {
        m.ranges = joined(std::mem::take(&mut m.ranges));
    }
    if !ups.is_empty() {
        let list = work.take(device, completed, up_kept);
        // SAFETY: an open list of ours; the residents are kept with it.
        let bytes = unsafe { record_upload(&list, &ups) };
        STATS[0].fetch_add(bytes, Ordering::Relaxed);
        STATS[1].fetch_add(ups.iter().map(|u| u.ranges.len() as u64).sum(), Ordering::Relaxed);
        out.upload = Some(close(&list, "the upload to residents"));
    }
    if !merges.is_empty() {
        let list = work.take(device, completed, merge_kept);
        // SAFETY: an open list of ours, run after the submission's lists;
        // the residents are kept with it.
        let (bytes, dispatches) = unsafe { record_merge(&list, &work.pass, &merges) };
        STATS[2].fetch_add(bytes, Ordering::Relaxed);
        STATS[3].fetch_add(dispatches, Ordering::Relaxed);
        out.merge = Some(close(&list, "the merge of residents"));
    }
    // What this submission writes reaches the arena by its end (in place,
    // or merged); every other resident over it takes those pages again
    // before the next submission (this one's upload is taken already).
    let mut written: Vec<(Option<&Resident>, u64, u64)> =
        merges.iter().flat_map(|m| m.ranges.iter().map(|&(a, b)| (Some(m.resident), m.resident.start + a, m.resident.start + b))).collect();
    written.extend(joined(in_place.to_vec()).into_iter().map(|(from, to)| (None, from, to)));
    STATS[4].fetch_add(cross_mark(&live, &written), Ordering::Relaxed);
    if out.upload.is_some() || out.merge.is_some() {
        STATS[5].fetch_add(1, Ordering::Relaxed);
    }
    out
}

/// Title ranges `written` (each with the resident whose merge writes it,
/// or none: written in place) go up again to every other resident of
/// `live` that holds some of them, by whole pages. Returns the bytes.
fn cross_mark(live: &[Arc<Resident>], written: &[(Option<&Resident>, u64, u64)]) -> u64 {
    let mut marked = 0;
    for &(writer, from, to) in written {
        for other in live {
            if writer.is_some_and(|w| std::ptr::eq(w, &**other)) || !other.overlaps(from, to) {
                continue;
            }
            let a = (from.max(other.start) - other.start) & !(PAGE - 1);
            let b = (to.min(other.start + other.len) - other.start).next_multiple_of(PAGE);
            other.mark(a, b);
            marked += b - a;
        }
    }
    marked
}

/// Closes a list of ours; a failure ends the run (the residents' bytes
/// would be wrong from here on).
fn close(list: &ID3D12GraphicsCommandList, what: &str) -> ID3D12CommandList {
    // SAFETY: an open list of ours.
    match unsafe { list.Close() }.and_then(|()| list.cast()) {
        Ok(list) => list,
        Err(e) => crate::host::fail(format_args!("{what} could not be recorded ({e})")),
    }
}

/// The submission [`prepare`] was for is on the queue, and the host's
/// fence is signalled with `value` behind it: its lists are free again
/// once the fence has passed it.
pub(crate) fn sent(value: u64) {
    if !on() {
        return;
    }
    if let Some(work) = lock(&WORK).as_mut() {
        let out = std::mem::take(&mut work.out);
        work.spare.extend(out.into_iter().map(|s| Spare { after: value, ..s }));
    }
}

/// The line, with the frame-rate line, while residents are on.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let (alive, bytes) = {
        let all = lock(&ALL);
        let alive: Vec<Arc<Resident>> = all.iter().filter_map(Weak::upgrade).collect();
        (alive.len(), alive.iter().map(|r| r.len).sum::<u64>())
    };
    let [changed, partial] = lock(&WORK).as_mut().map_or([0; 2], |w| {
        let now = w.pass.counts();
        let delta = [now[0].wrapping_sub(w.read[0]), now[1].wrapping_sub(w.read[1])];
        w.read = now;
        delta
    });
    let mb = |b: f64| b / f64::from(1u32 << 20);
    tracing::info!(
        "residents (VICTORY_INPLACE=4): {alive} alive, {} MB in the card's memory and {} MB of twins (budget {} MB); per frame: \
         {:.1} MB uploaded from arenas in {:.0} ranges, {:.1} MB compared in {:.0} merge passes, {:.2} MB merged into arenas \
         ({:.0} partial dwords), {:.1} MB marked to go up again after writes through other copies, {:.1} submissions with work for residents",
        bytes >> 20,
        bytes >> 20,
        most() >> 20,
        mb(s[0]),
        s[1],
        mb(s[2]),
        s[3],
        mb(f64::from(changed) * 4.0 / frames),
        f64::from(partial) / frames,
        mb(s[4]),
        s[5],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inplace::{From, K64, Rig, open, place};
    use windows_sys::Win32::System::Memory as mem;

    /// The byte mask of the bytes in which two dwords differ, as [`MERGE`]
    /// makes it.
    fn byte_mask(diff: u32) -> u32 {
        (0..4).map(|k| 0xffu32 << (8 * k)).filter(|m| diff & m != 0).fold(0, |a, m| a | m)
    }

    /// What the merge leaves in the title's dword `title` for `gpu` (what
    /// `R` holds) against `was` (`B`): the bytes the card changed are the
    /// card's, every other byte the title's.
    fn merged(title: u32, gpu: u32, was: u32) -> u32 {
        let mask = byte_mask(gpu ^ was);
        (title & !mask) | (gpu & mask)
    }

    #[test]
    fn a_dword_takes_only_the_bytes_the_card_changed() {
        assert_eq!(byte_mask(0), 0);
        assert_eq!(byte_mask(0x0000_0100), 0x0000_ff00);
        assert_eq!(byte_mask(0x8000_0001), 0xff00_00ff);
        assert_eq!(byte_mask(0x0101_0101), u32::MAX);
        // As the two atomics do it: And with `gpu | !mask`, Or with
        // `gpu & mask`.
        for (title, gpu, was) in [(0x1122_3344u32, 0x11aa_3344u32, 0x1122_3344u32), (0xdead_beef, 0x0000_0000, 0xffff_ff00), (7, 7, 7)] {
            let mask = byte_mask(gpu ^ was);
            assert_eq!((title & (gpu | !mask)) | (gpu & mask), merged(title, gpu, was));
        }
        assert_eq!(merged(0x1122_33ff, 0x11aa_3344, 0x1122_3344), 0x11aa_33ff, "the title's byte 0 is kept");
    }

    /// Pages noted as runs; ranges joined where they touch.
    #[test]
    fn pages_go_up_in_runs() {
        assert_eq!(page_runs(&[]), []);
        assert_eq!(page_runs(&[2, 3, 4, 7, 9, 10]), [(2, 5), (7, 8), (9, 11)]);
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20)]), [(0, 6), (8, 20)]);
    }

    /// Title memory of `pages` 64 KiB pages, write-watched as the
    /// runtime's is, opened as a part on `device`: the memory and where it
    /// is in its arena. It stays reserved to the end of the run: the
    /// write-watch journal (`crate::watch`) remembers its pages, and a
    /// later test given the same addresses would find them written.
    fn title(device: &ID3D12Device, pages: usize) -> Option<(*mut u8, crate::arena::Resolved)> {
        device.cast::<ID3D12Device13>().ok()?;
        let bytes = pages * K64;
        // SAFETY: an allocation of the test's own.
        let base = unsafe {
            mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE | mem::MEM_COMMIT | crate::inplace::MEM_WRITE_WATCH, mem::PAGE_READWRITE)
        }
        .cast::<u8>();
        assert!(!base.is_null());
        crate::inplace::fill(base, bytes, 0x5a);
        let heap = open(device, From::Part(base as usize, bytes)).ok()?;
        let resource = place(device, &heap, 0, bytes as u64).ok()?;
        // SAFETY: a live buffer.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        let at = crate::arena::Resolved { resource, heap: heap.into(), offset: 0, gpu, left: bytes as u64, kind: crate::arena::Kind::InPlace, section: None };
        Some((base, at))
    }

    /// A resident over all of `at`'s title memory, `R` and `B` in the
    /// device's own memory.
    fn resident(device: &ID3D12Device, base: *mut u8, at: crate::arena::Resolved) -> Resident {
        let len = at.left;
        let r = committed(device, D3D12_HEAP_TYPE_DEFAULT, len).unwrap();
        let b = committed(device, D3D12_HEAP_TYPE_DEFAULT, len).unwrap();
        BYTES.fetch_add(2 * len, Ordering::Relaxed);
        Resident::new(base as u64, len, r, b, at)
    }

    /// Test (a): the pages the title wrote, and only those, go up from the
    /// arena into `R`, and into `B` from `R`; nothing is copied by the CPU.
    #[test]
    fn warp_the_pages_the_title_wrote_go_up_from_the_arena() {
        let Some(device) = crate::inplace::tests::warp() else { return };
        let Some((base, at)) = title(&device, 8) else { return };
        let r = resident(&device, base, at);
        let (start, n) = (base as u64, 8 * K64 as u64);
        let k = K64 as u64;
        assert_eq!(r.take(), [(0, n)], "everything goes up at first");
        // Never taken: the first look takes every page it covers.
        assert!(r.refresh(start + 100, start + 3 * k));
        assert_eq!(r.take(), [(0, 3 * k)], "whole pages");
        assert!(r.refresh(start, start + n));
        assert_eq!(r.take(), [(3 * k, n)], "what the first look did not cover");
        assert!(r.refresh(start, start + n));
        assert_eq!(r.take(), [], "nothing written since");
        let mut rig = Rig::new(&device).unwrap();
        // SAFETY: an open list of the rig's; the resident lives through it.
        rig.run(|list| unsafe {
            record_upload(list, &[Upload { resident: &r, ranges: vec![(0, n)] }]);
        })
        .unwrap();
        // SAFETY (each): inside the allocation.
        let bytes = |at: u64, len: usize| unsafe { std::slice::from_raw_parts(base.add(at as usize), len).to_vec() };
        assert_eq!(rig.fetch(&r.buffer, 0, n as usize).unwrap(), bytes(0, n as usize));
        assert_eq!(rig.fetch(&r.twin, 0, n as usize).unwrap(), bytes(0, n as usize));
        // The title writes pages 2, 5 and 6.
        for (page, value) in [(2u64, 1u8), (5, 2), (6, 3)] {
            // SAFETY: inside the allocation.
            unsafe { base.add((page * k + 77) as usize).write_volatile(value) };
        }
        assert!(r.refresh(start, start + n));
        let ranges = r.take();
        assert_eq!(ranges, [(2 * k, 3 * k), (5 * k, 7 * k)]);
        // SAFETY: as above.
        let up = rig.run(|list| unsafe {
            record_upload(list, &[Upload { resident: &r, ranges: ranges.clone() }]);
        });
        up.unwrap();
        assert_eq!(rig.fetch(&r.buffer, 0, n as usize).unwrap(), bytes(0, n as usize));
        assert_eq!(rig.fetch(&r.twin, 0, n as usize).unwrap(), bytes(0, n as usize));
        drop(r);
    }

    /// The merge of a resident's declared range: `gpu` is what the
    /// submission's shaders left in `R`; `cpu` writes byte 0 of dwords
    /// 128..160 of the title's memory all along. The pass runs on `device`.
    fn merge_with_the_cpu_writing(device: &ID3D12Device) {
        let Some((base, at)) = title(device, 1) else { return };
        let r = resident(device, base, at);
        let n = K64 as u64;
        let mut rig = Rig::new(device).unwrap();
        let pass = Pass::new(device).unwrap();
        // SAFETY: an open list of the rig's; the resident lives through it.
        rig.run(|list| unsafe {
            record_upload(list, &[Upload { resident: &r, ranges: r.take() }]);
        })
        .unwrap();
        // The title's dwords, read as the CPU sees them (the card and a
        // thread of the test write them too).
        // SAFETY (each): inside the allocation.
        let word = |w: usize| unsafe { base.cast::<u32>().add(w).read_volatile() };
        let words = || (0..K64 / 4).map(word).collect::<Vec<u32>>();
        let before = words();
        // What the shaders wrote: dwords 0..16 whole, 64..96 and 128..160
        // in byte 2 only.
        let mut gpu = before.clone();
        for w in 0..16 {
            gpu[w] = !before[w];
        }
        for w in (64..96).chain(128..160) {
            gpu[w] = before[w] ^ 0x00ab_0000;
        }
        let as_bytes: Vec<u8> = gpu.iter().flat_map(|w| w.to_le_bytes()).collect();
        rig.store(&r.buffer, 0, &as_bytes).unwrap();
        // Meanwhile the title writes byte 1 of dword 300, which the card
        // did not touch.
        // SAFETY: inside the allocation.
        unsafe { base.add(300 * 4 + 1).write_volatile(0x77) };
        let stop = std::sync::atomic::AtomicBool::new(false);
        let at = base as usize;
        let last = std::thread::scope(|s| {
            let cpu = s.spawn(|| {
                let mut k = 0u8;
                while !stop.load(Ordering::Relaxed) {
                    k = k.wrapping_add(1).max(1);
                    for w in 128..160 {
                        // SAFETY: inside the allocation.
                        unsafe { std::ptr::write_volatile((at as *mut u8).add(w * 4), k) };
                    }
                }
                k
            });
            // SAFETY: an open list of the rig's, after the "shaders"; the
            // resident lives through it.
            let ran = rig.run(|list| unsafe {
                record_merge(list, &pass, &[Merge { resident: &r, ranges: vec![(0, n)] }]);
            });
            stop.store(true, Ordering::Relaxed);
            ran.unwrap();
            cpu.join().unwrap()
        });
        let words = words();
        for w in 0..16 {
            assert_eq!(words[w], gpu[w], "a whole dword the card wrote, dword {w}");
        }
        for w in 64..96 {
            assert_eq!(words[w], gpu[w], "byte 2 the card's, the rest the title's, dword {w}");
        }
        for w in 128..160 {
            assert_eq!(words[w] & 0xff, u32::from(last), "the CPU's last byte 0 is kept, dword {w}");
            assert_eq!(words[w] & 0xffff_ff00, gpu[w] & 0xffff_ff00, "the card's byte 2, dword {w}");
        }
        assert_eq!(words[300] & 0xff00, 0x7700, "a byte only the title wrote is the title's");
        let untouched = |w: usize| !(w < 16 || (64..96).contains(&w) || (128..160).contains(&w) || w == 300);
        assert!((0..K64 / 4).filter(|&w| untouched(w)).all(|w| words[w] == before[w]));
        let twin = rig.fetch(&r.twin, 0, K64).unwrap();
        assert_eq!(twin, as_bytes, "B takes what R holds");
        assert_eq!(pass.counts(), [16 + 32 + 32, 64], "dwords that differed, of them in part");
        drop(r);
    }

    /// Test (b) on the software adapter.
    #[test]
    fn warp_the_merge_stores_only_what_the_card_changed() {
        let Some(device) = crate::inplace::tests::warp() else { return };
        merge_with_the_cpu_writing(&device);
    }

    /// Test (b) on this PC's card (`cargo test -p victory-gpu -- --ignored
    /// merge_on_hardware`): the atomics into the arena go over the bus.
    #[test]
    #[ignore = "opens a device on the card"]
    fn merge_on_hardware() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain device creation on the default adapter.
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).unwrap() };
        merge_with_the_cpu_writing(&device.unwrap());
    }

    /// Test (c): submission 1's shaders write a dword of `R` and its merge
    /// puts it into the title's memory; the title then writes another byte
    /// of the same 64 KiB page; submission 2's upload of that page brings
    /// both into `R` and `B`, and its merge finds nothing to do.
    #[test]
    fn warp_a_merge_lands_before_the_next_upload_of_its_page() {
        let Some(device) = crate::inplace::tests::warp() else { return };
        let Some((base, at)) = title(&device, 2) else { return };
        let r = resident(&device, base, at);
        let (start, n) = (base as u64, 2 * K64 as u64);
        let mut rig = Rig::new(&device).unwrap();
        let pass = Pass::new(&device).unwrap();
        // SAFETY (each run): an open list of the rig's; the resident lives
        // through it.
        rig.run(|list| unsafe {
            record_upload(list, &[Upload { resident: &r, ranges: r.take() }]);
        })
        .unwrap();
        assert!(r.refresh(start, start + n));
        let _ = r.take();
        // Submission 1: a shader writes dword 0x40 of page 1; the merge.
        let x = K64 as u64 + 0x100;
        rig.store(&r.buffer, x, &0x1234_5678u32.to_le_bytes()).unwrap();
        rig.run(|list| unsafe {
            record_merge(list, &pass, &[Merge { resident: &r, ranges: vec![(K64 as u64, n)] }]);
        })
        .unwrap();
        // SAFETY (each): inside the allocation.
        let word = |at: u64| unsafe { base.add(at as usize).cast::<u32>().read_volatile() };
        assert_eq!(word(x), 0x1234_5678, "merged into the title's memory");
        // The title writes a byte of the same page.
        let y = K64 as u64 + 0x2001;
        // SAFETY: inside the allocation.
        unsafe { base.add(y as usize).write_volatile(0xee) };
        // Submission 2: the page the title wrote goes up, whole.
        assert!(r.refresh(start, start + n));
        let ranges = r.take();
        assert_eq!(ranges, [(K64 as u64, n)]);
        rig.run(|list| unsafe {
            record_upload(list, &[Upload { resident: &r, ranges }]);
        })
        .unwrap();
        for buffer in [&r.buffer, &r.twin] {
            let got = rig.fetch(buffer, K64 as u64, K64).unwrap();
            assert_eq!(&got[0x100..0x104], &0x1234_5678u32.to_le_bytes(), "the card's dword");
            assert_eq!(got[0x2001], 0xee, "the title's byte");
        }
        let counted = pass.counts();
        rig.run(|list| unsafe {
            record_merge(list, &pass, &[Merge { resident: &r, ranges: vec![(0, n)] }]);
        })
        .unwrap();
        assert_eq!(pass.counts(), counted, "R and B agree: nothing merged");
        assert_eq!(word(x), 0x1234_5678);
        drop(r);
    }

    /// Test (d): a buffer whose memory is not in an arena gets no resident
    /// (its copy in the card's memory stays, as before), and takes nothing
    /// of the budget; one in an arena gets one, and a buffer inside its
    /// pages shares it at its offset. What is written through one copy of
    /// the same memory goes up to the others, not to the writer.
    #[test]
    fn warp_a_buffer_outside_arenas_keeps_its_copy_and_nested_ones_share() {
        let Some(device) = crate::inplace::tests::warp() else { return };
        let Some((base, at)) = title(&device, 4) else { return };
        let k = K64 as u64;
        let (start, n) = (base as u64, 4 * k);
        let mut arenas = crate::arena::Arenas::default();
        let mut buffer = |bytes| committed(&device, D3D12_HEAP_TYPE_DEFAULT, bytes).ok();
        // Not reported committed: no arena, no resident.
        let none = make(start, n, u64::MAX, &mut |at, len| arenas.resolve_with(&device, at, len, u64::MAX, u64::MAX, &mut |_| ()), &mut buffer);
        assert!(none.is_none());
        assert!(holder(start, start + k).is_none(), "nothing registered");
        // In an arena: a resident, registered.
        arenas.committed(start, n);
        let outer = make(start, n, u64::MAX, &mut |at, len| arenas.resolve_with(&device, at, len, u64::MAX, u64::MAX, &mut |_| ()), &mut buffer).expect("a resident");
        assert_eq!((outer.start, outer.len), (start, n));
        let inner = holder(start + k, start + 3 * k).expect("in its pages");
        assert!(Arc::ptr_eq(&inner, &outer));
        assert!(holder(start, start + n + k).is_none(), "past its pages");
        // A second resident over some of the same pages (one made before
        // a bigger one): writes merged from the outer one go up to it.
        let other = Arc::new(resident(&device, base, at));
        let _ = (outer.take(), other.take());
        let live = [outer.clone(), other.clone()];
        let marked = cross_mark(&live, &[(Some(&*outer), start + k + 8, start + k + 16)]);
        assert_eq!(marked, k);
        assert_eq!(other.take(), [(k, 2 * k)], "the page goes up to the other copy");
        assert_eq!(outer.take(), [], "not to the one that wrote it");
        // Written in place (no resident): up to both.
        assert_eq!(cross_mark(&live, &[(None, start + 3 * k, start + 3 * k + 1)]), 2 * k);
        assert_eq!(outer.take(), [(3 * k, n)]);
        forget(start, start + 1, false);
        assert!(holder(start, start + k).is_none(), "forgotten when let go");
        drop((outer, inner, other, live));
        drop(arenas);
    }

    /// Residents that overlap become one: a big buffer whose pages hold a
    /// resident made before it (an inner buffer used first) and more gets
    /// one resident over both, which takes the older one's place; one over
    /// the gap between two residents joins all three. The replaced ones
    /// stay alive for what holds them, and in step with the new one.
    #[test]
    fn warp_overlapping_residents_become_one() {
        let Some(device) = crate::inplace::tests::warp() else { return };
        let Some((base, _at)) = title(&device, 8) else { return };
        let (k, start) = (K64 as u64, base as u64);
        let mut arenas = crate::arena::Arenas::default();
        arenas.committed(start, 8 * k);
        let mut resolve = |at, len| arenas.resolve_with(&device, at, len, u64::MAX, u64::MAX, &mut |_| ());
        let mut buffer = |bytes| committed(&device, D3D12_HEAP_TYPE_DEFAULT, bytes).ok();
        let mut over = |from: u64, to: u64| make_over(start + from * k, start + to * k, u64::MAX, &mut resolve, &mut buffer).expect("a resident");
        let (inner, replaced) = over(1, 2);
        assert!(replaced.is_empty());
        let (right, replaced) = over(5, 6);
        assert!(replaced.is_empty());
        let (outer, replaced) = over(0, 3);
        assert_eq!((outer.start, outer.len), (start, 3 * k));
        assert!(matches!(replaced.as_slice(), [r] if Arc::ptr_eq(r, &inner)));
        assert!(Arc::ptr_eq(&holder(start + k, start + 2 * k).unwrap(), &outer), "the older one's pages find the new one");
        let (all, replaced) = over(2, 6);
        assert_eq!((all.start, all.len), (start, 6 * k));
        assert_eq!(replaced.len(), 2);
        assert!(replaced.iter().any(|r| Arc::ptr_eq(r, &outer)) && replaced.iter().any(|r| Arc::ptr_eq(r, &right)));
        assert!(Arc::ptr_eq(&holder(start + 5 * k, start + 6 * k).unwrap(), &all));
        assert!(Arc::ptr_eq(&holder(start, start + k).unwrap(), &all));
        // Written through the new one: the page goes up to a replaced one.
        let live = [all.clone(), inner.clone()];
        let _ = (all.take(), inner.take());
        assert_eq!(cross_mark(&live, &[(Some(&*all), start + k + 4, start + k + 8)]), k);
        assert_eq!(inner.take(), [(0, k)]);
        forget(start, start + 8 * k, false);
        drop((inner, right, outer, all, live, replaced));
    }
}
