//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Option<(usize, Vec<u8>)>>,
    /// Local build: the buffer is title memory itself (`crate::direct`): nothing to copy.
    direct: bool,
    /// Local build: the copy lives in the mirror (`crate::mirror`): the CPU address of each
    /// 64 KiB title page from `start`'s on (empty otherwise: one block at `data`).
    pages: Vec<usize>,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address in `buffer` (a mirrored copy's from its first page).
    pub(crate) fn offset(&self, address: u64) -> u64 {
        if self.pages.is_empty() { address - self.start } else { address - (self.start & !(crate::mirror::TILE - 1)) }
    }

    /// CPU address of byte `off` of the copy; valid up to the end of its 64 KiB title page.
    fn ptr(&self, off: usize) -> *mut u8 {
        if self.pages.is_empty() {
            // SAFETY: inside the copy.
            return unsafe { self.data.add(off) };
        }
        const TILE: usize = crate::mirror::TILE as usize;
        let a = self.start as usize + off;
        (self.pages[a / TILE - self.start as usize / TILE] + a % TILE) as *mut u8
    }

    /// Calls `f(offset, cpu address, length)` for each piece of `[from, to)` (offsets in the
    /// copy) that lies in one 64 KiB title page.
    fn pieces(&self, from: usize, to: usize, mut f: impl FnMut(usize, *mut u8, usize)) {
        const TILE: usize = crate::mirror::TILE as usize;
        let mut at = from;
        while at < to {
            let n = if self.pages.is_empty() { to - at } else { (TILE - (self.start as usize + at) % TILE).min(to - at) };
            f(at, self.ptr(at), n);
            at += n;
        }
    }

    /// Appends the copy's bytes `[from, to)` to `out`.
    fn gather(&self, from: usize, to: usize, out: &mut Vec<u8>) {
        // SAFETY: each piece lies in one page of the copy.
        self.pieces(from, to, |_, p, n| out.extend_from_slice(unsafe { std::slice::from_raw_parts(p, n) }));
    }

    /// A copy in the mirror (`crate::mirror`) of `[start, start + len)`.
    fn mirrored(m: &crate::mirror::Mirror, start: u64, len: u64) -> Option<Shadow> {
        use crate::mirror::TILE;
        if !m.covers(start, len) {
            return None;
        }
        let first = start / TILE;
        let pages = m.pages(first, (start + len.max(1)).div_ceil(TILE) - first)?;
        // views and barriers name a buffer of the copy's own over the same pages; shader
        // addresses are the mirror's (reads past the copy find neighbours or zeros)
        let own = m.alias(&pages)?;
        let mut s = Shadow::new(own, std::ptr::null_mut(), m.va + start, start, len);
        s.pages = pages.into_iter().map(|(cpu, _, _)| cpu).collect();
        Some(s)
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.direct {
            // nothing to bring up to date; the list still keeps the buffer alive
            list.defer_refresh(self, address as usize, 0);
            return;
        }
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            direct: false,
            pages: Vec::new(),
        }
    }

    /// A buffer over title memory itself (`crate::direct`).
    fn direct(buffer: ID3D12Resource, gpu: u64, start: u64, len: u64) -> Shadow {
        let mut s = Shadow::new(buffer, start as *mut u8, gpu, start, len);
        s.direct = true;
        s
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.direct {
            return;
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.direct {
            return;
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                self.gather(start, at, &mut grown);
                grown.extend_from_slice(&bytes);
                self.gather(at + bytes.len(), end, &mut grown);
                give_back(bytes);
            }
            None => self.gather(start, end, &mut grown),
        }
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        if self.direct {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            // Local build: which pieces the GPU changed, compared on several threads (one core
            // compared ~100 MB per submission, 27 submissions a frame: ~170 ms).
            let changed = changed_pieces(self, before.as_ptr() as usize, start, from, to);
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                if !changed.contains(&at) {
                    at += n;
                    continue;
                }
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.ptr(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            give_back(bytes);
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.direct {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if !self.pages.is_empty() {
            return self.update_mirrored(from, to);
        }
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Local build: [`Shadow::update_free`] for a copy in the mirror. Its pages are shared by
    /// every copy covering them, so what was copied when is kept per page for all copies
    /// (`MIRROR_COPIED`), and a page checked whole in this submission is not checked again by
    /// another copy (`MIRROR_CHECKED`): the world's copies overlap, and the same 174 MB heap
    /// was checked 20 times a frame (9 GB checked, 879 ms a frame).
    fn update_mirrored(&self, from: usize, to: usize) -> usize {
        let sub = SUBMISSION.load(Ordering::Relaxed);
        let first = from / PAGE;
        let last = (to - 1) / PAGE;
        // runs of pages still to look at, as byte ranges cut to [from, to)
        let mut runs: Vec<(usize, usize)> = Vec::new();
        {
            let checked = lock_map(&MIRROR_CHECKED);
            for page in first..=last {
                let (a, b) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
                let whole = a == page * PAGE && b == (page + 1) * PAGE;
                if whole && checked.get(&page) == Some(&sub) {
                    continue;
                }
                match runs.last_mut() {
                    Some(r) if r.1 == a => r.1 = b,
                    _ => runs.push((a, b)),
                }
            }
        }
        let mut bytes = 0;
        for (a, b) in runs {
            bytes += self.update_run(a, b, sub);
        }
        bytes
    }

    /// One run of [`Shadow::update_mirrored`].
    fn update_run(&self, from: usize, to: usize, sub: u64) -> usize {
        let whole_pages = || (from.div_ceil(PAGE)..to / PAGE);
        let mark = |copied_at: Option<u64>| {
            let mut checked = lock_map(&MIRROR_CHECKED);
            let mut copied = lock_map(&MIRROR_COPIED);
            for page in whole_pages() {
                checked.insert(page, sub);
                if let Some(now) = copied_at {
                    copied.insert(page, now);
                }
            }
        };
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, written)) = crate::watch::writes(from, to, true)
        {
            self.watched.store(1, Ordering::Relaxed);
            // pages written after their last copy (or never copied), as runs
            let stale: Vec<usize> = {
                let copied = lock_map(&MIRROR_COPIED);
                let mut writes = written.iter().peekable();
                let mut out = Vec::new();
                for page in from / PAGE..=(to - 1) / PAGE {
                    let write = match writes.peek() {
                        Some(&&(p, w)) if p == page => {
                            writes.next();
                            w
                        }
                        _ => 0,
                    };
                    let at = copied.get(&page).copied().unwrap_or(0);
                    if at == 0 || write > at {
                        out.push(page);
                    }
                }
                out
            };
            let mut bytes = 0;
            let mut i = 0;
            while i < stale.len() {
                let mut j = i;
                while j + 1 < stale.len() && stale[j + 1] == stale[j] + 1 {
                    j += 1;
                }
                bytes += self.copy((stale[i] * PAGE).max(from), ((stale[j] + 1) * PAGE).min(to));
                i = j + 1;
            }
            mark(Some(now));
            return bytes;
        }
        self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        self.copy(from, to);
        mark(None);
        to - from
    }

    /// Local build: whether this copy lives in the mirror (`crate::mirror`).
    pub(crate) fn is_mirrored(&self) -> bool {
        !self.pages.is_empty()
    }

    /// Local build: whether shaders of a submission not yet copied back may be writing it.
    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // pieces within one page of the copy each
        self.pieces(from - start, to - start, |off, p, n| {
            crate::host::copy_mapped(p, (start + off) as *const u8, n, false);
        });
        to - from
    }
}

/// Offsets (from the copy's start) of the 64 KiB pieces of `[from, to)` whose bytes in the copy
/// at `data` differ from `before` (the snapshot of `[from, to)`), compared on several threads.
fn changed_pieces(shadow: &Shadow, before: usize, start: usize, from: usize, to: usize) -> std::collections::HashSet<usize> {
    const PIECE: usize = 64 << 10;
    let mut pieces = Vec::new();
    let mut at = from;
    while at < to {
        let n = (PIECE - (start + at) % PIECE).min(to - at);
        pieces.push((at, n));
        at += n;
    }
    let pieces: Vec<(usize, usize, usize)> = pieces.iter().map(|&(at, n)| (at, n, shadow.ptr(at) as usize)).collect();
    let differs = move |&(at, n, host): &(usize, usize, usize)| -> bool {
        // SAFETY: inside the copy (one page) and the snapshot.
        let host = unsafe { std::slice::from_raw_parts(host as *const u8, n) };
        let old = unsafe { std::slice::from_raw_parts((before + at - from) as *const u8, n) };
        host != old
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 12);
    if pieces.len() < 128 || threads < 2 {
        return pieces.iter().filter(|p| differs(p)).map(|p| p.0).collect();
    }
    // Use the persistent pool to avoid OS thread spawn overhead.
    let results = crate::host::pool_foreach_bool(&pieces, differs);
    pieces.iter().zip(results).filter_map(|(p, diff)| diff.then_some(p.0)).collect()
}

/// Local build: submissions so far (`next_submission`), and per mirror page (64 KiB title
/// page) the submission that last checked it whole and the write-watch asking it was last
/// copied after (see `Shadow::update_mirrored`).
static SUBMISSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static MIRROR_CHECKED: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
static MIRROR_COPIED: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);

/// Local build: brings title memory `[from, to)` up to date in the mirror's pages, for every
/// copy covering it at once (`host`'s batch merges the copies' refreshes into such ranges).
pub(crate) fn refresh_mirror(from: usize, to: usize) -> usize {
    use crate::mirror::TILE;
    let Some(m) = crate::mirror::get() else { return 0 };
    if to <= from || !m.covers(from as u64, (to - from) as u64) {
        return 0;
    }
    let first = from as u64 / TILE;
    let Some(pages) = m.pages(first, (to as u64).div_ceil(TILE) - first) else { return 0 };
    let mut view = Shadow::new(m.buffer.clone(), std::ptr::null_mut(), m.va + from as u64, from as u64, (to - from) as u64);
    view.pages = pages.into_iter().map(|(cpu, _, _)| cpu).collect();
    view.update_free(from, to)
}

/// A new submission begins: pages must be checked again.
pub(crate) fn next_submission() {
    SUBMISSION.fetch_add(1, Ordering::Relaxed);
}

/// Local build: title memory mapped anew: its pages are stale for every copy.
pub(crate) fn mirror_pages_remapped(at: usize, len: usize) {
    let (first, last) = (at / PAGE, (at + len.max(1) - 1) / PAGE);
    // (always CHECKED before COPIED, as `update_run` takes them)
    let mut checked = lock_map(&MIRROR_CHECKED);
    let mut copied = lock_map(&MIRROR_COPIED);
    for page in first..=last {
        copied.remove(&page);
        checked.remove(&page);
    }
}

struct MapGuard<'a>(std::sync::MutexGuard<'a, Option<std::collections::HashMap<usize, u64>>>);
impl std::ops::Deref for MapGuard<'_> {
    type Target = std::collections::HashMap<usize, u64>;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("made")
    }
}
impl std::ops::DerefMut for MapGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("made")
    }
}
fn lock_map(m: &'static Mutex<Option<std::collections::HashMap<usize, u64>>>) -> MapGuard<'static> {
    let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        *g = Some(std::collections::HashMap::new());
    }
    MapGuard(g)
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    // direct buffers over memory mapped anew show the old pages: made again
    let remapped = crate::direct::take_remapped();
    if !remapped.is_empty() {
        for e in map.values_mut() {
            if e.shadow.as_ref().is_some_and(|s| s.direct && remapped.iter().any(|&(a, b)| a < s.start + s.len && b > s.start)) {
                e.shadow = None;
            }
        }
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|_, s| !(s.direct && remapped.iter().any(|&(a, b)| a < s.start + s.len && b > s.start)));
    }
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        if let Some((buffer, gpu)) = crate::direct::buffer(&host()?.device, key.0, len) {
            entry.shadow = Some(Arc::new(Shadow::direct(buffer, gpu, key.0, len)));
            return entry.shadow.clone();
        }
        if let Some(s) = crate::mirror::get().and_then(|m| Shadow::mirrored(m, key.0, len)) {
            entry.shadow = Some(Arc::new(s));
            return entry.shadow.clone();
        }
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        // Local build: 1 MiB past the end: shaders read past a buffer's end (memory goes on
        // there on the console; here a GPU page fault removed the device)
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16) + (1 << 20))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// Local build: the extent a copy of loose memory covers: the whole run of reserved or mapped
/// memory around `address` when it is at most `SPAN` (every address in it shares the copy),
/// else the `SPAN`-aligned piece holding `[address, end)`. Shaders index far past the address
/// they start from (8 MB past a 4 MB window removed the device).
fn loose_extent(address: u64, end: u64) -> (u64, u64) {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    const SPAN: u64 = 128 << 20;
    let query = |at: u64| -> Option<MEMORY_BASIC_INFORMATION> {
        // SAFETY: a query only.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(at as *const _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        (n != 0 && info.State != MEM_FREE).then_some(info)
    };
    let fallback = || {
        let start = address & !(SPAN - 1);
        (start, end.next_multiple_of(WINDOW).max(start + WINDOW) - start)
    };
    static RUN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*RUN.get_or_init(|| std::env::var("VICTORY_LOOSE_RUN").is_ok_and(|v| v.trim() == "1")) {
        let start = address & !(WINDOW - 1);
        return (start, end.next_multiple_of(WINDOW) - start);
    }
    let Some(here) = query(address) else { return fallback() };
    let (mut lo, mut hi) = (here.BaseAddress as u64, here.BaseAddress as u64 + here.RegionSize as u64);
    while address - lo < SPAN && lo > 0 {
        match query(lo - 1) {
            Some(i) => lo = i.BaseAddress as u64,
            None => break,
        }
    }
    while hi - address < SPAN {
        match query(hi) {
            Some(i) => hi = i.BaseAddress as u64 + i.RegionSize as u64,
            None => break,
        }
    }
    if hi - lo <= SPAN && end <= hi {
        (lo, hi - lo)
    } else {
        fallback()
    }
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let (start, len) = loose_extent(address, end);
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    // (no direct heap here: a run of loose memory takes in the title's CPU structures, and
    // reading those in place made it crash at start)
    if let Some(s) = crate::mirror::get().and_then(|m| Shadow::mirrored(m, start, len)) {
        let shadow = Arc::new(s);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // Local build: a whole window more past the end: shaders read across a window's end (a GPU
    // page fault 1.1 MB past one removed the device)
    let (buffer, data) = host()?.shared_buffer(len + WINDOW)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    {
        static TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let total = TOTAL.fetch_add(len + WINDOW, std::sync::atomic::Ordering::Relaxed) + len + WINDOW;
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 8 || n.is_power_of_two() {
            tracing::info!(n, start = format_args!("{start:#x}"), len, total_mb = total >> 20, "loose title memory copy");
        }
    }
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}
