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
    before: Mutex<Option<Snap>>,
    /// Local build: the buffer is title memory itself (`crate::direct`): nothing to copy.
    direct: bool,
    /// Local build: the copy lives in the mirror (`crate::mirror`): the CPU address of each
    /// 64 KiB title page from `start`'s on (empty otherwise: one block at `data`).
    pages: Vec<usize>,
}

/// Local build: a snapshot of a copy's bytes, indexed like the copy (byte `off` of the copy is
/// `buf[off]`); only `[from, to)` is filled in, and `buf`'s length stays 0 (its capacity is
/// reserved address space: untouched parts cost nothing). Growing a snapshot gathers only the
/// newly covered parts, in place: it used to be rebuilt (copied whole) at every growth, dozens
/// of times a submission for the world's 174 MB heap.
struct Snap {
    buf: Vec<u8>,
    from: usize,
    to: usize,
}

impl Snap {
    fn ptr(&self) -> *mut u8 {
        self.buf.as_ptr().cast_mut()
    }
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
        if self.direct {
            // SAFETY: the placed resource is retained by this shadow.
            let base = unsafe { self.buffer.GetGPUVirtualAddress() };
            self.gpu(address) - base
        } else if self.pages.is_empty() {
            address - self.start
        } else {
            address - (self.start & !(crate::mirror::TILE - 1))
        }
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

    /// Copies the copy's bytes `[from, to)` to the same offsets from `base` (a snapshot buffer),
    /// on the pool's threads when there are many.
    fn gather(&self, base: usize, from: usize, to: usize) {
        if from >= to {
            return;
        }
        const CHUNK: usize = 2 << 20;
        let one = |from: usize, to: usize| {
            // SAFETY: each piece lies in one page of the copy; `base` holds the offsets.
            self.pieces(from, to, |off, p, n| unsafe { std::ptr::copy_nonoverlapping(p, (base + off) as *mut u8, n) });
        };
        if to - from < 4 * CHUNK {
            return one(from, to);
        }
        let chunks: Vec<(usize, usize)> = (from..to).step_by(CHUNK).map(|a| (a, (a + CHUNK).min(to))).collect();
        crate::host::pool().scope(&chunks, |&(a, b)| -> bool {
            one(a, b);
            true
        });
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
        if self.is_mirrored() && from < to {
            // Other copies share these mirror pages: their refreshes must leave them alone until
            // the copy-back (`Shadow::update_mirrored`).
            let at = self.start as usize;
            lock_pending().push((at + from, at + to));
        }
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (start, end) = match &*before {
            Some(s) => (s.from.min(from), s.to.max(to)),
            None => (from, to),
        };
        match before.as_mut() {
            Some(s) if (s.from, s.to) == (start, end) => return,
            Some(s) => {
                if s.buf.capacity() < end {
                    // (only for copies too big to reserve whole: see below)
                    let mut bigger = pooled(end);
                    // SAFETY: both hold the range; the new one is larger.
                    unsafe { std::ptr::copy_nonoverlapping(s.ptr().add(s.from), bigger.as_mut_ptr().add(s.from), s.to - s.from) };
                    give_back(std::mem::replace(&mut s.buf, bigger));
                }
                let base = s.ptr() as usize;
                self.gather(base, start, s.from);
                self.gather(base, s.to, end);
                (s.from, s.to) = (start, end);
            }
            None => {
                // Reserved whole (not touched): room to grow in place. Past 512 MB only what
                // is needed, grown by a copy when more is asked.
                let want = if self.len <= 512 << 20 { self.len as usize } else { end };
                let buf = pooled(want.max(end));
                self.gather(buf.as_ptr() as usize, start, end);
                *before = Some(Snap { buf, from: start, to: end });
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::SnapUs, started.elapsed().as_micros() as u64);
        }
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
        if let Some(snap) = &*snapshot
            && snap.from <= from
            && to <= snap.to
            && from < to
        {
            // (`[from, to)` is inside the snapshot's filled part; nothing else touches it while
            // the lock is held)
            let base = snap.ptr() as usize;
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut scratch = Scratch::new();
            let mut synced = Vec::new();
            // Local build: which pieces the GPU changed, compared on several threads (one core
            // compared ~100 MB per submission, 27 submissions a frame: ~170 ms).
            let changed = changed_pieces(self, base, start, from, to);
            let sync = !self.is_mirrored();
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
                // SAFETY: inside the snapshot's filled part; the lock is held, so it may be updated.
                let old = unsafe { std::slice::from_raw_parts_mut((base + at) as *mut u8, n) };
                if let Some(w) = back_piece(start + at, host, old, &mut scratch, sync, back_mode()) {
                    written += w;
                    if sync {
                        synced.push((start + at) / PAGE);
                    }
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
            && let Some(snap) = snapshot.take()
        {
            give_back(snap.buf);
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
        // Not over ranges any copy's shaders of this submission write: that copy's snapshot
        // tells what the GPU wrote, and title bytes brought in after it would be taken for the
        // GPU's and written back over newer title writes.
        let pending = lock_pending().clone();
        if pending.iter().any(|&(a, e)| a < to && e > from) {
            return gaps(from, to, pending).into_iter().map(|(a, b)| self.update_mirrored_free(a, b)).sum();
        }
        self.update_mirrored_free(from, to)
    }

    /// [`Shadow::update_mirrored`] of a range no shader of the submission writes.
    fn update_mirrored_free(&self, from: usize, to: usize) -> usize {
        let sub = SUBMISSION.load(Ordering::Relaxed);
        let first = from / PAGE;
        let last = (to - 1) / PAGE;
        // runs of pages still to look at, as byte ranges cut to [from, to)
        let mut runs: Vec<(usize, usize)> = Vec::new();
        {
            let tables = mirror_pages();
            for page in first..=last {
                let (a, b) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
                let whole = a == page * PAGE && b == (page + 1) * PAGE;
                if whole && tables.get(page).0 == sub {
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
            let mut tables = mirror_pages();
            for page in whole_pages() {
                let e = tables.at(page);
                e.0 = sub;
                if let Some(now) = copied_at {
                    e.1 = now;
                }
            }
        };
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, written)) = crate::watch::writes(from, to, true)
        {
            self.watched.store(1, Ordering::Relaxed);
            // pages written after their last copy (or never copied), as runs
            let stale: Vec<usize> = {
                let tables = mirror_pages();
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
                    let at = tables.get(page).1;
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
/// at `data` differ from `before` (the snapshot, indexed by offset), compared on several threads.
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
        let old = unsafe { std::slice::from_raw_parts((before + at) as *const u8, n) };
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
/// (the two per-page numbers live in one flat table, `MIRROR_PAGES`)
///
/// Local build: a two-level table, 4,096 pages (256 MB of title memory) per chunk, of
/// `(checked, copied)` per page. They were two `HashMap`s: ~85,000 lookups and inserts a frame.
struct PageTable {
    /// Indexed by chunk; the mirror covers 2M pages (512 chunks), so this stays small.
    chunks: Vec<Option<Box<[(u64, u64); PAGE_CHUNK]>>>,
}
const PAGE_CHUNK: usize = 4096;

impl PageTable {
    fn get(&self, page: usize) -> (u64, u64) {
        match self.chunks.get(page / PAGE_CHUNK) {
            Some(Some(c)) => c[page % PAGE_CHUNK],
            _ => (0, 0),
        }
    }
    fn at(&mut self, page: usize) -> &mut (u64, u64) {
        let k = page / PAGE_CHUNK;
        if self.chunks.len() <= k {
            self.chunks.resize_with(k + 1, || None);
        }
        let c = self.chunks[k].get_or_insert_with(|| Box::new([(0, 0); PAGE_CHUNK]));
        &mut c[page % PAGE_CHUNK]
    }
    /// Forgets pages `[first, last]` (what is not in the table is already forgotten).
    fn forget(&mut self, first: usize, last: usize) {
        let hi = (last / PAGE_CHUNK).min(self.chunks.len().saturating_sub(1));
        for k in first / PAGE_CHUNK..=hi {
            if let Some(Some(c)) = self.chunks.get_mut(k) {
                for page in (k * PAGE_CHUNK).max(first)..=((k + 1) * PAGE_CHUNK - 1).min(last) {
                    c[page % PAGE_CHUNK] = (0, 0);
                }
            }
        }
    }
}

static MIRROR_PAGES: Mutex<Option<PageTable>> = Mutex::new(None);

struct PagesLock(std::sync::MutexGuard<'static, Option<PageTable>>);
impl std::ops::Deref for PagesLock {
    type Target = PageTable;
    fn deref(&self) -> &PageTable {
        self.0.as_ref().expect("made")
    }
}
impl std::ops::DerefMut for PagesLock {
    fn deref_mut(&mut self) -> &mut PageTable {
        self.0.as_mut().expect("made")
    }
}
fn mirror_pages() -> PagesLock {
    let mut g = MIRROR_PAGES.lock().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(|| PageTable { chunks: Vec::new() });
    PagesLock(g)
}

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

/// Title ranges that mirrored copies' shaders write in the submission being prepared
/// (`Shadow::writing`): refreshes of other copies over the same mirror pages leave them alone.
static MIRROR_PENDING: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

fn lock_pending() -> std::sync::MutexGuard<'static, Vec<(usize, usize)>> {
    MIRROR_PENDING.lock().unwrap_or_else(|e| e.into_inner())
}

/// A submission's preparation begins (the previous one is copied back): no mirrored range
/// is being written.
pub(crate) fn submission_begins() {
    lock_pending().clear();
}

/// The parts of `[from, to)` outside every range of `holes`, in order.
fn gaps(from: usize, to: usize, mut holes: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    holes.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, e) in holes {
        if e <= at || a >= to {
            continue;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(e);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// The runs (offsets in the piece, end exclusive) of the dwords of a piece at title address
/// `address` whose bytes in `host` differ from `old`: what the GPU wrote. Dwords are aligned
/// to title memory; a piece's partial first and last dwords count by themselves.
#[cfg(test)]
fn gpu_runs(address: usize, host: &[u8], old: &[u8]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    gpu_runs_into(address, host, old, &mut runs);
    runs
}

/// The runs of changed dwords, as `gpu_runs` gives them, into `runs` (cleared first). Equal
/// stretches are skipped 32 bytes at a time (fixed-size compares the compiler makes vector
/// compares) instead of dword by dword.
fn gpu_runs_into(address: usize, host: &[u8], old: &[u8], runs: &mut Vec<(usize, usize)>) {
    runs.clear();
    let n = host.len().min(old.len());
    let (host, old) = (&host[..n], &old[..n]);
    let mut push = |a: usize, e: usize| match runs.last_mut() {
        Some(r) if r.1 == a => r.1 = e,
        _ => runs.push((a, e)),
    };
    // one title dword (or the part of it in the piece) from `at`
    let dword = |at: usize, push: &mut dyn FnMut(usize, usize)| -> usize {
        let end = ((address + at) / 4 * 4 + 4 - address).min(n);
        if host[at..end] != old[at..end] {
            push(at, end);
        }
        end
    };
    let mut at = 0;
    while at < n && !(address + at).is_multiple_of(8) {
        at = dword(at, &mut push);
    }
    // `at` is 8-aligned in title memory from here on, so each 4 bytes are a whole dword
    while at + 32 <= n {
        let h: &[u8; 32] = host[at..at + 32].try_into().unwrap();
        let o: &[u8; 32] = old[at..at + 32].try_into().unwrap();
        if h != o {
            for k in (0..32).step_by(4) {
                if h[k..k + 4] != o[k..k + 4] {
                    push(at + k, at + k + 4);
                }
            }
        }
        at += 32;
    }
    while at < n {
        at = dword(at, &mut push);
    }
}

/// How the copy-back writes title memory.
#[derive(Clone, Copy, PartialEq, Debug)]
enum BackMode {
    /// One `WriteProcessMemory` per run of changed dwords (8867c99's way; ~10 us a run: the
    /// world's ~100,000 runs a frame took 1.7 to 5 s).
    Kernel,
    /// Plain stores where the piece is committed read-write (one query per piece written),
    /// the kernel elsewhere (read-only or otherwise protected title memory).
    Direct,
}

/// Local build: `VICTORY_COPY_BACK_KERNEL=1` writes every run through the kernel again.
fn back_mode() -> BackMode {
    static MODE: std::sync::OnceLock<BackMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        if std::env::var("VICTORY_COPY_BACK_KERNEL").is_ok_and(|v| v.trim() == "1") { BackMode::Kernel } else { BackMode::Direct }
    })
}

/// The last title region found committed and read-write (`[from, to)`), kept for the pieces
/// after it: a query costs about as much as walking the region (~80 us for a 64 MB heap, the
/// time 1,000 pieces' worth of runs take to write). Only read-write answers are kept: a block
/// the write guards (`crate::watch`) opened since is asked again.
#[derive(Default)]
struct Writable {
    from: usize,
    to: usize,
}

impl Writable {
    /// Whether all of `[at, at + n)` is committed and read-write (a plain store cannot fault
    /// there except into the write guards' handler, which opens the block).
    fn covers(&mut self, at: usize, n: usize) -> bool {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE, VirtualQuery,
        };
        let end = at + n;
        let mut p = at;
        while p < end {
            if (self.from..self.to).contains(&p) {
                p = self.to;
                continue;
            }
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) } == 0 {
                return false;
            }
            let base = info.Protect & 0xff;
            let next = info.BaseAddress as usize + info.RegionSize;
            if info.State != MEM_COMMIT
                || info.Protect & PAGE_GUARD != 0
                || (base != PAGE_READWRITE && base != PAGE_EXECUTE_READWRITE)
                || next <= p
            {
                return false;
            }
            (self.from, self.to) = (info.BaseAddress as usize, next);
            p = next;
        }
        true
    }
}

/// What [`back_piece`] reuses from piece to piece of a copy-back.
struct Scratch {
    /// Room for a piece's title bytes.
    title: Vec<u8>,
    runs: Vec<(usize, usize)>,
    writable: Writable,
}

impl Scratch {
    fn new() -> Scratch {
        Scratch { title: vec![0; 64 << 10], runs: Vec::new(), writable: Writable::default() }
    }
}

/// The copy-back of one piece (inside one 64 KiB title page) at title address `at`: `host` is
/// the copy after the submission, `old` the snapshot before it (at most 64 KiB). Writes into title memory only the runs of dwords the GPU changed that title
/// memory does not already hold, each run by itself: the title's threads keep writing the rest
/// of the piece while we work (a whole 64 KiB piece read, patched and written back lost their
/// writes made in between: the world's root UAVs reach to the end of big heaps the title's
/// own records live in). With `sync`, the copy then takes the title's bytes (its own buffer:
/// no other copy compares against it). The snapshot takes the copy's bytes: another range of
/// the copy declared in the same submission finds nothing more to write here.
///
/// `None` when nothing was done (the GPU wrote nothing here, or title memory is unreadable);
/// else the bytes written.
fn back_piece(
    at: usize,
    host: &mut [u8],
    old: &mut [u8],
    scratch: &mut Scratch,
    sync: bool,
    mode: BackMode,
) -> Option<usize> {
    let n = host.len();
    let Scratch { title, runs, writable } = scratch;
    let title = &mut title[..n];
    if host == old || !crate::host::read_into(at as *const u8, title) {
        return None;
    }
    gpu_runs_into(at, host, old, runs);
    let mut written = 0;
    // decided at the first run to write: plain stores, or the kernel
    let mut direct = None;
    for &(a, b) in runs.iter() {
        if title[a..b] == host[a..b] {
            continue;
        }
        let direct = *direct.get_or_insert_with(|| {
            crate::watch::writing(at, n);
            mode == BackMode::Direct && writable.covers(at, n)
        });
        if direct {
            // SAFETY: committed read-write title memory (just asked; the title does not unmap
            // memory a submission it waits for writes), our own copy as the source.
            unsafe { std::ptr::copy_nonoverlapping(host[a..b].as_ptr(), (at + a) as *mut u8, b - a) };
        } else {
            crate::host::write_mapped((at + a) as *mut u8, host[a..b].as_ptr(), b - a);
        }
        title[a..b].copy_from_slice(&host[a..b]);
        written += b - a;
    }
    if sync {
        host.copy_from_slice(title);
    }
    old.copy_from_slice(host);
    Some(written)
}

/// The current submission's number (see [`next_submission`]).
pub(crate) fn current_submission() -> u64 {
    SUBMISSION.load(Ordering::Relaxed)
}

/// A new submission begins: pages must be checked again.
pub(crate) fn next_submission() {
    SUBMISSION.fetch_add(1, Ordering::Relaxed);
}

/// Local build: title memory mapped anew: its pages are stale for every copy.
pub(crate) fn mirror_pages_remapped(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len.max(1) - 1) / PAGE);
    mirror_pages().forget(first, last);
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
    if let Some(what) = crate::mirror::describe(va) {
        return format!("; inside {what}");
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_runs_are_the_changed_dwords_only() {
        let old = [0u8; 32];
        let mut host = old;
        assert!(gpu_runs(0x1000, &host, &old).is_empty());
        // one byte of dword 1, all of dwords 4 and 5, one byte of dword 7
        host[5] = 1;
        host[16..24].fill(7);
        host[31] = 9;
        assert_eq!(gpu_runs(0x1000, &host, &old), [(4, 8), (16, 24), (28, 32)]);
        // a piece starting mid-dword: its first two bytes are a dword by themselves
        let mut host = [0u8; 10];
        host[1] = 1;
        host[2] = 1;
        assert_eq!(gpu_runs(0x1002, &host, &[0u8; 10]), [(0, 6)]);
        host[1] = 0;
        assert_eq!(gpu_runs(0x1002, &host, &[0u8; 10]), [(2, 6)]);
        let mut host = [0u8; 10];
        host[9] = 1;
        assert_eq!(gpu_runs(0x1002, &host, &[0u8; 10]), [(6, 10)]);
    }

    /// The copy-back's rule on a piece: title bytes the GPU did not change are never written,
    /// so a title write made while the piece is being copied back survives.
    #[test]
    fn copy_back_leaves_bytes_the_gpu_did_not_write() {
        let old = vec![0u8; 64];
        let mut host = old.clone();
        host[8..12].copy_from_slice(&[1, 2, 3, 4]);
        // title memory as read for the copy-back, then written by a title thread meanwhile
        let mut title = vec![5u8; 64];
        let runs = gpu_runs(0x2000, &host, &old);
        title[40] = 0xee;
        for (a, b) in runs {
            title[a..b].copy_from_slice(&host[a..b]);
        }
        assert_eq!(&title[8..12], &[1, 2, 3, 4]);
        assert_eq!(title[40], 0xee);
        assert!(title[..8].iter().chain(&title[12..40]).all(|&b| b == 5));
    }

    /// 8867c99's dword walk, the reference the faster one must match.
    fn gpu_runs_reference(address: usize, host: &[u8], old: &[u8]) -> Vec<(usize, usize)> {
        let n = host.len().min(old.len());
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut at = 0;
        while at < n {
            let end = ((address + at) / 4 * 4 + 4 - address).min(n);
            if host[at..end] != old[at..end] {
                match runs.last_mut() {
                    Some(r) if r.1 == at => r.1 = end,
                    _ => runs.push((at, end)),
                }
            }
            at = end;
        }
        runs
    }

    /// A small deterministic generator (no dependency).
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
    }

    #[test]
    fn fast_gpu_runs_match_the_reference() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..3000 {
            let n = 1 + rng.below(300);
            let address = 0x1_0000 + rng.below(16);
            let old: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            let mut host = old.clone();
            for _ in 0..rng.below(1 + round % 12) {
                let i = rng.below(n);
                host[i] = host[i].wrapping_add(1 + rng.below(255) as u8);
            }
            let mut runs = Vec::new();
            gpu_runs_into(address, &host, &old, &mut runs);
            assert_eq!(runs, gpu_runs_reference(address, &host, &old), "n {n} at {address:#x}");
        }
    }

    /// Title memory for the tests: committed read-write pages of our own (never freed).
    fn title_memory(len: usize) -> &'static mut [u8] {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        // SAFETY: a fresh allocation, never freed.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            assert!(!p.is_null());
            std::slice::from_raw_parts_mut(p.cast(), len)
        }
    }

    /// Both ways of writing put the same bytes in title memory: the changed dwords title memory
    /// did not hold already, nothing else. Read-only title memory goes through the kernel, which
    /// leaves it alone in both.
    #[test]
    fn back_piece_writes_the_same_bytes_either_way() {
        const N: usize = 64 << 10;
        for protect in [false, true] {
            let mut results = Vec::new();
            for mode in [BackMode::Kernel, BackMode::Direct] {
                let title = title_memory(N);
                let mut rng = Rng(7);
                for b in title.iter_mut() {
                    *b = rng.next() as u8;
                }
                let mut old = title.to_vec();
                let mut host = old.clone();
                for _ in 0..500 {
                    let i = rng.below(N);
                    host[i] ^= 0x5a;
                }
                // a dword the title already holds with the GPU's value: not written
                host[100..104].copy_from_slice(&[9, 9, 9, 9]);
                title[100..104].copy_from_slice(&[9, 9, 9, 9]);
                if protect {
                    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
                    let mut was = 0;
                    // SAFETY: our own allocation.
                    unsafe { VirtualProtect(title.as_ptr().cast(), N, PAGE_READONLY, &mut was) };
                    assert!(!Writable::default().covers(title.as_ptr() as usize, N));
                }
                // (the kernel does not write read-only memory either: it stays as it was)
                let mut expected = title.to_vec();
                if !protect {
                    for (a, b) in gpu_runs_reference(title.as_ptr() as usize, &host, &old) {
                        expected[a..b].copy_from_slice(&host[a..b]);
                    }
                }
                let at = title.as_ptr() as usize;
                let w = back_piece(at, &mut host, &mut old, &mut Scratch::new(), true, mode).unwrap();
                let wrong = title.iter().zip(&expected).position(|(a, b)| a != b);
                assert!(wrong.is_none(), "{mode:?}, protect {protect}: byte {wrong:?}");
                // (read-only: the copy takes what we tried to write, as in 8867c99)
                if !protect {
                    assert_eq!(host, expected, "the copy takes the title's bytes");
                }
                assert_eq!(old, host, "the snapshot takes the copy's");
                results.push((w, title.to_vec()));
            }
            assert!(results[0] == results[1], "protect {protect}");
        }
    }

    /// Title threads writing a piece's other dwords while it is copied back keep their writes
    /// (what 8867c99 fixed stays fixed with plain stores).
    #[test]
    fn concurrent_title_writes_outside_the_changed_dwords_survive() {
        use std::sync::atomic::{AtomicBool, Ordering};
        const N: usize = 64 << 10;
        for mode in [BackMode::Direct, BackMode::Kernel] {
            for round in 0..20u8 {
                let title = title_memory(N);
                let at = title.as_ptr() as usize;
                let old = vec![0u8; N];
                let mut host = old.clone();
                // the GPU wrote every 16th dword
                for d in (0..N / 4).step_by(16) {
                    host[d * 4..d * 4 + 4].copy_from_slice(&[0xa0, 0xa1, 0xa2, round]);
                }
                let stop = AtomicBool::new(false);
                std::thread::scope(|s| {
                    let stop = &stop;
                    // a title thread: the other dwords, over and over, last with `round + 1`
                    s.spawn(move || {
                        let mut v = 0u8;
                        loop {
                            let last = stop.load(Ordering::Acquire);
                            let value = if last { round + 1 } else { v };
                            for d in (0..N / 4).filter(|d| d % 16 != 0) {
                                // SAFETY: our own memory; volatile so each store happens.
                                unsafe { std::ptr::write_volatile((at + d * 4) as *mut u32, u32::from(value) * 0x0101_0101) };
                            }
                            if last {
                                break;
                            }
                            v = v.wrapping_add(1);
                        }
                    });
                    for _ in 0..4 {
                        let (mut h, mut o) = (host.clone(), old.clone());
                        back_piece(at, &mut h, &mut o, &mut Scratch::new(), false, mode);
                    }
                    stop.store(true, Ordering::Release);
                });
                for d in 0..N / 4 {
                    let got = &title[d * 4..d * 4 + 4];
                    if d % 16 == 0 {
                        assert_eq!(got, &[0xa0, 0xa1, 0xa2, round], "{mode:?}: GPU dword {d}");
                    } else {
                        assert_eq!(got, &[round + 1; 4], "{mode:?}: title dword {d} lost its write");
                    }
                }
            }
        }
    }

    /// Copy-back of 64 MB with scattered GPU writes: 8867c99's way (byte-slice dword walk, one
    /// `WriteProcessMemory` a run) against now; title memory restored before each pass.
    /// `cargo test -p victory-gpu --release --lib copy_back_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    #[allow(clippy::type_complexity)]
    fn copy_back_cost() {
        const LEN: usize = 64 << 20;
        const PIECE: usize = 64 << 10;
        let title = title_memory(LEN);
        let at = title.as_ptr() as usize;
        let mut rng = Rng(42);
        for b in title.iter_mut() {
            *b = rng.next() as u8;
        }
        let pristine = title.to_vec();
        let old = pristine.clone();
        // how many of the 1,024 pieces the GPU wrote, how many dwords in each
        for (pieces, dwords) in [(1024, 16), (1024, 128), (256, 1024), (64, 4096), (8, 16384), (0, 0)] {
            let mut host = old.clone();
            let mut rng = Rng(pieces as u64 * 31 + dwords as u64 + 1);
            let mut chosen: Vec<usize> = (0..LEN / PIECE).collect();
            for i in 0..chosen.len() {
                let j = i + rng.below(chosen.len() - i);
                chosen.swap(i, j);
            }
            for &p in &chosen[..pieces] {
                for _ in 0..dwords {
                    host[p * PIECE + rng.below(PIECE / 4) * 4] ^= 0xff;
                }
            }
            let mut pass = |label: &str, f: &mut dyn FnMut(usize, &mut [u8], &mut [u8], &mut [u8]) -> usize| {
                title.copy_from_slice(&pristine);
                let (mut h, mut o) = (host.clone(), old.clone());
                let mut scratch = vec![0u8; PIECE];
                let t = std::time::Instant::now();
                let mut written = 0;
                for p in (0..LEN).step_by(PIECE) {
                    written += f(at + p, &mut h[p..p + PIECE], &mut o[p..p + PIECE], &mut scratch);
                }
                let took = t.elapsed();
                eprintln!("  {label}: {took:?} ({written} bytes written)");
                (took, title.to_vec())
            };
            eprintln!("64 MB, {pieces} pieces changed, ~{dwords} dwords each:");
            let (before, a) = pass("before (8867c99)", &mut |at, host, old, title| {
                if host == old || !crate::host::read_into(at as *const u8, title) {
                    return 0;
                }
                let mut w = 0;
                let mut opened = false;
                for (a, b) in gpu_runs_reference(at, host, old) {
                    if title[a..b] == host[a..b] {
                        continue;
                    }
                    if !opened {
                        crate::watch::writing(at, host.len());
                        opened = true;
                    }
                    crate::host::write_mapped((at + a) as *mut u8, host[a..b].as_ptr(), b - a);
                    title[a..b].copy_from_slice(&host[a..b]);
                    w += b - a;
                }
                host.copy_from_slice(title);
                old.copy_from_slice(host);
                w
            });
            let mut scratch = Scratch::new();
            let (after, b) = pass("after", &mut |at, host, old, _| {
                back_piece(at, host, old, &mut scratch, true, BackMode::Direct).unwrap_or(0)
            });
            assert!(a == b, "the same bytes either way");
            eprintln!("  {:.1}x faster", before.as_secs_f64() / after.as_secs_f64().max(1e-9));
        }
    }

    #[test]
    fn gaps_skip_pending_ranges() {
        assert_eq!(gaps(0, 100, vec![]), [(0, 100)]);
        assert_eq!(gaps(0, 100, vec![(40, 60), (10, 20)]), [(0, 10), (20, 40), (60, 100)]);
        assert_eq!(gaps(0, 100, vec![(0, 50), (30, 120)]), []);
        assert_eq!(gaps(10, 20, vec![(0, 5), (25, 30)]), [(10, 20)]);
        assert_eq!(gaps(10, 20, vec![(15, 25)]), [(10, 15)]);
    }

    #[test]
    fn page_table_remembers_and_forgets() {
        let mut t = PageTable { chunks: Vec::new() };
        assert_eq!(t.get(5), (0, 0));
        *t.at(5) = (3, 4);
        *t.at(9000) = (7, 8);
        assert_eq!((t.get(5), t.get(9000), t.get(6)), ((3, 4), (7, 8), (0, 0)));
        t.forget(0, 100);
        assert_eq!((t.get(5), t.get(9000)), ((0, 0), (7, 8)));
        t.forget(8000, usize::MAX / PAGE);
        assert_eq!(t.get(9000), (0, 0));
    }

    /// The world's per-frame table traffic (~85,000 lookups and inserts): flat table against the
    /// maps it replaced. `cargo test -p victory-gpu --lib page_table_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn page_table_cost() {
        let pages: Vec<usize> = (0..2800).map(|i| 0x10000 + i).collect();
        let t = std::time::Instant::now();
        let mut a = std::collections::HashMap::new();
        let mut b = std::collections::HashMap::new();
        let mut n = 0u64;
        for sub in 1..=15u64 {
            for _ in 0..2 {
                for p in &pages {
                    n += u64::from(a.get(p) == Some(&sub));
                    b.get(p);
                }
            }
            for p in &pages {
                a.insert(*p, sub);
                b.insert(*p, sub);
            }
        }
        let maps = t.elapsed();
        let t = std::time::Instant::now();
        let mut f = PageTable { chunks: Vec::new() };
        for sub in 1..=15u64 {
            for _ in 0..2 {
                for p in &pages {
                    n += u64::from(f.get(*p).0 == sub);
                    f.get(*p);
                }
            }
            for p in &pages {
                let e = f.at(*p);
                *e = (sub, sub);
            }
        }
        eprintln!("page tables: hash maps {maps:?}, flat {:?} ({n})", t.elapsed());
    }
}
