//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
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
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
}

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, std::collections::HashSet<usize>);

/// Where a copy-back piece's snapshot bytes are: inside one segment (its
/// address), or reassembled from several.
enum Old {
    At(usize),
    Gathered(usize),
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

/// Retain only bounded idle snapshot memory. Active submissions are not
/// capped and never lose a baseline because the pool is full.
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let idle: usize = pool.iter().map(Vec::capacity).sum();
    if v.capacity() <= POOL_BYTES.saturating_sub(idle) {
        pool.push(v);
    }
}

const POOL_BYTES: usize = 256 << 20;
const SNAPSHOT_CHUNK: usize = 4 << 20;

/// Threads snapshots and copy-back use (`VICTORY_COPY_THREADS`, 1 = all
/// on the submitting thread as before; default up to 6). Hundreds of MB
/// a world frame were copied and compared on the submitting thread alone
/// (Belobug a0a4591 did the same for its tiles). `VICTORY_SERIAL_COPIES=1`
/// is the opt-out for A/B runs of one build: one thread, as before.
fn copy_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        let serial = std::env::var("VICTORY_SERIAL_COPIES").ok();
        let n = threads_from(serial.as_deref(), std::env::var("VICTORY_COPY_THREADS").ok().as_deref(), cpus);
        tracing::info!(threads = n, "snapshots and copy-back threads (VICTORY_SERIAL_COPIES=1 for one)");
        n
    })
}

/// [`copy_threads`] from `VICTORY_SERIAL_COPIES`, `VICTORY_COPY_THREADS`
/// and the processors there are.
fn threads_from(serial: Option<&str>, threads: Option<&str>, cpus: usize) -> usize {
    if serial.is_some_and(|v| v.trim() == "1") {
        return 1;
    }
    threads
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map_or(cpus.clamp(1, 6), |n| n.clamp(1, 16))
}

/// Work smaller than this stays on the calling thread (starting threads
/// costs tens of microseconds).
const PARALLEL_LEAST: usize = 2 << 20;
/// Pieces memory copies are split into between threads.
const COPY_PIECE: usize = 1 << 20;

/// Calls `f` on pieces of `list` from up to `threads` threads (the caller's
/// one of them), small pieces taken in turn: they are not equally long.
/// Every item is handed to exactly one call.
fn in_parallel<T: Sync>(list: &[T], threads: usize, f: impl Fn(&[T]) + Sync) {
    if threads <= 1 || list.len() <= 1 {
        return f(list);
    }
    let piece = (list.len() / (threads * 8)).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || {
        loop {
            let from = next.fetch_add(piece, std::sync::atomic::Ordering::Relaxed);
            if from >= list.len() {
                break;
            }
            f(&list[from..(from + piece).min(list.len())]);
        }
    };
    let threads = threads.min(list.len().div_ceil(piece));
    #[cfg(test)]
    SPREAD.fetch_add(usize::from(threads > 1), std::sync::atomic::Ordering::Relaxed);
    std::thread::scope(|s| {
        for _ in 1..threads {
            s.spawn(work);
        }
        work();
    });
}

/// Calls of [`in_parallel`] that started threads (tests).
#[cfg(test)]
static SPREAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Declared ranges whose snapshot was skipped because a range already
/// declared in the submission covers them (diagnostics).
#[cfg(feature = "diagnostics")]
static COVERED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

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

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked: crate::watch::track(start as usize, len as usize, true),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
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
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.writing_on(address, len, copy_threads());
    }

    /// [`Shadow::writing`] with the snapshot copied on up to `threads`
    /// threads. A range inside one already declared in this submission has
    /// its snapshot already (Belobug a0a4591): nothing to look at.
    fn writing_on(&self, address: usize, len: usize, threads: usize) {
        let end = address + len;
        {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            // Every pending range had its holes snapshotted when declared, and
            // the snapshot lives until the last pending range goes.
            let covered = pending.iter().any(|&(a, b)| a <= address && end <= b);
            pending.insert((address, end));
            if covered {
                #[cfg(feature = "diagnostics")]
                {
                    use std::sync::atomic::Ordering::Relaxed;
                    COVERED[0].fetch_add(1, Relaxed);
                    COVERED[1].fetch_add(len as u64, Relaxed);
                }
                return;
            }
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        // The holes of `[from, to)` not snapshotted yet, in order: preserve
        // the first baseline of each byte; extending a range only snapshots
        // its holes. Chunk edges aligned to title pages.
        let mut holes: Vec<(usize, usize)> = Vec::new();
        let mut i = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
        let mut at = from;
        while at < to {
            if let Some((start, bytes)) = segments.get(i)
                && *start <= at
            {
                at = *start + bytes.len();
                i += 1;
                continue;
            }
            let absolute = self.start as usize + at;
            let next = ((absolute / SNAPSHOT_CHUNK + 1) * SNAPSHOT_CHUNK) - self.start as usize;
            let end = to.min(next).min(segments.get(i).map_or(to, |s| s.0));
            holes.push((at, end));
            at = end;
        }
        if holes.is_empty() {
            return;
        }
        let captured: usize = holes.iter().map(|&(a, b)| b - a).sum();
        let mut bufs: Vec<Vec<u8>> = holes.iter().map(|&(a, b)| pooled(b - a)).collect();
        // (copy offset, snapshot buffer, bytes) in pieces of at most
        // COPY_PIECE; each piece is one thread's alone.
        let mut copies: Vec<(usize, usize, usize)> = Vec::new();
        for (&(a, b), buf) in holes.iter().zip(&mut bufs) {
            let dst = buf.as_mut_ptr() as usize;
            let mut at = a;
            while at < b {
                let n = (b - at).min(COPY_PIECE);
                copies.push((at, dst + (at - a), n));
                at += n;
            }
        }
        let data = self.data as usize;
        let threads = if captured >= PARALLEL_LEAST { threads } else { 1 };
        in_parallel(&copies, threads, |chunk| {
            for &(at, dst, n) in chunk {
                // SAFETY: inside the mapping (`at + n <= to`); a pooled buffer
                // with room for its hole, each piece written by one thread.
                unsafe { std::ptr::copy_nonoverlapping((data + at) as *const u8, dst as *mut u8, n) };
            }
        });
        let mut new = holes
            .iter()
            .zip(bufs)
            .map(|(&(a, b), mut buf)| {
                // SAFETY: every byte of `[0, b - a)` was written above,
                // within the capacity `pooled` gave.
                unsafe { buf.set_len(b - a) };
                (a, buf)
            })
            .peekable();
        // Holes lie between existing segments: merge both sorted lists.
        let old = std::mem::take(segments);
        segments.reserve(old.len() + holes.len());
        for segment in old {
            while new.peek().is_some_and(|(a, _)| *a < segment.0) {
                segments.push(new.next().expect("peeked"));
            }
            segments.push(segment);
        }
        segments.extend(new);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::SnapshotBytes, captured as u64);
            crate::copystat::snapshot(self.start, self.len, captured as u64);
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
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        self.copy_back_on(address, len, back, copy_threads())
    }

    /// [`Shadow::copy_back_or_not`] with the pieces compared and written
    /// back on up to `threads` threads. Which pieces are compared, and with
    /// which snapshot bytes, is decided in order on this thread first; the
    /// pieces are disjoint, so the threads touch disjoint memory.
    fn copy_back_on(&self, address: usize, len: usize, back: bool, threads: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if back && from < to {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            // (copy offset, bytes, the snapshot's bytes: a pointer into a
            // segment, or an index into `gathered`).
            let mut work: Vec<(usize, usize, Old)> = Vec::new();
            let mut gathered: Vec<Vec<u8>> = Vec::new();
            let mut i = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                let end = at + n;
                let Some((taken, before)) = segments.get(i).filter(|(s, _)| *s <= at) else { break };
                let old = if end <= taken + before.len() {
                    Old::At(before.as_ptr() as usize + (at - taken))
                } else {
                    // A range extension may have split a page into several
                    // captures. Reassemble just this page, not the snapshot.
                    let mut piece = Vec::with_capacity(n);
                    let mut cursor = at;
                    for (s, bytes) in &segments[i..] {
                        if *s > cursor { break; }
                        let limit = end.min(*s + bytes.len());
                        piece.extend_from_slice(&bytes[cursor - s..limit - s]);
                        cursor = limit;
                        if cursor == end { break; }
                    }
                    if piece.len() != n { break; }
                    gathered.push(piece);
                    Old::Gathered(gathered.len() - 1)
                };
                while segments.get(i).is_some_and(|(s, b)| s + b.len() <= end) { i += 1; }
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                work.push((at, n, old));
                at += n;
            }
            // Per piece: 0 equal to title memory or unread, 1 synced, 2
            // synced and written back.
            let outcome: Vec<std::sync::atomic::AtomicU8> =
                work.iter().map(|_| std::sync::atomic::AtomicU8::new(0)).collect();
            let total: usize = work.iter().map(|w| w.1).sum();
            let threads = if total >= PARALLEL_LEAST { threads } else { 1 };
            let data = self.data as usize;
            let indexed: Vec<usize> = (0..work.len()).collect();
            in_parallel(&indexed, threads, |chunk| {
                let mut title = vec![0u8; PIECE];
                for &k in chunk {
                    let (at, n, ref old) = work[k];
                    // SAFETY: inside the mapping; the GPU is done with it and
                    // this piece is this thread's alone.
                    let host = unsafe { std::slice::from_raw_parts_mut((data + at) as *mut u8, n) };
                    let old: &[u8] = match *old {
                        // SAFETY: inside a segment (checked above), which
                        // the lock held keeps alive and unchanged.
                        Old::At(p) => unsafe { std::slice::from_raw_parts(p as *const u8, n) },
                        Old::Gathered(g) => &gathered[g],
                    };
                    let title = &mut title[..n];
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
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
                            let _writing = crate::watch::writing(start + at, n);
                            crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        }
                        outcome[k].store(1 + u8::from(changed), std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
            let mut synced = Vec::new();
            for (&(at, n, _), o) in work.iter().zip(&outcome) {
                match o.load(std::sync::atomic::Ordering::Relaxed) {
                    2 => {
                        written += n;
                        synced.push((start + at) / PAGE);
                    }
                    1 => synced.push((start + at) / PAGE),
                    _ => {}
                }
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
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
    /// shader writes: the pages changed since taken, or all of it where the
    /// memory is not tracked.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if let Some(t) = &self.tracked {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let runs = t.take(from, to);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
            // Cut to the range: next to it shaders may be writing the copy.
            let copied: usize = runs.into_iter().map(|(a, b)| self.copy(a, b)).sum();
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::Checked, (to - from) as u64);
                add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
            }
            return copied;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let copied = self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, copied as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        copied
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        // These are zero-initialized committed D3D12 buffers (no
        // CREATE_NOT_ZEROED). An untouched lazy CPU reservation supplies no
        // newer bytes; the shadow's initial zeros or GPU results stay valid.
        // When a page commits it stays dirty until the first readable copy.
        crate::host::copy_shadow(at, from as *const u8, to - from, crate::watch::lazy_covers)
    }
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

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
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
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
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
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
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
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
const WINDOW: u64 = 4 << 20;
const LOOSE_MAX_COUNT: usize = 64;
const LOOSE_MAX_BYTES: u64 = 256 << 20;

#[derive(Default)]
struct Loose {
    entries: BTreeMap<(u64, u64), (u64, Arc<Shadow>)>,
    clock: u64,
}

impl Loose {
    fn trim(&mut self, count: usize, bytes: u64) -> Vec<Arc<Shadow>> {
        let mut cold: Vec<_> = self.entries.iter()
            .filter(|(_, (_, v))| Arc::strong_count(v) == 1)
            .map(|(&key, (used, _))| (*used, key)).collect();
        cold.sort_unstable();
        let mut total: u64 = self.entries.keys().map(|&(_, len)| len).sum();
        let mut released = Vec::new();
        for (_, key) in cold {
            if self.entries.len() <= count && total <= bytes { break; }
            let (_, copy) = self.entries.remove(&key).unwrap();
            released.push(copy);
            total -= key.1;
        }
        if !released.is_empty() { changed(); }
        released
    }
}

static LOOSE: Mutex<Loose> = Mutex::new(Loose { entries: BTreeMap::new(), clock: 0 });

/// VA reuse makes old loose windows stale. Active lists keep their own Arc.
pub(crate) fn evict_loose(first: usize, len: usize) {
    if len == 0 { return; }
    let end = first.saturating_add(len) as u64;
    let mut cache = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let keys: Vec<_> = cache.entries.keys().copied()
        .filter(|&(a, n)| a < end && (first as u64) < a.saturating_add(n)).collect();
    let released: Vec<_> = keys.iter().filter_map(|k| cache.entries.remove(k)).collect();
    if !released.is_empty() { changed(); }
    drop(cache);
    drop(released);
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.checked_next_multiple_of(WINDOW)?.checked_sub(start)?;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    map.clock = map.clock.wrapping_add(1);
    let now = map.clock;
    if let Some((used, shadow)) = map.entries.get_mut(&(start, len)) {
        *used = now;
        return Some(shadow.clone());
    }
    let total: u64 = map.entries.keys().map(|&(_, n)| n).sum();
    let released = if map.entries.len() >= LOOSE_MAX_COUNT || total.saturating_add(len) > LOOSE_MAX_BYTES {
        map.trim(48, (192u64 << 20).min(LOOSE_MAX_BYTES.saturating_sub(len)))
    } else { Vec::new() };
    // Drop detached copies after releasing the cache lock: their trackers
    // can acquire the change journal mutex.
    drop(map);
    drop(released);
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, existing)) = map.entries.get(&(start, len)) {
        return Some(existing.clone());
    }
    map.entries.insert((start, len), (now, shadow.clone()));
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let [covered, covered_bytes] = [0, 1].map(|i| COVERED[i].swap(0, Relaxed));
    if covered > 0 {
        tracing::info!(
            "shader-written ranges inside one already declared: {:.0}/frame ({:.1} MB) not looked at again; copy threads {}",
            covered as f64 / frames,
            covered_bytes as f64 / frames / 1e6,
            copy_threads()
        );
    }
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    map.insert((address, len), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).entries.values().map(|(_, s)| s.clone()));
    all
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
            .entries.values()
            .map(|(_, s)| (s.gpu, s.start, s.len)),
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
    fn lazy_reserved_refresh_keeps_gpu_bytes_until_cpu_commit() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const LEN: usize = 4 * PAGE;
        static BASE: AtomicUsize = AtomicUsize::new(0);
        let host = host().expect("D3D12 host");
        // SAFETY: a new private reservation; no bytes are accessed before commit.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(title, 0);
        BASE.store(title, Relaxed);
        let covers = |at: usize, len: usize| {
            let base = BASE.load(Relaxed);
            base != 0 && at >= base && at.saturating_add(len) <= base + LEN
        };
        crate::watch::on_lazy_reserved(covers, covers);
        let (buffer, mapped) = host.shared_buffer(LEN as u64).unwrap();
        let copy = Shadow::new(buffer, mapped, 0, title as u64, LEN as u64);
        assert_eq!(copy.update(title, LEN), 0, "fresh shadow keeps its initial zeros");
        assert_eq!(copy.read_host(title as u64, 1), [0]);
        let (upload, data) = host.upload_buffer(4).unwrap();
        let value = 0x1234_5678u32;
        let list = host.new_list().unwrap();
        // SAFETY: both buffers cover four bytes; COMMON promotes to COPY_DEST.
        unsafe {
            data.cast::<u32>().write(value);
            list.list.CopyBufferRegion(&copy.buffer, PAGE as u64, &upload, 0, 4);
        }
        list.close();
        host.execute(&[&list]);
        let before = copy.read_host((title + PAGE) as u64, 1);
        assert_eq!(before, [value]);
        let bytes = (0..32).map(|_| copy.update(title, LEN)).sum::<usize>();
        assert_eq!(bytes, 0);
        assert_eq!(copy.read_host((title + PAGE) as u64, 1), before);
        // An actual commit/write supplies authoritative bytes and must win.
        let page = (title + PAGE) as *mut std::ffi::c_void;
        // SAFETY: within the original reservation, committed before writing.
        unsafe {
            assert_eq!(VirtualAlloc(page, PAGE, MEM_COMMIT, PAGE_READWRITE), page);
            page.cast::<u32>().write(42);
        }
        assert_eq!(copy.update(title, LEN), PAGE);
        assert_eq!(copy.read_host((title + PAGE) as u64, 1), [42]);
        // The old policy demonstrably overwrites the same GPU value.
        copy.write_host((title + 2 * PAGE) as u64, &value.to_le_bytes());
        // SAFETY: this is the live writable shadow mapping, not title memory.
        let legacy_dst = unsafe { mapped.add(2 * PAGE) };
        crate::host::copy_mapped(legacy_dst, (title + 2 * PAGE) as *const u8, PAGE, false);
        assert_eq!(copy.read_host((title + 2 * PAGE) as u64, 1), [0]);
        eprintln!("reserved refresh fixture: {} -> {bytes} bytes over 32 refreshes; real GPU copy preserved, CPU commit wins", 32 * LEN);
        drop(copy);
        BASE.store(0, Relaxed);
        // SAFETY: original reservation, no tracker left on it.
        assert_ne!(unsafe { VirtualFree(title as _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn overlapping_snapshots_capture_only_new_bytes_and_cap_idle_commit() {
        let host = host().expect("D3D12 host");
        let len = SNAPSHOT_CHUNK;
        let p = crate::watch::tests::pool(len / PAGE);
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        let copy = Shadow::new(buffer, data, 0, p as u64, len as u64);
        copy.update(p, len);
        // Grow the same range one page at a time. The old implementation
        // allocated and copied the whole growing prefix after each step.
        let steps = len / PAGE;
        for n in 1..=steps {
            copy.writing(p, n * PAGE);
        }
        let saved = copy.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        assert_eq!(saved, len);
        copy.write_host((p + PAGE - 1) as u64, &[0x34, 0x56]);
        assert_eq!(copy.copy_back(p, len), 2 * PAGE);
        for n in 1..steps { copy.keep_writes(p, n * PAGE); }
        assert!(copy.before.lock().unwrap().0.is_empty());
        let idle: usize = POOL.lock().unwrap().iter().map(Vec::capacity).sum();
        assert!(idle <= POOL_BYTES);
        eprintln!("snapshot fixture: {} -> {saved} baseline bytes; idle cap {} -> {POOL_BYTES} bytes", PAGE * steps * (steps + 1) / 2, 2usize << 30);
        drop(copy);
        // SAFETY: owned pool, no tracker remains.
        assert_ne!(unsafe { windows_sys::Win32::System::Memory::VirtualFree(p as _, 0, windows_sys::Win32::System::Memory::MEM_RELEASE) }, 0);
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64);
        copy.writing(p + 3 * PAGE, 100);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE);
        copy.writing(p + PAGE, PAGE);
        copy.writing(p + 80, 8);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "only newly covered bytes were added");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Title byte `i` of the fixtures: never zero, so a zero anywhere is a
    /// zero-fill.
    fn pattern(i: usize) -> u8 {
        (i % 251) as u8 + 1
    }

    /// One run of the parallel-snapshot fixture on `threads` threads: a
    /// 10 MiB write-watched buffer (three snapshot chunks), declared ranges
    /// that overlap and nest, shader writes across chunk and piece edges,
    /// a title write meanwhile. Returns (title bytes, copy bytes, snapshot
    /// bytes after each declaration, bytes written back).
    fn parallel_fixture(threads: usize) -> (Vec<u8>, Vec<u8>, Vec<usize>, usize) {
        let host = host().expect("D3D12 host");
        let pages = 160;
        let len = pages * PAGE;
        let p = crate::watch::tests::pool(pages);
        // SAFETY: our pool, not guarded yet.
        unsafe { (0..len).for_each(|i| ((p + i) as *mut u8).write(pattern(i))) };
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        let copy = Shadow::new(buffer, data, 0, p as u64, len as u64);
        assert_eq!(copy.update(p, len), len);
        let snap = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        let segments_ok = |s: &Shadow| {
            let b = s.before.lock().unwrap();
            // Disjoint, in order, and each the title's bytes before the run.
            b.0.windows(2).all(|w| w[0].0 + w[0].1.len() <= w[1].0)
                && b.0.iter().all(|(at, bytes)| bytes.iter().enumerate().all(|(k, &v)| v == pattern(at + k)))
        };
        let mib = 1 << 20;
        let a = (p + 100, 9 * mib - 100);
        let nested = (p + 3 * PAGE + 7, 2 * PAGE);
        let tail = (p + 9 * mib - 100, len - 9 * mib + 100);
        let early = (p + 10, 200);
        let mut sizes = Vec::new();
        for (at, n) in [a, nested, a, tail, early] {
            copy.writing_on(at, n, threads);
            sizes.push(snap(&copy));
            assert!(segments_ok(&copy), "snapshot holds the title's bytes before the submission");
        }
        // The GPU writes: across the 4 MiB chunk edge, a 64 KiB piece edge,
        // inside the nested range, at the very ends.
        let writes: [(usize, &[u8]); 6] = [
            (4 * mib - 3, &[0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6]),
            (17 * PAGE - 1, &[0xb1, 0xb2]),
            (3 * PAGE + 9, &[0xc1; 300]),
            (100, &[0xd1]),
            (len - 1, &[0xe1]),
            (8 * mib + 12345, &[0xf1; 70000]),
        ];
        for (at, bytes) in writes {
            copy.write_host((p + at) as u64, bytes);
        }
        // The title writes a byte of a written page meanwhile (guard fault),
        // one the GPU left alone: the title's byte stays.
        // SAFETY: our pool; the guard handler takes the write.
        unsafe { ((p + 17 * PAGE + 40) as *mut u8).write_volatile(0x77) };
        let mut written = 0;
        for (at, n) in [early, tail, a, nested, a] {
            written += copy.copy_back_on(at, n, true, threads);
        }
        assert!(copy.before.lock().unwrap().0.is_empty(), "the snapshot goes with the last range");
        // SAFETY: our pool and the copy's mapping, both `len` bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) }.to_vec();
        // SAFETY: as above.
        let host_bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        let mut expected: Vec<u8> = (0..len).map(pattern).collect();
        for (at, bytes) in writes {
            expected[at..at + bytes.len()].copy_from_slice(bytes);
        }
        expected[17 * PAGE + 40] = 0x77;
        assert!(title == expected, "title memory: the shaders' bytes went back, the title's stayed, nothing else changed");
        assert!(!title.contains(&0) && !host_bytes.contains(&0), "no zero-fill of title memory or of the copy");
        // Byte for byte, the copy is title memory again (the shaders' bytes,
        // the title's write, the rest as it was): nothing zeroed or stale.
        let differ = host_bytes.iter().zip(&expected).filter(|(h, e)| h != e).count();
        assert_eq!(differ, 0, "copy bytes differing from title memory");
        drop(copy);
        crate::watch::tests::free(p);
        (title, host_bytes, sizes, written)
    }

    /// (b): snapshots and copy-back on several threads keep the bytes the
    /// shaders wrote, keep the title's own, zero nothing, snapshot each byte
    /// once (a range inside one already declared adds nothing), and give
    /// the same result as one thread, the behaviour before (a).
    #[test]
    fn parallel_snapshots_keep_shader_bytes_and_match_one_thread() {
        if host().is_none() {
            return;
        }
        let spread = SPREAD.load(std::sync::atomic::Ordering::Relaxed);
        let many = parallel_fixture(6);
        assert!(SPREAD.load(std::sync::atomic::Ordering::Relaxed) > spread, "threads were started");
        let one = parallel_fixture(1);
        let len = 160 * PAGE;
        let a = 9 * (1 << 20) - 100;
        // a, then nested inside a (nothing), a again (nothing), the tail's
        // part past a, then `early`'s 90 bytes before a.
        assert_eq!(many.2, vec![a, a, a, len - 100, len - 10]);
        assert_eq!(many.0, one.0, "title memory identical");
        assert_eq!(many.1, one.1, "copies identical");
        assert_eq!(many.2, one.2, "snapshot bytes identical");
        assert_eq!(many.3, one.3, "bytes written back identical");
        eprintln!(
            "parallel snapshot fixture: {} snapshot bytes for {} declared, {} written back",
            many.2[4],
            2 * a + 2 * PAGE + (len - a) + 200,
            many.3
        );
    }

    /// The A/B opt-out: `VICTORY_SERIAL_COPIES=1` wins, one thread as
    /// before; otherwise up to 6 by default.
    #[test]
    fn serial_copies_opt_out() {
        assert_eq!(threads_from(Some("1"), Some("8"), 16), 1);
        assert_eq!(threads_from(Some(" 1 "), None, 16), 1);
        assert_eq!(threads_from(Some("0"), None, 16), 6);
        assert_eq!(threads_from(None, None, 4), 4);
        assert_eq!(threads_from(None, Some("1"), 16), 1);
        assert_eq!(threads_from(None, Some("0"), 16), 1);
        assert_eq!(threads_from(None, Some("x"), 2), 2);
    }

    /// The memory peak of several threads against one: snapshot buffers
    /// are the same either way; the threads add their stacks and a 64 KiB
    /// buffer each. Run alone (`--ignored --test-threads=1`): the process
    /// peak is shared with other tests.
    #[test]
    #[ignore]
    fn parallel_copies_peak_commit() {
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        if host().is_none() {
            return;
        }
        let peak = || {
            // SAFETY: plain data, all zeros valid.
            let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
            c.cb = size_of_val(&c) as u32;
            // SAFETY: our own process, a counters structure of its size.
            assert_ne!(unsafe { GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast(), c.cb) }, 0);
            (c.PeakPagefileUsage, c.PeakWorkingSetSize)
        };
        // Twice on one thread first: the pool and the D3D12 heaps reach
        // their size, so the peak after that is what threads add.
        parallel_fixture(1);
        parallel_fixture(1);
        let (commit, ws) = peak();
        for _ in 0..4 {
            parallel_fixture(6);
        }
        let (commit2, ws2) = peak();
        let extra = commit2.saturating_sub(commit);
        eprintln!(
            "peak commit one thread {:.1} MB, six threads {:.1} MB (+{:.2} MB); peak working set {:.1} -> {:.1} MB",
            commit as f64 / 1e6,
            commit2 as f64 / 1e6,
            extra as f64 / 1e6,
            ws as f64 / 1e6,
            ws2 as f64 / 1e6
        );
        assert!(extra < 32 << 20, "threads add at most stacks and piece buffers");
    }
}
