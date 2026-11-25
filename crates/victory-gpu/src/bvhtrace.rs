//! `VICTORY_BVH_TRACE=1` (diagnostics): where the ray-tracing structure
//! builder's results land in title memory.
//!
//! A builder that writes outside its structures corrupts the title: run 98
//! of the fork, and runs with `VICTORY_BVH_BUILDER=1` on 2026-11-25, where
//! the title read `0xff8000007f800000` (an empty box's bounds, +inf then
//! -inf, as the builder initialises them) where it keeps a pointer. This
//! logs three things, to compare afterwards:
//!
//! - each copy that becomes reachable by raw address (its title span);
//! - the root views and root constants each builder pipeline is given;
//! - every copy-back into title memory that carries infinities, with the
//!   address and the copy around it.
//!
//! Logging only: nothing here changes what runs.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

/// Whether a builder stage has been dispatched yet: infinities written back
/// before it are the rest of the frame's (depth, reductions), not its.
static BUILDER_SEEN: AtomicBool = AtomicBool::new(false);

/// Per 4 MiB of title memory: infinities written back before and after the
/// builder's first dispatch, and ranges carrying them.
static TOTALS: Mutex<Option<HashMap<u64, [u64; 4]>>> = Mutex::new(None);

/// Logs the 4 MiB blocks whose copy-backs carry infinities, those that got
/// them only once the builder ran first (diagnostics frame line).
pub(crate) fn summary() {
    if !on() {
        return;
    }
    let totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(totals) = totals.as_ref() else { return };
    let mut rows: Vec<(&u64, &[u64; 4])> = totals.iter().collect();
    // Only after the builder first, then by count.
    rows.sort_by_key(|(_, t)| (t[0] != 0, std::cmp::Reverse(t[1])));
    // Low bits of a key name the path: 0 shader copy-backs, 1 title copies.
    let path = |block: u64| if block & 1 == 0 { "back" } else { "copy" };
    let text: Vec<String> = rows
        .iter()
        .map(|(block, t)| {
            format!(
                "{:#x} {}: before {} ({} ranges), after {} ({} ranges)",
                *block & !1,
                path(**block),
                t[0],
                t[2],
                t[1],
                t[3]
            )
        })
        .collect();
    tracing::info!(
        builder_seen = BUILDER_SEEN.load(Ordering::Relaxed),
        blocks = totals.len(),
        "bvh trace: infinities by 4 MiB block: {}",
        text.join("; ")
    );
}

/// Counts infinity words (+inf `0x7f800000`, -inf `0xff800000`) in `bytes`
/// written at title address `dst`, aligned words only: count and first.
fn infinities(dst: usize, bytes: &[u8]) -> (usize, Option<usize>) {
    let skip = ((4 - dst % 4) % 4).min(bytes.len());
    let mut count = 0usize;
    let mut first = None;
    for (i, word) in bytes[skip..].chunks_exact(4).enumerate() {
        let w = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        if w == 0x7f80_0000 || w == 0xff80_0000 {
            count += 1;
            first.get_or_insert(dst + skip + i * 4);
        }
    }
    (count, first)
}

fn count(key: u64, n: usize) {
    let after = usize::from(BUILDER_SEEN.load(Ordering::Relaxed));
    let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let t = totals.get_or_insert_with(HashMap::new).entry(key).or_default();
    t[after] += n as u64;
    t[2 + after] += 1;
}

/// After a title copy (the title's own buffer copies, done on the CPU) of
/// `len` bytes from `src` to title memory at `dst`: counts and, once the
/// builder runs, logs the infinities it put there.
pub(crate) fn copied(dst: usize, src: usize, len: usize) {
    static LINES: AtomicU32 = AtomicU32::new(0);
    if !on() || len < 8 {
        return;
    }
    watched(if src == 0 { "read-back" } else { "title copy" }, dst, len);
    // Read back through the kernel: part of the destination may be unmapped.
    let mut bytes = vec![0u8; len.min(64 << 20)];
    if !crate::host::read_into(dst as *const u8, &mut bytes) {
        return;
    }
    let (n, first) = infinities(dst, &bytes);
    if n < 2 {
        return;
    }
    count(dst as u64 & !((4 << 20) - 1) | 1, n);
    if !BUILDER_SEEN.load(Ordering::Relaxed) || !take(&LINES) {
        return;
    }
    tracing::warn!(
        frame = crate::frame::current(),
        dst = format_args!("{dst:#x}"),
        src = format_args!("{src:#x}"),
        len,
        infinities = n,
        first = format_args!("{:#x}", first.unwrap_or(dst)),
        "bvh trace: infinities copied into title memory by a title copy"
    );
}

/// The programs of shader `hash` by their number in fault records
/// (`victory_shader::emit::program_id`), for those that reach title memory
/// by address.
pub(crate) fn program_names(hash: &str, container: &[u8]) {
    if !on() {
        return;
    }
    let Ok(c) = victory_shader::Container::parse(container) else { return };
    let Some(part) = c.part(victory_shader::FourCC::DXIL) else { return };
    let Ok(programs) = victory_shader::native_programs(part.data) else { return };
    for p in &programs {
        tracing::info!(
            hash,
            program = format_args!("{:08x}", victory_shader::emit::program_id(&p.code)),
            "bvh trace: program number"
        );
    }
}

/// A shader touched title memory with no copy: `page` (64 KiB), the low
/// dword of the address, the instruction's byte offset and the program's
/// number. Counted by program and instruction; the first few logged.
pub(crate) fn fault(page: u32, lo: u32, pc: u32, program: u32) {
    static LINES: AtomicU32 = AtomicU32::new(0);
    if !on() {
        return;
    }
    let address = u64::from(page) << 16 | u64::from(lo & 0xFFFF);
    if take(&LINES) {
        tracing::warn!(
            frame = crate::frame::current(),
            address = format_args!("{address:#x}"),
            program = format_args!("{program:08x}"),
            pc = format_args!("{pc:#x}"),
            "bvh trace: an access found no copy"
        );
    }
}

/// `VICTORY_BVH_WATCH=<start>,<len>` (hex or decimal): every write of ours
/// into that title range is logged, whatever it carries (`path` says which).
pub(crate) fn watched(path: &str, dst: usize, len: usize) {
    static RANGE: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    let range = *RANGE.get_or_init(|| {
        let v = std::env::var("VICTORY_BVH_WATCH").ok()?;
        let mut it = v.split(',').map(|x| {
            let x = x.trim();
            x.strip_prefix("0x").map_or_else(|| x.parse().ok(), |h| usize::from_str_radix(h, 16).ok())
        });
        let (start, n) = (it.next()??, it.next()??);
        tracing::info!(start = format_args!("{start:#x}"), len = n, "bvh trace: watching title memory");
        Some((start, start + n))
    });
    let Some((start, end)) = range else { return };
    if dst < end && start < dst + len {
        static LINES: AtomicU32 = AtomicU32::new(0);
        if take(&LINES) {
            let around = crate::shadow::find(dst as u64, 1).map(|s| {
                let (a, n) = s.span();
                format!("{a:#x}..{:#x}", a + n)
            });
            tracing::warn!(
                frame = crate::frame::current(),
                path,
                dst = format_args!("{dst:#x}"),
                len,
                copy = ?around,
                builder_seen = BUILDER_SEEN.load(Ordering::Relaxed),
                "bvh trace: a write of ours into the watched range"
            );
        }
    }
}

/// Whether `VICTORY_BVH_TRACE=1`.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_TRACE").is_some_and(|v| v == "1"))
}

/// Lines of each kind at most: a world frame copies back thousands of
/// ranges.
const MAX_LINES: u32 = 600;

fn take(counter: &AtomicU32) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < MAX_LINES
}

/// A copy got number `n` in the raw-address directory.
pub(crate) fn numbered(n: usize, start: u64, len: u64) {
    if on() {
        tracing::info!(
            n,
            start = format_args!("{start:#x}"),
            end = format_args!("{:#x}", start + len),
            "bvh trace: copy reachable by raw address"
        );
    }
}

/// A builder pipeline is dispatched with these root views and constants
/// (logged once per pipeline and set of arguments).
pub(crate) fn builder_inputs(pipeline: &str, views: &[u64], constants: &[u32]) {
    static SEEN: Mutex<Option<HashSet<(String, Vec<u64>, Vec<u32>)>>> = Mutex::new(None);
    static LINES: AtomicU32 = AtomicU32::new(0);
    if !on() {
        return;
    }
    if !BUILDER_SEEN.swap(true, Ordering::Relaxed) {
        // The baseline: what the frames before the builder wrote back.
        summary();
    }
    let key = (pipeline.to_owned(), views.to_vec(), constants.to_vec());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.get_or_insert_with(HashSet::new).insert(key) || !take(&LINES) {
        return;
    }
    // Constants that read as title addresses, as `memory::ready` takes them.
    let mut named = Vec::new();
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            named.push(format!("[{i}]={:#x}", u64::from(c) | u64::from(high) << 32));
        }
        if c >= 0x0100_0000 {
            named.push(format!("[{i}]<<8={:#x}", u64::from(c) << 8));
        }
    }
    let views: Vec<String> = views.iter().map(|v| format!("{v:#x}")).collect();
    tracing::info!(
        frame = crate::frame::current(),
        pipeline,
        views = format_args!("{views:?}"),
        constants = format_args!("{:x?}", &constants[..constants.len().min(24)]),
        addresses = format_args!("{named:?}"),
        "bvh trace: builder dispatch"
    );
}

/// Before `len` bytes from `src` are written to title memory at `dst`: logs
/// the range if it carries infinities (+inf `0x7f800000`, -inf
/// `0xff800000`, aligned words).
pub(crate) fn landing(dst: usize, src: *const u8, len: usize) {
    static LINES: AtomicU32 = AtomicU32::new(0);
    if !on() || len < 8 {
        return;
    }
    watched("copy-back (any)", dst, len);
    // SAFETY: the caller's source holds `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(src, len) };
    let skip = (4 - dst % 4) % 4;
    let mut infinities = 0usize;
    let mut first = None;
    for (i, word) in bytes[skip.min(len)..].chunks_exact(4).enumerate() {
        let w = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        if w == 0x7f80_0000 || w == 0xff80_0000 {
            infinities += 1;
            first.get_or_insert(dst + skip + i * 4);
        }
    }
    if infinities < 2 {
        return;
    }
    {
        let after = usize::from(BUILDER_SEEN.load(Ordering::Relaxed));
        let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
        let t = totals.get_or_insert_with(HashMap::new).entry(dst as u64 & !((4 << 20) - 1)).or_default();
        t[after] += infinities as u64;
        t[2 + after] += 1;
    }
    // A summary every 200 ranges with infinities, past the line limit too.
    static RANGES: AtomicU32 = AtomicU32::new(0);
    if RANGES.fetch_add(1, Ordering::Relaxed) % 200 == 199 {
        summary();
    }
    // Ranges are listed once the builder runs; before it only counted.
    if !BUILDER_SEEN.load(Ordering::Relaxed) || !take(&LINES) {
        return;
    }
    let around = crate::shadow::find(dst as u64, 1)
        .map(|s| {
            let (start, len) = s.span();
            format!("{start:#x}..{:#x}", start + len)
        })
        .unwrap_or_else(|| "none".to_owned());
    tracing::warn!(
        frame = crate::frame::current(),
        dst = format_args!("{dst:#x}"),
        len,
        infinities,
        first = format_args!("{:#x}", first.unwrap_or(dst)),
        copy = around,
        "bvh trace: infinities written back into title memory"
    );
}
