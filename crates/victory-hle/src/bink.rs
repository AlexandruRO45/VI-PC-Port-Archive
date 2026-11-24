//! What the title's Bink decoder is handed, and what it puts there
//! (`VICTORY_BINK_TRACE=1`, diagnostics only).
//!
//! The title decodes its movies with the Bink 2 decoder it ships
//! (`Bink2Scarlett.dll`), which the loader maps as a title DLL, so its
//! imports are the decoder's own code and not ours. That leaves the frame
//! buffers as the one place to look from: the title asks the decoder how big
//! they must be ([`BinkGetFrameBuffersInfo`]), allocates them, hands them
//! over ([`BinkRegisterFrameBuffers`]), and afterwards the decoder writes a
//! decoded frame into that memory, which the movie's pixel shader samples as
//! three 8-bit luminance planes.
//!
//! So this wraps those imports on the import-resolution path — five symbols,
//! each calling the decoder's own function straight through — and records,
//! for the texture path to match against ([`victory_gpu::movie`]):
//!
//! - the structure the title is handed and the one it registers, as raw
//!   words, which is what a wrong size or a refused registration looks like;
//! - every pointer-sized value in the registered structure whose page is
//!   committed, with its region size: the buffers the decoder will write;
//! - at every wait, whether each of those buffers still holds what it held;
//!   a decoder writing frames changes them about thirty times a second.
//!
//! Nothing is wrapped unless the switch is on and [`attach`] has been told
//! where the decoder's own functions are, so a run without the switch binds
//! exactly what it did before.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pe::ImportSymbol;

/// The library a title's Bink 2 decoder is in.
const DLL: &str = "bink2scarlett.dll";

/// The symbols wrapped, in the order of [`REAL`].
const SYMBOLS: [&str; 5] = [
    "BinkGetFrameBuffersInfo",
    "BinkRegisterFrameBuffers",
    "BinkDoFrameAsyncMulti",
    "BinkDoFrameAsyncWait",
    "BinkNextFrame",
];

const GET_FRAME_BUFFERS: usize = 0;
const REGISTER_FRAME_BUFFERS: usize = 1;
const DO_FRAME_MULTI: usize = 2;
const DO_FRAME_WAIT: usize = 3;
const NEXT_FRAME: usize = 4;

/// The decoder's own function for each wrapped symbol, as `attach` found it.
static REAL: [AtomicUsize; SYMBOLS.len()] = [const { AtomicUsize::new(0) }; SYMBOLS.len()];

/// `VICTORY_BINK_TRACE=1`.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_BINK_TRACE").is_some_and(|v| v == "1" || v == "true")
    })
}

/// Records the decoder's own exports, from the title DLLs the loader mapped.
/// Call it before the imports are bound: nothing is wrapped until it has.
pub fn attach(dlls: &[(String, victory_loader::LoadedImage)]) {
    if !on() {
        return;
    }
    let Some((name, image)) = dlls.iter().find(|(name, _)| name.eq_ignore_ascii_case(DLL)) else {
        tracing::warn!("no {DLL} among the title's DLLs: the decoder's buffers are not traced");
        return;
    };
    for (slot, symbol) in SYMBOLS.iter().enumerate() {
        match image.export(symbol) {
            Some(at) => {
                REAL[slot].store(at, Ordering::Relaxed);
                tracing::info!(dll = %name, symbol, at = format_args!("{at:#x}"), "bink: wrapped");
            }
            None => tracing::warn!(dll = %name, symbol, "bink: not exported, not wrapped"),
        }
    }
}

/// Our wrapper for one of the decoder's imports, when the switch is on and
/// the decoder's own function is known; the import-resolution path asks this
/// before the host and the title's own DLLs.
pub fn instrument(dll: &str, symbol: &ImportSymbol) -> Option<usize> {
    if !on() || !dll.eq_ignore_ascii_case(DLL) {
        return None;
    }
    let ImportSymbol::Name { name, .. } = symbol else {
        return None;
    };
    let slot = SYMBOLS.iter().position(|s| s == name)?;
    if REAL[slot].load(Ordering::Relaxed) == 0 {
        return None;
    }
    let wrapper: *const () = match slot {
        GET_FRAME_BUFFERS => get_frame_buffers_info as *const (),
        REGISTER_FRAME_BUFFERS => register_frame_buffers as *const (),
        DO_FRAME_MULTI => do_frame_async_multi as *const (),
        DO_FRAME_WAIT => do_frame_async_wait as *const (),
        NEXT_FRAME => next_frame as *const (),
        _ => return None,
    };
    Some(wrapper as usize)
}

/// The decoder's own function for `slot`, in the shape every wrapper here
/// forwards with: Windows passes integer and pointer arguments in four
/// registers whatever their width, and none of these takes a float.
type Real = unsafe extern "system" fn(u64, u64, u64, u64) -> u64;

fn real(slot: usize) -> Option<Real> {
    let at = REAL[slot].load(Ordering::Relaxed);
    (at != 0).then(|| {
        // SAFETY: `attach` took this address from the decoder's own export
        // table, and the signature above is what it is called with.
        unsafe { std::mem::transmute::<usize, Real>(at) }
    })
}

// The wrappers. Each logs and returns exactly what the decoder returned.

unsafe extern "system" fn get_frame_buffers_info(bink: u64, fbs: u64, _c: u64, _d: u64) -> u64 {
    let Some(f) = real(GET_FRAME_BUFFERS) else {
        return 0;
    };
    // SAFETY: the title's own arguments, passed on unchanged.
    let out = unsafe { f(bink, fbs, 0, 0) };
    dump("get frame buffers", fbs, 0x00);
    tracing::info!(
        bink = format_args!("{bink:#x}"),
        fbs = format_args!("{fbs:#x}"),
        "bink: frame buffers asked for"
    );
    out
}

unsafe extern "system" fn register_frame_buffers(bink: u64, fbs: u64, _c: u64, _d: u64) -> u64 {
    let Some(f) = real(REGISTER_FRAME_BUFFERS) else {
        return 0;
    };
    let before = dump("register frame buffers", fbs, 0x00);
    // SAFETY: the title's own arguments, passed on unchanged.
    let out = unsafe { f(bink, fbs, 0, 0) };
    let after = dump("register frame buffers, after", fbs, 0x00);
    let mut pointers = Vec::new();
    for (i, (was, now)) in before.iter().zip(&after).enumerate() {
        if was != now {
            tracing::info!(
                offset = format_args!("{:#x}", i * 4),
                was = format_args!("{was:#x}"),
                now = format_args!("{now:#x}"),
                "bink: the registered structure changed in the call"
            );
        }
    }
    // A buffer is a pointer and a pitch, and the pitch is the word after
    // it: the title fills in what it allocated, so every 64-bit word of the
    // structure whose page is committed is one candidate, and the decoder's
    // own frame buffers are the ones among them that hold pixels. The plane
    // is the pitch times the height the decoder asked for — 3,840 x 2,160
    // for the luma plane, 1,920 x 1,080 for the two chroma planes.
    let heights = (
        before.get(2).copied().unwrap_or(0),
        before.get(4).copied().unwrap_or(0),
    );
    for (i, &(off, word)) in unsafe { pairs(fbs, 0x120) }.iter().enumerate() {
        if committed(word as usize).is_none() {
            continue;
        }
        let pitch = unsafe { ((fbs as usize + off + 12) as *const u32).read_volatile() } as usize;
        let height = if pitch >= heights.0 as usize {
            heights.0 as usize
        } else {
            heights.1 as usize
        };
        let size = (pitch * height).max(1);
        pointers.push((off, word as usize, size));
        tracing::info!(
            offset = format_args!("{off:#x}"),
            at = format_args!("{word:#x}"),
            pitch,
            height,
            size,
            len = format_args!("{size:#x}"),
            word = i,
            "bink: a registered buffer"
        );
    }
    note_planes(&pointers);
    tracing::info!(
        bink = format_args!("{bink:#x}"),
        buffers = pointers.len(),
        "bink: frame buffers registered"
    );
    out
}

unsafe extern "system" fn do_frame_async_multi(bink: u64, a: u64, b: u64, c: u64) -> u64 {
    let Some(f) = real(DO_FRAME_MULTI) else {
        return 0;
    };
    // SAFETY: the title's own arguments, passed on unchanged.
    let out = unsafe { f(bink, a, b, c) };
    tracing::debug!(
        bink = format_args!("{bink:#x}"),
        a,
        b,
        c,
        out,
        "bink: do frame async multi"
    );
    out
}

unsafe extern "system" fn do_frame_async_wait(bink: u64, a: u64, _b: u64, _c: u64) -> u64 {
    let Some(f) = real(DO_FRAME_WAIT) else {
        return 0;
    };
    // SAFETY: the title's own arguments, passed on unchanged.
    let out = unsafe { f(bink, a, 0, 0) };
    advances(a);
    out
}

unsafe extern "system" fn next_frame(bink: u64, _a: u64, _b: u64, _c: u64) -> u64 {
    let Some(f) = real(NEXT_FRAME) else {
        return 0;
    };
    // SAFETY: the title's own argument, passed on unchanged.
    let out = unsafe { f(bink, 0, 0, 0) };
    static ADVANCED: AtomicUsize = AtomicUsize::new(0);
    let n = ADVANCED.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::info!(
        bink = format_args!("{bink:#x}"),
        advanced = n,
        out,
        "bink: next frame"
    );
    out
}

// Reading what the decoder was handed.

/// The first `len` bytes of `at` as 32-bit words, logging them eight to a
/// line; empty when `at` is not readable.
fn dump(what: &str, at: u64, len: usize) -> Vec<u32> {
    if at == 0 {
        return Vec::new();
    }
    // SAFETY: the title's own structure; a page of it is readable because
    // the call above just wrote it. Read as words, so no alignment is
    // assumed beyond what the title promises.
    let words = unsafe { words(at, len.max(0x40)) };
    if words.is_empty() {
        return Vec::new();
    }
    for line in words.chunks(8) {
        let text = line
            .iter()
            .map(|(off, w)| format!("{off:#04x}:{w:#010x}"))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::info!("bink: {what}: {text}");
    }
    words.into_iter().map(|(_, w)| w).collect()
}

/// Up to `len` bytes at `at` as (offset, 64-bit word) pairs, for the
/// pointers in a structure; empty when the first page is not there.
unsafe fn pairs(at: u64, len: usize) -> Vec<(usize, u64)> {
    if at == 0 || len == 0 || committed(at as usize).is_none() {
        return Vec::new();
    }
    // SAFETY: the caller vouches for the range being the title's structure.
    unsafe {
        (0..len / 8)
            .map(|i| (i * 8, ((at as usize + i * 8) as *const u64).read_volatile()))
            .collect()
    }
}

/// Up to `len` bytes at `at` as (offset, word) pairs, empty when the first
/// page is not there.
unsafe fn words(at: u64, len: usize) -> Vec<(usize, u32)> {
    if at == 0 || len == 0 || committed(at as usize).is_none() {
        return Vec::new();
    }
    // SAFETY: the caller vouches for the range being the title's structure.
    unsafe {
        (0..len / 4)
            .map(|i| (i * 4, ((at as usize + i * 4) as *const u32).read_volatile()))
            .collect()
    }
}

/// The size of the committed region at `at`, when `at` looks like a pointer
/// into this process.
fn committed(at: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEMORY_BASIC_INFORMATION, MEM_COMMIT, VirtualQuery,
    };
    if at < 0x1_0000 || at > 0x0000_7fff_ffff_ffff {
        return None;
    }
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a plain query about an address in this process.
    let got = unsafe {
        VirtualQuery(
            at as *const _,
            &mut info,
            std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    (got != 0 && info.State == MEM_COMMIT).then_some(info.RegionSize)
}

/// Records the registered buffers with [`victory_gpu::movie`], which matches
/// them against the textures the movie samples.
fn note_planes(pointers: &[(usize, usize, usize)]) {
    const NAMES: [&str; 4] = ["y", "cr", "cb", "a"];
    for (i, &(_, at, size)) in pointers.iter().take(NAMES.len()).enumerate() {
        victory_gpu::movie::note(at, size.min(64 << 20), NAMES[i]);
    }
}

// What the decoder put there.

/// A buffer as it last read: where, the bytes sampled, and how many of them
/// differ from the sample before.
struct Sampled {
    at: usize,
    last: Vec<u8>,
    changed: usize,
}

static SAMPLES: Mutex<Vec<Sampled>> = Mutex::new(Vec::new());

/// How many buffers were sampled so far, so the first frames say everything
/// and the rest only when something changed.
static WAITS: AtomicUsize = AtomicUsize::new(0);

/// Reads each registered buffer after a wait, and says whether, and how
/// much, it changed since the wait before. `wait_ms` is the wait's timeout,
/// logged so that a stalled decode is visible beside a moving one.
fn advances(wait_ms: u64) {
    let planes = victory_gpu::movie::planes();
    let n = WAITS.fetch_add(1, Ordering::Relaxed) + 1;
    let mut samples = SAMPLES.lock().unwrap_or_else(|e| e.into_inner());
    let mut moved = 0;
    let mut report = String::new();
    for plane in &planes {
        // Two 32 KiB windows of the plane: its top rows and its middle.
        let window = plane.size.min(32 << 10).max(1);
        let mid = plane.at + (plane.size / 2).min(plane.size.saturating_sub(window));
        let mut now = Vec::with_capacity(window * 2);
        for at in [plane.at, mid] {
            if committed(at).is_none() {
                continue;
            }
            // SAFETY: a committed page of the title's own buffer, read
            // volatile because the decoder writes it on its own threads.
            let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, window) };
            now.extend_from_slice(bytes);
        }
        let seat = samples.iter().position(|s| s.at == plane.at);
        let seat = match seat {
            Some(seat) => seat,
            None => {
                samples.push(Sampled {
                    at: plane.at,
                    last: Vec::new(),
                    changed: 0,
                });
                samples.len() - 1
            }
        };
        let last = std::mem::replace(&mut samples[seat].last, now.clone());
        let differ = if last.len() == now.len() {
            last.iter().zip(&now).filter(|(a, b)| a != b).count()
        } else {
            now.len()
        };
        samples[seat].changed = differ;
        if differ > 0 {
            moved += 1;
        }
        let percent = differ * 100 / now.len().max(1);
        // The same bytes at the address the texture reads them at, when both
        // are known: a decoder writing one view and a texture reading the
        // other are one memory only if both move together.
        let other = match victory_gpu::movie::delta() {
            Some(delta) => {
                let at = plane.at.wrapping_add_signed(delta);
                match (committed(at), now.len()) {
                    (Some(_), len) if len > 0 => {
                        // SAFETY: committed memory of the same buffer, read
                        // volatile because the decoder writes it.
                        let bytes =
                            unsafe { std::slice::from_raw_parts(at as *const u8, len.min(window * 2)) };
                        let n = (0..bytes.len().min(last.len()))
                            .filter(|&k| bytes[k] != last[k])
                            .count();
                        format!(" [texture view {at:#x} {n} differ from the last console read]")
                    }
                    _ => String::from(" [texture view not committed]"),
                }
            }
            None => String::new(),
        };
        report.push_str(&format!(
            " {} {:#x} {differ}/{percent}%{other}",
            plane.name, plane.at
        ));
    }
    let every = if n <= 3 { 1 } else { 15 };
    if n % every == 0 || moved > 0 {
        // The presented frame the wait came in, so the decoder's own motion
        // can be read against a window's frame numbers: a stretch where the
        // picture does not change is either a decoder that has stopped or a
        // surface nobody re-reads, and only this tells them apart.
        tracing::info!(
            frame = victory_gpu::frame::current(),
            waits = n,
            wait_ms,
            moved,
            "bink: after a wait,{report}"
        );
    }
}
