//! Movie surfaces: the frame buffers the title registers with its Bink
//! decoder, so that a texture whose memory is one of them says what it did
//! with it (`VICTORY_BINK_TRACE=1`, diagnostics only).
//!
//! A title's movie is decoded into buffers the title allocates and hands to
//! the decoder (`BinkRegisterFrameBuffers`), and the picture on screen is a
//! texture that reads that memory — three 8-bit luminance planes converted
//! to RGB by a pixel shader. So when the picture holds while the soundtrack
//! advances, there are two ends to tell apart: whether the decoder writes
//! the memory the texture reads, and whether the texture's own refresh
//! notices the writes. `victory_hle::bink` records what the title
//! registered and what the decoder put there; [`verdict`] is called by the
//! texture path with what it decided about a texture that reads one of those
//! ranges, so one log holds both ends beside each other.
//!
//! Nothing here runs unless the switch is on: [`note`] and [`overlaps`] are
//! inert with an empty table.

use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

/// `VICTORY_BINK_TRACE=1`: record the decoder's buffers and report every
/// texture that reads one.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_BINK_TRACE").is_some_and(|v| v == "1" || v == "true")
    })
}

/// One range the title registered with its decoder: where, how long, and
/// which of the decoder's buffers it is (`y`, `cr`, `cb`, ... as the
/// argument order names them).
#[derive(Clone, Copy)]
pub struct Plane {
    pub at: usize,
    pub size: usize,
    pub name: &'static str,
}

static PLANES: Mutex<Vec<Plane>> = Mutex::new(Vec::new());

/// The decoder will write `size` bytes at `at`.
pub fn note(at: usize, size: usize, name: &'static str) {
    if !on() || at == 0 || size == 0 {
        return;
    }
    let mut planes = PLANES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(same) = planes.iter_mut().find(|p| p.at == at) {
        same.size = same.size.max(size);
        let size = same.size;
        tracing::debug!(at = format_args!("{at:#x}"), size, name, "movie plane registered");
        return;
    }
    planes.push(Plane { at, size, name });
    tracing::info!(
        at = format_args!("{at:#x}"),
        size,
        len = format_args!("{:#x}", size),
        name,
        "movie plane registered"
    );
}

/// Every range recorded so far.
pub fn planes() -> Vec<Plane> {
    PLANES.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The distance between the address the decoder is handed for a buffer and
/// the address a texture reads the same bytes at: one memory, two views
/// (the console's own address, and the one the texture was placed at). Set
/// when a texture's address matches a registered buffer in its low 32 bits,
/// which is what a second view of the same pages looks like from here.
static DELTA: Mutex<Option<isize>> = Mutex::new(None);

/// How far a texture's address sits from the decoder's, once both are known.
pub fn delta() -> Option<isize> {
    if !on() {
        return None;
    }
    *DELTA.lock().unwrap_or_else(|e| e.into_inner())
}

/// The recorded plane `at`..`at + len` lies in, if any — directly, or at the
/// offset a second view of the same pages sits at ([`delta`]: a texture the
/// title placed on the decoder's memory reads it there).
pub fn overlaps(at: usize, len: usize) -> Option<Plane> {
    if !on() {
        return None;
    }
    let planes = PLANES.lock().unwrap_or_else(|e| e.into_inner());
    let inside = |p: &Plane, at: usize| at < p.at + p.size && p.at < at + len.max(1);
    planes
        .iter()
        .find(|p| inside(p, at))
        .or_else(|| {
            let delta = (*DELTA.lock().unwrap_or_else(|e| e.into_inner()))?;
            let console = at.checked_add_signed(-delta)?;
            planes.iter().find(|p| inside(p, console))
        })
        .copied()
}

/// Says where a texture the movie could sample keeps its memory, once per
/// texture, whether or not it is one of the decoder's buffers: the match is
/// by address, so both ends of it have to be in the log.
#[cfg(feature = "diagnostics")]
pub(crate) fn said(resource: usize, desc: &str, memory: usize, size: usize, host_written: bool) {
    if !on() {
        return;
    }
    static SAID: Mutex<Option<Vec<usize>>> = Mutex::new(None);
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        let said = said.get_or_insert_with(Vec::new);
        if said.contains(&resource) {
            return;
        }
        said.push(resource);
    }
    if memory != 0 {
        let planes = planes();
        if let Some(same) = planes
            .iter()
            .find(|p| p.at as u32 == memory as u32 || p.at as u32 == (memory + size) as u32)
        {
            let delta = memory as isize - same.at as isize;
            let mut known = DELTA.lock().unwrap_or_else(|e| e.into_inner());
            if *known != Some(delta) {
                *known = Some(delta);
                tracing::info!(
                    plane_at = format_args!("{:#x}", same.at),
                    texture_at = format_args!("{memory:#x}"),
                    delta,
                    "movie surface: a texture and a decoder buffer differ by one offset"
                );
            }
        }
    }
    let plane = overlaps(memory, size);
    // Whether the second view is *recorded*: a detector that cannot fire is
    // the bug this crew has paid for, so both ends of it say what the view
    // table holds before anything is made to depend on it.
    if let Some(plane) = plane {
        let mine = crate::inplace::views_at(memory, size);
        let theirs = crate::inplace::views_at(plane.at, plane.size);
        tracing::info!(
            texture_at = format_args!("{memory:#x}"),
            plane_at = format_args!("{:#x}", plane.at),
            texture_views = mine.len(),
            plane_views = theirs.len(),
            second_view = crate::inplace::another_view(memory, size),
            texture_view_entries = ?mine.iter().map(|(base, bytes, section, offset)| format!("{base:#x}+{bytes:#x} s{section} o{offset:#x}")).collect::<Vec<_>>(),
            plane_view_entries = ?theirs.iter().map(|(base, bytes, section, offset)| format!("{base:#x}+{bytes:#x} s{section} o{offset:#x}")).collect::<Vec<_>>(),
            "movie surface: what the view table holds at both ends"
        );
    }
    let plane_at = format!("{:#x}", plane.map_or(0, |p| p.at));
    let plane_name = plane.map_or("-", |p| p.name);
    tracing::info!(
        resource = format_args!("{resource:#x}"),
        memory = format_args!("{memory:#x}"),
        plane_at,
        plane = plane_name,
        size,
        host_written,
        desc,
        "movie surface: a texture upload was asked about"
    );
}

/// `VICTORY_MOVIE_DUMP=<frame>:<folder>`: at that presented frame, the bytes
/// each movie-sized texture is about to be uploaded from are written to
/// `<folder>/plane-<memory>.bin` and `<folder>/plane-<memory>.txt`. The
/// staging buffer holds the *detiled* memory — what the host texture is
/// copied from — so a capture of the same frame and one of these files
/// answer the question the counters cannot: does the picture on screen hold
/// the decoder's current frame? `VICTORY_MOVIE_DUMP_SIZE` bounds it to bytes
/// whose size ends in that many hex digits (default: the 3.1 MB-class luma
/// plane, `7e9000`, and the 2.1 MB chroma planes, `21c000`, are both taken).
#[cfg(feature = "diagnostics")]
pub(crate) fn dump_at(frame: u64) -> Option<(&'static std::path::PathBuf, u64)> {
    static WANT: OnceLock<Option<(std::path::PathBuf, u64)>> = OnceLock::new();
    let want = WANT.get_or_init(|| {
        let spec = std::env::var("VICTORY_MOVIE_DUMP").ok()?;
        let (frame, folder) = spec.split_once(':')?;
        let frame = frame.trim().parse().ok()?;
        let folder = std::path::PathBuf::from(folder);
        std::fs::create_dir_all(&folder).ok()?;
        tracing::info!(frame, folder = %folder.display(), "movie surface: a frame will be dumped");
        Some((folder, frame))
    });
    want.as_ref().filter(|(_, want)| *want == frame).map(|(folder, want)| (folder, *want))
}

/// Every this many frames, a texture that reads a movie plane says what its
/// refresh or its upload decided, with the caller's detail.
const EVERY: u64 = 15;

/// The frame last reported in, and the (texture, what) pairs already said
/// this frame.
static LAST: Mutex<(u64, Vec<(usize, String)>)> = Mutex::new((0, Vec::new()));
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Reports what happened to the texture whose memory is `memory`..`memory +
/// size`, at most once a frame per texture and only on every [`EVERY`]th
/// frame. `detail` is evaluated only when the line is written.
pub(crate) fn verdict(what: &str, memory: usize, size: usize, detail: impl FnOnce() -> String) {
    let Some(plane) = overlaps(memory, size) else {
        return;
    };
    let frame = crate::frame::current();
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.0 != frame {
            last.0 = frame;
            last.1.clear();
        }
        if frame % EVERY != 0 || last.1.iter().any(|(at, said)| *at == memory && said == what) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        last.1.push((memory, what.to_owned()));
    }
    tracing::info!(
        frame,
        plane = plane.name,
        at = format_args!("{:#x}", plane.at),
        memory = format_args!("{memory:#x}"),
        size,
        what,
        "movie surface: {}",
        detail()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_overlap_is_found_and_a_neighbour_is_not() {
        // The table is only filled with the switch on, and this test may run
        // beside a run that set it: record the plane either way.
        PLANES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Plane {
                at: 0x1000,
                size: 0x1000,
                name: "y",
            });
        let inside = |at: usize, len: usize| {
            PLANES
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .any(|p| at < p.at + p.size && p.at < at + len.max(1))
        };
        assert!(inside(0x1000, 0x1000));
        assert!(inside(0x1800, 0x100));
        assert!(!inside(0x2000, 0x100));
    }
}
