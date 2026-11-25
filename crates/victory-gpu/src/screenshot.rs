//! The screenshot the title asks the display for: the next presented frame,
//! in BGRA8, handed back through `display.rs`'s `take_screenshot`.
//!
//! `capture.rs` already reads a texture back into a readback buffer for its
//! frame saves, but that module is diagnostics-only and its job is writing
//! PNGs. A screenshot is asked for by the title in every build, and what it
//! needs is the bytes, not a file, so the read-back lives here too. The
//! sequence is the one `display.rs` describes: the title asks, the next
//! `PresentX` copies the frame, and the title picks the bytes up.

use windows::Win32::Graphics::Direct3D12::*;

/// One frame as the title takes it: colour in BGRA8, `pitch` bytes a row,
/// `height` rows.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pitch: usize,
    pub bgra: Vec<u8>,
}

struct Waiting {
    /// Rows are 0 until a frame has filled this in.
    frame: Option<Frame>,
}

static ASKED: std::sync::Mutex<Vec<Waiting>> = std::sync::Mutex::new(Vec::new());

/// Whether a frame is wanted: the title asked and has not taken one.
pub(crate) fn wanted() -> bool {
    let asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    asked.iter().any(|w| w.frame.is_none())
}

/// The title asked for a screenshot. Two asks before a frame is presented
/// take one frame, which is what the title's own sequence does on the
/// console too.
pub fn ask() {
    let mut asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    if asked.len() >= 2 {
        asked.remove(0);
    }
    asked.push(Waiting { frame: None });
}

/// Takes the oldest finished frame, if one is there.
pub fn take() -> Option<Frame> {
    let mut asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    let at = asked.iter().position(|w| w.frame.is_some())?;
    asked.remove(at).frame
}

/// Waits up to `limit` for a frame asked for by [`ask`], then takes it. The
/// wait polls: the frame is filled on the thread that presents, and a
/// condition variable would need that thread to take the same lock while it
/// holds the swap chain's.
pub fn take_blocking(limit: std::time::Duration) -> Option<Frame> {
    let started = std::time::Instant::now();
    loop {
        if let Some(frame) = take() {
            return Some(frame);
        }
        if started.elapsed() >= limit {
            // Nothing came: drop the ask so a later present does not fill a
            // screenshot nobody will collect.
            drop_ask();
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// Reads `texture` (in the copy-source state) into a readback buffer, on
/// `list`, and keeps the buffer so [`fill`] can map it once the submission
/// has run.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a readback buffer; no initial state.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
    }
    let buffer = buffer?;
    let f = footprint.Footprint;
    let to = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(buffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    let from = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(texture.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let _ = (f.Width, f.Height);
    // SAFETY: both locations are live resources of the sizes described.
    unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    Some((buffer, footprint))
}

/// Fills the oldest unfilled ask from a buffer [`read_back`] made. A format
/// other than 8-bit colour cannot be taken, and the ask is dropped rather
/// than left waiting for every later frame.
pub(crate) fn fill(buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let f = footprint.Footprint;
    let source_bgra = match f.Format {
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM
        | windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM
        | windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "screenshot not taken: not 8-bit color");
            drop_ask();
            return;
        }
    };
    let (width, height, pitch) = (f.Width, f.Height, f.RowPitch as usize);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy has run (a submission is waited for before its
    // readback is mapped); the mapping spans the footprint's rows.
    let bgra = unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            drop_ask();
            return;
        }
        let len = pitch * height as usize;
        let raw = std::slice::from_raw_parts(data.cast::<u8>(), len);
        let out = to_bgra(source_bgra, raw);
        buffer.Unmap(0, None);
        out
    };
    let mut asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(slot) = asked.iter_mut().find(|w| w.frame.is_none()) {
        slot.frame = Some(Frame {
            width,
            height,
            pitch,
            bgra,
        });
    }
}

fn drop_ask() {
    let mut asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(at) = asked.iter().position(|w| w.frame.is_none()) {
        asked.remove(at);
    }
}

/// `rows` as BGRA8: copied when the source already is, otherwise its red
/// and blue swapped four bytes at a time. Pure, so a test can hold it.
fn to_bgra(source_bgra: bool, rows: &[u8]) -> Vec<u8> {
    if source_bgra {
        return rows.to_vec();
    }
    let mut out = vec![0u8; rows.len()];
    for (dst, src) in out.chunks_exact_mut(4).zip(rows.chunks_exact(4)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = src[3];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The queue the title's asks go through, without a GPU: an ask shows
    /// as wanted until a frame is filled in, `take` gives back the oldest
    /// finished one, and a third ask before any frame drops the oldest, so
    /// asking in a loop cannot grow the queue.
    ///
    /// These are the parts of a screenshot that are not D3D12; the read-back
    /// and the mapping need a device and are exercised by the title's own
    /// calls.
    #[test]
    fn asks_queue_and_are_taken_oldest_first() {
        // The queue is process-wide, so this runs alone with its own empty
        // start.
        reset();
        assert!(!wanted());
        assert!(take().is_none());

        ask();
        assert!(wanted(), "an ask is wanted until a frame fills it");
        assert!(take().is_none(), "an unfilled ask is not a frame");

        // Fill the ask by hand: what `fill` writes after a present.
        fill_for_test(4, 2, 8, vec![0xAB; 16]);
        assert!(!wanted(), "a filled ask is no longer wanted");
        let frame = take().expect("the filled frame");
        assert_eq!((frame.width, frame.height, frame.pitch), (4, 2, 8));
        assert_eq!(frame.bgra.len(), 16);
        assert!(take().is_none(), "taken once");

        // Three asks: the oldest is dropped, so two remain and no more.
        ask();
        ask();
        ask();
        assert_eq!(asks(), 2, "a third ask drops the oldest");
        reset();
    }

    fn reset() {
        ASKED.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    fn asks() -> usize {
        ASKED.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Fills the oldest unfilled ask without a device.
    fn fill_for_test(width: u32, height: u32, pitch: usize, bgra: Vec<u8>) {
        let mut asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = asked.iter_mut().find(|w| w.frame.is_none()) {
            slot.frame = Some(Frame {
                width,
                height,
                pitch,
                bgra,
            });
        }
    }

    /// The conversion the title's screenshot depends on: a swap chain that
    /// presents RGBA has to reach it as BGRA, and one that already presents
    /// BGRA must not be swapped twice.
    #[test]
    fn rgba_rows_arrive_as_bgra() {
        // One RGBA texel (1, 2, 3, 4) becomes (3, 2, 1, 4).
        assert_eq!(super::to_bgra(false, &[1, 2, 3, 4]), vec![3, 2, 1, 4]);
        // Two texels, so the chunks line up across a row.
        assert_eq!(
            super::to_bgra(false, &[10, 20, 30, 40, 50, 60, 70, 80]),
            vec![30, 20, 10, 40, 70, 60, 50, 80]
        );
        // Already BGRA: unchanged, not swapped back.
        assert_eq!(super::to_bgra(true, &[3, 2, 1, 4]), vec![3, 2, 1, 4]);
        // A trailing partial texel comes back as a zero texel rather than a
        // shifted row: `chunks_exact` leaves it in the (already zero) output.
        assert_eq!(super::to_bgra(false, &[1, 2, 3, 4, 9]), vec![3, 2, 1, 4, 0]);
        assert!(super::to_bgra(false, &[]).is_empty());
    }
}
