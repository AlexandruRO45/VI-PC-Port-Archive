//! The frame rate and frame time written in the picture's top left corner,
//! as rectangles cleared on the back buffer: no shader, texture or font.
//! `VICTORY_FPS=0` turns it off.

use std::sync::OnceLock;

use windows::Win32::Foundation::RECT;

/// Maximum rectangles passed to a single `ClearRenderTargetView` call.
/// NVIDIA's D3D12 driver (`nvwgf2umx.dll`) terminates with `0xc0000409`
/// (`STATUS_STACK_BUFFER_OVERRUN`) when a single call passes 64 or more
/// rectangles (a typical readout like `"30.0 FPS  33 MS"` emits ~98).
pub(crate) const MAX_CLEAR_RECTS: usize = 16;

/// Whether the readout is drawn.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_FPS").map_or(true, |v| v != "0"))
}

/// A glyph's seven rows, top first, of five dots each (bit 4 is the left).
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
        '1' => [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
        '2' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
        '3' => [0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e],
        '4' => [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
        '5' => [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
        '6' => [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
        '7' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
        '9' => [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c],
        'F' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
        'P' => [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
        'S' => [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
        'M' => [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11],
        _ => [0; 7],
    }
}

/// The text's width and height in dots, with a dot of margin all round.
pub(crate) fn size(text: &str) -> (i32, i32) {
    (text.chars().count() as i32 * 6 + 1, 9)
}

/// The rectangles of `text` with its top left margin at `at` and dots of
/// `dot` pixels: one for each run of dots in a row.
pub(crate) fn rects(text: &str, at: (i32, i32), dot: i32) -> Vec<RECT> {
    let mut all = Vec::new();
    for (i, c) in text.chars().enumerate() {
        let left = at.0 + (1 + i as i32 * 6) * dot;
        for (row, bits) in glyph(c).into_iter().enumerate() {
            let top = at.1 + (1 + row as i32) * dot;
            let mut x = 0;
            while x < 5 {
                if bits & (0x10 >> x) == 0 {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < 5 && bits & (0x10 >> x) != 0 {
                    x += 1;
                }
                all.push(RECT {
                    left: left + start * dot,
                    top,
                    right: left + x * dot,
                    bottom: top + dot,
                });
            }
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_of_dots_is_one_rectangle() {
        // The top row of "F" is five dots.
        let all = rects("F", (10, 20), 4);
        assert_eq!(
            all[0],
            RECT {
                left: 14,
                top: 24,
                right: 34,
                bottom: 28
            }
        );
        assert!(rects(" ", (0, 0), 4).is_empty());
        assert_eq!(size("12 FPS"), (37, 9));
    }

    /// Verifies that recording typical readout strings in `MAX_CLEAR_RECTS`
    /// chunks succeeds on the host D3D12 device without triggering driver
    /// limits (e.g. NVIDIA `nvwgf2umx.dll` `0xc0000409` on >= 64 rects).
    #[test]
    fn clear_readout_rects_in_chunks_succeeds_on_host() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return;
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1920,
            Height: 1080,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut rt: Option<ID3D12Resource> = None;
        // SAFETY: plain render-target resource creation on the host device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .expect("create render target");
        }
        let rt = rt.unwrap();
        // SAFETY: plain 1-slot RTV heap and RTV on the host device.
        let rtv: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .expect("create RTV heap")
        };
        // SAFETY: `rt` is a live render target and `rtv` has 1 slot.
        let view = unsafe { rtv.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: `rt` matches the RTV heap.
        unsafe { host.device.CreateRenderTargetView(&rt, None, view) };

        let list = host.new_list().expect("new list");
        for text in ["30.0 FPS  33 MS", "230.0 FPS  4 MS", "888.8 FPS  888 MS"] {
            let marks = rects(text, (15, 15), 5);
            assert!(marks.len() >= 64);
            for chunk in marks.chunks(MAX_CLEAR_RECTS) {
                // SAFETY: open list, live RTV, rectangles inside 1920x1080.
                unsafe {
                    list.list
                        .ClearRenderTargetView(view, &[1.0, 0.9, 0.1, 1.0], Some(chunk));
                }
            }
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
