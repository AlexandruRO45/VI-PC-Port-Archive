//! Copy a color texture into a title buffer, via host READBACK memory.

use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_SAMPLE_DESC};
use windows::core::Interface;

use super::{COPY_SOURCE, CopyLocation, buffer_range, ensure_state};
use crate::host::{Host, HostList};
use crate::resource::{PlacedFootprint, element, family, pc_format};

struct Plan {
    source: D3D12_BOX,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    staging_bytes: u64,
    // Byte offsets in the title buffer, independent of the D3D12 pitch.
    offset: u64,
    row_pitch: u64,
    slice_pitch: u64,
    row_bytes: u64,
    span: u64,
}

/// Copy only the source box, packed at (0,0,0) in staging, then publish the
/// copied rows at the requested title offset. Never copy padding back.
fn plan(
    mut from: D3D12_SUBRESOURCE_FOOTPRINT,
    dst: PlacedFootprint,
    [x, y, z]: [u32; 3],
    src_box: Option<[u32; 6]>,
) -> Option<Plan> {
    let format = from.Format.0 as u32;
    let (texel, block) = element(dst.format)?;
    // Depth/stencil planes, BC, packed video, and console texel-size
    // substitutions are deliberately not reinterpreted as color rows.
    if block != 1
        || element(format) != Some((texel, 1))
        || matches!(format, 19..=22 | 40 | 44..=47 | 55 | 68 | 69 | 89)
        || matches!(dst.format, 68 | 69 | 89)
        || family(format) != family(pc_format(dst.format))
    {
        return None;
    }
    let [left, top, front, right, bottom, back] =
        src_box.unwrap_or([0, 0, 0, from.Width, from.Height, from.Depth]);
    let (w, h, d) = (
        right.checked_sub(left)?,
        bottom.checked_sub(top)?,
        back.checked_sub(front)?,
    );
    if w == 0
        || h == 0
        || d == 0
        || right > from.Width
        || bottom > from.Height
        || back > from.Depth
        || x.checked_add(w)? > dst.width
        || y.checked_add(h)? > dst.height
        || z.checked_add(d)? > dst.depth
    {
        return None;
    }
    let row_bytes = u64::from(w).checked_mul(texel)?;
    let row_pitch = u64::from(dst.row_pitch);
    if row_pitch < u64::from(dst.width) * texel {
        return None;
    }
    let slice_pitch = row_pitch.checked_mul(u64::from(dst.height))?;
    let offset = dst
        .offset
        .checked_add(u64::from(z).checked_mul(slice_pitch)?)?
        .checked_add(u64::from(y) * row_pitch)?
        .checked_add(u64::from(x) * texel)?;
    let span = u64::from(d - 1)
        .checked_mul(slice_pitch)?
        .checked_add(u64::from(h - 1) * row_pitch)?
        .checked_add(row_bytes)?;
    let host_pitch = u32::try_from(row_bytes)
        .ok()?
        .checked_next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)?;
    let staging_bytes = u64::from(host_pitch)
        .checked_mul(u64::from(h))?
        .checked_mul(u64::from(d))?;
    (from.Width, from.Height, from.Depth, from.RowPitch) = (w, h, d, host_pitch);
    Some(Plan {
        source: D3D12_BOX {
            left,
            top,
            front,
            right,
            bottom,
            back,
        },
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: from,
        },
        staging_bytes,
        offset,
        row_pitch,
        slice_pitch,
        row_bytes,
        span,
    })
}

pub(super) fn record(
    list: &HostList,
    dst: CopyLocation,
    at: [u32; 3],
    src: CopyLocation,
    src_box: Option<[u32; 6]>,
) {
    let (Some(host), Some(texture), Some((dst_base, dst_len))) = (
        crate::host::host(),
        crate::resource::host_copy(src.resource),
        crate::resource::buffer_bytes(dst.resource),
    ) else {
        return;
    };
    if src.kind != 0 {
        return;
    }
    // SAFETY: the Arc holds the host texture alive.
    let desc = unsafe { texture.texture.GetDesc() };
    let sub = src.footprint.offset as u32; // union's first word for a subresource location
    #[cfg(feature = "diagnostics")]
    let trace = {
        use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
        static NOTED: AtomicU32 = AtomicU32::new(0);
        let n = NOTED.fetch_add(1, Relaxed);
        if n < 16 {
            let title_format = crate::resource::title_format(src.resource);
            tracing::info!(n, recording_after_present = crate::queue::presented(), src = ?src.resource,
                title_format = ?title_format, host_format = desc.Format.0, width = desc.Width, height = desc.Height,
                layers_or_depth = desc.DepthOrArraySize, mip_levels = desc.MipLevels, sub, ?src_box,
                dst = ?dst.resource, dst_base = ?dst_base, dst_len, dst_footprint = ?dst.footprint,
                ?at, "texture-to-buffer copy requested");
        }
        (n < 16).then_some(n)
    };
    #[cfg(not(feature = "diagnostics"))]
    let _ = (dst_base, dst_len);
    let mips = u32::from(desc.MipLevels);
    let arrays = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(desc.DepthOrArraySize)
    };
    if desc.SampleDesc.Count != 1
        || sub >= mips * arrays
        || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        return;
    }
    // SAFETY: single real subresource and writable footprint. Ask the host
    // rather than inferring mip/volume/slice dimensions from the title ABI.
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    unsafe {
        host.device
            .GetCopyableFootprints(&desc, sub, 1, 0, Some(&mut footprint), None, None, None)
    };
    let Some(copy) = plan(footprint.Footprint, dst.footprint, at, src_box) else {
        #[cfg(feature = "diagnostics")]
        if let Some(n) = trace {
            tracing::warn!(
                n,
                ?footprint,
                "texture-to-buffer footprint unsupported or out of bounds"
            );
        }
        return;
    };
    if buffer_range(dst.resource, copy.offset, copy.span).is_none() {
        return;
    }
    let Some(buffer) = staging(host, copy.staging_bytes) else {
        return;
    };
    if crate::texture::title_written(&texture.texture) {
        crate::texture::upload(list, src.resource, &texture.texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let mut from = src;
    from.resource = texture.texture.as_raw().cast_const().cast();
    let to = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed pointer; the location does not release it.
        pResource: unsafe { std::mem::transmute_copy(&buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: copy.footprint,
        },
    };
    // SAFETY: both resources are live and in their copy states; footprint
    // and box are checked.
    unsafe {
        list.list
            .CopyTextureRegion(&to, 0, 0, 0, (&raw const from).cast(), Some(&copy.source));
    }
    // Hold the title buffer across deferred submission and publish before
    // completion writes wake CPU consumers. Preserve bytes outside the box.
    let dst = crate::object::Held::new(dst.resource);
    list.defer_after(move || {
        let _keep_source_alive = &texture;
        publish(
            &buffer,
            dst.get(),
            &copy,
            #[cfg(feature = "diagnostics")]
            trace,
        );
    });
}

fn staging(host: &Host, size: u64) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer = None;
    // SAFETY: a valid READBACK buffer description and output storage.
    if let Err(e) = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut buffer,
        )
    } {
        tracing::warn!(error = %e, size, "no texture copy readback buffer");
        return None;
    }
    buffer
}

fn publish(
    buffer: &ID3D12Resource,
    dst: crate::object::This,
    copy: &Plan,
    #[cfg(feature = "diagnostics")] trace: Option<u32>,
) {
    let Some(dst) = buffer_range(dst, copy.offset, copy.span) else {
        return;
    };
    let mut data = std::ptr::null_mut();
    let range = D3D12_RANGE {
        Begin: 0,
        End: copy.staging_bytes as usize,
    };
    // SAFETY: the host fence completed; output and range are valid.
    if let Err(e) = unsafe { buffer.Map(0, Some(&range), Some(&mut data)) } {
        tracing::warn!(error = %e, "could not map texture copy readback");
        return;
    }
    let f = copy.footprint.Footprint;
    let _writing = crate::watch::writing(dst as usize, copy.span as usize);
    let from = data.cast::<u8>();
    #[cfg(feature = "diagnostics")]
    if let Some(n) = trace {
        // SAFETY: the first copied row is present in the initialized mapping.
        let first = unsafe { std::slice::from_raw_parts(from, copy.row_bytes.min(16) as usize) };
        tracing::info!(n, after_present = crate::queue::presented(), dst = format_args!("{:p}", dst),
            row_bytes = copy.row_bytes, rows = f.Height, slices = f.Depth, first = ?first,
            "texture-to-buffer fence completed");
    }
    for z in 0..u64::from(f.Depth) {
        for y in 0..u64::from(f.Height) {
            let at = z * copy.slice_pitch + y * copy.row_pitch;
            let read = (z * u64::from(f.Height) + y) * u64::from(f.RowPitch);
            // SAFETY: spans and pitches were checked before allocating the copy.
            unsafe {
                crate::host::write_mapped(
                    dst.add(at as usize),
                    from.add(read as usize),
                    copy.row_bytes as usize,
                )
            };
        }
    }
    // SAFETY: the CPU did not write to the mapped buffer.
    unsafe { buffer.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 })) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;

    #[test]
    fn volume_crop_checks_destination_slice_and_offset_overflow() {
        let src = D3D12_SUBRESOURCE_FOOTPRINT {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Width: 8,
            Height: 4,
            Depth: 3,
            RowPitch: 256,
        };
        let dst = PlacedFootprint {
            offset: 5,
            format: 28,
            width: 4,
            height: 4,
            depth: 4,
            row_pitch: 20,
        };
        let region = Some([3, 2, 1, 5, 4, 3]);
        let checked = plan(src, dst, [1, 1, 1], region).unwrap();
        assert_eq!(
            (checked.offset, checked.slice_pitch, checked.span),
            (109, 80, 108)
        );
        assert_eq!(checked.staging_bytes, 1024);
        assert!(plan(src, dst, [1, 1, 3], region).is_none());
        assert!(
            plan(
                src,
                PlacedFootprint {
                    offset: u64::MAX,
                    ..dst
                },
                [1, 1, 1],
                region
            )
            .is_none()
        );
        assert!(
            plan(
                src,
                PlacedFootprint { format: 118, ..dst },
                [1, 1, 1],
                region
            )
            .is_none()
        );
    }
}
