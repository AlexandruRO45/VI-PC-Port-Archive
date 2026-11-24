//! Slot 62 copies a linear CPU image into a resource named by GPU address.
//!
//! Scarlett's video path (title RVA 0x979650) calls the graphics list with a
//! CPU pointer, the resource's GPU VA, a byte count, and flags. Callers fill
//! that CPU buffer only after the call returns, so the bytes are read at
//! ExecuteCommandLists. The copy itself is recorded at the call so later
//! commands in the same list see the upload.

use std::sync::atomic::{AtomicU32, Ordering};

use windows::core::Result;
use windows::Win32::Foundation::{E_FAIL, E_INVALIDARG, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use super::{parent, recording, List};
use crate::device::{self, LinearResource};

const PLACEMENT: u64 = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64;

struct Region {
    index: u32,
    offset: u64,
    span: u64,
    pitch: u32,
}

enum Plan {
    Whole,
    One(u32),
}

/// Guest rows include the tail of the last row. The driver footprint may end
/// a little earlier, or a packed chain may add placement padding.
fn fits(guest: u64, needed: u64, slack: u64) -> bool {
    guest >= needed && guest - needed < slack
}

fn plan(guest_offset: u64, guest_bytes: u64, total: u64, regions: &[Region]) -> Option<Plan> {
    if regions.is_empty() || guest_bytes == 0 || total == 0 {
        return None;
    }
    let mut last_end = 0u64;
    let mut slack = PLACEMENT;
    for region in regions {
        if region.pitch == 0 || region.span == 0 {
            return None;
        }
        let end = region.offset.checked_add(region.span)?;
        last_end = last_end.max(end);
        slack = slack.max(u64::from(region.pitch));
    }
    let whole = guest_offset == 0
        && regions
            .iter()
            .all(|region| region.offset.is_multiple_of(PLACEMENT))
        && last_end <= guest_bytes
        && fits(guest_bytes, last_end, slack)
        && guest_bytes.saturating_add(slack) >= total
        && fits(total.max(guest_bytes), guest_bytes.min(total), slack);
    if whole {
        return Some(Plan::Whole);
    }
    regions
        .iter()
        .find(|region| {
            let at_start = guest_offset == 0 && region.index == 0;
            (at_start || guest_offset == region.offset)
                && fits(
                    guest_bytes,
                    region.span,
                    u64::from(region.pitch).max(PLACEMENT),
                )
        })
        .map(|region| Plan::One(region.index))
}

fn footprint_bytes(pitch: u32, rows: u32, depth: u32) -> Option<u64> {
    u64::from(pitch)
        .checked_mul(u64::from(rows))?
        .checked_mul(u64::from(depth))
}

pub(super) struct Pending {
    source: usize,
    bytes: usize,
    mapped: usize,
    buffer: ID3D12Resource,
}

impl Pending {
    fn mapped(buffer: ID3D12Resource, source: usize, bytes: usize) -> Result<Self> {
        let mut mapped = core::ptr::null_mut();
        let unread = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: the upload buffer has `bytes` bytes and has not been mapped.
        unsafe { buffer.Map(0, Some(&unread), Some(&mut mapped))? };
        if mapped.is_null() {
            return Err(E_FAIL.into());
        }
        Ok(Self {
            source,
            bytes,
            mapped: mapped as usize,
            buffer,
        })
    }

    pub(super) fn snapshot(&self) -> windows::core::Result<()> {
        if self.source == 0 || self.mapped == 0 || self.bytes == 0 {
            return Err(E_INVALIDARG.into());
        }
        // SAFETY: `source` is the guest allocation captured at record time.
        // The title keeps it alive through ExecuteCommandLists, which is the
        // caller. `mapped` is this upload buffer's mapping of `bytes` bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.source as *const u8,
                self.mapped as *mut u8,
                self.bytes,
            );
        }
        note_snapshot(self.source, self.bytes);
        Ok(())
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if self.mapped != 0 {
            // SAFETY: Map succeeded on this buffer and no copy is still using
            // the pointer. Reset drops the record before the native list is reused.
            unsafe { self.buffer.Unmap(0, None) };
            self.mapped = 0;
        }
    }
}

fn note(detail: &str) {
    static COUNT: AtomicU32 = AtomicU32::new(0);
    if COUNT.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::diagnostics::log(detail);
    }
}

fn note_snapshot(source: usize, bytes: usize) {
    static COUNT: AtomicU32 = AtomicU32::new(0);
    if COUNT.fetch_add(1, Ordering::Relaxed) >= 4 {
        return;
    }
    let prefix = bytes.min(16);
    // SAFETY: snapshot just read `bytes` from this guest pointer.
    let (hex, zero) = unsafe {
        let data = core::slice::from_raw_parts(source as *const u8, bytes);
        let hex: String = data
            .iter()
            .take(prefix)
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let step = (bytes / 64).max(1);
        let zero = data.iter().step_by(step).all(|byte| *byte == 0);
        (hex, zero)
    };
    crate::diagnostics::log(&format!(
        "slot 62 snapshot {bytes:#x} bytes from {source:#x} first={hex} {}",
        if zero {
            "source samples are zero"
        } else {
            "source has data"
        }
    ));
}

fn upload_buffer(device: &ID3D12Device, bytes: u64) -> Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
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
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut native = None;
    // SAFETY: D3D12 borrows the descriptor only for the duration of the call.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut native,
        )?;
    }
    native.ok_or_else(|| E_FAIL.into())
}

fn plane_count(
    device: &ID3D12Device,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
) -> u32 {
    let mut info = D3D12_FEATURE_DATA_FORMAT_INFO {
        Format: format,
        PlaneCount: 1,
    };
    // SAFETY: info is a live FORMAT_INFO record and the size matches that struct.
    let queried = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_INFO,
            core::ptr::from_mut(&mut info).cast(),
            core::mem::size_of_val(&info) as u32,
        )
    };
    if queried.is_ok() && info.PlaneCount > 0 {
        u32::from(info.PlaneCount)
    } else {
        1
    }
}

fn subresource_count(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> Option<u32> {
    let mips = u32::from(desc.MipLevels);
    // NV12 and the other planar video formats have one subresource per plane.
    let (slices, planes) = match desc.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D => (
            u32::from(desc.DepthOrArraySize),
            plane_count(device, desc.Format),
        ),
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => (1, 1),
        _ => return None,
    };
    mips.checked_mul(slices)?
        .checked_mul(planes)
        .filter(|count| (1..=2048).contains(count))
}

fn regions_for(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
) -> Result<(u64, Vec<Region>, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>)> {
    let desc = unsafe { resource.GetDesc() };
    let count = subresource_count(device, &desc).ok_or(E_INVALIDARG)?;
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut row_bytes = vec![0u64; count as usize];
    let mut total = 0u64;
    // SAFETY: the three output slices have one entry per subresource.
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            count,
            0,
            Some(layouts.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(row_bytes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    let mut regions = Vec::with_capacity(count as usize);
    for (index, layout) in layouts.iter().enumerate() {
        let footprint = &layout.Footprint;
        let pitch = footprint.RowPitch;
        let row = row_bytes[index];
        if footprint.Width == 0
            || rows[index] == 0
            || footprint.Depth == 0
            || row == 0
            || row > u64::from(pitch)
        {
            crate::diagnostics::recording_error(format_args!(
                "slot 62 subresource {index} of {}x{} format {} has no copyable footprint",
                desc.Width, desc.Height, desc.Format.0
            ));
            return Err(E_INVALIDARG.into());
        }
        let Some(span) = footprint_bytes(pitch, rows[index], footprint.Depth) else {
            return Err(E_INVALIDARG.into());
        };
        regions.push(Region {
            index: index as u32,
            offset: layout.Offset,
            span,
            pitch,
        });
    }
    Ok((total, regions, layouts))
}

fn record_texture(
    list: &ID3D12GraphicsCommandList,
    destination: &ID3D12Resource,
    source: &ID3D12Resource,
    subresource: u32,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let mut src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(source.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    let mut dst = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(destination.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: subresource,
        },
    };
    // SAFETY: each location's active union field matches its type tag.
    // CopyTextureRegion copies the resource pointers before returning.
    unsafe {
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        std::mem::ManuallyDrop::drop(&mut src.pResource);
        std::mem::ManuallyDrop::drop(&mut dst.pResource);
    }
}

fn texture_copies(
    list: &List,
    state: &mut super::Recording,
    device: &ID3D12Device,
    destination: &ID3D12Resource,
    placed: Option<&std::sync::Arc<crate::device::placement::Texture>>,
    guest_offset: u64,
    guest_bytes: u64,
    source: &ID3D12Resource,
) -> Result<()> {
    if let Some(texture) = placed {
        state
            .prepare_copy(Some(texture), None)
            .map_err(windows::core::Error::from)?;
    }
    let (total, regions, layouts) = regions_for(device, destination)?;
    let Some(plan) = plan(guest_offset, guest_bytes, total, &regions) else {
        let desc = unsafe { destination.GetDesc() };
        let heads: Vec<String> = regions
            .iter()
            .take(4)
            .map(|region| {
                format!(
                    "#{} off={:#x} span={:#x} pitch={}",
                    region.index, region.offset, region.span, region.pitch
                )
            })
            .collect();
        crate::diagnostics::recording_error(format_args!(
            "slot 62 texture {}x{}x{} format={} mips={} offset={guest_offset:#x} bytes={guest_bytes:#x} total={total:#x} [{}]",
            desc.Width, desc.Height, desc.DepthOrArraySize, desc.Format.0, desc.MipLevels, heads.join(", ")
        ));
        return Err(E_INVALIDARG.into());
    };
    match plan {
        Plan::Whole => {
            for (region, layout) in regions.iter().zip(&layouts) {
                record_texture(&list.native, destination, source, region.index, *layout);
            }
        }
        Plan::One(index) => {
            let layout = layouts.get(index as usize).ok_or(E_INVALIDARG)?;
            let mut layout = *layout;
            layout.Offset = 0;
            record_texture(&list.native, destination, source, index, layout);
        }
    }
    Ok(())
}

pub(super) unsafe extern "system" fn linear_upload(
    list: &List,
    source: *const u8,
    address: u64,
    bytes: u32,
    flags: u32,
) {
    let mut state = recording(list);
    if !state.can_record(&list.native) {
        return;
    }
    let bytes_u = u64::from(bytes);
    if source.is_null() || address == 0 || bytes == 0 || flags != 0 {
        crate::diagnostics::recording_error(format_args!(
            "slot 62 linear upload source={source:p} address={address:#x} bytes={bytes:#x} flags={flags:#x}"
        ));
        state.error = E_INVALIDARG;
        return;
    }
    let result = (|| -> Result<()> {
        let resolved = unsafe { device::linear_resource(parent(&list.parent), address, bytes_u)? };
        let mut owner: Option<ID3D12Device> = None;
        unsafe { list.native.GetDevice(&mut owner)? };
        let device = owner.ok_or(E_FAIL)?;
        let upload = upload_buffer(&device, bytes_u)?;
        let pending = Pending::mapped(upload.clone(), source as usize, bytes as usize)?;
        state
            .uploads
            .try_reserve(1)
            .map_err(|_| windows::core::Error::from(E_OUTOFMEMORY))?;
        super::raytracing::retain(&state, &[&upload])?;
        match &resolved {
            LinearResource::Buffer { resource, offset } => {
                super::raytracing::retain(&state, &[resource])?;
                // SAFETY: both buffers belong to this open list and contain the range.
                unsafe {
                    list.native
                        .CopyBufferRegion(resource, *offset, &upload, 0, bytes_u)
                };
                note(&format!(
                    "slot 62 buffer upload {bytes:#x} bytes to {address:#x} offset={offset:#x}"
                ));
            }
            LinearResource::Texture {
                resource,
                placed,
                offset,
            } => {
                super::raytracing::retain(&state, &[resource])?;
                texture_copies(
                    list,
                    &mut state,
                    &device,
                    resource,
                    placed.as_ref(),
                    *offset,
                    bytes_u,
                    &upload,
                )?;
                note(&format!(
                    "slot 62 texture upload {bytes:#x} bytes to {address:#x} offset={offset:#x}"
                ));
            }
        }
        state.uploads.push(pending);
        Ok(())
    })();
    if let Err(error) = result {
        state.error = error.code();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(index: u32, offset: u64, span: u64, pitch: u32) -> Region {
        Region {
            index,
            offset,
            span,
            pitch,
        }
    }

    #[test]
    fn one_plane_matches_its_row_pitch() {
        let regions = [region(0, 0, 8_294_400, 7680)];
        assert!(matches!(
            plan(0, 8_294_400, 8_294_400, &regions),
            Some(Plan::Whole)
        ));
    }

    #[test]
    fn mip0_is_not_the_whole_chain() {
        let regions = [region(0, 0, 1000, 256), region(1, 1024, 500, 256)];
        assert!(matches!(plan(0, 1000, 1524, &regions), Some(Plan::One(0))));
    }

    #[test]
    fn packed_chain_uses_driver_offsets() {
        let regions = [region(0, 0, 2048, 256), region(1, 2048, 1024, 256)];
        assert!(matches!(plan(0, 3072, 3072, &regions), Some(Plan::Whole)));
    }

    #[test]
    fn second_subresource_can_be_addressed_alone() {
        let regions = [region(0, 0, 2048, 256), region(1, 2048, 1024, 256)];
        assert!(matches!(
            plan(2048, 1024, 3072, &regions),
            Some(Plan::One(1))
        ));
    }

    #[test]
    fn oversized_guest_buffer_is_rejected() {
        let regions = [region(0, 0, 1000, 256)];
        assert!(plan(0, 50_000, 1000, &regions).is_none());
    }
}
