//! Per-slot console descriptor strides.
//!
//! The compute stride table (root SRV t0, space32767) tells the translated
//! shader the stride of the descriptor record each heap slot names, so an
//! unformatted indexed load (IDXEN) can compute its address at runtime. The
//! host creates every heap view through the descriptor shims (srv/uav), which
//! know the view's `StructureByteStride`, so the stride is recorded here at
//! creation time, keyed by the absolute CPU descriptor handle.
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct Strides(std::sync::Mutex<BTreeMap<usize, u32>>);

impl Strides {
    /// Record the stride of the descriptor at `handle`. Views created before
    /// this call always rewrite the slot.
    pub(super) fn set(&self, handle: usize, stride: u32) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).insert(handle, stride);
    }

    /// Iterate every tracked entry, sorted by handle address.
    pub(super) fn all(&self) -> Vec<(usize, u32)> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
            .iter().map(|(&handle, &stride)| (handle, stride)).collect()
    }

    /// The tracked strides within the CPU range `[start, end)`, as
    /// (handle address, stride) pairs sorted by address.
    pub(super) fn collect(&self, start: usize, end: usize) -> Vec<(usize, u32)> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
            .range(start..end)
            .map(|(&handle, &stride)| (handle, stride))
            .collect()
    }
}

#[cfg(test)]
#[path = "../tests/device/strides.rs"]
mod tests;


/// The stride (bytes per element) of a buffer view, as the console's descriptor
/// record carries it: the structure stride when it is set, raw views count
/// their elements one byte, and a formatted view by its format's element.
pub(super) fn view_stride(format: i32, raw: bool, structure_stride: u32) -> u32 {
    if structure_stride > 0 {
        return structure_stride;
    }
    if raw {
        return 1;
    }
    super::views::element_stride(format, 0).unwrap_or(0)
}

use windows::Win32::Graphics::Direct3D12::*;

/// The console stride of the buffer view the game asked for, from the raw
/// description (before any host adjustment). 0 when the view is not a buffer
/// view or its stride is not known.
pub(super) unsafe fn srv_stride(desc: *const D3D12_SHADER_RESOURCE_VIEW_DESC) -> u32 {
    let Some(view) = (unsafe { desc.as_ref() }) else { return 0 };
    if view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER { return 0 };
    let b = unsafe { view.Anonymous.Buffer };
    view_stride(view.Format.0, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, b.StructureByteStride)
}

pub(super) unsafe fn uav_stride(desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC) -> u32 {
    let Some(view) = (unsafe { desc.as_ref() }) else { return 0 };
    if view.ViewDimension != D3D12_UAV_DIMENSION_BUFFER { return 0 };
    let b = unsafe { view.Anonymous.Buffer };
    view_stride(view.Format.0, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, b.StructureByteStride)
}
