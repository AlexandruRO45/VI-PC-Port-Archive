//! The per-slot console descriptor stride table (root SRV t0, space32767).
//!
//! The translated compute shader reads the stride of the descriptor record its
//! IDXEN load names from a host-provided table instead of refusing the load
//! (see `sibling_stride` in the compiler for the compile-time fast path).
//! Entry 0 is a valid flag; entry (1 + slot) is the stride in bytes of the
//! record at the guest heap slot `slot`. Entries the host never recorded read
//! 0, which the shader treats as a raw byte view - the same semantics a raw
//! console descriptor carries - so an untracked slot cannot silently compute
//! a wrong address.
use windows::Win32::Foundation::{E_INVALIDARG, E_POINTER};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, IUnknown, Result};

use super::{blas::upload_buffer, descriptor_paging::Arena, Device, MediaResources};

/// Populate and upload the stride table for the heaps bound to the list.
///
/// The table is indexed by the guest heap slot: the identity path's guest
/// index equals the native slot, so the shader's direct
/// `ResourceDescriptorHeap` binding is unchanged. The arena path (indirect
/// descriptor handles) is handled separately: the shader names the slot by
/// its own guest index.
///
/// # Safety
/// The owner must be the controlling unknown of the live guest device.
pub(crate) unsafe fn populate(
    owner: &IUnknown,
    arena: Option<&Arena>,
    heaps: &[Option<ID3D12DescriptorHeap>],
    resources: &MediaResources,
) -> Result<u64> {
    let device = unsafe { &*owner.as_raw().cast::<Device>() };
    let strides = &device.state.descriptor_strides;
    let mut entries: Vec<u32> = vec![1]; // entry 0: valid flag
    let mut next_index = 1u32;
    if let Some(arena) = arena {
        let stride = arena.guest_stride();
        for (handle, stride_of_view) in strides.all() {
            if !arena.contains(handle as u64) { continue; }
            let slot = (handle - arena.base()) / stride;
            let index = next_index.checked_add(slot as u32).ok_or(E_INVALIDARG)?;
            if entries.len() <= index as usize {
                entries.resize(index as usize + 1, 0);
            }
            entries[index as usize] = stride_of_view;
        }
        next_index = next_index.saturating_add(arena.count() as u32);
    }
    for heap in heaps.iter().flatten() {
        let desc = unsafe { heap.GetDesc() };
        // Only shader-visible CBV/SRV/UAV heaps hold stride records for
        // compute shaders. Sampler heaps are not indexed by this table.
        if desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV { continue; }
        let count = desc.NumDescriptors;
        let mut native = None::<ID3D12Device>;
        unsafe { heap.GetDevice(&mut native)? };
        let increment = unsafe {
            native
                .ok_or(E_INVALIDARG)?
                .GetDescriptorHandleIncrementSize(desc.Type) as usize
        };
        if increment == 0 { return Err(E_INVALIDARG.into()); }
        let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr;
        let end = start
            .checked_add(increment.checked_mul(count as usize).ok_or(E_INVALIDARG)?)
            .ok_or(E_INVALIDARG)?;
        for (handle, stride) in strides.collect(start, end) {
            let offset = handle.checked_sub(start).ok_or(E_INVALIDARG)?;
            let slot = offset / increment;
            let index = next_index.checked_add(slot as u32).ok_or(E_INVALIDARG)?;
            if entries.len() <= index as usize {
                entries.resize(index as usize + 1, 0);
            }
            entries[index as usize] = stride;
        }
        next_index = next_index.saturating_add(count);
    }
    if entries.len() <= 1 {
        // No tracked records: the decoration is only useful where the host
        // can name a stride. Refuse so the guest keeps its own table instead
        // of a silently zero one.
        return Err(E_INVALIDARG.into());
    }
    let bytes = (entries.len() as u64).checked_mul(4).ok_or(E_INVALIDARG)?;
    let resource = upload_buffer(&device.native, bytes)?;
    let mut data = std::ptr::null_mut();
    unsafe { resource.Map(0, Some(&D3D12_RANGE::default()), Some(&mut data))? };
    if data.is_null() {
        return Err(E_POINTER.into());
    }
    unsafe {
        std::slice::from_raw_parts_mut(data as *mut u32, entries.len()).copy_from_slice(&entries);
    }
    unsafe { resource.Unmap(0, None) };
    let address = unsafe { resource.GetGPUVirtualAddress() };
    if address == 0 {
        return Err(E_POINTER.into());
    }
    let mut held = resources.lock().unwrap_or_else(|e| e.into_inner());
    let pageable = ID3D12Pageable::from(resource);
    if !held.contains(&pageable) {
        held.try_reserve(1).map_err(|_| windows::Win32::Foundation::E_OUTOFMEMORY)?;
        held.push(pageable);
    }
    Ok(address)
}
