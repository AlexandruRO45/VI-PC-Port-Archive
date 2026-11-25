//! Whether the adapter's imports of section-backed memory alias that memory.
//!
//! XMem physical pages are section views. Wine maps sections as shared memory,
//! and RADV refuses to import shared memory as a host pointer. vkd3d-proton
//! then allocates unrelated memory and still reports a successful import, so
//! the GPU reads zeros where the title wrote data. Neither the import result
//! nor a CPU Map of the heap shows the substitution: only a GPU read does.
use super::*;
use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};

const CANARY: [u32; 4] = [0x7e57_c0de, 0x5ec7_10a5, 0xa11a_5ed0, 0x0ddb_a115];
const SECTION_BYTES: usize = 65536;

/// Probe once per device. A probe that cannot run keeps the previous
/// behaviour, so an unrelated failure never disables working imports.
pub(super) fn section_imports_alias(device: &ID3D12Device) -> bool {
    match probe(device) {
        Ok(true) => true,
        Ok(false) => {
            crate::diagnostics::log(
                "the host driver does not import section-backed guest memory: the GPU read unrelated memory \
                 after a successful import. Imports of XMem physical pages fail with E_NOTIMPL",
            );
            false
        }
        Err(error) => {
            crate::diagnostics::log(&format!("section import probe could not run: {:?}", error.code()));
            true
        }
    }
}

fn probe(device: &ID3D12Device) -> windows::core::Result<bool> {
    let section = unsafe {
        CreateFileMappingW(INVALID_HANDLE_VALUE, None, windows::Win32::System::Memory::PAGE_READWRITE, 0, SECTION_BYTES as u32, None)?
    };
    let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SECTION_BYTES) };
    let result = if view.Value.is_null() {
        Err(windows::core::Error::from_win32())
    } else {
        // SAFETY: the view is a fresh writable mapping of SECTION_BYTES bytes.
        unsafe { core::ptr::copy_nonoverlapping(CANARY.as_ptr(), view.Value.cast(), CANARY.len()) };
        // Every native object, and the GPU's use of the view, ends inside this call.
        gpu_reads_canary(device, view.Value)
    };
    unsafe {
        if !view.Value.is_null() {
            let _ = UnmapViewOfFile(view);
        }
        let _ = CloseHandle(section);
    }
    result
}

fn gpu_reads_canary(device: &ID3D12Device, view: *mut c_void) -> windows::core::Result<bool> {
    let bytes = size_of_val(&CANARY) as u64;
    let host: ID3D12Device3 = device.cast()?;
    let heap: ID3D12Heap = unsafe { host.OpenExistingHeapFromAddress(view)? };
    let buffer = |flags| D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut source: Option<ID3D12Resource> = None;
    unsafe {
        device.CreatePlacedResource(&heap, 0, &buffer(D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER),
            D3D12_RESOURCE_STATE_COMMON, None, &mut source)?
    };
    let source = source.ok_or(E_POINTER)?;
    let readback_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut readback: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer(D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback)?
    };
    let readback = readback.ok_or(E_POINTER)?;

    let kind = D3D12_COMMAND_LIST_TYPE_COPY;
    let queue: ID3D12CommandQueue = unsafe {
        device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: kind, ..Default::default() })?
    };
    let allocator: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(kind)? };
    let list: ID3D12GraphicsCommandList = unsafe { device.CreateCommandList(0, kind, &allocator, None)? };
    unsafe {
        // A buffer in COMMON promotes to COPY_SOURCE on its first copy.
        list.CopyBufferRegion(&readback, 0, &source, 0, bytes);
        list.Close()?;
        queue.ExecuteCommandLists(&[Some(list.cast()?)]);
    }
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
    unsafe {
        queue.Signal(&fence, 1)?;
        // A null event blocks until the copy completes.
        fence.SetEventOnCompletion(1, HANDLE::default())?;
    }
    let mut mapped = core::ptr::null_mut();
    unsafe { readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: bytes as usize }), Some(&mut mapped))? };
    // SAFETY: the readback buffer holds `bytes` bytes and stays mapped here.
    let read = unsafe { core::slice::from_raw_parts(mapped.cast::<u32>(), CANARY.len()) } == CANARY;
    unsafe { readback.Unmap(0, Some(&D3D12_RANGE::default())) };
    Ok(read)
}
