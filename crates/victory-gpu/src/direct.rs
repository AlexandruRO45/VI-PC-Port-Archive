//! Local build: title memory the host GPU reads in place, without copies (the technique of
//! XCL's unified memory, `ID3D12Device3::OpenExistingHeapFromAddress`).
//!
//! A title buffer in an ordinary allocation of the title's (private memory, committed in
//! whole, under 4 GiB) gets a buffer placed on a heap opened over that allocation: the GPU
//! reads and writes title memory itself, so the buffer's `Shadow` needs no refresh before a
//! submission and no copy back after it. Memory that cannot be opened (physical-page views,
//! partly committed reservations) keeps its copy.
//!
//! `VICTORY_DIRECT=0` turns it off.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
};

/// Placed buffer offsets are 64 KiB aligned.
const ALIGN: u64 = 1 << 16;
/// Largest heap the host opens from an address.
const MAX_HEAP: usize = (4096 << 20) - (1 << 16);

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DIRECT").map_or(true, |v| v.trim() != "0"))
}

struct Heap {
    heap: Option<ID3D12Heap>,
    base: usize,
    size: usize,
}

// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for Heap {}

/// Heaps opened, by allocation base (`None` heap: the allocation cannot be opened).
static HEAPS: Mutex<Option<HashMap<usize, Heap>>> = Mutex::new(None);

/// The allocation holding `address`: its base and size, if it is private memory committed in
/// whole.
fn allocation(address: usize) -> Option<(usize, usize)> {
    // SAFETY: queries only.
    unsafe {
        let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        let n = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        if VirtualQuery(address as *const c_void, &mut info, n) == 0 {
            return None;
        }
        let base = info.AllocationBase as usize;
        if base == 0 {
            return None;
        }
        // every region of the allocation, from its base on
        let mut at = base;
        loop {
            if VirtualQuery(at as *const c_void, &mut info, n) == 0 || info.AllocationBase as usize != base {
                break;
            }
            if info.State != MEM_COMMIT || info.Type != MEM_PRIVATE {
                return None;
            }
            at = info.BaseAddress as usize + info.RegionSize;
            if at - base > MAX_HEAP {
                return None;
            }
        }
        (at > base).then_some((base, at - base))
    }
}

/// A buffer over title memory `[start, start + len)` and the host GPU address of `start`.
pub(crate) fn buffer(device: &ID3D12Device, start: u64, len: u64) -> Option<(ID3D12Resource, u64)> {
    if !enabled() || len == 0 {
        return None;
    }
    if let Some(found) = in_section(device, start, len) {
        return Some(found);
    }
    // Private allocations hold the title's CPU heaps too (Grand Theft Auto VI's "Game
    // Virtual" and "Game Physical" heaps): shader writes that went to a copy before landed in
    // its records there and crashed it. Only with `VICTORY_DIRECT_PRIVATE=1`.
    // Smaller ones (its 174 MB GPU heap) are read in place; `VICTORY_DIRECT_PRIVATE_MB` sets the
    // largest (default 512, 0: none).
    static LIMIT: OnceLock<usize> = OnceLock::new();
    let limit = *LIMIT.get_or_init(|| std::env::var("VICTORY_DIRECT_PRIVATE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(512usize) << 20);
    // and small ones hold the title's CPU structures (reading them in place stopped its input)
    static MIN: OnceLock<usize> = OnceLock::new();
    let min = *MIN.get_or_init(|| std::env::var("VICTORY_DIRECT_PRIVATE_MIN_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64usize) << 20);
    let (_, size) = allocation(start as usize)?;
    if size > limit || size < min {
        return None;
    }
    in_allocation(device, start, len)
}

/// A buffer over private memory (a heap opened from its allocation's address).
fn in_allocation(device: &ID3D12Device, start: u64, len: u64) -> Option<(ID3D12Resource, u64)> {
    let (base, size) = allocation(start as usize)?;
    if start + len > (base + size) as u64 {
        return None;
    }
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let heaps = heaps.get_or_insert_with(HashMap::new);
    let entry = heaps.entry(base).or_insert_with(|| {
        let heap = (|| -> windows::core::Result<ID3D12Heap> {
            let device3: ID3D12Device3 = device.cast()?;
            // SAFETY: a committed allocation of ours that lives as long as the title's memory.
            unsafe { device3.OpenExistingHeapFromAddress::<ID3D12Heap>(base as *const c_void) }
        })();
        match &heap {
            Ok(_) => tracing::info!(base = format_args!("{base:#x}"), size, "title memory opened for the GPU in place (no copies)"),
            Err(e) => tracing::info!(base = format_args!("{base:#x}"), size, error = %e, "title memory not opened in place; copied"),
        }
        Heap { heap: heap.ok(), base, size }
    });
    let heap = entry.heap.as_ref()?;
    let offset = (start - entry.base as u64) & !(ALIGN - 1);
    let width = (start + len - entry.base as u64 - offset).next_multiple_of(ALIGN).min(entry.size as u64 - offset);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        // heaps opened over process memory are shared across adapters
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a buffer inside the heap.
    let made = unsafe { device.CreatePlacedResource(heap, offset, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer) };
    if let Err(e) = made {
        tracing::debug!(error = %e, offset, width, "no buffer placed over title memory");
        return None;
    }
    let buffer = buffer?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - entry.base as u64 - offset);
    Some((buffer, gpu))
}

/// The section behind title memory: handle, byte offset of the address, section size.
type Lookup = fn(usize, usize) -> Option<(isize, u64, u64)>;
static LOOKUP: OnceLock<Lookup> = OnceLock::new();

/// Set by the memory module: where physical pages are.
pub fn set_section_lookup(lookup: Lookup) {
    let _ = LOOKUP.set(lookup);
}

struct SectionHeap(Option<ID3D12Heap>);
// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for SectionHeap {}

/// Heaps opened from sections, by handle.
static SECTION_HEAPS: Mutex<Option<HashMap<isize, SectionHeap>>> = Mutex::new(None);

/// A buffer over physical pages (a heap opened from their section).
fn in_section(device: &ID3D12Device, start: u64, len: u64) -> Option<(ID3D12Resource, u64)> {
    let (handle, offset_of_start, size) = (LOOKUP.get()?)(start as usize, len as usize)?;
    let heap = {
        let mut heaps = SECTION_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        let heaps = heaps.get_or_insert_with(HashMap::new);
        heaps
            .entry(handle)
            .or_insert_with(|| {
                let heap = (|| -> windows::core::Result<ID3D12Heap> {
                    let device3: ID3D12Device3 = device.cast()?;
                    // SAFETY: a committed section of the memory module's.
                    unsafe { device3.OpenExistingHeapFromFileMapping::<ID3D12Heap>(windows::Win32::Foundation::HANDLE(handle as *mut c_void)) }
                })();
                match &heap {
                    Ok(_) => tracing::info!(size, "physical pages opened for the GPU in place (no copies)"),
                    Err(e) => tracing::info!(size, error = %e, "physical pages not opened in place; copied"),
                }
                SectionHeap(heap.ok())
            })
            .0
            .clone()?
    };
    let offset = offset_of_start & !(ALIGN - 1);
    let width = (offset_of_start + len - offset).next_multiple_of(ALIGN).min(size - offset);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a buffer inside the heap.
    if let Err(e) = unsafe { device.CreatePlacedResource(&heap, offset, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer) } {
        tracing::debug!(error = %e, offset, width, "no buffer placed over physical pages");
        return None;
    }
    let buffer = buffer?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (offset_of_start - offset);
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        tracing::info!(start = format_args!("{start:#x}"), len, n, "title buffer read in place on the GPU");
    }
    Some((buffer, gpu))
}

/// Title memory mapped anew: buffers placed over it show what was there before. Queued here
/// (the memory module calls this with its lock held) and taken by `shadow::find`.
static REMAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

pub(crate) fn remapped(at: usize, len: usize) {
    if enabled() {
        REMAPPED.lock().unwrap_or_else(|e| e.into_inner()).push((at as u64, (at + len) as u64));
    }
}

pub(crate) fn take_remapped() -> Vec<(u64, u64)> {
    std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()))
}
