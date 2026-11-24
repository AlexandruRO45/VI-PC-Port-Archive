//! Local build: one GPU address space mirroring title memory, as the console's GPU sees it.
//!
//! Host copies of title buffers used to be separate host buffers, each as long as the title
//! buffer (or a 4 MiB window of loose memory). Grand Theft Auto VI's shaders index far past the
//! address they start from (10-27 MB past a 4 MiB window): a GPU page fault, the device removed.
//!
//! Here one reserved buffer of `BYTES` stands for title memory: title address `a` is mirror
//! byte `a`. Each 64 KiB title page a copy covers gets one page of CPU-visible system memory (a
//! tile of a custom heap), mapped at its place in the mirror and shared by every copy covering
//! that title page. Reads past a copy's end find the neighbouring pages, or zeros where no page
//! is mapped (tiled resources tier 2 and up): never a fault. (Measured on an RX 6800 XT: tiles
//! of a heap made with `CreateHeap` work; tiles of heaps opened from process memory hang it.)
//!
//! `VICTORY_MIRROR=0` turns it off (separate host buffers as before).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
use victory_shader::emit::{TITLE_MEMORY_VIEW_SHIFT, TITLE_MEMORY_VIEWS};

/// Mirrored title addresses: below 128 GiB (Grand Theft Auto VI's reach about 65 GiB).
const BYTES: u64 = 128 << 30;
pub(crate) const TILE: u64 = 1 << 16;
/// Tiles per heap of pages (256 MiB).
const CHUNK_TILES: u32 = 4096;

struct Chunk {
    heap: ID3D12Heap,
    /// A buffer over the whole heap, mapped: the CPU's view of its pages.
    _view: ID3D12Resource,
    cpu: usize,
    used: u32,
}

struct Pages {
    /// title page index → CPU address of its page, its chunk and tile there
    map: HashMap<u64, (usize, usize, u32)>,
    chunks: Vec<Chunk>,
}

pub(crate) struct Mirror {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) va: u64,
    pages: Mutex<Pages>,
}

// SAFETY: D3D12 objects are free-threaded; the page table is behind its lock.
unsafe impl Send for Mirror {}
// SAFETY: as above.
unsafe impl Sync for Mirror {}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MIRROR").map_or(true, |v| v.trim() != "0"))
}

fn buffer_desc(width: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// The mirror, made on first use; `None` if it is off or the host refuses it.
pub(crate) fn get() -> Option<&'static Mirror> {
    static MIRROR: OnceLock<Option<Mirror>> = OnceLock::new();
    MIRROR
        .get_or_init(|| {
            if !enabled() {
                return None;
            }
            let host = crate::host::host()?;
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a reserved buffer.
            let made = unsafe {
                host.device.CreateReservedResource(&buffer_desc(BYTES), D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
            };
            match (made, buffer) {
                (Ok(()), Some(buffer)) => {
                    // SAFETY: a live buffer.
                    let va = unsafe { buffer.GetGPUVirtualAddress() };
                    crate::host::name(&buffer, || "title memory mirror".into());
                    tracing::info!(gib = BYTES >> 30, va = format_args!("{va:#x}"), "title memory mirrored on the GPU");
                    Some(Mirror { buffer, va, pages: Mutex::new(Pages { map: HashMap::new(), chunks: Vec::new() }) })
                }
                (e, _) => {
                    tracing::warn!(error = ?e.err(), "no title memory mirror: separate copies");
                    None
                }
            }
        })
        .as_ref()
}

impl Mirror {
    /// Puts the bounded raw SRVs in consecutive slots of a live view heap.
    pub(crate) fn read_views(
        &self,
        first: D3D12_CPU_DESCRIPTOR_HANDLE,
        step: u32,
    ) -> Option<()> {
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        let host = crate::host::host()?;
        if !*SUPPORTED.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            // SAFETY: the query writes this feature's structure.
            let supported = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            };
            supported.is_ok()
                && options.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_2.0
                && options.ResourceBindingTier.0 >= D3D12_RESOURCE_BINDING_TIER_2.0
        }) {
            return None;
        }
        let elements = 1 << (TITLE_MEMORY_VIEW_SHIFT - 2);
        let size = u64::from(elements) * 4 * u64::from(TITLE_MEMORY_VIEWS);
        if size != BYTES
            || first.ptr == 0
            || step == 0
            || !crate::descriptor::host::valid_raw_srv(&self.buffer, size)
        {
            return None;
        }
        for index in 0..TITLE_MEMORY_VIEWS {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: u64::from(index) * u64::from(elements),
                        NumElements: elements,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            let slot = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: first.ptr + (index * step) as usize,
            };
            // SAFETY: the caller reserved this range in a live view heap.
            unsafe { host.device.CreateShaderResourceView(&self.buffer, Some(&desc), slot) };
        }
        Some(())
    }

    /// Whether `[start, start + len)` lies in the mirror.
    pub(crate) fn covers(&self, start: u64, len: u64) -> bool {
        start.checked_add(len).is_some_and(|end| end <= BYTES)
    }

    /// The pages of title pages `first..first + count` (CPU address, chunk, tile), made and
    /// mapped in the mirror where they are new (zeros).
    pub(crate) fn pages(&self, first: u64, count: u64) -> Option<Vec<(usize, usize, u32)>> {
        let host = crate::host::host()?;
        let mut p = self.pages.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(count as usize);
        // new mappings by chunk: (mirror tiles, heap tiles)
        let mut new: HashMap<usize, (Vec<u32>, Vec<u32>)> = HashMap::new();
        for page in first..first + count {
            if let Some(&found) = p.map.get(&page) {
                out.push(found);
                continue;
            }
            if p.chunks.last().is_none_or(|c| c.used == CHUNK_TILES) {
                p.chunks.push(new_chunk(&host.device)?);
                let total = p.chunks.len() as u64 * u64::from(CHUNK_TILES) * TILE;
                tracing::info!(mib = total >> 20, "title memory mirror pages");
            }
            let k = p.chunks.len() - 1;
            let c = &mut p.chunks[k];
            let tile = c.used;
            c.used += 1;
            let cpu = c.cpu + tile as usize * TILE as usize;
            p.map.insert(page, (cpu, k, tile));
            let e = new.entry(k).or_default();
            e.0.push(page as u32);
            e.1.push(tile);
            out.push((cpu, k, tile));
        }
        for (k, (tiles, heap_tiles)) in new {
            host.map_tiles(&self.buffer, &tiles, &p.chunks[k].heap, &heap_tiles);
        }
        Some(out)
    }
}

/// A heap for shaders that read the mirror without a title view heap.
pub(crate) fn read_heap() -> Option<&'static ID3D12DescriptorHeap> {
    static HEAP: OnceLock<Option<ID3D12DescriptorHeap>> = OnceLock::new();
    HEAP.get_or_init(|| {
        let mirror = get()?;
        let host = crate::host::host()?;
        // SAFETY: plain creation of a shader-visible heap.
        unsafe {
            let heap: ID3D12DescriptorHeap = host.device.CreateDescriptorHeap(
                &D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: TITLE_MEMORY_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                },
            ).ok()?;
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            mirror.read_views(heap.GetCPUDescriptorHandleForHeapStart(), step)?;
            Some(heap)
        }
    }).as_ref()
}

impl Mirror {
    /// A reserved buffer of its own over `pages` (in order: its tile k is pages[k]), for what
    /// names a resource (views, barriers, indirect arguments, copies): one resource for all
    /// of title memory would be in one state for all of it.
    pub(crate) fn alias(&self, pages: &[(usize, usize, u32)]) -> Option<ID3D12Resource> {
        let host = crate::host::host()?;
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a reserved buffer.
        unsafe {
            host.device
                .CreateReservedResource(&buffer_desc(pages.len() as u64 * TILE), D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .ok()?;
        }
        let buffer = buffer?;
        let p = self.pages.lock().unwrap_or_else(|e| e.into_inner());
        let mut by_chunk: HashMap<usize, (Vec<u32>, Vec<u32>)> = HashMap::new();
        for (k, &(_, chunk, tile)) in pages.iter().enumerate() {
            let e = by_chunk.entry(chunk).or_default();
            e.0.push(k as u32);
            e.1.push(tile);
        }
        for (chunk, (tiles, heap_tiles)) in by_chunk {
            host.map_tiles(&buffer, &tiles, &p.chunks[chunk].heap, &heap_tiles);
        }
        Some(buffer)
    }
}

fn new_chunk(device: &ID3D12Device) -> Option<Chunk> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: u64::from(CHUNK_TILES) * TILE,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };
    let mut heap: Option<ID3D12Heap> = None;
    let mut view: Option<ID3D12Resource> = None;
    let mut cpu = std::ptr::null_mut();
    // SAFETY: plain creation; the view is mapped for good (CPU-visible heap).
    unsafe {
        if let Err(e) = device.CreateHeap(&desc, &mut heap) {
            tracing::warn!(error = %e, "no heap for title memory mirror pages");
            return None;
        }
        let heap = heap?;
        device
            .CreatePlacedResource(&heap, 0, &buffer_desc(desc.SizeInBytes), D3D12_RESOURCE_STATE_COMMON, None, &mut view)
            .ok()?;
        let view = view?;
        view.Map(0, None, Some(&mut cpu)).ok()?;
        Some(Chunk { heap, _view: view, cpu: cpu as usize, used: 0 })
    }
}
