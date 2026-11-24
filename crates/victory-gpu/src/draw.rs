//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::{DescriptorBinding, TITLE_MEMORY_VIEWS, UserData};
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::{HostPipeline, RootBinding, RootBuffer};

/// One host dispatch of a split title dispatch: the first group and the
/// group counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub(crate) start: [u32; 3],
    pub(crate) count: [u32; 3],
}

/// The most host dispatches one title dispatch is split into.
const MAX_CHUNKS: usize = 4096;

/// Splits `n` groups into runs of at most the host limit: (first, count).
fn split_dimension(n: u32) -> impl Iterator<Item = (u32, u32)> {
    const LIMIT: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    (0..n.div_ceil(LIMIT)).map(move |i| (i * LIMIT, (n - i * LIMIT).min(LIMIT)))
}

/// The host dispatches covering `x*y*z` groups, none over the host limit per
/// dimension (`None` past [`MAX_CHUNKS`]). An empty dispatch has none.
pub(crate) fn dispatch_chunks(x: u32, y: u32, z: u32) -> Option<Vec<Chunk>> {
    let count = |n: u32| n.div_ceil(D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) as usize;
    if count(x).checked_mul(count(y))?.checked_mul(count(z))? > MAX_CHUNKS {
        return None;
    }
    let mut chunks = Vec::new();
    for (sz, cz) in split_dimension(z) {
        for (sy, cy) in split_dimension(y) {
            for (sx, cx) in split_dimension(x) {
                chunks.push(Chunk {
                    start: [sx, sy, sz],
                    count: [cx, cy, cz],
                });
            }
        }
    }
    Some(chunks)
}

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<Option<u32>>),
    /// A root view's GPU address.
    View(RootBuffer, u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    title_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) â†’ first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

/// Root UAV bindings so far: bounded by the shader's extent, and not (the
/// rest of the title buffer's copy), with the bytes each declared.
static ROOT_UAVS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Root UAVs bound and unbound so far, and the bytes they declared.
pub(crate) fn root_uav_counts() -> [u64; 4] {
    ROOT_UAVS.each_ref().map(|c| c.load(Ordering::Relaxed))
}

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

struct RootView {
    gpu: u64,
    copy: RootCopy,
}

enum RootCopy {
    Shadow(std::sync::Arc<crate::shadow::Shadow>, u64, u64),
    Upload { to: *mut u8, address: u64, len: u64 },
    /// Local build: a null root view (`null_root_view`), nothing to copy.
    None,
}

enum RootArgument {
    Constants(Vec<u32>),
    Buffer(RootBuffer, RootView),
    Table(D3D12_GPU_DESCRIPTOR_HANDLE),
}

/// Validates the resource and the actual GPU address supplied to a setter.
/// Local build: a table entry whose view the host cannot make reads as a null view instead of
/// dropping the whole draw (the slot is null already); `VICTORY_STRICT_TABLES=1` drops it.
fn valid_or_null(index: u32, host: &crate::descriptor::host::HostHeap) -> bool {
    if strict_tables() {
        return false;
    }
    host.make_null(index);
    true
}

/// Local build: `VICTORY_STRICT_ROOTS=1` drops draws and dispatches whose root constants or
/// views the host cannot make (unset by the title, no host copy) instead of binding zeros
/// and a null view, and those reading and writing the title memory mirror together.
fn strict_roots() -> bool {
    static STRICT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STRICT.get_or_init(|| std::env::var_os("VICTORY_STRICT_ROOTS").is_some())
}

/// Local build: a root view at the mirror's unmapped first page (see
/// [`crate::mirror::null_va`]).
fn null_root_view() -> Option<RootView> {
    let gpu = crate::mirror::null_va()?;
    Some(RootView {
        gpu,
        copy: RootCopy::None,
    })
}

/// Why `p`'s draw or dispatch cannot bind, counted by reason (diagnostics); false.
fn refuse(p: &HostPipeline, why: &str) -> bool {
    #[cfg(feature = "diagnostics")]
    crate::pipeline::note_why(format!("{} bind: {why}", p.name));
    #[cfg(not(feature = "diagnostics"))]
    let _ = (p, why);
    false
}

fn strict_tables() -> bool {
    static STRICT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STRICT.get_or_init(|| std::env::var_os("VICTORY_STRICT_TABLES").is_some())
}

pub(crate) fn validate_root_buffer(
    kind: RootBuffer,
    gpu: u64,
    buffer_base: u64,
    desc: D3D12_RESOURCE_DESC,
    len: u64,
) -> Result<(), &'static str> {
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER {
        return Err("the host resource is not a buffer");
    }
    if len == 0
        || gpu == 0
        || buffer_base == 0
        || gpu
            .checked_sub(buffer_base)
            .and_then(|offset| offset.checked_add(len))
            .is_none_or(|end| end > desc.Width)
    {
        return Err("the host GPU address is outside the buffer");
    }
    if kind == RootBuffer::Uav
        && !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        return Err("the host buffer does not support unordered access");
    }
    if kind == RootBuffer::Srv && desc.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) {
        return Err("the host buffer does not support shader reads");
    }
    let alignment = match kind {
        RootBuffer::Cbv => D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
        // Local build: root SRV/UAV addresses of raw and structured buffers need
        // 4-byte alignment (the 16-byte floor dropped Grand Theft Auto VI's clears
        // at title addresses ending in 4 or 8). `VICTORY_ROOT_ALIGN16=1` restores it.
        RootBuffer::Srv | RootBuffer::Uav => {
            if std::env::var_os("VICTORY_ROOT_ALIGN16").is_some() {
                D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT
            } else {
                4
            }
        }
    };
    if !gpu.is_multiple_of(u64::from(alignment)) {
        return Err("the root GPU address is misaligned");
    }
    Ok(())
}

#[cfg(test)]
mod root_view_tests {
    use super::*;

    #[test]
    fn raw_root_uav_rejects_invalid_bindings() {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let base = 0x1000;
        assert!(validate_root_buffer(RootBuffer::Uav, base, base, desc, 4).is_ok());
        for gpu in [0, base - 16, base + 2, base + 256] {
            assert!(validate_root_buffer(RootBuffer::Uav, gpu, base, desc, 4).is_err());
        }
        assert!(validate_root_buffer(RootBuffer::Uav, base + 240, base, desc, 17).is_err());
        let read_only = D3D12_RESOURCE_DESC {
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..desc
        };
        assert!(validate_root_buffer(RootBuffer::Uav, base, base, read_only, 4).is_err());
    }

    #[test]
    fn root_buffer_requires_alignment_extent_and_resource_type() {
        let base = 0x1000;
        let mut desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 512,
            ..Default::default()
        };
        assert!(validate_root_buffer(RootBuffer::Cbv, base + 256, base, desc, 256).is_ok());
        assert!(validate_root_buffer(RootBuffer::Cbv, base + 16, base, desc, 256).is_err());
        assert!(validate_root_buffer(RootBuffer::Cbv, base + 256, base, desc, 257).is_err());
        assert!(validate_root_buffer(RootBuffer::Srv, base, base, desc, 0).is_err());
        desc.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        assert!(validate_root_buffer(RootBuffer::Srv, base, base, desc, 4).is_err());
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        assert!(validate_root_buffer(RootBuffer::Cbv, base, base, desc, 256).is_err());
    }

    #[test]
    fn root_views_require_matching_title_setter_and_stage() {
        let mut state = State::default();
        state.set_root_signature(false, 1);
        state.set_root_signature(true, 2);
        state.set_view(false, 0, RootBuffer::Srv, 0x1000);
        state.set_view(true, 0, RootBuffer::Uav, 0x2000);
        assert_eq!(
            root_address(state.roots_ref(true).first(), RootBuffer::Uav),
            Ok(0x2000)
        );
        assert!(root_address(state.roots_ref(false).first(), RootBuffer::Uav).is_err());
        state.set_root_signature(true, 2);
        assert_eq!(
            root_address(state.roots_ref(true).first(), RootBuffer::Uav),
            Ok(0x2000)
        );
        state.set_root_signature(true, 3);
        assert!(root_address(state.roots_ref(true).first(), RootBuffer::Uav).is_err());
        assert_eq!(
            root_address(state.roots_ref(false).first(), RootBuffer::Srv),
            Ok(0x1000)
        );
    }
}

/// Local build: `VICTORY_ROOT_UAV_BACK=0` keeps what shaders write through root UAVs on the
/// host (not copied back into title memory), to tell whether those writes break the title.
fn root_uav_copy_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_ROOT_UAV_BACK").map_or(true, |v| v.trim() != "0"))
}

/// Local build: `VICTORY_ROOT_UAV_EXTENT=0` declares a root UAV to the end
/// of its copy even when the shader bounds what it reaches.
fn root_uav_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_ROOT_UAV_EXTENT").map_or(true, |v| v.trim() != "0"))
}

fn root_address(root: Option<&Root>, kind: RootBuffer) -> Result<u64, &'static str> {
    match root {
        Some(&Root::View(actual, address)) if actual == kind => Ok(address),
        Some(Root::None) | None => Err("unset title root view"),
        _ => Err("wrong type of title root view"),
    }
}

/// The root view of `address`. A root UAV reaches `extent` bytes when the
/// shader bounds it, otherwise the rest of its copy.
fn root_view(
    list: &HostList,
    address: u64,
    kind: RootBuffer,
    extent: Option<u64>,
) -> Result<RootView, &'static str> {
    if address == 0 {
        return Err("the title GPU address is null");
    }
    let min_len = if kind == RootBuffer::Cbv { 1 } else { 4 };
    let shadow = crate::shadow::find(address, min_len).ok_or("no host copy of the title buffer")?;
    // Root views carry no size. Keep the existing limiting spans for reads.
    let limit = match kind {
        RootBuffer::Cbv => 1 << 16,
        RootBuffer::Srv => 1 << 20,
        RootBuffer::Uav => extent.map_or(u64::MAX, |e| e.max(4)),
    };
    let len = shadow.left(address).min(limit);
    if kind == RootBuffer::Uav {
        let k = if extent.is_some() { 0 } else { 1 };
        ROOT_UAVS[k].fetch_add(1, Ordering::Relaxed);
        ROOT_UAVS[k + 2].fetch_add(len, Ordering::Relaxed);
    }
    let resource = if shadow.is_mirrored() {
        &crate::mirror::get().ok_or("no title memory mirror")?.buffer
    } else {
        &shadow.buffer
    };
    // SAFETY: the shadow or mirror retains the live buffer.
    let (base, desc) = unsafe { (resource.GetGPUVirtualAddress(), resource.GetDesc()) };
    let gpu = shadow.gpu(address);
    let span = if kind == RootBuffer::Cbv {
        len.next_multiple_of(256)
    } else {
        len
    };
    if kind == RootBuffer::Cbv && validate_root_buffer(kind, gpu, base, desc, span).is_err() {
        let (gpu, to) = list
            .scratch(len)
            .ok_or("no aligned constant-buffer upload")?;
        if gpu == 0
            || to.is_null()
            || !gpu.is_multiple_of(u64::from(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT))
        {
            return Err("the constant-buffer upload GPU address is invalid");
        }
        return Ok(RootView {
            gpu,
            copy: RootCopy::Upload { to, address, len },
        });
    }
    validate_root_buffer(kind, gpu, base, desc, span)?;
    Ok(RootView {
        gpu,
        copy: RootCopy::Shadow(shadow, address, len),
    })
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, kind: RootBuffer, address: u64) {
        set(self.roots(compute), slot, Root::View(kind, address));
    }

    pub(crate) fn set_root_signature(&mut self, compute: bool, signature: usize) {
        let current = &mut self.title_roots[usize::from(compute)];
        if *current != signature {
            *current = signature;
            self.roots(compute).clear();
        }
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let Some(end) = (offset as usize).checked_add(values.len()) else {
                return;
            };
            if c.len() < end {
                c.resize(end, None);
            }
            for (dst, &value) in c[offset as usize..end].iter_mut().zip(values) {
                *dst = Some(value);
            }
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().flatten().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    fn table_base(
        &self,
        compute: bool,
        slot: u8,
    ) -> Option<(u64, &'static crate::descriptor::Heap)> {
        let Root::Table(table) = *self.roots_ref(compute).get(usize::from(slot))? else {
            return None;
        };
        let heap = self.heap_of(table)?;
        let step = if heap.samplers() { 16 } else { 32 };
        (table != 0 && (table - heap.start).is_multiple_of(step)).then_some((table, heap))
    }

    fn table_offset(&self, compute: bool, slot: u8) -> Option<u32> {
        let (table, heap) = self.table_base(compute, slot)?;
        let index = heap.index(table)?;
        (index < heap.host()?.count())
            .then(|| u32::try_from(table - heap.start).ok())
            .flatten()
    }

    fn descriptor_entries(
        &self,
        list: &HostList,
        used: &[UsedResource],
        bindings: &[DescriptorBinding],
        compute: bool,
    ) -> Result<std::collections::BTreeMap<u8, Vec<(u32, ResourceType)>>, &'static str> {
        let mut tables = std::collections::BTreeMap::<u8, Vec<(u32, ResourceType)>>::new();
        for u in used.iter().filter(|u| u.root == RootType::Table) {
            let step = if u.kind == ResourceType::Sampler {
                16
            } else {
                32
            };
            let offset = u
                .table_offset_dwords
                .checked_mul(4)
                .ok_or("table offset overflow")?;
            tables
                .entry(u.slot)
                .or_default()
                .push((offset / step * step, u.kind));
        }
        for binding in bindings {
            let (properties, _) = binding.properties;
            let kind = match (properties & 0xff, properties & 0x1000 != 0) {
                (14, _) => ResourceType::Sampler,
                (13, _) => ResourceType::Cbv,
                (_, true) => ResourceType::Uav,
                _ => ResourceType::Srv,
            };
            let sampler = kind == ResourceType::Sampler;
            let (base, heap) = match binding.table {
                Some(slot) => self
                    .table_base(compute, slot)
                    .ok_or("invalid title table base")?,
                None => {
                    let title_heap = self.heaps[usize::from(sampler)];
                    if title_heap == 0 {
                        return Err("missing title heap");
                    }
                    let heap = crate::descriptor::heap(title_heap as This);
                    (heap.start, heap)
                }
            };
            if heap.samplers() != sampler {
                return Err("wrong heap type for descriptor");
            }
            let step = if sampler { 16 } else { 32 };
            let offset = binding.byte_offset & !(step - 1);
            let handle = base
                .checked_add(u64::from(offset))
                .ok_or("descriptor address overflow")?;
            let index = heap
                .index(handle)
                .ok_or("descriptor outside the title heap")?;
            let view = crate::descriptor::view(handle).ok_or("descriptor is unset")?;
            if let Err(why) = crate::descriptor::host::validate_binding(&view, binding.properties) {
                // Local build: the console reads a view whose dimension or format differs from the
                // shader's declaration (2D vs 2D array, typed format classes) and Grand Theft Auto VI
                // relies on it; dropping those draws left the story's picture black. Only a
                // wrong class of view (and missing resources) still drops the draw, unless
                // `VICTORY_STRICT_DESCRIPTORS=1`.
                let lenient = matches!(
                    why,
                    "descriptor shape does not match the shader"
                        | "typed descriptor format does not match the shader"
                        | "structured descriptor stride does not match the shader"
                );
                static STRICT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if !lenient || *STRICT.get_or_init(|| std::env::var_os("VICTORY_STRICT_DESCRIPTORS").is_some()) {
                    return Err(why);
                }
            }
            if let Some(slot) = binding.table {
                let entries = tables.entry(slot).or_default();
                if !entries.contains(&(offset, kind)) {
                    entries.push((offset, kind));
                }
            } else {
                let host = heap.host().ok_or("missing host heap")?;
                if host.mirror(index, handle, true).is_none() {
                    return Err("no host descriptor for direct heap access");
                }
                readable(list, kind, &view);
            }
        }
        Ok(tables)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false, None)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    /// `groups`: a direct mesh dispatch's group counts.
    fn prepare_for(
        &mut self,
        list: &HostList,
        indexed: bool,
        mesh: bool,
        groups: Option<[u32; 3]>,
    ) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return refuse(p, "a draw or mesh dispatch with another kind of pipeline");
        }
        if !self.bind(list, p, false, groups) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return refuse(p, "index buffer outside title buffers");
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    ///
    /// `groups`: the group counts of a direct dispatch (bounding the root
    /// UAVs a shader reaches by its ids), `None` for draws and indirect ones.
    fn bind(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        compute: bool,
        groups: Option<[u32; 3]>,
    ) -> bool {
        if p.compute != compute {
            return refuse(p, "a compute pipeline for a draw, or a graphics one for a dispatch");
        }
        let l = &list.list;
        let title_heaps = self.heaps.map(|h| {
            (h != 0)
                .then(|| crate::descriptor::heap(h as This).host())
                .flatten()
        });
        for (i, (indexed, heap)) in p.indexed_heaps.into_iter().zip(title_heaps).enumerate() {
            if indexed && heap.is_none() {
                once!(pipeline = %p.name, heap = i, "draw or dispatch dropped: missing indexed heap");
                return refuse(p, "missing indexed heap");
            }
        }
        let has_root = |kind| {
            p.stages
                .iter()
                .any(|s| s.roots.iter().any(|&(_, b)| b == kind))
        };
        let mut heaps = title_heaps.map(|h| h.map(|h| h.heap.clone()));
        let mut title_memory = None;
        if has_root(RootBinding::TitleMemory) {
            title_memory = if let Some(heap) = title_heaps[0] {
                heap.title_memory()
                    .filter(|&handle| heap.valid_table(handle, TITLE_MEMORY_VIEWS))
            } else if self.heaps[0] == 0 {
                crate::mirror::read_heap().and_then(|heap| {
                    let host = crate::host::host()?;
                    // SAFETY: the fallback is a live view heap.
                    let (handle, step) = unsafe {
                        (
                            heap.GetGPUDescriptorHandleForHeapStart(),
                            host.device.GetDescriptorHandleIncrementSize(
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            ),
                        )
                    };
                    if !crate::descriptor::host::valid_view_table(
                        heap,
                        handle,
                        step,
                        TITLE_MEMORY_VIEWS,
                    ) {
                        return None;
                    }
                    heaps[0] = Some(heap.clone());
                    Some(handle)
                })
            } else {
                None
            };
            if title_memory.is_none() {
                once!(pipeline = %p.name, "dispatch dropped: no valid title memory SRV table");
                return refuse(p, "no valid title memory SRV table");
            }
        }
        let image_data = if has_root(RootBinding::DescriptorData) {
            let data = title_heaps[0].and_then(|heap| {
                let (buffer, handle) = heap.image_data()?;
                heap.valid_table(handle, 1).then_some((buffer, handle))
            });
            if data.is_none() {
                once!(pipeline = %p.name, "draw or dispatch dropped: no valid image descriptor data table");
                return refuse(p, "no valid image descriptor data table");
            }
            data
        } else {
            None
        };
        for (i, heap) in heaps.iter().enumerate() {
            if let Some(heap) = heap {
                // SAFETY: a live heap.
                let (desc, gpu) =
                    unsafe { (heap.GetDesc(), heap.GetGPUDescriptorHandleForHeapStart()) };
                let kind = [
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                ][i];
                if desc.Type != kind
                    || !desc
                        .Flags
                        .contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
                    || desc.NumDescriptors == 0
                    || gpu.ptr == 0
                {
                    once!(pipeline = %p.name, heap = i, "draw or dispatch dropped: invalid host heap");
                    return refuse(p, "invalid host heap");
                }
            }
        }
        let mut arguments = Vec::new();
        let mut mirror_reads = title_memory.is_some();
        let mut mirror_writes = false;
        for stage in &p.stages {
            let t = stage.translated();
            let made = match self.tables(list, &t.used, &t.host.descriptor_bindings, compute) {
                Ok(made) => made,
                Err(reason) => {
                    once!(pipeline = %p.name, reason, "draw or dispatch dropped: invalid descriptor table");
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why(format!(
                        "{} invalid descriptor table: {reason}",
                        p.name
                    ));
                    return false;
                }
            };
            for &(param, binding) in &stage.roots {
                let arg = match binding {
                    RootBinding::Constants(dwords) => {
                        let mut values = vec![0u32; dwords as usize];
                        for u in &t.host.user_data {
                            let value = match self.user_word(&made, compute, u) {
                                Some(value) => value,
                                // Local build: a root constant the title left unset reads
                                // zero (the console reads whatever its user data holds);
                                // `VICTORY_STRICT_ROOTS=1` drops the draw or dispatch.
                                None if u.root != RootType::Table && !strict_roots() => {
                                    #[cfg(feature = "diagnostics")]
                                    crate::pipeline::note_why(format!(
                                        "{} bind (lenient): unset root constant {}.{} read as zero",
                                        p.name, u.slot, u.dword
                                    ));
                                    0
                                }
                                None => {
                                    once!(pipeline = %p.name, slot = u.slot, ?u.root, u.dword,
                                        "draw or dispatch dropped: missing or invalid root data");
                                    return refuse(
                                        p,
                                        if u.root == RootType::Table {
                                            "unset title table for a user-data dword"
                                        } else {
                                            "unset root constant"
                                        },
                                    );
                                }
                            };
                            values[u.index as usize] = value;
                        }
                        for a in &t.host.buffer_addresses {
                            let value = self.table_base(compute, a.table).and_then(|(base, heap)| {
                                if heap.samplers() {
                                    return None;
                                }
                                let handle = base.checked_add(u64::from(a.byte_offset))?;
                                heap.index(handle)?;
                                let view = crate::descriptor::view(handle)?;
                                crate::descriptor::host::buffer_address_word(&view, a.high)
                            });
                            let Some(value) = value else {
                                once!(pipeline = %p.name, slot = a.table,
                                    "draw or dispatch dropped: invalid buffer address constant");
                                return refuse(p, "invalid buffer address constant");
                            };
                            values[a.index as usize] = value;
                        }
                        RootArgument::Constants(values)
                    }
                    RootBinding::Buffer { slot, kind } => {
                        let user = |dword: u8| {
                            let u = t
                                .host
                                .user_data
                                .iter()
                                .find(|u| u.index == u32::from(dword))?;
                            self.user_word(&made, compute, u)
                        };
                        let extent = (kind == RootBuffer::Uav && root_uav_extent())
                            .then(|| t.host.root_uav_extents.iter().find(|e| e.slot == slot))
                            .flatten()
                            .and_then(|e| e.len(groups, user));
                        let view =
                            root_address(self.roots_ref(compute).get(usize::from(slot)), kind)
                                .and_then(|address| root_view(list, address, kind, extent));
                        let view = match view {
                            Ok(view) => view,
                            // Local build: a root view the host cannot make (unset by the
                            // title, or an address no host copy holds) points at the
                            // mirror's unmapped first page instead of dropping the draw or
                            // dispatch: reads give zeros, writes go nowhere (tiled resources
                            // tier 2). `VICTORY_STRICT_ROOTS=1` drops it.
                            Err(reason) if !strict_roots() && let Some(view) = null_root_view() => {
                                #[cfg(feature = "diagnostics")]
                                crate::pipeline::note_why(format!(
                                    "{} bind (lenient): root {kind:?} {slot} null: {reason}",
                                    p.name
                                ));
                                #[cfg(not(feature = "diagnostics"))]
                                let _ = reason;
                                view
                            }
                            Err(reason) => {
                                once!(pipeline = %p.name, slot, ?kind, reason,
                                    "draw or dispatch dropped: invalid root view");
                                return refuse(p, &format!("invalid root {kind:?} {slot}: {reason}"));
                            }
                        };
                        if matches!(&view.copy, RootCopy::Shadow(shadow, ..) if shadow.is_mirrored())
                        {
                            mirror_writes |= kind == RootBuffer::Uav;
                            mirror_reads |= kind != RootBuffer::Uav;
                        }
                        RootArgument::Buffer(kind, view)
                    }
                    RootBinding::TitleMemory => {
                        let Some(handle) = title_memory else {
                            return refuse(p, "no title memory table");
                        };
                        RootArgument::Table(handle)
                    }
                    RootBinding::DescriptorData => {
                        let Some((_, handle)) = &image_data else {
                            return refuse(p, "no image descriptor data table");
                        };
                        RootArgument::Table(*handle)
                    }
                };
                arguments.push((param, arg));
            }
        }
        // Local build: the title memory table reads the mirror through resources of
        // their own (`Mirror::read_views`), and a buffer has no layout a read and a
        // write in one dispatch could disagree on (as on the console, which binds both):
        // only `VICTORY_STRICT_MIRROR=1` drops reads and writes of the mirror together.
        if mirror_reads && mirror_writes && std::env::var_os("VICTORY_STRICT_MIRROR").is_some() {
            once!(pipeline = %p.name, "draw or dispatch dropped: conflicting root access to the title memory mirror");
            return refuse(p, "conflicting root access to the title memory mirror");
        }
        if let Some((buffer, _)) = image_data {
            if list.first_use(buffer.as_raw() as usize) {
                list.keep(&buffer);
            }
        }
        let bound_heaps = heaps
            .each_ref()
            .map(|h| h.as_ref().map_or(0, |h| h.as_raw() as usize));
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != bound_heaps {
            let heaps: Vec<_> = heaps.into_iter().flatten().map(Some).collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = bound_heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for (param, arg) in arguments {
            match arg {
                RootArgument::Constants(values) => {
                    // SAFETY: preflight checked the parameter's type and size.
                    unsafe {
                        if compute {
                            l.SetComputeRoot32BitConstants(
                                param,
                                values.len() as u32,
                                values.as_ptr().cast(),
                                0,
                            );
                        } else {
                            l.SetGraphicsRoot32BitConstants(
                                param,
                                values.len() as u32,
                                values.as_ptr().cast(),
                                0,
                            );
                        }
                    }
                }
                RootArgument::Table(handle) => {
                    // SAFETY: preflight checked the full range in the bound view heap.
                    unsafe {
                        if compute {
                            l.SetComputeRootDescriptorTable(param, handle);
                        } else {
                            l.SetGraphicsRootDescriptorTable(param, handle);
                        }
                    }
                }
                RootArgument::Buffer(kind, RootView { gpu, copy }) => {
                    match copy {
                        RootCopy::Shadow(shadow, address, len) => {
                            shadow.refresh(list, address, len);
                            if kind == RootBuffer::Uav && root_uav_copy_back() {
                                shadow.gpu_writes(list, address, len);
                            }
                        }
                        RootCopy::Upload { to, address, len } => {
                            list.defer_fill(to, address as *const u8, len as usize);
                        }
                        RootCopy::None => {}
                    }
                    // SAFETY: preflight checked the GPU address and binding kind.
                    unsafe {
                        match (compute, kind) {
                            (false, RootBuffer::Cbv) => {
                                l.SetGraphicsRootConstantBufferView(param, gpu)
                            }
                            (false, RootBuffer::Srv) => {
                                l.SetGraphicsRootShaderResourceView(param, gpu)
                            }
                            (false, RootBuffer::Uav) => {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu)
                            }
                            (true, RootBuffer::Cbv) => {
                                l.SetComputeRootConstantBufferView(param, gpu)
                            }
                            (true, RootBuffer::Srv) => {
                                l.SetComputeRootShaderResourceView(param, gpu)
                            }
                            (true, RootBuffer::Uav) => {
                                l.SetComputeRootUnorderedAccessView(param, gpu)
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        let split = [x, y, z]
            .into_iter()
            .any(|n| n > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
        if split && dispatch_chunks(x, y, z).is_none() {
            once!(x, y, z, "dispatch dropped: too many groups to split");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true, Some([x, y, z])) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.trace_dispatch(x, y, z);
        if split {
            // The host caps a dimension; run the range in pieces, the shader
            // adding each piece's first group to its group id.
            let stage = p.stages.first();
            let param = stage.and_then(|s| {
                s.roots
                    .iter()
                    .find_map(|&(param, b)| matches!(b, RootBinding::Constants(_)).then_some(param))
            });
            let (Some(stage), Some(param)) = (stage, param) else {
                once!(pipeline = %p.name, x, y, z, "dispatch dropped: group count exceeds the host limit");
                DROPPED.fetch_add(1, Ordering::Relaxed);
                return;
            };
            let offsets = &stage.translated().host.group_offsets;
            if offsets.is_empty() {
                once!(pipeline = %p.name, x, y, z,
                    "dispatch dropped: group count exceeds the host limit and the shader reads no group id");
                DROPPED.fetch_add(1, Ordering::Relaxed);
                return;
            }
            let set = |start: [u32; 3]| {
                for o in offsets {
                    // SAFETY: an open list with the constants bound; the
                    // offset dwords lie inside them (checked at pipeline
                    // creation).
                    unsafe {
                        list.list.SetComputeRoot32BitConstant(
                            param,
                            start[usize::from(o.component)],
                            o.index,
                        )
                    };
                }
            };
            for chunk in dispatch_chunks(x, y, z).into_iter().flatten() {
                set(chunk.start);
                // SAFETY: an open list with everything set.
                unsafe { list.list.Dispatch(chunk.count[0], chunk.count[1], chunk.count[2]) };
            }
            set([0; 3]);
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(param, binding) in &stage.roots {
                match binding {
                    RootBinding::Buffer { slot, kind } => {
                        if let Ok(address) =
                            root_address(self.roots_ref(compute).get(usize::from(slot)), kind)
                        {
                            let host = crate::shadow::find(address, 1)
                                .map(|s| (s.gpu(address), s.left(address)));
                            let _ =
                                write!(text, " root{slot} {kind:?}={address:#x} host {host:#x?}");
                        }
                    }
                    RootBinding::DescriptorData | RootBinding::TitleMemory => {
                        let _ = write!(text, " root{param}={binding:?}");
                    }
                    RootBinding::Constants(_) => {}
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true, Some([x, 1, 1]))
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    /// The value of user-data dword `u` for this bind: a table's offset
    /// (`made` holds the tables just made) or a root constant.
    fn user_word(&self, made: &[(u8, u32)], compute: bool, u: &UserData) -> Option<u32> {
        let fresh = made
            .iter()
            .find(|m| m.0 == u.slot && u.root == RootType::Table);
        match fresh {
            Some(&(_, offset)) => Some(offset),
            None if u.root == RootType::Table => self.table_offset(compute, u.slot),
            None => match self.roots_ref(compute).get(usize::from(u.slot)) {
                Some(Root::Constants(words)) => words.get(u.dword as usize).copied().flatten(),
                _ => None,
            },
        }
    }

    /// Prepares the entries indexed by the shader. Fresh tables keep their
    /// title layout and copy constant buffers into aligned list uploads.
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        bindings: &[DescriptorBinding],
        compute: bool,
    ) -> Result<Vec<(u8, u32)>, &'static str> {
        let tables = self.descriptor_entries(list, used, bindings, compute)?;
        let mut made = Vec::new();
        for (slot, entries) in tables {
            let (table, heap) = self
                .table_base(compute, slot)
                .ok_or("invalid title table base")?;
            let host = heap.host().ok_or("missing host heap")?;
            let step = if heap.samplers() { 16 } else { 32 };
            let mut fresh = false;
            let mut required = std::collections::BTreeMap::new();
            for (offset, kind) in entries {
                if heap.samplers() != (kind == ResourceType::Sampler) {
                    return Err("wrong heap type for table");
                }
                let handle = table
                    .checked_add(u64::from(offset))
                    .ok_or("table address overflow")?;
                let index = heap
                    .index(handle)
                    .ok_or("table entry outside the title heap")?;
                let view =
                    crate::descriptor::view(handle).ok_or("required table entry is unset")?;
                if crate::descriptor::host::view_kind(&view) != kind {
                    return Err("wrong descriptor class in title table");
                }
                fresh |= kind == ResourceType::Cbv || index >= host.count();
                required.insert(offset / step, (index, kind, handle, view));
            }
            if !fresh {
                for (_, (index, kind, handle, view)) in required {
                    if host.mirror(index, handle, true).is_none() && strict_tables() {
                        return Err("no valid host descriptor for table entry");
                    }
                    readable(list, kind, &view);
                }
                continue;
            }
            let n = required.last_key_value().map_or(0, |(k, _)| k + 1);
            if heap.samplers() || n > host.dynamic_capacity() {
                return Err("no host range for remapped table");
            }
            // The last required entry was checked against the title heap,
            // so all intervening slots are in the same live heap.
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * u64::from(step);
                    // SAFETY: a descriptor slot in the validated title range.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            let first = if let Some(&first) = self.fresh_tables.get(&key) {
                first
            } else {
                let first = host
                    .dynamic(n)
                    .ok_or("no free host range for remapped table")?;
                let prepared = (|| {
                    for k in 0..n {
                        let index = first + k;
                        let view = crate::descriptor::view(table + u64::from(k) * u64::from(step));
                        let valid = match view {
                            Some(View::Constant(address, size)) => {
                                if address == 0
                                    || crate::descriptor::host::cbv_size(size).is_none()
                                    || !crate::texture::committed(
                                        address as *const u8,
                                        size as usize,
                                    )
                                {
                                    return Err("invalid constant buffer in remapped table");
                                }
                                let (gpu, to) =
                                    list.scratch(u64::from(size)).ok_or("no constant upload")?;
                                if to.is_null() || !host.constant_at(index, gpu, size) {
                                    return Err("invalid constant-buffer upload in remapped table");
                                }
                                list.defer_fill(to, address as *const u8, size as usize);
                                true
                            }
                            Some(view) => host.make_at(index, &view),
                            None => {
                                host.make_null(index);
                                false
                            }
                        };
                        if !valid && required.contains_key(&k) {
                            if !valid_or_null(index, host) {
                                return Err("no valid host descriptor for remapped table entry");
                            }
                        }
                    }
                    Ok(())
                })();
                if let Err(reason) = prepared {
                    host.release(first, n);
                    return Err(reason);
                }
                self.fresh_tables.insert(key, first);
                self.dynamic
                    .push((std::ptr::from_ref(heap) as usize, first, n));
                first
            };
            for (_, (_, kind, _, view)) in required {
                readable(list, kind, &view);
            }
            made.push((
                slot,
                first
                    .checked_mul(step)
                    .ok_or("remapped table offset overflow")?,
            ));
        }
        Ok(made)
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    /// Local build: a dispatch of a listed frame, with what it reads and writes (diagnostics).
    #[cfg(feature = "diagnostics")]
    fn trace_dispatch(&self, x: u32, y: u32, z: u32) {
        if !tracing::enabled!(tracing::Level::DEBUG) || !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let name = crate::pipeline::host_pipeline(self.pipeline as This).map(|p| p.name.clone()).unwrap_or_default();
        tracing::debug!("dispatch {x}x{y}x{z} pipeline {:#x} [{name}] reads {:?}", self.pipeline, self.reads(true));
    }

    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This).map(|h| (t, h)))
            .map(|(t, h)| {
                // SAFETY: a live resource.
                let d = unsafe { h.GetDesc() };
                format!("{:#x} {}x{} f{}", t as usize, d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own.
        if !crate::indirect::bound(list, &args.buffer, args.gpu(arguments), (kind, stride, max)) {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED if self.topology == RECTANGLES => {
                why("rectangles");
                false
            }
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true, None),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true, None),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &args.buffer);
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                // The reason is counted as "<pipeline> bind: ...".
                Some(p) => why(&format!("prepare failed (compute {}, mesh {}) {}", p.compute, p.mesh, p.name)),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &args.buffer,
                args.offset(arguments),
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        crate::indirect::done(list, &args.buffer);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{:#x} {}x{} f{} flags {:#x}", r as usize, d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod split_tests {
    use super::*;

    const L: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

    #[test]
    fn a_dimension_splits_into_runs_of_the_limit() {
        assert_eq!(split_dimension(0).count(), 0);
        assert_eq!(split_dimension(1).collect::<Vec<_>>(), [(0, 1)]);
        assert_eq!(split_dimension(L).collect::<Vec<_>>(), [(0, L)]);
        assert_eq!(
            split_dimension(82606).collect::<Vec<_>>(),
            [(0, 65535), (65535, 82606 - 65535)]
        );
        assert_eq!(
            split_dimension(2 * L + 1).collect::<Vec<_>>(),
            [(0, L), (L, L), (2 * L, 1)]
        );
    }

    #[test]
    fn the_logged_dispatch_is_two_chunks() {
        let chunks = dispatch_chunks(82606, 1, 1).unwrap();
        assert_eq!(
            chunks,
            [
                Chunk { start: [0, 0, 0], count: [65535, 1, 1] },
                Chunk { start: [65535, 0, 0], count: [17071, 1, 1] },
            ]
        );
    }

    #[test]
    fn chunks_cover_every_group_exactly_once() {
        for (x, y, z) in [(70000, 3, 2), (1, 140000, 1), (L + 1, L + 1, 2), (5, 6, 7)] {
            let chunks = dispatch_chunks(x, y, z).unwrap();
            let total: u64 = chunks
                .iter()
                .map(|c| c.count.iter().map(|&n| u64::from(n)).product::<u64>())
                .sum();
            assert_eq!(total, u64::from(x) * u64::from(y) * u64::from(z));
            for c in &chunks {
                assert!(c.count.iter().all(|&n| (1..=L).contains(&n)));
                for d in 0..3 {
                    assert!(c.start[d] + c.count[d] <= [x, y, z][d]);
                }
            }
        }
    }

    #[test]
    fn an_unsplit_dispatch_is_one_chunk_and_empty_is_none() {
        assert_eq!(dispatch_chunks(3, 4, 5).unwrap().len(), 1);
        assert!(dispatch_chunks(0, 4, 5).unwrap().is_empty());
    }

    #[test]
    fn absurd_counts_are_refused() {
        assert!(dispatch_chunks(u32::MAX, u32::MAX, 1).is_none());
        assert!(dispatch_chunks(u32::MAX, 1, 1).is_none());
        assert_eq!(dispatch_chunks(300_000, 1, 1).unwrap().len(), 5);
    }
}
