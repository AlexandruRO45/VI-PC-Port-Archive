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

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::drop_stats::{Outcome, Reason};
use crate::host::HostList;
use crate::object::This;
use crate::pipeline::{host::HostPipeline, require_host_pipeline};

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
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
    #[cfg(feature = "diagnostics")]
    hdr_targets: Vec<hdr_trace::Target>,
    #[cfg(feature = "diagnostics")]
    hdr_bindings: Vec<hdr_trace::Binding>,
    #[cfg(feature = "diagnostics")]
    hdr_program: Option<hdr_trace::Program>,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Fresh tables: (title heap, table address, width, record hash) →
    /// first dynamic slot.
    fresh_tables: std::collections::HashMap<(usize, u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// The title-memory page table made for this list (`crate::memory`)
    /// and its view heap.
    memory: Option<(usize, crate::memory::Map)>,
    /// Views made for root SRVs and UAVs in this list ([`State::root_view`]):
    /// (view heap, host copy, unordered access) → dynamic slot.
    root_slots: std::collections::HashMap<(usize, usize, bool), u32>,
    /// The run of dynamic slots root views take from (view heap, next
    /// slot, end): taken [`ROOT_VIEW_RUN`] at a time, so that single slots
    /// do not break up the runs fresh tables need.
    root_run: (usize, u32, u32),
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            match (heap, crate::memory::heap()) {
                (MEMORY_HEAP, Some(ours)) => ours.release(first, n),
                _ => crate::descriptor::release_dynamic(heap, first, n),
            }
        }
    }
}

/// The key of our own view heap (`crate::memory::heap`) among the title's.
const MEMORY_HEAP: usize = crate::memory::OURS;

/// Dynamic slots taken at a time for root views.
const ROOT_VIEW_RUN: u32 = 256;

/// Dynamic slots taken for root views ([`State::root_view`]) since the last
/// call.
pub(crate) fn take_root_view_slots() -> u64 {
    ROOT_VIEW_SLOTS.swap(0, Ordering::Relaxed)
}
static ROOT_VIEW_SLOTS: AtomicU64 = AtomicU64::new(0);

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

#[cfg(feature = "diagnostics")]
impl State {
    pub(crate) fn note_dropped(&self, list: &HostList, kind: &str, why: Reason, direct: bool) {
        #[cfg(feature = "diagnostics")]
        hdr_trace::dropped(&crate::pipeline::shaders_of(self.pipeline as This), &format!("{kind} {why:?}"));
        list.note_op(|| format!("DROPPED {kind} ({why:?}) {}", crate::pipeline::shaders_of(self.pipeline as This)));
        if direct && let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
            last.push_str(" DROPPED");
        }
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. What shaders write through a root UAV goes back
/// to title memory after the submission.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool, uav: bool) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv || uav { 1 << 20 } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || uav || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        if uav {
            shadow.gpu_writes(list, address, len);
        }
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
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

    /// Others recorded into the host list (`crate::external`): its heaps,
    /// root signatures and pipeline are set again before the next draw or
    /// dispatch.
    pub(crate) fn host_state_lost(&mut self) {
        // No title heaps (0) differ from what the host list has, too.
        (self.host_pipeline, self.host_roots, self.host_heaps) = (0, [0; 2], [usize::MAX; 2]);
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

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
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
        #[cfg(feature = "diagnostics")]
        if hdr_trace::enabled() { self.hdr_targets.clear(); }
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
            crate::capture::note_target(list, record.resource);
            let desc = record.desc();
            rtvs.push(host.rtv(texture, desc.as_ref()));
            #[cfg(feature = "diagnostics")]
            if hdr_trace::enabled() {
                self.hdr_targets.push(hdr_trace::Target::rtv(record.resource, texture, desc.as_ref()));
            }
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
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
    /// it: a table's byte offset in its heap, a constant's value, a root
    /// view's address (title memory).
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::View(address)), RootType::Descriptor) if dword < 2 => {
                (address >> (32 * dword)) as u32
            }
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
                c.get(dword as usize).copied().unwrap_or(0)
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

    /// The host addresses of the bound view heap's title-memory page table
    /// and the list's page bitmap for it, set up on first use (the table
    /// names every host copy at each submission).
    fn memory_map(&mut self, list: &HostList) -> Option<(u64, u64)> {
        let heap = match self.heaps[0] {
            0 => MEMORY_HEAP,
            h => h,
        };
        if let Some((h, map)) = &self.memory
            && *h == heap
        {
            return Some((map.table, map.dirty));
        }
        let map = crate::memory::bind(list, heap)?;
        let out = (map.table, map.dirty);
        self.memory = Some((heap, map));
        Some(out)
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
    fn prepare(&mut self, list: &HostList, indexed: bool) -> Outcome {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> Outcome {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return Err(Reason::ShaderSkipped);
        }
        let p = require_host_pipeline(self.pipeline as This)?;
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return Err(Reason::PipelineKind);
        }
        self.bind(list, p, false)?;
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
            let shadow = crate::shadow::find(v.address, u64::from(v.size))
                .ok_or(Reason::Bind("index buffer unavailable"))?;
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
        // A depth copy of ours (`crate::twin`) ran a compute pipeline.
        if list.take_rebind() {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            (self.host_pipeline, self.host_roots[1]) = (p.pso.as_raw() as usize, 0);
        }
        Ok(())
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// Returns the binding failure that prevents this draw or dispatch.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> Outcome {
        self.bind_as(list, p, compute, compute)
    }

    /// The fallback CS binds compute registers from the title's graphics
    /// arguments, sharing all heap, page-table and resource preparation.
    fn bind_as(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        compute: bool,
        title_compute: bool,
    ) -> Outcome {
        let l = &list.list;
        // A depth copy of ours (`crate::twin`) since the last one ran a
        // compute pipeline.
        if list.take_rebind() {
            (self.host_pipeline, self.host_roots[1]) = (0, 0);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them. Title memory's views need a view heap: ours where
        // the title binds none (the BVH builder's dispatches may not).
        let memory_heap = self.heaps[0] == 0
            && p.stages.iter().any(|s| s.memory.is_some() || has_root_views(&s.translated().host));
        self.bind_heaps(list, memory_heap);
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
        #[cfg(feature = "diagnostics")]
        crate::capture::binding(&p.name);
        // Shaders taking views from the heap itself read (or write) views no
        // table names: everything the bound heap's views reach is brought up
        // to date at submission (`crate::bindless`).
        if p.stages.iter().any(|s| s.translated().host.heap) {
            let writes = p.stages.iter().any(|s| s.translated().host.heap_writes);
            if let Some(host) = (self.heaps[0] != 0)
                .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                .flatten()
            {
                list.defer_heap(&host.viewed);
            }
            crate::bindless::note_draw(writes);
        }
        #[cfg(feature = "diagnostics")]
        if hdr_trace::enabled() {
            self.hdr_bindings.clear();
            self.hdr_program = None;
        }
        for stage in &p.stages {
            let t = stage.translated();
            let made = self.tables(list, &t.used, title_compute);
            #[cfg(feature = "diagnostics")]
            if hdr_trace::enabled()
                && let Some(program) = hdr_trace::Program::from_hash(&stage.shader.hash)
                && let Some(made) = &made
            {
                self.hdr_program = Some(program);
                self.hdr_views(program, &t.used, title_compute, made);
            }
            // A depth copy of ours made readable a texture the tables hold
            // (`crate::twin`), with a compute pipeline: this one's again
            // (root arguments are set below).
            if list.take_rebind() {
                // SAFETY: an open list, a live root signature and pipeline.
                unsafe {
                    if compute {
                        l.SetComputeRootSignature(&p.root);
                    }
                    l.SetPipelineState(&p.pso);
                }
                self.host_pipeline = p.pso.as_raw() as usize;
                self.host_roots[1] = if compute { root } else { 0 };
            }
            let Some(made) = made else {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: a required descriptor table could not be made"
                    );
                }
                return Err(Reason::Bind("required descriptor table could not be made"));
            };
            // Root views a shader reads as title memory: the copies holding
            // their addresses go into the page table.
            let mut roots = Vec::new();
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if u.root == RootType::Descriptor
                        && u.dword == 0
                        && let Some(Root::View(address)) =
                            self.roots_ref(title_compute).get(usize::from(u.slot))
                        && let Some(shadow) = crate::shadow::find(*address, 1)
                    {
                        let uav = t.used.iter().any(|r| {
                            r.slot == u.slot
                                && r.root == RootType::Descriptor
                                && r.kind == ResourceType::Uav
                        });
                        roots.push((*address, uav.then_some(shadow)));
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None if u.dword == victory_shader::emit::ROOT_VIEW_OFFSET => {
                            // Where the root address lies in the copy its view shows.
                            let offset =
                                match self.roots_ref(title_compute).get(usize::from(u.slot)) {
                                    Some(&Root::View(address)) => crate::shadow::find(address, 1)
                                        .and_then(|s| u32::try_from(s.offset(address)).ok()),
                                    _ => None,
                                };
                            offset.ok_or(Reason::RootView("offset unavailable"))?
                        }
                        None if is_root_view(u) => {
                            let uav = u.dword == victory_shader::emit::ROOT_UAV_VIEW;
                            self.root_view(list, title_compute, u.slot, uav)?.0
                        }
                        None => self.user_value(title_compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, kind) in [(&stage.root_cbvs, 0), (&stage.root_srvs, 1), (&stage.root_uavs, 2)] {
                for &(slot, param) in views {
                    if kind == 1 && slot == victory_shader::emit::VIEW_STRIDES {
                        // The bound view heap's stride table; without one
                        // the shader would read an unbound root view.
                        let heap = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                            .flatten();
                        let Some(table) = heap.and_then(|h| h.strides()) else {
                            once!("a shader reads view strides without a view heap: dropped");
                            return Err(Reason::Bind("view strides unavailable"));
                        };
                        list.keep(table);
                        // SAFETY: an open list; the table lives while the list
                        // holds it.
                        unsafe {
                            let gpu = table.GetGPUVirtualAddress();
                            if compute {
                                l.SetComputeRootShaderResourceView(param, gpu);
                            } else {
                                l.SetGraphicsRootShaderResourceView(param, gpu);
                            }
                        }
                        continue;
                    }
                    let Some(Root::View(address)) =
                        self.roots_ref(title_compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        continue;
                    };
                    let Some(gpu) = root_view(list, *address, kind == 1, kind == 2) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, kind) {
                            (false, 0) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, 1) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (false, _) => l.SetGraphicsRootUnorderedAccessView(param, gpu),
                            (true, 0) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, 1) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, _) => l.SetComputeRootUnorderedAccessView(param, gpu),
                        }
                    }
                }
            }
            if let Some(param) = stage.view_addresses {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.view_addresses());
                let Some(gpu) = gpu else {
                    once!("a shader reading a view's address dropped: no view heap");
                    return Err(Reason::Bind("view addresses unavailable"));
                };
                // SAFETY: an open list; the heap's buffer lives with the heap.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, gpu);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
            if let Some(param) = stage.memory {
                let Some((table, dirty)) = self.memory_map(list) else {
                    once!("a shader reading title memory dropped: no view heap or page table");
                    return Err(Reason::Bind("view heap or title memory page table unavailable"));
                };
                crate::memory::note_use(&p.name);
                // What shaders write through a root UAV goes back by its
                // bytes, as through root views ([`root_view`]).
                for (address, shadow) in &roots {
                    if let (Some(shadow), Some((_, map))) = (shadow, &self.memory) {
                        let len = shadow.left(*address).min(1 << 20);
                        shadow.refresh(list, *address, len);
                        shadow.gpu_writes(list, *address, len);
                        map.precise(*address, len);
                    }
                }
                // SAFETY: an open list; list memory alive until it is reset.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, table);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, table);
                    }
                    if let Some(param) = stage.memory_dirty {
                        if compute {
                            l.SetComputeRootUnorderedAccessView(param, dirty);
                        } else {
                            l.SetGraphicsRootUnorderedAccessView(param, dirty);
                        }
                    }
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding("");
        Ok(())
    }

    /// The heap index of a raw view (`uav`: unordered access) of the host
    /// copy holding the root view the title set at root parameter `slot`,
    /// and the root address's byte offset in it: a dynamic slot of the view
    /// heap the shader indexes, made once per list and copy; the range from
    /// the root address brought up to date at submission
    /// (`victory_shader::emit`, "Root SRVs and root UAVs").
    /// The copy is brought up to date from the root address to its end: a
    /// shader reaches wherever data its root view leads to lie (a vertex
    /// fetch's streams sit at offsets in the root view's own header).
    /// shortcut: what shaders write through a root UAV goes back to title
    /// memory for 1 MiB from its address, as before the views.
    fn root_view(&mut self, list: &HostList, compute: bool, slot: u8, uav: bool) -> Result<(u32, u32), Reason> {
        let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
            once!(slot, "a shader reads a root view the title did not set: dropped");
            return Err(Reason::RootView("address not set"));
        };
        let shadow = crate::shadow::find(address, 1).ok_or(Reason::RootView("buffer unavailable"))?;
        let offset = u32::try_from(shadow.offset(address)).map_err(|_| Reason::RootView("offset exceeds u32"))?;
        shadow.refresh(list, address, shadow.left(address));
        if uav {
            shadow.gpu_writes(list, address, shadow.left(address).min(1 << 20));
        }
        let (key, host) = match self.heaps[0] {
            0 => (MEMORY_HEAP, crate::memory::heap()),
            h => (
                std::ptr::from_ref(crate::descriptor::heap(h as This)) as usize,
                crate::descriptor::heap(h as This).host(),
            ),
        };
        let host = host.ok_or(Reason::RootView("view heap unavailable"))?;
        let copy = std::sync::Arc::as_ptr(&shadow) as usize;
        if let Some(&index) = self.root_slots.get(&(key, copy, uav)) {
            return Ok((index, offset));
        }
        let (run_heap, next, end) = self.root_run;
        if run_heap != key || next >= end {
            let Some(first) = host.dynamic(ROOT_VIEW_RUN) else {
                once!("no dynamic slots for root views: dropped");
                return Err(Reason::RootView("no dynamic view slots"));
            };
            self.dynamic.push((key, first, ROOT_VIEW_RUN));
            self.root_run = (key, first, first + ROOT_VIEW_RUN);
        }
        let index = self.root_run.1;
        self.root_run.1 += 1;
        ROOT_VIEW_SLOTS.fetch_add(1, Ordering::Relaxed);
        host.raw_view_at(index, Some(&shadow.buffer), 0, shadow.range().1 & !3, uav);
        self.root_slots.insert((key, copy, uav), index);
        Ok((index, offset))
    }

    /// Gives the host list the title's heaps if it has others.
    fn bind_heaps(&mut self, list: &HostList, memory_heap: bool) {
        let want = if memory_heap {
            [MEMORY_HEAP, self.heaps[1]]
        } else {
            self.heaps
        };
        if self.host_heaps != want {
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            if memory_heap && let Some(h) = crate::memory::heap() {
                heaps.insert(0, Some(h.heap.clone()));
            }
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.list.SetDescriptorHeaps(&heaps) };
            self.host_heaps = want;
            self.host_roots = [0; 2];
        }
    }

    /// `ClearUnorderedAccessViewUint` or `...Float` (`float`): the view the
    /// title made at `cpu` (its CPU handle; the GPU one names the same view
    /// in the bound heap) cleared to `values` on the host, through a
    /// dynamic slot of the bound view heap that the list holds until reset.
    pub(crate) fn clear_unordered(
        &mut self,
        list: &HostList,
        cpu: u64,
        values: [u32; 4],
        float: bool,
        rects: &[RECT],
    ) {
        let Some(view) = crate::descriptor::view(cpu) else {
            once!("an unordered access clear of a slot without a view");
            return;
        };
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This));
        let (Some(heap), Some(host)) = (heap, crate::host::host()) else {
            once!("an unordered access clear without a bound view heap");
            return;
        };
        let Some(visible) = heap.host() else { return };
        let Some(slot) = visible.dynamic(1) else {
            return;
        };
        self.dynamic
            .push((std::ptr::from_ref(heap) as usize, slot, 1));
        visible.make_at(slot, &view);
        let at = host.cpu_view();
        let Some(resource) = crate::descriptor::host::make_view(&view, at) else {
            once!("an unordered access clear of a view the host cannot make");
            return;
        };
        let written = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        readable(list, &written, &view);
        self.bind_heaps(list, false);
        list.keep(&resource);
        let gpu = visible.gpu(slot);
        // SAFETY: an open list; both slots view the live resource.
        unsafe {
            if float {
                let values = values.map(f32::from_bits);
                list.list
                    .ClearUnorderedAccessViewFloat(gpu, at, &resource, &values, rects);
            } else {
                list.list
                    .ClearUnorderedAccessViewUint(gpu, at, &resource, &values, rects);
            }
        }
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) -> Outcome {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return Err(Reason::ShaderSkipped);
        }
        let p = require_host_pipeline(self.pipeline as This)?;
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return Err(Reason::PipelineKind);
        }
        self.bind(list, p, true)?;
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.probe_small(list, true);
        #[cfg(feature = "diagnostics")]
        let hdr = self.hdr_before(list, format_args!("dispatch {x}x{y}x{z}"), x != 0 && y != 0 && z != 0);
        #[cfg(feature = "diagnostics")]
        list.crumb(|| pipeline_crumb(p, crate::crumbs::Command::Dispatch, [x, y, z]));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        #[cfg(feature = "diagnostics")]
        self.probe_written(list);
        #[cfg(feature = "diagnostics")]
        hdr_trace::after(list, hdr);
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    #[cfg(feature = "diagnostics")]
    fn crumb(&self, list: &HostList, command: crate::crumbs::Command, args: [u32; 3]) {
        list.crumb(|| {
            let stages = crate::pipeline::host_pipeline(self.pipeline as This)
                .into_iter().flat_map(|p| &p.stages);
            crate::crumbs::Op::new(command, args, stages.map(|s| &s.shader.hash))
        });
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
            for u in t.host.user_data.iter().filter(|u| !is_root_view(u)) {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                    // A slot the host never made: what the title's memory
                    // holds there (our record, zeros, or words it wrote itself).
                    if slot == "null" && self.heaps[0] != 0 {
                        let at = crate::descriptor::heap(self.heaps[0] as This).start + u64::from(index) * 32;
                        let mut words = [0u8; 32];
                        if crate::host::read_into(at as *const u8, &mut words) {
                            let w: Vec<u32> =
                                words.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let _ = write!(text, " (title words {w:x?}");
                            if let Some(view) = crate::descriptor::view(at) {
                                let _ = write!(text, ": {}", describe(Some(&view)));
                            }
                            let _ = write!(text, ")");
                        }
                    }
                }
            }
            let viewed = t.host.user_data.iter().filter(|u| is_root_view(u)).map(|u| u.slot);
            let slots: std::collections::BTreeSet<u8> =
                stage.root_cbvs.iter().chain(&stage.root_srvs).map(|r| r.0).chain(viewed).collect();
            for slot in slots {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    // Its first dwords (a vertex stream header, say).
                    let mut head = [0u8; 16];
                    let head = crate::host::read_into(*address as *const u8, &mut head)
                        .then(|| head.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>());
                    let _ = write!(text, " root{slot}={address:#x} host {host:x?} head {head:x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute {
            let mem = |r: usize| crate::resource::texture_memory(r as This).map_or(0, |(p, _)| p as u64);
            let targets: Vec<String> = self.targets.iter().map(|&t| format!("{t:#x} mem {:#x}", mem(t))).collect();
            let _ = write!(text, " targets {targets:?} depth {:#x} mem {:#x}", self.depth, mem(self.depth));
        }
        text
    }


    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) -> Outcome {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        self.prepare_for(list, false, true)?;
        let p = require_host_pipeline(self.pipeline as This)?;
        if let Some(f) = &p.mesh_fallback {
            return self.run_mesh_fallback(list, p, f, x);
        }
        match list.list.cast::<ID3D12GraphicsCommandList6>() {
            Ok(l6) => {
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("mesh {x}"), false));
                #[cfg(feature = "diagnostics")]
                list.crumb(|| pipeline_crumb(p, crate::crumbs::Command::Mesh, [x, 1, 1]));
                // SAFETY: an open list with everything set.
                unsafe { l6.DispatchMesh(x, 1, 1) };
                DRAWN.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => Err(Reason::Unsupported("native mesh command list unavailable")),
        }
    }

    fn run_mesh_fallback(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        f: &crate::pipeline::host::MeshFallback,
        x: u32,
    ) -> Outcome {
        if x == 0 {
            return Ok(());
        }
        let stride = u64::from(f.layout.record_bytes());
        let (record, bytes) = crate::host::host()
            .and_then(|h| h.mesh_records(list, u64::from(x) * stride))
            .ok_or(Reason::Scratch("mesh records unavailable"))?;
        let groups = x.min(65_535).min((bytes / stride) as u32);
        crate::host::note_mesh_clipped(u64::from(x - groups));
        if groups == 0 {
            return Err(Reason::Scratch("mesh records cannot hold one group"));
        }
        self.bind_as(list, &f.compute, true, false)?;
        let l = &list.list;
        #[cfg(feature = "diagnostics")]
        list.crumb(|| pipeline_crumb(&f.compute, crate::crumbs::Command::MeshFallbackDispatch, [groups, 1, 1]));
        // SAFETY: live scratch and PSOs retained by the list.
        unsafe {
            l.ResourceBarrier(&[crate::host::transition(
                &record, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            l.SetComputeRootUnorderedAccessView(f.uav, record.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstant(f.base, 0, 0);
            l.Dispatch(groups, 1, 1);
        }
        self.draw_mesh_records(list, p, f, &record, |l| {
            let count = f.layout.max_primitives * 3;
            #[cfg(feature = "diagnostics")]
            list.crumb(|| pipeline_crumb(p, crate::crumbs::Command::MeshFallbackDraw, [count, groups, 0]));
            // SAFETY: the draw's state is set by draw_mesh_records.
            unsafe { l.DrawInstanced(count, groups, 0, 0) };
        })
    }

    fn draw_mesh_records(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        f: &crate::pipeline::host::MeshFallback,
        record: &ID3D12Resource,
        draw: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> Outcome {
        let l = &list.list;
        let (uav, srv) = (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        // `bind_as` may switch heaps and invalidate graphics roots; bind the
        // pixel pass again before the draw, not just its PSO.
        let bound = self.bind(list, p, false);
        // SAFETY: finish the UAV writes even on the failed pixel bind path,
        // since the next fallback call can reuse the same records.
        unsafe { l.ResourceBarrier(&[crate::host::transition(record, uav, srv)]) };
        if bound.is_ok() {
            // SAFETY: GPU records have been written, and the list holds the buffer.
            unsafe {
                l.SetGraphicsRootShaderResourceView(f.srv, record.GetGPUVirtualAddress());
                l.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                draw(l);
            }
            self.set_topology(list, self.topology);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: this list just transitioned the held buffer to SRV.
        unsafe {
            l.ResourceBarrier(&[crate::host::transition(record, srv, D3D12_RESOURCE_STATE_COMMON)]);
        }
        bound
    }

    fn execute_mesh_fallback(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        f: &crate::pipeline::host::MeshFallback,
        args: u64,
        count: Option<u64>,
        stride: u32,
        max: u32,
    ) -> Outcome {
        if max == 0 {
            return Ok(());
        }
        let h = crate::host::host().ok_or(Reason::Unknown)?;
        let group_bytes = u64::from(f.layout.record_bytes());
        let (record, bytes) = h.mesh_records(list, 16 << 20)
            .ok_or(Reason::Scratch("mesh records unavailable"))?;
        let group_cap = (bytes / group_bytes) as u32;
        use crate::host::{MESH_COMMAND_CAP, MESH_DRAW_BYTES, transition};
        let records = max.min(MESH_COMMAND_CAP);
        if records < max {
            once!(max, records, "mesh indirect records clipped by command scratch capacity");
        }
        let (commands, base) = list.argument_space(MESH_DRAW_BYTES + u64::from(records) * 16)
            .ok_or(Reason::Scratch("mesh argument space unavailable"))?;
        let draw = host_signature(0, 16).ok_or(Reason::Indirect("mesh draw signature unavailable"))?;
        let (upload, offset, to) = list.scratch_buffer(MESH_DRAW_BYTES)
            .ok_or(Reason::Scratch("mesh argument upload unavailable"))?;
        let l = &list.list;
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let copy = D3D12_RESOURCE_STATE_COPY_DEST;
        let common = D3D12_RESOURCE_STATE_COMMON;
        // SAFETY: both scratch allocations are retained by the list, and
        // each use restores the entire argument chunk to COMMON.
        unsafe {
            to.cast::<[u32; 8]>().write_unaligned([
                f.layout.max_primitives * 3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]);
            l.ResourceBarrier(&[transition(&commands, common, copy)]);
            l.CopyBufferRegion(&commands, base, &upload, offset, MESH_DRAW_BYTES);
            l.ResourceBarrier(&[transition(&commands, copy, uav)]);
        }
        if !crate::indirect::prepare_mesh(list, args, count, stride, records, &commands, base, group_cap) {
            // SAFETY: a failed prepare did not change the argument state.
            unsafe { l.ResourceBarrier(&[transition(&commands, uav, common)]) };
            return Err(Reason::Indirect("mesh argument preparation failed"));
        }
        (self.host_pipeline, self.host_roots[1]) = (0, 0);
        let ran = self.bind_as(list, &f.compute, true, false).and_then(|()| {
            // The title may release the pipeline before submitting the list.
            list.keep(&f.dispatch);
            #[cfg(feature = "diagnostics")]
            list.crumb(|| pipeline_crumb(&f.compute, crate::crumbs::Command::MeshFallbackIndirectDispatch, [records, group_cap, stride]));
            // SAFETY: prepared dispatch stream uses this CS root signature.
            unsafe {
                l.ResourceBarrier(&[transition(&record, common, uav)]);
                l.SetComputeRootUnorderedAccessView(f.uav, record.GetGPUVirtualAddress());
                l.ExecuteIndirect(
                    &f.dispatch,
                    records,
                    &commands,
                    base + MESH_DRAW_BYTES,
                    None,
                    0,
                );
            }
            self.draw_mesh_records(list, p, f, &record, |l| {
                #[cfg(feature = "diagnostics")]
                list.crumb(|| pipeline_crumb(p, crate::crumbs::Command::MeshFallbackIndirectDraw, [1, 16, 0]));
                // SAFETY: the draw's state is set and the argument chunk is held.
                unsafe { l.ExecuteIndirect(&draw, 1, &commands, base, None, 0) };
            })
        });
        // SAFETY: prepare moved the held chunk to INDIRECT_ARGUMENT.
        unsafe {
            l.ResourceBarrier(&[transition(&commands, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, common)]);
        }
        ran
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(*table) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some((view, placed)) = host.mirror(index, handle, true) {
            readable_resolved(list, u, &view, placed.as_ref());
        }
    }

    /// The descriptors a shader reads through tables, made from the
    /// title's slots. Constant buffers there are raw views of their data
    /// (translated shaders read them so: the host wants constant buffer
    /// views 256-byte aligned, the console only 16), brought up to date at
    /// submission like other buffer views. Returns the byte offsets of
    /// tables made afresh in dynamic slots, by root parameter.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once.
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
    ) -> Option<Vec<(u8, u32)>> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = Vec::new();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            let &Root::Table(table) = self.roots_ref(compute).get(usize::from(u.slot))? else {
                return None;
            };
            let heap = self.heap_of(table)?;
            let step = if heap.samplers() { 4 } else { 8 };
            let offset = u.table_offset_dwords;
            if heap.samplers() != (u.kind == ResourceType::Sampler)
                || !offset.is_multiple_of(step) || u32::from(u.dwords) > step
            {
                return None;
            }
            let slots = offset / step + 1;
            let first = heap.table_span(table, slots)?;
            if first + slots > heap.host()?.count() && !fresh.contains(&u.slot) {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let &Root::Table(table) = self.roots_ref(compute).get(usize::from(slot))? else {
                return None;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let heap = self.heap_of(table)?;
            let host = heap.host()?;
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            heap.table_span(table, n)?;
            if !host.can_allocate_table(n) {
                return None;
            }
            let heap_id = std::ptr::from_ref(heap) as usize;
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (heap_id, table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in &entries {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        let placed = host.placed_at(first + u.table_offset_dwords / 8);
                        readable_resolved(list, u, &view, placed.as_ref());
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let first = host.dynamic(n)?;
            self.fresh_tables.insert(key, first);
            self.dynamic.push((heap_id, first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(view) => {
                        let placed = host.make_at(index, &view);
                        if let Some(u) = used {
                            readable_resolved(list, u, &view, placed.as_ref());
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        Some(made)
    }

    #[cfg(feature = "diagnostics")]
    fn hdr_views(&mut self, program: hdr_trace::Program, used: &[UsedResource], compute: bool, made: &[(u8, u32)]) {
        for u in used {
            if u.root != RootType::Table || !program.slot(u.slot, u.table_offset_dwords) { continue; }
            let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else { continue; };
            let at = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(*table) else { continue; };
            let Some(host) = heap.host() else { continue; };
            let index = match made.iter().find(|(s, _)| *s == u.slot) {
                Some(&(_, offset)) => offset / 32 + u.table_offset_dwords / 8,
                None => { let Some(i) = heap.index(at) else { continue; }; i }
            };
            let view = crate::descriptor::view(at);
            let (cpu, backing, observed) = host.hdr_slot(index);
            self.hdr_bindings.push(hdr_trace::Binding::new(u, index, cpu, view.as_ref(), backing, observed));
        }
    }

    #[cfg(feature = "diagnostics")]
    fn hdr_before(&self, list: &HostList, op: std::fmt::Arguments<'_>, nonempty: bool) -> Option<hdr_trace::Event> {
        let program = self.hdr_program?;
        if !nonempty { return None; }
        hdr_trace::before(list, program, op, &self.hdr_bindings, &self.hdr_targets)
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
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
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
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

    /// In a listed frame: each small buffer (64 bytes at most) the draw's
    /// tables name, as title memory now and as its host copy when the list
    /// is submitted (before the GPU runs it) and after it has run.
    #[cfg(feature = "diagnostics")]
    fn probe_small(&self, list: &HostList, compute: bool) {
        if !crate::capture::listed(crate::queue::presented() + 1) || hdr_trace::enabled() {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else { return };
        for stage in &p.stages {
            for u in &stage.translated().used {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.roots_ref(compute).get(usize::from(u.slot)))
                else {
                    continue;
                };
                let Some(view) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let range = match view {
                    View::Constant(address, size) => Some((address, u64::from(size))),
                    ref v => crate::descriptor::host::buffer_range(v),
                };
                let Some((at, len)) = range.filter(|r| r.1 <= 64) else {
                    continue;
                };
                let Some(shadow) = crate::shadow::find(at, len) else { continue };
                let who = format!("{} {:?} s{}+{} {at:#x}+{len}", stage.shader.hash, u.kind, u.slot, u.table_offset_dwords);
                let (s, w) = (shadow.clone(), who.clone());
                list.defer_host(move || tracing::info!(target: "victory_gpu::trace", "small {w} at submission: {}", s.peek(at, len)));
                list.defer_after(move || tracing::info!(target: "victory_gpu::trace", "small {who} after the list: {}", shadow.peek(at, len)));
            }
        }
    }

    /// In a listed frame, right after a dispatch: each 1D or 3D texture it
    /// writes through a table, read back in the list itself (what the
    /// dispatch left there, before anything later in the frame).
    #[cfg(feature = "diagnostics")]
    fn probe_written(&self, list: &HostList) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        if hdr_trace::enabled() { return; }
        let (Some(p), Some(host)) = (
            crate::pipeline::host_pipeline(self.pipeline as This),
            crate::host::host(),
        ) else {
            return;
        };
        for stage in &p.stages {
            for u in stage
                .translated()
                .used
                .iter()
                .filter(|u| u.kind == ResourceType::Uav)
            {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.compute.get(usize::from(u.slot)))
                else {
                    continue;
                };
                let Some(View::Unordered(r, ..)) =
                    crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4)
                else {
                    continue;
                };
                let Some(t) = crate::resource::host_resource(r as This) else {
                    continue;
                };
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D
                    && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D
                {
                    continue;
                }
                const COPY_SOURCE: u32 = 0x800;
                const UNORDERED: u32 = 0x8;
                // All UAV writes done before the copy.
                let all = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
                    },
                };
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[all]) };
                crate::list::ensure_state(list, r as This, COPY_SOURCE);
                let read = crate::capture::read_back(&host.device, &list.list, t);
                crate::list::ensure_state(list, r as This, UNORDERED);
                let Some((buffer, f)) = read else { continue };
                list.keep(&buffer);
                struct Readback(ID3D12Resource);
                // SAFETY: D3D12 objects are free-threaded.
                unsafe impl Send for Readback {}
                // SAFETY: as above.
                unsafe impl Sync for Readback {}
                let buffer = Readback(buffer);
                let who = format!("{} s{}+{} {r:#x} {}x{}x{} f{}", stage.shader.hash, u.slot, u.table_offset_dwords, d.Width, d.Height, d.DepthOrArraySize, d.Format.0);
                list.defer_after(move || {
                    let mut data = std::ptr::null_mut();
                    // SAFETY: the list has run; the mapping spans the first slice.
                    unsafe {
                        if buffer.0.Map(0, None, Some(&mut data)).is_err() {
                            return;
                        }
                        let n = (f.Footprint.RowPitch * f.Footprint.Height) as usize / 4;
                        let w = std::slice::from_raw_parts(data.cast::<u32>(), n);
                        let set = w.iter().filter(|&&x| x != 0).count();
                        let first: Vec<f32> = w.iter().take(8).map(|&x| f32::from_bits(x)).collect();
                        tracing::info!(target: "victory_gpu::trace", "written {who} right after the dispatch: {set} of {n} dwords set, first {first:?}");
                        buffer.0.Unmap(0, None);
                    }
                });
            }
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
                        let view = crate::descriptor::view(handle);
                        let mut text = describe(view.as_ref());
                        // The title's view words, and whether the host slot
                        // the shader reads was made from a view.
                        if let Some(View::Shader(_, Some(w)) | View::Unordered(_, _, Some(w))) = &view {
                            text.push_str(&format!(" words {w:?}"));
                        }
                        let slot = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This))
                            .and_then(|h| {
                                let i = handle.checked_sub(h.start)? / 32;
                                h.host()?.describe_slot(u32::try_from(i).ok()?)
                            });
                        let current: String = view.as_ref().map(|v| format!("{v:?}")).unwrap_or_default().chars().take(120).collect();
                        text.push_str(match slot.as_deref() {
                            Some("null") => " slot null",
                            Some(s) if s.starts_with(&current) => " slot made",
                            Some(_) => " slot stale",
                            None => " slot unknown",
                        });
                        text
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
    ) -> Outcome {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        // Rectangles: completed by the pipeline's geometry shader, else
        // expanded into vertex buffers of their own.
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        self.prepare(list, false)?;
        let (count, from) = if rects {
            self.rectangles(list, vertices, first)?;
            (vertices / 3 * 6, 0)
        } else {
            (vertices, first)
        };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
        #[cfg(feature = "diagnostics")]
        self.probe_small(list, false);
        #[cfg(feature = "diagnostics")]
        let hdr = self.hdr_before(list, format_args!("draw {count}x{instances}"), count != 0 && instances != 0);
        #[cfg(feature = "diagnostics")]
        self.crumb(list, crate::crumbs::Command::Draw, [count, instances, from]);
        // SAFETY: an open list with everything set.
        unsafe { list.list.DrawInstanced(count, instances, from, first_instance) };
        #[cfg(feature = "diagnostics")]
        hdr_trace::after(list, hdr);
        DRAWN.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Whether the bound pipeline's geometry shader completes rectangles.
    fn completes_rectangles(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rectangles)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> Outcome {
        let p = require_host_pipeline(self.pipeline as This)?;
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
                return Err(Reason::Bind("rectangle vertex buffer unavailable or too small"));
            }
            let (gpu, to) = list.scratch(rects * 6 * stride)
                .ok_or(Reason::Scratch("rectangle vertices"))?;
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
        Ok(())
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) -> Outcome {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES && !self.completes_rectangles() {
            once!("indexed rectangles not drawn");
            return Err(Reason::Unsupported("indexed rectangles"));
        }
        self.prepare(list, true)?;
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
        #[cfg(feature = "diagnostics")]
        self.crumb(list, crate::crumbs::Command::Indexed, [indices, instances, first]);
        #[cfg(feature = "diagnostics")]
        let hdr = self.hdr_before(list, format_args!("indexed {indices}x{instances}"), indices != 0 && instances != 0);
        // SAFETY: an open list with everything set.
        unsafe { list.list.DrawIndexedInstanced(indices, instances, first, base, first_instance) };
        #[cfg(feature = "diagnostics")]
        hdr_trace::after(list, hdr);
        DRAWN.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> Outcome {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return Err(Reason::ShaderSkipped);
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let pipeline = self.pipeline as This;
        let len = u64::from(stride) * u64::from(max.max(1));
        let args = crate::shadow::find(arguments, len)
            .ok_or(Reason::Indirect("argument buffer unavailable"))?;
        let counter = match count {
            Some(at) => Some((crate::shadow::find(at, 4)
                .ok_or(Reason::Indirect("count buffer unavailable"))?, at)),
            None => None,
        };
        if kind == MESH {
            let p = require_host_pipeline(self.pipeline as This)?;
            if let Some(fallback) = &p.mesh_fallback {
                args.refresh(list, arguments, len);
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                self.prepare_for(list, false, true)?;
                return self.execute_mesh_fallback(
                    list,
                    p,
                    fallback,
                    args.gpu(arguments),
                    counter.as_ref().map(|(s, at)| s.gpu(*at)),
                    stride,
                    max,
                );
            }
        }
        if !matches!(kind, DRAW | INDEXED | DISPATCH | MESH) {
            return Err(Reason::Unsupported("indirect argument kind"));
        }
        let signature = host_signature(kind, stride)
            .ok_or(Reason::Indirect("host signature unavailable"))?;
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own.
        if !crate::indirect::bound(list, &args.buffer, args.gpu(arguments), (kind, stride, max)) {
            return Err(Reason::Indirect("argument clamp unavailable or misaligned"));
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let prepared = match kind {
            DRAW | INDEXED => {
                if self.topology == RECTANGLES && !self.completes_rectangles() {
                    Err(Reason::Unsupported("indirect rectangles"))
                } else {
                    self.prepare(list, kind == INDEXED)
                }
            }
            MESH => self.prepare_for(list, false, true),
            DISPATCH => require_host_pipeline(self.pipeline as This).and_then(|p| {
                if p.compute { self.bind(list, p, true) } else { Err(Reason::PipelineKind) }
            }),
            _ => Err(Reason::Unknown),
        };
        if let Err(why) = prepared {
            crate::indirect::done(list, &args.buffer);
            return Err(why);
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        #[cfg(feature = "diagnostics")]
        self.crumb(list, crate::crumbs::Command::Indirect, [kind, max, stride]);
        #[cfg(feature = "diagnostics")]
        let mut hdr = self.hdr_before(list, format_args!("indirect kind={kind} max={max} stride={stride}"), max != 0);
        #[cfg(feature = "diagnostics")]
        if let Some(event) = &mut hdr {
            event.indirect(list, (&args.buffer, args.offset(arguments)),
                counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at))), kind, stride, max);
        }
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
        #[cfg(feature = "diagnostics")]
        hdr_trace::after(list, hdr);
        // A listed frame: what the records held (after the clamp).
        #[cfg(feature = "diagnostics")]
        if crate::capture::dump_on() {
            let count = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
            let name = crate::pipeline::shaders_of(pipeline);
            crate::capture::read_indirect(list, (&args.buffer, args.offset(arguments)), count, (kind, stride, max), name);
        }
        crate::indirect::done(list, &args.buffer);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
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

#[cfg(feature = "diagnostics")]
fn pipeline_crumb(p: &HostPipeline, command: crate::crumbs::Command, args: [u32; 3]) -> crate::crumbs::Op {
    crate::crumbs::Op::new(command, args, p.stages.iter().map(|s| &s.shader.hash))
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    let placed = crate::descriptor::host::placed_texture(view);
    readable_resolved(list, u, view, placed.as_ref());
}

fn readable_resolved(list: &HostList, u: &UsedResource, view: &View, owner: Option<&std::sync::Arc<crate::resource::Placed>>) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    // Use the exact generation installed in the slot. Do not bind an old
    // descriptor and upload or transition the texture of a new mapping.
    let placed = owner.map(|p| crate::descriptor::host::placed_shader(view, p));
    if let Some(p) = owner
        && !crate::resource::was_remapped(p.get())
        && let Some(a) = crate::resource::alias_source(p.get())
    {
        crate::texture::alias_copy(list, a.get(), p.get());
    }
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    // Constant buffers (raw views) to their copy's end at most; others in
    // the copy their host view reads.
    let range = match *view {
        View::Constant(address, size) => crate::shadow::find(address, 1).and_then(|s| {
            let len = u64::from(size).min(s.left(address));
            crate::shadow::find(address, len).map(|s| (s, address, len))
        }),
        _ => crate::descriptor::host::view_copy(view),
    };
    if let Some((shadow, address, len)) = range
        && len > 0
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload_placed(list, *resource as This, t, owner);
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
            crate::capture::note_target(list, *resource as This);
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
            // Its console memory: another view of it (a placed one) is
            // another host texture.
            let at = crate::resource::texture_memory(r).map_or(0, |(p, _)| p as u64);
            format!("{}x{} f{} flags {:#x} mem {at:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    // Small buffers (exposure, counters): their title memory now, as floats.
    let small = |range: Option<(u64, u64)>| match range {
        Some((at, len)) if len <= 64 && crate::texture::committed(at as *const u8, len as usize) => {
            // SAFETY: `len` committed bytes of title memory.
            let b = unsafe { std::slice::from_raw_parts(at as *const u8, len as usize) };
            let f: Vec<f32> = b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            format!(" = {f:?}")
        }
        _ => String::new(),
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This).unwrap_or_else(|| {
            let range = crate::descriptor::host::buffer_range(v);
            format!("buffer {range:?}{}", small(range))
        }),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || {
                let range = crate::descriptor::host::buffer_range(v);
                format!("uav buffer {range:?}{}", small(range))
            },
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
        // The first four values, as a colour rectangle's constants are, read
        // through the kernel (asking `VirtualQuery` first cost milliseconds
        // per draw in the title's pools, with every draw noted).
        Some(View::Constant(address, size)) if *size >= 16 => {
            let mut bytes = [0u8; 16];
            if crate::host::read_into(*address as *const u8, &mut bytes) {
                let v: [f32; 4] = std::array::from_fn(|i| f32::from_le_bytes([0, 1, 2, 3].map(|k| bytes[4 * i + k])));
                format!("Constant({address:#x}, {size}) {v:?}")
            } else {
                format!("Constant({address:#x}, {size})")
            }
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// A user-data entry that takes the heap index of a root view's view.
fn is_root_view(u: &victory_shader::emit::UserData) -> bool {
    u.root == RootType::Descriptor && u.dword >= victory_shader::emit::ROOT_SRV_VIEW
}

/// Whether a shader reads root views through views the host makes.
fn has_root_views(t: &victory_shader::emit::HostShader) -> bool {
    t.user_data.iter().any(is_root_view)
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

#[cfg(feature = "diagnostics")]
pub(crate) mod hdr_trace {
    use super::{HostList, This, UsedResource, View};
    use crate::descriptor::host::{HdrViewInfo, hdr_current};
    use crate::host::{host, transition};
    use std::sync::atomic::AtomicU32;
    use std::sync::{Mutex, OnceLock};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;

    pub(crate) fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_HDR_TRACE").is_some_and(|v| v == "1")
            && std::env::var_os("VICTORY_CAPTURE_WORLD").is_some()
            && crate::capture::listed_only())
    }

    pub(crate) fn frame_is_selected(frame: u64) -> bool {
        enabled() && crate::capture::listed(frame)
    }

    /// Frames with a TiledLighting draw so far and, from
    /// `VICTORY_HDR_TRACE_AFTER` (default 1), the one to flush without a
    /// listed frame. This is the first frame where the real lit path
    /// executes, not an offset that a slow world may never reach.
    static TILED: AtomicU32 = AtomicU32::new(0);
    static WITH_TILED: AtomicU32 = AtomicU32::new(0);
    static NOTES: AtomicU32 = AtomicU32::new(0);

    /// Whether the Present now processing is the trigger frame, without
    /// consuming it: `capture::wanted` asks before `flush` runs.
    pub(crate) fn auto_frame(_n: u64) -> bool {
        if !enabled() || !crate::capture::world_started() { return false; }
        TILED.load(std::sync::atomic::Ordering::Relaxed) > 0
            && WITH_TILED.load(std::sync::atomic::Ordering::Relaxed) + 1 == auto_after()
    }

    fn auto_after() -> u32 {
        static AFTER: OnceLock<u32> = OnceLock::new();
        *AFTER.get_or_init(|| std::env::var("VICTORY_HDR_TRACE_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
    }

    /// Recording is armed for the whole run: the title records a frame's
    /// commands before that frame is presented, so filtering by the
    /// presented number here misses them. Kept lines are flushed only in
    /// the selected frame's Present.
    pub(crate) fn active_frame() -> Option<u64> {
        enabled().then_some(0)
    }

    #[derive(Clone, Copy, Debug)]
    #[repr(usize)]
    pub(super) enum Program { Temporal, Diffusion, Tiled, TiledNoDir, Taa, Composite, CompositeNoise, Upscale }

    impl Program {
        pub(super) fn from_hash(hash: &str) -> Option<Self> {
            Some(match hash {
                "34fb51b8a0312e82" => Self::Temporal,
                "68029c67dca01425" => Self::Diffusion,
                "1cd9bc634417a981" => Self::Tiled,
                "48e74d16cbfa45e3" => Self::TiledNoDir,
                "df11dcd012cd7382" => Self::Taa,
                "7c649c422f6208e4" => Self::Composite,
                "5970ee6dbc47ec61" => Self::CompositeNoise,
                "4f4ec7b16673362f" => Self::Upscale,
                _ => return None,
            })
        }

        fn graphics(self) -> bool {
            matches!(self, Self::Tiled | Self::TiledNoDir | Self::Composite | Self::CompositeNoise | Self::Upscale)
        }

        fn tiled(self) -> bool {
            matches!(self, Self::Tiled | Self::TiledNoDir)
        }

        pub(super) fn slot(self, slot: u8, dwords: u32) -> bool {
            matches!((self, slot, dwords),
                (Self::Temporal, 0, 448 | 464 | 520 | 640 | 688) |
                (Self::Temporal, 1, 8..=208) |
                (Self::Diffusion, 0, 224 | 480 | 568) |
                (Self::Diffusion, 1, 8 | 16) |
                (Self::Taa, 1, 56 | 112 | 136 | 160) |
                (Self::Composite | Self::CompositeNoise, 1, 32 | 304))
        }

        fn takes_sample(self, slot: u8, offset: u32) -> Option<Coord> {
            match (self, slot, offset) {
                (Self::Temporal, 0, 448) => Some(Coord::Tile),
                (Self::Diffusion, 0, 568) => Some(Coord::PackedY),
                (Self::Temporal, 0, 464) |
                (Self::Temporal, 1, 48 | 80 | 96 | 128 | 136 | 208) |
                (Self::Diffusion, 1, 8 | 16) |
                (Self::Taa, 1, 56 | 112 | 136 | 160) |
                (Self::Composite | Self::CompositeNoise, 1, 32 | 304) => Some(Coord::Pixel),
                _ => None,
            }
        }
    }

    // These IDs are advisory sample points, not asserted classifications.
    // The Temporal depth output at the same points supplies the witness.
    #[derive(Clone, Copy, Debug)]
    enum Coord { Pixel, Tile, PackedY }

    #[derive(Clone)]
    pub(super) struct Target {
        object: usize,
        backing: ID3D12Resource,
        format: Option<i32>,
        sub: Option<(u32, u32, u32)>,
    }

    impl Target {
        pub(super) fn rtv(resource: This, texture: &ID3D12Resource, desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>) -> Self {
            // SAFETY: live resource and view whose dimension selects the union member.
            let (format, sub) = unsafe { match desc {
                None => {
                    let d = texture.GetDesc();
                    (d.Format.0, (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D).then_some((0, 0, 0)))
                }
                Some(d) => (d.Format.0, match d.ViewDimension {
                    D3D12_RTV_DIMENSION_TEXTURE2D => {
                        let a = d.Anonymous.Texture2D; Some((a.MipSlice, 0, a.PlaneSlice))
                    }
                    D3D12_RTV_DIMENSION_TEXTURE2DARRAY => {
                        let a = d.Anonymous.Texture2DArray; Some((a.MipSlice, a.FirstArraySlice, a.PlaneSlice))
                    }
                    _ => None,
                }),
            }};
            Self { object: resource as usize, backing: texture.clone(), format: (format != 0).then_some(format), sub }
        }
    }

    #[derive(Clone)]
    pub(super) struct Binding {
        slot: u8,
        offset: u32,
        kind: victory_shader::ResourceType,
        host_slot: u32,
        cpu: usize,
        object: usize,
        memory: Option<u64>,
        requested: Option<u32>,
        info: Option<HdrViewInfo>,
        backing: Option<ID3D12Resource>,
        twin: bool,
    }

    impl Binding {
        pub(super) fn new(u: &UsedResource, host_slot: u32, cpu: usize, view: Option<&View>,
            backing: Option<ID3D12Resource>, info: Option<HdrViewInfo>) -> Self
        {
            let (object, requested, memory) = match view {
                Some(View::Shader(id, desc) | View::Unordered(id, _, desc)) =>
                    (*id, desc.map(|d| crate::resource::pc_format(d[0])), None),
                Some(View::Placed(_, _, d)) =>
                    (0, Some(crate::resource::pc_format(d[0])), Some(u64::from(d[10]) | (u64::from(d[11]) << 32))),
                _ => (0, None, None),
            };
            let twin = object != 0 && backing.as_ref().is_some_and(|b|
                crate::resource::host_copy(object as This).is_some_and(|c| c.texture.as_raw() != b.as_raw()));
            Self { slot: u.slot, offset: u.table_offset_dwords, kind: u.kind, host_slot,
                cpu, object, memory, requested, info, backing, twin }
        }

        fn stable(&self) -> bool { self.info.is_some() && hdr_current(self.cpu) == self.info }
    }

    pub(crate) struct Event {
        frame: u64,
        program: Program,
        command: String,
        bindings: Vec<Binding>,
        targets: Vec<Target>,
        samples: Vec<(String, Sample)>,
        args: Option<(Readback, u32, u32, bool, u32)>,
        submission: u64,
    }

    // COM objects here are only D3D12 resources, documented free-threaded.
    // SAFETY: mappings are read only after the submission fence under its lock.
    unsafe impl Send for Event {}

    struct Quota { seen: [u8; 7] }
    static QUOTA: Mutex<Quota> = Mutex::new(Quota { seen: [0; 7] });

    pub(super) fn before(list: &HostList, program: Program, command: std::fmt::Arguments<'_>,
        bindings: &[Binding], targets: &[Target]) -> Option<Event>
    {
        active_frame()?;
        use std::sync::atomic::Ordering::Relaxed;
        // The first TiledLighting draw of the frame decides whether this is
        // the frame to keep; other frames record no copies or lines.
        let first_tiled = program.tiled() && TILED.fetch_add(1, Relaxed) == 0;
        let listed_now = crate::capture::listed(crate::queue::presented() + 1);
        let candidate = listed_now
            || (first_tiled && crate::capture::world_started()
                && WITH_TILED.load(Relaxed) + 1 == auto_after());
        if !candidate { return None; }
        let mut q = QUOTA.lock().unwrap_or_else(|e| e.into_inner());
        let seen = &mut q.seen[program as usize];
        if *seen != 0 { return None; }
        *seen = 1;
        drop(q);
        let mut event = Event { frame: crate::queue::presented() + 1, program, command: command.to_string(), bindings: bindings.to_vec(),
            targets: if program.graphics() { targets.to_vec() } else { Vec::new() },
            samples: Vec::new(), args: None, submission: 0 };
        event.record(list, false);
        Some(event)
    }

    pub(super) fn after(list: &HostList, event: Option<Event>) {
        if let Some(mut event) = event {
            event.record(list, true);
            list.hdr.lock().unwrap_or_else(|e| e.into_inner()).push(event);
        }
    }

    impl Event {
        fn record(&mut self, list: &HostList, after: bool) {
            let Some(h) = host() else { return; };
            if let Some(t) = self.targets.first()
                && let Some(state) = list.hdr_state(&t.backing)
                && let Some(sample) = t.sub.and_then(|sub| sample(&h.device, &list.list, &t.backing, state, sub, Coord::Pixel))
            {
                self.samples.push((if after { "MRT0-after" } else { "MRT0-before" }.into(), sample));
            }
            for b in &self.bindings {
                if after != (b.kind == victory_shader::ResourceType::Uav) { continue; }
                let Some(coord) = self.program.takes_sample(b.slot, b.offset) else { continue; };
                let (Some(backing), Some(sub)) = (&b.backing, b.info.and_then(|i| i.sub)) else { continue; };
                let Some(state) = list.hdr_state(backing) else { continue; };
                if let Some(sample) = sample(&h.device, &list.list, backing, state, sub, coord) {
                    self.samples.push((format!("s{}+{}-{}", b.slot, b.offset, if after { "after" } else { "before" }), sample));
                }
            }
        }

        pub(super) fn indirect(&mut self, list: &HostList, args: (&ID3D12Resource, u64),
            count: Option<(&ID3D12Resource, u64)>, kind: u32, stride: u32, max: u32)
        {
            // Read only the prefix. We can prove work if a counted prefix
            // includes one nonempty record, not prove none when truncated.
            let n = max.min(32);
            if !matches!(kind, 0..=2) || !(12..=64).contains(&stride) || stride % 4 != 0 { return; }
            let len = u64::from(n) * u64::from(stride);
            let Some(h) = host() else { return; };
            let Some(buffer) = Readback::new(&h.device, len + 4) else { return; };
            let read = D3D12_RESOURCE_STATE_COPY_SOURCE;
            let arg = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            // SAFETY: args are bounded and transitioned by indirect::bound;
            // the counter follows the existing indirect capture's state contract.
            unsafe {
                let l = &list.list;
                l.ResourceBarrier(&[transition(args.0, arg, read)]);
                l.CopyBufferRegion(&buffer.0, 0, args.0, args.1, len);
                if let Some((c, at)) = count {
                    let same = c.as_raw() == args.0.as_raw();
                    let write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    if !same { l.ResourceBarrier(&[transition(c, write, read)]); }
                    l.CopyBufferRegion(&buffer.0, len, c, at, 4);
                    if !same { l.ResourceBarrier(&[transition(c, read, write)]); }
                }
                l.ResourceBarrier(&[transition(args.0, read, arg)]);
            }
            self.args = Some((buffer, n, stride, count.is_some(), kind));
        }
    }

    static PENDING: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static BACKINGS: Mutex<Option<std::collections::HashMap<usize, String>>> = Mutex::new(None);

    pub(crate) fn label(backing: usize) -> Option<String> {
        BACKINGS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&backing).cloned())
    }

    pub(crate) fn dropped(shaders: &str, why: &str) {
        if !enabled() { return; }
        const HASHES: [&str; 7] = ["34fb51b8a0312e82", "68029c67dca01425", "1cd9bc634417a981",
            "48e74d16cbfa45e3", "df11dcd012cd7382", "7c649c422f6208e4", "5970ee6dbc47ec61"];
        if !HASHES.iter().any(|h| shaders.contains(h)) { return; }
        PENDING.lock().unwrap_or_else(|e| e.into_inner())
            .push(format!("dropped pipeline={shaders} reason={why}"));
    }

    /// Lines kept for the frame now presenting; emitted only if it is the
    /// selected frame, and dropped either way.
    pub(crate) fn flush(frame: u64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let tiled = TILED.swap(0, Relaxed) > 0 && crate::capture::world_started();
        let listed = frame_is_selected(frame);
        let with = if tiled { WITH_TILED.fetch_add(1, Relaxed) + 1 } else { WITH_TILED.load(Relaxed) };
        let auto = tiled && with == auto_after();
        let selected = listed || auto;
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if selected {
            if auto && !listed { NOTES.fetch_add(1, Relaxed); }
            for line in pending.iter() {
                tracing::info!(target: "victory_gpu::hdr_trace", frame, auto, listed, "{line}");
            }
        }
        pending.clear();
        *QUOTA.lock().unwrap_or_else(|e| e.into_inner()) = Quota { seen: [0; 7] };
        selected
    }

    pub(crate) fn complete(list: &HostList, _frame: u64, submission: u64, ordinal: usize) {
        let mut events = list.hdr.lock().unwrap_or_else(|e| e.into_inner());
        if events.is_empty() { return; }
        let id = list.list.as_raw();
        let mut labels = BACKINGS.lock().unwrap_or_else(|e| e.into_inner());
        let labels = labels.get_or_insert_with(std::collections::HashMap::new);
        let mut lines = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        for (op, e) in events.drain(..).enumerate() {
            let stable = e.bindings.iter().all(|b| b.info.is_none() || b.stable());
            lines.push(format!("op={op} submission={submission} fence={submission} ordinal={ordinal} list={id:?} program={:?} command={:?}", e.program, e.command));
            for b in &e.bindings {
                let observed = b.stable().then_some(b.info).flatten();
                let effective = observed.and_then(|i| i.effective);
                let expected_f32 = matches!(e.program, Program::Temporal | Program::Diffusion)
                    && e.program.takes_sample(b.slot, b.offset).is_some();
                let integer = effective.is_some_and(|f| matches!(f, 3|4|7|8|12|14|17|18|25|30|32|36|38|42|43|50|52|57|59|62|64));
                if let Some(backing) = b.backing.as_ref().map(Interface::as_raw) {
                    labels.entry(backing as usize).or_insert_with(|| format!("{:?} s{}+{} {:?}", e.program, b.slot, b.offset, b.kind));
                }
                lines.push(format!(
                    "view slot={} dwords={} kind={:?} host_slot={} cpu={:#x} object={:#x} memory={:?} requested={:?} effective={:?} generation_at_bind={:?} generation_now={:?} stable={} backing={:?} twin={} null_or_unretained={} sub={:?} expected_f32={} class_mismatch={}",
                    b.slot, b.offset, b.kind, b.host_slot, b.cpu, b.object, b.memory, b.requested,
                    effective, b.info.map(|i| i.generation), hdr_current(b.cpu).map(|i| i.generation),
                    b.stable(), b.backing.as_ref().map(Interface::as_raw), b.twin,
                    b.backing.is_none(), observed.and_then(|i| i.sub), expected_f32,
                    expected_f32 && integer));
            }
            for (mrt, t) in e.targets.iter().enumerate() {
                labels.entry(t.backing.as_raw() as usize).or_insert_with(|| format!("{:?} MRT{mrt}", e.program));
                lines.push(format!("render_target mrt={mrt} object={:#x} backing={:?} effective={:?} sub={:?}",
                    t.object, t.backing.as_raw(), t.format, t.sub));
            }
            for (name, s) in &e.samples {
                lines.push(format!("sample name={name} stable={stable} coordinates={:?} footprint_format={} words={:?}",
                    s.points, s.format, s.words()));
            }
            if let Some((args, n, stride, counted, kind)) = &e.args {
                let len = (*n * *stride) as usize;
                let words = args.read(len + 4);
                let count = if *counted { words.as_ref().and_then(|b| word(&b[len..])) } else { Some(*n) };
                let live = count.map(|c| c.min(*n) as usize);
                let found = words.as_ref().zip(live).and_then(|(b, live)| b[..len].chunks_exact(*stride as usize)
                    .take(live).position(|r| r.chunks_exact(4).take(if *kind == 2 { 3 } else { 2 })
                        .all(|w| word(w).is_some_and(|w| w > 0))));
                lines.push(format!("indirect count={count:?} captured_records={n} first_nonempty_record={found:?}"));
            }
        }
    }

    fn word(b: &[u8]) -> Option<u32> { Some(u32::from_le_bytes(b.get(..4)?.try_into().ok()?)) }

    struct Readback(ID3D12Resource);
    impl Readback {
        fn new(device: &ID3D12Device, size: u64) -> Option<Self> {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size, Height: 1, DepthOrArraySize: 1,
                MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let mut buffer = None;
            // SAFETY: plain creation with an owned out pointer.
            unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer).ok()?; }
            buffer.map(Self)
        }

        fn read(&self, size: usize) -> Option<Vec<u8>> {
            let mut data = std::ptr::null_mut();
            // SAFETY: after the fence, allocation has at least `size` bytes.
            unsafe {
                self.0.Map(0, None, Some(&mut data)).ok()?;
                let out = std::slice::from_raw_parts(data.cast::<u8>(), size).to_vec();
                self.0.Unmap(0, None);
                Some(out)
            }
        }
    }

    pub(crate) struct Sample {
        buffer: Readback,
        points: [(u32, u32); 2],
        format: i32,
        bytes: usize,
    }

    impl Sample {
        pub(crate) fn words(&self) -> Option<[String; 2]> {
            let b = self.buffer.read(512 + self.bytes)?;
            Some([&b[..self.bytes], &b[512..]].map(|slot| {
                let hex = slot.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join("");
                if self.format != 10 { return hex; }
                let fp16 = |i: usize| {
                    let w = u16::from_le_bytes([slot[i], slot[i + 1]]);
                    let sign = if w & 0x8000 == 0 { 1.0 } else { -1.0 };
                    let exp = (w >> 10) & 31;
                    let f = f32::from(w & 1023) / 1024.0;
                    sign * match exp { 0 => f * 2f32.powi(-14), 31 if f == 0.0 => f32::INFINITY,
                        31 => f32::NAN, _ => (1.0 + f) * 2f32.powi(i32::from(exp) - 15) }
                };
                format!("{hex} rgb={:?}", [fp16(0), fp16(2), fp16(4)])
            }))
        }
    }

    fn sample(device: &ID3D12Device, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource,
        state: D3D12_RESOURCE_STATES, (mip, array, plane): (u32, u32, u32), coord: Coord) -> Option<Sample>
    {
        // SAFETY: live resource. No partial copies of depth/stencil or MSAA.
        let desc = unsafe { texture.GetDesc() };
        let format = desc.Format.0;
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 || plane != 0
            || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
            || mip >= u32::from(desc.MipLevels) || array >= u32::from(desc.DepthOrArraySize)
            || !matches!(format, 9..=14 | 23..=43 | 48..=54 | 56..=57 | 60..=65 | 67 | 87..=93)
        { return None; }
        let (bytes, 1) = crate::resource::element(format as u32)? else { return None; };
        let (w, h) = (u32::try_from(desc.Width).ok()?.checked_shr(mip)?.max(1), desc.Height.checked_shr(mip)?.max(1));
        let points = [3, 13].map(|row| {
            let (sx, sy) = match coord { Coord::Pixel => (1, 1), Coord::Tile => (16, 16), Coord::PackedY => (1, 16) };
            // Mid-screen x; sky/geometry candidates at 3/16 and 13/16.
            // Scale to the previously observed scene size for tile masks.
            let (x, y) = if sx == 1 && sy == 1 { (w / 2, h * row / 16) } else { (1280 / sx, 1440 * row / 16 / sy) };
            (x.min(w - 1), y.min(h - 1))
        });
        let index = mip + array * u32::from(desc.MipLevels);
        let buffer = Readback::new(device, 1024)?;
        let read = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // SAFETY: boxed copies are inside a color subresource; aligned offsets
        // and pitch, matching format, live objects. Restore exact state.
        unsafe {
            if state != read { list.ResourceBarrier(&[transition(texture, state, read)]); }
            let from = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index } };
            for (i, (x, y)) in points.into_iter().enumerate() {
                let to = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&buffer.0),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: i as u64 * 512, Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: desc.Format, Width: 1, Height: 1, Depth: 1, RowPitch: 256 } } } };
                list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&D3D12_BOX {
                    left: x, top: y, front: 0, right: x + 1, bottom: y + 1, back: 1 }));
            }
            if state != read { list.ResourceBarrier(&[transition(texture, read, state)]); }
        }
        Some(Sample { buffer, points, format, bytes: bytes as usize })
    }

    pub(crate) fn present_probe(device: &ID3D12Device, list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource, state: D3D12_RESOURCE_STATES, frame: u64) -> Option<Sample>
    {
        if !frame_is_selected(frame) { return None; }
        sample(device, list, back, state, (0, 0, 0), Coord::Pixel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    /// Reads 16 dwords 64 bytes apart from `offset` past the root view
    /// (heap index in the table) as the translated shaders do: the root
    /// address's offset in the view plus their own.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), RootConstants(num32BitConstants=2, b0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint root; uint offset; }
[RootSignature(RS)]
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src.Load(root + offset + id.x * 64));
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        compile_cs(SOURCE)
    }

    /// A compute shader with its root signature inside (`RS`).
    fn compile_cs(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("root"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: a compiled shader with its root signature inside.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    /// A title buffer of `bytes` in memory of its own.
    fn buffer(bytes: u64) -> This {
        let desc: [u32; 14] = [1, 0, 0, 0, bytes as u32, (bytes >> 32) as u32, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// Root views (`victory_shader::emit`, "Root SRVs and root UAVs") through
    /// our device: a raw view of the host copy holding the root address,
    /// brought up to date from there to the copy's end, so data 1.5 MiB past
    /// the root address (past the 1 MiB the old root views brought up to
    /// date) reads right; reads past the copy's end read 0 (a host root
    /// descriptor read whatever lay there). Two root addresses in one copy
    /// share its view.
    #[test]
    fn root_views_reach_their_copy_and_read_zeros_past_it() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const MIB: u64 = 1 << 20;
        let (src, dst) = (buffer(4 * MIB), buffer(256));
        let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, dst, null, &uav, title_heap.start + 32);
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        let src = src_bytes as u64;
        let cases = [
            // Data 1.5 MiB past the root address.
            (src + 0x100, 0x18_0000u32, (0..16).map(|k| 0x5000 + k).collect::<Vec<u32>>()),
            // The copy's last 256 bytes: four lanes inside, the rest past it.
            (src + 4 * MIB - 256, 0, (0..16).map(|k| if k < 4 { 0x7000 + k } else { 0 }).collect()),
        ];
        for (round, (address, offset, want)) in cases.into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // The title writes after recording, before submitting.
            let mut state = State::default();
            state.set_heaps(&[heap]);
            state.set_view(true, 0, address);
            let (index, at) = state.root_view(&list, true, 0, false).expect("a root view");
            assert_eq!(u64::from(at), address - src, "the root address's offset in the copy");
            state.set_view(true, 1, address - 0x40);
            let other = state.root_view(&list, true, 1, false).expect("a root view");
            assert_eq!(other, (index, at - 0x40), "one view per copy");
            let constants = [at, offset];
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(index));
                list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(1));
                list.list.SetComputeRoot32BitConstants(2, 2, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            for k in 0..16u64 {
                let at = address + u64::from(offset) + 64 * k;
                if at + 4 <= src + 4 * MIB {
                    let value = if round == 0 { 0x5000 + k as u32 } else { 0x7000 + k as u32 };
                    // SAFETY: inside the title buffer.
                    unsafe { (at as *mut u32).write(value) };
                }
            }
            host.execute(&[&list]);
            assert_eq!(out.read_host(dst_bytes as u64, 16), want, "round {round}");
        }
    }

    /// Title memory of `bytes` (64 KiB aligned) for textures at an address.
    fn title_memory(bytes: usize) -> u64 {
        let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
        // SAFETY: a non-zero size; leaked, as title memory outlives a test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    /// A 2D resource description as words: size, format, layout, flags.
    fn texture_words(width: u32, height: u32, format: u32, layout: u32, flags: u32) -> [u32; 14] {
        [3, 0, 0, 0, width, 0, height, 1 | 1 << 16, format, 1, 0, layout, flags, 0]
    }

    /// A title texture (resource object) at `address`.
    fn texture_at_address(address: u64, words: &[u32; 14]) -> This {
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut out = std::ptr::null();
        let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
        assert_eq!(made, S_OK);
        out
    }

    /// A placed shader view (device 0x230) of a 2D texture at `address`.
    fn placed_view(address: u64, words: [u32; 14]) -> View {
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[words[8], 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, words, desc)
    }

    fn used(kind: ResourceType) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }
    }

    /// Slots of a shader-visible view heap of ours.
    struct Slots(ID3D12DescriptorHeap, u32);

    impl Slots {
        fn new(n: u32) -> Self {
            let host = crate::host::host().unwrap();
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: n,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain creation.
            let heap = unsafe { host.device.CreateDescriptorHeap(&desc) }.unwrap();
            // SAFETY: a plain query.
            let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            Slots(heap, step)
        }

        fn make(&self, i: u32, view: &View) {
            // SAFETY: a live heap.
            let mut at = unsafe { self.0.GetCPUDescriptorHandleForHeapStart() };
            at.ptr += (i * self.1) as usize;
            assert!(crate::descriptor::host::make_view(view, at).is_some(), "slot {i} made");
        }

        fn gpu(&self, i: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
            // SAFETY: a live shader-visible heap.
            let mut at = unsafe { self.0.GetGPUDescriptorHandleForHeapStart() };
            at.ptr += u64::from(i * self.1);
            at
        }
    }

    /// Records a dispatch of `pso` over `groups` with the table at slot
    /// `table` as parameter 0 and, if given, root UAV `out` as parameter 1.
    fn run(
        list: &HostList,
        slots: &Slots,
        (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
        table: u32,
        out: Option<&ID3D12Resource>,
        groups: (u32, u32),
    ) {
        // SAFETY: an open list and live objects.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(slots.0.clone())]);
            list.list.SetComputeRootSignature(root);
            list.list.SetPipelineState(pso);
            list.list.SetComputeRootDescriptorTable(0, slots.gpu(table));
            if let Some(out) = out {
                list.list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            }
            list.list.Dispatch(groups.0, groups.1, 1);
        }
    }

    const READ: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u1)"
Texture2D<float4> t : register(t0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store4((id.y * 64 + id.x) * 16, asuint(t.Load(int3(id.xy, 0))));
}
"#;

    #[test]
    fn capped_tables_validate_span_width_and_refuse_exhaustion() {
        let raw = crate::descriptor::test_capped_view_heap(8, 2, 5);
        let heap = crate::descriptor::heap(raw);
        let host = crate::host::host().expect("D3D12 test host");
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[raw]);
        state.set_table(true, 0, heap.start + 2 * 32);
        let srv = |offset| UsedResource {
            table_offset_dwords: offset,
            ..used(ResourceType::Srv)
        };
        let small = state.tables(&list, &[srv(0)], true).unwrap()[0].1;
        let large = state.tables(&list, &[srv(0), srv(16)], true).unwrap()[0].1;
        assert_ne!(small, large);
        assert_eq!(state.tables(&list, &[srv(0)], true).unwrap()[0].1, small);
        state.set_table(true, 0, heap.start + 7 * 32);
        assert!(state.tables(&list, &[srv(8)], true).is_none());
        state.set_table(true, 0, heap.start + 7 * 32 + 4);
        assert!(state.tables(&list, &[srv(0)], true).is_none());
        state.set_table(true, 0, heap.start + 4 * 32);
        assert!(state.tables(&list, &[srv(8)], true).is_none(), "only one free slot remains");
        assert_eq!(state.dynamic.len(), 2);
        drop(state);
        // A reset/release gives the dynamic slots back to the next list.
        let mut state = State::default();
        state.set_heaps(&[raw]);
        state.set_table(true, 0, heap.start + 4 * 32);
        assert!(state.tables(&list, &[srv(8)], true).is_some());
        drop(state);
        let release: extern "system" fn(This) -> u32 = crate::object::slot(raw, 0x10);
        release(raw);
    }

    /// One texture memory written through an unordered view of a resource
    /// object in one format and read through a placed view in another: a
    /// castable pair shares one host texture (no copy), a pair of other
    /// shapes over the same bytes (block-compressed against plain) is two,
    /// bridged by a copy at the read. On a host without relaxed format
    /// casting (Intel Iris Xe) the castable pair is two as well, and the
    /// copy bridges it the same way.
    #[test]
    fn a_texture_written_in_one_format_reads_in_another_through_a_placed_view() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Format written and size, what the shader writes, format read and
        // size, the texel expected, whether one host texture.
        type Case = (u32, u32, &'static str, u32, u32, [f32; 4], bool);
        let cases: [Case; 2] = [
            // R32_UINT holding R9G9B9E5 (1, 0.5, 0.25): the sky's tables.
            (
                42,
                64,
                "RWTexture2D<uint> o : register(u0); void put(uint2 p) { o[p] = 256u | 128u << 9 | 64u << 18 | 16u << 27; }",
                67,
                64,
                [1.0, 0.5, 0.25, 1.0],
                true,
            ),
            // R32G32_UINT blocks read as BC1: solid red blocks.
            (
                17,
                16,
                "RWTexture2D<uint2> o : register(u0); void put(uint2 p) { o[p] = uint2(0xF800, 0); }",
                71,
                64,
                [1.0, 0.0, 0.0, 1.0],
                false,
            ),
        ];
        for (round, (written, size, put, format, read_size, want, shared)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 20);
            let object = texture_at_address(address, &texture_words(size, size, written, 0x105, 4));
            let view = placed_view(address, texture_words(read_size, read_size, format, 0x105, 0));
            let placed = crate::descriptor::host::placed_texture(&view).expect("a placed texture");
            let a = crate::resource::host_copy(object).unwrap();
            let b = crate::resource::host_copy(placed.get()).unwrap();
            // One host texture only where the host took the castable formats.
            let shared = shared && !a.castable.is_empty();
            assert_eq!(std::sync::Arc::ptr_eq(&a, &b), shared, "round {round}: one host texture");
            let unordered = View::Unordered(object as usize, 0, Some([written, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
            slots.make(0, &unordered);
            slots.make(1, &view);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ put(id.xy); }}"
            ));
            let list = host.new_list().unwrap();
            let copies = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed);
            readable(&list, &used(ResourceType::Uav), &unordered);
            run(&list, &slots, &write, 0, None, (size / 8, size / 8));
            readable(&list, &used(ResourceType::Srv), &view);
            run(&list, &slots, &read, 1, Some(&out), (read_size / 8, read_size / 8));
            list.close();
            host.execute(&[&list]);
            let copied = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed) > copies;
            assert_eq!(copied, !shared, "round {round}: a copy only where not shared");
            let n = read_size as usize;
            for i in [0usize, 7, n * 9 + 33, n * n - 1] {
                let (x, y) = (i % n, i / n);
                // SAFETY: inside the mapped output.
                let got = unsafe { (data.add((y * 64 + x) * 16) as *const [f32; 4]).read() };
                assert_eq!(got, want, "round {round} texel {x},{y}");
            }
        }
    }

    /// A depth texture with stencil (the console's 16-bit depth, the
    /// host's 24-bit) written through unordered views of its planes, then
    /// read as depth and stencil: the 16-bit depth arrives in 24 bits as
    /// the host converts the same value, the stencil as written. Rendered
    /// as depth (a clear), then used through the unordered view: back in
    /// 16 bits.
    #[test]
    fn a_depth_texture_written_through_unordered_views_reads_as_depth() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let address = title_memory(1 << 22);
        let depth = texture_at_address(address, &texture_words(64, 64, 118, 0, 2));
        let copy = crate::resource::host_copy(depth).expect("a host depth texture");
        let planes = [
            View::Unordered(depth as usize, 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0])),
            View::Unordered(depth as usize, 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0])),
        ];
        slots.make(0, &planes[0]);
        slots.make(1, &planes[1]);
        let reads = [
            View::Shader(depth as usize, Some([119, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])),
            View::Shader(depth as usize, Some([120, 4, 0x1688, 0, 0, 1, 1, 0, 0, 0])),
        ];
        slots.make(2, &reads[0]);
        slots.make(3, &reads[1]);
        let write = compile_cs(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
RWTexture2D<float> d : register(u0);
RWTexture2D<uint> s : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    d[id.xy] = (1 + id.x + 64 * id.y) * (1.0 / 65535.0);
    s[id.xy] = (id.x ^ id.y) & 0xff;
}
"#,
        );
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2)), UAV(u1)"
Texture2D<float> d : register(t0);
Texture2D<uint2> s : register(t1);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store2((id.y * 64 + id.x) * 8, uint2(asuint(d.Load(int3(id.xy, 0))), s.Load(int3(id.xy, 0)).g));
}
"#,
        );
        let (out, data) = host.shared_buffer(64 * 64 * 8).unwrap();
        // The depth texture's planes as bytes, read back after.
        // SAFETY: a live resource.
        let dd = unsafe { copy.texture.GetDesc() };
        let mut f = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let mut total = 0;
        // SAFETY: outputs for two subresources.
        unsafe { host.device.GetCopyableFootprints(&dd, 0, 2, 0, Some(f.as_mut_ptr()), None, None, Some(&mut total)) };
        let (raw, raw_data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        for v in &planes {
            readable(&list, &used(ResourceType::Uav), v);
        }
        run(&list, &slots, &write, 0, None, (8, 8));
        for v in &reads {
            readable(&list, &used(ResourceType::Srv), v);
        }
        assert!(list.take_rebind(), "the conversion ran in the list");
        run(&list, &slots, &read, 2, Some(&out), (8, 8));
        crate::list::ensure_state(&list, depth, 0x800);
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let to = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&raw) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        // SAFETY: an open list, the texture in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(f[0]), 0, 0, 0, &at(&copy.texture, 0), None);
            list.list.CopyTextureRegion(&to(f[1]), 0, 0, 0, &at(&copy.texture, 1), None);
        }
        list.close();
        host.execute(&[&list]);
        let d24 = |u: u32| u * 256 + (u * 255 + 32767) / 65535;
        for (x, y) in [(0usize, 0usize), (5, 0), (63, 2), (17, 40), (63, 63)] {
            let k = (1 + x + 64 * y) as u32;
            // SAFETY: inside the mapped buffers.
            let [depth_bits, stencil] = unsafe { (data.add((y * 64 + x) * 8) as *const [u32; 2]).read() };
            let got = f32::from_bits(depth_bits);
            assert!((got - k as f32 / 65535.0).abs() < 1e-7, "depth at {x},{y}: {got}");
            assert_eq!(stencil, ((x ^ y) & 0xff) as u32, "stencil at {x},{y}");
            // SAFETY: inside the mapped buffer, by the footprints.
            let (word, byte) = unsafe {
                (
                    (raw_data.add(f[0].Offset as usize + y * f[0].Footprint.RowPitch as usize + x * 4) as *const u32).read(),
                    raw_data.add(f[1].Offset as usize + y * f[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(word & 0xff_ffff, d24(k), "24-bit depth at {x},{y}");
            assert_eq!(u32::from(byte), ((x ^ y) & 0xff) as u32);
        }
        // Rendered as depth (a clear of both planes), then used through
        // the unordered view: copied to the twin, converted back.
        list.reset();
        crate::list::ensure_state(&list, depth, 0x10);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let handle = host.dsv(&copy.texture, Some(&dsv));
        // SAFETY: an open list, the texture in the depth write state.
        unsafe {
            list.list.ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5, 0x11, None)
        };
        readable(&list, &used(ResourceType::Uav), &planes[0]);
        let twin = copy.twin.get().and_then(Option::as_ref).expect("a twin");
        let mut g = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        for (p, plane) in twin.planes.iter().enumerate() {
            // SAFETY: a live resource.
            let pd = unsafe { plane.texture.GetDesc() };
            // SAFETY: an output for one subresource.
            unsafe { host.device.GetCopyableFootprints(&pd, 0, 1, f[p].Offset, Some(&mut g[p]), None, None, None) };
            if let Some(b) = list.move_texture(plane, 0x800) {
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[b]) };
            }
        }
        // SAFETY: an open list, the twin's planes in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(g[0]), 0, 0, 0, &at(&twin.planes[0].texture, 0), None);
            list.list.CopyTextureRegion(&to(g[1]), 0, 0, 0, &at(&twin.planes[1].texture, 0), None);
        }
        list.close();
        host.execute(&[&list]);
        for (x, y) in [(0usize, 0usize), (63, 63)] {
            // SAFETY: inside the mapped buffer, by the footprints.
            let (half, byte) = unsafe {
                (
                    (raw_data.add(g[0].Offset as usize + y * g[0].Footprint.RowPitch as usize + x * 2) as *const u16).read(),
                    raw_data.add(g[1].Offset as usize + y * g[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(half, 32768, "16-bit depth at {x},{y}");
            assert_eq!(byte, 0x11, "stencil at {x},{y}");
        }
    }


    /// Unordered-access-only 1D and 3D title textures (the colour grading
    /// bakes: a 4096 R32_FLOAT curve, a 64^3 RGBA16F table) keep what a
    /// dispatch writes through their unordered views, whole (a 3D view of
    /// every depth slice, -1), and a shader view reads it.
    #[test]
    fn unordered_1d_and_3d_textures_keep_their_writes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Description words (dimension, width, height, depth | mips << 16,
        // format), the unordered view's and shader view's words, the
        // shaders' bodies, groups, a texel to check and its value.
        type Case = ([u32; 5], [u32; 10], [u32; 10], &'static str, &'static str, (u32, u32), [f32; 4]);
        let cases: [Case; 2] = [
            (
                [2, 4096, 1, 1 | 1 << 16, 41],
                [41, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [41, 2, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture1D<float> o : register(u0); void put(uint3 p) { if (p.y == 0 && p.z == 0) o[p.x] = p.x * 0.25; }",
                "Texture1D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int2(p.y * 64 + p.x, 0)); }",
                (512, 1),
                [(2 * 64 + 7) as f32 * 0.25, 0.0, 0.0, 1.0],
            ),
            (
                [4, 64, 64, 64 | 1 << 16, 10],
                [10, 8, 0, 0, u32::MAX, 0, 0, 0, 0, 0],
                [10, 8, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture3D<float4> o : register(u0); void put(uint3 p) { o[p] = float4(p, 1) / 64; }",
                "Texture3D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int4(p.x, p.y, 40, 0)); }",
                (8, 8),
                [7.0 / 64.0, 2.0 / 64.0, 40.0 / 64.0, 1.0 / 64.0],
            ),
        ];
        for (round, (d, uav, srv, put, get, groups, want)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 22);
            let words = [d[0], 0, 0, 0, d[1], 0, d[2], d[3], d[4], 1, 0, 0, 4, 0];
            let texture = texture_at_address(address, &words);
            let unordered = View::Unordered(texture as usize, 0, Some(uav));
            let shader = View::Shader(texture as usize, Some(srv));
            slots.make(0, &unordered);
            slots.make(1, &shader);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ for (uint z = 0; z < 64; z++) put(uint3(id.xy, z)); }}"
            ));
            let read = compile_cs(&format!(
                "#define RS \"DescriptorTable(SRV(t0)), UAV(u1)\"\n{get}\nRWByteAddressBuffer o : register(u1);\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ o.Store4((id.y * 64 + id.x) * 16, asuint(get(id))); }}"
            ));
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Uav), &unordered);
            run(&list, &slots, &write, 0, None, groups);
            readable(&list, &used(ResourceType::Srv), &shader);
            run(&list, &slots, &read, 1, Some(&out), (8, 8));
            list.close();
            host.execute(&[&list]);
            // SAFETY: inside the mapped output: texel (7, 2) of the read.
            let got = unsafe { (data.add((2 * 64 + 7) * 16) as *const [f32; 4]).read() };
            assert_eq!(got, want, "round {round}");
        }
    }

    /// A structured view the title places off its buffer's elements (the
    /// culling constants at a ring position) is made over a copy of its own
    /// bytes; one on an element views the buffer's copy.
    #[test]
    fn structured_views_off_their_buffers_elements_get_their_own_copy() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let src = buffer(1 << 16);
        let (bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let start = bytes as u64;
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Four 456-byte elements placed at `address` (resource and view
        // descriptions as the title passes them to device 0x230).
        let view = |address: u64| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let width = |address: u64| {
            let made = crate::descriptor::host::make_view(&view(address), at).expect("a view");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        assert_eq!(width(start + 456), 1 << 16, "on an element: the buffer's copy");
        assert_eq!(width(start + 100), (4 * 456u64).next_multiple_of(256), "off: its own bytes");
        let (copy, address, len) = crate::descriptor::host::view_copy(&view(start + 100)).unwrap();
        assert_eq!((copy.range(), address, len), ((start + 100, 4 * 456), start + 100, 4 * 456));
        drop(copy);
        // A slot made from the view holds its copy: after more than a
        // thousand other views' copies, draws still find (and refresh) the
        // copy the slot shows; rewritten, the slot lets it go.
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let slots = crate::descriptor::heap(heap).host().expect("a host view heap");
        slots.make_at(0, &view(start + 100));
        let shown = std::sync::Arc::downgrade(&crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0);
        for k in 1..1200 {
            if (100 + 4 * k) % 456 != 0 {
                crate::descriptor::host::view_copy(&view(start + 100 + 4 * k)).unwrap();
            }
        }
        let found = crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0;
        assert!(shown.upgrade().is_some_and(|s| std::sync::Arc::ptr_eq(&s, &found)), "the slot's copy");
        drop(found);
        slots.make_null(0);
        assert!(shown.upgrade().is_none(), "released with the slot");
    }
}
