//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::draw::validate_root_buffer;
use crate::host::host;
use crate::pipeline::host::RootBuffer;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    image_data: OnceLock<ImageData>,
    title_memory: Mutex<Option<D3D12_GPU_DESCRIPTOR_HANDLE>>,
}

/// Image dimensions and base mip at the slot indices used by the shader.
/// The mapped upload buffer also stays live in any list that binds it.
struct ImageData {
    buffer: ID3D12Resource,
    mapped: usize,
    handle: D3D12_GPU_DESCRIPTOR_HANDLE,
}

fn table_in_heap(
    desc: D3D12_DESCRIPTOR_HEAP_DESC,
    start: u64,
    handle: u64,
    step: u32,
    count: u32,
) -> bool {
    if desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        || !desc
            .Flags
            .contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
        || start == 0
        || step == 0
        || count == 0
    {
        return false;
    }
    let step = u64::from(step);
    handle.checked_sub(start).is_some_and(|offset| {
        offset.is_multiple_of(step)
            && (offset / step)
                .checked_add(u64::from(count))
                .is_some_and(|end| end <= u64::from(desc.NumDescriptors))
    })
}

pub(crate) fn valid_view_table(
    heap: &ID3D12DescriptorHeap,
    handle: D3D12_GPU_DESCRIPTOR_HANDLE,
    step: u32,
    count: u32,
) -> bool {
    // SAFETY: a live descriptor heap.
    let (desc, start) = unsafe { (heap.GetDesc(), heap.GetGPUDescriptorHandleForHeapStart()) };
    table_in_heap(desc, start.ptr, handle.ptr, step, count)
}

pub(crate) fn valid_raw_srv(buffer: &ID3D12Resource, size: u64) -> bool {
    // SAFETY: a live resource.
    let (base, desc) = unsafe { (buffer.GetGPUVirtualAddress(), buffer.GetDesc()) };
    validate_root_buffer(RootBuffer::Srv, base, base, desc, size).is_ok()
}

pub(crate) fn cbv_size(size: u32) -> Option<u32> {
    (size > 0 && size <= 65536).then(|| size.next_multiple_of(256))
}

impl ImageData {
    fn set(&self, index: u32, words: [u32; 4]) {
        // SAFETY: a persistent mapping with four dwords per slot. Callers
        // serialize writes with the same lock as descriptor mirroring.
        unsafe {
            (self.mapped as *mut [u32; 4])
                .add(index as usize)
                .write(words)
        };
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - MIN_DYNAMIC, MIN_DYNAMIC)
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
            image_data: OnceLock::new(),
            title_memory: Mutex::new(None),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..self.dynamic as usize).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn dynamic_capacity(&self) -> u32 {
        self.dynamic
    }

    pub(crate) fn valid_table(&self, handle: D3D12_GPU_DESCRIPTOR_HANDLE, count: u32) -> bool {
        valid_view_table(&self.heap, handle, self.step, count)
    }

    pub(crate) fn title_memory(&self) -> Option<D3D12_GPU_DESCRIPTOR_HANDLE> {
        use victory_shader::emit::TITLE_MEMORY_VIEWS;

        let mut handle = self.title_memory.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(handle) = *handle {
            return Some(handle);
        }
        let mirror = crate::mirror::get()?;
        let slot = self.dynamic(TITLE_MEMORY_VIEWS)?;
        // SAFETY: this is a live shader-visible heap with the reserved slots.
        let base = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        let start = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + u64::from(self.step) * u64::from(slot),
        };
        if !self.valid_table(start, TITLE_MEMORY_VIEWS)
            || mirror.read_views(self.cpu(slot), self.step).is_none()
        {
            self.release(slot, TITLE_MEMORY_VIEWS);
            return None;
        }
        *handle = Some(start);
        Some(start)
    }

    /// A bounded raw view of the image fields. Allocate it on first use,
    /// seeding it while the mirrored descriptor slots are locked.
    pub(crate) fn image_data(&self) -> Option<(ID3D12Resource, D3D12_GPU_DESCRIPTOR_HANDLE)> {
        if let Some(data) = self.image_data.get() {
            return Some((data.buffer.clone(), data.handle));
        }
        if self.sampler || self.count == 0 {
            return None;
        }
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(data) = self.image_data.get() {
            return Some((data.buffer.clone(), data.handle));
        }
        let host = host()?;
        let elements = self.count.checked_mul(4)?;
        let (buffer, mapped) = host.upload_buffer(u64::from(elements) * 4)?;
        if mapped.is_null() || !valid_raw_srv(&buffer, u64::from(elements) * 4) {
            return None;
        }
        let slot = self.dynamic(1)?;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a live shader-visible descriptor heap.
        let gpu = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        let handle = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu + u64::from(self.step) * u64::from(slot),
        };
        if !self.valid_table(handle, 1) {
            self.release(slot, 1);
            return None;
        }
        // SAFETY: checked raw buffer and reserved slot in a live view heap.
        unsafe {
            host.device
                .CreateShaderResourceView(&buffer, Some(&desc), self.cpu(slot))
        };
        let made_data = ImageData {
            buffer,
            mapped: mapped as usize,
            handle,
        };
        for (index, &key) in made.iter().enumerate() {
            let words = if key == 0 {
                [0; 4]
            } else {
                image_words_at(key as u32, (key >> 32) as u32)
            };
            made_data.set(index as u32, words);
        }
        let result = (made_data.buffer.clone(), handle);
        let _ = self.image_data.set(made_data);
        Some(result)
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) -> bool {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        let valid = kept.is_some();
        self.keep(index, kept.flatten());
        valid
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            *slot = resource;
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) -> bool {
        let (Some(size), Some(host)) = (cbv_size(size), host()) else {
            return false;
        };
        if gpu == 0
            || !gpu.is_multiple_of(256)
            || self.sampler
            || index >= self.count + self.dynamic
        {
            return false;
        }
        self.keep(index, None);
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size,
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
        true
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
                if let Some(data) = self.image_data.get() {
                    data.set(index, [0; 4]);
                }
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            let valid = kept.is_some();
            made[index as usize] = if valid { key } else { 0 };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
            if let Some(data) = self.image_data.get() {
                data.set(index, image_words_at(i, generation));
            }
            if for_draw && !valid {
                return None;
            }
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

pub(crate) fn view_kind(view: &View) -> victory_shader::ResourceType {
    use victory_shader::ResourceType::*;
    match view {
        View::Constant(..) => Cbv,
        View::Shader(..) | View::Placed(false, ..) => Srv,
        View::Unordered(..) | View::Placed(true, ..) => Uav,
        View::Sampler(..) | View::ConsoleSampler(..) => Sampler,
    }
}

/// Checks the DXIL resource kind and element type against the view that
/// will supply a statically addressed descriptor.
pub(crate) fn validate_binding(
    view: &View,
    (properties, element): (u32, u32),
) -> Result<(), &'static str> {
    if let Some(placed) = placed_texture(view) {
        return validate_binding(&placed, (properties, element));
    }
    let kind = properties & 0xff;
    let writable = properties & 0x1000 != 0;
    let shape = match view {
        View::Constant(..) if kind == 13 && !writable => return Ok(()),
        View::Sampler(..) | View::ConsoleSampler(..) if kind == 14 => return Ok(()),
        View::Shader(resource, desc) if !writable && *resource != 0 => desc.as_ref().map(srv_shape),
        View::Unordered(resource, counter, desc) if writable && *resource != 0 && *counter == 0 => {
            desc.as_ref().map(uav_shape)
        }
        View::Placed(uav, _, desc) if *uav == writable => {
            let words = desc[..10].try_into().expect("ten descriptor words");
            Some(if *uav {
                uav_shape(words)
            } else {
                srv_shape(words)
            })
        }
        _ => return Err("descriptor class does not match the shader"),
    };
    let Some((actual, mut format, stride)) = shape else {
        // Default views get their format and dimension from a live resource.
        let resource = match view {
            View::Shader(r, None) | View::Unordered(r, _, None) => *r,
            _ => return Err("no descriptor shape"),
        };
        let texture = crate::resource::host_resource(resource as crate::object::This)
            .ok_or("no resource for default descriptor")?;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let actual = match (d.Dimension, d.DepthOrArraySize > 1, d.SampleDesc.Count > 1) {
            (D3D12_RESOURCE_DIMENSION_TEXTURE1D, false, _) => 1,
            (D3D12_RESOURCE_DIMENSION_TEXTURE1D, true, _) => 6,
            (D3D12_RESOURCE_DIMENSION_TEXTURE2D, false, false) => 2,
            (D3D12_RESOURCE_DIMENSION_TEXTURE2D, false, true) => 3,
            (D3D12_RESOURCE_DIMENSION_TEXTURE2D, true, false) => 7,
            (D3D12_RESOURCE_DIMENSION_TEXTURE2D, true, true) => 8,
            (D3D12_RESOURCE_DIMENSION_TEXTURE3D, _, _) => 4,
            _ => 0,
        };
        return check_shape(kind, element, actual, d.Format, 0);
    };
    if let View::Shader(resource, _) | View::Unordered(resource, _, _) = view
        && let Some(texture) = crate::resource::host_resource(*resource as crate::object::This)
    {
        // SAFETY: a live texture.
        let resource = unsafe { texture.GetDesc() };
        if !texture_view_fits(resource, actual) || !fits(texture, Some(&mut format), writable) {
            return Err("host resource does not permit the descriptor");
        }
    }
    check_shape(kind, element, actual, format, stride)
}

fn check_shape(
    kind: u32,
    element: u32,
    actual: u32,
    format: DXGI_FORMAT,
    stride: u32,
) -> Result<(), &'static str> {
    if actual != kind {
        return Err("descriptor shape does not match the shader");
    }
    match kind {
        11 => Ok(()), // Raw buffer.
        12 if element == stride => Ok(()),
        12 => Err("structured descriptor stride does not match the shader"),
        1..=10 if component_matches(format, element & 0xff) => Ok(()),
        _ => Err("typed descriptor format does not match the shader"),
    }
}

fn component_matches(format: DXGI_FORMAT, component: u32) -> bool {
    let class = match format {
        DXGI_FORMAT_R32G32B32A32_UINT
        | DXGI_FORMAT_R32G32B32_UINT
        | DXGI_FORMAT_R16G16B16A16_UINT
        | DXGI_FORMAT_R32G32_UINT
        | DXGI_FORMAT_R10G10B10A2_UINT
        | DXGI_FORMAT_R8G8B8A8_UINT
        | DXGI_FORMAT_R16G16_UINT
        | DXGI_FORMAT_R32_UINT
        | DXGI_FORMAT_R8G8_UINT
        | DXGI_FORMAT_R16_UINT
        | DXGI_FORMAT_R8_UINT
        | DXGI_FORMAT_X24_TYPELESS_G8_UINT
        | DXGI_FORMAT_X32_TYPELESS_G8X24_UINT => 5,
        DXGI_FORMAT_R32G32B32A32_SINT
        | DXGI_FORMAT_R32G32B32_SINT
        | DXGI_FORMAT_R16G16B16A16_SINT
        | DXGI_FORMAT_R32G32_SINT
        | DXGI_FORMAT_R8G8B8A8_SINT
        | DXGI_FORMAT_R16G16_SINT
        | DXGI_FORMAT_R32_SINT
        | DXGI_FORMAT_R8G8_SINT
        | DXGI_FORMAT_R16_SINT
        | DXGI_FORMAT_R8_SINT => 4,
        _ => 9,
    };
    // Float instructions also accept the normalized float view formats.
    match component {
        2 | 4 => class == 4,
        3 | 5 => class == 5,
        8..=10 | 11..=16 => class == 9 && format != DXGI_FORMAT_UNKNOWN,
        _ => false,
    }
}

fn srv_shape(words: &[u32; 10]) -> (u32, DXGI_FORMAT, u32) {
    let d = srv_desc(words);
    if d.ViewDimension == D3D12_SRV_DIMENSION_BUFFER {
        // SAFETY: selected by the view dimension.
        let b = unsafe { d.Anonymous.Buffer };
        return buffer_shape(
            d.Format,
            b.StructureByteStride,
            b.Flags == D3D12_BUFFER_SRV_FLAG_RAW,
        );
    }
    let kind = match d.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE1D => 1,
        D3D12_SRV_DIMENSION_TEXTURE2D => 2,
        D3D12_SRV_DIMENSION_TEXTURE2DMS => 3,
        D3D12_SRV_DIMENSION_TEXTURE3D => 4,
        D3D12_SRV_DIMENSION_TEXTURECUBE => 5,
        D3D12_SRV_DIMENSION_TEXTURE1DARRAY => 6,
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => 7,
        D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => 8,
        D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => 9,
        _ => 0,
    };
    (kind, d.Format, 0)
}

fn uav_shape(words: &[u32; 10]) -> (u32, DXGI_FORMAT, u32) {
    let d = uav_desc(words);
    if d.ViewDimension == D3D12_UAV_DIMENSION_BUFFER {
        // SAFETY: selected by the view dimension.
        let b = unsafe { d.Anonymous.Buffer };
        return buffer_shape(
            d.Format,
            b.StructureByteStride,
            b.Flags == D3D12_BUFFER_UAV_FLAG_RAW,
        );
    }
    let kind = match d.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE1D => 1,
        D3D12_UAV_DIMENSION_TEXTURE2D => 2,
        D3D12_UAV_DIMENSION_TEXTURE3D => 4,
        D3D12_UAV_DIMENSION_TEXTURE1DARRAY => 6,
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => 7,
        _ => 0,
    };
    (kind, d.Format, 0)
}

fn buffer_shape(format: DXGI_FORMAT, stride: u32, raw: bool) -> (u32, DXGI_FORMAT, u32) {
    let kind = match (format, stride, raw) {
        (DXGI_FORMAT_R32_TYPELESS, 0, true) => 11,
        (DXGI_FORMAT_UNKNOWN, 1.., false) => 12,
        (f, 0, false) if f != DXGI_FORMAT_UNKNOWN && f != DXGI_FORMAT_R32_TYPELESS => 10,
        _ => 0,
    };
    (kind, format, stride)
}

fn texture_view_fits(resource: D3D12_RESOURCE_DESC, kind: u32) -> bool {
    let dimension = match kind {
        1 | 6 => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
        2 | 3 | 5 | 7..=9 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        4 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        _ => return false,
    };
    let multisampled = matches!(kind, 3 | 8);
    resource.Dimension == dimension
        && (resource.SampleDesc.Count > 1) == multisampled
        && (!matches!(kind, 5 | 9) || resource.DepthOrArraySize >= 6)
}

fn image_words_at(index: u32, generation: u32) -> [u32; 4] {
    let views = super::VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    views
        .list
        .get(index as usize)
        .filter(|entry| entry.generation == generation)
        .map_or([0; 4], |entry| entry.image_words)
}

/// The four dwords the shader's descriptor data holds per heap slot: for an
/// image the packed dimensions and the first mip, for a buffer view (dword
/// 2) its element stride, the field a console buffer descriptor keeps in
/// dword 1 bits [29:16].
pub(super) fn descriptor_words(view: &View) -> [u32; 4] {
    if let Some([size, mip]) = image_words(view) {
        return [size, mip, 0, 0];
    }
    [0, 0, buffer_stride(view).unwrap_or(0), 0]
}

/// Bytes per element of a buffer view: its structure stride, 4 for a raw
/// view, the format's size for a typed one.
fn buffer_stride(view: &View) -> Option<u32> {
    let srv = |words: &[u32; 10]| element_size(&srv_desc(words));
    let uav = |words: &[u32; 10]| {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            Some(u64::from(buffer.StructureByteStride))
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            Some(4)
        } else {
            crate::resource::element(desc.Format.0 as u32).map(|(bytes, _)| bytes)
        }
    };
    let bytes = match view {
        View::Shader(r, Some(desc)) => {
            crate::resource::buffer_bytes(*r as crate::object::This)?;
            srv(desc)
        }
        View::Placed(false, resource, desc) if resource[0] == 1 => srv(desc[..10].try_into().ok()?),
        View::Unordered(r, _, Some(desc)) => {
            crate::resource::buffer_bytes(*r as crate::object::This)?;
            uav(desc)?
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => uav(desc[..10].try_into().ok()?)?,
        _ => return None,
    };
    u32::try_from(bytes).ok().filter(|&b| b < 1 << 14)
}

/// RDNA2 stores the dimensions of mip zero and the view's first mip.
pub(super) fn image_words(view: &View) -> Option<[u32; 2]> {
    let placed_size = |resource: &[u32; 14]| {
        (2..=4).contains(&resource[0]).then_some((
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
            resource[6],
        ))
    };
    let (size, mip) = match view {
        View::Shader(resource, desc) => (
            crate::resource::texture_dimensions(*resource as crate::object::This)?,
            desc.as_ref().map_or(Some(0), image_srv_mip)?,
        ),
        View::Unordered(resource, _, desc) => (
            crate::resource::texture_dimensions(*resource as crate::object::This)?,
            desc.as_ref().map_or(Some(0), image_uav_mip)?,
        ),
        View::Placed(false, resource, desc) => (
            placed_size(resource)?,
            image_srv_mip(desc[..10].try_into().ok()?)?,
        ),
        View::Placed(true, resource, desc) => (
            placed_size(resource)?,
            image_uav_mip(desc[..10].try_into().ok()?)?,
        ),
        _ => return None,
    };
    let (width, height) = size;
    if width == 0 || width > 1 << 14 || height == 0 || height > 1 << 14 || mip > 15 {
        return None;
    }
    Some([(height - 1) << 16 | (width as u32 - 1), mip])
}

fn image_srv_mip(words: &[u32; 10]) -> Option<u32> {
    let d = srv_desc(words);
    // SAFETY: the view dimension names the union member read here.
    Some(unsafe {
        match d.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE1D => d.Anonymous.Texture1D.MostDetailedMip,
            D3D12_SRV_DIMENSION_TEXTURE1DARRAY => d.Anonymous.Texture1DArray.MostDetailedMip,
            D3D12_SRV_DIMENSION_TEXTURE2D => d.Anonymous.Texture2D.MostDetailedMip,
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => d.Anonymous.Texture2DArray.MostDetailedMip,
            D3D12_SRV_DIMENSION_TEXTURE2DMS | D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => 0,
            D3D12_SRV_DIMENSION_TEXTURE3D => d.Anonymous.Texture3D.MostDetailedMip,
            D3D12_SRV_DIMENSION_TEXTURECUBE => d.Anonymous.TextureCube.MostDetailedMip,
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => d.Anonymous.TextureCubeArray.MostDetailedMip,
            _ => return None,
        }
    })
}

fn image_uav_mip(words: &[u32; 10]) -> Option<u32> {
    let d = uav_desc(words);
    // SAFETY: the view dimension names the union member read here.
    Some(unsafe {
        match d.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE1D => d.Anonymous.Texture1D.MipSlice,
            D3D12_UAV_DIMENSION_TEXTURE1DARRAY => d.Anonymous.Texture1DArray.MipSlice,
            D3D12_UAV_DIMENSION_TEXTURE2D => d.Anonymous.Texture2D.MipSlice,
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => d.Anonymous.Texture2DArray.MipSlice,
            D3D12_UAV_DIMENSION_TEXTURE3D => d.Anonymous.Texture3D.MipSlice,
            _ => return None,
        }
    })
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
        return None;
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base.checked_add(buffer.FirstElement.checked_mul(element)?)?,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    if desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base.checked_add(buffer.FirstElement.checked_mul(element)?)?,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER || address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if !matches!(
        buffer.Flags,
        D3D12_BUFFER_SRV_FLAG_NONE | D3D12_BUFFER_SRV_FLAG_RAW
    ) {
        return None;
    }
    let kind = buffer_shape(
        desc.Format,
        buffer.StructureByteStride,
        buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW,
    )
    .0;
    buffer.FirstElement = buffer_view_start(
        &shadow,
        address,
        len,
        element,
        buffer.FirstElement,
        buffer.NumElements,
        kind,
        false,
    )?;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    if desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER || address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    if buffer.CounterOffsetInBytes != 0
        || !matches!(
            buffer.Flags,
            D3D12_BUFFER_UAV_FLAG_NONE | D3D12_BUFFER_UAV_FLAG_RAW
        )
    {
        return None;
    }
    let kind = buffer_shape(
        desc.Format,
        buffer.StructureByteStride,
        buffer.Flags == D3D12_BUFFER_UAV_FLAG_RAW,
    )
    .0;
    buffer.FirstElement = buffer_view_start(
        &shadow,
        address,
        len,
        element,
        buffer.FirstElement,
        buffer.NumElements,
        kind,
        true,
    )?;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// Rebase a bounded buffer view onto its host resource before calling a
/// descriptor creator. The element interval must fit the title view too.
fn buffer_view_start(
    shadow: &crate::shadow::Shadow,
    title: u64,
    len: u64,
    element: u64,
    first: u64,
    count: u32,
    kind: u32,
    writable: bool,
) -> Option<u64> {
    if title == 0
        || element == 0
        || kind == 0
        || count == 0
        || count > 1 << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP
        || kind == 12 && (!element.is_multiple_of(4) || element > 2048)
        || first.checked_add(u64::from(count))? > len / element
    {
        return None;
    }
    // SAFETY: the shadow owns a live buffer.
    let (gpu, desc) = unsafe {
        (
            shadow.buffer.GetGPUVirtualAddress(),
            shadow.buffer.GetDesc(),
        )
    };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        || gpu == 0
        || writable
            && !desc
                .Flags
                .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        || !writable
            && desc
                .Flags
                .contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        return None;
    }
    let base = shadow.offset(title);
    let offset = base.checked_add(first.checked_mul(element)?)?;
    if !offset.is_multiple_of(element)
        || offset.checked_add(u64::from(count).checked_mul(element)?)? > desc.Width
        || kind == 11
            && !gpu
                .checked_add(offset)?
                .is_multiple_of(u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT))
    {
        return None;
    }
    Some(offset / element)
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            if address == 0 {
                return None;
            }
            let rounded = cbv_size(size)?;
            let shadow = crate::shadow::find(address, u64::from(size))?;
            // SAFETY: the shadow retains the live resource.
            let (base, resource) = unsafe {
                (
                    shadow.buffer.GetGPUVirtualAddress(),
                    shadow.buffer.GetDesc(),
                )
            };
            let gpu = base.checked_add(shadow.offset(address))?;
            validate_root_buffer(RootBuffer::Cbv, gpu, base, resource, u64::from(rounded)).ok()?;
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: rounded,
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // SAFETY: a live texture.
                let resource = unsafe { texture.GetDesc() };
                if desc
                    .as_ref()
                    .is_some_and(|words| !texture_view_fits(resource, srv_shape(words).0))
                {
                    return None;
                }
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, counter, desc) => {
            if *counter != 0 {
                return None;
            }
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            // SAFETY: a live texture.
            let resource = unsafe { texture.GetDesc() };
            if desc
                .as_ref()
                .is_some_and(|words| !texture_view_fits(resource, uav_shape(words).0))
            {
                return None;
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    ..Default::default()
                },
            },
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_data_keeps_mip_zero_dimensions_and_view_mip() {
        let mut resource = [0; 14];
        resource[0] = 3; // Texture2D.
        resource[4] = 257;
        resource[6] = 129;
        let mut srv = [0; 16];
        srv[0] = DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32;
        srv[1] = D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32;
        srv[4] = 3;
        assert_eq!(
            image_words(&View::Placed(false, resource, srv)),
            Some([128 << 16 | 256, 3])
        );

        let mut uav = [0; 16];
        uav[0] = srv[0];
        uav[1] = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
        uav[2] = 5;
        assert_eq!(
            image_words(&View::Placed(true, resource, uav)),
            Some([128 << 16 | 256, 5])
        );
        resource[0] = 1; // Buffer.
        assert_eq!(image_words(&View::Placed(true, resource, uav)), None);
    }

    #[test]
    fn descriptor_words_hold_a_buffer_view_stride() {
        let mut resource = [0; 14];
        resource[0] = 1; // Buffer.
        let mut srv = [0; 16];
        srv[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        srv[7] = 48; // StructureByteStride follows FirstElement and NumElements.
        assert_eq!(
            descriptor_words(&View::Placed(false, resource, srv)),
            [0, 0, 48, 0]
        );
        // Raw.
        srv[7] = 0;
        srv[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        assert_eq!(
            descriptor_words(&View::Placed(false, resource, srv))[2],
            4
        );
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn indexed_views_must_match_the_shader_annotation() {
        let mut resource = [0; 14];
        resource[0] = D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32;
        let mut words = [0; 16];
        words[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        words[6] = 2400;
        words[7] = 416;
        let structured = View::Placed(false, resource, words);
        assert!(validate_binding(&structured, (11, 0)).is_err());
        assert!(validate_binding(&structured, (12, 336)).is_err());
        assert!(validate_binding(&structured, (12, 416)).is_ok());

        words[0] = DXGI_FORMAT_R8_UINT.0 as u32;
        words[1] = D3D12_UAV_DIMENSION_BUFFER.0 as u32;
        words[6] = 393216;
        words[7] = 0;
        let uint = View::Placed(true, resource, words);
        assert!(validate_binding(&uint, (0x100a, 0x409)).is_err());
        assert!(validate_binding(&uint, (0x100a, 0x105)).is_ok());
        assert!(validate_binding(&uint, (10, 0x105)).is_err());
    }

    #[test]
    fn metadata_and_mirror_tables_fit_the_bound_view_heap() {
        let mut desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NumDescriptors: 257,
            NodeMask: 0,
        };
        let start = 0x1000;
        let step = 32;
        assert!(table_in_heap(desc, start, start, step, 1));
        assert!(table_in_heap(desc, start, start + 32, step, 256));
        assert!(!table_in_heap(desc, start, start + 64, step, 256));
        assert!(!table_in_heap(desc, start, start + 1, step, 1));
        assert!(!table_in_heap(desc, start, 0, step, 1));
        assert!(!table_in_heap(desc, start, u64::MAX, 1, 10000));
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        assert!(!table_in_heap(desc, start, start, step, 1));
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        assert!(!table_in_heap(desc, start, start, step, 1));
    }

    #[test]
    fn constant_view_size_is_bounded_and_rounded() {
        assert_eq!(cbv_size(0), None);
        assert_eq!(cbv_size(16), Some(256));
        assert_eq!(cbv_size(65535), Some(65536));
        assert_eq!(cbv_size(65537), None);
        assert_eq!(cbv_size(u32::MAX), None);
    }

    #[test]
    fn texture_resource_cannot_supply_a_raw_buffer_or_another_dimension() {
        let texture = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            DepthOrArraySize: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        assert!(texture_view_fits(texture, 2));
        assert!(!texture_view_fits(texture, 4));
        assert!(!texture_view_fits(texture, 5));
        assert!(!texture_view_fits(texture, 11));
    }
}
