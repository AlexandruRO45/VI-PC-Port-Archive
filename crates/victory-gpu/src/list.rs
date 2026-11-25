//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (Grand Theft Auto VI asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: Grand Theft Auto VI's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Upload>>,
    /// The host lists the calls record into, in order: one, and one more
    /// for every wait the title records after commands ([`wait`]). Shared
    /// with the queues that hold them until their waits hold
    /// (`crate::hold`), so they outlive a list the title releases first.
    #[cfg(windows)]
    hosts: std::sync::Mutex<Vec<std::sync::Arc<crate::host::HostList>>>,
    /// Which of them the calls record into.
    #[cfg(windows)]
    segment: std::sync::atomic::AtomicUsize,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
pub(crate) fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// The host list a title list's calls record into (its current segment),
/// if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let current: *const crate::host::HostList = std::sync::Arc::as_ptr(hosts.get(k)?);
    // SAFETY: the list holds this host list until it is reset or released
    // (an `Arc` does not move it), and the title records into a list from
    // one thread at a time.
    Some(unsafe { &*current })
}

/// A title list's host lists, in order: what a submission of it runs.
#[cfg(windows)]
pub(crate) fn segments(this: This) -> Vec<std::sync::Arc<crate::host::HostList>> {
    if this.is_null() {
        return Vec::new();
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    hosts.iter().take(k + 1).cloned().collect()
}

/// Waits recorded after commands: the title list goes on in a new host
/// list, which runs once the wait holds.
static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Lists reset or released while a queue still held some of their host
/// lists: D3D12 lets a list be reset as soon as it is submitted (Grand
/// Theft Auto VI does, 0-21 times a world frame).
static EARLY_RESETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Cuts and early resets since the last call (the frame line).
pub(crate) fn take_cuts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CUTS.swap(0, Relaxed), EARLY_RESETS.swap(0, Relaxed))
}

/// `Wait32BitValueX(address, comparison, reference, mask, flags)` and
/// `Wait64BitValueX(address, comparison, reference, flags)`: the GPU waits
/// until the value compares true. Recorded at the start of the current host
/// list, or, after commands, at the start of a new one the rest of the
/// title list records into (the queue runs each once its waits hold,
/// `crate::hold`); bottom-of-pipe writes recorded before it land when the
/// part before it has run.
extern "system" fn wait32(this: This, address: u64, comparison: u32, reference: u32, mask: u32, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 4, comparison, reference: u64::from(reference), mask: u64::from(mask) });
}
pub(crate) extern "system" fn wait64(this: This, address: u64, comparison: u32, reference: u64, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 8, comparison, reference, mask: u64::MAX });
}

fn wait(this: This, wait: crate::hold::ValueWait) {
    #[cfg(windows)]
    {
        let Some(current) = host_list(this) else { return };
        if !current.is_empty() {
            let Some(next) = next_segment(this) else { return };
            CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            current.close();
            // SAFETY: the title passes one of our lists.
            let list = unsafe { &*this.cast::<List>() };
            list.state.lock().unwrap_or_else(|e| e.into_inner()).cut(next);
            next.defer_wait(wait);
            return;
        }
        current.defer_wait(wait);
    }
    #[cfg(not(windows))]
    let _ = (this, wait);
}

/// Moves a title list's recording on to its next host list (made, or one
/// it held since an earlier recording, reset then): that host list.
#[cfg(windows)]
fn next_segment<'a>(this: This) -> Option<&'a crate::host::HostList> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed) + 1;
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    if hosts.len() <= k {
        let made = crate::host::host()?.new_list().ok()?;
        hosts.push(std::sync::Arc::new(made));
    }
    list.segment.store(k, std::sync::atomic::Ordering::Relaxed);
    let next: *const crate::host::HostList = std::sync::Arc::as_ptr(&hosts[k]);
    drop(hosts);
    // SAFETY: as in `host_list`.
    Some(unsafe { &*next })
}

/// After the title list's last submission: its host lists start again,
/// and a host list a queue still holds (`crate::hold`) is left to the
/// queue and replaced. The last one left so, which should keep what the
/// title list owned.
#[cfg(windows)]
fn restart(this: This) -> Option<std::sync::Arc<crate::host::HostList>> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let used = list.segment.swap(0, std::sync::atomic::Ordering::Relaxed);
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let mut left = None;
    for (k, h) in hosts.iter_mut().enumerate() {
        if std::sync::Arc::strong_count(h) > 1 {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(segment = k, "a list reset while a queue still holds part of it: that part keeps its own host list");
            let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop();
            match spare.map(Ok).or_else(|| crate::host::host().map(|host| host.new_list())) {
                Some(Ok(fresh)) => left = Some(std::mem::replace(h, std::sync::Arc::new(fresh))),
                _ => continue,
            }
        } else if k <= used {
            h.reset();
        }
    }
    left
}

/// Host lists [`restart`] left to a queue, back once they ran: the next
/// ones it hands out (a host list and its allocators are not free to make;
/// Grand Theft Auto VI resets up to 21 lists a frame while they are held).
#[cfg(windows)]
static SPARE: std::sync::Mutex<Vec<crate::host::HostList>> = std::sync::Mutex::new(Vec::new());

/// A host list a queue has run: one no title list holds any more is reset
/// and kept for [`restart`].
#[cfg(windows)]
pub(crate) fn recycle(list: std::sync::Arc<crate::host::HostList>) {
    const KEPT: usize = 64;
    if let Ok(list) = std::sync::Arc::try_unwrap(list) {
        list.reset();
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spare.len() < KEPT {
            spare.push(list);
        }
    }
}

/// What a title list owned that the host list a queue still holds, if any,
/// may read: its upload memory (list 0x4a0) and its draw state's dynamic
/// descriptor slots and title-memory page table. Freed once that host list
/// (the list's last held one, so the last to run) is gone.
#[cfg(windows)]
fn leave(to: Option<std::sync::Arc<crate::host::HostList>>, uploads: Vec<Upload>, state: crate::draw::State) {
    struct Kept(#[allow(dead_code)] Vec<Upload>, #[allow(dead_code)] crate::draw::State);
    // SAFETY: only dropped, with the host list, after it has run.
    unsafe impl Send for Kept {}
    if let Some(held) = to {
        let kept = Kept(uploads, state);
        held.defer_after(move || {
            let _ = &kept;
        });
    }
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        hosts: std::sync::Mutex::new(
            crate::host::host()
                .and_then(|h| match h.new_list() {
                    Ok(list) => Some(std::sync::Arc::new(list)),
                    Err(e) => {
                        tracing::warn!(error = %e, "no host command list");
                        None
                    }
                })
                .into_iter()
                .collect(),
        ),
        #[cfg(windows)]
        segment: Default::default(),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
    });
    let hr = object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    if hr == S_OK {
        crate::pipeline::bound(pipeline);
    }
    hr
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    // A queue may still hold some of its host lists; they are its own, and
    // the last of them takes what the list owned.
    #[cfg(windows)]
    if !this.is_null() && object::refs(this) == 1 {
        // SAFETY: the title passes one of our lists.
        let list = unsafe { &*this.cast::<List>() };
        let held = list.hosts.lock().unwrap_or_else(|e| e.into_inner()).iter().rfind(|h| std::sync::Arc::strong_count(h) > 1).cloned();
        if held.is_some() {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!("a list released while a queue still holds part of it: that part runs on its own");
            let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
            let state = std::mem::take(&mut *list.state.lock().unwrap_or_else(|e| e.into_inner()));
            leave(held, uploads, state);
        }
    }
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // A queue may still hold what the list recorded (`restart`, `leave`).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    let uploads = std::mem::take(&mut *uploads.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(windows)]
    {
        let held = restart(this);
        // SAFETY: the title passes one of our lists.
        let mut state = unsafe { &(*this.cast::<List>()).state }.lock().unwrap_or_else(|e| e.into_inner());
        let old = std::mem::take(&mut *state);
        state.reset(pipeline);
        drop(state);
        leave(held, uploads, old);
    }
    #[cfg(not(windows))]
    drop(uploads);
    crate::pipeline::bound(pipeline);
    #[cfg(not(windows))]
    let _ = this;
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(&texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("barrier kinds {:?}", barriers.iter().map(|b| b.kind).collect::<Vec<_>>()));
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            host.push(UAV_BARRIER);
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(&texture, state);
        let barrier = crate::host::transition(&texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Copies `len` bytes between guest buffers: on the GPU when an earlier
/// command in `list` produced the source bytes on the GPU, else at submission
/// in host memory.
#[cfg(windows)]
fn copy_buffer_bytes(list: &crate::host::HostList, to: *mut u8, from: *const u8, len: u64) {
    let Ok(bytes) = usize::try_from(len) else {
        return;
    };
    if bytes == 0 {
        return;
    }
    let (dst_addr, src_addr) = (to as u64, from as u64);
    let src_range = src_addr as usize..(src_addr as usize).saturating_add(bytes);
    let gpu_src = list
        .written_shadow(std::iter::once(src_range))
        .and_then(|s| {
            let off = s.offset_for_range(src_addr, len)?;
            Some((s, off))
        });
    if let Some((src_shadow, src_off)) = gpu_src
        && let Some(dst_shadow) = crate::shadow::find(dst_addr, len)
        && let Some(dst_off) = dst_shadow.offset_for_range(dst_addr, len)
        && record_gpu_buffer_copy(list, &dst_shadow, dst_off, &src_shadow, src_off, len)
    {
        src_shadow.refresh(list, src_addr, len);
        dst_shadow.refresh(list, dst_addr, len);
        dst_shadow.gpu_writes(list, dst_addr, len, false);
        return;
    }
    list.defer_copy(to, from, bytes);
}

/// Records a GPU copy from `src` into `dst`, transitioning any buffer currently
/// in explicit `UNORDERED_ACCESS` and leaving copied `Shadow` buffers in
/// `UNORDERED_ACCESS`.
#[cfg(windows)]
fn record_gpu_buffer_copy(
    list: &crate::host::HostList,
    dst: &std::sync::Arc<crate::shadow::Shadow>,
    dst_off: u64,
    src: &std::sync::Arc<crate::shadow::Shadow>,
    src_off: u64,
    len: u64,
) -> bool {
    use crate::host::transition;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    };
    use windows::core::Interface;

    if std::ptr::eq(dst.buffer.as_raw(), src.buffer.as_raw()) {
        let Some((temp, _)) = crate::host::host().and_then(|h| h.shared_buffer(len)) else {
            return false;
        };
        list.keep(&temp);
        // SAFETY: open list; `src`/`dst` and `temp` stay alive through execution.
        unsafe {
            if list.is_unordered(&src.buffer) {
                list.list.ResourceBarrier(&[transition(
                    &src.buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            list.list
                .CopyBufferRegion(&temp, 0, &src.buffer, src_off, len);
            list.list.ResourceBarrier(&[
                transition(
                    &src.buffer,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
                transition(
                    &temp,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ]);
            list.list
                .CopyBufferRegion(&dst.buffer, dst_off, &temp, 0, len);
            list.list.ResourceBarrier(&[transition(
                &dst.buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        list.set_unordered(&dst.buffer);
        return true;
    }

    let src_was_uav = list.is_unordered(&src.buffer);
    let dst_was_uav = list.is_unordered(&dst.buffer);
    // SAFETY: open list; `src` and `dst` are kept alive by `refresh`.
    unsafe {
        let mut before = Vec::with_capacity(2);
        if src_was_uav {
            before.push(transition(
                &src.buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ));
        }
        if dst_was_uav {
            before.push(transition(
                &dst.buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ));
        }
        if !before.is_empty() {
            list.list.ResourceBarrier(&before);
        }
        list.list
            .CopyBufferRegion(&dst.buffer, dst_off, &src.buffer, src_off, len);
        list.list.ResourceBarrier(&[
            transition(
                &src.buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            transition(
                &dst.buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ]);
    }
    list.set_unordered(&src.buffer);
    list.set_unordered(&dst.buffer);
    true
}

/// Buffer to buffer: on the GPU after a same-list GPU write, else at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        copy_buffer_bytes(list, to, from, len);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if crate::texture::title_written(&from) {
                crate::texture::upload(list, src, &from);
            }
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::texture::host_writes(list, dst);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(&to, &from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            copy_buffer_bytes(list, to, from, a.min(b));
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// Selects the host buffer and offset for a placed texture copy source:
/// the GPU-written [`crate::shadow::Shadow`] buffer (rebased onto a 512-byte
/// aligned buffer on the GPU if `offset` is unaligned) when an earlier command
/// in `list` produced the read texels on the GPU, else an upload snapshot
/// filled from title memory at submission. Also returns any shadow buffer that
/// must be transitioned back to `UNORDERED_ACCESS` after `CopyTextureRegion`.
#[cfg(windows)]
fn placed_copy_source(
    list: &crate::host::HostList,
    resource: This,
    offset: u64,
    bytes: u64,
    read: &std::ops::Range<u64>,
    rows: impl Iterator<Item = std::ops::Range<u64>>,
) -> Option<(
    windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    u64,
    Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
)> {
    use crate::host::transition;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT,
    };

    let data = buffer_range(resource, offset, bytes)?;
    let address = data as u64;
    let read_address = address + read.start;
    let read_bytes = read.end - read.start;
    let absolute = |range: std::ops::Range<u64>| {
        (address + range.start) as usize..(address + range.end) as usize
    };
    let source = list
        .written_shadow(std::iter::once(absolute(read.clone())))
        .or_else(|| list.written_shadow(rows.map(absolute)))
        .and_then(|shadow| {
            let shadow_off = shadow.offset_for_range(address, bytes)?;
            Some((shadow, shadow_off))
        });
    let Some((shadow, shadow_off)) = source else {
        let (upload, staging) = list.upload(crate::host::host()?, bytes)?;
        // The list owns this mapped D3D12 upload buffer. Wine can report
        // its address as MEM_FREE, so use the host-upload path.
        list.defer_fill(staging, data, bytes as usize);
        return Some((upload, 0, None));
    };
    shadow.refresh(list, read_address, read_bytes);
    if shadow_off.is_multiple_of(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT.into()) {
        if list.is_unordered(&shadow.buffer) {
            // SAFETY: open list; `shadow.buffer` is kept alive by `refresh`.
            unsafe {
                list.list.ResourceBarrier(&[transition(
                    &shadow.buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
        }
        return Some((
            shadow.buffer.clone(),
            shadow_off,
            Some(shadow.buffer.clone()),
        ));
    }
    // Rebase an unaligned placed offset onto a 512-byte aligned host buffer on
    // the GPU, preserving box-relative byte offsets inside the footprint.
    let (staging, _) = crate::host::host()?.shared_buffer(bytes)?;
    list.keep(&staging);
    // SAFETY: open list; both buffers are held by `list` and the range is inside both.
    unsafe {
        if list.is_unordered(&shadow.buffer) {
            list.list.ResourceBarrier(&[transition(
                &shadow.buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        list.list.CopyBufferRegion(
            &staging,
            read.start,
            &shadow.buffer,
            shadow_off + read.start,
            read_bytes,
        );
        list.list.ResourceBarrier(&[
            transition(
                &shadow.buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            transition(
                &staging,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ),
        ]);
    }
    list.set_unordered(&shadow.buffer);
    Some((staging, 0, None))
}

/// Texture to texture on the GPU; placed buffer to texture from its host copy
/// after a same-list GPU write, or through an upload buffer filled at
/// submission; texture to buffer through readback memory ([`read_back`]).
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_BOX, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_TEXTURE_COPY_LOCATION,
        };
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            if dst.kind == FOOTPRINT && src.kind != FOOTPRINT {
                read_back(list, &dst, [x, y, z], &src, source_box);
            }
            return;
        };
        let mut from = src;
        let _buffer; // keeps the source alive while this call records it
        let mut restore_uav = None;
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        // SAFETY: the title passes source box coordinates or null.
        let source_box = unsafe { source_box.as_ref() };
        if src.kind == FOOTPRINT {
            let Some(bytes) = src.footprint.bytes() else {
                return;
            };
            let Some((read_span, rows)) = src.footprint.box_reads(source_box.copied()) else {
                return;
            };
            let Some((buffer, offset, restore)) = placed_copy_source(
                list,
                src.resource,
                src.footprint.offset,
                bytes,
                &read_span,
                rows,
            ) else {
                return;
            };
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint.offset = offset;
            restore_uav = restore;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer; the range is inside it.
                read = Some((
                    unsafe { buffer.GetGPUVirtualAddress() } + offset + read_span.start,
                    read_span.end - read_span.start,
                ));
            }
            _buffer = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            if crate::texture::title_written(&texture) {
                crate::texture::upload(list, src.resource, &texture);
            }
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        ensure_state(list, dst.resource, COPY_DEST);
        crate::texture::host_writes(list, dst.resource);
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = source_box.map(|bounds| std::ptr::from_ref(bounds).cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            );
            if let Some(ref shadow_buf) = restore_uav {
                list.list.ResourceBarrier(&[crate::host::transition(
                    shadow_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                list.set_unordered(shadow_buf);
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

/// Texture read backs since the last frame-rate line: how many, bytes
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) static READ_BACKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// (texture, buffer) pairs read back so far, each logged once (diagnostics).
#[cfg(feature = "diagnostics")]
static READ_BACK_PAIRS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Texture to buffer (a read back): the region is copied on the host into
/// readback memory laid out as the title's footprint, and once the list has
/// run its rows go into the title's buffer (the title reads them on the CPU
/// after its fence).
#[cfg(windows)]
fn read_back(list: &crate::host::HostList, dst: &CopyLocation, at: [u32; 3], src: &CopyLocation, source_box: *const [u32; 6]) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BOX, D3D12_RESOURCE_DIMENSION_TEXTURE3D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    };
    use windows::core::Interface;
    macro_rules! refused {
        ($why:literal) => {{
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(concat!("texture to buffer copy refused: ", $why)));
            return;
        }};
    }
    let (Some(host), Some(texture)) = (crate::host::host(), crate::resource::host_resource(src.resource)) else {
        refused!("no host texture");
    };
    // SAFETY: the title's box (same layout as the host's) or null; a live texture.
    let (b, desc) = unsafe { (source_box.cast::<D3D12_BOX>().as_ref(), texture.GetDesc()) };
    let size = match b {
        Some(b) => [b.right.saturating_sub(b.left), b.bottom.saturating_sub(b.top), b.back.saturating_sub(b.front)],
        None => {
            // A subresource index shares the footprint's first dword.
            let mip = src.footprint.offset as u32 % u32::from(desc.MipLevels.max(1));
            let depth = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { u32::from(desc.DepthOrArraySize) } else { 1 };
            [((desc.Width >> mip) as u32).max(1), (desc.Height >> mip).max(1), (depth >> mip).max(1)]
        }
    };
    // The host takes footprints of 256-byte pitches only (as the console).
    if !dst.footprint.row_pitch.is_multiple_of(256) {
        refused!("a row pitch not a multiple of 256");
    }
    // shortcut: depth and stencil planes copy out in typeless formats of
    // their own; refused until a title is seen reading one back.
    if desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
        refused!("a depth or stencil texture");
    }
    // The host copies into a footprint only within a buffer that holds all
    // of it, so readback memory spans the footprint, not just the region.
    let (Some(rows), Some(len)) = (dst.footprint.rows(at, size), dst.footprint.bytes()) else {
        refused!("a region outside its footprint");
    };
    let (Some(title), Some((buffer, data))) = (buffer_range(dst.resource, dst.footprint.offset, len), list.readback(host, len)) else {
        return;
    };
    if crate::texture::title_written(&texture) {
        crate::texture::upload(list, src.resource, &texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let (mut to, mut from) = (*dst, *src);
    to.resource = buffer.as_raw().cast_const().cast();
    to.footprint.offset = 0;
    from.resource = texture.as_raw().cast_const().cast();
    #[cfg(feature = "diagnostics")]
    {
        let text = || {
            let mem = crate::resource::texture_memory(src.resource).map_or(0, |(at, _)| at as usize);
            format!(
                "read back {:?} (mem {mem:#x}, {}x{} f{} sub {}) {size:?} at {at:?} into {:?}+{:#x}, {len} bytes",
                src.resource,
                desc.Width,
                desc.Height,
                desc.Format.0,
                src.footprint.offset as u32,
                dst.resource,
                dst.footprint.offset
            )
        };
        READ_BACKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        READ_BACKS[1].fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        let pair = (src.resource as usize, dst.resource as usize);
        let mut seen = READ_BACK_PAIRS.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&pair) {
            seen.push(pair);
            tracing::info!("a texture read back (first of this pair): {}", text());
        }
        list.note_op(text);
    }
    // SAFETY: an open list; both locations in the host's layout, naming live
    // resources; the region lies in the footprint (checked).
    unsafe {
        list.list.CopyTextureRegion(
            (&raw const to).cast(),
            at[0],
            at[1],
            at[2],
            (&raw const from).cast(),
            b.map(std::ptr::from_ref),
        )
    };
    let (data, title) = (data as usize, title as usize);
    list.defer_after(move || {
        for &(at, n) in &rows {
            let (at, n) = (at as usize, n as usize);
            let _writing = crate::watch::writing(title + at, n);
            crate::host::write_mapped((title + at) as *mut u8, (data + at) as *const u8, n);
        }
    });
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_writes(list, dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = DXGI_FORMAT(format as i32);
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(&to, dst_sub, &from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    &to,
                    dst_sub,
                    0,
                    0,
                    &from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(&texture, record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        ensure_state(list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(&texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        list.defer(move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    #[cfg(windows)]
    state::dispatch(this, x, y, z);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z);
        count_dispatch();
    }
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: queries are answered when
/// resolved.
extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// shortcut: no query runs on the host GPU; results are plausible values,
/// not measured ones (time spans come out as the submission's length).
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    let Some(end) = offset.checked_add(u64::from(count) * bytes) else {
        return;
    };
    if end > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and every submission is waited for, so there is nothing
/// to do.
extern "system" fn sync(_: This, _flags: u32, _control: u32, _address: u64, _size: u64) {}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. Grand Theft Auto VI's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Some(block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Memory [`allocate_upload`] hands out, zeroed, freed when the list is
/// reset. On Windows it lies in [4 GiB, 2^40), where translated shaders
/// find title memory by address (`crate::memory`; the console's GPU
/// addresses have 40 bits), and is a buffer of its own to the host copies
/// (`crate::shadow`), so its copy is exactly it.
struct Upload {
    #[cfg(windows)]
    at: usize,
    #[cfg(windows)]
    len: usize,
    #[cfg(not(windows))]
    block: Vec<u8>,
}

impl Upload {
    #[cfg(windows)]
    fn new(len: usize) -> Option<Upload> {
        use windows_sys::Win32::System::Memory::{
            MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
            MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 32) as *mut _,
            HighestEndingAddress: ((1usize << 40) - 1) as *mut _,
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // Watched like the title's pools: copies take only written pages.
        // SAFETY: a fresh allocation; the parameter points at `requirements`,
        // which outlives the call.
        let at = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        } as usize;
        if at == 0 {
            return None;
        }
        crate::shadow::register(at, at as u64, len as u64);
        UPLOADS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        Some(Upload { at, len })
    }

    #[cfg(not(windows))]
    fn new(len: usize) -> Option<Upload> {
        Some(Upload {
            block: vec![0; len],
        })
    }

    fn start(&self) -> *mut u8 {
        #[cfg(windows)]
        return self.at as *mut u8;
        #[cfg(not(windows))]
        return self.block.as_ptr().cast_mut();
    }
}

#[cfg(windows)]
impl Drop for Upload {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        crate::shadow::unregister(self.at, self.at as u64);
        UPLOADS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_sub(self.len as u64, std::sync::atomic::Ordering::Relaxed);
        // Not guarded any more: memory allocated here next is new.
        crate::watch::title_memory_remapped(self.at, self.len);
        // SAFETY: our allocation; lists that still use its copy hold that.
        unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
    }
}

// SAFETY: plain memory, handed out once.
#[cfg(windows)]
unsafe impl Send for Upload {}

/// Live list upload blocks and their bytes (the statistics).
#[cfg(windows)]
pub(crate) static UPLOADS: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth bounds, depth order and shading rate hints): any
/// arguments, nothing returned.
extern "system" fn ignored_state(_: This) {}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        with_state(this, |list, s| s.viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        // SAFETY: the title passes four floats or null.
        with_state(this, |list, s| s.blend_factor(list, unsafe { factor.as_ref() }));
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, s| s.stencil_ref(list, reference));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    /// `ClearUnorderedAccessViewUint(GPU handle, CPU handle, resource,
    /// values, rectangle count, rectangles)`.
    pub(super) extern "system" fn clear_unordered_uint(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, false, items(rects, count));
    }
    /// `ClearUnorderedAccessViewFloat`, the same with four floats.
    pub(super) extern "system" fn clear_unordered_float(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, true, items(rects, count));
    }
    fn clear_unordered(this: This, cpu: u64, values: *const [u32; 4], float: bool, rects: &[RECT]) {
        // SAFETY: the title passes four values (null is tolerated).
        let Some(&values) = (unsafe { values.as_ref() }) else {
            return;
        };
        with_state(this, |list, s| {
            s.clear_unordered(list, cpu, values, float, rects);
        });
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, x, y, z));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as clear_unordered_float;
    pub(super) use super::ignored_state as clear_unordered_uint;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Grand Theft Auto VI's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}


/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        // Which kinds of barriers (0 global, 1 texture, 2 buffer) and how many.
        #[cfg(feature = "diagnostics")]
        if !_groups.is_null() {
            // SAFETY: the title passes `count` groups of 16 bytes: kind, count, pointer.
            let groups = unsafe { std::slice::from_raw_parts(_groups.cast::<[u32; 4]>(), count as usize) };
            list.note_op(|| format!("barrier groups {:?}", groups.iter().map(|g| (g[0], g[1])).collect::<Vec<_>>()));
        }
        // SAFETY: the same layout; no resource named.
        let barrier = unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    #[cfg(not(windows))]
    let _ = (this, count);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
pub(crate) extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, ignored_state as *const ()),             // 0x220 depth bounds
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait32 as *const ()),                    // 0x238 (newer 0x248)
    (72, wait64 as *const ()),                    // 0x240 (newer 0x250)
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 12] = [
    (53, state::clear_unordered_uint as *const ()), // 0x1a8
    (54, state::clear_unordered_float as *const ()), // 0x1b0
    (55, ignored_state as *const ()),               // 0x1b8 DiscardResource
    (28, query_mark as *const ()),                  // 0xe0 BeginQuery
    (29, query_mark as *const ()),                  // 0xe8 EndQuery
    (30, resolve_queries as *const ()),             // 0xf0
    (78, dispatch_x as *const ()),                  // 0x270
    (98, state::dispatch_mesh_x as *const ()),      // 0x310
    (99, driver_hint as *const ()),                 // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// Grand Theft Auto VI's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from Grand Theft Auto VI's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tracing::{Event, Metadata, Subscriber, span};

    struct PipelineBinds(Arc<AtomicUsize>);

    impl Subscriber for PipelineBinds {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.is_event()
                && metadata.target() == "victory_gpu::pipeline"
                && metadata.fields().field("pipeline").is_some()
                && metadata.fields().field("frame").is_some()
        }

        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            unreachable!()
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
        fn enter(&self, _: &span::Id) {}
        fn exit(&self, _: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if self.enabled(event.metadata()) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn lists_bind_the_initial_pipeline_on_creation() {
        use crate::pipeline::{ComputeHead, create_compute};

        // A synthetic compute description matching the public ABI.
        #[repr(C)]
        struct ComputeDesc {
            root: This,
            code: *const u8,
            size: usize,
        }

        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let desc = ListDesc {
            kind: 0,
            flags: 0,
            nodes: 0,
            barrier_batch: 0,
        };
        let bytes = [0xa5, 0xb6, 0xc7];
        let compute = ComputeDesc {
            root: null,
            code: bytes.as_ptr(),
            size: bytes.len(),
        };
        let compute = std::ptr::from_ref(&compute).cast::<ComputeHead>();
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        let binds = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing::Dispatch::new(PipelineBinds(binds.clone()));

        tracing::dispatcher::with_default(&subscriber, || {
            for console in [false, true] {
                let create = |pipeline, out| {
                    if console {
                        create_list_x(null, &desc, allocator, pipeline, &LIST_IIDS[1], out)
                    } else {
                        create_list(null, 0, 0, allocator, pipeline, &LIST_IIDS[1], out)
                    }
                };
                let mut pipeline = null;
                let mut list = null;
                assert_eq!(
                    create_compute(null, compute, &PAGEABLE_IIDS[0], &mut pipeline),
                    S_OK
                );
                let before = binds.load(Ordering::Relaxed);

                assert_eq!(create(null, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before);
                assert_eq!(release(list), 0);

                assert_eq!(create(pipeline, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before + 1);
                assert_eq!(release(list), 0);
                assert_eq!(release(pipeline), 0);
            }
        });
        assert_eq!(release(allocator), 0);
    }

    /// ClearUnorderedAccessViewUint/Float and DiscardResource come before
    /// the methods newer lists insert: the AMD upscaler calls them there.
    #[test]
    fn unordered_access_clears_have_the_same_slots_in_both_layouts() {
        for table in [&LIST_FUNCTIONS, &NEWER_LIST_FUNCTIONS] {
            assert_eq!(table.0[0x1a8 / 8], state::clear_unordered_uint as *const ());
            assert_eq!(
                table.0[0x1b0 / 8],
                state::clear_unordered_float as *const ()
            );
            assert_eq!(table.0[0x1b8 / 8], ignored_state as *const ());
        }
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, below 2^40 (where
        // shaders find title memory), gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        #[cfg(windows)]
        assert!((1 << 32..1 << 40).contains(&(p as usize)), "{p:p}");
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// An offset near `u64::MAX` must not overflow and write outside the
    /// query destination buffer.
    #[test]
    fn resolve_queries_rejects_overflowing_offset() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let buffer = crate::resource::test_buffer(64);
        // Used to overflow `offset + count * 8` (panic in debug, wrap in release).
        resolve_queries(list, null, 0, 0, 2, buffer, u64::MAX - 8);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for object in [buffer, list, allocator] {
            assert_eq!(release(object), 0);
        }
    }

    #[cfg(windows)]
    mod placed_copy_gpu {
        use super::*;
        use std::sync::atomic::{AtomicU32, Ordering};
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        use windows::core::{Interface, PCSTR, s};

        const WIDTH: usize = 4;
        const HEIGHT: usize = 4;
        const ROW_PITCH: usize = 256;
        const ALIGNED_OFFSET: usize = 512;
        const UNALIGNED_OFFSET: usize = 516;
        const FOOTPRINT_BYTES: usize = (HEIGHT - 1) * ROW_PITCH + WIDTH * 4;
        const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

        static DEBUG_ERRORS: AtomicU32 = AtomicU32::new(0);

        unsafe extern "system" fn count_debug_errors(
            _: D3D12_MESSAGE_CATEGORY,
            severity: D3D12_MESSAGE_SEVERITY,
            id: D3D12_MESSAGE_ID,
            desc: PCSTR,
            _: *mut std::ffi::c_void,
        ) {
            if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0
                && id != D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_INVALIDLIBRARYBLOB
                && id.0 != 645
            {
                DEBUG_ERRORS.fetch_add(1, Ordering::Relaxed);
                // SAFETY: D3D12 supplies a valid NUL-terminated string during the callback.
                let text = unsafe { desc.to_string() }.unwrap_or_default();
                eprintln!("D3D12 debug error (id={}): {text}", id.0);
            }
        }

        struct DebugGuard(Option<(ID3D12InfoQueue1, u32)>);
        impl Drop for DebugGuard {
            fn drop(&mut self) {
                if let Some((ref queue, cookie)) = self.0 {
                    // SAFETY: cookie was registered on `queue`.
                    let _ = unsafe { queue.UnregisterMessageCallback(cookie) };
                }
            }
        }

        fn attach_debug(device: &ID3D12Device, tag: &str) -> DebugGuard {
            DEBUG_ERRORS.store(0, Ordering::Relaxed);
            // SAFETY: plain DXGI adapter query for logging the active test backend.
            if let Ok(desc) = unsafe {
                CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(device.GetAdapterLuid()))
                    .and_then(|a| a.GetDesc1())
            } {
                eprintln!(
                    "{tag} adapter: {} ({:04x}:{:04x}, flags={:#x})",
                    String::from_utf16_lossy(&desc.Description).trim_end_matches('\0'),
                    desc.VendorId,
                    desc.DeviceId,
                    desc.Flags
                );
            }
            DebugGuard(device.cast::<ID3D12InfoQueue1>().ok().and_then(|queue| {
                let mut cookie = 0;
                // SAFETY: static callback function, unregistered on drop.
                unsafe {
                    queue.RegisterMessageCallback(
                        Some(count_debug_errors),
                        D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                        std::ptr::null_mut(),
                        &mut cookie,
                    )
                }
                .ok()
                .map(|()| (queue, cookie))
            }))
        }

        struct Held(This);
        impl Drop for Held {
            fn drop(&mut self) {
                slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
            }
        }

        fn make_resource(desc: D3D12_RESOURCE_DESC) -> Held {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let iid = ID3D12Resource::IID;
            let mut out = std::ptr::null();
            assert_eq!(
                crate::resource::create_committed(
                    std::ptr::null(),
                    (&raw const heap).cast(),
                    0,
                    (&raw const desc).cast(),
                    D3D12_RESOURCE_STATE_COMMON.0 as u32,
                    std::ptr::null(),
                    (&raw const iid).cast(),
                    &mut out,
                ),
                S_OK
            );
            Held(out)
        }

        fn make_texture() -> Held {
            make_resource(D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: WIDTH as u64,
                Height: HEIGHT as u32,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: FORMAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            })
        }

        fn make_buffer(initial: &[u8]) -> Held {
            let res = make_resource(D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: initial.len() as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            });
            let (ptr, size) = crate::resource::buffer_bytes(res.0).unwrap();
            assert_eq!(size, initial.len() as u64);
            // SAFETY: freshly created committed guest buffer of `initial.len()` bytes.
            unsafe { std::ptr::copy_nonoverlapping(initial.as_ptr(), ptr, initial.len()) };
            res
        }

        fn make_guest_list() -> (Held, Held) {
            let null = std::ptr::null();
            let mut alloc = null;
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut alloc),
                S_OK
            );
            let alloc = Held(alloc);
            let mut list = null;
            assert_eq!(
                create_list(null, 0, 0, alloc.0, null, &LIST_IIDS[1], &mut list),
                S_OK
            );
            (Held(list), alloc)
        }

        fn pattern(tag: u8) -> [[u8; 4]; WIDTH * HEIGHT] {
            std::array::from_fn(|idx| {
                let (x, y) = ((idx % WIDTH) as u8, (idx / WIDTH) as u8);
                [
                    tag,
                    11u8.wrapping_add(x * 23),
                    19u8.wrapping_add(y * 31),
                    255,
                ]
            })
        }

        fn placed_bytes(tag: u8, offset: usize) -> Vec<u8> {
            let mut buf = vec![0x5a; offset + FOOTPRINT_BYTES];
            for (idx, px) in pattern(tag).into_iter().enumerate() {
                let at = offset + (idx / WIDTH) * ROW_PITCH + (idx % WIDTH) * 4;
                buf[at..at + 4].copy_from_slice(&px);
            }
            buf
        }

        fn writer_pso(device: &ID3D12Device) -> Option<(ID3D12RootSignature, ID3D12PipelineState)> {
            const HLSL: &str = r#"
                cbuffer Params : register(b0) { uint dst_byte_offset; uint word_count; }
                ByteAddressBuffer src_words : register(t0);
                RWByteAddressBuffer dst_words : register(u0);
                [RootSignature("RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)")]
                [numthreads(64, 1, 1)]
                void main(uint3 tid : SV_DispatchThreadID) {
                    if (tid.x < word_count) {
                        uint off = tid.x * 4;
                        dst_words.Store(dst_byte_offset + off, src_words.Load(off));
                    }
                }
            "#;
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: static HLSL source and valid output blob pointer.
            unsafe {
                D3DCompile(
                    HLSL.as_ptr().cast(),
                    HLSL.len(),
                    s!("placed_writer"),
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                let code = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                let root: ID3D12RootSignature = device.CreateRootSignature(0, code).ok()?;
                let pso = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;
                Some((root, pso))
            }
        }

        fn dispatch_uav_write(
            recorded: &crate::host::HostList,
            (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
            (dst_buf, byte_offset): (&Held, usize),
            upload_src: &ID3D12Resource,
            write_len: usize,
        ) {
            assert_eq!(write_len % 4, 0);
            let (data, buffer_size) = crate::resource::buffer_bytes(dst_buf.0).unwrap();
            let buffer_base = data as u64;
            let shadow = crate::shadow::find(buffer_base, buffer_size).unwrap();
            recorded.keep(root);
            recorded.keep(pso);
            recorded.keep(upload_src);
            let mark = recorded.mark_gpu_writes();
            shadow.refresh(recorded, buffer_base, buffer_size);
            shadow.gpu_writes(
                recorded,
                buffer_base + byte_offset as u64,
                write_len as u64,
                false,
            );
            let words = (write_len / 4) as u32;
            let params = [byte_offset as u32, words];
            // SAFETY: open command list; root views point into live buffers.
            unsafe {
                recorded.list.SetComputeRootSignature(root);
                recorded.list.SetPipelineState(pso);
                recorded
                    .list
                    .SetComputeRoot32BitConstants(0, 2, params.as_ptr().cast(), 0);
                recorded
                    .list
                    .SetComputeRootShaderResourceView(1, upload_src.GetGPUVirtualAddress());
                recorded
                    .list
                    .SetComputeRootUnorderedAccessView(2, shadow.gpu(buffer_base));
                recorded.list.Dispatch(words.div_ceil(64), 1, 1);
                recorded.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(None),
                        }),
                    },
                }]);
            }
            recorded.commit_gpu_writes(mark);
        }

        fn record_placed_copy(
            list: &Held,
            dst_tex: &Held,
            src_buf: &Held,
            offset: usize,
            [dst_x, dst_y]: [u32; 2],
            src_box: Option<&[u32; 6]>,
        ) {
            let native_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: offset as u64,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: FORMAT,
                    Width: WIDTH as u32,
                    Height: HEIGHT as u32,
                    Depth: 1,
                    RowPitch: ROW_PITCH as u32,
                },
            };
            // SAFETY: `PlacedFootprint` matches `D3D12_PLACED_SUBRESOURCE_FOOTPRINT` layout.
            let footprint = unsafe {
                std::mem::transmute::<
                    D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
                    crate::resource::PlacedFootprint,
                >(native_fp)
            };
            let src = CopyLocation {
                resource: src_buf.0,
                kind: 1,
                footprint,
            };
            let dst = CopyLocation {
                resource: dst_tex.0,
                kind: 0,
                // SAFETY: all-zero selects subresource index 0.
                footprint: unsafe { std::mem::zeroed() },
            };
            type CopyFn = extern "system" fn(
                This,
                *const CopyLocation,
                u32,
                u32,
                u32,
                *const CopyLocation,
                *const [u32; 6],
            );
            slot::<CopyFn>(list.0, 0x80)(
                list.0,
                &dst,
                dst_x,
                dst_y,
                0,
                &src,
                src_box.map_or(std::ptr::null(), std::ptr::from_ref),
            );
        }

        fn record_texture_readback(
            host: &crate::host::Host,
            recorded: &crate::host::HostList,
            tex: &Held,
        ) -> ID3D12Resource {
            ensure_state(recorded, tex.0, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
            let host_tex = crate::resource::host_resource(tex.0).unwrap();
            let mut rb: Option<ID3D12Resource> = None;
            // SAFETY: plain readback buffer creation and texture-to-buffer copy recording.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_READBACK,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: FOOTPRINT_BYTES as u64,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut rb,
                    )
                    .unwrap();
                let rb = rb.unwrap();
                let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&rb),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: FORMAT,
                                Width: WIDTH as u32,
                                Height: HEIGHT as u32,
                                Depth: 1,
                                RowPitch: ROW_PITCH as u32,
                            },
                        },
                    },
                };
                let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&host_tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                recorded
                    .list
                    .CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
                rb
            }
        }

        fn read_pixels(rb: &ID3D12Resource) -> [[u8; 4]; WIDTH * HEIGHT] {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: GPU work completed; `rb` holds `FOOTPRINT_BYTES` bytes.
            unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let out = std::array::from_fn(|idx| {
                    let at = (idx / WIDTH) * ROW_PITCH + (idx % WIDTH) * 4;
                    ptr.cast::<u8>().add(at).cast::<[u8; 4]>().read_unaligned()
                });
                rb.Unmap(0, None);
                out
            }
        }

        #[test]
        fn placed_copy_reads_same_list_gpu_writes_and_preserves_cpu_snapshots() {
            let Some(host) = crate::host::host() else {
                return;
            };
            let _dbg = attach_debug(&host.device, "placed_copy_full");
            let Some(writer) = writer_pso(&host.device) else {
                return;
            };

            // 1. Direct same-list Compute UAV write -> CopyTextureRegion at both
            //    512-byte aligned and 516-byte unaligned offsets, including replay.
            let (upload, mapped) = host.upload_buffer(FOOTPRINT_BYTES as u64).unwrap();
            for offset in [ALIGNED_OFFSET, UNALIGNED_OFFSET] {
                let src = make_buffer(&placed_bytes(17, offset));
                let (data, size) = crate::resource::buffer_bytes(src.0).unwrap();
                let dst = make_texture();
                let (list, _alloc) = make_guest_list();
                let recorded = host_list(list.0).unwrap();

                dispatch_uav_write(recorded, &writer, (&src, offset), &upload, FOOTPRINT_BYTES);
                record_placed_copy(&list, &dst, &src, offset, [0, 0], None);
                let rb = record_texture_readback(host, recorded, &dst);
                assert_eq!(close(list.0), S_OK);

                for tag in [198u8, 223u8] {
                    let fresh = placed_bytes(tag, offset);
                    // SAFETY: `mapped` holds `FOOTPRINT_BYTES` bytes and idle between executes.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            fresh[offset..].as_ptr(),
                            mapped,
                            FOOTPRINT_BYTES,
                        );
                    }
                    host.execute(&[recorded]);
                    assert_eq!(
                        read_pixels(&rb),
                        pattern(tag),
                        "UAV offset={offset} tag={tag}"
                    );
                    // SAFETY: synchronous execution completed; guest buffer remains live.
                    let guest_after = unsafe { std::slice::from_raw_parts(data, size as usize) };
                    assert_eq!(guest_after, fresh.as_slice());
                }
            }

            // 2. Chained same-list Compute UAV -> guest CopyBufferRegion -> CopyTextureRegion,
            //    plus two ordered CPU CopyBufferRegion snapshots in the same list.
            let uav_intermediate = make_buffer(&placed_bytes(10, ALIGNED_OFFSET));
            let placed_src = make_buffer(&placed_bytes(20, UNALIGNED_OFFSET));
            let cpu_snap_a = make_buffer(&placed_bytes(41, ALIGNED_OFFSET));
            let cpu_snap_b = make_buffer(&placed_bytes(73, ALIGNED_OFFSET));
            let chained_fresh = placed_bytes(165, ALIGNED_OFFSET);
            // SAFETY: `mapped` holds `FOOTPRINT_BYTES` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chained_fresh[ALIGNED_OFFSET..].as_ptr(),
                    mapped,
                    FOOTPRINT_BYTES,
                );
            }

            let tex_chained = make_texture();
            let tex_cpu_a = make_texture();
            let tex_cpu_b = make_texture();
            let (list, _alloc) = make_guest_list();
            let recorded = host_list(list.0).unwrap();

            dispatch_uav_write(
                recorded,
                &writer,
                (&uav_intermediate, ALIGNED_OFFSET),
                &upload,
                FOOTPRINT_BYTES,
            );
            copy_buffer_region(
                list.0,
                placed_src.0,
                UNALIGNED_OFFSET as u64,
                uav_intermediate.0,
                ALIGNED_OFFSET as u64,
                FOOTPRINT_BYTES as u64,
            );
            record_placed_copy(
                &list,
                &tex_chained,
                &placed_src,
                UNALIGNED_OFFSET,
                [0, 0],
                None,
            );

            // Subsequent CPU buffer copies into `uav_intermediate` must revert to
            // ordered submission-time CPU snapshots.
            let full_len = (ALIGNED_OFFSET + FOOTPRINT_BYTES) as u64;
            copy_buffer_region(list.0, uav_intermediate.0, 0, cpu_snap_a.0, 0, full_len);
            record_placed_copy(
                &list,
                &tex_cpu_a,
                &uav_intermediate,
                ALIGNED_OFFSET,
                [0, 0],
                None,
            );
            copy_buffer_region(list.0, uav_intermediate.0, 0, cpu_snap_b.0, 0, full_len);
            record_placed_copy(
                &list,
                &tex_cpu_b,
                &uav_intermediate,
                ALIGNED_OFFSET,
                [0, 0],
                None,
            );

            let rb_chained = record_texture_readback(host, recorded, &tex_chained);
            let rb_cpu_a = record_texture_readback(host, recorded, &tex_cpu_a);
            let rb_cpu_b = record_texture_readback(host, recorded, &tex_cpu_b);
            assert_eq!(close(list.0), S_OK);
            host.execute(&[recorded]);

            assert_eq!(read_pixels(&rb_chained), pattern(165));
            assert_eq!(read_pixels(&rb_cpu_a), pattern(41));
            assert_eq!(read_pixels(&rb_cpu_b), pattern(73));
            assert_eq!(DEBUG_ERRORS.load(Ordering::Relaxed), 0);
        }

        #[test]
        fn placed_boxed_copy_reads_gpu_writes_and_rebases_subregions() {
            let Some(host) = crate::host::host() else {
                return;
            };
            let _dbg = attach_debug(&host.device, "placed_copy_boxed");
            let Some(writer) = writer_pso(&host.device) else {
                return;
            };
            let seed_buf = make_buffer(&placed_bytes(57, ALIGNED_OFFSET));

            // Case A: single-row box [0, 0, 0, 2, 1, 1] at aligned offset 512,
            //         8-byte GPU write covering only the 2 boxed pixels.
            // Case B: nonzero-origin box [1, 2, 0, 3, 3, 1] at unaligned offset 516,
            //         copied to destination (0, 2) via GPU rebase.
            for (offset, src_box, dst_xy, tag) in [
                (ALIGNED_OFFSET, [0, 0, 0, 2, 1, 1], [0, 0], 181u8),
                (UNALIGNED_OFFSET, [1, 2, 0, 3, 3, 1], [0, 2], 204u8),
            ] {
                let [left, top, _, right, bottom, _] = src_box.map(|v| v as usize);
                let span_begin = offset + top * ROW_PITCH + left * 4;
                let span_end = offset + (bottom - 1) * ROW_PITCH + right * 4;
                let span_len = span_end - span_begin;

                let old = placed_bytes(17, offset);
                let fresh = placed_bytes(tag, offset);
                let src = make_buffer(&old);
                let dst = make_texture();
                let (upload, mapped) = host.upload_buffer(span_len as u64).unwrap();
                // SAFETY: `mapped` holds `span_len` bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        fresh[span_begin..span_end].as_ptr(),
                        mapped,
                        span_len,
                    );
                }

                let (list, _alloc) = make_guest_list();
                let recorded = host_list(list.0).unwrap();
                record_placed_copy(&list, &dst, &seed_buf, ALIGNED_OFFSET, [0, 0], None);
                dispatch_uav_write(recorded, &writer, (&src, span_begin), &upload, span_len);
                record_placed_copy(&list, &dst, &src, offset, dst_xy, Some(&src_box));
                let rb = record_texture_readback(host, recorded, &dst);
                assert_eq!(close(list.0), S_OK);
                host.execute(&[recorded]);

                let got = read_pixels(&rb);
                let expected = pattern(tag);
                let seed = pattern(57);
                let [dx, dy] = dst_xy.map(|v| v as usize);
                let (bw, bh) = (right - left, bottom - top);
                for (i, px) in got.into_iter().enumerate() {
                    let (x, y) = (i % WIDTH, i / WIDTH);
                    if (dx..dx + bw).contains(&x) && (dy..dy + bh).contains(&y) {
                        let sx = left + (x - dx);
                        let sy = top + (y - dy);
                        assert_eq!(px, expected[sy * WIDTH + sx]);
                    } else {
                        assert_eq!(px, seed[i]);
                    }
                }
            }

            // Case C: two pitched rows [1, 1, 0, 3, 3, 1] at unaligned offset 516
            //         written by two separate 8-byte GPU dispatches (leaving the
            //         row-pitch gap untouched) plus a 4-byte partial GPU overwrite
            //         of the first pixel in row 1.
            let src_box = [1u32, 1, 0, 3, 3, 1];
            let src = make_buffer(&placed_bytes(17, UNALIGNED_OFFSET));
            let dst = make_texture();
            let row1_fresh = placed_bytes(140, UNALIGNED_OFFSET);
            let row2_fresh = placed_bytes(190, UNALIGNED_OFFSET);
            let override_px = [250u8, 12, 34, 255];

            let r1_off = UNALIGNED_OFFSET + ROW_PITCH + 4;
            let r2_off = UNALIGNED_OFFSET + 2 * ROW_PITCH + 4;
            let (up1, m1) = host.upload_buffer(8).unwrap();
            let (up2, m2) = host.upload_buffer(8).unwrap();
            let (up_ov, m_ov) = host.upload_buffer(4).unwrap();
            // SAFETY: each upload buffer fits its 8-byte or 4-byte payload.
            unsafe {
                std::ptr::copy_nonoverlapping(row1_fresh[r1_off..r1_off + 8].as_ptr(), m1, 8);
                std::ptr::copy_nonoverlapping(row2_fresh[r2_off..r2_off + 8].as_ptr(), m2, 8);
                std::ptr::copy_nonoverlapping(override_px.as_ptr(), m_ov, 4);
            }

            let (list, _alloc) = make_guest_list();
            let recorded = host_list(list.0).unwrap();
            record_placed_copy(&list, &dst, &seed_buf, ALIGNED_OFFSET, [0, 0], None);
            dispatch_uav_write(recorded, &writer, (&src, r1_off), &up1, 8);
            dispatch_uav_write(recorded, &writer, (&src, r2_off), &up2, 8);
            dispatch_uav_write(recorded, &writer, (&src, r1_off), &up_ov, 4);
            record_placed_copy(&list, &dst, &src, UNALIGNED_OFFSET, [2, 0], Some(&src_box));
            let rb = record_texture_readback(host, recorded, &dst);
            assert_eq!(close(list.0), S_OK);
            host.execute(&[recorded]);

            let got = read_pixels(&rb);
            let p140 = pattern(140);
            let p190 = pattern(190);
            let seed = pattern(57);
            assert_eq!(got[2], override_px);
            assert_eq!(got[3], p140[WIDTH + 2]);
            assert_eq!(got[WIDTH + 2], p190[2 * WIDTH + 1]);
            assert_eq!(got[WIDTH + 3], p190[2 * WIDTH + 2]);
            assert_eq!(got[0], seed[0]);
            assert_eq!(got[3 * WIDTH], seed[3 * WIDTH]);
            assert_eq!(DEBUG_ERRORS.load(Ordering::Relaxed), 0);
        }

        fn counter_pso(
            device: &ID3D12Device,
        ) -> Option<(
            ID3D12RootSignature,
            ID3D12PipelineState,
            ID3D12CommandSignature,
        )> {
            const HLSL: &str = r#"
                RWByteAddressBuffer tally : register(u0);
                [RootSignature("UAV(u0)")]
                [numthreads(1, 1, 1)]
                void main() {
                    uint prev;
                    tally.InterlockedAdd(0, 1u, prev);
                }
            "#;
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: static HLSL source and valid output pointers.
            unsafe {
                D3DCompile(
                    HLSL.as_ptr().cast(),
                    HLSL.len(),
                    s!("indirect_tally"),
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                let code = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                let root: ID3D12RootSignature = device.CreateRootSignature(0, code).ok()?;
                let pso = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;
                let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                    ..Default::default()
                };
                let mut sig: Option<ID3D12CommandSignature> = None;
                device
                    .CreateCommandSignature(
                        &D3D12_COMMAND_SIGNATURE_DESC {
                            ByteStride: 12,
                            NumArgumentDescs: 1,
                            pArgumentDescs: &arg,
                            NodeMask: 0,
                        },
                        None::<&ID3D12RootSignature>,
                        &mut sig,
                    )
                    .ok()?;
                Some((root, pso, sig?))
            }
        }

        #[test]
        fn gpu_buffer_copy_tracks_indirect_and_dropped_uav_states() {
            let Some(host) = crate::host::host() else {
                return;
            };
            let _dbg = attach_debug(&host.device, "indirect_and_dropped_uav");
            let (Some(writer), Some((tally_root, tally_pso, dispatch_sig))) =
                (writer_pso(&host.device), counter_pso(&host.device))
            else {
                return;
            };

            // Buffer B starts with indirect dispatch arguments `[1, 1, 1]` at offset 0
            // and stale placed texels at `ALIGNED_OFFSET`.
            let mut b_init = placed_bytes(17, ALIGNED_OFFSET);
            for (i, word) in [1u32, 1, 1].into_iter().enumerate() {
                b_init[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
            }
            let total_bytes = ALIGNED_OFFSET + FOOTPRINT_BYTES;
            let buf_b = make_buffer(&b_init);
            let buf_a = make_buffer(&vec![0u8; total_bytes + 64]);
            let buf_dropped = make_buffer(&placed_bytes(88, ALIGNED_OFFSET));
            let (tally_buf, tally_ptr) = host.shared_buffer(256).unwrap();
            // SAFETY: freshly allocated 256-byte shared buffer.
            unsafe { tally_ptr.cast::<u32>().write_volatile(0) };

            // Upload payload refills `[3, 1, 1]` at offset 0 and `pattern(212)` at `ALIGNED_OFFSET`.
            let mut a_payload = placed_bytes(212, ALIGNED_OFFSET);
            for (i, word) in [3u32, 1, 1].into_iter().enumerate() {
                a_payload[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
            }
            let (upload, mapped) = host.upload_buffer(total_bytes as u64).unwrap();
            // SAFETY: `mapped` holds `total_bytes` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(a_payload.as_ptr(), mapped, total_bytes);
            }

            let tex_from_b = make_texture();
            let tex_from_dropped = make_texture();
            let (list, alloc) = make_guest_list();
            let recorded = host_list(list.0).unwrap();
            recorded.keep(&tally_root);
            recorded.keep(&tally_pso);
            recorded.keep(&dispatch_sig);
            recorded.keep(&tally_buf);

            // 1. First `ExecuteIndirect` pass reads `[1, 1, 1]` from `buf_b` via the
            //    clamp's own argument buffer, keeping `buf_b`'s shadow in `COMMON`.
            let (b_ptr, b_size) = crate::resource::buffer_bytes(buf_b.0).unwrap();
            let b_base = b_ptr as u64;
            let b_shadow = crate::shadow::find(b_base, b_size).unwrap();
            b_shadow.refresh(recorded, b_base, b_size);
            let run_indirect = |recorded: &crate::host::HostList| {
                let clamped = crate::indirect::bound(
                    recorded,
                    b_shadow.gpu(b_base),
                    (2, 12, 1),
                )
                .unwrap();
                // SAFETY: open command list; `clamped` is in `INDIRECT_ARGUMENT`.
                unsafe {
                    recorded.list.SetComputeRootSignature(&tally_root);
                    recorded.list.SetPipelineState(&tally_pso);
                    recorded
                        .list
                        .SetComputeRootUnorderedAccessView(0, tally_buf.GetGPUVirtualAddress());
                    recorded.list.ExecuteIndirect(
                        &dispatch_sig,
                        1,
                        &clamped,
                        0,
                        None::<&ID3D12Resource>,
                        0,
                    );
                    recorded.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                }
                crate::indirect::done(recorded, &clamped);
            };
            run_indirect(recorded);
            assert!(!recorded.is_unordered(&b_shadow.buffer));

            // 2. A dropped draw/dispatch discards its unrecorded `Step::GpuWrite`
            //    and keeps `buf_dropped`'s shadow buffer in `COMMON`.
            let (d_ptr, d_size) = crate::resource::buffer_bytes(buf_dropped.0).unwrap();
            let d_base = d_ptr as u64;
            let d_shadow = crate::shadow::find(d_base, d_size).unwrap();
            let mark = recorded.mark_gpu_writes();
            d_shadow.refresh(recorded, d_base, d_size);
            d_shadow.gpu_writes(
                recorded,
                d_base + ALIGNED_OFFSET as u64,
                FOOTPRINT_BYTES as u64,
                false,
            );
            recorded.discard_gpu_writes(mark);
            assert!(!recorded.is_unordered(&d_shadow.buffer));
            assert!(
                recorded
                    .written_shadow(std::iter::once(
                        (d_base as usize + ALIGNED_OFFSET)
                            ..(d_base as usize + ALIGNED_OFFSET + FOOTPRINT_BYTES),
                    ))
                    .is_none()
            );
            record_placed_copy(
                &list,
                &tex_from_dropped,
                &buf_dropped,
                ALIGNED_OFFSET,
                [0, 0],
                None,
            );

            // 3. Dispatch UAV write into `buf_a` -> same-buffer `CopyBufferRegion` inside `buf_a`
            //    -> `CopyBufferRegion(buf_a -> buf_b)` refills `buf_b` for a second indirect pass
            //    and a placed texture copy.
            dispatch_uav_write(recorded, &writer, (&buf_a, 0), &upload, total_bytes);
            copy_buffer_region(list.0, buf_a.0, total_bytes as u64, buf_a.0, 0, 12);
            copy_buffer_region(list.0, buf_b.0, 0, buf_a.0, 0, total_bytes as u64);
            assert!(recorded.is_unordered(&b_shadow.buffer));

            // 4. Second `ExecuteIndirect` pass executes the refilled `[3, 1, 1]` arguments
            //    from `buf_b`, followed by `CopyTextureRegion` from `buf_b`.
            run_indirect(recorded);
            record_placed_copy(&list, &tex_from_b, &buf_b, ALIGNED_OFFSET, [0, 0], None);

            let rb_b = record_texture_readback(host, recorded, &tex_from_b);
            let rb_dropped = record_texture_readback(host, recorded, &tex_from_dropped);
            assert_eq!(close(list.0), S_OK);
            host.execute(&[recorded]);

            // SAFETY: synchronous execution finished; `tally_ptr` holds `1 + 3 = 4`.
            assert_eq!(unsafe { tally_ptr.cast::<u32>().read_volatile() }, 4);
            assert_eq!(read_pixels(&rb_b), pattern(212));
            assert_eq!(read_pixels(&rb_dropped), pattern(88));
            assert_eq!(DEBUG_ERRORS.load(Ordering::Relaxed), 0);

            // 5. Resetting the command list clears `unordered` so a fresh submission
            //    starts from `COMMON` without stale `UNORDERED_ACCESS -> COPY_*` barriers.
            assert_eq!(reset(list.0, alloc.0, std::ptr::null()), S_OK);
            assert!(!recorded.is_unordered(&b_shadow.buffer));
            let fresh_after_reset = placed_bytes(153, ALIGNED_OFFSET);
            // SAFETY: `mapped` holds `total_bytes` bytes and is idle after `execute`.
            unsafe {
                std::ptr::copy_nonoverlapping(fresh_after_reset.as_ptr(), mapped, total_bytes);
            }
            let tex_reset = make_texture();
            dispatch_uav_write(recorded, &writer, (&buf_a, 0), &upload, total_bytes);
            copy_buffer_region(list.0, buf_b.0, 0, buf_a.0, 0, total_bytes as u64);
            record_placed_copy(&list, &tex_reset, &buf_b, ALIGNED_OFFSET, [0, 0], None);
            let rb_reset = record_texture_readback(host, recorded, &tex_reset);
            assert_eq!(close(list.0), S_OK);
            host.execute(&[recorded]);
            assert_eq!(read_pixels(&rb_reset), pattern(153));
            assert_eq!(DEBUG_ERRORS.load(Ordering::Relaxed), 0);
        }
    }
}
