//! Where the host memory goes: counters of what this crate holds for the
//! title, logged with the process's commit charge (see
//! `victory_hle`'s xmem memory log), to tell the title's own use from ours
//! when loading the world nears the host's commit limit.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Bytes of the title's D3D12 heaps and committed resources (title memory
/// we allocate, from its allocator or ours).
pub(crate) static TITLE_GPU: AtomicU64 = AtomicU64::new(0);
/// Bytes of the host copies of title buffers and loose memory (system
/// memory visible to the GPU), and how many there are.
pub(crate) static SHADOWS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SHADOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Host textures (the driver's allocations, video memory while it fits),
/// how many, and how many are textures the title placed itself, never
/// released (`resource::texture_at`).
pub(crate) static HOST_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HOST_TEXTURE_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static PLACED_COUNT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn add(counter: &AtomicU64, bytes: u64) {
    counter.fetch_add(bytes, Relaxed);
}

pub(crate) fn sub(counter: &AtomicU64, bytes: u64) {
    // Saturating: a counter never wraps if a release is counted twice.
    let _ = counter.fetch_update(Relaxed, Relaxed, |v| Some(v.saturating_sub(bytes)));
}

/// Bytes and number of the host buffers alive, by kind: upload, read-back,
/// shared, unordered access; heaps of textures and of copies' tiles; and
/// command lists alive.
pub(crate) static BUFFERS: [[AtomicU64; 2]; 6] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; 6];
pub(crate) static LISTS: AtomicU64 = AtomicU64::new(0);

/// Counts `resource` (`bytes`) as one of `kind` until it is destroyed.
#[cfg(windows)]
pub(crate) fn track(
    resource: &impl windows::core::Interface,
    kind: usize,
    bytes: u64,
) {
    use std::ffi::c_void;
    use windows::core::{GUID, HRESULT};
    /// An object the resource releases when it is destroyed.
    #[repr(C)]
    struct Guard {
        vtable: *const [*const c_void; 3],
        refs: AtomicU64,
        kind: usize,
        bytes: u64,
    }
    unsafe extern "system" fn query(_: *mut c_void, _: *const GUID, out: *mut *mut c_void) -> HRESULT {
        // SAFETY: the caller passes an out pointer.
        unsafe { *out = std::ptr::null_mut() };
        HRESULT(0x8000_4002_u32 as i32)
    }
    unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
        // SAFETY: a live guard.
        unsafe { &*this.cast::<Guard>() }.refs.fetch_add(1, Relaxed) as u32 + 1
    }
    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        // SAFETY: a live guard.
        let left = unsafe { &*this.cast::<Guard>() }.refs.fetch_sub(1, Relaxed) - 1;
        if left == 0 {
            // SAFETY: the last reference; made by `Box::into_raw`.
            let guard = unsafe { Box::from_raw(this.cast::<Guard>()) };
            sub(&BUFFERS[guard.kind][0], guard.bytes);
            sub(&BUFFERS[guard.kind][1], 1);
        }
        left as u32
    }
    struct Table([*const c_void; 3]);
    // SAFETY: function pointers.
    unsafe impl Sync for Table {}
    static TABLE: Table = Table([
        query as *const c_void,
        add_ref as *const c_void,
        release as *const c_void,
    ]);
    const KEY: GUID = GUID::from_u128(0x6a1d_4a0e_5d3c_4c57_9d0f_3c5b_71e2_0a11);
    use windows::core::Interface;
    let Ok(object) = resource.cast::<windows::Win32::Graphics::Direct3D12::ID3D12Object>() else {
        return;
    };
    let guard = Box::into_raw(Box::new(Guard {
        vtable: &raw const TABLE.0,
        refs: AtomicU64::new(1),
        kind,
        bytes,
    }));
    add(&BUFFERS[kind][0], bytes);
    add(&BUFFERS[kind][1], 1);
    // SAFETY: a live resource and an object with the layout of an IUnknown,
    // whose first reference is dropped after.
    unsafe {
        let unknown = windows::core::IUnknown::from_raw(guard.cast());
        let _ = object.SetPrivateDataInterface(&KEY, &unknown);
        drop(unknown);
    }
}

/// What Windows says of the whole machine, as `[(in use, total); 2]` in
/// bytes: physical memory (what standby caches hold counts as free) and the
/// commit limit (physical memory plus the paging files).
#[cfg(windows)]
pub(crate) fn system_memory() -> Option<[(u64, u64); 2]> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: a sized structure of our own.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return None;
    }
    Some([
        (status.ullTotalPhys.saturating_sub(status.ullAvailPhys), status.ullTotalPhys),
        (status.ullTotalPageFile.saturating_sub(status.ullAvailPageFile), status.ullTotalPageFile),
    ])
}

/// How full the machine's memory is, in percent: its physical memory or its
/// commit limit, whichever is higher. Asked at most once a second.
#[cfg(windows)]
pub(crate) fn system_memory_percent() -> u64 {
    static LAST: std::sync::Mutex<(Option<std::time::Instant>, u64)> = std::sync::Mutex::new((None, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.0.is_none_or(|at| at.elapsed() > std::time::Duration::from_secs(1)) {
        last.0 = Some(std::time::Instant::now());
        if let Some([(ram, ram_total), (commit, commit_total)]) = system_memory() {
            last.1 = (ram * 100 / ram_total.max(1)).max(commit * 100 / commit_total.max(1));
        }
    }
    last.1
}

/// One line of MiB figures: what this crate holds, and what the host GPU
/// driver says it uses in video and system memory.
pub fn report() -> String {
    let mib = |v: u64| v >> 20;
    #[allow(unused_mut)]
    let mut line = format!(
        "title gpu heaps {} MiB, buffer copies {} MiB ({}), host textures {} MiB ({}, {} placed)",
        mib(TITLE_GPU.load(Relaxed)),
        mib(SHADOWS.load(Relaxed)),
        SHADOW_COUNT.load(Relaxed),
        mib(HOST_TEXTURES.load(Relaxed)),
        HOST_TEXTURE_COUNT.load(Relaxed),
        PLACED_COUNT.load(Relaxed),
    );
    {
        use std::fmt::Write;
        for (name, b) in ["upload", "readback", "shared", "unordered", "texture heaps", "tile heaps"].iter().zip(&BUFFERS) {
            let _ = write!(line, ", {name} {} MiB ({})", mib(b[0].load(Relaxed)), b[1].load(Relaxed));
        }
        let _ = write!(line, ", {} lists", LISTS.load(Relaxed));
    }
    #[cfg(windows)]
    {
        use std::fmt::Write;
        let _ = write!(
            line,
            ", texture staging {} MiB, snapshots {} MiB",
            mib(crate::texture::staged_bytes()),
            mib(crate::shadow::pool_bytes()),
        );
        if let Some([local, system]) = crate::host::video_memory() {
            let _ = write!(
                line,
                ", driver: video {} of {} MiB, system {} MiB",
                mib(local.0),
                mib(local.1),
                mib(system.0)
            );
        }
        if let Some([(ram, ram_total), (commit, commit_total)]) = system_memory() {
            let _ = write!(
                line,
                ", machine: memory {} of {} MiB in use, commit {} of {} MiB in use",
                mib(ram),
                mib(ram_total),
                mib(commit),
                mib(commit_total)
            );
        }
    }
    line
}
