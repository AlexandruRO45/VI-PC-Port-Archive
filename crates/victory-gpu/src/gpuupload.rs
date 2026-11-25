//! GPU upload heaps on this PC (`VICTORY_GPU_UPLOAD_PROBE`): a probe that
//! says whether the host can let the CPU write straight into the card's
//! memory (Direct3D 12's `D3D12_HEAP_TYPE_GPU_UPLOAD`: the CPU's writes are
//! forwarded over PCIe into video memory, with no system-memory staging
//! copy and no copy command), and what that costs. It changes nothing a run
//! does: it makes buffers of its own, times them, logs, and lets them go.
//!
//! What the Direct3D documentation gives (Agility SDK 1.613 and later, the
//! device's `D3D12_FEATURE_D3D12_OPTIONS16::GPUUploadHeapSupported`):
//!
//! - on a card with its own memory the heap exists only when all of the
//!   card's memory is CPU-visible (Resizable BAR, on in the firmware setup
//!   and exposed by the driver); on a part with one memory it always does;
//! - creating one where it is not supported fails;
//! - the CPU's pages are write-combined, as in upload heaps: sequential
//!   writes are fast, reads are very slow;
//! - the GPU reads it at the card's own speed, unlike an upload heap.
//!
//! What it does not give, and what the probe measures here:
//!
//! - whether a buffer in it may have `ALLOW_UNORDERED_ACCESS`. The title
//!   buffers' host copies are all unordered-access buffers (shaders may
//!   write them, `shadow.rs`), and the copy-back compares their bytes on
//!   the CPU, which write-combined memory makes slow. If unordered access
//!   is refused, or the CPU read is slow, only copies no shader writes
//!   could live here (vertex, index and constant data);
//! - which initial state it takes, the common one or the generic read one;
//! - how fast the CPU writes and reads it, against an upload heap.
//!
//! shortcut: nothing of this is used by the copies yet. The `cards` path
//! (`shadow::card_for`) keeps a card-side buffer beside a system-memory
//! staging copy and moves the changed ranges with copy commands; a GPU
//! upload heap would replace the staging copy and the commands for the
//! copies no shader writes. The probe's verdict says whether that is worth
//! building on this PC.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// `D3D12_HEAP_TYPE_GPU_UPLOAD`, by value, so that this builds against
/// `windows` releases whose generated names do not have it yet.
const GPU_UPLOAD: D3D12_HEAP_TYPE = D3D12_HEAP_TYPE(5);

/// The most the probe reads back from write-combined memory: reads there
/// are slow enough that the whole size would stall the submitting thread.
const READ_BYTES: usize = 4 << 20;

/// `VICTORY_GPU_UPLOAD_PROBE`: set (and not 0), the probe runs once, at the
/// first once-a-second call. `VICTORY_GPU_UPLOAD_PROBE_MB` is the size of
/// the timed buffers (64, at most 1024).
fn probe_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_UPLOAD_PROBE").is_some_and(|v| v != "0"))
}

/// Called once a second by the thread that submits, before a submission
/// (`host::sweep`): with the switch set, runs the probe at the first call
/// that has the host, and logs what it found.
pub(crate) fn tick() {
    static DONE: AtomicBool = AtomicBool::new(false);
    if !probe_on() || DONE.load(Ordering::Relaxed) {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    if DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    let mb = std::env::var("VICTORY_GPU_UPLOAD_PROBE_MB")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(64usize);
    let took = Instant::now();
    for line in probe(&host.device, &|| host.video_memory(), mb.clamp(1, 1024)) {
        tracing::info!("gpu upload probe: {line}");
    }
    tracing::info!(
        "gpu upload probe: took {:.0} ms",
        took.elapsed().as_secs_f64() * 1000.0
    );
}

/// A buffer of `size` bytes in a heap of `kind`.
fn buffer(
    device: &ID3D12Device,
    kind: D3D12_HEAP_TYPE,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: kind,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a committed buffer; the structures live
    // through the call.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut made,
        )?;
    }
    made.ok_or_else(windows::core::Error::empty)
}

/// A GPU upload buffer: the initial state it took (the common one, else the
/// generic read one), with the buffer.
fn gpu_upload(
    device: &ID3D12Device,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
) -> Result<(&'static str, ID3D12Resource), String> {
    let common = buffer(device, GPU_UPLOAD, size, flags, D3D12_RESOURCE_STATE_COMMON);
    let first = match common {
        Ok(b) => return Ok(("common", b)),
        Err(e) => e,
    };
    match buffer(
        device,
        GPU_UPLOAD,
        size,
        flags,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    ) {
        Ok(b) => Ok(("generic read", b)),
        Err(second) => Err(format!(
            "common state: {first}; generic read state: {second}"
        )),
    }
}

/// The CPU's pointer to a buffer's bytes.
fn map(buffer: &ID3D12Resource) -> Option<*mut u8> {
    let mut data = std::ptr::null_mut();
    // SAFETY: a live buffer in a CPU-visible heap; the mapping stays until
    // the buffer is released.
    unsafe { buffer.Map(0, None, Some(&mut data)) }.ok()?;
    (!data.is_null()).then_some(data.cast())
}

/// Gigabytes per second the CPU writes `source` into `data` at.
///
/// # Safety
/// `data` must be a mapping of at least `source.len()` bytes.
unsafe fn write_rate(data: *mut u8, source: &[u8]) -> f64 {
    let started = Instant::now();
    // SAFETY: by this function's contract.
    unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), data, source.len()) };
    source.len() as f64 / started.elapsed().as_secs_f64().max(1e-9) / 1e9
}

/// Gigabytes per second the CPU reads `len` bytes of `data` at.
///
/// # Safety
/// `data` must be a mapping of at least `len` bytes.
unsafe fn read_rate(data: *const u8, len: usize) -> f64 {
    let mut into = vec![0u8; len];
    let started = Instant::now();
    // SAFETY: by this function's contract.
    unsafe { std::ptr::copy_nonoverlapping(data, into.as_mut_ptr(), len) };
    let took = started.elapsed().as_secs_f64().max(1e-9);
    std::hint::black_box(&into);
    len as f64 / took / 1e9
}

/// The lines the probe has to say. `memory` gives the card's memory
/// account (`Host::video_memory`: budget and use of the card's memory, then
/// of the PC's shared memory); `mb` is the size of the timed buffers.
/// Nothing of the title's is touched.
pub(crate) fn probe(
    device: &ID3D12Device,
    memory: &dyn Fn() -> Option<[u64; 4]>,
    mb: usize,
) -> Vec<String> {
    let mut lines = Vec::new();

    let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE::default();
    // SAFETY: a plain query that fills our structure.
    let arch_ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE,
            (&raw mut arch).cast(),
            size_of_val(&arch) as u32,
        )
    }
    .is_ok();
    lines.push(format!(
        "adapter architecture (query ok {arch_ok}): one memory for CPU and GPU {}, cache coherent {}",
        arch.UMA.as_bool(),
        arch.CacheCoherentUMA.as_bool()
    ));

    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: a plain query that fills our structure.
    let options_ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    }
    .is_ok();
    let supported = options_ok && options.GPUUploadHeapSupported.as_bool();
    lines.push(format!(
        "GPUUploadHeapSupported: {supported} (query ok {options_ok}; on a card with its own memory this needs Resizable BAR on in the firmware setup and in the driver)"
    ));
    if !supported {
        lines.push(
            "VERDICT: no GPU upload heap here; the card side of a copy stays a default-heap buffer filled by copy commands from a system-memory staging copy, as now"
                .to_owned(),
        );
        return lines;
    }

    let size = (mb as u64) << 20;
    let none = D3D12_RESOURCE_FLAG_NONE;
    let before = memory();

    let plain = gpu_upload(device, size, none);
    let uav = gpu_upload(device, size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    let describe = |what: &str, made: &Result<(&'static str, ID3D12Resource), String>| match made {
        Ok((state, _)) => format!("{what}: made, initial state {state}"),
        Err(why) => format!("{what}: refused ({why})"),
    };
    lines.push(describe(&format!("{mb} MiB buffer"), &plain));
    lines.push(describe(
        &format!("{mb} MiB buffer with unordered access"),
        &uav,
    ));

    if let (Some(before), Some(after)) = (before, memory()) {
        // The system's account may lag the creation by a moment.
        lines.push(format!(
            "the card's memory in use went {} MiB -> {} MiB (budget {} MiB); the PC's shared memory {} MiB -> {} MiB",
            before[1] >> 20,
            after[1] >> 20,
            after[0] >> 20,
            before[3] >> 20,
            after[3] >> 20
        ));
    }

    // Speeds, on the plain buffer, against an upload heap's.
    let (mut write_gpu, mut read_gpu) = (None, None);
    if let Ok((_, b)) = &plain
        && let Some(data) = map(b)
    {
        let source = vec![0x5au8; size as usize];
        // SAFETY: the mapping is of `size` bytes, the source is as long, and
        // the read is at most that.
        unsafe {
            write_gpu = Some(write_rate(data, &source));
            read_gpu = Some(read_rate(data, READ_BYTES.min(size as usize)));
        }
    }
    let mut write_upload = None;
    if let Ok(b) = buffer(
        device,
        D3D12_HEAP_TYPE_UPLOAD,
        size,
        none,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    ) && let Some(data) = map(&b)
    {
        let source = vec![0x5au8; size as usize];
        // SAFETY: as above.
        write_upload = Some(unsafe { write_rate(data, &source) });
    }
    let rate = |r: Option<f64>| match r {
        Some(r) => format!("{r:.2} GB/s"),
        None => "not measured".to_owned(),
    };
    lines.push(format!(
        "CPU writes of {mb} MiB, one pass: GPU upload heap {}, upload heap {}; CPU read of {} MiB from the GPU upload heap: {}",
        rate(write_gpu),
        rate(write_upload),
        READ_BYTES.min(size as usize) >> 20,
        rate(read_gpu)
    ));

    lines.push(format!(
        "VERDICT: GPU upload heap usable {}, with unordered access {}. Copies no shader writes (vertex, index, constant data) could be filled in the card's memory with no staging copy and no copy command; copies shaders write cannot unless unordered access is allowed, and their copy-back reads on the CPU, which costs what the read rate above says. One pass on one machine: repeat before trusting the numbers",
        plain.is_ok(),
        uav.is_ok()
    ));
    lines
}
