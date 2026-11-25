//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, `external` host
//! libraries recording into title lists. Planned:
//! `present` (swap chain).

#[cfg(windows)]
mod bindless;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
pub mod list;
#[cfg(windows)]
mod memory;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
mod switch;
#[cfg(windows)]
mod tdr;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod twin;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::{crash_detail, on_failure};

/// Creates the host D3D12 device (wave size, NVIDIA view-heap cap) before
/// shaders are translated or the shader cache folder is chosen. AMD heaps
/// can be larger than 1_000_000 views and waves can be 64; NVIDIA (RTX 40/50)
/// is 32-lane waves and a 1_000_000-view cap. Translations must see that.
#[cfg(windows)]
pub fn prepare_host() {
    let _ = host::host();
    if let Some(n) = std::env::var("VICTORY_HOST_MAX_WAVE")
        .ok()
        .or_else(|| std::env::var("VICTORY_HOST_WAVE").ok())
        .and_then(|v| v.parse().ok())
    {
        victory_shader::emit::set_host_max_wave(n);
    }
    if let Some(n) = std::env::var("VICTORY_VIEW_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        victory_shader::emit::set_view_heap_limit(n);
        tracing::info!(null_slot = n, "VICTORY_VIEW_LIMIT: view indices clamped");
    } else {
        descriptor::host::probe_view_limit();
    }
}

#[cfg(not(windows))]
pub fn prepare_host() {}
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
#[cfg(windows)]
pub use watch::{
    Writing as KernelWrite, title_memory_aliased, title_memory_reading, title_memory_remapped,
    title_memory_remapping, writing as kernel_will_write,
};
