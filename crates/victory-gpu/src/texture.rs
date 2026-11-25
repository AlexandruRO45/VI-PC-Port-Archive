//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// The shared host texture's upload layout and last successful refresh.
/// Only metadata survives the submission, not the upload heap or read bytes.
pub(crate) struct Staging {
    size: usize,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    last: Mutex<Last>,
}

#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    /// Exact bytes of each journal page, including the cut first/last pages.
    /// Published only after a successful upload, never after an OOM.
    pages: Vec<blake3::Hash>,
    uploaded: bool,
    retry: bool,
}

fn staging_margin(enabled: Option<&str>, mb: Option<&str>) -> Option<u64> {
    if enabled == Some("0") { return None; }
    Some(mb.and_then(|v| v.parse::<u64>().ok()).unwrap_or(256).saturating_mul(1 << 20))
}

/// Remaining space for one staging allocation, or None to keep the old path.
fn headroom(margin: Option<u64>, query: impl FnOnce() -> Option<(u64, u64)>) -> Option<u64> {
    let margin = margin?; // Opt-out must not even look up the DXGI adapter.
    let (usage, budget) = query()?;
    // On UMA there may be no NON_LOCAL segment at all.
    if usage == 0 && budget == 0 { return None; }
    Some(budget.saturating_sub(margin).saturating_sub(usage))
}

fn staging_headroom(device: &ID3D12Device) -> Option<u64> {
    static MARGIN: OnceLock<Option<u64>> = OnceLock::new();
    let margin = MARGIN.get_or_init(|| staging_margin(
        std::env::var("VICTORY_STAGING_BUDGET").ok().as_deref(),
        std::env::var("VICTORY_STAGING_BUDGET_MARGIN_MB").ok().as_deref(),
    ));
    headroom(*margin, || {
        // Host has one device for the process. Cache failed samples too.
        type Sample = (Instant, Option<(u64, u64)>);
        static SAMPLE: Mutex<Option<Sample>> = Mutex::new(None);
        let mut sample = SAMPLE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, value)) = *sample && at.elapsed() < Duration::from_millis(100) {
            return value;
        }
        let value = non_local_memory(device);
        *sample = Some((Instant::now(), value));
        value
    })
}

fn non_local_memory(device: &ID3D12Device) -> Option<(u64, u64)> {
    use windows::Win32::Graphics::Dxgi::*;
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    // SAFETY: the host device is live; select its exact adapter.
    let adapter = ADAPTER.get_or_init(|| unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid(device.GetAdapterLuid())).ok()
    }).as_ref()?;
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: valid output for the host's node zero.
    unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut info) }.ok()?;
    Some((info.CurrentUsage, info.Budget))
}

thread_local! {
    static READ: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
    static LINEAR: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
}
const RETAIN_SCRATCH: usize = 16 << 20;

fn room(cell: &'static std::thread::LocalKey<std::cell::Cell<Vec<u8>>>, len: usize) -> Option<Vec<u8>> {
    let mut bytes = cell.take();
    if len > bytes.len() {
        bytes.try_reserve_exact(len - bytes.len()).ok()?;
    }
    bytes.resize(len, 0);
    Some(bytes)
}

fn reuse(cell: &'static std::thread::LocalKey<std::cell::Cell<Vec<u8>>>, bytes: Vec<u8>) {
    if bytes.capacity() <= RETAIN_SCRATCH {
        cell.set(bytes);
    }
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve writes `resource`: its host texture's contents
/// are the host's, not title memory's, which stays as it was.
pub(crate) fn host_written(resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return;
    };
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    crate::texture::host_written(to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// The first use starts in COPY_DEST; its prelude performs the upload only
/// if title memory changed. Keep the shared HostTexture, not a title pointer.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    upload_placed(list, resource, texture, None);
}

pub(crate) fn upload_placed(list: &HostList, resource: This, texture: &ID3D12Resource, placed: Option<&Arc<crate::resource::Placed>>) {
    if placed.is_some_and(|p| crate::resource::was_remapped(p.get())) {
        return;
    }
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) || !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    let Some(copy) = crate::resource::host_copy(resource) else { return };
    crate::list::ensure_state(list, resource, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
    list.defer_texture(staging, copy, placed.cloned());
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn upload_changed(list: &HostList, resource: This, texture: &ID3D12Resource) -> Option<ID3D12Resource> {
    if is_host_written(resource) {
        return None;
    }
    let staging = staging(resource, texture)?;
    let buffer = staging.refresh(host()?)?;
    crate::list::ensure_state(list, resource, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
    staging.record(&list.list, texture, &buffer);
    Some(buffer)
}

/// Split at journal boundaries, with the exact bytes inside the texture.
fn page_ranges(from: usize, to: usize) -> impl Iterator<Item = (usize, usize)> {
    use crate::watch::PAGE;
    let mut at = from;
    std::iter::from_fn(move || {
        if at >= to { return None; }
        let end = at.saturating_add(PAGE - at % PAGE).min(to);
        let range = (at, end);
        at = end;
        Some(range)
    })
}

impl Staging {
    /// Copy each subresource to the texture, already in COPY_DEST.
    pub(crate) fn record(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, buffer: &ID3D12Resource) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed COM pointer; the location never releases it.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: same; caller retains the upload until the fence.
                pResource: unsafe { std::mem::transmute_copy(buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }

    /// Allocate and fill an upload only when there are changed resident bytes.
    pub(crate) fn refresh(&self, host: &crate::host::Host) -> Option<ID3D12Resource> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, changed) = self.refresh_now(host, || staging_headroom(&host.device));
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        changed
    }

    fn refresh_now(&self, host: &crate::host::Host, headroom: impl FnOnce() -> Option<u64>) -> (usize, Option<ID3D12Resource>) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let Some(end) = start.checked_add(size).filter(|end| *end > start) else { return (0, None) };
        if let Some(tracked) = &self.tracked {
            let dirty = tracked.take(start, end);
            if dirty.is_empty() && last.uploaded && !last.retry {
                return (0, None);
            }
            let mut probed = 0;
            // On a small dirty subset, verify only those pages. Keep the
            // full read for initial upload and large or genuinely changed
            // ranges, so detile still sees one complete initialized image.
            if !last.retry && !last.pages.is_empty()
                && dirty.iter().map(|(a, b)| b - a).sum::<usize>() < size / 2
            {
                let mut page = room(&READ, crate::watch::PAGE).unwrap_or_default();
                if !page.is_empty() {
                    let same = dirty.iter().all(|&(a, b)| {
                        page_ranges(a, b).all(|(a, b)| {
                            let out = &mut page[..b - a];
                            crate::host::copy_mapped(out.as_mut_ptr(), a as *const u8, out.len(), false);
                            probed += out.len();
                            last.pages.get(a / crate::watch::PAGE - start / crate::watch::PAGE)
                                == Some(&blake3::hash(out))
                        })
                    });
                    reuse(&READ, page);
                    if same {
                        #[cfg(feature = "diagnostics")]
                        {
                            use crate::copystat::{C, add};
                            add(C::TexWatched, probed as u64);
                            add(C::TexIdentical, probed as u64);
                        }
                        return (probed, None);
                    }
                }
            }
            last.retry = true;
            let Some(mut read) = room(&READ, size) else { return (probed, None) };
            crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
            // The journal is page-granular. Aliases, neighboring allocations
            // and still-reserved lazy pages can flag a texture whose bytes
            // did not change. Hash the actual zero-filled data before taking
            // an UPLOAD allocation or detiling it again.
            let hash = blake3::hash(&read);
            let same = last.hash == Some(hash);
            let upload = if same { None } else { self.stage(host, &read, last.uploaded, headroom) };
            if upload.is_some() {
                last.uploaded = true;
                last.hash = Some(hash);
                if size >= 4 * crate::watch::PAGE {
                    last.pages = page_ranges(start, end)
                        .map(|(a, b)| blake3::hash(&read[a - start..b - start])).collect();
                }
            }
            reuse(&READ, read);
            last.retry = !same && upload.is_none();
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::TexWatched, (size + probed) as u64);
                if same { add(C::TexIdentical, size as u64); }
                if upload.is_some() { add(C::TexChanged, self.size as u64); }
            }
            return (size + probed, upload);
        }
        let Some(mut read) = room(&READ, size) else { return (0, None) };
        let Some((hash, bytes)) = read_resident(self.memory, &mut read) else {
            last.hash = None;
            reuse(&READ, read);
            return (0, None);
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, bytes as u64);
        let upload = if last.hash == Some(hash) { None } else { self.stage(host, &read, last.uploaded, headroom) };
        reuse(&READ, read);
        if upload.is_some() {
            last.uploaded = true;
            last.hash = Some(hash);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexChanged, self.size as u64);
        }
        (bytes, upload)
    }

    fn span(&self, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, rows: u32) -> Option<std::ops::Range<usize>> {
        let at = usize::try_from(fp.Offset).ok()?;
        let padded = (fp.Footprint.RowPitch as usize).checked_mul(rows as usize)?
            .checked_mul(fp.Footprint.Depth as usize)?;
        Some(at..at + padded.min(self.size.checked_sub(at)?))
    }

    fn stage(&self, host: &crate::host::Host, tiled: &[u8], uploaded: bool, headroom: impl FnOnce() -> Option<u64>) -> Option<ID3D12Resource> {
        if uploaded && headroom().is_some_and(|left| self.size as u64 > left) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexDeferred, 1);
            return None;
        }
        let max = self.footprints.iter().zip(&self.rows).try_fold(0usize, |n, (fp, &r)| {
            Some(n.max(self.span(fp, r)?.len()))
        })?;
        let mut linear = room(&LINEAR, max)?;
        let Some((upload, data)) = host.upload_buffer(self.size as u64) else {
            reuse(&LINEAR, linear);
            return None;
        };
        // SAFETY: the new upload mapping is exclusive and has `size` bytes.
        let out = unsafe { std::slice::from_raw_parts_mut(data, self.size) };
        let ok = self.detile(tiled, out, &mut linear);
        reuse(&LINEAR, linear);
        ok.then_some(upload)
    }

    fn detile(&self, tiled: &[u8], out: &mut [u8], scratch: &mut [u8]) -> bool {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let Some(span) = self.span(footprint, rows) else { return false };
            let linear = &mut scratch[..span.len()];
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return false;
            }
            out[span].copy_from_slice(linear);
        }
        true
    }
}

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
    VirtualQuery,
};

const WRITE: u32 = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
const READABLE: u32 = WRITE | PAGE_READONLY | PAGE_EXECUTE_READ;

fn region(at: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: valid output storage; querying does not access the address.
    let queried = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    (queried != 0).then_some(info)
}

fn accessible(info: &MEMORY_BASIC_INFORMATION, mask: u32) -> bool {
    info.State == MEM_COMMIT && info.Protect & PAGE_GUARD == 0 && info.Protect & mask != 0
}

/// A range committed and readable at query time.
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    committed_for(at as usize, len, READABLE)
}

/// A range committed and writable at query time.
pub(crate) fn committed_writable(at: *mut u8, len: usize) -> bool {
    committed_for(at as usize, len, WRITE)
}

fn committed_for(mut at: usize, len: usize, mask: u32) -> bool {
    let Some(end) = at.checked_add(len) else { return false };
    while at < end {
        let Some(info) = region(at).filter(|i| accessible(i, mask)) else { return false };
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize).filter(|&p| p > at) else {
            return false;
        };
        at = next;
    }
    true
}

/// Visit readable, reserved and other unreadable runs without accessing title memory.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Readable { Yes, Reserved, No }

pub(crate) fn read_runs(mut at: usize, len: usize, mut visit: impl FnMut(usize, usize, Readable)) {
    use windows_sys::Win32::System::Memory::MEM_RESERVE;
    let Some(end) = at.checked_add(len) else {
        visit(at, len, Readable::No);
        return;
    };
    while at < end {
        let Some(info) = region(at) else { break };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if next <= at { break; }
        let state = if accessible(&info, READABLE) { Readable::Yes }
            else if info.State == MEM_RESERVE { Readable::Reserved } else { Readable::No };
        visit(at, next - at, state);
        at = next;
    }
    if at < end {
        visit(at, end - at, Readable::No);
    }
}

/// Common case: one checked kernel read. On a partially mapped texture,
/// hash run bounds and readable bytes only, using zeros for the holes.
fn read_resident(src: *const u8, out: &mut [u8]) -> Option<(blake3::Hash, usize)> {
    if crate::host::read_into(src, out) {
        return Some((blake3::hash(out), out.len()));
    }
    let base = src as usize;
    let mut bytes = 0;
    let mut hash = blake3::Hasher::new();
    hash.update(b"partial texture");
    read_runs(base, out.len(), |at, len, state| {
        let readable = state == Readable::Yes;
        let start = at - base;
        let part = &mut out[start..start + len];
        if readable {
            let mut record = |offset: usize, p: &[u8]| {
                hash.update(&offset.to_le_bytes());
                hash.update(&p.len().to_le_bytes());
                hash.update(p);
                bytes += p.len();
            };
            if crate::host::read_into(at as *const u8, part) {
                record(start, part);
            } else {
                // Concurrent unmap within a region: retry individual pages.
                let mut offset = start;
                let mut p = part;
                while !p.is_empty() {
                    let n = (4096 - (base + offset) % 4096).min(p.len());
                    let (page, tail) = p.split_at_mut(n);
                    if crate::host::read_into((base + offset) as *const u8, page) {
                        record(offset, page);
                    } else {
                        page.fill(0);
                    }
                    offset += n;
                    p = tail;
                }
            }
        } else {
            part.fill(0);
        }
    });
    (bytes != 0).then(|| (hash.finalize(), bytes))
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let size = usize::try_from(total).ok().filter(|&n| n > 0 && n <= isize::MAX as usize)?;
    tracing::debug!(?desc, total, "title texture staging layout");
    let layout_size = layout.size as usize;
    let s = Arc::new(Staging {
        size,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        last: Mutex::default(),
    });
    *held = Some(s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree, VirtualProtect};

    #[test]
    fn detile_does_not_write_the_240_padding_bytes_outside_the_footprint() {
        let layout = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            format: 28, // R8G8B8A8_UNORM
            width: 4, height: 4, depth_or_array: 1, mips: 1, samples: 1, swizzle: Swizzle::LINEAR,
        }).unwrap().planes[0].layout.clone();
        let source = vec![0x57; layout.size as usize];
        let staging = Staging {
            size: 784,
            footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Width: 4, Height: 4, Depth: 1, RowPitch: 256, ..Default::default() },
            }],
            rows: vec![4], layout, mips: 1, memory: source.as_ptr(), tracked: None, last: Mutex::default(),
        };
        let mut upload = [0xcc; 1024];
        assert!(staging.detile(&source, &mut upload, &mut [0; 784]));
        for y in 0..4 {
            assert_eq!(&upload[y * 256..y * 256 + 16], &[0x57; 16]);
        }
        assert_eq!(&upload[784..], &[0xcc; 240]);
    }

    #[test]
    fn watched_texture_skips_false_dirty_uploads_but_uploads_real_changes() {
        let host = host().expect("D3D12 host");
        let at = crate::watch::tests::pool(1);
        let layout = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            format: 28,
            width: 64, height: 64, depth_or_array: 1, mips: 1, samples: 1, swizzle: Swizzle::LINEAR,
        }).unwrap().planes[0].layout.clone();
        let size = layout.size as usize;
        assert!(size < crate::watch::PAGE);
        let staging = Staging {
            size: 16384,
            footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Width: 64, Height: 64, Depth: 1, RowPitch: 256, ..Default::default() },
            }],
            rows: vec![64], layout, mips: 1, memory: at as *const u8,
            tracked: Some(crate::watch::track(at, size, true).unwrap()), last: Mutex::default(),
        };
        assert!(staging.refresh_now(host, || Some(0)).1.is_some());
        assert_eq!(staging.refresh_now(host, || None).0, 0, "clean page needs no read");
        let mut reads = 0;
        for _ in 0..32 {
            // Write after the texture's end but in its tracked page.
            // SAFETY: the fixture owns the whole page.
            unsafe { (at as *mut u8).add(size).write_volatile(1); }
            let (read, upload) = staging.refresh_now(host, || None);
            reads += read;
            assert_eq!(read, size);
            assert!(upload.is_none());
        }
        // SAFETY: in the actual texture, unlike the writes above.
        unsafe { (at as *mut u8).write_volatile(0x5a); }
        let previous = staging.last.lock().unwrap().hash.unwrap();
        assert!(staging.refresh_now(host, || headroom(staging_margin(None, None), || Some((512 << 20, 768 << 20)))).1.is_none());
        {
            let last = staging.last.lock().unwrap();
            assert!(last.retry);
            assert_eq!(last.hash, Some(previous)); // no copy recorded; previous contents remain published
        }
        // No intervening write: the consumed journal entry must not lose the change.
        assert!(staging.refresh_now(host, || Some(staging.size as u64)).1.is_some());
        assert!(!staging.last.lock().unwrap().retry);
        assert_ne!(staging.last.lock().unwrap().hash, Some(previous));
        eprintln!("watched fixture: {reads} bytes read, 0 uploaded (previously {} bytes of upload allocations)", 32 * staging.size);
        drop(staging);
        // SAFETY: no tracker remains on the owned pool.
        assert_ne!(unsafe { VirtualFree(at as _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn watched_large_texture_probes_only_its_dirty_page() {
        use crate::watch::PAGE;
        let host = host().expect("D3D12 host");
        let pool = crate::watch::tests::pool(9);
        let at = pool + 256;
        let layout = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            format: 28,
            width: 512, height: 256, depth_or_array: 1, mips: 1, samples: 1, swizzle: Swizzle::LINEAR,
        }).unwrap().planes[0].layout.clone();
        let size = layout.size as usize;
        let staging = Staging {
            size: 512 * 256 * 4,
            footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Width: 512, Height: 256, Depth: 1, RowPitch: 2048, ..Default::default() },
            }],
            rows: vec![256], layout, mips: 1, memory: at as *const u8,
            tracked: Some(crate::watch::track(at, size, true).unwrap()), last: Mutex::default(),
        };
        assert!(staging.refresh_now(host, || None).1.is_some());
        // SAFETY: a neighboring byte of the test's first journal page.
        unsafe { (pool as *mut u8).write_volatile(0x5a); }
        let (read, upload) = staging.refresh_now(host, || None);
        assert_eq!(read, PAGE - 256);
        assert!(upload.is_none());
        // SAFETY: the first byte inside the texture, on the same journal page.
        unsafe { (at as *mut u8).write_volatile(0x5a); }
        assert!(staging.refresh_now(host, || None).1.is_some());
        eprintln!("watched sparse fixture: {size} -> {read} source bytes; no upload for unchanged texels");
        drop(staging);
        // SAFETY: no tracker remains on the owned pool.
        assert_ne!(unsafe { VirtualFree(pool as _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn untracked_budget_retries_and_opt_out_uploads_the_same_bytes() {
        let host = host().expect("D3D12 host");
        let layout = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d, format: 28, width: 64, height: 64,
            depth_or_array: 1, mips: 1, samples: 1, swizzle: Swizzle::LINEAR,
        }).unwrap().planes[0].layout.clone();
        let mut source = vec![0x5a; layout.size as usize];
        let staging = Staging {
            size: 64 * 256, footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Width: 64, Height: 64, Depth: 1, RowPitch: 256, ..Default::default()
                },
            }], rows: vec![64], layout, mips: 1, memory: source.as_ptr(), tracked: None, last: Mutex::default(),
        };
        let bytes = |upload: &ID3D12Resource| {
            let mut data = std::ptr::null_mut();
            // SAFETY: a live mappable upload of staging.size bytes. Balance this extra Map.
            unsafe {
                upload.Map(0, None, Some(&mut data)).unwrap();
                let bytes = std::slice::from_raw_parts(data.cast::<u8>(), staging.size).to_vec();
                upload.Unmap(0, Some(&D3D12_RANGE::default()));
                bytes
            }
        };
        let full = || headroom(staging_margin(None, None), || Some((768 << 20, 768 << 20)));
        let first = staging.refresh_now(host, full).1.unwrap();
        source.fill(0xaa);
        assert!(staging.refresh_now(host, full).1.is_none());
        assert_eq!(bytes(&first), vec![0x5a; staging.size]);
        let second = staging.refresh_now(host, || Some(staging.size as u64)).1.unwrap();
        assert_eq!(bytes(&second), vec![0xaa; staging.size]);
        source.fill(0x5a);
        let off = || headroom(staging_margin(Some("0"), None), || panic!("queried when disabled"));
        let third = staging.refresh_now(host, off).1.unwrap();
        assert_eq!(bytes(&third), bytes(&first));
        assert!(staging.refresh_now(host, off).1.is_none());
    }

    #[test]
    fn partial_texture_read_and_access_permissions() {
        const P: usize = 4096;
        // SAFETY: owned synthetic reservation, committed and protected only here.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), 2 * P, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!at.is_null());
            assert!(!VirtualAlloc(at.cast(), P, MEM_COMMIT, PAGE_READWRITE).is_null());
            at.write_bytes(0x99, P);
            let mut out = [0x31; 2 * P];
            let (partial, n) = read_resident(at, &mut out).unwrap();
            assert_eq!(n, P);
            assert_eq!(&out[..P], &[0x99; P]);
            assert_eq!(&out[P..], &[0; P]);
            assert!(!committed(at, 2 * P));
            assert!(!VirtualAlloc(at.add(P).cast(), P, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert_ne!(partial, read_resident(at, &mut out).unwrap().0);
            assert_ne!(VirtualProtect(at.cast(), P, PAGE_READONLY, &mut 0), 0);
            assert!(committed(at, 2 * P));
            assert!(!committed_writable(at, 1));
            assert!(committed_writable(at.add(P), P));
            assert!(!committed(usize::MAX as *const u8, 2));
            assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
        }
    }
}
