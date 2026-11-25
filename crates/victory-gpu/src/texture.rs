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

use std::sync::{Arc, Mutex};

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

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: u64,
    data: *mut u8,
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

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not tracked), with room to read tracked memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    read: Vec<u8>,
    // True only if the previous tracked refresh recorded a bindless upload.
    band_base: bool,
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

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
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
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    record(list, resource, texture, &staging, None);
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn upload_changed(list: &HostList, resource: This, texture: &ID3D12Resource) -> bool {
    if is_host_written(resource) {
        return false;
    }
    let Some(staging) = staging(resource, texture) else {
        return false;
    };
    let (changed, band) = staging.refresh_for_upload(bands_enabled());
    if !changed {
        return false;
    }
    record(list, resource, texture, &staging, band);
    list.keep(&staging.buffer);
    true
}

/// Copies the selected band, or every subresource, into the host texture,
/// moving it to the copy destination state first.
fn record(
    list: &HostList,
    resource: This,
    texture: &ID3D12Resource,
    staging: &Staging,
    band: Option<RowBand>,
) {
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        copy_band(&list.list, &to, &from, footprint.Footprint.Width, band);
    }
}

fn copy_band(
    list: &ID3D12GraphicsCommandList,
    to: &D3D12_TEXTURE_COPY_LOCATION,
    from: &D3D12_TEXTURE_COPY_LOCATION,
    width: u32,
    band: Option<RowBand>,
) {
    let bounds = band.map(|b| D3D12_BOX {
        left: 0,
        top: b.top,
        front: 0,
        right: width,
        bottom: b.bottom,
        back: 1,
    });
    let y = band.map_or(0, |b| b.top);
    // SAFETY: an open list; both locations name live resources, and the
    // region is in bounds in both copies and aligned to compressed blocks.
    unsafe { list.CopyTextureRegion(to, 0, y, 0, from, bounds.as_ref().map(std::ptr::from_ref)) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowBand {
    top: u32,
    bottom: u32,
}

/// Convert a tracked byte envelope to a conservative full-width band. In
/// non-tail tiled 2D textures, one tile row spans pitch / block width
/// consecutive tiles. Changes in row or slice padding may expand the band,
/// but can never exclude an in-bounds texel.
fn dirty_band(
    layout: &victory_formats::texture::Layout,
    dirty: std::ops::Range<usize>,
) -> Option<RowBand> {
    use victory_formats::texture::{Dimension, Swizzle};
    let [m] = layout.mips.as_slice() else {
        return None;
    };
    let d = &layout.desc;
    if !layout.addressable()
        || d.dimension != Dimension::Tex2d
        || d.samples != 1
        || layout.slices != 1
        || m.depth != 1
        || m.tail.is_some()
        || !d.width.is_multiple_of(d.block)
        || !d.height.is_multiple_of(d.block)
    {
        return None;
    }
    let (group_rows, stride) = if d.swizzle == Swizzle::LINEAR {
        (1, u64::from(m.pitch) * u64::from(d.element_bytes))
    } else {
        let [bw, bh, _] = layout.block;
        (bh, u64::from(m.pitch / bw) * layout.alignment)
    };
    let low = (dirty.start as u64).checked_sub(m.offset)?;
    let high = (dirty.end.checked_sub(1)? as u64).checked_sub(m.offset)?;
    if dirty.is_empty() || stride == 0 {
        return None;
    }
    let texels = u64::from(group_rows) * u64::from(d.block);
    let top = u32::try_from((low / stride) * texels).ok()?;
    let bottom = ((high / stride + 1) * texels).min(u64::from(d.height)) as u32;
    // Keep near-full copies on the regular path rather than issue a box.
    (top < bottom && bottom - top <= d.height / 2).then_some(RowBand { top, bottom })
}

fn bands_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("VICTORY_TEXTURE_UPLOAD_BANDS").is_some_and(|v| v == "1"))
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all): whether it did.
    pub(crate) fn refresh(&self) -> bool {
        self.refresh_for_upload(false).0
    }

    fn refresh_for_upload(&self, use_bands: bool) -> (bool, Option<RowBand>) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, changed, band) = self.refresh_now(use_bands);
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        (changed, band)
    }

    /// Bytes read, whether anything changed, and an optional upload band.
    fn refresh_now(&self, use_bands: bool) -> (usize, bool, Option<RowBand>) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            let dirty = t.take(start, start + size);
            if dirty.is_empty() {
                return (0, false, None);
            }
            let band = if use_bands
                && last.band_base
                && matches!(self.footprints.as_slice(), [f] if
                    f.Footprint.Width == self.layout.desc.width
                        && f.Footprint.Height == self.layout.desc.height
                        && f.Footprint.Depth == 1)
            {
                let from = dirty.first().unwrap().0 - start;
                let to = dirty.last().unwrap().1 - start;
                dirty_band(&self.layout, from..to)
            } else {
                None
            };
            // A deferred eager copy might have refreshed staging without a
            // bindless upload. In that case, the next changed bindless upload
            // takes the original full-copy path.
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            let complete = self.detile(&last.read);
            last.band_base = use_bands && complete;
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, true, band.filter(|_| complete));
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, false, None);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, false, None);
        }
        last.hash = Some(hash);
        let _ = self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true, None)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) -> bool {
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
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return false;
            }
        }
        true
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
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
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    let s = Arc::new(Staging {
        buffer,
        size: total,
        data,
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

    fn layout(
        width: u32,
        height: u32,
        block: u32,
        swizzle: victory_formats::texture::Swizzle,
    ) -> victory_formats::texture::Layout {
        use victory_formats::texture::{Desc, Dimension, Layout};
        Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width,
            height,
            depth: 1,
            mips: 1,
            element_bytes: if block == 4 { 16 } else { 4 },
            block,
            samples: 1,
            swizzle,
        })
        .unwrap()
    }

    #[test]
    fn dirty_bands_enclose_mapped_elements() {
        use victory_formats::texture::Swizzle;
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_D] {
            for block in [1, 4] {
                let l = layout(132, 1024, block, swizzle);
                let m = &l.mips[0];
                for y in [0, 31, m.height / 2, m.height - 1] {
                    for x in [0, m.width / 2, m.width - 1] {
                        let at = l.offset(0, x, y, 0).unwrap() as usize;
                        let b = dirty_band(&l, at..at + l.desc.element_bytes as usize).unwrap();
                        assert!(
                            b.top <= y * block && b.bottom >= (y + 1) * block,
                            "{swizzle:?}, {block}, {x}, {y}: {b:?}"
                        );
                        assert_eq!(b.top % block, 0);
                        assert_eq!(b.bottom % block, 0);
                    }
                }
                assert_eq!(dirty_band(&l, 0..l.size as usize), None);
                assert_eq!(dirty_band(&l, 123..123), None);
            }
        }
    }

    #[test]
    fn tracked_refresh_keeps_first_and_uncertain_uploads_full() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let Some(host) = host() else {
            assert!(
                std::env::var_os("VICTORY_BAND_BENCH").is_none(),
                "D3D12 unavailable"
            );
            return;
        };
        let l = layout(512, 512, 1, victory_formats::texture::Swizzle::SW_64KB_D);
        // SAFETY: create a write-watched range for this isolated texture.
        let mem = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                l.size as usize,
                MEM_RESERVE | MEM_COMMIT | 0x0020_0000,
                PAGE_READWRITE,
            )
        };
        assert!(!mem.is_null());
        let (buffer, data) = host.upload_buffer(512 * 512 * 4).unwrap();
        let at = l.offset(0, 91, 301, 0).unwrap() as usize;
        let staging = Staging {
            buffer,
            size: 512 * 512 * 4,
            data,
            footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Width: 512,
                    Height: 512,
                    Depth: 1,
                    RowPitch: 2048,
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                },
            }],
            rows: vec![512],
            layout: l,
            mips: 1,
            memory: mem.cast(),
            tracked: Some(crate::watch::track(mem as usize, 512 * 512 * 4, true).unwrap()),
            last: Mutex::default(),
        };
        assert_eq!(staging.refresh_for_upload(true), (true, None));
        assert_eq!(staging.refresh_for_upload(true), (false, None));
        // SAFETY: both offsets refer to the live test range; the write tracker handles the fault.
        unsafe {
            mem.cast::<u8>().add(at).write_volatile(73);
        }
        let (changed, band) = staging.refresh_for_upload(true);
        assert!(changed);
        let region = band.unwrap();
        assert!((region.top..region.bottom).contains(&301));
        // SAFETY: live tracked allocation.
        unsafe {
            mem.cast::<u8>().add(at).write_volatile(74);
        }
        assert!(staging.refresh());
        // SAFETY: live tracked allocation.
        unsafe {
            mem.cast::<u8>().add(at).write_volatile(75);
        }
        assert_eq!(staging.refresh_for_upload(true), (true, None));
        drop(staging);
        // SAFETY: no tracker or references remain for this allocation.
        assert_ne!(unsafe { VirtualFree(mem, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn gpu_texture_band_preserves_bytes_outside_the_box() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        let Some(host) = host() else {
            assert!(
                std::env::var_os("VICTORY_BAND_BENCH").is_none(),
                "D3D12 unavailable"
            );
            return;
        };
        let bench = std::env::var_os("VICTORY_BAND_BENCH").is_some();
        let side = if bench { 4096u32 } else { 256u32 };
        let bytes = u64::from(side) * u64::from(side) * 4;
        let (upload_a, a) = host.upload_buffer(bytes).unwrap();
        let (upload_b, b) = host.upload_buffer(bytes).unwrap();
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(side),
            Height: side,
            MipLevels: 1,
            DepthOrArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let buffer_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            MipLevels: 1,
            DepthOrArraySize: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let make = |d: &D3D12_RESOURCE_DESC, ty| -> ID3D12Resource {
            let mut out = None;
            // SAFETY: descriptors live through the call; no optimized clear value.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: ty,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        d,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut out,
                    )
                    .unwrap();
            }
            out.unwrap()
        };
        let image = make(&texture_desc, D3D12_HEAP_TYPE_DEFAULT);
        let readback = make(&buffer_desc, D3D12_HEAP_TYPE_READBACK);
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Width: side,
                Height: side,
                Depth: 1,
                RowPitch: side * 4,
            },
        };
        let loc = |r: &ID3D12Resource, buf| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: the location borrows its resource for recording only.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: if buf {
                D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT
            } else {
                D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
            },
            Anonymous: if buf {
                D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                }
            } else {
                D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                }
            },
        };
        let l = layout(side, side, 1, victory_formats::texture::Swizzle::SW_64KB_D);
        let at = l.offset(0, 0, side / 2, 0).unwrap() as usize;
        let page = at / 65536 * 65536;
        let band = dirty_band(&l, page..page + 65536).unwrap();
        // SAFETY: the two upload mappings remain live and are `bytes` long.
        unsafe {
            std::ptr::write_bytes(a, 17, bytes as usize);
            std::ptr::write_bytes(b, 83, bytes as usize);
        }
        let list = host.new_list().unwrap();
        copy_band(
            &list.list,
            &loc(&image, false),
            &loc(&upload_a, true),
            side,
            None,
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
        copy_band(
            &list.list,
            &loc(&image, false),
            &loc(&upload_b, true),
            side,
            Some(band),
        );
        // SAFETY: open list, live textures, and correct initial and target states.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                &image,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list
                .CopyTextureRegion(&loc(&readback, true), 0, 0, 0, &loc(&image, false), None);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: execute waited for the copy; the readback holds `bytes` initialized bytes.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            for (y, row) in std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize)
                .chunks_exact(side as usize * 4)
                .enumerate()
            {
                let expected = if (band.top..band.bottom).contains(&(y as u32)) {
                    83
                } else {
                    17
                };
                assert!(row.iter().all(|&x| x == expected), "row {y}");
            }
            readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        }
        if !bench {
            return;
        }
        list.reset();
        // SAFETY: the prior command left the texture in COPY_SOURCE.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                &image,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
        }
        list.close();
        host.execute(&[&list]);
        for i in 0..24 {
            for partial in [i % 2 == 1, i % 2 == 0] {
                list.reset();
                let start = std::time::Instant::now();
                copy_band(
                    &list.list,
                    &loc(&image, false),
                    &loc(&upload_b, true),
                    side,
                    partial.then_some(band),
                );
                list.close();
                host.execute(&[&list]);
                println!(
                    "BAND_RESULT,{i},{partial},{},{:.2}",
                    if partial {
                        u64::from(band.bottom - band.top) * u64::from(side) * 4
                    } else {
                        bytes
                    },
                    start.elapsed().as_secs_f64() * 1e6
                );
            }
        }
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}
