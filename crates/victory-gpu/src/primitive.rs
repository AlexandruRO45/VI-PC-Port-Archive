//! Console primitive types the host lacks. A rectangle list gives three
//! corners per rectangle, the second the one between the other two, and
//! the hardware adds the fourth at `v0 + v2 - v1` (Grand Theft Auto VI's
//! full-screen rectangles come as top left, bottom left, bottom right); on
//! the host each rectangle becomes two triangles, with the fourth vertex
//! made from the other three on the CPU.
//!
//! shortcut: the fourth vertex is computed from vertex inputs, not from
//! vertex shader outputs as the hardware does; that is the same for vertex
//! shaders linear in their inputs (blits, 2D quads). Inputs in formats
//! [`components`] does not know keep the first vertex's value, and so do
//! components past the end of the vertex (an element that reaches into the
//! next one, which the layout allows).

/// Console-only primitive topologies expanded into standard D3D12 triangle or
/// line list index buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpandedTopology {
    TriangleFan,
    LineLoop,
    Quads,
    QuadStrip,
}

impl ExpandedTopology {
    const TRIANGLE_FAN: u32 = 6;
    const LINE_LOOP: u32 = 18;
    const QUADS: u32 = 19;
    const QUAD_STRIP: u32 = 20;

    /// Returns the expanded console topology kind, or `None` for standard PC
    /// topologies and `RECTANGLES` (handled via vertex expansion).
    pub(crate) fn from_console(topology: u32) -> Option<Self> {
        match topology {
            Self::TRIANGLE_FAN => Some(Self::TriangleFan),
            Self::LINE_LOOP => Some(Self::LineLoop),
            Self::QUADS => Some(Self::Quads),
            Self::QUAD_STRIP => Some(Self::QuadStrip),
            _ => None,
        }
    }

    /// Standard D3D12 primitive topology value (`LINELIST = 2`, `TRIANGLELIST = 4`).
    pub(crate) fn host_topology(self) -> u32 {
        match self {
            Self::LineLoop => 2,
            Self::TriangleFan | Self::Quads | Self::QuadStrip => 4,
        }
    }

    /// Maximum number of indices emitted for `count` input vertices or indices.
    pub(crate) fn max_output_indices(self, count: u32) -> u32 {
        match self {
            Self::TriangleFan => count.saturating_sub(2).saturating_mul(3),
            Self::LineLoop => {
                if count >= 2 {
                    count.saturating_mul(2)
                } else {
                    0
                }
            }
            Self::Quads => (count / 4).saturating_mul(6),
            Self::QuadStrip => {
                if count >= 4 {
                    ((count - 2) / 2).saturating_mul(6)
                } else {
                    0
                }
            }
        }
    }

    /// Expands `count` sequential 16-bit vertex indices starting at `start`.
    pub(crate) fn expand_non_indexed_u16(self, count: u32, start: u16, dst: &mut [u16]) -> usize {
        expand_indices(
            self,
            (0..count).map(|i| start.wrapping_add(i as u16)),
            None,
            start,
            dst,
        )
    }

    /// Expands `count` sequential 32-bit vertex indices starting at `start`.
    pub(crate) fn expand_non_indexed_u32(self, count: u32, start: u32, dst: &mut [u32]) -> usize {
        expand_indices(
            self,
            (0..count).map(|i| start.wrapping_add(i)),
            None,
            start,
            dst,
        )
    }

    /// Expands 16-bit indices from `src`, honoring optional strip-cut `cut` and
    /// padding unused tail slots with degenerate primitives.
    #[cfg(test)]
    pub(crate) fn expand_indexed_u16(
        self,
        src: &[u16],
        cut: Option<u16>,
        fallback_pad: u16,
        dst: &mut [u16],
    ) -> usize {
        expand_indices(self, src.iter().copied(), cut, fallback_pad, dst)
    }

    /// Expands 32-bit indices from `src`, honoring optional strip-cut `cut` and
    /// padding unused tail slots with degenerate primitives.
    #[cfg(test)]
    pub(crate) fn expand_indexed_u32(
        self,
        src: &[u32],
        cut: Option<u32>,
        fallback_pad: u32,
        dst: &mut [u32],
    ) -> usize {
        expand_indices(self, src.iter().copied(), cut, fallback_pad, dst)
    }

    #[cfg(windows)]
    fn kind_id(self) -> u32 {
        match self {
            Self::TriangleFan => 0,
            Self::LineLoop => 1,
            Self::Quads => 2,
            Self::QuadStrip => 3,
        }
    }
}

fn expand_indices<T: Copy + Eq>(
    kind: ExpandedTopology,
    indices: impl IntoIterator<Item = T>,
    cut: Option<T>,
    fallback_pad: T,
    dst: &mut [T],
) -> usize {
    let mut written = 0usize;
    let mut emit = |slice: &[T]| {
        if written + slice.len() <= dst.len() {
            dst[written..written + slice.len()].copy_from_slice(slice);
            written += slice.len();
        }
    };
    match kind {
        ExpandedTopology::TriangleFan => {
            let mut pivot = None;
            let mut prev = None;
            for v in indices {
                if cut == Some(v) {
                    pivot = None;
                    prev = None;
                    continue;
                }
                match (pivot, prev) {
                    (None, _) => pivot = Some(v),
                    (Some(_), None) => prev = Some(v),
                    (Some(p), Some(q)) => {
                        emit(&[p, q, v]);
                        prev = Some(v);
                    }
                }
            }
        }
        ExpandedTopology::LineLoop => {
            let mut first = None;
            let mut prev = None;
            let mut run_len = 0usize;
            for v in indices {
                if cut == Some(v) {
                    if run_len >= 2
                        && let (Some(f), Some(p)) = (first, prev)
                    {
                        emit(&[p, f]);
                    }
                    first = None;
                    prev = None;
                    run_len = 0;
                    continue;
                }
                if let Some(p) = prev {
                    emit(&[p, v]);
                } else {
                    first = Some(v);
                }
                prev = Some(v);
                run_len += 1;
            }
            if run_len >= 2
                && let (Some(f), Some(p)) = (first, prev)
            {
                emit(&[p, f]);
            }
        }
        ExpandedTopology::Quads => {
            let mut q = [None; 3];
            let mut len = 0usize;
            for v in indices {
                if cut == Some(v) {
                    len = 0;
                    continue;
                }
                if len < 3 {
                    q[len] = Some(v);
                    len += 1;
                } else if let [Some(v0), Some(v1), Some(v2)] = q {
                    emit(&[v0, v1, v2, v0, v2, v]);
                    len = 0;
                }
            }
        }
        ExpandedTopology::QuadStrip => {
            let mut a = None;
            let mut b = None;
            let mut c = None;
            for v in indices {
                if cut == Some(v) {
                    a = None;
                    b = None;
                    c = None;
                    continue;
                }
                match (a, b, c) {
                    (None, _, _) => a = Some(v),
                    (Some(_), None, _) => b = Some(v),
                    (Some(_), Some(_), None) => c = Some(v),
                    (Some(v0), Some(v1), Some(v2)) => {
                        emit(&[v0, v1, v, v0, v, v2]);
                        a = Some(v2);
                        b = Some(v);
                        c = None;
                    }
                }
            }
        }
    }
    if written < dst.len() {
        let pad = if written > 0 { dst[0] } else { fallback_pad };
        dst[written..].fill(pad);
    }
    written
}

/// Expands `rects` rectangles of `stride`-byte vertices from `src` into six
/// vertices each in `dst`: `(v0, v1, v2)` and `(v0, v2, v3)`, so both
/// triangles turn the same way. `elements` are (byte offset, format).
pub(crate) fn expand_rects(src: &[u8], dst: &mut [u8], stride: usize, elements: &[(u32, u32)]) {
    for (rect, out) in src
        .chunks_exact(3 * stride)
        .zip(dst.chunks_exact_mut(6 * stride))
    {
        let v = |i: usize| &rect[i * stride..(i + 1) * stride];
        for (slot, i) in [0, 1, 2, 0, 2].into_iter().enumerate() {
            out[slot * stride..(slot + 1) * stride].copy_from_slice(v(i));
        }
        let fourth = &mut out[5 * stride..];
        fourth.copy_from_slice(v(0));
        for &(offset, format) in elements {
            let at = offset as usize;
            let Some((n, kind)) = components(format) else {
                continue;
            };
            // Only the components inside the vertex: the rest belong to the
            // next one.
            let inside = stride.saturating_sub(at) / kind.bytes();
            for c in 0..n.min(inside) {
                let get = |i: usize| kind.read(&v(i)[at..], c);
                let value = get(0) + get(2) - get(1);
                kind.write(&mut fourth[at..], c, value);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    F32,
    F16,
    Unorm8,
    Unorm16,
}

/// Component count and kind of the vertex formats rectangles use.
fn components(format: u32) -> Option<(usize, Kind)> {
    Some(match format {
        2 => (4, Kind::F32),
        6 => (3, Kind::F32),
        16 => (2, Kind::F32),
        41 => (1, Kind::F32),
        10 => (4, Kind::F16),
        34 => (2, Kind::F16),
        54 => (1, Kind::F16),
        11 => (4, Kind::Unorm16),
        35 => (2, Kind::Unorm16),
        28 | 29 | 87 | 91 => (4, Kind::Unorm8),
        _ => return None,
    })
}

impl Kind {
    fn bytes(self) -> usize {
        match self {
            Kind::F32 => 4,
            Kind::F16 | Kind::Unorm16 => 2,
            Kind::Unorm8 => 1,
        }
    }

    fn read(self, at: &[u8], c: usize) -> f32 {
        let b = &at[c * self.bytes()..];
        match self {
            Kind::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Kind::F16 => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Kind::Unorm16 => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
            Kind::Unorm8 => f32::from(b[0]) / 255.0,
        }
    }

    fn write(self, at: &mut [u8], c: usize, v: f32) {
        let n = self.bytes();
        let b = &mut at[c * n..(c + 1) * n];
        match self {
            Kind::F32 => b.copy_from_slice(&v.to_le_bytes()),
            Kind::F16 => b.copy_from_slice(&f32_to_half(v).to_le_bytes()),
            Kind::Unorm16 => {
                b.copy_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes())
            }
            Kind::Unorm8 => b[0] = (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: scale the mantissa.
        (0, m) => {
            let v = m as f32 * 2f32.powi(-24);
            return if sign != 0 { -v } else { v };
        }
        (31, m) => sign | 0x7f80_0000 | m << 13,
        (e, m) => sign | (e + 112) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00; // too large: infinity
    }
    if exp <= 0 {
        return sign; // too small: zero
    }
    sign | (exp as u16) << 10 | ((man + 0x1000) >> 13) as u16
}

#[cfg(windows)]
mod gpu {
    use std::sync::OnceLock;

    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_POINTLIST, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
    };
    use windows::core::{PCSTR, s};

    use super::ExpandedTopology;
    use crate::host::{HostList, host, transition};

    const CHUNK: u32 = 256;

    const CAPTURE_VS: &str = r#"
uint main(uint vid : SV_VertexID) : INDEX {
    return vid;
}
"#;

    const EXPAND_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Params : register(b0) {
    uint kind;
    uint count;
    uint max_out;
    uint num_chunks;
    uint cut_enabled;
    uint cut_value;
    uint fallback_pad;
    uint is_u16;
};
ByteAddressBuffer capture : register(t0);
RWByteAddressBuffer meta : register(u0);
RWByteAddressBuffer work : register(u1);
RWByteAddressBuffer dst : register(u2);

static const uint CHUNK = 256u;
static const uint MAX_DISPATCH_THREADS = 65535u * 64u;

bool is_cut(uint v) {
    return cut_enabled != 0u && v == cut_value;
}

uint load_src(uint i) {
    return capture.Load(i * 4u);
}

void write_out(uint idx, uint val) {
    if (idx < max_out) {
        if (is_u16 != 0u) {
            work.Store(idx * 4u, val);
        } else {
            dst.Store(idx * 4u, val);
        }
    }
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void cs_scan_chunks(uint3 id : SV_DispatchThreadID) {
    for (uint c = id.x; c < num_chunks; c += MAX_DISPATCH_THREADS) {
        uint start = c * CHUNK;
        uint end = min(start + CHUNK, count);
        uint has_cut = 0u;
        uint last_cut_plus_1 = 0u;
        for (uint i = start; i < end; ++i) {
            if (is_cut(load_src(i))) {
                has_cut = 1u;
                last_cut_plus_1 = i + 1u;
            }
        }
        uint at = (c + 1u) * 16u;
        meta.Store2(at, uint2(has_cut, last_cut_plus_1));
    }
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void cs_resolve_starts() {
    uint rs = 0u;
    for (uint c = 0u; c < num_chunks; ++c) {
        uint at = (c + 1u) * 16u;
        uint2 m = meta.Load2(at);
        meta.Store(at, rs);
        if (m.x != 0u) {
            rs = m.y;
        }
    }
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void cs_count_chunks(uint3 id : SV_DispatchThreadID) {
    for (uint c = id.x; c < num_chunks; c += MAX_DISPATCH_THREADS) {
        uint at = (c + 1u) * 16u;
        uint rs = meta.Load(at);
        uint start = c * CHUNK;
        uint end = min(start + CHUNK, count);
        uint out_n = 0u;
        uint first_out = 0u;
        for (uint i = start; i < end; ++i) {
            uint v = load_src(i);
            if (is_cut(v)) {
                rs = i + 1u;
            } else {
                uint pos = i - rs;
                if (kind == 0u) {
                    if (pos >= 2u) {
                        if (out_n == 0u) first_out = load_src(rs);
                        out_n += 3u;
                    }
                } else if (kind == 1u) {
                    if (pos >= 1u) {
                        if (out_n == 0u) first_out = load_src(i - 1u);
                        out_n += 2u;
                        bool run_end = (i + 1u == count);
                        if (!run_end) {
                            run_end = is_cut(load_src(i + 1u));
                        }
                        if (run_end) {
                            out_n += 2u;
                        }
                    }
                } else if (kind == 2u) {
                    if ((pos & 3u) == 3u) {
                        if (out_n == 0u) first_out = load_src(i - 3u);
                        out_n += 6u;
                    }
                } else {
                    if (pos >= 3u && (pos & 1u) == 1u) {
                        if (out_n == 0u) first_out = load_src(i - 3u);
                        out_n += 6u;
                    }
                }
            }
        }
        meta.Store2(at + 4u, uint2(out_n, first_out));
    }
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void cs_prefix_offsets() {
    uint total = 0u;
    uint pad = fallback_pad;
    bool have_pad = false;
    for (uint c = 0u; c < num_chunks; ++c) {
        uint at = (c + 1u) * 16u;
        uint2 cf = meta.Load2(at + 4u);
        meta.Store(at + 4u, total);
        if (!have_pad && cf.x > 0u) {
            pad = cf.y;
            have_pad = true;
        }
        total += cf.x;
    }
    meta.Store2(0u, uint2(min(total, max_out), pad));
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void cs_emit_chunks(uint3 id : SV_DispatchThreadID) {
    for (uint c = id.x; c < num_chunks; c += MAX_DISPATCH_THREADS) {
        uint at = (c + 1u) * 16u;
        uint2 ro = meta.Load2(at);
        uint rs = ro.x;
        uint w = ro.y;
        uint start = c * CHUNK;
        uint end = min(start + CHUNK, count);
        for (uint i = start; i < end; ++i) {
            uint v = load_src(i);
            if (is_cut(v)) {
                rs = i + 1u;
            } else {
                uint pos = i - rs;
                if (kind == 0u) {
                    if (pos >= 2u) {
                        write_out(w, load_src(rs));
                        write_out(w + 1u, load_src(i - 1u));
                        write_out(w + 2u, v);
                        w += 3u;
                    }
                } else if (kind == 1u) {
                    if (pos >= 1u) {
                        write_out(w, load_src(i - 1u));
                        write_out(w + 1u, v);
                        w += 2u;
                        bool run_end = (i + 1u == count);
                        if (!run_end) {
                            run_end = is_cut(load_src(i + 1u));
                        }
                        if (run_end) {
                            write_out(w, v);
                            write_out(w + 1u, load_src(rs));
                            w += 2u;
                        }
                    }
                } else if (kind == 2u) {
                    if ((pos & 3u) == 3u) {
                        uint v0 = load_src(i - 3u);
                        uint v1 = load_src(i - 2u);
                        uint v2 = load_src(i - 1u);
                        write_out(w, v0);
                        write_out(w + 1u, v1);
                        write_out(w + 2u, v2);
                        write_out(w + 3u, v0);
                        write_out(w + 4u, v2);
                        write_out(w + 5u, v);
                        w += 6u;
                    }
                } else {
                    if (pos >= 3u && (pos & 1u) == 1u) {
                        uint v0 = load_src(i - 3u);
                        uint v1 = load_src(i - 2u);
                        uint v2 = load_src(i - 1u);
                        write_out(w, v0);
                        write_out(w + 1u, v1);
                        write_out(w + 2u, v);
                        write_out(w + 3u, v0);
                        write_out(w + 4u, v);
                        write_out(w + 5u, v2);
                        w += 6u;
                    }
                }
            }
        }
    }
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void cs_finalize(uint3 id : SV_DispatchThreadID) {
    uint2 tp = meta.Load2(0u);
    uint total = tp.x;
    uint pad = tp.y;
    if (is_u16 != 0u) {
        uint out_dwords = (max_out + 1u) / 2u;
        for (uint d = id.x; d < out_dwords; d += MAX_DISPATCH_THREADS) {
            uint i0 = d * 2u;
            uint i1 = i0 + 1u;
            uint v0 = pad;
            if (i0 < total) v0 = work.Load(i0 * 4u);
            uint v1 = pad;
            if (i1 < total) v1 = work.Load(i1 * 4u);
            dst.Store(d * 4u, (v0 & 0xFFFFu) | ((v1 & 0xFFFFu) << 16u));
        }
    } else {
        for (uint d = id.x; d < max_out; d += MAX_DISPATCH_THREADS) {
            if (d >= total) {
                dst.Store(d * 4u, pad);
            }
        }
    }
}
"#;

    struct Expander {
        capture_root: ID3D12RootSignature,
        capture_pso: ID3D12PipelineState,
        cs_root: ID3D12RootSignature,
        scan_pso: ID3D12PipelineState,
        resolve_pso: ID3D12PipelineState,
        count_pso: ID3D12PipelineState,
        prefix_pso: ID3D12PipelineState,
        emit_pso: ID3D12PipelineState,
        finalize_pso: ID3D12PipelineState,
    }

    // SAFETY: D3D12 root signatures and PSOs are free-threaded.
    unsafe impl Send for Expander {}
    // SAFETY: as above.
    unsafe impl Sync for Expander {}

    fn compile(source: &str, name: PCSTR, entry: PCSTR, target: PCSTR) -> windows::core::Result<Vec<u8>> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: static shader strings and caller-owned out pointers.
        let res = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                name,
                None,
                None,
                entry,
                target,
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = res {
            if let Some(errors) = errors {
                // SAFETY: blob owns its bytes.
                let msg = unsafe {
                    std::slice::from_raw_parts(
                        errors.GetBufferPointer().cast::<u8>(),
                        errors.GetBufferSize(),
                    )
                };
                tracing::warn!("topology expander compile: {}", String::from_utf8_lossy(msg));
            }
            return Err(e);
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: blob owns its bytes.
        Ok(unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
                .to_vec()
        })
    }

    fn make_expander() -> windows::core::Result<Expander> {
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(CAPTURE_VS, s!("expand_capture"), s!("main"), s!("vs_5_0"))?;
        let cs_scan = compile(EXPAND_CS, s!("expand_cs"), s!("cs_scan_chunks"), s!("cs_5_1"))?;
        let cs_resolve = compile(EXPAND_CS, s!("expand_cs"), s!("cs_resolve_starts"), s!("cs_5_1"))?;
        let cs_count = compile(EXPAND_CS, s!("expand_cs"), s!("cs_count_chunks"), s!("cs_5_1"))?;
        let cs_prefix = compile(EXPAND_CS, s!("expand_cs"), s!("cs_prefix_offsets"), s!("cs_5_1"))?;
        let cs_emit = compile(EXPAND_CS, s!("expand_cs"), s!("cs_emit_chunks"), s!("cs_5_1"))?;
        let cs_finalize = compile(EXPAND_CS, s!("expand_cs"), s!("cs_finalize"), s!("cs_5_1"))?;

        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT,
            ..Default::default()
        };
        let mut rs_blob: Option<ID3DBlob> = None;
        // SAFETY: valid root signature and PSO descriptions with compiled bytecode.
        unsafe {
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut rs_blob, None)?;
            let rs_blob = rs_blob.ok_or_else(windows::core::Error::empty)?;
            let rs_bytes = std::slice::from_raw_parts(
                rs_blob.GetBufferPointer().cast::<u8>(),
                rs_blob.GetBufferSize(),
            );
            let capture_root: ID3D12RootSignature = device.CreateRootSignature(0, rs_bytes)?;
            let so_entry = D3D12_SO_DECLARATION_ENTRY {
                Stream: 0,
                SemanticName: s!("INDEX"),
                SemanticIndex: 0,
                StartComponent: 0,
                ComponentCount: 1,
                OutputSlot: 0,
            };
            let stride: u32 = 4;
            let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&capture_root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                StreamOutput: D3D12_STREAM_OUTPUT_DESC {
                    pSODeclaration: &so_entry,
                    NumEntries: 1,
                    pBufferStrides: &stride,
                    NumStrides: 1,
                    RasterizedStream: D3D12_SO_NO_RASTERIZED_STREAM,
                },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                IBStripCutValue: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
                NumRenderTargets: 0,
                DSVFormat: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let capture_pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&pso_desc)?;
            let cs_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs_scan)?;
            let make_cs = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineState> {
                device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&cs_root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
            };
            Ok(Expander {
                capture_root,
                capture_pso,
                scan_pso: make_cs(&cs_scan)?,
                resolve_pso: make_cs(&cs_resolve)?,
                count_pso: make_cs(&cs_count)?,
                prefix_pso: make_cs(&cs_prefix)?,
                emit_pso: make_cs(&cs_emit)?,
                finalize_pso: make_cs(&cs_finalize)?,
                cs_root,
            })
        }
    }

    fn expander() -> Option<&'static Expander> {
        static EXPANDER: OnceLock<Option<Expander>> = OnceLock::new();
        EXPANDER
            .get_or_init(|| match make_expander() {
                Ok(e) => Some(e),
                Err(e) => {
                    tracing::warn!(error = %e, "no GPU topology expander");
                    None
                }
            })
            .as_ref()
    }

    /// Reusable per-`HostList` GPU scratch buffers for indexed topology
    /// expansion. Outside an active expanded draw, `so_counter` and `capture`
    /// remain in `STREAM_OUT`, and `meta`/`work`/`result` remain in
    /// `UNORDERED_ACCESS`, so repeated draws, buffer growth, and closed-list
    /// replay all maintain exact D3D12 resource states.
    #[derive(Default)]
    pub(crate) struct ExpandScratch {
        zero: Option<ID3D12Resource>,
        so_counter: Option<ID3D12Resource>,
        capture: Option<(ID3D12Resource, u64)>,
        meta: Option<(ID3D12Resource, u64)>,
        work: Option<(ID3D12Resource, u64)>,
        result: Option<(ID3D12Resource, u64)>,
    }

    // SAFETY: D3D12 resources are free-threaded.
    unsafe impl Send for ExpandScratch {}

    fn default_buffer(
        device: &ID3D12Device,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        initial: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(256).next_multiple_of(256),
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
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain default buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial, None, &mut out)
                .ok()?;
        }
        out
    }

    fn ensure_buffer(
        list: &HostList,
        device: &ID3D12Device,
        slot: &mut Option<(ID3D12Resource, u64)>,
        needed: u64,
        flags: D3D12_RESOURCE_FLAGS,
        initial: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let needed = needed.max(256).next_multiple_of(256);
        if let Some((buf, cap)) = slot.as_ref()
            && *cap >= needed
        {
            return Some(buf.clone());
        }
        let cap = needed
            .max(4096)
            .checked_next_power_of_two()
            .unwrap_or(needed);
        let buf = default_buffer(device, cap, flags, initial)?;
        if let Some((old, _)) = slot.replace((buf.clone(), cap)) {
            // Earlier commands in this list may still reference `old`.
            list.keep(&old);
        }
        Some(buf)
    }

    fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        }
    }

    /// Output of GPU-side indexed topology expansion, ready to bind for
    /// `DrawIndexedInstanced`. Call [`finish_indexed_gpu`] immediately after
    /// the draw so `result` returns to `UNORDERED_ACCESS`.
    pub(crate) struct ExpandedIndexDraw {
        pub(crate) view: D3D12_INDEX_BUFFER_VIEW,
        pub(crate) count: u32,
        pub(crate) result: ID3D12Resource,
    }

    /// Captures `count` indices from `src_ibv` starting at `first` via a
    /// private point-list stream-output pass (keeping the source index buffer
    /// in its ordinary IA `INDEX_BUFFER` state) and expands them on the GPU
    /// into a host triangle/line-list index buffer.
    ///
    /// Replaces the command list's graphics root signature, compute root
    /// signature, pipeline state, and IA primitive topology; callers must
    /// re-bind their draw state before issuing the draw.
    pub(crate) fn expand_indexed_gpu(
        list: &HostList,
        kind: ExpandedTopology,
        src_ibv: D3D12_INDEX_BUFFER_VIEW,
        count: u32,
        first: u32,
        strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
        fallback_pad: u32,
    ) -> Option<ExpandedIndexDraw> {
        let is_u16 = match src_ibv.Format {
            DXGI_FORMAT_R16_UINT => true,
            DXGI_FORMAT_R32_UINT => false,
            _ => return None,
        };
        let max_out = kind.max_output_indices(count);
        if max_out == 0 {
            return None;
        }
        let host = host()?;
        let exp = expander()?;
        let elem_bytes: u64 = if is_u16 { 2 } else { 4 };
        let capture_bytes = u64::from(count).checked_mul(4)?;
        let out_bytes = u64::from(max_out).checked_mul(elem_bytes)?;
        let out_bytes_u32 = u32::try_from(out_bytes).ok()?;
        let num_chunks = count.div_ceil(CHUNK);
        let meta_bytes = u64::from(num_chunks).checked_add(1)?.checked_mul(16)?;
        let work_bytes = if is_u16 {
            u64::from(max_out).checked_mul(4)?
        } else {
            256
        };

        let (zero, so_counter, capture, meta, work, result) = {
            let mut scratch = list.expand_scratch();
            if scratch.zero.is_none() {
                let (buf, ptr) = host.upload_buffer(256)?;
                // SAFETY: freshly mapped 256-byte upload buffer.
                unsafe { ptr.write_bytes(0, 256) };
                scratch.zero = Some(buf);
            }
            if scratch.so_counter.is_none() {
                scratch.so_counter = default_buffer(
                    &host.device,
                    256,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_STREAM_OUT,
                );
            }
            let zero = scratch.zero.clone()?;
            let so_counter = scratch.so_counter.clone()?;
            let capture = ensure_buffer(
                list,
                &host.device,
                &mut scratch.capture,
                capture_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_STREAM_OUT,
            )?;
            let meta = ensure_buffer(
                list,
                &host.device,
                &mut scratch.meta,
                meta_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            let work = ensure_buffer(
                list,
                &host.device,
                &mut scratch.work,
                work_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            let result = ensure_buffer(
                list,
                &host.device,
                &mut scratch.result,
                out_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            (zero, so_counter, capture, meta, work, result)
        };

        let (cut_enabled, cut_value, pad) = if is_u16 {
            (
                u32::from(strip_cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF),
                u32::from(u16::MAX),
                fallback_pad.min(u32::from(u16::MAX)),
            )
        } else {
            (
                u32::from(strip_cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF),
                u32::MAX,
                fallback_pad,
            )
        };
        let params: [u32; 8] = [
            kind.kind_id(),
            count,
            max_out,
            num_chunks,
            cut_enabled,
            cut_value,
            pad,
            u32::from(is_u16),
        ];
        let chunk_groups = num_chunks.div_ceil(64).min(65535);
        let out_dwords = if is_u16 { max_out.div_ceil(2) } else { max_out };
        let final_groups = out_dwords.div_ceil(64).min(65535);
        let l = &list.list;

        // SAFETY: open command list, live private buffers in tracked states,
        // and valid compiled SO/compute pipelines.
        let result_gpu = unsafe {
            let counter_gpu = so_counter.GetGPUVirtualAddress();
            let capture_gpu = capture.GetGPUVirtualAddress();
            let meta_gpu = meta.GetGPUVirtualAddress();
            let work_gpu = work.GetGPUVirtualAddress();
            let result_gpu = result.GetGPUVirtualAddress();

            // 1. Reset the 64-bit SO filled-size counter to 0 on the GPU so
            // repeated draws and closed-list replays start at offset 0.
            l.ResourceBarrier(&[transition(
                &so_counter,
                D3D12_RESOURCE_STATE_STREAM_OUT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            l.CopyBufferRegion(&so_counter, 0, &zero, 0, 8);
            l.ResourceBarrier(&[transition(
                &so_counter,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_STREAM_OUT,
            )]);

            // 2. Capture raw indices (including strip-cut markers) through SO
            // while keeping the title's source index buffer in INDEX_BUFFER state.
            l.SetGraphicsRootSignature(&exp.capture_root);
            l.SetPipelineState(&exp.capture_pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
            l.IASetIndexBuffer(Some(&src_ibv));
            l.SOSetTargets(
                0,
                Some(&[D3D12_STREAM_OUTPUT_BUFFER_VIEW {
                    BufferLocation: capture_gpu,
                    SizeInBytes: capture_bytes,
                    BufferFilledSizeLocation: counter_gpu,
                }]),
            );
            l.DrawIndexedInstanced(count, 1, first, 0, 0);
            l.SOSetTargets(0, Some(&[D3D12_STREAM_OUTPUT_BUFFER_VIEW::default()]));
            l.ResourceBarrier(&[transition(
                &capture,
                D3D12_RESOURCE_STATE_STREAM_OUT,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);

            // 3. Chunked GPU expansion into `result` (with u16 packing + tail pad).
            l.SetComputeRootSignature(&exp.cs_root);
            l.SetComputeRoot32BitConstants(0, 8, params.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, capture_gpu);
            l.SetComputeRootUnorderedAccessView(2, meta_gpu);
            l.SetComputeRootUnorderedAccessView(3, work_gpu);
            l.SetComputeRootUnorderedAccessView(4, result_gpu);

            l.SetPipelineState(&exp.scan_pso);
            l.Dispatch(chunk_groups, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);

            l.SetPipelineState(&exp.resolve_pso);
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);

            l.SetPipelineState(&exp.count_pso);
            l.Dispatch(chunk_groups, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);

            l.SetPipelineState(&exp.prefix_pso);
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);

            l.SetPipelineState(&exp.emit_pso);
            l.Dispatch(chunk_groups, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);

            l.SetPipelineState(&exp.finalize_pso);
            l.Dispatch(final_groups, 1, 1);

            l.ResourceBarrier(&[
                transition(
                    &capture,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_STREAM_OUT,
                ),
                transition(
                    &result,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDEX_BUFFER,
                ),
            ]);
            result_gpu
        };

        Some(ExpandedIndexDraw {
            view: D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: result_gpu,
                SizeInBytes: out_bytes_u32,
                Format: src_ibv.Format,
            },
            count: max_out,
            result,
        })
    }

    /// Returns the expanded result buffer to `UNORDERED_ACCESS` after the
    /// expanded draw completes so subsequent draws or replays see a consistent
    /// initial state.
    pub(crate) fn finish_indexed_gpu(list: &HostList, result: &ID3D12Resource) {
        // SAFETY: open command list and live scratch result buffer.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                result,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }
}

#[cfg(windows)]
pub(crate) use gpu::{ExpandScratch, expand_indexed_gpu, finish_indexed_gpu};

#[cfg(test)]
mod tests {
    use super::*;

    /// A rectangle of float2 positions and RGBA8 colors, given as top
    /// left, bottom left, bottom right: the fourth corner completes the
    /// parallelogram at the top right.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let vertex = |x: f32, y: f32, r: u8| {
            let mut v = Vec::new();
            v.extend(x.to_le_bytes());
            v.extend(y.to_le_bytes());
            v.extend([r, 0, 0, 255]);
            v
        };
        let src: Vec<u8> = [
            vertex(-1.0, 1.0, 0),
            vertex(-1.0, -1.0, 100),
            vertex(1.0, -1.0, 150),
        ]
        .concat();
        let mut dst = vec![0u8; 6 * 12];
        expand_rects(&src, &mut dst, 12, &[(0, 16), (8, 28)]);
        assert_eq!(dst[..36], src[..]);
        assert_eq!(dst[36..48], src[..12]);
        assert_eq!(dst[48..60], src[24..36]);
        assert_eq!(dst[60..], vertex(1.0, 1.0, 50)[..]);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x3e00), 1.5);
        assert_eq!(half_to_f32(0xc000), -2.0);
    }

    /// An element that reaches past the stride (float4 at 4 in 8-byte
    /// vertices, RGBA8 at the stride itself): the components inside get
    /// the fourth corner, the rest keep the first vertex's bytes. The old
    /// code indexed past the vertex and panicked.
    #[test]
    fn elements_past_the_stride_are_left_alone() {
        let vertex = |x: f32, y: f32| [x.to_le_bytes(), y.to_le_bytes()].concat();
        let src = [vertex(0.0, 2.0), vertex(1.0, 3.0), vertex(5.0, 7.0)].concat();
        let mut dst = vec![0u8; 6 * 8];
        expand_rects(&src, &mut dst, 8, &[(0, 16), (4, 2), (8, 28)]);
        assert_eq!(dst[..40], [&src[..], &src[..8], &src[16..]].concat()[..]);
        assert_eq!(dst[40..], vertex(4.0, 6.0)[..]);
    }

    #[test]
    fn expand_non_indexed_topologies() {
        // QUADS: 2 quads + 1 trailing vertex ignored.
        let mut out = vec![0u32; ExpandedTopology::Quads.max_output_indices(9) as usize];
        assert_eq!(out.len(), 12);
        let n = ExpandedTopology::Quads.expand_non_indexed_u32(9, 10, &mut out);
        assert_eq!(n, 12);
        assert_eq!(out, [10, 11, 12, 10, 12, 13, 14, 15, 16, 14, 16, 17]);

        // QUAD_STRIP: 6 vertices -> 2 quads with consistent winding, 7th ignored.
        let mut out = vec![0u16; ExpandedTopology::QuadStrip.max_output_indices(7) as usize];
        assert_eq!(out.len(), 12);
        let n = ExpandedTopology::QuadStrip.expand_non_indexed_u16(7, 0, &mut out);
        assert_eq!(n, 12);
        assert_eq!(out, [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]);

        // TRIANGLE_FAN: 5 vertices -> 3 triangles sharing pivot 0.
        let mut out = vec![0u32; ExpandedTopology::TriangleFan.max_output_indices(5) as usize];
        assert_eq!(out.len(), 9);
        assert_eq!(
            ExpandedTopology::TriangleFan.expand_non_indexed_u32(5, 0, &mut out),
            9
        );
        assert_eq!(out, [0, 1, 2, 0, 2, 3, 0, 3, 4]);
        assert_eq!(ExpandedTopology::TriangleFan.max_output_indices(2), 0);

        // LINE_LOOP: 4 vertices -> 4 segments closing 3 -> 0.
        let mut out = vec![0u16; ExpandedTopology::LineLoop.max_output_indices(4) as usize];
        assert_eq!(out.len(), 8);
        assert_eq!(
            ExpandedTopology::LineLoop.expand_non_indexed_u16(4, 1, &mut out),
            8
        );
        assert_eq!(out, [1, 2, 2, 3, 3, 4, 4, 1]);
        assert_eq!(ExpandedTopology::LineLoop.max_output_indices(1), 0);
    }

    #[test]
    fn expand_indexed_topologies_with_strip_cuts() {
        // TRIANGLE_FAN (16-bit, 0xFFFF cut): two fans separated by cut + incomplete tail.
        let src: [u16; 10] = [0, 1, 2, 3, u16::MAX, 10, 11, 12, u16::MAX, 20];
        let mut out = vec![0u16; ExpandedTopology::TriangleFan.max_output_indices(10) as usize];
        let n = ExpandedTopology::TriangleFan.expand_indexed_u16(&src, Some(u16::MAX), 0, &mut out);
        assert_eq!(n, 9);
        assert_eq!(&out[..9], [0, 1, 2, 0, 2, 3, 10, 11, 12]);
        assert!(out[9..].iter().all(|&i| i == 0), "degenerate tail padding");

        // QUAD_STRIP (32-bit, 0xFFFFFFFF cut): two 4-vertex strips separated by cut.
        let src32: [u32; 9] = [0, 1, 2, 3, u32::MAX, 4, 5, 6, 7];
        let mut out32 = vec![0u32; ExpandedTopology::QuadStrip.max_output_indices(9) as usize];
        let n =
            ExpandedTopology::QuadStrip.expand_indexed_u32(&src32, Some(u32::MAX), 0, &mut out32);
        assert_eq!(n, 12);
        assert_eq!(&out32[..12], [0, 1, 3, 0, 3, 2, 4, 5, 7, 4, 7, 6]);
        assert!(out32[12..].iter().all(|&i| i == 0));

        // QUADS (16-bit cut discards incomplete 2-vertex quad before cut).
        let qsrc: [u16; 7] = [0, 1, u16::MAX, 2, 3, 4, 5];
        let mut qout = vec![0u16; ExpandedTopology::Quads.max_output_indices(7) as usize];
        let n = ExpandedTopology::Quads.expand_indexed_u16(&qsrc, Some(u16::MAX), 0, &mut qout);
        assert_eq!(n, 6);
        assert_eq!(qout, [2, 3, 4, 2, 4, 5]);

        // LINE_LOOP (16-bit cut closes first loop before starting second).
        let lsrc: [u16; 7] = [0, 1, 2, u16::MAX, 5, 6, 7];
        let mut lout = vec![0u16; ExpandedTopology::LineLoop.max_output_indices(7) as usize];
        let n = ExpandedTopology::LineLoop.expand_indexed_u16(&lsrc, Some(u16::MAX), 0, &mut lout);
        assert_eq!(n, 12);
        assert_eq!(&lout[..12], [0, 1, 1, 2, 2, 0, 5, 6, 6, 7, 7, 5]);
        assert_eq!(&lout[12..], [0, 0]);
    }

    #[cfg(windows)]
    fn render_topology_test(
        topology_type: windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
        host_topology: u32,
        strip_cut: windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
        vertices: &[[f32; 2]],
        draw_calls: &[(&[u16], i32)],
    ) -> Option<Vec<[u8; 4]>> {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::s;

        let host = crate::host::host()?;
        const HLSL: &str = r#"
float4 vs_main(float2 pos : POSITION) : SV_Position { return float4(pos, 0.0, 1.0); }
float4 ps_main() : SV_Target { return float4(1.0, 1.0, 1.0, 1.0); }
"#;
        let compile = |entry: windows::core::PCSTR, target: windows::core::PCSTR| -> Vec<u8> {
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: static shader source and entry point strings.
            unsafe {
                D3DCompile(
                    HLSL.as_ptr().cast(),
                    HLSL.len(),
                    s!("topology_test"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .unwrap();
                let blob = blob.unwrap();
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
                .to_vec()
            }
        };
        let vs = compile(s!("vs_main"), s!("vs_5_0"));
        let ps = compile(s!("ps_main"), s!("ps_5_0"));

        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            ..Default::default()
        };
        let mut rs_blob: Option<ID3DBlob> = None;
        // SAFETY: plain root signature, PSO, RT, and readback creation.
        let (root, pso, rt, readback, vb, vb_gpu) = unsafe {
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut rs_blob, None)
                .unwrap();
            let rs_blob = rs_blob.unwrap();
            let rs_bytes = std::slice::from_raw_parts(
                rs_blob.GetBufferPointer().cast::<u8>(),
                rs_blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, rs_bytes).unwrap();
            let elem = D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            };
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let mut rast = D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            };
            if topology_type == D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE {
                rast.CullMode = D3D12_CULL_MODE_NONE;
            }
            let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: rast,
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: &elem,
                    NumElements: 1,
                },
                IBStripCutValue: strip_cut,
                PrimitiveTopologyType: topology_type,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let pso: ID3D12PipelineState =
                host.device.CreateGraphicsPipelineState(&pso_desc).unwrap();

            let rt_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 8,
                Height: 8,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let def_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut rt: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &def_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rt_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .unwrap();
            let rt = rt.unwrap();

            let rb_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256 * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let rb_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let mut readback: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &rb_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rb_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
            let readback = readback.unwrap();

            let vb_bytes = std::mem::size_of_val(vertices) as u64;
            let (vb, vb_ptr) = host.upload_buffer(vb_bytes).unwrap();
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr().cast::<u8>(),
                vb_ptr,
                vb_bytes as usize,
            );
            let vb_gpu = vb.GetGPUVirtualAddress();
            (root, pso, rt, readback, vb, vb_gpu)
        };

        let list = host.new_list().unwrap();
        let rtv = host.rtv(&rt, None);
        // SAFETY: open command list, live resources, valid scratch index views.
        unsafe {
            let l = &list.list;
            l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, None);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 8.0,
                Height: 8.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: 8,
                bottom: 8,
                ..Default::default()
            }]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host_topology as i32));
            l.IASetVertexBuffers(
                0,
                Some(&[D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: vb_gpu,
                    SizeInBytes: std::mem::size_of_val(vertices) as u32,
                    StrideInBytes: 8,
                }]),
            );
            for &(indices, base) in draw_calls {
                let bytes = (indices.len() * 2) as u64;
                let (ib_gpu, ib_ptr) = list.scratch(bytes).unwrap();
                std::ptr::copy_nonoverlapping(
                    indices.as_ptr().cast::<u8>(),
                    ib_ptr,
                    bytes as usize,
                );
                l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: ib_gpu,
                    SizeInBytes: bytes as u32,
                    Format: DXGI_FORMAT_R16_UINT,
                }));
                l.DrawIndexedInstanced(indices.len() as u32, 1, 0, base, 0);
            }
            l.ResourceBarrier(&[crate::host::transition(
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: 8,
                            Height: 8,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rt),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();
        host.execute(&[&list]);
        drop(vb);

        let mut data = std::ptr::null_mut();
        // SAFETY: `host.execute` waited for the GPU copy into `readback`.
        let pixels = unsafe {
            readback.Map(0, None, Some(&mut data)).unwrap();
            let mut out = Vec::with_capacity(64);
            for y in 0..8usize {
                let row = std::slice::from_raw_parts(data.cast::<u8>().add(y * 256), 32);
                for x in 0..8usize {
                    out.push([row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3]]);
                }
            }
            readback.Unmap(0, None);
            out
        };
        Some(pixels)
    }

    #[cfg(windows)]
    #[test]
    fn quads_and_strip_cut_triangle_fan_render_with_backface_culling() {
        use windows::Win32::Graphics::Direct3D12::*;
        // Vertices in clockwise NDC order (front-facing under D3D12_CULL_MODE_BACK):
        // 0..4: non-indexed QUADS covering left half (x: -1..0, y: -1..1).
        // 4..8: indexed TRIANGLE_FAN #1 covering top-right (x: 0.25..1, y: 0.25..1).
        // 8..11: indexed TRIANGLE_FAN #2 covering bottom-right (x: 0.25..1, y: -1..-0.25).
        let vertices: [[f32; 2]; 11] = [
            [-1.0, 1.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, -1.0],
            [0.25, 1.0],
            [1.0, 1.0],
            [1.0, 0.25],
            [0.25, 0.25],
            [0.25, -0.25],
            [1.0, -0.25],
            [1.0, -1.0],
        ];
        let mut quad_ib = vec![0u16; ExpandedTopology::Quads.max_output_indices(4) as usize];
        ExpandedTopology::Quads.expand_non_indexed_u16(4, 0, &mut quad_ib);

        let fan_src: [u16; 8] = [4, 5, 6, 7, u16::MAX, 8, 9, 10];
        let mut fan_ib = vec![0u16; ExpandedTopology::TriangleFan.max_output_indices(8) as usize];
        ExpandedTopology::TriangleFan.expand_indexed_u16(&fan_src, Some(u16::MAX), 0, &mut fan_ib);

        let Some(px) = render_topology_test(
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ExpandedTopology::Quads.host_topology(),
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF,
            &vertices,
            &[(&quad_ib, 0), (&fan_ib, 0)],
        ) else {
            return;
        };
        let at = |x: usize, y: usize| px[y * 8 + x][0];
        // Both triangles of the left-half quad must rasterize (not backface-culled).
        assert_eq!(at(1, 1), 255, "quad first triangle");
        assert_eq!(at(1, 6), 255, "quad second triangle");
        // Both strip-cut-separated triangle fans on the right half must rasterize.
        assert_eq!(at(6, 1), 255, "first triangle fan");
        assert_eq!(at(7, 6), 255, "second triangle fan after strip cut");
        // Middle gap between the two fans must stay black (strip cut did not bridge).
        assert_eq!(at(6, 4), 0, "gap between strip-cut fans stays black");
    }

    #[cfg(windows)]
    #[test]
    fn quad_strip_and_line_loop_render_on_gpu() {
        use windows::Win32::Graphics::Direct3D12::*;
        // Indexed QUAD_STRIP with 6 vertices (2 quads, CW front-facing under CULL_MODE_BACK)
        // covering the entire 8x8 target.
        let strip_verts: [[f32; 2]; 6] = [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ];
        let strip_src: [u16; 6] = [0, 1, 2, 3, 4, 5];
        let mut strip_ib = vec![0u16; ExpandedTopology::QuadStrip.max_output_indices(6) as usize];
        ExpandedTopology::QuadStrip.expand_indexed_u16(&strip_src, None, 0, &mut strip_ib);

        let Some(px) = render_topology_test(
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ExpandedTopology::QuadStrip.host_topology(),
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            &strip_verts,
            &[(&strip_ib, 0)],
        ) else {
            return;
        };
        let pat = |x: usize, y: usize| px[y * 8 + x][0];
        assert_eq!(pat(1, 1), 255, "quad strip quad 0");
        assert_eq!(pat(6, 6), 255, "quad strip quad 1");

        // LINE_LOOP drawing the perimeter of pixel centers (1,1)->(6,1)->(6,6)->(1,6)->(1,1).
        let to_ndc = |p: f32| (p + 0.5) / 8.0 * 2.0 - 1.0;
        let loop_verts: [[f32; 2]; 4] = [
            [to_ndc(1.0), -to_ndc(1.0)],
            [to_ndc(6.0), -to_ndc(1.0)],
            [to_ndc(6.0), -to_ndc(6.0)],
            [to_ndc(1.0), -to_ndc(6.0)],
        ];
        let mut loop_ib = vec![0u16; ExpandedTopology::LineLoop.max_output_indices(4) as usize];
        ExpandedTopology::LineLoop.expand_non_indexed_u16(4, 0, &mut loop_ib);

        let Some(lpx) = render_topology_test(
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            ExpandedTopology::LineLoop.host_topology(),
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            &loop_verts,
            &[(&loop_ib, 0)],
        ) else {
            return;
        };
        let lat = |x: usize, y: usize| lpx[y * 8 + x][0];
        assert_eq!(lat(3, 1), 255, "top edge");
        assert_eq!(lat(6, 3), 255, "right edge");
        assert_eq!(lat(3, 6), 255, "bottom edge");
        assert_eq!(lat(1, 3), 255, "closing edge v3 -> v0");
        assert_eq!(lat(3, 3), 0, "interior stays black");
    }

    #[cfg(windows)]
    #[test]
    fn gpu_expansion_matches_cpu_across_topologies_formats_cuts_and_chunks() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;

        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().unwrap();

        let make_readback = |bytes: u64| -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes.max(256).next_multiple_of(256),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out = None;
            // SAFETY: plain readback buffer creation.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut out,
                    )
                    .unwrap();
            }
            out.unwrap()
        };

        // 1. Multi-chunk R16_UINT with strip cuts right on/around 256-index chunk boundaries,
        // nonzero `first`, and every topology.
        let mut src16: Vec<u16> = (0u16..1035).map(|i| (i * 7 + 3) & 0x7FFF).collect();
        for &cut_pos in &[2usize, 10, 254, 255, 256, 257, 258, 511, 512, 767, 770] {
            src16[cut_pos] = u16::MAX;
        }
        let first16: u32 = 3;
        let count16: u32 = src16.len() as u32 - first16;
        let (ib16, ib16_ptr) = host.upload_buffer((src16.len() * 2) as u64).unwrap();
        // SAFETY: freshly mapped upload buffer of `src16.len() * 2` bytes.
        let ib16_gpu = unsafe {
            std::ptr::copy_nonoverlapping(src16.as_ptr().cast::<u8>(), ib16_ptr, src16.len() * 2);
            ib16.GetGPUVirtualAddress()
        };
        let ibv16 = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: ib16_gpu,
            SizeInBytes: (src16.len() * 2) as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        for kind in [
            ExpandedTopology::TriangleFan,
            ExpandedTopology::LineLoop,
            ExpandedTopology::Quads,
            ExpandedTopology::QuadStrip,
        ] {
            for cut_enabled in [false, true] {
                let strip_cut = if cut_enabled {
                    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
                } else {
                    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED
                };
                let max_out = kind.max_output_indices(count16) as usize;
                let mut expected = vec![0u16; max_out];
                kind.expand_indexed_u16(
                    &src16[first16 as usize..],
                    cut_enabled.then_some(u16::MAX),
                    77,
                    &mut expected,
                );
                let expanded = expand_indexed_gpu(&list, kind, ibv16, count16, first16, strip_cut, 77)
                    .expect("gpu u16 expansion");
                assert_eq!(expanded.count as usize, max_out);
                let bytes = (max_out * 2) as u64;
                let rb = make_readback(bytes);
                // SAFETY: open list; copy expanded result to readback then restore UAV state.
                unsafe {
                    list.list.ResourceBarrier(&[crate::host::transition(
                        &expanded.result,
                        D3D12_RESOURCE_STATE_INDEX_BUFFER,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    list.list.CopyBufferRegion(&rb, 0, &expanded.result, 0, bytes);
                    list.list.ResourceBarrier(&[crate::host::transition(
                        &expanded.result,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                }
                list.close();
                host.execute(&[&list]);
                let mut ptr = std::ptr::null_mut();
                // SAFETY: GPU execution finished writing `rb`.
                let got = unsafe {
                    rb.Map(0, None, Some(&mut ptr)).unwrap();
                    let out = std::slice::from_raw_parts(ptr.cast::<u16>(), max_out).to_vec();
                    rb.Unmap(0, None);
                    out
                };
                assert_eq!(got, expected, "u16 mismatch for {kind:?} cut={cut_enabled}");
                list.reset();
            }
        }

        // 2. Multi-chunk R32_UINT including an all-cut case exercising `fallback_pad`.
        let mut src32: Vec<u32> = (0u32..600).map(|i| i * 13 + 5).collect();
        for &cut_pos in &[0usize, 1, 255, 256, 257, 511, 512] {
            src32[cut_pos] = u32::MAX;
        }
        let all_cuts32 = [u32::MAX; 10];
        for (src, kind, fallback_pad) in [
            (&src32[..], ExpandedTopology::TriangleFan, 19u32),
            (&src32[..], ExpandedTopology::LineLoop, 19u32),
            (&src32[..], ExpandedTopology::Quads, 19u32),
            (&src32[..], ExpandedTopology::QuadStrip, 19u32),
            (&all_cuts32[..], ExpandedTopology::TriangleFan, 42u32),
        ] {
            let count = src.len() as u32;
            let max_out = kind.max_output_indices(count) as usize;
            let mut expected = vec![0u32; max_out];
            kind.expand_indexed_u32(src, Some(u32::MAX), fallback_pad, &mut expected);
            let (ib, ib_ptr) = host.upload_buffer((src.len() * 4) as u64).unwrap();
            // SAFETY: freshly mapped upload buffer.
            let ib_gpu = unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), ib_ptr, src.len() * 4);
                ib.GetGPUVirtualAddress()
            };
            let ibv = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: ib_gpu,
                SizeInBytes: (src.len() * 4) as u32,
                Format: DXGI_FORMAT_R32_UINT,
            };
            let expanded = expand_indexed_gpu(
                &list,
                kind,
                ibv,
                count,
                0,
                D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF,
                fallback_pad,
            )
            .expect("gpu u32 expansion");
            let bytes = (max_out * 4) as u64;
            let rb = make_readback(bytes);
            // SAFETY: open list; copy expanded result to readback then restore UAV state.
            unsafe {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &expanded.result,
                    D3D12_RESOURCE_STATE_INDEX_BUFFER,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.list.CopyBufferRegion(&rb, 0, &expanded.result, 0, bytes);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &expanded.result,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: GPU execution finished writing `rb`.
            let got = unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let out = std::slice::from_raw_parts(ptr.cast::<u32>(), max_out).to_vec();
                rb.Unmap(0, None);
                out
            };
            assert_eq!(got, expected, "u32 mismatch for {kind:?} len={count}");
            list.reset();
        }
        drop(ib16);
    }

    #[cfg(windows)]
    #[test]
    fn same_list_gpu_copy_expanded_triangle_fan_renders_new_indices() {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::s;

        let Some(host) = crate::host::host() else {
            return;
        };

        const HLSL: &str = r#"
struct VsOut { float4 pos : SV_Position; float4 col : COLOR; };
VsOut vs_main(float2 pos : POSITION, float4 col : COLOR) {
    VsOut o;
    o.pos = float4(pos, 0.0, 1.0);
    o.col = col;
    return o;
}
float4 ps_main(VsOut i) : SV_Target { return i.col; }
"#;
        let compile = |entry: windows::core::PCSTR, target: windows::core::PCSTR| -> Vec<u8> {
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: static shader source and entry point strings.
            unsafe {
                D3DCompile(
                    HLSL.as_ptr().cast(),
                    HLSL.len(),
                    s!("same_list_copy_test"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .unwrap();
                let blob = blob.unwrap();
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
                .to_vec()
            }
        };
        let vs = compile(s!("vs_main"), s!("vs_5_0"));
        let ps = compile(s!("ps_main"), s!("ps_5_0"));

        // 3 full-screen CW triangles: 0..3 red, 3..6 green, 6..9 blue.
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Vtx([f32; 2], [f32; 4]);
        let verts: [Vtx; 9] = [
            Vtx([-1.0, -1.0], [1.0, 0.0, 0.0, 1.0]),
            Vtx([-1.0, 3.0], [1.0, 0.0, 0.0, 1.0]),
            Vtx([3.0, -1.0], [1.0, 0.0, 0.0, 1.0]),
            Vtx([-1.0, -1.0], [0.0, 1.0, 0.0, 1.0]),
            Vtx([-1.0, 3.0], [0.0, 1.0, 0.0, 1.0]),
            Vtx([3.0, -1.0], [0.0, 1.0, 0.0, 1.0]),
            Vtx([-1.0, -1.0], [0.0, 0.0, 1.0, 1.0]),
            Vtx([-1.0, 3.0], [0.0, 0.0, 1.0, 1.0]),
            Vtx([3.0, -1.0], [0.0, 0.0, 1.0, 1.0]),
        ];

        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            ..Default::default()
        };
        let mut rs_blob: Option<ID3DBlob> = None;
        // SAFETY: plain D3D12 object creation for rendering test.
        let (root, pso16, pso32, rt, readback, vb, vb_gpu) = unsafe {
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut rs_blob, None)
                .unwrap();
            let rs_blob = rs_blob.unwrap();
            let rs_bytes = std::slice::from_raw_parts(
                rs_blob.GetBufferPointer().cast::<u8>(),
                rs_blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, rs_bytes).unwrap();
            let elems = [
                D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: s!("POSITION"),
                    Format: DXGI_FORMAT_R32G32_FLOAT,
                    InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    ..Default::default()
                },
                D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: s!("COLOR"),
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    AlignedByteOffset: 8,
                    InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    ..Default::default()
                },
            ];
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let rast = D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            };
            let make_pso = |cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE| -> ID3D12PipelineState {
                let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    VS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: vs.as_ptr().cast(),
                        BytecodeLength: vs.len(),
                    },
                    PS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: ps.as_ptr().cast(),
                        BytecodeLength: ps.len(),
                    },
                    BlendState: blend,
                    SampleMask: u32::MAX,
                    RasterizerState: rast,
                    InputLayout: D3D12_INPUT_LAYOUT_DESC {
                        pInputElementDescs: elems.as_ptr(),
                        NumElements: elems.len() as u32,
                    },
                    IBStripCutValue: cut,
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                };
                desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                host.device.CreateGraphicsPipelineState(&desc).unwrap()
            };
            let pso16 = make_pso(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF);
            let pso32 = make_pso(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF);

            let def_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let rt_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 8,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut rt: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &def_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rt_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .unwrap();
            let rt = rt.unwrap();

            let rb_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let rb_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256 * 4,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut readback: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &rb_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rb_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
            let readback = readback.unwrap();

            let vb_bytes = std::mem::size_of_val(&verts) as u64;
            let (vb, vb_ptr) = host.upload_buffer(vb_bytes).unwrap();
            std::ptr::copy_nonoverlapping(
                verts.as_ptr().cast::<u8>(),
                vb_ptr,
                vb_bytes as usize,
            );
            let vb_gpu = vb.GetGPUVirtualAddress();
            (root, pso16, pso32, rt, readback, vb, vb_gpu)
        };

        // Title index buffers registered in `shadow`:
        // - `title_ib16`: holds old RED indices `[9, 9, 1, 2, 3]` (with first=2, base=-1 -> [0,1,2] RED).
        // - `title_ib32`: 32,769 indices (forces scratch growth after the small R16 fan!).
        let title_ib16: Vec<u16> = vec![9, 9, 1, 2, 3];
        let mut title_ib32: Vec<u32> = vec![u32::MAX; 32_769];
        title_ib32[0] = 0;
        title_ib32[1] = 1;
        title_ib32[2] = 2;
        let res16 = title_ib16.as_ptr() as usize;
        let res32 = title_ib32.as_ptr() as usize;
        crate::shadow::register(res16, title_ib16.as_ptr() as u64, (title_ib16.len() * 2) as u64);
        crate::shadow::register(res32, title_ib32.as_ptr() as u64, (title_ib32.len() * 4) as u64);
        let shadow16 = crate::shadow::find(title_ib16.as_ptr() as u64, 10).unwrap();
        let shadow32 = crate::shadow::find(title_ib32.as_ptr() as u64, 12).unwrap();

        // Producer upload buffers copied on the GPU in the SAME command list.
        let (prod16, prod16_ptr) = host.upload_buffer(256).unwrap();
        let (prod32, prod32_ptr) = host.upload_buffer(256).unwrap();
        // Run 1: left half (R16 fan, first=2, base=-1) -> GREEN [4,5,6] -> [3,4,5];
        //        right half (R32 fan, 32769 indices) -> GREEN [3,4,5].
        // SAFETY: freshly mapped upload buffers.
        unsafe {
            let green16: [u16; 3] = [4, 5, 6];
            std::ptr::copy_nonoverlapping(green16.as_ptr().cast::<u8>(), prod16_ptr, 6);
            let green32: [u32; 3] = [3, 4, 5];
            std::ptr::copy_nonoverlapping(green32.as_ptr().cast::<u8>(), prod32_ptr, 12);
        }

        let list = host.new_list().unwrap();
        let rtv = host.rtv(&rt, None);
        shadow16.refresh(&list, title_ib16.as_ptr() as u64, 10);
        shadow32.refresh(&list, title_ib32.as_ptr() as u64, (title_ib32.len() * 4) as u64);

        // SAFETY: open command list, live buffers, explicit INDEX_BUFFER transitions.
        unsafe {
            let l = &list.list;
            l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, None);

            // Native same-list GPU CopyBufferRegion overwriting shadow16[2..5] and shadow32[0..3],
            // explicitly putting both sources in INDEX_BUFFER state before expanded draws.
            l.CopyBufferRegion(
                &shadow16.buffer,
                shadow16.offset(title_ib16.as_ptr() as u64) + 4,
                &prod16,
                0,
                6,
            );
            l.CopyBufferRegion(
                &shadow32.buffer,
                shadow32.offset(title_ib32.as_ptr() as u64),
                &prod32,
                0,
                12,
            );
            l.ResourceBarrier(&[
                crate::host::transition(
                    &shadow16.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_INDEX_BUFFER,
                ),
                crate::host::transition(
                    &shadow32.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_INDEX_BUFFER,
                ),
            ]);

            // Draw 1: R16_UINT fan (first=2, base=-1) onto left half (x=0..4).
            let exp16 = expand_indexed_gpu(
                &list,
                ExpandedTopology::TriangleFan,
                D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: shadow16.gpu(title_ib16.as_ptr() as u64),
                    SizeInBytes: (title_ib16.len() * 2) as u32,
                    Format: DXGI_FORMAT_R16_UINT,
                },
                3,
                2,
                D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF,
                1,
            )
            .expect("R16 same-list copy expansion");
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 4.0,
                Height: 4.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: 4,
                bottom: 4,
                ..Default::default()
            }]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso16);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(
                ExpandedTopology::TriangleFan.host_topology() as i32,
            ));
            l.IASetVertexBuffers(
                0,
                Some(&[D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: vb_gpu,
                    SizeInBytes: std::mem::size_of_val(&verts) as u32,
                    StrideInBytes: std::mem::size_of::<Vtx>() as u32,
                }]),
            );
            l.IASetIndexBuffer(Some(&exp16.view));
            l.DrawIndexedInstanced(exp16.count, 1, 0, -1, 0);
            finish_indexed_gpu(&list, &exp16.result);

            // Draw 2: R32_UINT fan with 32,769 indices (forces scratch growth) onto right half (x=4..8).
            let exp32 = expand_indexed_gpu(
                &list,
                ExpandedTopology::TriangleFan,
                D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: shadow32.gpu(title_ib32.as_ptr() as u64),
                    SizeInBytes: (title_ib32.len() * 4) as u32,
                    Format: DXGI_FORMAT_R32_UINT,
                },
                title_ib32.len() as u32,
                0,
                D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF,
                0,
            )
            .expect("R32 same-list copy expansion");
            l.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 4.0,
                Width: 4.0,
                Height: 4.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                left: 4,
                right: 8,
                bottom: 4,
                ..Default::default()
            }]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso32);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(
                ExpandedTopology::TriangleFan.host_topology() as i32,
            ));
            l.IASetVertexBuffers(
                0,
                Some(&[D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: vb_gpu,
                    SizeInBytes: std::mem::size_of_val(&verts) as u32,
                    StrideInBytes: std::mem::size_of::<Vtx>() as u32,
                }]),
            );
            l.IASetIndexBuffer(Some(&exp32.view));
            l.DrawIndexedInstanced(exp32.count, 1, 0, 0, 0);
            finish_indexed_gpu(&list, &exp32.result);

            // Copy render target to readback and return RT to RENDER_TARGET so
            // the same closed list can be replayed cleanly.
            l.ResourceBarrier(&[crate::host::transition(
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: 8,
                            Height: 4,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rt),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            l.ResourceBarrier(&[crate::host::transition(
                &rt,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
        }
        list.close();

        let read_pixels = || -> Vec<[u8; 4]> {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: GPU execution finished writing `readback`.
            unsafe {
                readback.Map(0, None, Some(&mut ptr)).unwrap();
                let mut px = Vec::with_capacity(32);
                for y in 0..4usize {
                    let row = std::slice::from_raw_parts(ptr.cast::<u8>().add(y * 256), 32);
                    for x in 0..8usize {
                        px.push([row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3]]);
                    }
                }
                readback.Unmap(0, None);
                px
            }
        };

        // Submission 1: both R16 and R32 fans must render GREEN (not stale RED).
        host.execute(&[&list]);
        let px1 = read_pixels();
        assert!(
            px1.iter().all(|&c| c == [0, 255, 0, 255]),
            "run 1 expected all green, got {:?}",
            &px1[..8]
        );

        // Submission 2 (closed-list replay): swap producer uploads to BLUE and
        // replay the exact same closed list without resetting.
        // SAFETY: producer buffers are not executing between synchronous submissions.
        unsafe {
            let blue16: [u16; 3] = [7, 8, 9];
            std::ptr::copy_nonoverlapping(blue16.as_ptr().cast::<u8>(), prod16_ptr, 6);
            let blue32: [u32; 3] = [6, 7, 8];
            std::ptr::copy_nonoverlapping(blue32.as_ptr().cast::<u8>(), prod32_ptr, 12);
        }
        host.execute(&[&list]);
        let px2 = read_pixels();
        assert!(
            px2.iter().all(|&c| c == [0, 0, 255, 255]),
            "replay expected all blue, got {:?}",
            &px2[..8]
        );

        drop((vb, prod16, prod32));
        crate::shadow::unregister(res16, title_ib16.as_ptr() as u64);
        crate::shadow::unregister(res32, title_ib32.as_ptr() as u64);
    }
}
