//! Two-pass mesh-shader fallback record layout and pass-through vertex shader
//! (`vs_6_6`) emitted directly in DXIL.
//!
//! When the host GPU lacks D3D12 mesh shaders, each console mesh thread group
//! writes one fixed-stride record into a shared raw buffer (`u0, space 3`),
//! and the pass-through vertex shader pulls vertices and per-primitive
//! attributes from that buffer (`t0, space 3`) by `SV_VertexID` and
//! `SV_InstanceID` using `DrawInstanced(max_primitives * 3, groups, 0, 0)`.

use super::HostShader;
use super::bitcode::{Attr, Bin, Cast, Term, Ty, V, pred};
use super::container::SigElement;
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};
use crate::native::HwStage;

/// Header dwords at the start of every mesh-group record: `[nv, np, 0, 0]`.
pub const MESH_HEADER_DWORDS: u32 = 4;

/// Quiet NaN (`f32`) written to `SV_Position` of culled or out-of-range
/// primitives so the rasterizer discards the triangle cleanly.
const QNAN_BITS: u32 = 0x7FC0_0000;

/// One output element (per-vertex or per-primitive) packed into a
/// mesh-fallback group record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordElement {
    pub name: String,
    pub index: u32,
    pub kind: u8,
    pub comp: u8,
    pub interp: u8,
    pub row: Option<u32>,
    pub col: u8,
    pub cols: u8,
    /// First dword offset inside a single vertex or primitive slice.
    pub offset: u32,
}

/// Fixed per-group record layout shared between Pass 1 (compute) and Pass 2
/// (pass-through vertex shader): `[header(4), vertices, prim_indices, prim_attrs]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshRecordLayout {
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub vert_dwords: u32,
    pub prim_idx_base: u32,
    pub prim_attr_base: u32,
    pub prim_dwords: u32,
    pub record_dwords: u32,
    pub vertices: Vec<RecordElement>,
    pub prims: Vec<RecordElement>,
}

impl MeshRecordLayout {
    pub(crate) fn from_signatures(
        outputs: &[SigElement],
        prims: &[SigElement],
        max_vertices: u32,
        max_primitives: u32,
    ) -> Self {
        let pack = |els: &[SigElement], force_const: bool| -> (Vec<RecordElement>, u32) {
            let mut off = 0u32;
            let out = els
                .iter()
                .map(|e| {
                    let offset = off;
                    off += u32::from(e.cols);
                    RecordElement {
                        name: e.name.clone(),
                        index: e.index,
                        kind: e.kind,
                        comp: e.comp,
                        interp: if force_const { 1 } else { e.interp },
                        row: e.row,
                        col: e.col,
                        cols: e.cols,
                        offset,
                    }
                })
                .collect();
            (out, off)
        };
        let (vertices, vert_dwords) = pack(outputs, false);
        let (prims, prim_dwords) = pack(prims, true);
        let prim_idx_base = MESH_HEADER_DWORDS + max_vertices * vert_dwords;
        let prim_attr_base = prim_idx_base + max_primitives;
        let record_dwords = (prim_attr_base + max_primitives * prim_dwords).next_multiple_of(4);
        Self {
            max_vertices,
            max_primitives,
            vert_dwords,
            prim_idx_base,
            prim_attr_base,
            prim_dwords,
            record_dwords,
            vertices,
            prims,
        }
    }

    /// Byte size of one group's record (always 16-byte aligned).
    pub fn record_bytes(&self) -> u32 {
        self.record_dwords * 4
    }
}

#[derive(Clone, Copy)]
enum Source {
    Vertex { offset: u32, cols: u8 },
    Prim { offset: u32, cols: u8 },
    Zero,
}

fn matches_semantic(a: &RecordElement, b: &SigElement) -> bool {
    a.kind == b.kind && a.index == b.index && (a.kind != 0 || a.name.eq_ignore_ascii_case(&b.name))
}

fn default_interp(kind: u8, comp: u8) -> u8 {
    match (kind, comp) {
        (3, _) => 4,
        (0 | 6 | 7, 9) => 2,
        (0, _) => 1,
        _ => 0,
    }
}

fn make_sig(
    name: &str,
    index: u32,
    kind: u8,
    comp: u8,
    interp: u8,
    row: u32,
    cols: u8,
) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp,
        interp,
        row: Some(row),
        col: 0,
        cols,
        usage: (1u8 << cols) - 1,
        half: false,
    }
}

fn match_outputs(
    layout: &MeshRecordLayout,
    ps_in: Option<&[SigElement]>,
) -> Vec<(SigElement, Source)> {
    let mut out: Vec<(SigElement, Source)> = Vec::new();
    for e in ps_in
        .unwrap_or_default()
        .iter()
        .filter(|e| matches!(e.kind, 0 | 3 | 6 | 7))
    {
        let (src, interp) = if let Some(v) = layout.vertices.iter().find(|v| matches_semantic(v, e))
        {
            (
                Source::Vertex {
                    offset: v.offset,
                    cols: v.cols,
                },
                default_interp(e.kind, e.comp),
            )
        } else if let Some(p) = layout.prims.iter().find(|p| matches_semantic(p, e)) {
            (
                Source::Prim {
                    offset: p.offset,
                    cols: p.cols,
                },
                1,
            )
        } else {
            (Source::Zero, default_interp(e.kind, e.comp))
        };
        let mut sig = e.clone();
        sig.interp = interp;
        sig.usage = (1u8 << sig.cols) - 1;
        sig.half = false;
        out.push((sig, src));
    }
    let next_row = |els: &[(SigElement, Source)]| {
        els.iter()
            .filter_map(|(e, _)| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0)
    };
    if !out.iter().any(|(e, _)| e.kind == 3) {
        let row = next_row(&out);
        let (cols, src) = match layout.vertices.iter().find(|v| v.kind == 3) {
            Some(v) => (
                v.cols,
                Source::Vertex {
                    offset: v.offset,
                    cols: v.cols,
                },
            ),
            None => (4, Source::Zero),
        };
        out.push((make_sig("SV_Position", 0, 3, 9, 4, row, cols), src));
    }
    for v in layout.vertices.iter().filter(|v| matches!(v.kind, 6 | 7)) {
        if !out
            .iter()
            .any(|(e, _)| e.kind == v.kind && e.index == v.index)
        {
            let row = next_row(&out);
            let sig = make_sig(
                &v.name,
                v.index,
                v.kind,
                v.comp,
                default_interp(v.kind, v.comp),
                row,
                v.cols,
            );
            out.push((
                sig,
                Source::Vertex {
                    offset: v.offset,
                    cols: v.cols,
                },
            ));
        }
    }
    // DXIL forbids mixed interpolation modes in a single row; integers and
    // per-primitive outputs require constant (1), so unify any row containing one.
    let const_rows: Vec<u32> = out
        .iter()
        .filter(|(e, _)| e.interp == 1)
        .filter_map(|(e, _)| e.row)
        .collect();
    for (e, _) in &mut out {
        if e.row.is_some_and(|r| const_rows.contains(&r)) {
            e.interp = 1;
        }
    }
    out
}

/// Emits a pass-through `vs_6_6` pulling vertices and per-primitive attributes
/// from `t0, space 3` and matching `ps_dxil`'s `ISG1` (if provided).
pub fn emit_passthrough_vs(
    layout: &MeshRecordLayout,
    ps_dxil: Option<&[u8]>,
) -> Result<HostShader, ShaderError> {
    let ps_inputs = match ps_dxil {
        Some(bytes) => {
            let c = Container::parse(bytes)?;
            Some(match c.part(FourCC::ISG1) {
                Some(part) => super::signature(part.data)?,
                None => Vec::new(),
            })
        }
        None => None,
    };
    let outputs = match_outputs(layout, ps_inputs.as_deref());
    let mut s = Shader::new(false);
    let (i32t, f32t, void) = (s.t.i32, s.t.f32, s.t.void);
    let ret_i32 = s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);

    s.inputs = vec![
        make_sig("SV_VertexID", 0, 1, 5, 0, 0, 1),
        make_sig("SV_InstanceID", 0, 2, 5, 0, 1, 1),
    ];
    s.outputs = outputs.iter().map(|(e, _)| e.clone()).collect();

    let (entry, check_prim, live, culled) =
        (s.entry, s.m.new_block(), s.m.new_block(), s.m.new_block());
    let handle = s.mesh_record_srv_handle();
    let (iu, k0, k1, k2, k3, k4) = (
        s.m.undef(i32t),
        s.m.i32(0),
        s.m.i32(1),
        s.m.i32(2),
        s.m.i32(3),
        s.m.i32(4),
    );
    let raw_load = |s: &mut Shader, blk: usize, byte: V, mask: u8| {
        let m = s.m.i8(mask);
        s.op(
            blk,
            "rawBufferLoad.i32",
            Attr::ReadOnly,
            ret_i32,
            139,
            &[handle, byte, iu, m, k4],
        )
    };

    let vid_args = [k0, k0, s.m.i8(0), iu];
    let vid = s.op(entry, "loadInput.i32", Attr::ReadNone, i32t, 4, &vid_args);
    let iid_args = [k1, k0, s.m.i8(0), iu];
    let iid = s.op(entry, "loadInput.i32", Attr::ReadNone, i32t, 4, &iid_args);

    let stride = s.m.i32(layout.record_bytes());
    let rec_bytes = s.m.bin(entry, Bin::Mul, iid, stride);
    let hdr = raw_load(&mut s, entry, rec_bytes, 0x3);
    let (raw_nv, raw_np) = (s.m.extract(entry, hdr, 0), s.m.extract(entry, hdr, 1));
    let (max_v, max_p) = (s.m.i32(layout.max_vertices), s.m.i32(layout.max_primitives));
    let nv = {
        let c = s.m.cmp(entry, pred::IULT, raw_nv, max_v);
        s.m.select(entry, c, raw_nv, max_v)
    };
    let np = {
        let c = s.m.cmp(entry, pred::IULT, raw_np, max_p);
        s.m.select(entry, c, raw_np, max_p)
    };
    let prim = s.m.bin(entry, Bin::UDiv, vid, k3);
    let corner = s.m.bin(entry, Bin::URem, vid, k3);
    let in_range = s.m.cmp(entry, pred::IULT, prim, np);
    s.m.terminate(entry, Term::CondBr(in_range, check_prim, culled));

    // Unpack `exp prim`: bit 31 culls; each ten-bit corner has a nine-bit index.
    let prim_off = s.m.bin(check_prim, Bin::Shl, prim, k2);
    let idx_base = s.m.i32(layout.prim_idx_base * 4);
    let p_off = s.m.bin(check_prim, Bin::Add, idx_base, prim_off);
    let p_byte = s.m.bin(check_prim, Bin::Add, rec_bytes, p_off);
    let p_load = raw_load(&mut s, check_prim, p_byte, 0x1);
    let packed = s.m.extract(check_prim, p_load, 0);
    let not_null = s.m.cmp(check_prim, pred::SGE, packed, k0);
    let index_mask = s.m.i32(crate::io::MESH_VERTEX_INDEX_MASK);
    let mut vs = [k0; 3];
    let mut ok = not_null;
    for k in 0..3u32 {
        let sh = if k == 0 {
            packed
        } else {
            let sft = s.m.i32(10 * k);
            s.m.bin(check_prim, Bin::LShr, packed, sft)
        };
        vs[k as usize] = s.m.bin(check_prim, Bin::And, sh, index_mask);
        let lt = s.m.cmp(check_prim, pred::IULT, vs[k as usize], nv);
        ok = s.m.bin(check_prim, Bin::And, ok, lt);
    }
    s.m.terminate(check_prim, Term::CondBr(ok, live, culled));

    // Live vertex: pick corner's vertex index and load attributes.
    let is1 = s.m.cmp(live, pred::EQ, corner, k1);
    let is2 = s.m.cmp(live, pred::EQ, corner, k2);
    let v01 = s.m.select(live, is1, vs[1], vs[0]);
    let vi = s.m.select(live, is2, vs[2], v01);
    let slice_base = |s: &mut Shader, idx: V, stride_dw: u32, base_dw: u32| {
        (stride_dw > 0).then(|| {
            let st = s.m.i32(stride_dw * 4);
            let rel = s.m.bin(live, Bin::Mul, idx, st);
            let b = s.m.i32(base_dw * 4);
            let off = s.m.bin(live, Bin::Add, b, rel);
            s.m.bin(live, Bin::Add, rec_bytes, off)
        })
    };
    let v_base = slice_base(&mut s, vi, layout.vert_dwords, MESH_HEADER_DWORDS);
    let p_base = slice_base(&mut s, prim, layout.prim_dwords, layout.prim_attr_base);
    let (qnan, fzero) = (s.m.f32(QNAN_BITS), s.m.f32(0));

    for (id, (el, src)) in outputs.iter().enumerate() {
        let (base, offset, avail) = match *src {
            Source::Vertex { offset, cols } => (v_base, offset, cols),
            Source::Prim { offset, cols } => (p_base, offset, cols),
            Source::Zero => (None, 0, 0),
        };
        let n = el.cols.min(avail);
        let loaded = (n > 0).then_some(base).flatten().map(|base| {
            let byte = if offset == 0 {
                base
            } else {
                let o = s.m.i32(offset * 4);
                s.m.bin(live, Bin::Add, base, o)
            };
            raw_load(&mut s, live, byte, (1u8 << n) - 1)
        });
        for col in 0..el.cols {
            let val = match loaded {
                Some(r) if col < n => {
                    let x = s.m.extract(live, r, u32::from(col));
                    if el.comp == 9 {
                        s.m.cast(live, Cast::Bitcast, x, f32t)
                    } else {
                        x
                    }
                }
                _ if el.comp == 9 => fzero,
                _ => k0,
            };
            store_output(&mut s, live, id as u32, col, el.comp, val, void);
        }
    }
    s.m.terminate(live, Term::Ret);

    for (id, (el, _)) in outputs.iter().enumerate() {
        for col in 0..el.cols {
            let val = if el.kind == 3 {
                qnan
            } else if el.comp == 9 {
                fzero
            } else {
                k0
            };
            store_output(&mut s, culled, id as u32, col, el.comp, val, void);
        }
    }
    s.m.terminate(culled, Term::Ret);

    Ok(HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
    })
}

fn store_output(s: &mut Shader, blk: usize, id: u32, col: u8, comp: u8, val: V, void: Ty) {
    let suffix = if comp == 9 { ".f32" } else { ".i32" };
    let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), val];
    s.op(
        blk,
        &format!("storeOutput{suffix}"),
        Attr::NoUnwind,
        void,
        5,
        &args,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(name: &str, index: u32, kind: u8, comp: u8, row: u32, col: u8, cols: u8) -> SigElement {
        SigElement {
            col,
            ..make_sig(
                name,
                index,
                kind,
                comp,
                default_interp(kind, comp),
                row,
                cols,
            )
        }
    }

    #[test]
    fn record_layout_packs_vertices_indices_and_prims_aligned() {
        let out = vec![
            el("SV_Position", 0, 3, 9, 0, 0, 4),
            el("TEXCOORD", 0, 0, 9, 1, 0, 2),
        ];
        let prim = vec![el("TEXCOORD", 1, 0, 5, 0, 0, 1)];
        let l = MeshRecordLayout::from_signatures(&out, &prim, 64, 32);
        assert_eq!(l.vert_dwords, 6);
        assert_eq!(l.prim_idx_base, MESH_HEADER_DWORDS + 64 * 6);
        assert_eq!(l.prim_attr_base, l.prim_idx_base + 32);
        assert_eq!(l.prim_dwords, 1);
        assert_eq!(l.record_dwords % 4, 0);
        assert_eq!(
            (
                l.vertices[0].offset,
                l.vertices[1].offset,
                l.prims[0].offset
            ),
            (0, 4, 0)
        );
    }

    #[test]
    fn passthrough_vs_matches_ps_isg1_including_psg1_clip_and_missing() {
        let ms_out = vec![
            el("SV_Position", 0, 3, 9, 0, 0, 4),
            el("TEXCOORD", 0, 0, 9, 1, 0, 4),
            el("SV_ClipDistance", 0, 6, 9, 2, 0, 2),
        ];
        let layout =
            MeshRecordLayout::from_signatures(&ms_out, &[el("TEXCOORD", 1, 0, 5, 0, 0, 1)], 64, 64);

        let mut ps = Shader::new(true);
        ps.inputs = vec![
            el("TEXCOORD", 0, 0, 9, 0, 0, 4),
            SigElement {
                interp: 1,
                ..el("TEXCOORD", 1, 0, 5, 1, 0, 1)
            },
            SigElement {
                interp: 1,
                ..el("TEXCOORD", 2, 0, 9, 1, 1, 2)
            },
            SigElement {
                interp: 1,
                ..el("SV_IsFrontFace", 0, 13, 5, 2, 0, 1)
            },
        ];
        ps.outputs = vec![el("SV_Target", 0, 16, 9, 0, 0, 4)];
        let b = ps.entry;
        for c in 0..4u8 {
            let args = [ps.m.i32(0), ps.m.i32(0), ps.m.i8(c), ps.m.f32(0)];
            ps.op(b, "storeOutput.f32", Attr::NoUnwind, ps.t.void, 5, &args);
        }
        ps.m.terminate(b, Term::Ret);
        let ps_dxil = ps.finish();

        let vs = emit_passthrough_vs(&layout, Some(&ps_dxil)).unwrap();
        assert_eq!(vs.stage, HwStage::Vs);
        let osg = super::super::signature(
            Container::parse(&vs.dxil)
                .unwrap()
                .part(FourCC::OSG1)
                .unwrap()
                .data,
        )
        .unwrap();
        assert_eq!(osg.len(), 5);
        assert!(
            osg.iter()
                .any(|e| e.name == "TEXCOORD" && e.index == 0 && e.row == Some(0))
        );
        assert!(
            osg.iter()
                .any(|e| e.name == "TEXCOORD" && e.index == 1 && e.row == Some(1) && e.col == 0)
        );
        assert!(
            osg.iter()
                .any(|e| e.name == "TEXCOORD" && e.index == 2 && e.row == Some(1) && e.col == 1)
        );
        assert!(osg.iter().any(|e| e.kind == 3 && e.row == Some(2)));
        assert!(osg.iter().any(|e| e.kind == 6 && e.row == Some(3)));

        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            let mut copy = vs.dxil.clone();
            v.validate(&mut copy).unwrap();
            assert_eq!(copy[4..20], vs.dxil[4..20]);
        }
    }

    #[test]
    fn passthrough_vs_without_ps_validates() {
        let layout =
            MeshRecordLayout::from_signatures(&[el("SV_Position", 0, 3, 9, 0, 0, 4)], &[], 3, 1);
        let vs = emit_passthrough_vs(&layout, None).unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
    }

    #[test]
    fn mesh_fallback_compute_bindings_validate() {
        let mut s = Shader::new(false);
        s.threads = Some([64, 1, 1]);
        s.wave_size = Some(64);
        let b = s.entry;
        let uav = s.mesh_record_uav_handle();
        let cbv = s.mesh_base_cbv_handle();
        let (k0, k4, iu, mask1, void) = (
            s.m.i32(0),
            s.m.i32(4),
            s.m.undef(s.t.i32),
            s.m.i8(1),
            s.t.void,
        );
        let base_r = s.op(
            b,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            s.t.cbuf_ret,
            59,
            &[cbv, k0],
        );
        let base = s.m.extract(b, base_r, 0);
        let off = s.m.bin(b, Bin::Mul, base, k4);
        s.op(
            b,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[uav, off, iu, base, iu, iu, iu, mask1, k4],
        );
        s.m.terminate(b, Term::Ret);
        let dxil = s.finish();
        let c = Container::parse(&dxil).unwrap();
        assert!(c.part(FourCC::PSV0).is_some());
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            let mut copy = dxil.clone();
            v.validate(&mut copy).unwrap();
            assert_eq!(copy[4..20], dxil[4..20]);
        }
    }

    #[test]
    fn emit_mesh_fallback_emits_cs_and_preserves_native_ms() {
        use crate::native::{MeshIo, NativeProgram, StageIo};
        let osg1 =
            super::super::container::signature(&[el("SV_Position", 0, 3, 9, 0, 0, 4)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: true,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 64,
                max_vertices: 3,
                max_primitives: 1,
                topology: 2,
                lds_bytes: 0,
            }),
            code: [
                0x7E00_0280u32, // v_mov_b32 v0, 0
                0xF800_08CF,    // exp pos0 done
                0x0000_0000,    // v0, v0, v0, v0
                0xBF81_0000,    // s_endpgm
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
        };
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let fb = super::super::emit_mesh_fallback(&c, &native, &p, &st, &io, &res).unwrap();
        assert_eq!(fb.compute.stage, HwStage::Cs);
        let ms = super::super::emit(&c, &native, &p, &st, &io, &res).unwrap();
        assert_eq!(ms.stage, HwStage::Ms);
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut fb.compute.dxil.clone()).unwrap();
            v.validate(&mut ms.dxil.clone()).unwrap();
        }
    }

    #[cfg(windows)]
    const TRIANGLE: u32 = (1 << 10) | (2 << 20);

    /// Builds a synthetic GFX10 mesh program (`threads = 128`, 2+ waves) in
    /// wave32 or wave64 exercising `MeshGroup`, `MergedWaveInfo`, `v_mbcnt`,
    /// `s_sendmsg(GS_ALLOC_REQ)`, `OSG1`, `PSG1`, bit-31 null & oob culling.
    #[cfg(windows)]
    fn synthetic_mesh(
        wave64: bool,
        nv: u32,
        np: u32,
        scalar_input: bool,
        first_primitive: u32,
    ) -> super::super::MeshFallbackShader {
        use crate::native::{MeshIo, NativeProgram, StageIo, Usage, VsOutput};
        let mut w: Vec<u32> = Vec::new();
        let s_mov_lit = |w: &mut Vec<u32>, sdst: u32, lit: u32| {
            w.push(0xBE80_03FF | (sdst << 16));
            w.push(lit);
        };
        let v_mov_lit = |w: &mut Vec<u32>, vdst: u32, lit: u32| {
            w.push(0x7E00_02FF | (vdst << 17));
            w.push(lit);
        };
        let v_cmp_eq_lit = |w: &mut Vec<u32>, vsrc: u32, lit: u32| {
            w.push(0x7D84_00FF | (vsrc << 9));
            w.push(lit);
        };
        let v_cndmask = |w: &mut Vec<u32>, vdst: u32, vfalse: u32, vtrue: u32| {
            w.push(0x0200_0000 | (vdst << 17) | (vtrue << 9) | (256 + vfalse));
        };
        let exp = |w: &mut Vec<u32>, target: u32, mask: u32, done: bool, v: [u32; 4]| {
            w.push(0xF800_0000 | (u32::from(done) << 11) | (target << 4) | mask);
            w.push(v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24));
        };

        // s2 = group_id = v5 >> 7 (mesh_stride = 128); s4 = wave_id << (6|5).
        w.push(0x7E04_0505); // v_readfirstlane_b32 s2, v5
        w.push(0x9002_8702); // s_lshr_b32 s2, s2, 7
        w.push(0x9004_9803); // s_lshr_b32 s4, s3, 24
        w.push(0x8F04_0004 | ((if wave64 { 0x86 } else { 0x85 }) << 8));
        // v0 = lane_index via v_mbcnt_lo (+ hi for wave64), then v0 += s4 -> flat id.
        w.extend([0xD765_0000, (0x80 << 9) | 0xC1]);
        if wave64 {
            w.extend([0xD766_0000, (256 << 9) | 0xC1]);
        }
        w.push(0x4A00_0004); // v_add_nc_u32 v0, s4, v0
        // s_sendmsg(GS_ALLOC_REQ): m0 = nv | (np << 12).
        s_mov_lit(&mut w, 124, nv | (np << 12));
        w.push(0xBF90_0009);
        // Group 0: x in [-1, 0]; Group 1: x in [0, 1].
        v_mov_lit(&mut w, 1, (-1.0f32).to_bits());
        v_mov_lit(&mut w, 2, 0.0f32.to_bits());
        v_mov_lit(&mut w, 3, 1.0f32.to_bits());
        w.push(0x7E08_0202); // v_mov_b32 v4, s2
        v_cmp_eq_lit(&mut w, 4, 0);
        v_cndmask(&mut w, 1, 2, 1);
        v_cndmask(&mut w, 2, 3, 2);
        // v0 == 0 -> (v1, -1), v0 == 1 -> (v2, -1), v0 == 2 -> (v1, 3).
        v_mov_lit(&mut w, 6, (-1.0f32).to_bits());
        v_mov_lit(&mut w, 7, 3.0f32.to_bits());
        v_mov_lit(&mut w, 8, 0.0f32.to_bits());
        v_mov_lit(&mut w, 9, 1.0f32.to_bits());
        w.push(0x7E14_0301);
        v_cmp_eq_lit(&mut w, 0, 1);
        v_cndmask(&mut w, 10, 10, 2);
        w.push(0x7E16_0306);
        v_cmp_eq_lit(&mut w, 0, 2);
        v_cndmask(&mut w, 11, 11, 7);
        exp(&mut w, 12, 0xF, false, [10, 11, 8, 9]); // exp pos0
        // Per-vertex TEXCOORD0 = (0.25, 0.0, 0.75, 1.0); per-prim TEXCOORD1 = group_id + 1.
        if scalar_input {
            // Heap base is zero; s[20:23] names slot 0, with no UsedResource.
            w.extend([
                0xBE93_0380, // s_mov_b32 s19, 0
                0xF408_0509,
                0xFA00_0000, // s_load_dwordx4 s[20:23], s[18:19], 0
                0xF420_060A,
                0xFA00_0004, // s_buffer_load_dword s24, s[20:23], 4
                0x7E18_0218, // v_mov_b32 v12, s24
            ]);
        } else {
            v_mov_lit(&mut w, 12, 0.25f32.to_bits());
        }
        v_mov_lit(&mut w, 13, 0.0f32.to_bits());
        v_mov_lit(&mut w, 14, 0.75f32.to_bits());
        exp(&mut w, 32, 0xF, false, [12, 13, 14, 9]); // exp param0
        w.push(0x4A1E_0881); // v_add_nc_u32 v15, 1, v4
        exp(&mut w, 33, 0x1, false, [15, 15, 15, 15]); // exp param1
        // Prim 0: test word; prim 1: bit-31 null; prim 2: oob vertex 5.
        v_mov_lit(&mut w, 16, (1 << 10) | (5 << 20));
        v_mov_lit(&mut w, 17, (1 << 31) | TRIANGLE);
        v_mov_lit(&mut w, 18, first_primitive);
        v_cmp_eq_lit(&mut w, 0, 1);
        v_cndmask(&mut w, 16, 16, 17);
        v_cmp_eq_lit(&mut w, 0, 0);
        v_cndmask(&mut w, 16, 16, 18);
        exp(&mut w, 20, 0x1, true, [16, 16, 16, 16]); // exp prim done
        w.push(0xBF81_0000); // s_endpgm

        let osg1 = super::super::container::signature(
            &[
                el("SV_Position", 0, 3, 9, 0, 0, 4),
                el("TEXCOORD", 0, 0, 9, 1, 0, 4),
            ],
            false,
        );
        let psg1 = super::super::container::signature(
            &[SigElement {
                interp: 1,
                ..el("TEXCOORD", 1, 0, 5, 0, 0, 1)
            }],
            false,
        );
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1), (FourCC::PSG1, &psg1)]);
        let c = Container::parse(&bytes).unwrap();
        let out = |param, row| VsOutput {
            param,
            usage: Usage::Generic,
            row,
            fp16: false,
            pack2: false,
        };
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64,
            vgprs: 20,
            layout: if scalar_input {
                vec![crate::RootParam {
                    root: crate::RootType::ResourceHeap,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 0,
                    user_data: 10, // mesh user SGPRs start at s8
                    slot: 0,
                }]
            } else {
                vec![]
            },
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs: vec![out(0, 1), out(0x81, 0)],
                clip_cntl: 0,
                threads: 128,
                max_vertices: 6,
                max_primitives: 3,
                topology: 2,
                lds_bytes: 0,
            }),
            code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        super::super::emit_mesh_fallback(&c, &native, &p, &st, &io, &res).unwrap()
    }

    #[cfg(windows)]
    fn test_ps() -> HostShader {
        let mut s = Shader::new(true);
        s.inputs = vec![
            el("TEXCOORD", 0, 0, 9, 0, 0, 4),
            SigElement {
                interp: 1,
                ..el("TEXCOORD", 1, 0, 5, 1, 0, 1)
            },
        ];
        s.outputs = vec![el("SV_Target", 0, 16, 9, 0, 0, 4)];
        let b = s.entry;
        let (k0, k1, iu, f32t, i32t, void) = (
            s.m.i32(0),
            s.m.i32(1),
            s.m.undef(s.t.i32),
            s.t.f32,
            s.t.i32,
            s.t.void,
        );
        let (c0, c2) = (s.m.i8(0), s.m.i8(2));
        let r = s.op(
            b,
            "loadInput.f32",
            Attr::ReadNone,
            f32t,
            4,
            &[k0, k0, c0, iu],
        );
        let bl = s.op(
            b,
            "loadInput.f32",
            Attr::ReadNone,
            f32t,
            4,
            &[k0, k0, c2, iu],
        );
        let pid = s.op(
            b,
            "loadInput.i32",
            Attr::ReadNone,
            i32t,
            4,
            &[k1, k0, c0, iu],
        );
        let pf = s.m.cast(b, Cast::UiToFp, pid, f32t);
        let q = s.m.f32(0.25f32.to_bits());
        let g = s.m.bin(b, Bin::Mul, pf, q);
        let one = s.m.f32(1.0f32.to_bits());
        for (c, v) in [r, g, bl, one].into_iter().enumerate() {
            let args = [k0, k0, s.m.i8(c as u8), v];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, void, 5, &args);
        }
        s.m.terminate(b, Term::Ret);
        HostShader {
            stage: HwStage::Ps,
            dxil: s.finish(),
            user_data: vec![],
            samplers: vec![],
            static_samplers: vec![],
            root_cbvs: vec![],
            root_srvs: vec![],
            root_uavs: vec![],
            memory: false,
            memory_writes: false,
            view_addresses: false,
            heap: false,
            heap_writes: false,
        }
    }

    #[test]
    #[cfg(windows)]
    fn bindless_scalar_load_survives_mesh_fallback() {
        let host = super::super::host::Host::new(false).expect("hardware SM 6.6 device");
        let fb = synthetic_mesh(false, 3, 1, true, TRIANGLE);
        let ps = test_ps();
        let value = 0.375f32;
        let pixels = host
            .draw_mesh_fallback(&fb, &ps, 1, 0, (16, 16), Some(&[0, value.to_bits(), 0, 0]))
            .unwrap();
        assert_eq!(pixels[14 * 16 + 2], [value, 0.25, 0.75, 1.0]);
    }

    #[test]
    #[cfg(windows)]
    fn mesh_fallback_ignores_all_edge_flags_but_preserves_culling() {
        use super::super::host::Host;
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
            IDXGIAdapter1, IDXGIFactory4,
        };
        let host = Host::new(false).expect("hardware SM 6.6 device");
        // SAFETY: query the adapter of the live device used by the test.
        let adapter = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapterByLuid(host.device.GetAdapterLuid())
                .unwrap();
            adapter.GetDesc1().unwrap()
        };
        assert_eq!(adapter.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        let name = String::from_utf16_lossy(&adapter.Description);
        eprintln!(
            "edge-flag regression adapter: {}",
            name.trim_end_matches('\0')
        );
        let (min_wave, max_wave) = host.wave_lane_count().unwrap();
        let wave64 = !(min_wave..=max_wave).contains(&32);
        let ps = test_ps();
        let render = |nv, prim| {
            let fb = synthetic_mesh(wave64, nv, 1, false, prim);
            host.draw_mesh_fallback(&fb, &ps, 1, 0, (16, 16), None)
                .unwrap()
        };
        let plain = render(3, TRIANGLE);
        assert_eq!(plain[14 * 16 + 2], [0.25, 0.25, 0.75, 1.0]);
        let flags = [1 << 9, 1 << 19, 1 << 29];
        let all = flags.into_iter().fold(0, |a, b| a | b);
        for flag in flags.into_iter().chain([all]) {
            assert_eq!(render(3, TRIANGLE | flag), plain, "edge flags {flag:#x}");
        }
        // Keep nondegenerate geometry in the record for both controls.
        // Lowering nv to 2 makes index 2 invalid, even though it fits the
        // record capacity and the shader writes its position.
        for (nv, prim) in [(3, TRIANGLE | all | (1 << 31)), (2, TRIANGLE | all)] {
            assert!(
                render(nv, prim).iter().all(|&px| px == [0.0; 4]),
                "nv={nv}, primitive {prim:#x}"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn mesh_fallback_draws_multiwave_wave32_and_wave64_with_psg1_and_sendmsg() {
        let Ok(host) = super::super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let (min_w, max_w) = host.wave_lane_count().unwrap();
        eprintln!("host wave lane count: min={min_w}, max={max_w}");
        let ps = test_ps();
        let mut tested = 0;
        for (wave64, wsize) in [(true, 64u32), (false, 32u32)] {
            if wsize < min_w || wsize > max_w {
                continue;
            }
            tested += 1;
            let fb = synthetic_mesh(wave64, 3, 2, false, TRIANGLE);
            assert_eq!(fb.compute.stage, HwStage::Cs);
            if let Ok(mut v) = super::super::validator::Validator::load(
                super::super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut fb.compute.dxil.clone()).unwrap();
            }
            for base_group in [0u32, 3u32] {
                let img = host
                    .draw_mesh_fallback(&fb, &ps, 2, base_group, (16, 16), None)
                    .unwrap();
                assert_eq!(
                    img[14 * 16 + 2],
                    [0.25, 0.25, 0.75, 1.0],
                    "wave{wsize} base={base_group} g0"
                );
                assert_eq!(
                    img[14 * 16 + 10],
                    [0.25, 0.50, 0.75, 1.0],
                    "wave{wsize} base={base_group} g1"
                );
                assert_eq!(
                    img[1 * 16 + 7],
                    [0.0; 4],
                    "wave{wsize} base={base_group} bg0"
                );
                assert_eq!(
                    img[1 * 16 + 15],
                    [0.0; 4],
                    "wave{wsize} base={base_group} bg1"
                );
            }
            let empty = synthetic_mesh(wave64, 3, 0, false, TRIANGLE);
            let img0 = host
                .draw_mesh_fallback(&empty, &ps, 2, 0, (16, 16), None)
                .unwrap();
            assert!(
                img0.iter().all(|&px| px == [0.0; 4]),
                "wave{wsize} np=0 must draw nothing"
            );
        }
        assert!(
            tested > 0,
            "expected at least one supported wave size in [{min_w}..={max_w}]"
        );
    }
}
