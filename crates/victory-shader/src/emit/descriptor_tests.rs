//! End-to-end D3D12 tests on hardware GPU and WARP (`dxil.dll` validated)
//! for rebuilding every slice of merged `s_load_dwordx8` / `s_load_dwordx16`
//! descriptor loads, including split same-base reloads and cube image slices.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, StaticSampler, lower, validator};
use crate::ShaderError;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

const fn inline_u32(n: u8) -> u16 {
    128 + n as u16
}

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32 & 0x7F) << 16 | (op as u32) << 8 | src0 as u32
}

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    sop1(0x03, sdst, src0)
}

const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

const fn smem_load(op: u8, dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    [
        0x3D << 26 | (op as u32) << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        (125 << 25) | (byte_off & 0x1F_FFFF),
    ]
}

const fn s_load_dwordx4(dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    smem_load(0x02, dst, base, byte_off)
}

const fn s_load_dwordx8(dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    smem_load(0x03, dst, base, byte_off)
}

const fn s_load_dwordx16(dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    smem_load(0x04, dst, base, byte_off)
}

const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

const fn mubuf(
    op: u8,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    offen: bool,
    idxen: bool,
    byte_off: u16,
) -> [u32; 2] {
    [
        0x38 << 26
            | (op as u32 & 0x7F) << 18
            | (idxen as u32) << 13
            | (offen as u32) << 12
            | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn mimg(op: u8, vdata: u8, vaddr: u8, srsrc: u8, ssamp: u8, dmask: u8, dim: u8) -> [u32; 2] {
    [
        0x3C << 26 | (op as u32 & 0xFF) << 18 | (dmask as u32 & 0xF) << 8 | (dim as u32 & 7) << 3,
        ((ssamp as u32 >> 2) & 0x1F) << 21
            | ((srsrc as u32 >> 2) & 0x1F) << 16
            | (vdata as u32) << 8
            | vaddr as u32,
    ]
}

fn translate_cs(
    words: &[u32],
    threads: u32,
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<HostShader, ShaderError> {
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: sgprs.max(2) << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, false, 32)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

fn run_cs(shader: &HostShader, views: &[View]) -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label.into(), words[0].clone()));
    }
    assert!(
        !out.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    out
}

fn root_table() -> Vec<RootParam> {
    vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }]
}

fn used_range(
    kind: ResourceType,
    table_offset_dwords: u32,
    dwords: u8,
    stride: u32,
) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    }
}

#[test]
fn merged_loads_and_split_reloads_rebuild_all_buffer_slices() {
    // 1. `s_load_dwordx8` at +0x10 covering two adjacent 4-dword slices (raw SRV at +0x10 -> view 0,
    //    structured SRV at +0x20 -> view 1).
    let mut x8 = vec![s_mov_b32(4, 0), s_mov_b32(5, 0x80)];
    x8.extend_from_slice(&s_load_dwordx8(8, 4, 0x10));
    x8.extend_from_slice(&s_load_dwordx4(24, 4, 0x40)); // UAV at offset 16 -> view 2
    x8.push(vop2(0x1A, 16, inline_u32(2), 0)); // v16 = v0 << 2
    x8.push(vop2(0x1B, 2, inline_u32(3), 0)); // v2 = v0 & 3
    x8.push(vop2(0x1A, 3, inline_u32(2), 2)); // v3 = (v0 & 3) << 2
    x8.extend_from_slice(&mubuf(0x0C, 1, 3, 8, true, false, 0)); // raw[v0 & 3]
    x8.extend_from_slice(&mubuf(0x0C, 4, 2, 12, false, true, 8)); // struct[v0 & 3].dword2
    x8.extend_from_slice(&vop3(0x125, 1, vgpr(4), vgpr(1), 0));
    x8.extend_from_slice(&vop3(0x125, 1, sgpr(10), vgpr(1), 0)); // raw byte count = 72
    x8.extend_from_slice(&vop3(0x125, 1, sgpr(14), vgpr(1), 0)); // structured count = 5
    x8.extend_from_slice(&mubuf(0x1C, 1, 16, 24, true, false, 0));
    x8.push(S_ENDPGM);

    let buf0: Vec<u32> = (0..18).map(|i| 100 + i).collect(); // 72 bytes
    let mut buf1 = Vec::new();
    for r in 0..5u32 {
        buf1.extend_from_slice(&[400 + r, 500 + r, 600 + r, 700 + r]);
    }
    let views = [
        View::Buffer {
            data: &buf0,
            stride: Stride::Raw,
        },
        View::Buffer {
            data: &buf1,
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let x8_used = vec![
        used_range(ResourceType::Srv, 4, 4, 0),
        used_range(ResourceType::Srv, 8, 4, 16),
        used_range(ResourceType::Uav, 16, 4, 0),
    ];
    let x8_shader = translate_cs(&x8, 32, root_table(), x8_used).expect("x8 translates");
    let want: Vec<u32> = (0..32)
        .map(|l| (100 + (l & 3)) + (600 + (l & 3)) + 72 + 5)
        .collect();
    for (label, got) in run_cs(&x8_shader, &views) {
        assert_eq!(got, want, "{label} (x8)");
    }

    // 2. `s_load_dwordx16` with split same-base reloads:
    //    - Direct slice 0 on load 0 + reload B supplying slice 8;
    //    - Pure scalar-read load 0 + reload A supplying slice 0 + reload B supplying slice 8
    //      (either full `s_load_dwordx16` at 0x0 or narrow `s_load_dwordx4` at 0x20).
    let build_x16 = |direct_first: bool, narrow_second: bool, bad_word: Option<u8>| -> Vec<u32> {
        let mut w = vec![s_mov_b32(4, 0), s_mov_b32(5, 0x80)];
        w.extend_from_slice(&s_load_dwordx16(8, 4, 0x0)); // s[8:23]: scalar metadata reads
        w.extend_from_slice(&s_load_dwordx4(56, 4, 0x40)); // UAV at offset 16
        if !direct_first {
            w.extend_from_slice(&s_load_dwordx16(24, 4, 0x0)); // reload A supplies slice 0
        }
        if narrow_second {
            w.extend_from_slice(&s_load_dwordx4(48, 4, 0x20)); // reload B at +0x20 supplies slice 8
        } else {
            w.extend_from_slice(&s_load_dwordx16(40, 4, 0x0)); // reload B at 0x0 supplies slice 8
        }
        let rsrc0 = if direct_first { 8 } else { 24 };
        let rsrc1 = 48;
        w.push(vop2(0x1A, 16, inline_u32(2), 0)); // v16 = v0 << 2
        w.push(vop2(0x1B, 2, inline_u32(3), 0)); // v2 = v0 & 3
        w.push(vop2(0x1A, 3, inline_u32(2), 2)); // v3 = (v0 & 3) << 2
        w.extend_from_slice(&mubuf(0x0C, 1, 3, rsrc0, true, false, 0));
        w.extend_from_slice(&mubuf(0x0C, 4, 2, rsrc1, false, true, 8));
        w.extend_from_slice(&vop3(0x125, 1, vgpr(4), vgpr(1), 0));
        w.extend_from_slice(&vop3(0x125, 1, sgpr(10), vgpr(1), 0)); // slice 0 raw byte count = 72
        w.extend_from_slice(&vop3(0x125, 1, sgpr(18), vgpr(1), 0)); // slice 8 structured count = 5
        if let Some(reg) = bad_word {
            w.extend_from_slice(&vop3(0x125, 1, sgpr(reg), vgpr(1), 0));
        }
        w.extend_from_slice(&mubuf(0x1C, 1, 16, 56, true, false, 0));
        w.push(S_ENDPGM);
        w
    };
    let x16_used = || {
        vec![
            used_range(ResourceType::Srv, 0, 4, 0),
            used_range(ResourceType::Srv, 8, 4, 16),
            used_range(ResourceType::Uav, 16, 4, 0),
        ]
    };
    for (direct_first, narrow_second) in [(true, false), (false, false), (false, true)] {
        let code = build_x16(direct_first, narrow_second, None);
        let shader = translate_cs(&code, 32, root_table(), x16_used()).expect("x16 translates");
        for (label, got) in run_cs(&shader, &views) {
            assert_eq!(
                got, want,
                "{label} (direct {direct_first}, narrow {narrow_second})"
            );
        }
    }
    // Reading un-rebuilt address/format dwords (`s9` or `s17`) still fails.
    for reg in [8u8, 9, 17] {
        let code = build_x16(false, true, Some(reg));
        let err = translate_cs(&code, 1, root_table(), x16_used()).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor's address or format read as data",
                    ..
                }
            ),
            "reg s{reg}: {err:?}"
        );
    }
}

#[test]
fn merged_cube_and_2d_slices_sample_and_read_dimensions() {
    let mut w = vec![s_mov_b32(4, 0), s_mov_b32(5, 0x80)];
    // Merged 16-dword load: slot 0 (`s[8:15]`) is a 4x4 Cube, slot 1 (`s[16:23]`) is a 6x3 2D SRV.
    w.extend_from_slice(&s_load_dwordx16(8, 4, 0x0));
    w.extend_from_slice(&s_load_dwordx4(24, 4, 0x40)); // UAV at offset 16
    for s in 28..=31 {
        w.push(s_mov_b32(s, 0x80)); // point-wrap sampler
    }
    w.push(vop2(0x1A, 16, inline_u32(2), 0)); // v16 = v0 << 2

    // Cube sample coords `v[3:5] = (1.5, 1.5, float(v0 & 3))`: `image_sample_lz` (op 0x27, dim = 3).
    w.extend_from_slice(&[vop1(0x01, 3, 0xFF), 0x3FC0_0000]);
    w.extend_from_slice(&[vop1(0x01, 4, 0xFF), 0x3FC0_0000]);
    w.push(vop2(0x1B, 5, inline_u32(3), 0));
    w.push(vop1(0x06, 5, vgpr(5))); // v_cvt_f32_u32
    w.push(vop1(0x01, 6, inline_u32(0)));
    w.extend_from_slice(&mimg(0x27, 2, 3, 8, 28, 0x1, 3));

    // 2D sample coords `v[7:8] = (0.5, 0.5)`: `image_sample_lz` (op 0x27, dim = 1).
    w.push(vop1(0x01, 7, 0xF0));
    w.push(vop1(0x01, 8, 0xF0));
    w.extend_from_slice(&mimg(0x27, 9, 7, 16, 28, 0x1, 1));

    // Reconstruct scalar metadata from BOTH slices:
    // - 2D slice (`s[16:23]`): `width - 1 = 5`, `height - 1 = 2`;
    // - Cube slice (`s[8:15]`): `face_width - 1 = 3`, `TYPE = 11` (`SQ_RSRC_IMG_CUBE`).
    w.push(vop1(0x07, 2, vgpr(2))); // v_cvt_u32_f32
    w.push(vop1(0x07, 9, vgpr(9)));
    w.push(sop2(0x20, 32, 17, 128 + 30)); // s32 = s17 >> 30
    w.extend_from_slice(&[sop2(0x0E, 33, 18, 0xFF), 0x3FFF]); // s33 = s18 & 0x3fff
    w.push(sop2(0x1E, 33, 33, 128 + 2)); // s33 <<= 2
    w.push(sop2(0x10, 32, 32, 33)); // s32 = 2D width - 1 = 5
    w.extend_from_slice(&[sop2(0x27, 34, 18, 0xFF), (14 << 16) | 14]); // s34 = 2D height - 1 = 2
    w.push(sop2(0x20, 35, 9, 128 + 30));
    w.extend_from_slice(&[sop2(0x0E, 36, 10, 0xFF), 0x3FFF]);
    w.push(sop2(0x1E, 36, 36, 128 + 2));
    w.push(sop2(0x10, 35, 35, 36)); // s35 = cube face_width - 1 = 3
    w.push(sop2(0x20, 37, 11, 128 + 28)); // s37 = cube TYPE = 11

    w.extend_from_slice(&vop3(0x125, 2, vgpr(9), vgpr(2), 0));
    for s in [32, 34, 35, 37] {
        w.extend_from_slice(&vop3(0x125, 2, sgpr(s), vgpr(2), 0));
    }
    w.extend_from_slice(&mubuf(0x1C, 2, 16, 24, true, false, 0));
    w.push(S_ENDPGM);

    let used = vec![
        used_range(ResourceType::Srv, 0, 8, 0),
        used_range(ResourceType::Srv, 8, 8, 0),
        used_range(ResourceType::Uav, 16, 4, 0),
    ];
    let shader = translate_cs(&w, 32, root_table(), used).expect("cube + 2D translates");

    let mut cube_texels = Vec::with_capacity(6 * 16);
    for face in 0..6u32 {
        cube_texels.extend(std::iter::repeat_n(
            [(10 * (face + 1)) as f32, 0.0, 0.0, 1.0],
            16,
        ));
    }
    let tex2d = vec![[7.0f32, 0.0, 0.0, 1.0]; 6 * 3];
    let views = [
        View::Cube(4, &cube_texels),
        View::Texture(6, 3, &tex2d),
        View::RawUav(32),
    ];
    let want: Vec<u32> = (0..32)
        .map(|l| 10 * ((l & 3) + 1) + 7 + 5 + 2 + 3 + 11)
        .collect();
    for (label, got) in run_cs(&shader, &views) {
        assert_eq!(got, want, "{label}");
    }

    // Per-slice refusal isolation: an unreconstructable CBV slice at `s[8:11]`
    // in a merged `s_load_dwordx16` does not block scalar `NUM_RECORDS` reads
    // of a sibling structured SRV slice at `s[16:19]`, while reading `s10` of
    // the CBV slice itself still reports `"a constant buffer's descriptor read as data"`.
    let make_cbv_srv = |read_cbv_size: bool| -> Vec<u32> {
        let mut c = vec![s_mov_b32(4, 0), s_mov_b32(5, 0x80)];
        c.extend_from_slice(&s_load_dwordx16(8, 4, 0x0));
        c.extend_from_slice(&s_load_dwordx4(24, 4, 0x40));
        c.extend_from_slice(&mubuf(0x0C, 1, 0, 8, false, false, 0));
        c.extend_from_slice(&mubuf(0x0C, 2, 0, 16, false, true, 0));
        let size_reg = if read_cbv_size { 10 } else { 18 };
        c.extend_from_slice(&vop3(0x125, 1, vgpr(2), vgpr(1), 0));
        c.extend_from_slice(&vop3(0x125, 1, sgpr(size_reg), vgpr(1), 0));
        c.extend_from_slice(&mubuf(0x1C, 1, 0, 24, false, false, 0));
        c.push(S_ENDPGM);
        c
    };
    let cbv_srv_used = || {
        vec![
            used_range(ResourceType::Cbv, 0, 4, 0),
            used_range(ResourceType::Srv, 8, 4, 4),
            used_range(ResourceType::Uav, 16, 4, 0),
        ]
    };
    translate_cs(&make_cbv_srv(false), 1, root_table(), cbv_srv_used())
        .expect("SRV slice metadata lowers beside unread CBV slice");
    let cbv_err = translate_cs(&make_cbv_srv(true), 1, root_table(), cbv_srv_used()).unwrap_err();
    assert!(
        matches!(
            cbv_err,
            ShaderError::Unsupported {
                what: "a constant buffer's descriptor read as data",
                ..
            }
        ),
        "{cbv_err:?}"
    );
}
