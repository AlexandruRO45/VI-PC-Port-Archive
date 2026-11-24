//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if buffers {
        range(ResourceType::Srv, 0, 4, 4, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
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
    })
    .expect("lower");
    // One access with each lane's own heap index, no wave operation.
    assert_eq!(lowered.nonuniform_handles, 1);
    assert!(!lowered.wave_ops);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    for (label, got) in run(&shader, &views) {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    assert!(!results.is_empty());
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        assert!(!results.is_empty());
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}

/// Builds a wave32 compute shader with `s0` = `ResourceHeap`, `s1` = descriptor
/// table (holding a raw SRV at +0 and a raw UAV at +8 dwords, i.e. heap slots
/// `2` and `3` when table offset is `2 * 32 = 64`), validates it with
/// `dxil.dll`, and returns the `HostShader`.
fn build_heap_cbv_test(words: &[u32], threads: u32) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: RootType::ResourceHeap,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 1,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 1,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 1,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1, // s0, s1
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
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
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    shader
}

fn run_heap_cbv_test(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Table in s1 starts at heap index 2 (byte offset 64).
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if u.root == RootType::Table && u.slot == 1 {
                ud[u.index as usize] = 2 * 32;
            }
        }
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
        out.push((label, words[0].clone()));
    }
    out
}

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_and_saveexec_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x3C << 8 | src0 as u32
}

const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

const fn s_cmp_lg_u32(src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (0x07 << 16) | (src1 as u32) << 8 | src0 as u32
}

const fn s_cbranch_scc0(rel16: i16) -> u32 {
    0xBF84_0000 | (rel16 as u16 as u32)
}

const fn s_branch(rel16: i16) -> u32 {
    0xBF82_0000 | (rel16 as u16 as u32)
}

const fn s_load_dwordx4(dst: u8, base: u8, soffset: u8, offset20: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | ((base as u32) >> 1),
        (soffset as u32) << 25 | (offset20 & 0xF_FFFF),
    ]
}

const fn s_buffer_load_dword(dst: u8, sbase4: u8, offset20: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x08 << 18 | (dst as u32) << 6 | ((sbase4 as u32) >> 1),
        (125u32 << 25) | (offset20 & 0xF_FFFF),
    ]
}

const fn buffer_load_dword(vdata: u8, vaddr: u8, srsrc: u8, offen: bool, offset12: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x0C << 18 | if offen { 1 << 12 } else { 0 } | (offset12 as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn buffer_store_dword(vdata: u8, vaddr: u8, srsrc: u8, offen: bool, offset12: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | if offen { 1 << 12 } else { 0 } | (offset12 as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (0x01 << 9) | (src0 as u32 & 0x1FF)
}

const fn v_readfirstlane_b32(sdst: u8, vsrc: u8) -> u32 {
    0x7E00_0000 | (sdst as u32) << 17 | (0x02 << 9) | (256 + vsrc as u32)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// Reproduces #1595: a compute shader loads a heap index from a raw buffer,
/// compares it with 0, speculatively loads a heap CBV before `s_cbranch_scc0`,
/// and only uses the loaded dword on the non-zero branch. With a null Texture1D
/// SRV at heap slot 0, skipping the speculative CBV load avoids a CBV-over-SRV
/// mismatch and WARP device removal (`0x887A0005`).
#[test]
fn speculative_heap_cbv_before_branch_skips_null_srv_at_slot_0() {
    // s[2:3] = {s0 (ResourceHeap), 0}; s[4:5] = {s1 (Table), 0}.
    let mut w = vec![
        s_mov_b32(2, 0),
        s_mov_b32(3, 0x80),
        s_mov_b32(4, 1),
        s_mov_b32(5, 0x80),
    ];
    // Load table SRV (slot 2) into s[8:11] and UAV (slot 3) into s[12:15].
    w.extend_from_slice(&s_load_dwordx4(8, 4, 125, 0));
    w.extend_from_slice(&s_load_dwordx4(12, 4, 125, 32));
    // Load dynamic heap index from raw buffer SRV into v1 -> s6.
    w.push(v_mov_b32(2, 0x80)); // v2 = 0
    w.extend_from_slice(&buffer_load_dword(1, 2, 8, false, 0));
    w.push(v_readfirstlane_b32(6, 1));
    // Compare idx != 0, then speculatively load ResourceHeap[idx] CBV BEFORE branching!
    w.push(s_cmp_lg_u32(6, 0x80)); // SCC = (s6 != 0)
    w.push(sop2(0x1E, 7, 6, 0x85)); // s7 = s6 << 5 (idx * 32)
    w.extend_from_slice(&s_load_dwordx4(16, 2, 7, 0));
    w.extend_from_slice(&s_buffer_load_dword(20, 16, 0)); // speculative CBV load!
    w.push(s_cbranch_scc0(2)); // if idx == 0 -> skip 2 dwords (.zero)
    // .nonzero (2 dwords): v1 = s20; s_branch .store
    w.push(v_mov_b32(1, 20));
    w.push(s_branch(2)); // skip 2 dwords (.zero)
    // .zero (2 dwords): v1 = 0x7777_7777 (literal)
    w.push(v_mov_b32(1, 0xFF));
    w.push(0x7777_7777);
    // .store: store v1 to UAV[0]
    w.extend_from_slice(&buffer_store_dword(1, 2, 12, false, 0));
    w.push(0xBF81_0000); // s_endpgm

    let shader = build_heap_cbv_test(&w, 1);
    let cbv = [0xCAFE_BABEu32, 0x1122_3344, 0, 0];
    for (idx, want) in [(0u32, 0x7777_7777u32), (1, 0xCAFE_BABE)] {
        let idx_buf = [idx];
        let views = [
            View::NullTexture1dSrv,
            View::Cbv(&cbv),
            View::Buffer {
                data: &idx_buf,
                stride: Stride::Raw,
            },
            View::RawUav(1),
        ];
        let results = run_heap_cbv_test(&shader, &views);
        assert!(!results.is_empty());
        for (label, got) in results {
            assert_eq!(got, vec![want], "{label}, idx={idx}");
        }
    }
}

/// Speculative heap CBV load followed by pure ALU (`s_add_u32`), a *late*
/// compare (`s_cmp_lg_u32` scheduled after the load), and `s_cselect_b32`.
#[test]
fn speculative_heap_cbv_before_late_cmp_and_cselect_skips_null_srv_at_slot_0() {
    let mut w = vec![
        s_mov_b32(2, 0),
        s_mov_b32(3, 0x80),
        s_mov_b32(4, 1),
        s_mov_b32(5, 0x80),
    ];
    w.extend_from_slice(&s_load_dwordx4(8, 4, 125, 0));
    w.extend_from_slice(&s_load_dwordx4(12, 4, 125, 32));
    w.push(v_mov_b32(2, 0x80));
    w.extend_from_slice(&buffer_load_dword(1, 2, 8, false, 0));
    w.push(v_readfirstlane_b32(6, 1));
    // Load CBV *before* comparing s6 != 0, then transform via pure s_add_u32!
    w.push(sop2(0x1E, 7, 6, 0x85)); // s7 = s6 << 5
    w.extend_from_slice(&s_load_dwordx4(16, 2, 7, 0));
    w.extend_from_slice(&s_buffer_load_dword(20, 16, 0));
    w.push(sop2(0x00, 21, 20, 0x8A)); // s21 = s20 + 10
    w.push(s_cmp_lg_u32(6, 0x80)); // late compare AFTER load!
    w.push(sop2(0x0A, 22, 21, 0xFF)); // s_cselect_b32 s22, s21, 0x5555_5555
    w.push(0x5555_5555);
    w.push(v_mov_b32(1, 22));
    w.extend_from_slice(&buffer_store_dword(1, 2, 12, false, 0));
    w.push(0xBF81_0000);

    let shader = build_heap_cbv_test(&w, 1);
    let cbv = [0xCAFE_BABEu32, 0, 0, 0];
    for (idx, want) in [(0u32, 0x5555_5555u32), (1, 0xCAFE_BABE + 10)] {
        let idx_buf = [idx];
        let views = [
            View::NullTexture1dSrv,
            View::Cbv(&cbv),
            View::Buffer {
                data: &idx_buf,
                stride: Stride::Raw,
            },
            View::RawUav(1),
        ];
        let results = run_heap_cbv_test(&shader, &views);
        assert!(!results.is_empty());
        for (label, got) in results {
            assert_eq!(got, vec![want], "{label}, idx={idx}");
        }
    }
}

/// Soundness control: when heap slot 0 holds a *real* CBV and the loaded value
/// is still live when `idx == 0` (even though `idx != 0` is computed for a
/// separate select), the load must NOT be skipped.
#[test]
fn live_heap_cbv_at_slot_0_is_not_skipped_even_when_index_compared_to_zero() {
    let mut w = vec![
        s_mov_b32(2, 0),
        s_mov_b32(3, 0x80),
        s_mov_b32(4, 1),
        s_mov_b32(5, 0x80),
    ];
    w.extend_from_slice(&s_load_dwordx4(8, 4, 125, 0));
    w.extend_from_slice(&s_load_dwordx4(12, 4, 125, 32));
    w.push(v_mov_b32(2, 0x80));
    w.extend_from_slice(&buffer_load_dword(1, 2, 8, false, 0));
    w.push(v_readfirstlane_b32(6, 1));
    w.push(s_cmp_lg_u32(6, 0x80)); // SCC = (s6 != 0)
    w.push(sop2(0x1E, 7, 6, 0x85));
    w.extend_from_slice(&s_load_dwordx4(16, 2, 7, 0));
    w.extend_from_slice(&s_buffer_load_dword(20, 16, 0)); // live for both idx==0 and idx==1!
    w.push(sop2(0x0A, 21, 0x81, 0x80)); // s21 = (s6 != 0) ? 1 : 0
    w.push(sop2(0x00, 22, 20, 21)); // s22 = s20 + s21
    w.push(v_mov_b32(1, 22));
    w.extend_from_slice(&buffer_store_dword(1, 2, 12, false, 0));
    w.push(0xBF81_0000);

    let shader = build_heap_cbv_test(&w, 1);
    let cbv0 = [0x4242_4242u32, 0, 0, 0];
    let cbv1 = [0xCAFE_BABEu32, 0, 0, 0];
    for (idx, want) in [(0u32, 0x4242_4242u32), (1, 0xCAFE_BABE + 1)] {
        let idx_buf = [idx];
        let views = [
            View::Cbv(&cbv0),
            View::Cbv(&cbv1),
            View::Buffer {
                data: &idx_buf,
                stride: Stride::Raw,
            },
            View::RawUav(1),
        ];
        let results = run_heap_cbv_test(&shader, &views);
        assert!(!results.is_empty());
        for (label, got) in results {
            assert_eq!(got, vec![want], "{label}, idx={idx}");
        }
    }
}

/// Divergent per-lane `v_cndmask_b32` with a waterfall heap CBV read: even
/// lanes carry `idx = 0` (null Texture1D SRV at slot 0), odd lanes carry
/// `idx = 1` (valid CBV at slot 1). Guarding the load with `v_cmp_ne_u32`
/// prevents even lanes from executing `createHandleFromHeap(0)` as a CBV.
#[test]
fn divergent_cndmask_guards_per_lane_heap_cbv_from_null_srv_at_slot_0() {
    // s[2:3] = {s0 (ResourceHeap), 0}; s[4:5] = {s1 (Table), 0}.
    let mut w = vec![
        s_mov_b32(2, 0),
        s_mov_b32(3, 0x80),
        s_mov_b32(4, 1),
        s_mov_b32(5, 0x80),
    ];
    w.extend_from_slice(&s_load_dwordx4(12, 4, 125, 32)); // UAV at slot 3
    // v1 = v0 & 1 (even lanes = 0, odd lanes = 1); v2 = v1 << 5 (byte offset).
    w.push(vop2(0x1B, 1, 0x81, 0)); // v_and_b32 v1, 1, v0
    w.push(vop2(0x1A, 2, 0x85, 1)); // v_lshlrev_b32 v2, 5, v1
    // v_cmp_ne_u32_e64 s24, 0, v1  (mask of lanes with idx != 0)
    w.extend_from_slice(&vop3(0x0C5, 24, 0x80, 256 + 1, 0));
    // Console waterfall loop over per-lane heap offset v2 -> s_load_dwordx4 + s_buffer_load_dword.
    w.push(s_mov_b32(20, 0x7E)); // s20 = exec_lo
    // L (offset here):
    let loop_start = w.len();
    w.push(v_readfirstlane_b32(7, 2)); // s7 = firstlane(v2)
    w.extend_from_slice(&vop3(0x0C2, 21, 7, 256 + 2, 0)); // v_cmp_eq_u32_e64 s21, s7, v2
    w.push(s_and_saveexec_b32(21, 21));
    w.extend_from_slice(&s_load_dwordx4(16, 2, 7, 0));
    w.extend_from_slice(&s_buffer_load_dword(22, 16, 0));
    w.push(v_mov_b32(3, 22));
    w.push(sop2(0x12, 0x7E, 0x7E, 21)); // s_xor_b32 exec_lo, exec_lo, s21
    let rel = (loop_start as i32 - (w.len() as i32 + 1)) as i16;
    w.push(0xBF89_0000 | (rel as u16 as u32)); // s_cbranch_execnz L
    w.push(s_mov_b32(0x7E, 20)); // exec_lo = s20
    // v_cndmask_b32_e64 v4, 0x3333_3333 (via v5), v3, s24
    w.push(v_mov_b32(5, 0xFF));
    w.push(0x3333_3333);
    w.extend_from_slice(&vop3(0x101, 4, 256 + 5, 256 + 3, 24));
    // Store v4 to UAV[lane * 4].
    w.push(vop2(0x1A, 6, 0x82, 0)); // v6 = v0 << 2
    w.extend_from_slice(&buffer_store_dword(4, 6, 12, true, 0));
    w.push(0xBF81_0000);

    let shader = build_heap_cbv_test(&w, 32);
    let cbv = [0xCAFE_BABEu32, 0, 0, 0];
    let dummy_srv = [0u32];
    let views = [
        View::NullTexture1dSrv,
        View::Cbv(&cbv),
        View::Buffer {
            data: &dummy_srv,
            stride: Stride::Raw,
        },
        View::RawUav(32),
    ];
    let results = run_heap_cbv_test(&shader, &views);
    assert!(!results.is_empty());
    let want: Vec<u32> = (0..32)
        .map(|l| if l & 1 == 0 { 0x3333_3333 } else { 0xCAFE_BABE })
        .collect();
    for (label, got) in results {
        assert_eq!(got, want, "{label}");
    }
}

/// Straight-line EXEC-guarded `buffer_load_dword` (no `s_cbranch_execz`):
/// inactive even lanes hold `idx = 0` (null Texture1D SRV at slot 0), while
/// active odd lanes hold `idx = 1` (raw buffer SRV at slot 1). Branching
/// around handle creation and load prevents inactive lanes from reading the
/// Texture1D SRV through a raw buffer handle on WARP/GBV.
#[test]
fn exec_guarded_buffer_load_skips_inactive_lanes_on_null_srv_at_slot_0() {
    let mut w = vec![
        s_mov_b32(2, 0),
        s_mov_b32(3, 0x80),
        s_mov_b32(4, 1),
        s_mov_b32(5, 0x80),
    ];
    w.extend_from_slice(&s_load_dwordx4(12, 4, 125, 32)); // UAV at slot 3
    w.push(v_mov_b32(3, 0xFF)); // v3 = 0x2222_2222 (preserved on inactive lanes)
    w.push(0x2222_2222);
    w.push(v_mov_b32(4, 0x80)); // v4 = 0 (byte offset)
    w.push(vop2(0x1B, 1, 0x81, 0)); // v1 = v0 & 1
    w.push(vop2(0x1A, 2, 0x85, 1)); // v2 = v1 << 5
    // Narrow EXEC without branching (straight-line guard).
    w.extend_from_slice(&vop3(0x0C5, 24, 0x80, 256 + 1, 0)); // v_cmp_ne_u32 s24, 0, v1
    w.push(s_and_saveexec_b32(25, 24));
    // Console waterfall over v2, loading raw buffer dword into v3 under EXEC guard.
    w.push(s_mov_b32(20, 0x7E));
    let loop_start = w.len();
    w.push(v_readfirstlane_b32(7, 2));
    w.extend_from_slice(&vop3(0x0C2, 21, 7, 256 + 2, 0));
    w.push(s_and_saveexec_b32(21, 21));
    w.extend_from_slice(&s_load_dwordx4(16, 2, 7, 0));
    w.extend_from_slice(&buffer_load_dword(3, 4, 16, false, 0));
    w.push(sop2(0x12, 0x7E, 0x7E, 21));
    let rel = (loop_start as i32 - (w.len() as i32 + 1)) as i16;
    w.push(0xBF89_0000 | (rel as u16 as u32));
    w.push(s_mov_b32(0x7E, 20));
    // Restore EXEC and store v3 to UAV[lane * 4].
    w.push(s_mov_b32(0x7E, 25));
    w.push(vop2(0x1A, 6, 0x82, 0));
    w.extend_from_slice(&buffer_store_dword(3, 6, 12, true, 0));
    w.push(0xBF81_0000);

    let shader = build_heap_cbv_test(&w, 32);
    let raw_buf = [0xDEAD_1234u32];
    let dummy_srv = [0u32];
    let views = [
        View::NullTexture1dSrv,
        View::Buffer {
            data: &raw_buf,
            stride: Stride::Raw,
        },
        View::Buffer {
            data: &dummy_srv,
            stride: Stride::Raw,
        },
        View::RawUav(32),
    ];
    let results = run_heap_cbv_test(&shader, &views);
    assert!(!results.is_empty());
    let want: Vec<u32> = (0..32)
        .map(|l| if l & 1 == 0 { 0x2222_2222 } else { 0xDEAD_1234 })
        .collect();
    for (label, got) in results {
        assert_eq!(got, want, "{label}");
    }
}
