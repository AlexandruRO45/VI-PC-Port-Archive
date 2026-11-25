//! End-to-end tests of descriptors read as data: synthetic GFX10 programs
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU and WARP (see `waterfall_tests::run`).

use super::host::{Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a wave32 compute program of 32 threads with a table at s0: a
/// buffer of dwords at entry 0 and a raw UAV at entry 2.
fn build(words: &[u32]) -> HostShader {
    let range = |kind, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 4),
            range(ResourceType::Uav, 16, 0),
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
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// A buffer's last dword, read at the offset its descriptor's record count
/// gives, as the title reads the last records of some buffers. The record
/// count comes from the host view's size (`getDimensions`), and the read at
/// an offset made from it is the shape AMD's compiler (driver
/// 32.0.32029.2010, RDNA 4) crashed on while both went through one handle:
/// this test's process died creating the pipeline.
#[test]
fn every_lane_reads_the_last_dword_its_buffers_record_count_names() {
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0040, // s_load_dwordx4 s[12:15], s[4:5], 0x40
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x8110_C10A, // s_add_i32 s16, s10, -1
        0x8F10_8210, // s_lshl_b32 s16, s16, 2
        0x7E02_0210, // v_mov_b32 v1, s16
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8003_0203, // buffer_store_dword v2, v3, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let shader = build(&words);
    let data: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(32),
    ];
    for (label, got) in super::waterfall_tests::run(&shader, &views) {
        // 32 records of 4 bytes: every lane read the last, 131.
        assert_eq!(got, vec![131; 32], "{label}");
    }
}
