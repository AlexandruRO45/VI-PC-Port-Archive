//! Group barriers where lanes of a wave part (`lower/barriers.rs`): the
//! console runs `s_barrier` per wave, the host's barrier counts threads.
//! Synthetic GFX10 programs, one group of 64 threads in two waves of 32,
//! lowered, validated by `dxil.dll` when present, and run on the host GPU
//! and WARP. Each must complete (a group whose threads meet different
//! numbers of barriers hangs the GPU) and store what the console would:
//!
//! - a per-lane branch with a barrier on one side and one after the join
//!   (the lighter side gets as many barriers as it lacks);
//! - a loop holding a barrier, left per lane (its exit decided for the
//!   wave, as the console decides it);
//! - a per-lane side meeting barriers while the other meets none to the end
//!   (it exits; a host barrier does not wait for exited threads, which
//!   D3D12 leaves undefined: hence this test on both adapters);
//! - the same with a barrier after the join, whose counts cannot be
//!   balanced (console mode: branches decided for the wave, vector
//!   instructions guarded by EXEC).

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Threads in the group: two waves of 32.
const THREADS: u32 = 64;

/// A program's words, with labels for branches.
#[derive(Default)]
struct Asm {
    words: Vec<u32>,
    /// (word of the branch, SOPP opcode, label).
    fixups: Vec<(usize, u32, &'static str)>,
    labels: Vec<(&'static str, usize)>,
}

impl Asm {
    fn put(&mut self, words: &[u32]) -> &mut Self {
        self.words.extend_from_slice(words);
        self
    }

    fn label(&mut self, name: &'static str) -> &mut Self {
        self.labels.push((name, self.words.len()));
        self
    }

    /// A SOPP branch (`op`) to `label`.
    fn branch(&mut self, op: u32, label: &'static str) -> &mut Self {
        self.fixups.push((self.words.len(), op, label));
        self.words.push(0);
        self
    }

    fn finish(&mut self) -> Vec<u32> {
        for &(at, op, label) in &self.fixups {
            let to = self.labels.iter().find(|l| l.0 == label).expect("label").1;
            // In dwords from the instruction after the branch.
            let rel = to as i64 - (at as i64 + 1);
            self.words[at] = 0xBF80_0000 | (op << 16) | (rel as i16 as u16 as u32);
        }
        std::mem::take(&mut self.words)
    }
}

const S_CBRANCH_SCC1: u32 = 0x05;
const S_CBRANCH_EXECZ: u32 = 0x08;
const S_CBRANCH_EXECNZ: u32 = 0x09;

/// The UAV (table entry 0, at s0) into s[40:43], v20 = thread × 4, v1 = 1.
const PROLOGUE: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
    0x3428_0082, // v_lshlrev_b32 v20, 2, v0
    0x7E02_0281, // v_mov_b32 v1, 1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

/// v1 stored at v20, then the end.
const EPILOGUE: &[u32] = &[
    0xE070_1000,
    0x800A_0114, // buffer_store_dword v1, v20, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

const S_BARRIER: u32 = 0xBF8A_0000;
/// v_cmp_gt_u32 vcc_lo, 16, v0: threads 0-15 of the group.
const LANES_BELOW_16: u32 = 0x7D88_0090;
/// s_and_saveexec_b32 s6, vcc_lo
const SAVE_EXEC_VCC: u32 = 0xBE86_3C6A;
/// s_or_b32 exec_lo, exec_lo, s6
const RESTORE_EXEC: u32 = 0x887E_067E;
/// v_mov_b32 v1, 2
const V1_IS_2: u32 = 0x7E02_0282;
/// s_cmp_eq_u32 s0, 0: never true (s0 holds the table's address), but not
/// known when lowering: a uniform choice.
const S0_IS_0: u32 = 0xBF06_8000;

/// Threads 0-15 meet a barrier the others skip, and every thread one more
/// after the join: v1 = 2 for threads 0-15, else 1.
fn per_lane_branch() -> Vec<u32> {
    let mut a = Asm::default();
    a.put(PROLOGUE)
        .put(&[LANES_BELOW_16, SAVE_EXEC_VCC])
        .branch(S_CBRANCH_EXECZ, "join")
        .put(&[S_BARRIER, V1_IS_2])
        .label("join")
        .put(&[RESTORE_EXEC, S_BARRIER])
        .put(EPILOGUE);
    a.finish()
}

/// Each thread goes round (thread & 3) + 1 times, with a barrier in every
/// round, counting rounds in v1 (EXEC narrowed as threads finish).
fn loop_left_per_lane() -> Vec<u32> {
    let mut a = Asm::default();
    a.put(PROLOGUE)
        .put(&[
            0x3604_0083, // v_and_b32 v2, 3, v0
            0x4A04_0481, // v_add_nc_u32 v2, 1, v2
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBE87_037E, // s_mov_b32 s7, exec_lo
        ])
        .label("round")
        .put(&[
            S_BARRIER,
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
            0x7D82_0501, // v_cmp_lt_u32 vcc_lo, v1, v2
            0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
        ])
        .branch(S_CBRANCH_EXECNZ, "round")
        .put(&[0xBEFE_0307]) // s_mov_b32 exec_lo, s7
        .put(EPILOGUE);
    a.finish()
}

/// Threads 0-15 meet one or two barriers (a uniform choice), the others none
/// to the end: v1 = 2 for threads 0-15, else 1. With `after`, every thread
/// meets one more after the join, so the sides' counts ({1, 2} against
/// {0}, then {2, 3} against {1}) differ by no one number.
fn uneven_sides(after: bool) -> Vec<u32> {
    let mut a = Asm::default();
    a.put(PROLOGUE)
        .put(&[LANES_BELOW_16, SAVE_EXEC_VCC])
        .branch(S_CBRANCH_EXECZ, "join")
        .put(&[S0_IS_0])
        .branch(S_CBRANCH_SCC1, "one")
        .put(&[S_BARRIER])
        .label("one")
        .put(&[S_BARRIER, V1_IS_2])
        .label("join")
        .put(&[RESTORE_EXEC]);
    if after {
        a.put(&[S_BARRIER]);
    }
    a.put(EPILOGUE);
    a.finish()
}

fn build(words: &[u32]) -> HostShader {
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
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [THREADS, 1, 1],
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
        view_marks: false,
    })
    .expect("lower");
    HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: Vec::new(),
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
    }
}

/// Lowers, validates and runs `words` on the hardware adapter and WARP;
/// each must complete and store `want(thread)` per thread.
fn runs_as_the_console(name: &str, words: &[u32], want: impl Fn(u32) -> u32) {
    let shader = build(words);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
    }
    let views = [View::RawUav(THREADS)];
    let want: Vec<u32> = (0..THREADS).map(want).collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{name}, {label}: dispatch failed (a hang?): {e}"));
        assert_eq!(out[0][..THREADS as usize], want[..], "{name}, {label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// The programs decode to the instructions they are meant to be.
#[test]
fn barrier_programs_decode_as_written() {
    let names = |words: Vec<u32>| -> Vec<&'static str> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .expect("decode")
            .iter()
            .map(|i| i.name().unwrap_or("?"))
            .collect()
    };
    let prologue = [
        "s_mov_b32",
        "s_mov_b32",
        "s_load_dwordx4",
        "v_lshlrev_b32",
        "v_mov_b32",
        "s_waitcnt",
    ];
    let epilogue = ["buffer_store_dword", "s_endpgm"];
    let with = |body: &[&'static str]| -> Vec<&'static str> {
        prologue.iter().chain(body).chain(&epilogue).copied().collect()
    };
    assert_eq!(
        names(per_lane_branch()),
        with(&[
            "v_cmp_gt_u32",
            "s_and_saveexec_b32",
            "s_cbranch_execz",
            "s_barrier",
            "v_mov_b32",
            "s_or_b32",
            "s_barrier",
        ])
    );
    assert_eq!(
        names(loop_left_per_lane()),
        with(&[
            "v_and_b32",
            "v_add_nc_u32",
            "v_mov_b32",
            "s_mov_b32",
            "s_barrier",
            "v_add_nc_u32",
            "v_cmp_lt_u32",
            "s_and_b32",
            "s_cbranch_execnz",
            "s_mov_b32",
        ])
    );
    let uneven = [
        "v_cmp_gt_u32",
        "s_and_saveexec_b32",
        "s_cbranch_execz",
        "s_cmp_eq_u32",
        "s_cbranch_scc1",
        "s_barrier",
        "s_barrier",
        "v_mov_b32",
        "s_or_b32",
    ];
    assert_eq!(names(uneven_sides(false)), with(&uneven));
    let after: Vec<&'static str> = uneven.iter().copied().chain(["s_barrier"]).collect();
    assert_eq!(names(uneven_sides(true)), with(&after));
}

#[test]
fn a_barrier_on_one_side_of_a_lane_branch_is_balanced() {
    runs_as_the_console("per-lane branch", &per_lane_branch(), |t| if t < 16 { 2 } else { 1 });
}

#[test]
fn a_loop_with_a_barrier_left_per_lane_ends_for_the_wave() {
    runs_as_the_console("loop left per lane", &loop_left_per_lane(), |t| (t & 3) + 1);
}

#[test]
fn a_side_meeting_no_more_barriers_exits() {
    runs_as_the_console("uneven sides, none after", &uneven_sides(false), |t| if t < 16 { 2 } else { 1 });
}

#[test]
fn barriers_that_cannot_be_balanced_run_as_the_console_runs_them() {
    runs_as_the_console("uneven sides, one after", &uneven_sides(true), |t| if t < 16 { 2 } else { 1 });
}
