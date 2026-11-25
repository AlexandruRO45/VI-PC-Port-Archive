//! Branches no lane takes one way of, left out before lowering
//! (`emit_without_dead_branches`): the programs below, emitted with and
//! without, give the same results on the default adapter (skipped without
//! one), and what is left as it is stays as it was.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, Target};
use crate::native::{ComputeIo, RootParam, RootType, StageIo};

const ALL_WAVES: Target = Target {
    lanes_min: 32,
    lanes_max: 64,
};

/// A wave32 group of 32 threads: `body` between a prologue that loads the
/// UAV's descriptor into s[4:7] and sets v3 to the lane's byte offset and
/// an epilogue that stores v2 there.
fn program(body: &[u32]) -> crate::NativeProgram {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1, // one user SGPR (the UAV table)
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The program's earlier stages.
struct Stages {
    p: crate::ir::Program,
    st: crate::structure::Structured,
    io: crate::io::Io,
    res: crate::resource::Resources,
}

fn stages(native: &crate::NativeProgram) -> Stages {
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    Stages {
        res: crate::resource::track(&p, &native.layout, native.stage).unwrap(),
        io: crate::io::raise(&p, native).unwrap(),
        st: crate::structure::structure(&p).unwrap(),
        p,
    }
}

const CONTAINER: crate::Container = crate::Container {
    digest: [0; 16],
    version: (1, 0),
    parts: vec![],
};

/// The program as [`super::emit`] emits it, and as it is (no branch left
/// out).
fn both(native: &crate::NativeProgram, target: Target) -> (HostShader, HostShader) {
    let s = stages(native);
    let pruned = super::emit(&CONTAINER, native, &s.p, &s.st, &s.io, &s.res, target).unwrap();
    let plain =
        super::emit_threaded(&CONTAINER, native, &s.p, &s.st, &s.io, &s.res, target, true).unwrap();
    (pruned, plain)
}

/// What each lane stored.
fn run(host: &Host, cs: &HostShader) -> Vec<u32> {
    let (root, pso) = host.compute_pipeline(cs).unwrap();
    let data = vec![0; (cs.user_data_dwords() as usize).max(1)];
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(data))],
            views: &[View::RawUav(32)],
            samplers: &[],
            groups: [1, 1, 1],
        },
    )
    .unwrap()
    .remove(0)
}

fn validate(cs: &HostShader) {
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil.clone()).unwrap();
    }
}

/// s8 holds EXEC, so `vcc = exec & s8` is EXEC: `s_cbranch_vccz` is taken
/// by no lane, and every lane stores its id + 1.
const ONE_WAY: [u32; 6] = [
    0xBE88_037E, // s_mov_b32 s8, exec_lo
    0x876A_087E, // s_and_b32 vcc_lo, exec_lo, s8
    0xBF86_0002, // s_cbranch_vccz dead
    0x4A04_0081, // v_add_nc_u32 v2, 1, v0
    0xBF82_0001, // s_branch join
    0x4A04_0082, // dead: v_add_nc_u32 v2, 2, v0
                 // join:
];

#[test]
fn a_branch_no_lane_takes_one_way_of_is_left_out() {
    let native = program(&ONE_WAY);
    assert_eq!(stages(&native).st.one_way().len(), 1);
    let (pruned, plain) = both(&native, ALL_WAVES);
    assert!(
        pruned.dxil.len() < plain.dxil.len(),
        "{} bytes pruned, {} as it is",
        pruned.dxil.len(),
        plain.dxil.len()
    );
    validate(&pruned);
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    let want: Vec<u32> = (1..=32).collect();
    assert_eq!(run(&host, &plain), want);
    assert_eq!(run(&host, &pruned), want);
}

/// A loop no lane leaves, whose latch is a branch every lane takes back:
/// lowered with its exit, through which a lane leaves at the iteration
/// limit, and not as a jump (it would leave the program there and store
/// nothing). The branch before it is still left out.
#[test]
fn a_loop_stopped_at_the_iteration_limit_leaves_through_its_exit() {
    let mut body = ONE_WAY.to_vec();
    body.extend_from_slice(&[
        0x4A04_0481, // loop: v_add_nc_u32 v2, 1, v2
        0x876A_087E, // s_and_b32 vcc_lo, exec_lo, s8
        0xBF87_FFFD, // s_cbranch_vccnz loop
    ]);
    let native = program(&body);
    let s = stages(&native);
    let latch =
        s.p.blocks
            .iter()
            .position(|b| {
                b.insts
                    .last()
                    .is_some_and(|i| i.name() == "s_cbranch_vccnz")
            })
            .unwrap();
    // Both branches are one way; only the first is left out.
    assert_eq!(s.st.one_way().len(), 2);
    assert!(s.st.one_way().iter().any(|&(b, _)| b == latch));
    let (pruned, plain) = both(&native, ALL_WAVES);
    assert!(pruned.dxil != plain.dxil, "not pruned");
    validate(&pruned);
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    let (a, b) = (run(&host, &plain), run(&host, &pruned));
    assert_eq!(a, b);
    // Every lane counted the same number of iterations from its id + 1.
    let n = a[0] - 1;
    assert!(n > 0);
    assert!((0..32).all(|t| a[t as usize] == t + 1 + n), "{a:?}");
}

/// A program that asks for the console flow (a lane read, and lanes that
/// branch differently) only through the code behind a branch no lane takes
/// would not ask for it pruned: it is emitted as it is, in the console
/// flow, which runs that code.
#[test]
fn a_program_asking_for_the_console_flow_is_emitted_as_it_is() {
    let words = [
        0xBE88_037E, // s_mov_b32 s8, exec_lo
        0x876A_087E, // s_and_b32 vcc_lo, exec_lo, s8
        0xBF86_0002, // s_cbranch_vccz dead
        0x4A04_0081, // v_add_nc_u32 v2, 1, v0
        0xBF82_0003, // s_branch join
        0xD760_0009,
        0x0001_0B00, // dead: v_readlane_b32 s9, v0, 5
        0x7E04_0209, // v_mov_b32 v2, s9
        0x7D8A_0080, // join: v_cmp_ne_u32 vcc, 0, v0
        0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
        0xBF88_0001, // s_cbranch_execz end
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x887E_0A7E, // end: s_or_b32 exec_lo, exec_lo, s10
    ];
    let native = program(&words);
    let s = stages(&native);
    let jumps = s.st.one_way();
    assert_eq!(jumps.len(), 1);
    let parts = super::lower_parts(&CONTAINER, &native, &s.p, &s.st, &s.io, &s.res, ALL_WAVES);
    assert!(super::lower::wants_console_flow(&parts.unwrap()));
    // Pruned, it would not.
    let mut q = crate::ir::retarget(&s.p, &jumps).unwrap();
    crate::sgpr_spill::forward(&mut q);
    let st = crate::structure::structure(&q).unwrap();
    let io = crate::io::raise(&q, &native).unwrap();
    let res = crate::resource::track(&q, &native.layout, native.stage).unwrap();
    let parts = super::lower_parts(&CONTAINER, &native, &q, &st, &io, &res, ALL_WAVES);
    assert!(!super::lower::wants_console_flow(&parts.unwrap()));
    let (emitted, plain) = both(&native, ALL_WAVES);
    assert!(emitted.dxil == plain.dxil, "pruned");
}

/// The other way: a program that asks for the console flow only pruned.
/// A lane read where lanes branch differently asks for it, not in a
/// program with a permutation, and here the permutation is behind a branch
/// no lane takes. Pruned, the
/// program would ask for the console flow, which [`super::emit`] does not
/// give a pruned program: it is emitted as it is, per lane.
#[test]
fn a_program_asking_for_the_console_flow_only_pruned_is_emitted_as_it_is() {
    let words = [
        0xBE88_037E, // s_mov_b32 s8, exec_lo
        0x876A_087E, // s_and_b32 vcc_lo, exec_lo, s8
        0xBF86_0004, // s_cbranch_vccz dead
        0xD760_0009,
        0x0001_0B00, // v_readlane_b32 s9, v0, 5
        0x7E04_0209, // v_mov_b32 v2, s9
        0xBF82_0003, // s_branch join
        0xDACC_0000,
        0x0200_0003, // dead: ds_bpermute_b32 v2, v3, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7D8A_0080, // join: v_cmp_ne_u32 vcc, 0, v0
        0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
        0xBF88_0001, // s_cbranch_execz end
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x887E_0A7E, // end: s_or_b32 exec_lo, exec_lo, s10
    ];
    let native = program(&words);
    let s = stages(&native);
    assert!(
        s.p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "ds_bpermute_b32")
    );
    let jumps = s.st.one_way();
    assert_eq!(jumps.len(), 1);
    let parts = super::lower_parts(&CONTAINER, &native, &s.p, &s.st, &s.io, &s.res, ALL_WAVES);
    assert!(!super::lower::wants_console_flow(&parts.unwrap()));
    // Pruned, it would.
    let mut q = crate::ir::retarget(&s.p, &jumps).unwrap();
    crate::sgpr_spill::forward(&mut q);
    let st = crate::structure::structure(&q).unwrap();
    let io = crate::io::raise(&q, &native).unwrap();
    let res = crate::resource::track(&q, &native.layout, native.stage).unwrap();
    let parts = super::lower_parts(&CONTAINER, &native, &q, &st, &io, &res, ALL_WAVES);
    assert!(super::lower::wants_console_flow(&parts.unwrap()));
    let (emitted, plain) = both(&native, ALL_WAVES);
    assert!(emitted.dxil == plain.dxil, "pruned");
    validate(&emitted);
}
