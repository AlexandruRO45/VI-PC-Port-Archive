//! A stage-timed translation benchmark over a synthetic program, for
//! telling which pass a change makes faster or slower.

use super::container::SigElement;
use super::emit;
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, StageIo};

fn output_signature() -> Container<'static> {
    let el = |name: &str, kind, comp| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp,
        interp: 0,
        row: None,
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    };
    let osg1 = crate::emit::container::signature(&[el("SV_Depth", 17, 9)], false);
    let bytes: &'static [u8] = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]).leak();
    Container::parse(bytes).unwrap()
}

/// A pixel program of a counted loop over ALU chains: v_mov_b32 v1, 5;
/// s_mov_b32 s4, 0; then `250` iterations of {s_cmpk_lg_u32 s4, 100 (SOPK),
/// s_cbranch_scc0, 40 VOP3 v_mad_f32 pairs, s_addk_i32 s4, 1 (SOPK),
/// s_branch}, exp mrtz, s_endpgm.
fn program() -> NativeProgram {
    let mut w = vec![0x7E02_0285, 0xBE84_0380];
    let head = w.len() as u32;
    let mut exits = vec![];
    for _ in 0..250 {
        w.push(0xB004_0064); // s_cmpk_lg_u32 s4, 100
        exits.push(w.len() as u32);
        w.push(0xBF00_0000); // s_cbranch_scc0 <exit>
        for k in 0..40 {
            w.push(0xD400_0000u32 | 0x141 << 16 | (k % 8) as u32); // v_mad_f32 v[k%8]
            w.push(128 | 128 << 9 | 128 << 18); // inline integer zeros
        }
        w.push(0xB00F_0001); // s_addk_i32 s4, 1
        let back = (head as i64 - w.len() as i64 - 1) as u16 as u32; // s_branch head
        w.push(0xBF80_0000 | back);
    }
    let exit = w.len() as u32; // s_cbranch_scc0 reaches the export
    w.push(0xF800_1883); // exp mrtz v0, v1, off, off compr done vm
    w.push(0x0000_0100);
    w.push(0xBF81_0000); // s_endpgm
    for cb in exits {
        w[cb as usize] |= (exit - cb - 1) as u16 as u32;
    }
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// Microseconds per stage, over at least 3 calls and 2 s.
#[test]
#[ignore = "benchmark: cargo test --release -p victory-shader bench_translate -- --ignored --nocapture"]
fn bench_translate() {
    let c = &output_signature();
    let native = program();
    let mut stages = [0f64; 5];
    let mut calls = 0;
    let t = std::time::Instant::now();
    while calls < 3 || t.elapsed().as_secs_f64() < 2.0 {
        let n = std::time::Instant::now();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        stages[0] += n.elapsed().as_secs_f64();
        let st = crate::structure::structure(&p).unwrap();
        let n = std::time::Instant::now();
        stages[1] += n.elapsed().as_secs_f64();
        let n = std::time::Instant::now();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        stages[2] += n.elapsed().as_secs_f64();
        let io = crate::io::raise(&p, &native).unwrap();
        stages[3] += n.elapsed().as_secs_f64();
        let n = std::time::Instant::now();
        let s = emit(c, &native, &p, &st, &io, &res).unwrap();
        stages[4] += n.elapsed().as_secs_f64();
        std::hint::black_box(&s);
        calls += 1;
    }
    let per = |x: f64| x / f64::from(calls) * 1e3;
    eprintln!(
        "calls={calls} ir={:.2}ms structure={:.2}ms resource={:.2}ms io={:.2}ms emit={:.2}ms",
        per(stages[0]),
        per(stages[1]),
        per(stages[2]),
        per(stages[3]),
        per(stages[4]),
    );
}
