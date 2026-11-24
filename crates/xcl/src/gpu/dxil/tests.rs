//! End-to-end checks of emitted DXIL: the SDK validator accepts it, DXC
//! disassembles it, and D3D12 runs it on the GPU with the recompiler's
//! kernel ABI (u0 `io`, u1 `res` root UAVs).

use super::container::Stage;
use super::module::{BinOp, CastOp, Pred};
use super::shader::Shader;
use super::validate;
use crate::gpu::recomp::runner;

/// The program of `tests/data/dxil/dxc_cs_raw_uav.hlsl`, emitted directly:
/// `a = io[lane]; f = asfloat(a) * 2 + 1; if (a > 5) f = -f;
///  res[lane] = asuint(f) + WaveReadLaneAt(a, 3)` — with a real branch and phi
/// rather than a select, to exercise control flow.
fn sample_kernel(wave: u32) -> Vec<u8> {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = s.raw_uav(1, 0);
    let lane = s.thread_id_in_group(0);
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, lane, two);
    let a = s.raw_load(io, off, 1)[0];
    let f32 = s.m.f32();
    let i32 = s.m.i32();
    let fa = s.m.bitcast(a, f32);
    let c2 = s.m.f32c(2.0f32.to_bits());
    let c1 = s.m.f32c(1.0f32.to_bits());
    let mul = s.m.fbin(BinOp::Mul, fa, c2, 0);
    let f = s.m.fbin(BinOp::Add, mul, c1, 0);
    let five = s.m.i32c(5);
    let big = s.m.cmp(Pred::UGt, a, five);
    let three = s.m.i32c(3);
    let other = s.wave_read_lane_at(a, three);

    let entry = s.m.current_block();
    let neg_b = s.m.new_block();
    let join = s.m.new_block();
    s.m.cond_br(big, neg_b, join);
    s.m.set_block(neg_b);
    let nz = s.m.f32c(0x8000_0000);
    let neg = s.m.fbin(BinOp::Sub, nz, f, 0);
    s.m.br(join);
    s.m.set_block(join);
    let phi = s.m.phi(f32);
    s.m.add_incoming(phi, f, entry);
    s.m.add_incoming(phi, neg, neg_b);
    let bits = s.m.cast(CastOp::BitCast, phi, i32);
    let sum = s.m.bin(BinOp::Add, bits, other);
    s.raw_store(res, off, &[sum]);
    s.finish([64, 1, 1], Some(wave))
}

fn expected(io: &[u32], wave: usize) -> Vec<u32> {
    (0..64)
        .map(|l| {
            let a = io[l];
            let mut f = f32::from_bits(a) * 2.0 + 1.0;
            if a > 5 {
                f = -f;
            }
            f.to_bits().wrapping_add(io[l / wave * wave + 3])
        })
        .collect()
}

#[test]
fn emitted_kernel_passes_the_sdk_validator() {
    let c = sample_kernel(64);
    if let Err(e) = validate::validate(&c) {
        let dis = validate::disassemble(&c).unwrap_or_else(|d| d);
        panic!("validator rejected the emitted kernel:\n{e}\n--- disassembly ---\n{dis}");
    }
}

#[test]
fn emitted_kernel_disassembles_like_dxc_output() {
    let dis = validate::disassemble(&sample_kernel(64)).expect("disassembly");
    for needle in [
        "@dx.op.createHandleFromBinding(i32 217",
        "@dx.op.rawBufferLoad.i32(i32 139",
        "@dx.op.waveReadLaneAt.i32(i32 117",
        "phi float",
        "!{i32 64, i32 1, i32 1}",
    ] {
        assert!(dis.contains(needle), "missing `{needle}` in:\n{dis}");
    }
}

#[test]
fn emitted_kernel_runs_on_the_gpu() {
    let wave = match (runner::wave64_supported(), runner::wave32_supported()) {
        (Ok(true), _) => 64,
        (_, Ok(true)) => 32,
        (Err(e), _) => return runner::skip_test(&e),
        _ => return runner::skip_test("neither wave64 nor wave32 support"),
    };
    let c = sample_kernel(wave);
    validate::validate(&c).expect("valid DXIL");
    // io dwords 0..64 are the SGPR slots the runner uploads first.
    let mut sgpr = vec![0u32; 128];
    for (l, v) in sgpr.iter_mut().take(64).enumerate() {
        *v = match l % 4 {
            0 => l as u32,                     // small integers (≤ 5 stay positive)
            1 => (l as f32 * 0.5).to_bits(),   // floats, all > 5 as integers → negated
            2 => 3,
            _ => (-(l as f32)).to_bits(),
        };
    }
    let state = runner::KernelState { sgpr: sgpr.clone(), vgpr: vec![vec![0; 256]; 64], scc: 0 };
    let r = runner::run(&c, &state, &[]).expect("GPU run");
    assert_eq!(&r.raw[..64], &expected(&sgpr, wave as usize)[..]);
}

/// Blocks created out of dominance order (a block used as a branch target
/// before the block that dominates it exists), an unreachable block feeding a
/// phi, and `replace_all_uses` on a trivial phi: the writer lays blocks out in
/// reverse post-order, drops the unreachable one and its phi edge, and the
/// result validates and runs (res[lane] = (lane << 2) + 7 + io[lane]).
fn out_of_order_kernel() -> Vec<u8> {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = s.raw_uav(1, 0);
    let lane = s.thread_id_in_group(0);
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, lane, two);
    let entry = s.m.current_block();
    // created first, reached last
    let tail = s.m.new_block();
    let dead = s.m.new_block();
    let mid = s.m.new_block();
    s.m.br(mid);

    s.m.set_block(mid);
    let seven = s.m.i32c(7);
    let v = s.m.bin(BinOp::Add, off, seven);
    let x = s.raw_load(io, off, 1)[0];
    s.m.br(tail);

    s.m.set_block(dead);
    let junk = s.m.i32c(0xdead);
    s.m.br(tail);

    s.m.set_block(tail);
    let i32 = s.m.i32();
    let trivial = s.m.phi(i32); // only ever `v`: replaced below
    s.m.add_incoming(trivial, v, mid);
    let p = s.m.phi(i32);
    s.m.add_incoming(p, x, mid);
    s.m.add_incoming(p, junk, dead);
    let sum = s.m.bin(BinOp::Add, trivial, p);
    s.raw_store(res, off, &[sum]);
    s.m.replace_all_uses(trivial, v);
    let _ = entry;
    s.finish([64, 1, 1], None)
}

#[test]
fn blocks_are_written_in_reverse_post_order() {
    let c = out_of_order_kernel();
    if let Err(e) = validate::validate(&c) {
        let dis = validate::disassemble(&c).unwrap_or_else(|d| d);
        panic!("validator rejected the out-of-order kernel:\n{e}\n--- disassembly ---\n{dis}");
    }
    let dis = validate::disassemble(&c).expect("disassembly");
    assert!(!dis.contains("57005"), "the unreachable block's constant must be gone:\n{dis}");
    match runner::wave32_supported().or_else(|_| runner::wave64_supported()) {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave32 / wave64"),
        Err(e) => return runner::skip_test(&e),
    }
    let mut sgpr = vec![0u32; 128];
    for (l, v) in sgpr.iter_mut().take(64).enumerate() {
        *v = 1000 + l as u32 * 3;
    }
    let state = runner::KernelState { sgpr: sgpr.clone(), vgpr: vec![vec![0; 256]; 64], scc: 0 };
    let r = runner::run(&c, &state, &[]).expect("GPU run");
    let want: Vec<u32> = (0..64).map(|l| ((l as u32) << 2) + 7 + sgpr[l]).collect();
    assert_eq!(&r.raw[..64], &want[..]);
}

/// Multi-window accesses lower through one `switch` per instruction (not four diamond
/// branches) and contiguous multi-dword SMEM / FLAT / `ld_raw` accesses vectorize into
/// 4-wide `rawBufferLoad` / `rawBufferStore` ops.
#[test]
fn multi_window_accesses_emit_one_switch_and_wide_raw_ops() {
    if crate::gpu::recomp::dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| crate::gpu::rdna2::encode(n, e, f, None).unwrap();
    let mut code = Vec::new();
    code.extend(enc("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("OFFSET", 0), ("SOFFSET", 125)]));
    code.extend(enc("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 28), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]));
    code.extend(enc("GLOBAL_LOAD_DWORDX4", "ENC_FLAT_GLBL", &[("VDST", 4), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)]));
    code.extend(enc("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("ADDR", 2), ("DATA", 4), ("SADDR", 0x7D), ("OFFSET", 0)]));
    code.extend(enc("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 8), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));

    let dxil = super::kernel::translate(
        &code,
        super::kernel::KernelOptions { wave64: true, split: false, mem_windows: 4, sampler_base: 0 },
    )
    .expect("translate");
    validate::validate(&dxil).expect("valid DXIL");
    let dis = validate::disassemble(&dxil).expect("disassembly");
    assert_eq!(dis.matches("switch i32").count(), 5, "one switch per multi-dword instruction:\n{dis}");
    assert!(dis.contains("@dx.op.rawBufferLoad.i32") && dis.contains("i8 15"), "expected 4-wide rawBufferLoad:\n{dis}");
    assert!(dis.contains("@dx.op.rawBufferStore.i32") && dis.contains("i8 15"), "expected 4-wide rawBufferStore:\n{dis}");
}

/// Multi-window vectorized SMEM, MUBUF, `ld_raw`, GLOBAL load/store, and 64-bit atomic
/// (merging `i64` directly through `switch` phis) execute accurately across windows on GPU.
#[test]
fn multi_window_vector_loads_stores_and_atomics_run_on_the_gpu() {
    if crate::gpu::recomp::dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let split = match (runner::wave64_supported(), runner::wave32_supported()) {
        (Ok(true), _) => false,
        (_, Ok(true)) => true,
        (Err(e), _) => return runner::skip_test(&e),
        _ => return runner::skip_test("neither wave64 nor wave32 support"),
    };
    let Some(arena) = crate::gpu::uma::Uma::with_windows(2, 1 << 20) else {
        return runner::skip_test("no UMA arena");
    };
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let hi = |a: u64| (a >> 32) as u32;
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };

    for i in 0..8u64 {
        *word(w1 + 0x1000 + 4 * i) = 0x1000_0000 + i as u32;
    }
    for i in 0..4u64 {
        *word(w1 + 0x2000 + 4 * i) = 0x2000_0000 + i as u32;
    }
    for i in 0..70u64 {
        *word(w1 + 0x2100 + 4 * i) = 0x3000_0000 + i as u32;
    }
    // 5 dwords in w1 + 0x3000 for unaligned (+1 byte) BUFFER_LOAD_FORMAT_XYZW (32_32_32_32_UINT)
    for i in 0..5u64 {
        *word(w1 + 0x3000 + 4 * i) = u32::from_le_bytes([
            (0x10 + 4 * i) as u8,
            (0x11 + 4 * i) as u8,
            (0x12 + 4 * i) as u8,
            (0x13 + 4 * i) as u8,
        ]);
    }
    for l in 0..64u64 {
        for k in 0..4u64 {
            *word(w1 + 0x4000 + 16 * l + 4 * k) = 0x4000_0000 + (4 * l + k) as u32;
        }
    }

    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| crate::gpu::rdna2::encode(n, e, f, None).unwrap();
    let mut code = Vec::new();
    code.extend(enc("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("OFFSET", 0), ("SOFFSET", 125)]));
    code.extend(enc("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 28), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]));
    code.extend(enc("BUFFER_LOAD_DWORDX3", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 30), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(enc("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 14), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 1)]));
    code.extend(enc("GLOBAL_LOAD_DWORDX4", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)]));
    code.extend(enc("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("ADDR", 2), ("DATA", 20), ("SADDR", 0x7D), ("OFFSET", 0)]));
    code.extend(enc("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("ADDR", 4), ("DATA", 20), ("SADDR", 0x7D), ("OFFSET", 0)]));
    code.extend(enc("GLOBAL_ATOMIC_ADD_X2", "ENC_FLAT_GLBL", &[("VDST", 24), ("ADDR", 6), ("DATA", 8), ("SADDR", 0x7D), ("OFFSET", 0), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));

    let dxil = super::kernel::translate(
        &code,
        super::kernel::KernelOptions { wave64: true, split, mem_windows: 2, sampler_base: 0 },
    )
    .expect("translate");

    let ident = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    let mut st = runner::KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(w1 + 0x1000) as u32, hi(w1)]);
    st.sgpr[4..8].copy_from_slice(&[(w1 + 0x2000) as u32, hi(w1) & 0xFFFF, 12, 0]);
    st.sgpr[8..12].copy_from_slice(&[(w1 + 0x2100) as u32, hi(w1) & 0xFFFF, 0x104, 0]);
    st.sgpr[12..16].copy_from_slice(&[(w1 + 0x3000) as u32, hi(w1) & 0xFFFF, 64, ident | (75 << 12)]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u64;
        r[0] = (w1 + 0x4000 + 16 * l) as u32;
        r[1] = hi(w1);
        r[2] = (w0 + 0x5000 + 16 * l) as u32;
        r[3] = hi(w0);
        r[4] = (w1 + 0x6000 + 16 * l) as u32;
        r[5] = hi(w1);
        r[6] = (w1 + 0x7000) as u32;
        r[7] = hi(w1);
        r[8] = 1;
        r[9] = 1;
        r[30] = 8 * l as u32;
        r[10] = 0xDEAD;
        r[11] = 0xDEAD;
        r[12] = 0xDEAD;
    }

    let r = match runner::run_in_arena(&dxil, &st, &arena) {
        Ok(r) => r,
        Err(e) if e.contains("OpenExistingHeapFromAddress") => return runner::skip_test(&e),
        Err(e) => panic!("arena run: {e}"),
    };
    for i in 0..8 {
        assert_eq!(r.sgpr[20 + i], 0x1000_0000 + i as u32, "s_load_dwordx8[{i}]");
    }
    assert_eq!(r.sgpr[28..32], [0x2000_0000, 0x2000_0001, 0x2000_0002, 0], "s_buffer_load_dwordx4 straddling num_records");
    for l in 0..64usize {
        for k in 0..3usize {
            let off = 8 * l + 4 * k;
            let want = if off + 4 <= 0x104 { 0x3000_0000 + (off / 4) as u32 } else { 0 };
            assert_eq!(r.vgpr[l][10 + k], want, "lane {l} buffer_load_dwordx3[{k}]");
        }
        for k in 0..4usize {
            let b0 = 0x10 + 4 * k;
            let want_fmt = u32::from_le_bytes([(b0 + 1) as u8, (b0 + 2) as u8, (b0 + 3) as u8, (b0 + 4) as u8]);
            assert_eq!(r.vgpr[l][14 + k], want_fmt, "lane {l} unaligned ld_raw[{k}]");
            let want_g = 0x4000_0000 + (4 * l + k) as u32;
            assert_eq!(r.vgpr[l][20 + k], want_g, "lane {l} global_load_dwordx4[{k}]");
            assert_eq!(*word(w0 + 0x5000 + 16 * l as u64 + 4 * k as u64), want_g, "lane {l} global_store w0[{k}]");
            assert_eq!(*word(w1 + 0x6000 + 16 * l as u64 + 4 * k as u64), want_g, "lane {l} global_store w1[{k}]");
        }
    }
    let final64 = (*word(w1 + 0x7000) as u64) | ((*word(w1 + 0x7004) as u64) << 32);
    assert_eq!(final64, 64 * 0x1_0000_0001u64, "global_atomic_add_x2 in window 1");
    assert_eq!((*word(w0 + 0x7000), *word(w0 + 0x7004)), (0, 0), "window 0 untouched by atomic");
    let old_sum: u64 = (0..64)
        .map(|l| (r.vgpr[l][24] as u64) | ((r.vgpr[l][25] as u64) << 32))
        .sum();
    assert_eq!(old_sum, (0..64u64).map(|k| k * 0x1_0000_0001u64).sum::<u64>(), "64-bit GLC old values across 64 lanes");
}
