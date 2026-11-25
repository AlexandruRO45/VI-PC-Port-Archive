//! Tests for portable multi-wave Wave64 ballot, vote, prefix count, and cross-half read lowering.

use super::super::translate;
use crate::{ShaderExtract, Stage};

fn make_compute(group: [u32; 3], words: &[u32]) -> ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        flat_inputs: Vec::new(),
        flat_outputs: Vec::new(),
        stage: Stage::Compute,
        shader_model: (6, 0),
        required_features: None,
        bitcode_stripped: true,
        fspx_config: None,
        ps09_pixel_inputs: None,
        fssm: Vec::new(),
        entry: Some("main".into()),
        isa_data: isa.clone(),
        isa,
        isa_offset_in_fssm: 0,
        psv_threads: Some(group),
        isa_preamble: [0, group[0], group[1], group[2]],
        has_input_sig: false,
        has_output_sig: false,
        has_psv0: false,
        has_root_sig: false,
        input_sig: Vec::new(),
        output_sig: Vec::new(),
        resources: Vec::new(),
        gs_info: None,
        mesh_info: None,
        primitive_sig: Vec::new(),
        root_sig: None,
    }
}

// Helper RDNA2 encodings (verified against `tuxbox_dxbc::rdna2::disassemble`).
const S_NOP: u32 = 0xbf80_0000;
const S_ENDPGM: u32 = 0xbf81_0000;

fn v_cmp_u32_e64(op: u32, sdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xd400_0000 | (op << 16) | sdst, (src1 << 9) | src0]
}
fn v_cmp_ne_u32_vcc_v0_imm0() -> u32 {
    // VOPC op 197 (0xc5): (0x3e << 25) | (0xc5 << 17) | (vsrc1 << 9) | src0
    // src0 = 0x80 (inline 0), vsrc1 = 0 (v0)
    (0x3eu32 << 25) | (197u32 << 17) | (0u32 << 9) | 0x80
}
fn s_sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
    0xbe80_0000 | (sdst << 16) | (op << 8) | ssrc0
}
fn s_sop2(op: u32, sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0x8000_0000 | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
}
fn s_sopc(op: u32, ssrc0: u32, ssrc1: u32) -> u32 {
    0xbf00_0000 | (op << 16) | (ssrc1 << 8) | ssrc0
}
fn s_branch(op: u32, rel_dwords: i16) -> u32 {
    0xbf80_0000 | (op << 16) | (rel_dwords as u16 as u32)
}
fn v_vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7e00_0000 | (vdst << 17) | (op << 9) | src0
}
fn v_readlane_b32(sdst: u32, vsrc: u32, lane_src: u32) -> [u32; 2] {
    [0xd760_0000 | sdst, (lane_src << 9) | (256 + vsrc)]
}
fn v_mbcnt(high: bool, vdst: u32, mask_src: u32, acc_src: u32) -> [u32; 2] {
    let op = if high { 0x366u32 } else { 0x365u32 };
    [0xd400_0000 | (op << 16) | vdst, (acc_src << 9) | mask_src]
}

#[test]
fn ballot_and_count_bits_lower_portably_across_multiwave_and_nd_groups() {
    // v_cmp_ne_u32_e64 s[4:5], v0, 0 ; s_bcnt1_i32_b64 s6, s[4:5] ; s_endpgm
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [cmp[0], cmp[1], s_sop1(16, 6, 4), S_ENDPGM];
    for (group, waves, needs_tb_lane) in [
        ([64, 1, 1], 1u32, false),
        ([128, 1, 1], 2, true),
        ([8, 8, 1], 1, true),
        ([4, 4, 4], 1, true),
        ([256, 1, 1], 4, true),
        ([16, 16, 1], 4, true),
        ([1024, 1, 1], 16, true),
    ] {
        let translated = translate(&make_compute(group, &words)).expect("portable ballot");
        assert_eq!(translated.required_wave_size, None, "group={group:?}");
        assert_eq!(translated.profile, "cs_6_0", "group={group:?}");
        assert_eq!(translated.portable_compute_group, Some(group));
        assert!(!translated.hlsl.contains("[WaveSize(64)]"), "{}", translated.hlsl);
        assert!(
            translated.hlsl.contains(&format!("groupshared uint tb_w64_mask[{waves}][2];")),
            "{}",
            translated.hlsl
        );
        assert!(translated.hlsl.contains("tb_w64_ballot("), "{}", translated.hlsl);
        assert_eq!(
            translated.hlsl.contains("uint tb_lane = gtid.x +"),
            needs_tb_lane,
            "group={group:?}\n{}",
            translated.hlsl
        );
        let lane = if needs_tb_lane { "tb_lane" } else { "gtid.x" };
        assert!(translated.hlsl.contains(&format!("tb_w64_ballot({lane},")), "{}", translated.hlsl);
    }
}

#[test]
fn ballot_votes_all_any_count_prefix_first_and_cross_half_readlane() {
    // Compare v0 != 0 into s[4:5], test AnyTrue/AllTrue/AllEqual, countbits, prefix countbits,
    // v_readfirstlane_b32, s_ff1_i32_b64, and cross-half v_readlane_b32 from lane 47.
    let cmp_ballot = v_cmp_u32_e64(197, 4, 256, 0x80); // s[4:5] = ballot(v0 != 0)
    let cmp_vote = v_cmp_u32_e64(196, 8, 256, 0x85); // s[8:9] = mask(v0 > 5)
    let mb_lo = v_mbcnt(false, 3, 4, 0x80); // v3 = mbcnt_lo(s4, 0)
    let mb_hi = v_mbcnt(true, 3, 5, 256 + 3); // v3 = mbcnt_hi(s5, v3)
    let read47 = v_readlane_b32(14, 0, 0x80 + 47); // s14 = v0[lane 47]
    let words = [
        cmp_ballot[0], cmp_ballot[1],
        s_sop1(16, 6, 4),             // s_bcnt1_i32_b64 s6, s[4:5] (forces Ballot64 on s[4:5])
        mb_lo[0], mb_lo[1],
        mb_hi[0], mb_hi[1],
        cmp_vote[0], cmp_vote[1],
        s_sopc(19, 8, 0x80),          // s_cmp_lg_u64 s[8:9], 0  -> WaveActiveAnyTrue
        s_sopc(18, 8, 0xc1),          // s_cmp_eq_u64 s[8:9], -1 -> WaveActiveAllTrue
        s_sopc(18, 8, 0x7e),          // s_cmp_eq_u64 s[8:9], exec -> AllEqual
        s_sopc(18, 4, 0x7e),          // s_cmp_eq_u64 s[4:5], exec -> AllEqualBallot
        s_sop1(16, 10, 0x7e),         // s_bcnt1_i32_b64 s10, exec -> CountBits(Exec)
        v_cmp_ne_u32_vcc_v0_imm0(),   // v_cmp_ne_u32 vcc, 0, v0
        s_sop1(20, 11, 0x6a),         // s_ff1_i32_b64 s11, vcc -> FirstActive(Vcc)
        s_sop1(20, 13, 4),            // s_ff1_i32_b64 s13, s[4:5] -> ff1 on Ballot64
        v_vop1(2, 12, 256),           // v_readfirstlane_b32 s12, v0
        read47[0], read47[1],         // v_readlane_b32 s14, v0, 47 (cross-half)
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([128, 1, 1], &words)).expect("full wave64 op mix");
    assert_eq!(translated.required_wave_size, None);
    assert!(!translated.hlsl.contains("[WaveSize(64)]"));
    assert!(translated.hlsl.contains("groupshared uint tb_w64_mask[2][2];"));
    assert!(translated.hlsl.contains("groupshared uint tb_w64_lanes[2][64];"));
    assert!(translated.hlsl.contains("groupshared uint tb_swizzle[128];"));
    assert!(translated.hlsl.contains("tb_w64_ballot(tb_lane,"));
    assert!(translated.hlsl.contains("tb_w64_any(tb_lane,"));
    assert!(translated.hlsl.contains("tb_w64_all(tb_lane,"));
    assert!(translated.hlsl.contains("tb_w64_count_bits(tb_lane,"));
    assert!(translated.hlsl.contains("tb_w64_first_active(tb_lane,"));
    assert!(translated.hlsl.contains("tb_w64_read_first_lane(tb_lane,"));
    assert!(translated.hlsl.contains("tb_swizzle[(tb_lane & ~63u) | (0x2fu & 63u)]"));
}

#[test]
fn ballot_combined_with_exec_and_dynamic_readlane_lower_portably() {
    // s[4:5] = ballot(v0 != 0); s[6:7] = s[4:5] & exec; s8 = s_ff1_i32_b64(s[6:7]);
    // v_readlane_b32 s9, v0, s8 (dynamic SGPR lane read alongside ballot).
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let read_dyn = v_readlane_b32(9, 0, 8);
    let words = [
        cmp[0], cmp[1],
        s_sop2(0x0f, 6, 4, 0x7e), // s_and_b64 s[6:7], s[4:5], exec
        s_sop1(20, 8, 6),         // s_ff1_i32_b64 s8, s[6:7]
        read_dyn[0], read_dyn[1], // v_readlane_b32 s9, v0, s8
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([4, 4, 4], &words)).expect("ballot + dynamic readlane");
    assert_eq!(translated.required_wave_size, None);
    assert!(!translated.hlsl.contains("[WaveSize(64)]"));
    assert!(translated.hlsl.contains("tb_w64_ballot(tb_lane,"));
    assert!(translated.hlsl.contains("tb_reduction_read(tb_lane, v0, (s8 & 63u))"));
}

#[test]
fn control_dependent_tainted_scc_branch_around_ballot_is_refused() {
    // Lane-divergent vcc branch sets s8 = 1 inside one arm, then s_cmp_eq_i32 s8, 1
    // feeds s_cbranch_scc1 around a ballot. Control-dependence taint MUST refuse!
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        s_sop1(3, 8, 0x80),         // s_mov_b32 s8, 0
        v_cmp_ne_u32_vcc_v0_imm0(), // v_cmp_ne_u32 vcc, 0, v0
        s_branch(6, 1),             // s_cbranch_vccz +1 (divergent!)
        s_sop1(3, 8, 0x81),         // s_mov_b32 s8, 1 (control-dependent on divergent branch!)
        s_sopc(0, 8, 0x81),         // s_cmp_eq_i32 s8, 1 (SCC is Divergent!)
        s_branch(5, 3),             // s_cbranch_scc1 +3 (divergent skip!)
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),           // s_bcnt1_i32_b64 s6, s[4:5]
        S_ENDPGM,
    ];
    let err = translate(&make_compute([64, 1, 1], &words))
        .err()
        .expect("control-dependent tainted SCC branch around ballot must be refused");
    assert!(err.contains("divergent"), "unexpected error: {err}");
}

#[test]
fn vcc_ballot_word_moves_and_prefix_count_lower_portably() {
    // v_cmp_ne_u32 vcc, 0, v0 ; v_mov_b32 v1, vcc_lo ; v_mov_b32 v2, vcc_hi ;
    // v_mbcnt_lo v3, vcc_lo, 0 ; v_mbcnt_hi v3, vcc_hi, v3 ; s_endpgm
    let mb_lo = v_mbcnt(false, 3, 0x6a, 0x80);
    let mb_hi = v_mbcnt(true, 3, 0x6b, 256 + 3);
    let words = [
        v_cmp_ne_u32_vcc_v0_imm0(),
        v_vop1(1, 1, 0x6a), // v_mov_b32 v1, vcc_lo
        v_vop1(1, 2, 0x6b), // v_mov_b32 v2, vcc_hi
        mb_lo[0], mb_lo[1],
        mb_hi[0], mb_hi[1],
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([8, 8, 1], &words)).expect("vcc ballot");
    assert_eq!(translated.required_wave_size, None);
    assert!(translated.hlsl.contains("groupshared uint tb_w64_mask[1][2];"));
    assert!(translated.hlsl.contains("tb_w64_ballot(tb_lane, (vcc_lo & 1u) != 0u)"));
    assert!(!translated.hlsl.contains("if (exec_lo & 1u) {\n        uint2 w64_b"));
}

#[test]
fn multiwave_wave_uniform_skip_hoists_barriers_outside_branch_guard() {
    // Wave-uniform skip after ballot count:
    // s[4:5] = ballot(v0 != 0); s6 = countbits(s[4:5]); s_cmp_eq_u32 s6, 0;
    // s_cbranch_scc1 +4 (skips a second ballot + countbits inside the branch); s_endpgm
    let cmp1 = v_cmp_u32_e64(197, 4, 256, 0x80);
    let cmp2 = v_cmp_u32_e64(196, 8, 256, 0x85);
    let words = [
        cmp1[0], cmp1[1],
        s_sop1(16, 6, 4),    // s_bcnt1_i32_b64 s6, s[4:5]  (Wave-uniform on [128,1,1])
        s_sopc(0, 6, 0x80),  // s_cmp_eq_i32 s6, 0          (Wave-uniform SCC)
        s_branch(5, 3),      // s_cbranch_scc1 +3 (skips cmp2 + bcnt)
        cmp2[0], cmp2[1],
        s_sop1(16, 10, 8),   // s_bcnt1_i32_b64 s10, s[8:9]
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([128, 1, 1], &words)).expect("hoisted multiwave ballot");
    assert_eq!(translated.required_wave_size, None);
    // The second ballot MUST be hoisted outside `if (!(j...))` so both waves sync together!
    let guard_pos = translated.hlsl.find("if (!(j").expect("forward skip guard");
    let ballot_count = translated.hlsl.matches("tb_w64_ballot(tb_lane,").count();
    assert_eq!(ballot_count, 2, "{}", translated.hlsl);
    let second_ballot = translated.hlsl.rfind("tb_w64_ballot(tb_lane,").unwrap();
    assert!(
        second_ballot < guard_pos,
        "second ballot must be hoisted before guard:\n{}",
        translated.hlsl
    );
}

#[test]
fn divergent_vcc_branch_around_ballot_is_refused() {
    // Per-lane v_cmp_ne_u32 vcc, 0, v0 ; s_cbranch_vccz +3 ; ballot + countbits ; s_endpgm
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        v_cmp_ne_u32_vcc_v0_imm0(),
        s_branch(6, 3), // s_cbranch_vccz (divergent!)
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),
        S_ENDPGM,
    ];
    for group in [[64, 1, 1], [128, 1, 1], [8, 8, 1]] {
        let err = translate(&make_compute(group, &words))
            .err()
            .expect("divergent vcc branch around ballot must fail");
        assert!(err.contains("divergent"), "unexpected error: {err}");
    }
}

#[test]
fn divergent_execz_branch_around_ballot_is_refused() {
    // v_cmpx_ne_u32_e64 v0, 0 ; s_cbranch_execz +3 ; ballot + countbits ; s_endpgm
    let cmpx = v_cmp_u32_e64(197 + 16, 0x7e, 256, 0x80);
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmpx[0], cmpx[1],
        s_branch(8, 3), // s_cbranch_execz (divergent!)
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),
        S_ENDPGM,
    ];
    let err = translate(&make_compute([64, 1, 1], &words))
        .err()
        .expect("divergent execz branch around ballot must fail");
    assert!(err.contains("divergent"), "unexpected error: {err}");
}

#[test]
fn divergent_loop_enclosing_ballot_is_refused() {
    // Loop controlled by lane-divergent s_cbranch_vccnz back-edge enclosing a ballot.
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),
        v_cmp_ne_u32_vcc_v0_imm0(),
        s_branch(7, -5), // s_cbranch_vccnz back to 0 (divergent loop!)
        S_ENDPGM,
    ];
    let err = translate(&make_compute([64, 1, 1], &words))
        .err()
        .expect("divergent loop enclosing ballot must fail");
    assert!(err.contains("divergent") || err.contains("non-uniform loop"), "err={err}");
}

#[test]
fn multiwave_wave_varying_loop_enclosing_ballot_is_refused_but_single_wave_passes() {
    // Loop whose trip count depends on `s6 = s_bcnt1(ballot)` (Wave-uniform, NOT Group-uniform!).
    // Safe on [64,1,1] (1 wave => Wave == Group), MUST refuse on [128,1,1] (wave 0 != wave 1)!
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),   // s_bcnt1_i32_b64 s6, s[4:5]
        s_sopc(1, 6, 0x80), // s_cmp_lg_i32 s6, 0
        s_branch(5, -5),    // s_cbranch_scc1 back to 0
        S_ENDPGM,
    ];
    let single = translate(&make_compute([64, 1, 1], &words)).expect("single wave loop ok");
    assert_eq!(single.required_wave_size, None);
    let err = translate(&make_compute([128, 1, 1], &words))
        .err()
        .expect("multiwave wave-varying loop must be refused");
    assert!(err.contains("non-uniform loop"), "unexpected error: {err}");
}

#[test]
fn early_exit_before_ballot_is_refused() {
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        s_sopc(0, 0, 0x80), // s_cmp_eq_i32 s0, 0
        s_branch(5, 1),     // s_cbranch_scc1 skip_endpgm
        S_ENDPGM,           // early exit!
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),
        S_ENDPGM,
    ];
    let err = translate(&make_compute([64, 1, 1], &words))
        .err()
        .expect("early exit before ballot must fail");
    assert!(err.contains("early exit"), "unexpected error: {err}");
}

#[test]
fn non_multiple_of_64_group_does_not_lower_portable_wave64() {
    // 96-thread group with a Wave64 ballot is not a multiple of 64.
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [cmp[0], cmp[1], s_sop1(16, 6, 4), S_NOP, S_ENDPGM];
    let translated = translate(&make_compute([96, 1, 1], &words)).expect("96-thread fallback");
    assert!(!translated.hlsl.contains("tb_w64_ballot"));
}

#[test]
fn saveexec_scc_and_andn2_all_true_idiom_lower_portably() {
    // s[4:5] = v_cmp_ne_u32(v0, 0); s_andn2_b64 s[6:7], exec, s[4:5]; s_cbranch_scc1 +1; s_nop; s_endpgm
    // s_andn2_b64 sets SCC = any active lane where pred is FALSE (i.e. !WaveActiveAllTrue).
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmp[0], cmp[1],
        s_sop2(21, 6, 0x7e, 4),   // s_andn2_b64 s[6:7], exec, s[4:5]
        s_branch(5, 1),           // s_cbranch_scc1 +1
        S_NOP,
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([4, 4, 4], &words)).expect("andn2 all-true idiom");
    assert_eq!(translated.required_wave_size, None);
    assert!(translated.hlsl.contains("tb_w64_any(tb_lane, (s6 & 1u) != 0u)"));
}

#[test]
fn f1_null_dst_64bit_logic_vote_does_not_clobber_nul() {
    // s_andn2_b64 null (125), exec, s[4:5]; s_cbranch_scc1 +1
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmp[0], cmp[1],
        s_sop2(21, 125, 0x7e, 4), // s_andn2_b64 null, exec, s[4:5]
        s_branch(5, 1),           // s_cbranch_scc1 +1
        S_NOP,
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([64, 1, 1], &words)).expect("null dst andn2 vote");
    assert_eq!(translated.required_wave_size, None);
    assert!(translated.hlsl.contains("w64_null1 = (exec_lo & ~s4);"), "{}", translated.hlsl);
    assert!(translated.hlsl.contains("tb_w64_any(gtid.x, (w64_null1 & 1u) != 0u)"), "{}", translated.hlsl);
    assert!(!translated.hlsl.contains("tb_w64_any(gtid.x, (nul"), "{}", translated.hlsl);
}

#[test]
fn f2_ballot64_written_to_exec_or_vcc_extracts_lane_bit() {
    // s[4:5] = ballot(v0 != 0); s_bcnt1 s6, s[4:5] (forces Ballot64);
    // s_and_b64 exec, s[4:5], exec; s_and_b64 exec, s[4:5], s[4:5]; s_mov_b64 vcc, s[4:5];
    // s_cmov_b64 exec, s[4:5]; s_cselect_b64 vcc, s[4:5], 0; s_endpgm
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),           // s_bcnt1_i32_b64 s6, s[4:5]
        s_sop2(15, 0x7e, 4, 0x7e),  // s_and_b64 exec, s[4:5], exec
        s_sop2(15, 0x7e, 4, 4),     // s_and_b64 exec, s[4:5], s[4:5]
        s_sop1(4, 0x6a, 4),         // s_mov_b64 vcc, s[4:5]
        s_sop1(6, 0x7e, 4),         // s_cmov_b64 exec, s[4:5]
        s_sop2(11, 0x6a, 4, 0x80),  // s_cselect_b64 vcc, s[4:5], 0
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([64, 1, 1], &words)).expect("ballot64 to exec/vcc");
    assert_eq!(translated.required_wave_size, None);
    assert!(!translated.hlsl.contains("exec_hi = t"), "{}", translated.hlsl);
    assert!(!translated.hlsl.contains("vcc_hi = s5"), "{}", translated.hlsl);
    assert!(translated.hlsl.contains("exec_lo = (((((gtid.x & 63u) < 32u)"), "{}", translated.hlsl);
    assert!(translated.hlsl.contains("vcc_lo = (((((gtid.x & 63u) < 32u)"), "{}", translated.hlsl);
}

#[test]
fn f3_wrexec_b64_and_orn1_saveexec_lower_portably() {
    // s[4:5] = ballot(v0 != 0); s_bcnt1 s6, s[4:5]; s_andn2_wrexec_b64 s[8:9], s[4:5];
    // s_cbranch_scc1 +1; s_orn1_saveexec_b64 s[10:11], s[4:5]; s_endpgm
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmp[0], cmp[1],
        s_sop1(16, 6, 4),          // s_bcnt1_i32_b64 s6, s[4:5] (Ballot64 on s[4:5])
        s_sop1(58, 8, 4),          // s_andn2_wrexec_b64 s[8:9], s[4:5]
        s_branch(5, 1),            // s_cbranch_scc1 +1
        s_sop1(56, 10, 4),         // s_orn1_saveexec_b64 s[10:11], s[4:5]
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([64, 1, 1], &words)).expect("wrexec + orn1");
    assert_eq!(translated.required_wave_size, None);
    assert!(translated.hlsl.contains("tb_w64_any(gtid.x, (exec_lo & 1u) != 0u)"), "{}", translated.hlsl);
    assert!(translated.hlsl.contains("& ~exec_lo"), "{}", translated.hlsl);
    assert!(translated.hlsl.contains("| exec_lo"), "{}", translated.hlsl);
    assert!(!translated.hlsl.contains("s4 & ~exec_lo"), "{}", translated.hlsl);
}

#[test]
fn f4_literal_u32_max_is_low32_mask_not_all64_ones() {
    // s_cmp_eq_u64 s[4:5], Literal(0xffffffff) must promote s[4:5] to Ballot64 and compare
    // against (0xffffffffu, 0x0u), NOT emit tb_w64_all!
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        cmp[0], cmp[1],
        s_sopc(18, 4, 0xff), 0xffff_ffff, // s_cmp_eq_u64 s[4:5], Literal(0xffffffff)
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([64, 1, 1], &words)).expect("literal u32::MAX compare");
    assert_eq!(translated.required_wave_size, None);
    let main_body = &translated.hlsl[translated.hlsl.find("void main(").unwrap()..];
    assert!(main_body.contains("tb_w64_ballot("), "{main_body}");
    assert!(main_body.contains("s4 == 0xffffffffu && s5 == 0x0u"), "{main_body}");
    assert!(!main_body.contains("tb_w64_all("), "{main_body}");
}

#[test]
fn f5_logic_b64_combining_exec_with_scalar_pair_ballots_exec() {
    // s_mov_b32 s4, 2; s_mov_b32 s5, 0; s_and_b64 s[6:7], exec, s[4:5]; s_cbranch_scc1 +1; s_nop; s_endpgm
    let words = [
        s_sop1(3, 4, 0x82),        // s_mov_b32 s4, 2
        s_sop1(3, 5, 0x80),        // s_mov_b32 s5, 0
        s_sop2(15, 6, 0x7e, 4),    // s_and_b64 s[6:7], exec, s[4:5]
        s_branch(5, 1),            // s_cbranch_scc1 +1
        S_NOP,
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([64, 1, 1], &words)).expect("exec & scalar pair");
    assert_eq!(translated.required_wave_size, None);
    assert!(translated.hlsl.contains("tb_w64_ballot(gtid.x, (exec_lo & 1u) != 0u)"), "{}", translated.hlsl);
    assert!(translated.hlsl.contains("(w64_b2_0.x & s4)"), "{}", translated.hlsl);
    assert!(!translated.hlsl.contains("(exec_lo & s4)"), "{}", translated.hlsl);
}

#[test]
fn f6_cmov_b64_preserves_ballot_provenance_and_wqm_sdwa_stay_refused() {
    // F6a: s_cmov_b64 s[6:7], s[4:5] -> v_cndmask_b32 v1, 0, 1, s[6:7] extracts lane bit!
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    // v_cndmask_b32_e64 v1, 0, 1, s[6:7] (VOP3 op 0x101)
    let cnd: [u32; 2] = [0xd501_0001, (6u32 << 18) | (0x81u32 << 9) | 0x80];
    let cmov_words = [
        cmp[0], cmp[1],
        s_sop1(16, 8, 4),     // s_bcnt1_i32_b64 s8, s[4:5] (forces Ballot64 on s[4:5])
        s_sop1(6, 6, 4),      // s_cmov_b64 s[6:7], s[4:5]
        cnd[0], cnd[1],       // v_cndmask_b32 v1, 0, 1, s[6:7]
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([64, 1, 1], &cmov_words)).expect("cmov ballot provenance");
    assert_eq!(translated.required_wave_size, None);
    assert!(translated.hlsl.contains("(((gtid.x & 63u) < 32u) ? s6 : s7)"), "{}", translated.hlsl);

    // F6b: s_wqm_b64 s[4:5], exec followed by s_bcnt1_i32_b64 s6, s[4:5] must refuse (unpromotable Mask)!
    let wqm_bcnt_words = [
        s_sop1(0x1c, 4, 0x7e), // s_wqm_b64 s[4:5], exec
        s_sop1(16, 6, 4),      // s_bcnt1_i32_b64 s6, s[4:5]
        S_ENDPGM,
    ];
    assert!(
        translate(&make_compute([64, 1, 1], &wqm_bcnt_words)).is_err(),
        "s_wqm_b64 followed by s_bcnt1 must be refused"
    );

    // F6c: SDWA v_mov_b32 v1, vcc_lo (src0 = 0xf9 SDWA, bit 23 scalar flag + 0x6a vcc_lo) is refused by F7!
    let sdwa_mov = [v_vop1(1, 1, 0xf9), 0x0086_006a, S_ENDPGM];
    assert!(
        translate(&make_compute([64, 1, 1], &sdwa_mov)).is_err(),
        "SDWA v_mov_b32 reading vcc_lo must be refused"
    );

    // F6d: exec_snapshots Site::Capture(Shared64) followed by s_bcnt1_i32_b64 must refuse
    // instead of misreading per-lane 1-bit `s20` as a 64-bit ballot!
    let snap_bcnt = [
        0x7d86_00a3,      // v_cmp_gt_u32 vcc, 35, v0
        0xbe94_246a,      // s_and_saveexec_b64 s[20:21], vcc
        0xbe98_147e,      // s_ff1_i32_b64 s24, exec (triggers Shared64 exec_snapshots)
        0xbefe_0414,      // s_mov_b64 exec, s[20:21]
        0x7e08_0214,      // v_mov_b32 v4, s20 (forces Site::Capture(Shared64) on s[20:21])
        s_sop1(16, 6, 20), // s_bcnt1_i32_b64 s6, s[20:21] (not handled by exec_snapshots!)
        S_ENDPGM,
    ];
    assert!(
        translate(&make_compute([64, 1, 1], &snap_bcnt)).is_err(),
        "s_bcnt1_i32_b64 on exec_snapshots Site::Capture must be refused"
    );
}

#[test]
fn f7_unhandled_32bit_exec_vcc_reads_and_s_wqm_b32_are_refused() {
    // s_wqm_b32 s8, exec_lo (SOP1 op 0x1b) must NEVER fall through to native WaveActiveBallot!
    let wqm32 = [s_sop1(0x1b, 8, 0x7e), S_ENDPGM];
    assert!(
        translate(&make_compute([64, 1, 1], &wqm32)).is_err(),
        "s_wqm_b32 s8, exec_lo must be refused"
    );
    // s_bcnt1_i32_b32 s8, exec_lo (SOP1 op 15) must NOT read per-lane 1-bit exec_lo!
    let bcnt32 = [s_sop1(15, 8, 0x7e), S_ENDPGM];
    assert!(
        translate(&make_compute([64, 1, 1], &bcnt32)).is_err(),
        "s_bcnt1_i32_b32 s8, exec_lo must be refused"
    );
}

#[test]
fn f8_vop3_readfirstlane_writes_sgpr_and_hoists_multiwave_dependent_ballot() {
    // VOP3 v_readfirstlane_b32 (op 386 = 0x182) s6, v0 -> s_cmp_eq_i32 s6, 0 -> s_cbranch_scc1 -> ballot
    // On [128,1,1], s6 MUST be treated as Wave-uniform so the following ballot is hoisted outside `if (!(j...))`!
    let rfl_vop3: [u32; 2] = [0xd400_0000 | (386u32 << 16) | 6, 256];
    let cmp = v_cmp_u32_e64(197, 4, 256, 0x80);
    let words = [
        rfl_vop3[0], rfl_vop3[1],
        s_sopc(0, 6, 0x80), // s_cmp_eq_i32 s6, 0
        s_branch(5, 3),     // s_cbranch_scc1 +3
        cmp[0], cmp[1],
        s_sop1(16, 8, 4),   // s_bcnt1_i32_b64 s8, s[4:5]
        S_ENDPGM,
    ];
    let translated = translate(&make_compute([128, 1, 1], &words)).expect("VOP3 readfirstlane");
    assert_eq!(translated.required_wave_size, None);
    assert!(translated.hlsl.contains("s6 = tb_w64_read_first_lane("), "{}", translated.hlsl);
    assert!(!translated.hlsl.contains("v6 = tb_w64_read_first_lane("), "{}", translated.hlsl);
    let guard_pos = translated.hlsl.find("if (!(j").expect("forward skip guard");
    let ballot_pos = translated.hlsl.find("tb_w64_ballot(tb_lane,").expect("hoisted ballot");
    assert!(ballot_pos < guard_pos, "ballot must be hoisted before guard:\n{}", translated.hlsl);
}
