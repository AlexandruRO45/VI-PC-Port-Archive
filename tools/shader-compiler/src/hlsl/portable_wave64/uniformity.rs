//! Uniformity, convergence, hoisting, and groupshared-budget verification.

use super::super::{
    compute_group, wave_sum, Emitter, Enc, Insn, Opnd, Stage, COUNT_LANES, GROUP_READ,
    OP_WAVE_REDUCE,
};
use super::masks::{analyze_masks, build_cfg, Cfg, MaskPlan};
use super::{Analysis, MaskOp, Plan};
use crate::rdna2::clobbers_scalar;
use std::collections::{BTreeSet, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Uniformity {
    Divergent = 0,
    Wave = 1,
    Group = 2,
}

#[derive(Clone, PartialEq, Eq)]
struct UniformState {
    sgpr: [Uniformity; 106],
    scc: Uniformity,
}

impl UniformState {
    fn entry() -> Self {
        Self {
            sgpr: [Uniformity::Group; 106],
            scc: Uniformity::Divergent,
        }
    }

    fn all_divergent() -> Self {
        Self {
            sgpr: [Uniformity::Divergent; 106],
            scc: Uniformity::Divergent,
        }
    }

    fn opnd(&self, op: &Opnd, wide: bool) -> Uniformity {
        match *op {
            Opnd::ConstInt(_) | Opnd::ConstF32(_) | Opnd::Literal(_) => Uniformity::Group,
            Opnd::Scc => self.scc,
            Opnd::Sgpr(r @ 0..=105) => {
                let lo = self.sgpr[r as usize];
                if wide && r < 105 {
                    lo.min(self.sgpr[r as usize + 1])
                } else if wide {
                    Uniformity::Divergent
                } else {
                    lo
                }
            }
            _ => Uniformity::Divergent,
        }
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (a, &b) in self.sgpr.iter_mut().zip(&other.sgpr) {
            let m = (*a).min(b);
            if m != *a {
                *a = m;
                changed = true;
            }
        }
        let s = self.scc.min(other.scc);
        if s != self.scc {
            self.scc = s;
            changed = true;
        }
        changed
    }
}

fn scalar_preserves_scc(ins: &Insn) -> bool {
    matches!(
        ins.mnemonic.as_str(),
        "s_mov_b32"
            | "s_mov_b64"
            | "s_movk_i32"
            | "s_cselect_b32"
            | "s_cselect_b64"
            | "s_cmov_b32"
            | "s_cmov_b64"
            | "s_cmovk_i32"
            | "s_getpc_b64"
            | "s_brev_b32"
            | "s_brev_b64"
            | "s_bitset0_b32"
            | "s_bitset0_b64"
            | "s_bitset1_b32"
            | "s_bitset1_b64"
            | "s_bfm_b32"
            | "s_bfm_b64"
            | "s_mul_i32"
            | "s_mulk_i32"
            | "s_mul_hi_u32"
            | "s_mul_hi_i32"
            | "s_ff1_i32_b32"
            | "s_ff1_i32_b64"
            | "s_ff0_i32_b32"
            | "s_ff0_i32_b64"
            | "s_flbit_i32_b32"
            | "s_flbit_i32_b64"
            | "s_flbit_i32"
            | "s_flbit_i32_i64"
            | "s_sext_i32_i8"
            | "s_sext_i32_i16"
            | "s_pack_ll_b32_b16"
            | "s_pack_lh_b32_b16"
            | "s_pack_hh_b32_b16"
            | "s_setreg_b32"
            | "s_setreg_imm32_b32"
            | "s_getreg_b32"
    )
}

pub(super) fn scalar_writes_scc(ins: &Insn) -> bool {
    let m = ins.mnemonic.as_str();
    match ins.enc {
        Enc::Sopc => m.starts_with("s_cmp_") || m.starts_with("s_bitcmp"),
        Enc::Sopk => m.starts_with("s_cmpk_") || m == "s_addk_i32",
        Enc::Sop2 => matches!(
            m,
            "s_add_u32"
                | "s_sub_u32"
                | "s_add_i32"
                | "s_sub_i32"
                | "s_addc_u32"
                | "s_subb_u32"
                | "s_min_i32"
                | "s_min_u32"
                | "s_max_i32"
                | "s_max_u32"
                | "s_and_b32"
                | "s_and_b64"
                | "s_or_b32"
                | "s_or_b64"
                | "s_xor_b32"
                | "s_xor_b64"
                | "s_andn2_b32"
                | "s_andn2_b64"
                | "s_orn2_b32"
                | "s_orn2_b64"
                | "s_nand_b32"
                | "s_nand_b64"
                | "s_nor_b32"
                | "s_nor_b64"
                | "s_xnor_b32"
                | "s_xnor_b64"
                | "s_lshl_b32"
                | "s_lshl_b64"
                | "s_lshr_b32"
                | "s_lshr_b64"
                | "s_ashr_i32"
                | "s_ashr_i64"
                | "s_bfe_u32"
                | "s_bfe_i32"
                | "s_bfe_u64"
                | "s_bfe_i64"
                | "s_absdiff_i32"
                | "s_lshl1_add_u32"
                | "s_lshl2_add_u32"
                | "s_lshl3_add_u32"
                | "s_lshl4_add_u32"
        ),
        Enc::Sop1 => matches!(
            m,
            "s_not_b32"
                | "s_not_b64"
                | "s_bcnt0_i32_b32"
                | "s_bcnt0_i32_b64"
                | "s_bcnt1_i32_b32"
                | "s_bcnt1_i32_b64"
                | "s_abs_i32"
        ),
        _ => false,
    }
}

fn scalar_reads_dst(ins: &Insn) -> bool {
    let m = ins.mnemonic.as_str();
    (ins.enc == Enc::Sopk && !m.starts_with("s_movk") && !m.starts_with("s_getreg"))
        || matches!(
            m,
            "s_bitset0_b32"
                | "s_bitset0_b64"
                | "s_bitset1_b32"
                | "s_bitset1_b64"
                | "s_cmov_b32"
                | "s_cmov_b64"
        )
}

fn branch_uniformity(
    em: &Emitter<'_>,
    idx: usize,
    state: &UniformState,
    ctl: Uniformity,
    wave_level: Uniformity,
) -> Uniformity {
    let ins = &em.insns[idx];
    if ins.enc != Enc::Sopp {
        return Uniformity::Divergent;
    }
    match ins.op {
        1 | 2 => ctl,
        4 | 5 => state.scc.min(ctl),
        8 if em.reduction_execz_guards.contains(&idx) || em.lockstep.is_some() => wave_level.min(ctl),
        6 | 7 | 9 if em.lockstep.is_some() => wave_level.min(ctl),
        _ => Uniformity::Divergent,
    }
}

fn loop_uniformity(
    em: &Emitter<'_>,
    cfg: &Cfg,
    l: usize,
    b: usize,
    states: &[UniformState],
    ctl_level: &[Uniformity],
    wave_level: Uniformity,
) -> Uniformity {
    let mut u = branch_uniformity(em, b, &states[b], ctl_level[b], wave_level);
    for j in l..=b {
        if em.insns[j].mnemonic == "s_endpgm" {
            return Uniformity::Divergent;
        }
        if let Some(&t) = cfg.target.get(&j) {
            if t >= b {
                u = u.min(branch_uniformity(em, j, &states[j], ctl_level[j], wave_level));
            }
        }
    }
    for (&from, &to) in &cfg.target {
        if from < l && to > l {
            u = u.min(branch_uniformity(
                em,
                from,
                &states[from],
                ctl_level[from],
                wave_level,
            ));
        }
    }
    u
}

fn analyze_uniformity(
    em: &Emitter<'_>,
    cfg: &Cfg,
    masks: &MaskPlan,
    single_wave: bool,
) -> (Vec<UniformState>, Vec<Uniformity>) {
    let insns = &em.insns;
    let n = insns.len();
    let wave_level = if single_wave { Uniformity::Group } else { Uniformity::Wave };
    let packed_ballots = em.packed.as_ref().map(|p| &p.ballots);
    let mut ctl_level = vec![Uniformity::Group; n];

    loop {
        let mut states: Vec<Option<UniformState>> = vec![None; n];
        if n > 0 {
            states[0] = Some(UniformState::entry());
        }
        let mut work = VecDeque::from([0usize]);
        let mut iters = 0usize;
        let mut timed_out = false;
        while let Some(i) = work.pop_front() {
            iters += 1;
            if iters > n.saturating_mul(64) {
                timed_out = true;
                break;
            }
            let ins = &insns[i];
            let incoming = states[i].as_ref().unwrap();
            let mut out = incoming.clone();
            let m = ins.mnemonic.as_str();
            let ctl = ctl_level[i];
            let wave_res = wave_level.min(ctl);

            let is_wave_read = (ins.enc == Enc::Vop3 && ins.op == wave_sum::READ_LANE)
                || matches!(m, "tb_indexed_readlane" | "tb_group_readlane");
            let is_ballot_def = masks.ballot_compares.contains(&i)
                || masks.ballot_mask_copies.contains(&i)
                || masks.ballot_logic_mask.contains(&i)
                || packed_ballots.is_some_and(|b| b.contains(&i))
                || m == "tb_capture_allocation_mask";

            if m == "v_nop" {
                // Header / rewritten no-op.
            } else if masks.ballot_word_reads.contains(&i) {
                if let Some(Opnd::Sgpr(d @ 0..=105)) = ins.dst.first() {
                    out.sgpr[*d as usize] = wave_res;
                }
            } else if is_ballot_def {
                if let [Opnd::Sgpr(lo @ 0..=104)] = ins.dst.as_slice() {
                    out.sgpr[*lo as usize] = wave_res;
                    out.sgpr[*lo as usize + 1] = wave_res;
                }
                if em.lockstep.as_ref().is_some_and(|v| v.contains_key(&i))
                    || masks.mask_votes.contains_key(&i)
                    || masks.ballot_logic_mask.contains(&i)
                {
                    out.scc = wave_res;
                }
            } else if is_wave_read {
                let sel_u = ins
                    .src
                    .get(1)
                    .map_or(Uniformity::Divergent, |o| incoming.opnd(o, false));
                let res_u = if sel_u >= Uniformity::Wave {
                    wave_res
                } else {
                    Uniformity::Divergent
                };
                if let Some(Opnd::Sgpr(d @ 0..=105)) = ins.dst.first() {
                    out.sgpr[*d as usize] = res_u;
                }
            } else if m == "v_readfirstlane_b32"
                || (ins.enc == Enc::Vop3 && ins.op == OP_WAVE_REDUCE)
                || m == "tb_first_exec_lane"
                || matches!(masks.mask_votes.get(&i), Some(MaskOp::FirstActive(_)))
            {
                if let Some(Opnd::Sgpr(d @ 0..=105)) = ins.dst.first() {
                    out.sgpr[*d as usize] = wave_res;
                }
            } else if m == COUNT_LANES
                || matches!(masks.mask_votes.get(&i), Some(MaskOp::CountBits(_)))
            {
                if let Some(Opnd::Sgpr(d @ 0..=105)) = ins.dst.first() {
                    out.sgpr[*d as usize] = wave_res;
                }
                out.scc = wave_res;
            } else if matches!(
                masks.mask_votes.get(&i),
                Some(
                    MaskOp::AnyTrue(_)
                        | MaskOp::NoneTrue(_)
                        | MaskOp::AllTrue(_)
                        | MaskOp::NotAllTrue(_)
                        | MaskOp::AllEqual(..)
                        | MaskOp::AnyDiff(..)
                        | MaskOp::AllEqualBallot(..)
                        | MaskOp::AnyDiffBallot(..)
                )
            ) {
                out.scc = wave_res;
            } else {
                match ins.enc {
                    Enc::Smem => {
                        let is_load = m.starts_with("s_load_") || m.starts_with("s_buffer_load_");
                        let base_width = if m.starts_with("s_buffer") { 4 } else { 2 };
                        let mut u = if is_load { ctl } else { Uniformity::Divergent };
                        for o in &ins.src {
                            match *o {
                                Opnd::Sgpr(r @ 0..=105) => {
                                    for k in 0..base_width {
                                        if (r as usize + k) < 106 {
                                            u = u.min(incoming.sgpr[r as usize + k]);
                                        } else {
                                            u = Uniformity::Divergent;
                                        }
                                    }
                                }
                                _ => u = u.min(incoming.opnd(o, false)),
                            }
                        }
                        if let Some(&w1) = ins.raw.get(1) {
                            let off = ((w1 >> 25) & 0x7f) as usize;
                            if off < 106 {
                                u = u.min(incoming.sgpr[off]);
                            } else if off != 125 {
                                u = Uniformity::Divergent;
                            }
                        } else {
                            u = Uniformity::Divergent;
                        }
                        for r in 0..106u8 {
                            if clobbers_scalar(ins, r) {
                                out.sgpr[r as usize] = u;
                            }
                        }
                    }
                    Enc::Sop1 | Enc::Sop2 | Enc::Sopk | Enc::Sopc => {
                        let wide = m.ends_with("_b64") || m.ends_with("_u64") || m.ends_with("_i64");
                        let reads_scc = matches!(
                            m,
                            "s_cselect_b32"
                                | "s_cselect_b64"
                                | "s_cmov_b32"
                                | "s_cmov_b64"
                                | "s_cmovk_i32"
                                | "s_addc_u32"
                                | "s_subb_u32"
                        );
                        let res_u = if m == "s_getpc_b64" {
                            ctl
                        } else if m.contains("exec")
                            || m.contains("wrexec")
                            || m.starts_with("s_wqm")
                            || m.starts_with("s_movrel")
                            || m.starts_with("s_getreg")
                        {
                            Uniformity::Divergent
                        } else {
                            let mut u = ctl;
                            for o in &ins.src {
                                u = u.min(incoming.opnd(o, wide));
                            }
                            if scalar_reads_dst(ins) {
                                for o in &ins.dst {
                                    u = u.min(incoming.opnd(o, wide));
                                }
                            }
                            if reads_scc {
                                u = u.min(incoming.scc);
                            }
                            u
                        };
                        for r in 0..106u8 {
                            if clobbers_scalar(ins, r) {
                                out.sgpr[r as usize] = res_u;
                            }
                        }
                        if em.lockstep.as_ref().is_some_and(|v| v.contains_key(&i))
                            || matches!(masks.mask_votes.get(&i), Some(MaskOp::PostAnyTrue(_)))
                        {
                            out.scc = wave_res;
                        } else if !scalar_preserves_scc(ins) {
                            out.scc = if scalar_writes_scc(ins) {
                                res_u
                            } else {
                                Uniformity::Divergent
                            };
                        }
                    }
                    Enc::Sopp => {}
                    _ => {
                        for r in 0..106u8 {
                            if clobbers_scalar(ins, r) {
                                out.sgpr[r as usize] = Uniformity::Divergent;
                            }
                        }
                    }
                }
            }

            let mut push = |next: usize| {
                if next >= n {
                    return;
                }
                let changed = match &mut states[next] {
                    Some(existing) => existing.join(&out),
                    slot => {
                        *slot = Some(out.clone());
                        true
                    }
                };
                if changed {
                    work.push_back(next);
                }
            };
            if let Some(&t) = cfg.target.get(&i).or_else(|| cfg.back.get(&i)) {
                push(t);
                if ins.enc == Enc::Sopp && ins.op == 2 {
                    continue;
                }
            }
            if m != "s_endpgm" {
                push(i + 1);
            }
        }

        if timed_out {
            return (vec![UniformState::all_divergent(); n], vec![Uniformity::Divergent; n]);
        }
        let finished: Vec<UniformState> = states
            .into_iter()
            .map(|s| s.unwrap_or_else(UniformState::all_divergent))
            .collect();
        let mut next_ctl = ctl_level.clone();
        for (&from, &to) in &cfg.target {
            let bu = branch_uniformity(em, from, &finished[from], ctl_level[from], wave_level);
            for slot in next_ctl.iter_mut().take(to.min(n)).skip(from + 1) {
                *slot = (*slot).min(bu);
            }
        }
        for &(l, b) in &cfg.loops {
            let lu = loop_uniformity(em, cfg, l, b, &finished, &ctl_level, wave_level);
            for slot in &mut next_ctl[l..=b] {
                *slot = (*slot).min(lu);
            }
        }
        if next_ctl == ctl_level {
            return (finished, ctl_level);
        }
        ctl_level = next_ctl;
    }
}

fn refuse(required_wave_size: Option<u32>, msg: String) -> Analysis {
    if required_wave_size == Some(64) {
        Analysis::KeepNative
    } else {
        Analysis::RefuseDivergent(msg)
    }
}

/// Analyze `em` for portable multi-wave Wave64 lowering.
pub(in crate::hlsl) fn analyze(
    em: &Emitter<'_>,
    required_wave_size: Option<u32>,
    scan_width: Option<u32>,
    shared_mask: bool,
) -> Analysis {
    if em.ex.stage != Stage::Compute {
        return Analysis::NotNeeded;
    }
    let group = compute_group(em.ex);
    let total = group.into_iter().try_fold(1u32, u32::checked_mul).unwrap_or(0);
    if total == 0 || total > 1024 || total % 64 != 0 {
        return if required_wave_size == Some(64) {
            Analysis::KeepNative
        } else {
            Analysis::NotNeeded
        };
    }

    let Some(cfg) = build_cfg(&em.insns) else {
        return if required_wave_size == Some(64) {
            Analysis::KeepNative
        } else {
            Analysis::NotNeeded
        };
    };

    let masks = analyze_masks(em, &cfg);
    let has_wave_reduce = em
        .insns
        .iter()
        .any(|i| i.enc == Enc::Vop3 && i.op == OP_WAVE_REDUCE);
    if !(required_wave_size == Some(64) || masks.has_general_ops || has_wave_reduce) {
        return Analysis::NotNeeded;
    }
    if masks.conflict {
        return refuse(
            required_wave_size,
            "wave64 ballot/mask representation conflict across control flow".into(),
        );
    }

    let mut barrier_sites = BTreeSet::new();
    let mut hoistable_sites = BTreeSet::new();
    let mut reduction_permlane = BTreeSet::new();
    let mut uses_ballot = masks.needs_ballot;
    let mut uses_exchange = false;
    let mut uses_extrema_f32 = false;

    for (_, region) in wave_sum::reduction_regions(&em.insns, 64) {
        for idx in region {
            if em.insns[idx].mnemonic == "v_permlanex16_b32"
                && em.insns[idx].src[1..] == [Opnd::ConstInt(0), Opnd::ConstInt(0)]
            {
                reduction_permlane.insert(idx);
            }
        }
    }

    let packed_ballots = em.packed.as_ref().map(|p| &p.ballots);
    let scan64 = scan_width == Some(64);
    for (i, ins) in em.insns.iter().enumerate() {
        let m = ins.mnemonic.as_str();
        let is_ballot_cmp = masks.ballot_compares.contains(&i)
            || packed_ballots.is_some_and(|b| b.contains(&i));
        let is_ballot_exec = masks.ballot_mask_copies.contains(&i)
            || masks.ballot_word_reads.contains(&i)
            || masks.ballot_logic_mask.contains(&i)
            || matches!(
                m,
                "tb_capture_allocation_mask" | "tb_allocation_prefix_lo" | "tb_allocation_prefix_hi"
            );
        let is_mbcnt_ballot = matches!(m, "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32")
            && matches!(
                ins.src.first(),
                Some(Opnd::ExecLo | Opnd::ExecHi | Opnd::VccLo | Opnd::VccHi)
            );
        let is_wqm = m == "s_wqm_b64" && ins.src == [Opnd::ExecLo];
        let is_vote_def = em.vote_defs.contains(&i) && !is_ballot_cmp;
        let is_lockstep_vote = em.lockstep.as_ref().is_some_and(|v| v.contains_key(&i));
        let is_mask_vote = masks.mask_votes.contains_key(&i);
        let is_voted_branch = ins.enc == Enc::Sopp
            && ((ins.op == 8 && em.reduction_execz_guards.contains(&i))
                || (matches!(ins.op, 6..=9) && em.lockstep.is_some()));
        let is_dpp = ins.enc == Enc::Vop2 && ins.op == wave_sum::DPP_REDUCE;
        let is_readlane = (ins.enc == Enc::Vop3 && ins.op == wave_sum::READ_LANE) || m == GROUP_READ;
        let is_readfirst = m == "v_readfirstlane_b32";
        let is_permlane = m == "v_permlanex16_b32";
        let is_scan_swizzle = scan64 && m == "ds_swizzle_b32";
        let is_extrema = ins.enc == Enc::Vop3 && ins.op == OP_WAVE_REDUCE;

        if is_ballot_cmp
            || is_ballot_exec
            || is_mbcnt_ballot
            || is_wqm
            || is_vote_def
            || is_lockstep_vote
            || is_mask_vote
            || is_voted_branch
        {
            uses_ballot = true;
            barrier_sites.insert(i);
            hoistable_sites.insert(i);
        }
        if is_dpp || is_readlane || is_permlane || is_scan_swizzle {
            uses_exchange = true;
            barrier_sites.insert(i);
            hoistable_sites.insert(i);
            if is_permlane && !reduction_permlane.contains(&i) {
                uses_ballot = true;
            }
        }
        if is_readfirst {
            uses_ballot = true;
            uses_exchange = true;
            barrier_sites.insert(i);
            hoistable_sites.insert(i);
        }
        if is_extrema {
            uses_extrema_f32 = true;
            barrier_sites.insert(i);
            hoistable_sites.insert(i);
        }
        if matches!(m, "s_barrier" | "tb_indexed_readlane" | "tb_first_exec_lane")
            || (em.compute_swizzle && (m == "ds_swizzle_b32" || m == GROUP_READ))
        {
            barrier_sites.insert(i);
        }
    }

    if let Some(&last_barrier) = barrier_sites.iter().next_back() {
        if em.insns[..last_barrier].iter().any(|i| i.mnemonic == "s_endpgm") {
            return refuse(
                required_wave_size,
                "wave64 ballot or vote follows an early exit".into(),
            );
        }
    }

    let waves = total / 64;
    let single_wave = waves == 1;
    let wave_level = if single_wave { Uniformity::Group } else { Uniformity::Wave };
    let (states, ctl_level) = analyze_uniformity(em, &cfg, &masks, single_wave);

    let mut hoisted_ops = BTreeSet::new();
    for &idx in &barrier_sites {
        if ctl_level[idx] < Uniformity::Wave {
            return refuse(
                required_wave_size,
                format!(
                    "wave64 ballot/vote/exchange at {:#x} executes in lane-divergent control flow",
                    em.insns[idx].offset
                ),
            );
        }
        for &(l, b) in &cfg.loops {
            if !(l <= idx && idx <= b) {
                continue;
            }
            if loop_uniformity(em, &cfg, l, b, &states, &ctl_level, wave_level) != Uniformity::Group
            {
                return refuse(
                    required_wave_size,
                    format!(
                        "wave64 ballot/vote/exchange at {:#x} lies inside a non-uniform loop",
                        em.insns[idx].offset
                    ),
                );
            }
            if cfg.loops.iter().any(|&(pl, pb)| {
                (pl, pb) != (l, b)
                    && pb <= b
                    && loop_uniformity(em, &cfg, pl, pb, &states, &ctl_level, wave_level)
                        != Uniformity::Group
            }) {
                return refuse(
                    required_wave_size,
                    format!(
                        "wave64 loop at {:#x} shares guard with a non-uniform loop",
                        em.insns[idx].offset
                    ),
                );
            }
        }
        let needs_hoist = cfg.target.iter().any(|(&from, &to)| {
            from < idx
                && to > idx
                && branch_uniformity(em, from, &states[from], ctl_level[from], wave_level)
                    < Uniformity::Group
        });
        if needs_hoist {
            if hoistable_sites.contains(&idx) {
                hoisted_ops.insert(idx);
            } else if matches!(
                em.insns[idx].mnemonic.as_str(),
                "tb_indexed_readlane" | "tb_first_exec_lane"
            ) || (em.compute_swizzle
                && matches!(em.insns[idx].mnemonic.as_str(), "ds_swizzle_b32" | GROUP_READ))
            {
                // Already hoisted outside `active` by `control_flow.rs`.
            } else {
                return refuse(
                    required_wave_size,
                    format!(
                        "non-hoistable barrier at {:#x} under wave-varying branch",
                        em.insns[idx].offset
                    ),
                );
            }
        }
    }

    let plan = Plan {
        waves,
        needs_tb_lane: group != [64, 1, 1],
        uses_ballot,
        uses_exchange,
        uses_extrema_f32,
        ballot_compares: masks.ballot_compares,
        ballot_mask_copies: masks.ballot_mask_copies,
        ballot_word_reads: masks.ballot_word_reads,
        ballot_logic_mask: masks.ballot_logic_mask,
        ballot_mask_reads: masks.ballot_mask_reads,
        mask_votes: masks.mask_votes,
        hoisted_ops,
        reduction_permlane,
    };

    let uses_guest_lds = em
        .insns
        .iter()
        .any(|i| i.enc == Enc::Ds && i.mnemonic != "ds_swizzle_b32");
    let lds_bytes = if uses_guest_lds {
        let reflected = em
            .reflected_lds_words
            .map(|w| w * 4)
            .or_else(|| tuxbox_dxbc::fssm::compute_lds_bytes(&em.ex.fssm));
        match reflected {
            Some(b) if b > 0 => b,
            _ => {
                return refuse(
                    required_wave_size,
                    "portable wave64 scratch exceeds unknown guest LDS budget".into(),
                );
            }
        }
    } else {
        0
    };
    let swizzle_bytes = if em.compute_swizzle || em.indexed_lane { total * 4 } else { 0 };
    let snapshot_bytes = if shared_mask { waves * 2 * 4 } else { 0 };
    if lds_bytes
        .saturating_add(swizzle_bytes)
        .saturating_add(snapshot_bytes)
        .saturating_add(plan.scratch_bytes())
        > 32768
    {
        return refuse(
            required_wave_size,
            "portable wave64 scratch exceeds 32 KiB groupshared budget".into(),
        );
    }

    Analysis::Portable(plan)
}
