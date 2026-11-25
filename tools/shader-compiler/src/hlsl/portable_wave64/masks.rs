//! Structured CFG construction and SGPR mask vs. 64-bit ballot provenance tracking.

use super::super::{branch_pc, reads_sgpr, Emitter, Enc, Insn, Opnd, COUNT_LANES};
use super::uniformity::scalar_writes_scc;
use super::MaskOp;
use crate::rdna2::clobbers_scalar;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

pub(super) struct Cfg {
    pub(super) target: HashMap<usize, usize>,
    pub(super) back: HashMap<usize, usize>,
    pub(super) loops: Vec<(usize, usize)>,
}

pub(super) fn build_cfg(insns: &[Insn]) -> Option<Cfg> {
    if insns.iter().any(|op| {
        matches!(
            op.mnemonic.as_str(),
            "s_setpc_b64" | "s_swappc_b64" | "s_call_b64" | "s_rfe_b64" | "s_trap"
        ) || (op.mnemonic.contains("branch")
            && !(op.enc == Enc::Sopp && Emitter::branch_cond(op.op).is_some()))
    }) {
        return None;
    }
    let n = insns.len();
    let mut off2idx: HashMap<usize, usize> =
        insns.iter().enumerate().map(|(i, x)| (x.offset, i)).collect();
    if let Some(last) = insns.last() {
        off2idx.insert(last.offset + last.dwords * 4, n);
    }
    let prog_end = insns.last().map_or(0, |l| l.offset + l.dwords * 4);
    let mut target = HashMap::new();
    let mut back = HashMap::new();
    for (i, ins) in insns.iter().enumerate() {
        if ins.enc == Enc::Sopp && ins.op == 1 && i + 1 < n {
            target.insert(i, n);
            continue;
        }
        if ins.enc == Enc::Sopp && Emitter::branch_cond(ins.op).is_some() {
            let t = branch_pc(ins);
            let ti = if t >= 0 && t as usize >= prog_end {
                n
            } else {
                *off2idx.get(&(t as usize))?
            };
            if ti <= i {
                back.insert(i, ti);
            } else {
                target.insert(i, ti);
            }
        }
    }
    let mut heads: HashMap<usize, usize> = HashMap::new();
    for (&b, &l) in &back {
        let e = heads.entry(l).or_insert(b);
        *e = (*e).max(b);
    }
    let continues: Vec<_> = back
        .iter()
        .filter(|(&b, &l)| heads[&l] != b)
        .map(|(&b, &l)| (b, heads[&l]))
        .collect();
    for (b, closing) in continues {
        back.remove(&b);
        target.insert(b, closing);
    }
    let mut loops: Vec<(usize, usize)> = heads.into_iter().collect();
    loops.sort_unstable();
    for &(l1, b1) in &loops {
        for &(l2, b2) in &loops {
            let disjoint = b1 < l2 || b2 < l1;
            let nested = (l1 <= l2 && b2 <= b1) || (l2 <= l1 && b1 <= b2);
            if !(disjoint || nested) {
                return None; // Interleaved loops fall back to block dispatch.
            }
        }
    }
    Some(Cfg { target, back, loops })
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum RegOrigin {
    Scalar,
    Mask { roots: BTreeSet<usize>, promotable: bool },
    Ballot64,
    Conflict,
}

impl RegOrigin {
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Conflict, _) | (_, Self::Conflict) => Self::Conflict,
            (Self::Scalar, Self::Scalar) => Self::Scalar,
            (Self::Ballot64, Self::Ballot64)
            | (Self::Ballot64, Self::Scalar)
            | (Self::Scalar, Self::Ballot64) => Self::Ballot64,
            (
                Self::Mask { roots: a, promotable: pa },
                Self::Mask { roots: b, promotable: pb },
            ) => {
                let mut roots = a.clone();
                roots.extend(b.iter().copied());
                Self::Mask { roots, promotable: *pa && *pb }
            }
            _ => Self::Conflict,
        }
    }
}

pub(super) fn is_64bit_logic(m: &str) -> bool {
    matches!(
        m,
        "s_and_b64"
            | "s_or_b64"
            | "s_xor_b64"
            | "s_andn2_b64"
            | "s_orn2_b64"
            | "s_nand_b64"
            | "s_nor_b64"
            | "s_xnor_b64"
            | "s_not_b64"
    )
}

fn is_mask_const(o: &Opnd) -> bool {
    matches!(*o, Opnd::ConstInt(0 | -1) | Opnd::Literal(0 | u32::MAX))
}

fn is_ballot_word_mov(ins: &Insn) -> bool {
    matches!(ins.mnemonic.as_str(), "v_mov_b32" | "s_mov_b32")
        && matches!(ins.dst.first(), Some(Opnd::Sgpr(_) | Opnd::Vgpr(_)))
        && matches!(
            ins.src.first(),
            Some(Opnd::ExecLo | Opnd::ExecHi | Opnd::VccLo | Opnd::VccHi)
        )
}

fn scc_read_after(insns: &[Insn], cfg: &Cfg, start: usize) -> bool {
    let mut visited = vec![false; insns.len()];
    let mut work = VecDeque::new();
    let push = |idx: usize, visited: &mut [bool], work: &mut VecDeque<usize>| {
        if idx < insns.len() && !visited[idx] {
            visited[idx] = true;
            work.push_back(idx);
        }
    };
    if let Some(&t) = cfg.target.get(&start).or_else(|| cfg.back.get(&start)) {
        push(t, &mut visited, &mut work);
    }
    if !(insns[start].enc == Enc::Sopp && matches!(insns[start].op, 1 | 2)) {
        push(start + 1, &mut visited, &mut work);
    }
    while let Some(i) = work.pop_front() {
        let ins = &insns[i];
        let m = ins.mnemonic.as_str();
        if (ins.enc == Enc::Sopp && matches!(ins.op, 4 | 5))
            || ins.src.contains(&Opnd::Scc)
            || matches!(
                m,
                "s_cselect_b32"
                    | "s_cselect_b64"
                    | "s_cmov_b32"
                    | "s_cmov_b64"
                    | "s_cmovk_i32"
                    | "s_addc_u32"
                    | "s_subb_u32"
            )
        {
            return true;
        }
        if scalar_writes_scc(ins)
            || m.contains("saveexec")
            || m.contains("wrexec")
            || m.starts_with("s_wqm")
        {
            continue;
        }
        if let Some(&t) = cfg.target.get(&i).or_else(|| cfg.back.get(&i)) {
            push(t, &mut visited, &mut work);
            if ins.enc == Enc::Sopp && ins.op == 2 {
                continue;
            }
        }
        if m != "s_endpgm" {
            push(i + 1, &mut visited, &mut work);
        }
    }
    false
}

/// Result of tracking SGPR mask vs. 64-bit ballot provenance across the CFG.
pub(super) struct MaskPlan {
    pub(super) ballot_compares: BTreeSet<usize>,
    pub(super) ballot_mask_copies: BTreeSet<usize>,
    pub(super) ballot_word_reads: BTreeSet<usize>,
    pub(super) ballot_logic_mask: BTreeSet<usize>,
    pub(super) ballot_mask_reads: BTreeSet<usize>,
    pub(super) mask_votes: BTreeMap<usize, MaskOp>,
    pub(super) needs_ballot: bool,
    pub(super) has_general_ops: bool,
    pub(super) conflict: bool,
}

pub(super) fn analyze_masks(em: &Emitter<'_>, cfg: &Cfg) -> MaskPlan {
    let insns = &em.insns;
    let n = insns.len();
    let packed_ballots = em.packed.as_ref().map(|p| &p.ballots);
    let mut promoted_compares = BTreeSet::new();
    let mut promoted_copies = BTreeSet::new();
    let mut promoted_logic = BTreeSet::new();
    let mut conflict = false;

    // Iteratively promote `v_cmp_*` / `exec|vcc`-copy SGPR pairs to full 64-bit ballots
    // whenever any ballot-data consumer observes them.
    let states = loop {
        let mut states: Vec<Option<Vec<RegOrigin>>> = vec![None; n];
        if n > 0 {
            states[0] = Some(vec![RegOrigin::Scalar; 106]);
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

            let pair_origin = |opnd: &Opnd| -> RegOrigin {
                match *opnd {
                    Opnd::Sgpr(r @ 0..=104) if r % 2 == 0 => {
                        incoming[r as usize].join(&incoming[r as usize + 1])
                    }
                    Opnd::Sgpr(r @ 0..=105) => incoming[r as usize].clone(),
                    Opnd::ExecLo | Opnd::ExecHi | Opnd::VccLo | Opnd::VccHi => RegOrigin::Mask {
                        roots: BTreeSet::new(),
                        promotable: false,
                    },
                    _ => RegOrigin::Scalar,
                }
            };

            let is_ballot_cmp = promoted_compares.contains(&i)
                || packed_ballots.is_some_and(|b| b.contains(&i));
            let is_ballot_copy =
                promoted_copies.contains(&i) || m == "tb_capture_allocation_mask";
            let is_ballot_logic = promoted_logic.contains(&i);

            let mut handled = false;
            if is_ballot_cmp || is_ballot_copy || is_ballot_logic {
                if let [Opnd::Sgpr(lo @ 0..=104)] = ins.dst.as_slice() {
                    if lo % 2 == 0 {
                        out[*lo as usize] = RegOrigin::Ballot64;
                        out[*lo as usize + 1] = RegOrigin::Ballot64;
                        handled = true;
                    }
                }
            } else if m.starts_with("v_cmp_") && !m.starts_with("v_cmpx_") && ins.aux.sdwa.is_none()
            {
                if let [Opnd::Sgpr(lo @ 0..=104)] = ins.dst.as_slice() {
                    if lo % 2 == 0 {
                        let o = RegOrigin::Mask {
                            roots: BTreeSet::from([i]),
                            promotable: true,
                        };
                        out[*lo as usize] = o.clone();
                        out[*lo as usize + 1] = o;
                        handled = true;
                    }
                }
            } else if ((m == "s_mov_b64"
                && matches!(ins.src.as_slice(), [Opnd::ExecLo | Opnd::VccLo]))
                || m.contains("saveexec_b64"))
                && !em.exec_snapshots.sites.contains_key(&i)
            {
                if let [Opnd::Sgpr(lo @ 0..=104)] = ins.dst.as_slice() {
                    if lo % 2 == 0 {
                        let o = RegOrigin::Mask {
                            roots: BTreeSet::from([i]),
                            promotable: true,
                        };
                        out[*lo as usize] = o.clone();
                        out[*lo as usize + 1] = o;
                        handled = true;
                    }
                }
            } else if m == "s_mov_b64" || m == "s_cselect_b64" || is_64bit_logic(m) {
                if let [Opnd::Sgpr(lo @ 0..=104)] = ins.dst.as_slice() {
                    if lo % 2 == 0 {
                        let any_ballot = ins
                            .src
                            .iter()
                            .any(|o| matches!(pair_origin(o), RegOrigin::Ballot64));
                        let mut combined: Option<RegOrigin> = None;
                        for src in &ins.src {
                            if is_mask_const(src) {
                                continue;
                            }
                            let o = if is_64bit_logic(m)
                                && matches!(src, Opnd::ExecLo | Opnd::VccLo)
                            {
                                if any_ballot {
                                    RegOrigin::Ballot64
                                } else {
                                    RegOrigin::Mask {
                                        roots: BTreeSet::from([i]),
                                        promotable: true,
                                    }
                                }
                            } else {
                                pair_origin(src)
                            };
                            combined = Some(match combined {
                                None => o,
                                Some(prev) => prev.join(&o),
                            });
                        }
                        let res = combined.unwrap_or(RegOrigin::Scalar);
                        out[*lo as usize] = res.clone();
                        out[*lo as usize + 1] = res;
                        handled = true;
                    }
                }
            }

            if !handled && m != "v_nop" {
                for r in 0..106u8 {
                    if clobbers_scalar(ins, r) {
                        out[r as usize] = RegOrigin::Scalar;
                    }
                }
            }

            let mut push = |next: usize| {
                if next >= n {
                    return;
                }
                let changed = match &mut states[next] {
                    Some(existing) => {
                        let mut diff = false;
                        for (e, o) in existing.iter_mut().zip(&out) {
                            let joined = e.join(o);
                            if joined != *e {
                                *e = joined;
                                diff = true;
                            }
                        }
                        diff
                    }
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
            conflict = true;
            break Vec::new();
        }

        let mut new_cmp = BTreeSet::new();
        let mut new_copy = BTreeSet::new();
        let mut new_logic = BTreeSet::new();
        let mut unpromotable = false;

        for (i, ins) in insns.iter().enumerate() {
            let Some(st) = states[i].as_ref() else { continue };
            let m = ins.mnemonic.as_str();
            if m == "v_nop" || m == COUNT_LANES || em.exec_snapshots.sites.contains_key(&i) {
                continue;
            }
            let mut data_reads: Vec<u8> = Vec::new();
            match m {
                "s_bcnt1_i32_b64" | "s_ff1_i32_b64" | "s_ff0_i32_b64" | "s_flbit_i32_b64" => {
                    if let Some(Opnd::Sgpr(lo @ 0..=104)) = ins.src.first() {
                        data_reads.extend([*lo, *lo + 1]);
                    }
                }
                "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                    if let Some(Opnd::Sgpr(reg @ 0..=105)) = ins.src.first() {
                        data_reads.push(*reg);
                    }
                }
                "s_cmp_eq_u64" | "s_cmp_lg_u64" => {
                    let vote_pair = matches!(ins.src.as_slice(), [_, b] if is_mask_const(b))
                        || matches!(ins.src.as_slice(), [a, _] if is_mask_const(a))
                        || ins.src.iter().all(|o| match o {
                            Opnd::ExecLo | Opnd::VccLo => true,
                            Opnd::Sgpr(lo @ 0..=104) if lo % 2 == 0 => matches!(
                                st[*lo as usize],
                                RegOrigin::Mask { .. } | RegOrigin::Ballot64
                            ),
                            _ => false,
                        });
                    if !vote_pair {
                        for src in &ins.src {
                            if let Opnd::Sgpr(lo @ 0..=104) = *src {
                                data_reads.extend([lo, lo + 1]);
                            }
                        }
                    }
                }
                _ if is_64bit_logic(m) => {
                    // If this op is promoted or combines with Ballot64/Scalar, promote SGPR Mask siblings.
                    let force = promoted_logic.contains(&i)
                        || ins.src.iter().any(|o| {
                            !is_mask_const(o)
                                && matches!(*o, Opnd::Sgpr(lo @ 0..=104) if lo % 2 == 0
                                    && matches!(st[lo as usize], RegOrigin::Ballot64 | RegOrigin::Scalar)
                                    && matches!(st[lo as usize + 1], RegOrigin::Ballot64 | RegOrigin::Scalar))
                        });
                    if force {
                        for src in &ins.src {
                            if let Opnd::Sgpr(lo @ 0..=104) = *src {
                                if matches!(st[lo as usize], RegOrigin::Mask { .. }) {
                                    data_reads.extend([lo, lo + 1]);
                                }
                            }
                        }
                    }
                }
                _ if m == "s_mov_b64"
                    || m == "s_cselect_b64"
                    || m.contains("saveexec_b64")
                    || m.contains("wrexec_b64") => {}
                "v_cndmask_b32" => {
                    for src in ins.src.iter().take(2) {
                        if let Opnd::Sgpr(r @ 0..=105) = *src {
                            data_reads.push(r);
                        }
                    }
                }
                _ => {
                    for r in 0..106u8 {
                        if reads_sgpr(ins, r) {
                            data_reads.push(r);
                        }
                    }
                }
            }
            for reg in data_reads {
                match &st[reg as usize] {
                    RegOrigin::Mask { roots, promotable } => {
                        if !promotable || roots.is_empty() {
                            unpromotable = true;
                        } else {
                            for &root in roots {
                                let rm = insns[root].mnemonic.as_str();
                                if rm.starts_with("v_cmp_") {
                                    new_cmp.insert(root);
                                } else if is_64bit_logic(rm) {
                                    new_logic.insert(root);
                                } else {
                                    new_copy.insert(root);
                                }
                            }
                        }
                    }
                    RegOrigin::Conflict => unpromotable = true,
                    RegOrigin::Scalar | RegOrigin::Ballot64 => {}
                }
            }
        }

        if unpromotable {
            conflict = true;
            break states;
        }
        let added = new_cmp.iter().any(|x| !promoted_compares.contains(x))
            || new_copy.iter().any(|x| !promoted_copies.contains(x))
            || new_logic.iter().any(|x| !promoted_logic.contains(x));
        if !added {
            break states;
        }
        promoted_compares.extend(new_cmp);
        promoted_copies.extend(new_copy);
        promoted_logic.extend(new_logic);
    };

    let mut ballot_word_reads = BTreeSet::new();
    let mut ballot_logic_mask = BTreeSet::new();
    let mut ballot_mask_reads = BTreeSet::new();
    let mut mask_votes = BTreeMap::new();
    let mut needs_ballot = !promoted_compares.is_empty()
        || !promoted_copies.is_empty()
        || packed_ballots.is_some_and(|b| !b.is_empty())
        || !em.vote_uses.is_empty()
        || !em.reduction_execz_guards.is_empty()
        || em.lockstep.is_some();
    let mut has_general_ops = !promoted_compares.is_empty() || !promoted_copies.is_empty();

    if !conflict {
        for (i, ins) in insns.iter().enumerate() {
            let Some(st) = states[i].as_ref() else { continue };
            let m = ins.mnemonic.as_str();
            if em.exec_snapshots.sites.contains_key(&i) {
                continue;
            }
            let is_ballot_pair = |op: &Opnd| -> bool {
                matches!(*op, Opnd::Sgpr(lo @ 0..=104) if lo % 2 == 0
                    && st[lo as usize] == RegOrigin::Ballot64
                    && st[lo as usize + 1] == RegOrigin::Ballot64)
            };
            let is_mask_opnd = |op: &Opnd| -> bool {
                match *op {
                    Opnd::ExecLo | Opnd::VccLo => true,
                    Opnd::Sgpr(lo @ 0..=104) if lo % 2 == 0 => {
                        matches!(st[lo as usize], RegOrigin::Mask { .. })
                    }
                    _ => false,
                }
            };
            let has_conflict = |op: &Opnd, pair: bool| -> bool {
                match *op {
                    Opnd::Sgpr(r @ 0..=105) => {
                        st[r as usize] == RegOrigin::Conflict
                            || (pair && r < 105 && st[r as usize + 1] == RegOrigin::Conflict)
                    }
                    _ => false,
                }
            };

            if is_ballot_word_mov(ins) {
                ballot_word_reads.insert(i);
                needs_ballot = true;
                has_general_ops = true;
            } else if m == "v_cndmask_b32" && ins.enc == Enc::Vop3 {
                if let Some(src2) = ins.src.get(2) {
                    if has_conflict(src2, true) {
                        conflict = true;
                    } else if is_ballot_pair(src2) {
                        ballot_mask_reads.insert(i);
                    }
                }
            } else if m == "s_mov_b64" && ins.dst == [Opnd::ExecLo] {
                if let Some(src0) = ins.src.first() {
                    if has_conflict(src0, true) {
                        conflict = true;
                    } else if is_ballot_pair(src0) {
                        ballot_mask_reads.insert(i);
                    }
                }
            } else if m.contains("saveexec_b64") {
                if let Some(src0) = ins.src.first() {
                    if has_conflict(src0, true) {
                        conflict = true;
                    } else if is_ballot_pair(src0) {
                        ballot_mask_reads.insert(i);
                    }
                }
                if em.lockstep.is_none() && scc_read_after(insns, cfg, i) {
                    mask_votes.insert(i, MaskOp::PostAnyTrue(Opnd::ExecLo));
                    needs_ballot = true;
                    has_general_ops = true;
                }
            } else if is_64bit_logic(m) {
                let any_ballot = ins.src.iter().any(is_ballot_pair);
                let any_exec_vcc = ins
                    .src
                    .iter()
                    .any(|o| matches!(*o, Opnd::ExecLo | Opnd::VccLo));
                let any_sgpr_mask = ins.src.iter().any(|o| {
                    matches!(*o, Opnd::Sgpr(lo @ 0..=104) if lo % 2 == 0
                        && matches!(st[lo as usize], RegOrigin::Mask { .. }))
                });
                if any_ballot && any_sgpr_mask {
                    conflict = true;
                } else if (any_ballot || promoted_logic.contains(&i)) && any_exec_vcc {
                    ballot_logic_mask.insert(i);
                    needs_ballot = true;
                    has_general_ops = true;
                } else if (any_exec_vcc || any_sgpr_mask)
                    && em.lockstep.is_none()
                    && scc_read_after(insns, cfg, i)
                {
                    if let Some(&dst) = ins.dst.first() {
                        mask_votes.insert(i, MaskOp::PostAnyTrue(dst));
                        needs_ballot = true;
                        has_general_ops = true;
                    }
                }
            } else if matches!(m, "s_cmp_eq_u64" | "s_cmp_lg_u64") && ins.src.len() == 2 {
                let eq = m == "s_cmp_eq_u64";
                let (a, b) = (ins.src[0], ins.src[1]);
                let (am, bm) = (is_mask_opnd(&a), is_mask_opnd(&b));
                let (ab, bb) = (is_ballot_pair(&a), is_ballot_pair(&b));
                if has_conflict(&a, true) || has_conflict(&b, true) {
                    conflict = true;
                } else if am && bm {
                    let op = if eq { MaskOp::AllEqual(a, b) } else { MaskOp::AnyDiff(a, b) };
                    mask_votes.insert(i, op);
                    needs_ballot = true;
                    has_general_ops = true;
                } else if (ab && bm) || (am && bb) {
                    let (bal, msk) = if ab { (a, b) } else { (b, a) };
                    let op = if eq {
                        MaskOp::AllEqualBallot(bal, msk)
                    } else {
                        MaskOp::AnyDiffBallot(bal, msk)
                    };
                    mask_votes.insert(i, op);
                    needs_ballot = true;
                    has_general_ops = true;
                } else if am || bm {
                    let (mask, other) = if am { (a, b) } else { (b, a) };
                    let op = match (eq, other) {
                        (true, Opnd::ConstInt(0) | Opnd::Literal(0)) => Some(MaskOp::NoneTrue(mask)),
                        (false, Opnd::ConstInt(0) | Opnd::Literal(0)) => Some(MaskOp::AnyTrue(mask)),
                        (true, Opnd::ConstInt(-1) | Opnd::Literal(u32::MAX)) => Some(MaskOp::AllTrue(mask)),
                        (false, Opnd::ConstInt(-1) | Opnd::Literal(u32::MAX)) => Some(MaskOp::NotAllTrue(mask)),
                        _ => None,
                    };
                    if let Some(op) = op {
                        mask_votes.insert(i, op);
                        needs_ballot = true;
                        has_general_ops = true;
                    } else {
                        conflict = true;
                    }
                }
            } else if m == "s_bcnt1_i32_b64" {
                if let Some(&src0 @ (Opnd::ExecLo | Opnd::VccLo)) = ins.src.first() {
                    mask_votes.insert(i, MaskOp::CountBits(src0));
                    needs_ballot = true;
                    has_general_ops = true;
                }
            } else if m == "s_ff1_i32_b64" {
                if let Some(&src0 @ (Opnd::ExecLo | Opnd::VccLo)) = ins.src.first() {
                    mask_votes.insert(i, MaskOp::FirstActive(src0));
                    needs_ballot = true;
                    has_general_ops = true;
                }
            } else if matches!(m, "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32") {
                if let Some(Opnd::ExecLo | Opnd::ExecHi | Opnd::VccLo | Opnd::VccHi) =
                    ins.src.first()
                {
                    needs_ballot = true;
                    has_general_ops = true;
                }
            } else if matches!(
                m,
                "tb_capture_allocation_mask" | "tb_allocation_prefix_lo" | "tb_allocation_prefix_hi"
            ) {
                needs_ballot = true;
                has_general_ops = true;
            } else if m == "s_wqm_b64" && ins.src == [Opnd::ExecLo] {
                needs_ballot = true;
                has_general_ops = true;
            } else if matches!(m, "v_readfirstlane_b32" | "v_permlanex16_b32") {
                needs_ballot = true;
            }
        }
    }

    MaskPlan {
        ballot_compares: promoted_compares,
        ballot_mask_copies: promoted_copies,
        ballot_word_reads,
        ballot_logic_mask,
        ballot_mask_reads,
        mask_votes,
        needs_ballot,
        has_general_ops,
        conflict,
    }
}
