//! Intra-function DXIL SSA IR optimization and module compaction pass suite.
//!
//! Runs after IR construction and before metadata/bitcode serialization:
//! 1. Fixpoint loop over CFG simplification & RPO normalization, trivial/duplicate
//!    PHI elimination, constant folding & algebraic/bitcast/select simplification,
//!    dominator-scoped CSE (excluding implicit derivatives and non-pure intrinsics),
//!    and iterative mark-and-sweep DCE.
//! 2. Dense compaction of live instructions, constants, declarations, and globals.
//! 3. Synchronization of [`Shader::features`] with the remaining live IR.
//!
//! All graph traversals use explicit stacks so large shaders cannot overflow.

use std::collections::{BTreeSet, HashMap};

use super::module::{BinOp, Block, CastOp, ConstDef, FnAttrs, MdDef, Module, Op, Pred, Ty, Value};
use super::shader::{kind, Feature, Shader};

const MAX_ROUNDS: usize = 8;

/// Summary metrics of a DXIL [`Module`] (for diagnostics and optimization tests).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub blocks: usize,
    pub insts: usize,
    pub phis: usize,
    pub consts: usize,
    pub funcs: usize,
    pub globals: usize,
}

/// Counts live blocks/instructions/PHIs and pooled constants/functions/globals.
pub fn module_stats(m: &Module) -> Stats {
    let mut insts = 0usize;
    let mut phis = 0usize;
    for b in &m.blocks {
        insts += b.len();
        for &id in b {
            if matches!(m.insts[id as usize].op, Op::Phi(..)) {
                phis += 1;
            }
        }
    }
    Stats {
        blocks: m.blocks.len(),
        insts,
        phis,
        consts: m.consts.len(),
        funcs: m.funcs.len(),
        globals: m.globals.len(),
    }
}

/// Optimizes `s.m`, compacts pooled module tables (updating `s.entry`), and
/// synchronizes `s.features` against the remaining live IR.
pub fn optimize(s: &mut Shader) {
    if s.m.error().is_some() || s.m.blocks.is_empty() {
        return;
    }
    optimize_core(&mut s.m);
    compact_module(&mut s.m, Some(&mut s.entry));
    sync_features(s);
}

/// Optimizes and compacts `m` without modifying wrapper shader metadata.
pub fn optimize_module(m: &mut Module) {
    if m.error().is_some() || m.blocks.is_empty() {
        return;
    }
    optimize_core(m);
    compact_module(m, None);
}

fn optimize_core(m: &mut Module) {
    for _ in 0..MAX_ROUNDS {
        let mut changed = false;
        changed |= simplify_cfg(m);
        changed |= simplify_phis(m);
        changed |= simplify_insts(m);
        changed |= cse(m);
        changed |= dce(m);
        if !changed {
            break;
        }
    }
    // Final normalization pass if the last DCE/CSE opened trivial CFG/PHI cleanup.
    if simplify_cfg(m) {
        let _ = simplify_phis(m);
        let _ = dce(m);
        let _ = simplify_cfg(m);
    }
}

// ------------------------------------------------------------------ helpers

fn mask_for(w: u32) -> u64 {
    if w >= 64 {
        u64::MAX
    } else if w == 0 {
        0
    } else {
        (1u64 << w) - 1
    }
}

fn sext_i64(v: u64, w: u32) -> i64 {
    if w >= 64 {
        v as i64
    } else if w == 0 {
        0
    } else {
        ((v << (64 - w)) as i64) >> (64 - w)
    }
}

fn const_int_u64(m: &Module, v: Value) -> Option<u64> {
    match v {
        Value::Const(c) => match &m.consts[c as usize] {
            ConstDef::Int(t, x) => {
                let w = m.int_width(*t)?;
                Some((*x as u64) & mask_for(w))
            }
            ConstDef::Null(t) if m.int_width(*t).is_some() => Some(0),
            _ => None,
        },
        _ => None,
    }
}

fn scalar_bits(m: &Module, ty: Ty) -> Option<u32> {
    if let Some(w) = m.int_width(ty) {
        return Some(w);
    }
    if m.is_float(ty) {
        return match m.alloc_size_layout(ty, true) {
            2 => Some(16),
            4 => Some(32),
            8 => Some(64),
            _ => None,
        };
    }
    None
}

fn const_bitcast(m: &mut Module, v: Value, dst: Ty) -> Option<Value> {
    if m.value_ty(v) == dst {
        return Some(v);
    }
    let Value::Const(c) = v else { return None };
    let dst_bits = scalar_bits(m, dst)?;
    let def = m.consts[c as usize].clone();
    match def {
        ConstDef::Undef(t) => (scalar_bits(m, t) == Some(dst_bits)).then(|| m.undef(dst)),
        ConstDef::Null(t) if scalar_bits(m, t) == Some(dst_bits) => {
            if m.int_width(dst).is_some() {
                Some(m.const_int(dst, 0))
            } else if m.is_float(dst) {
                Some(m.const_float(dst, 0))
            } else {
                None
            }
        }
        ConstDef::Int(t, x) => {
            let w = m.int_width(t)?;
            if w != dst_bits {
                return None;
            }
            let raw = (x as u64) & mask_for(w);
            if m.int_width(dst).is_some() {
                Some(m.const_int(dst, raw))
            } else if m.is_float(dst) {
                Some(m.const_float(dst, raw))
            } else {
                None
            }
        }
        ConstDef::Float(t, bits) => {
            let w = scalar_bits(m, t)?;
            if w != dst_bits {
                return None;
            }
            if m.int_width(dst).is_some() {
                Some(m.const_int(dst, bits))
            } else if m.is_float(dst) {
                Some(m.const_float(dst, bits))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_undef(m: &Module, v: Value) -> bool {
    matches!(v, Value::Const(c) if matches!(m.consts[c as usize], ConstDef::Undef(_)))
}

fn is_op_name(full: &str, base: &str) -> bool {
    full == base || full.strip_prefix(base).is_some_and(|s| s.starts_with('.'))
}

fn resolve_val(mut v: Value, repl: &[Option<Value>]) -> Value {
    let mut steps = 0usize;
    while let Value::Inst(id) = v {
        let Some(next) = repl.get(id as usize).copied().flatten() else { break };
        if next == v || steps >= repl.len() {
            break;
        }
        v = next;
        steps += 1;
    }
    v
}

fn visit_op_values_mut(op: &mut Op, f: &mut dyn FnMut(&mut Value)) {
    match op {
        Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::AtomicRmw(_, a, b) => {
            f(a);
            f(b);
        }
        Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) => f(v),
        Op::Store(p, v, _) => {
            f(p);
            f(v);
        }
        Op::Select(c, t, e) | Op::CmpXchg(c, t, e) => {
            f(c);
            f(t);
            f(e);
        }
        Op::Call(_, args) => {
            for a in args {
                f(a);
            }
        }
        Op::Phi(inc) => {
            for (v, _) in inc {
                f(v);
            }
        }
        Op::Gep(p, idx) => {
            f(p);
            for i in idx {
                f(i);
            }
        }
        Op::Alloca(_, n, _) => f(n),
        Op::CondBr(c, _, _) => f(c),
        Op::Switch(v, _, cases) => {
            f(v);
            for (cv, _) in cases {
                f(cv);
            }
        }
        Op::Br(_) | Op::Ret | Op::Unreachable => {}
    }
}

fn rewrite_op_values(op: &mut Op, repl: &[Option<Value>]) -> bool {
    let mut changed = false;
    visit_op_values_mut(op, &mut |v| {
        let nv = resolve_val(*v, repl);
        if nv != *v {
            *v = nv;
            changed = true;
        }
    });
    changed
}

fn rewrite_md_values(m: &mut Module, repl: &[Option<Value>]) {
    for md in &mut m.md {
        if let MdDef::Val(_, v) = md {
            *v = resolve_val(*v, repl);
        }
    }
}

fn for_each_op_value(op: &Op, mut f: impl FnMut(Value)) {
    match op {
        Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::AtomicRmw(_, a, b) => {
            f(*a);
            f(*b);
        }
        Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) => f(*v),
        Op::Store(p, v, _) => {
            f(*p);
            f(*v);
        }
        Op::Select(c, t, x) | Op::CmpXchg(c, t, x) => {
            f(*c);
            f(*t);
            f(*x);
        }
        Op::Call(_, args) => {
            for &a in args {
                f(a);
            }
        }
        Op::Phi(inc) => {
            for &(v, _) in inc {
                f(v);
            }
        }
        Op::Gep(p, idx) => {
            f(*p);
            for &i in idx {
                f(i);
            }
        }
        Op::Alloca(_, n, _) => f(*n),
        Op::CondBr(c, _, _) => f(*c),
        Op::Switch(v, _, cases) => {
            f(*v);
            for &(cv, _) in cases {
                f(cv);
            }
        }
        Op::Br(_) | Op::Ret | Op::Unreachable => {}
    }
}

fn block_succs(m: &Module, b: usize) -> Vec<usize> {
    let Some(&last) = m.blocks[b].last() else { return Vec::new() };
    match &m.insts[last as usize].op {
        Op::Br(t) => vec![t.0 as usize],
        Op::CondBr(_, t, f) => vec![t.0 as usize, f.0 as usize],
        Op::Switch(_, d, cases) => {
            let mut s = Vec::with_capacity(1 + cases.len());
            s.push(d.0 as usize);
            for &(_, c) in cases {
                s.push(c.0 as usize);
            }
            s
        }
        _ => Vec::new(),
    }
}

fn invert_pred(p: Pred) -> Pred {
    match p {
        Pred::Eq => Pred::Ne,
        Pred::Ne => Pred::Eq,
        Pred::UGt => Pred::ULe,
        Pred::UGe => Pred::ULt,
        Pred::ULt => Pred::UGe,
        Pred::ULe => Pred::UGt,
        Pred::SGt => Pred::SLe,
        Pred::SGe => Pred::SLt,
        Pred::SLt => Pred::SGe,
        Pred::SLe => Pred::SGt,
        Pred::FFalse => Pred::FTrue,
        Pred::FTrue => Pred::FFalse,
        Pred::FOEq => Pred::FUNe,
        Pred::FUNe => Pred::FOEq,
        Pred::FOGt => Pred::FULe,
        Pred::FULe => Pred::FOGt,
        Pred::FOGe => Pred::FULt,
        Pred::FULt => Pred::FOGe,
        Pred::FOLt => Pred::FUGe,
        Pred::FUGe => Pred::FOLt,
        Pred::FOLe => Pred::FUGt,
        Pred::FUGt => Pred::FOLe,
        Pred::FONe => Pred::FUEq,
        Pred::FUEq => Pred::FONe,
        Pred::FOrd => Pred::FUno,
        Pred::FUno => Pred::FOrd,
    }
}

fn swap_pred(p: Pred) -> Pred {
    match p {
        Pred::UGt => Pred::ULt,
        Pred::ULt => Pred::UGt,
        Pred::UGe => Pred::ULe,
        Pred::ULe => Pred::UGe,
        Pred::SGt => Pred::SLt,
        Pred::SLt => Pred::SGt,
        Pred::SGe => Pred::SLe,
        Pred::SLe => Pred::SGe,
        Pred::FOGt => Pred::FOLt,
        Pred::FOLt => Pred::FOGt,
        Pred::FOGe => Pred::FOLe,
        Pred::FOLe => Pred::FOGe,
        Pred::FUGt => Pred::FULt,
        Pred::FULt => Pred::FUGt,
        Pred::FUGe => Pred::FULe,
        Pred::FULe => Pred::FUGe,
        other => other,
    }
}

/// If `v` computes logical NOT (`xor i1 x, true`) returns `Some(x)`.
fn match_not_i1(m: &Module, v: Value) -> Option<Value> {
    let Value::Inst(id) = v else { return None };
    if m.int_width(m.insts[id as usize].ty) != Some(1) {
        return None;
    }
    if let Op::Bin(BinOp::Xor, a, b, _) = m.insts[id as usize].op {
        if const_int_u64(m, b) == Some(1) {
            return Some(a);
        }
        if const_int_u64(m, a) == Some(1) {
            return Some(b);
        }
    }
    None
}

// ------------------------------------------------------------ CFG simplify

fn compute_rpo(m: &Module, alive: &[bool]) -> Vec<usize> {
    let n = m.blocks.len();
    if n == 0 || !alive[0] {
        return Vec::new();
    }
    let succs: Vec<Vec<usize>> = (0..n)
        .map(|b| if alive[b] { block_succs(m, b) } else { Vec::new() })
        .collect();
    let mut state = vec![0u8; n];
    let mut postorder = Vec::with_capacity(n);
    let mut stack = vec![(0usize, 0usize)];
    state[0] = 1;
    while let Some((b, idx)) = stack.last_mut() {
        let b = *b;
        let sb = &succs[b];
        if *idx < sb.len() {
            // Visit successors in reverse order so reversed postorder places the
            // primary/true branch before the fallback/exit branch.
            let next = sb[sb.len() - 1 - *idx];
            *idx += 1;
            if alive.get(next).copied().unwrap_or(false) && state[next] == 0 {
                state[next] = 1;
                stack.push((next, 0));
            }
        } else {
            state[b] = 2;
            postorder.push(b);
            stack.pop();
        }
    }
    postorder.reverse();
    postorder
}

/// Keeps at most `keep` entries from `pred` in every `Op::Phi` of block `succ`.
fn trim_phi_edges(m: &mut Module, succ: usize, pred: Block, keep: usize) -> bool {
    let mut changed = false;
    for k in 0..m.blocks[succ].len() {
        let id = m.blocks[succ][k] as usize;
        let Op::Phi(inc) = &mut m.insts[id].op else { break };
        let have = inc.iter().filter(|&&(_, b)| b == pred).count();
        if have > keep {
            let mut left = keep;
            inc.retain(|&(_, b)| {
                if b == pred {
                    if left > 0 {
                        left -= 1;
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            });
            changed = true;
        }
    }
    changed
}

fn replace_phi_pred_block(m: &mut Module, succ: usize, from: Block, to: Block) {
    for k in 0..m.blocks[succ].len() {
        let id = m.blocks[succ][k] as usize;
        let Op::Phi(inc) = &mut m.insts[id].op else { break };
        for (_, b) in inc.iter_mut() {
            if *b == from {
                *b = to;
            }
        }
    }
}

fn block_has_phis(m: &Module, b: usize) -> bool {
    m.blocks[b].first().is_some_and(|&id| matches!(m.insts[id as usize].op, Op::Phi(..)))
}

fn rewrite_term_target(term: &mut Op, from: Block, to: Block) -> usize {
    let mut count = 0;
    match term {
        Op::Br(t) if *t == from => {
            *t = to;
            count += 1;
        }
        Op::CondBr(_, t, f) => {
            if *t == from {
                *t = to;
                count += 1;
            }
            if *f == from {
                *f = to;
                count += 1;
            }
        }
        Op::Switch(_, d, cases) => {
            if *d == from {
                *d = to;
                count += 1;
            }
            for (_, t) in cases {
                if *t == from {
                    *t = to;
                    count += 1;
                }
            }
        }
        _ => {}
    }
    count
}

/// Simplifies terminators, drops unreachable blocks, trims dead PHI edges,
/// bypasses empty jump blocks, merges single-predecessor blocks, and lays out
/// reachable blocks in RPO.
pub(super) fn simplify_cfg(m: &mut Module) -> bool {
    let n = m.blocks.len();
    if n == 0 {
        return false;
    }
    let mut changed = false;

    // 1. Fold constant / redundant terminators.
    for b in 0..n {
        let Some(&term_id) = m.blocks[b].last() else { continue };
        let op = m.insts[term_id as usize].op.clone();
        match op {
            Op::CondBr(c, t, f) => {
                if let Some(cv) = const_int_u64(m, c) {
                    let target = if cv & 1 != 0 { t } else { f };
                    m.insts[term_id as usize].op = Op::Br(target);
                    let pred = Block(b as u32);
                    trim_phi_edges(m, target.0 as usize, pred, 1);
                    let other = if target == t { f } else { t };
                    if other != target {
                        trim_phi_edges(m, other.0 as usize, pred, 0);
                    }
                    changed = true;
                } else if t == f {
                    m.insts[term_id as usize].op = Op::Br(t);
                    trim_phi_edges(m, t.0 as usize, Block(b as u32), 1);
                    changed = true;
                } else if let Some(inner) = match_not_i1(m, c) {
                    m.insts[term_id as usize].op = Op::CondBr(inner, f, t);
                    changed = true;
                }
            }
            Op::Switch(v, d, cases) => {
                let mut old_targets = Vec::with_capacity(1 + cases.len());
                old_targets.push(d);
                for &(_, target) in &cases {
                    if !old_targets.contains(&target) {
                        old_targets.push(target);
                    }
                }
                if let Some(cv) = const_int_u64(m, v) {
                    let target = cases
                        .iter()
                        .find(|&&(k, _)| const_int_u64(m, k) == Some(cv))
                        .map_or(d, |&(_, dst)| dst);
                    m.insts[term_id as usize].op = Op::Br(target);
                    let pred = Block(b as u32);
                    for s in old_targets {
                        trim_phi_edges(m, s.0 as usize, pred, usize::from(s == target));
                    }
                    changed = true;
                } else {
                    let mut seen_vals = Vec::with_capacity(cases.len());
                    let filtered: Vec<(Value, Block)> = cases
                        .iter()
                        .copied()
                        .filter(|&(cv, dst)| {
                            if dst == d {
                                return false;
                            }
                            if let Some(kv) = const_int_u64(m, cv) {
                                if seen_vals.contains(&kv) {
                                    return false;
                                }
                                seen_vals.push(kv);
                            }
                            true
                        })
                        .collect();
                    if filtered.len() != cases.len() {
                        let pred = Block(b as u32);
                        if filtered.is_empty() {
                            m.insts[term_id as usize].op = Op::Br(d);
                            for s in old_targets {
                                trim_phi_edges(m, s.0 as usize, pred, usize::from(s == d));
                            }
                        } else {
                            m.insts[term_id as usize].op = Op::Switch(v, d, filtered.clone());
                            for s in old_targets {
                                let keep = usize::from(s == d) + filtered.iter().filter(|&&(_, x)| x == s).count();
                                trim_phi_edges(m, s.0 as usize, pred, keep);
                            }
                        }
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }

    // 2. Reachability and incoming PHI edge reconciliation.
    let mut alive = vec![true; n];
    let rpo = compute_rpo(m, &alive);
    let mut reach = vec![false; n];
    for &b in &rpo {
        reach[b] = true;
    }
    if rpo.len() < n {
        alive = reach.clone();
        changed = true;
    }

    let mut edge_counts: HashMap<(u32, u32), usize> = HashMap::new();
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &b in &rpo {
        for succ in block_succs(m, b) {
            if alive[succ] {
                *edge_counts.entry((b as u32, succ as u32)).or_insert(0) += 1;
                preds[succ].push(b as u32);
            }
        }
    }

    for &b in &rpo {
        for k in 0..m.blocks[b].len() {
            let id = m.blocks[b][k] as usize;
            let Op::Phi(inc) = &mut m.insts[id].op else { break };
            let mut rem: HashMap<u32, usize> = HashMap::new();
            for &p in &preds[b] {
                rem.insert(p, edge_counts[&(p, b as u32)]);
            }
            let orig_len = inc.len();
            inc.retain(|&(_, pb)| {
                if !alive.get(pb.0 as usize).copied().unwrap_or(false) {
                    return false;
                }
                if let Some(cnt) = rem.get_mut(&pb.0) {
                    if *cnt > 0 {
                        *cnt -= 1;
                        return true;
                    }
                }
                false
            });
            if inc.len() != orig_len {
                changed = true;
            }
        }
    }

    // 3. Bypass empty jump-only blocks (Br target with no other instructions).
    for &b in &rpo {
        if !alive[b] {
            continue;
        }
        loop {
            let succs = block_succs(m, b);
            let mut bypassed = false;
            for e in succs {
                if e == 0 || e == b || !alive[e] || m.blocks[e].len() != 1 {
                    continue;
                }
                let Op::Br(t) = m.insts[m.blocks[e][0] as usize].op else { continue };
                let tu = t.0 as usize;
                if tu == e {
                    continue;
                }
                let no_phis = !block_has_phis(m, tu);
                let single_pred_no_dup = preds[e].len() == 1
                    && preds[e][0] == b as u32
                    && !block_succs(m, b).contains(&tu);
                if no_phis || single_pred_no_dup {
                    let Some(&term_id) = m.blocks[b].last() else { break };
                    let count = rewrite_term_target(&mut m.insts[term_id as usize].op, Block(e as u32), t);
                    if count > 0 {
                        if !no_phis {
                            replace_phi_pred_block(m, tu, Block(e as u32), Block(b as u32));
                        }
                        let mut rem = count;
                        preds[e].retain(|&p| {
                            if rem > 0 && p == b as u32 {
                                rem -= 1;
                                false
                            } else {
                                true
                            }
                        });
                        preds[tu].extend(std::iter::repeat_n(b as u32, count));
                        if preds[e].is_empty() {
                            alive[e] = false;
                            if let Some(pos) = preds[tu].iter().position(|&p| p == e as u32) {
                                preds[tu].swap_remove(pos);
                            }
                        }
                        changed = true;
                        bypassed = true;
                        break;
                    }
                }
            }
            if !bypassed {
                break;
            }
        }
    }

    // 4. Single-predecessor block merging (pred ends in `Br(succ)`, preds[succ] == [pred]).
    let mut val_map: Vec<Option<Value>> = vec![None; m.insts.len()];
    let mut had_val_map = false;
    for &pred in &rpo {
        if !alive[pred] {
            continue;
        }
        loop {
            let Some(&last) = m.blocks[pred].last() else { break };
            let Op::Br(succ_b) = m.insts[last as usize].op else { break };
            let succ = succ_b.0 as usize;
            if succ == 0 || succ == pred || !alive[succ] || preds[succ].len() != 1 {
                break;
            }
            m.blocks[pred].pop();
            let succ_insts = std::mem::take(&mut m.blocks[succ]);
            for id in succ_insts {
                let ty = m.insts[id as usize].ty;
                if let Op::Phi(inc) = &m.insts[id as usize].op {
                    let rep = inc
                        .iter()
                        .find(|&&(_, b)| b == Block(pred as u32))
                        .map(|&(v, _)| resolve_val(v, &val_map))
                        .unwrap_or_else(|| m.undef(ty));
                    val_map[id as usize] = Some(rep);
                    had_val_map = true;
                } else {
                    m.blocks[pred].push(id);
                }
            }
            for s2 in block_succs(m, pred) {
                replace_phi_pred_block(m, s2, Block(succ as u32), Block(pred as u32));
                for p in &mut preds[s2] {
                    if *p == succ as u32 {
                        *p = pred as u32;
                    }
                }
            }
            preds[succ].clear();
            alive[succ] = false;
            changed = true;
        }
    }
    if had_val_map {
        for &b in &rpo {
            if !alive[b] {
                continue;
            }
            for &id in &m.blocks[b] {
                rewrite_op_values(&mut m.insts[id as usize].op, &val_map);
            }
        }
        rewrite_md_values(m, &val_map);
    }

    // 5. Dense RPO compaction & block index remapping.
    let final_rpo = compute_rpo(m, &alive);
    let already_ordered = final_rpo.len() == m.blocks.len() && final_rpo.iter().enumerate().all(|(i, &b)| i == b);
    if !already_ordered {
        let mut old_to_new: Vec<Option<Block>> = vec![None; m.blocks.len()];
        let mut new_blocks = Vec::with_capacity(final_rpo.len());
        for (new_idx, &old_idx) in final_rpo.iter().enumerate() {
            old_to_new[old_idx] = Some(Block(new_idx as u32));
            new_blocks.push(std::mem::take(&mut m.blocks[old_idx]));
        }
        m.blocks = new_blocks;
        for b in &m.blocks {
            for &id in b {
                match &mut m.insts[id as usize].op {
                    Op::Phi(inc) => {
                        for (_, pb) in inc {
                            *pb = old_to_new[pb.0 as usize].expect("live phi pred");
                        }
                    }
                    Op::Br(t) => *t = old_to_new[t.0 as usize].expect("live br target"),
                    Op::CondBr(_, t, f) => {
                        *t = old_to_new[t.0 as usize].expect("live condbr true");
                        *f = old_to_new[f.0 as usize].expect("live condbr false");
                    }
                    Op::Switch(_, d, cases) => {
                        *d = old_to_new[d.0 as usize].expect("live switch default");
                        for (_, t) in cases {
                            *t = old_to_new[t.0 as usize].expect("live switch target");
                        }
                    }
                    _ => {}
                }
            }
        }
        m.cur = Block(0);
        changed = true;
    }

    changed
}

// ------------------------------------------------------------ PHI simplify

fn intersect_idom(idom: &[Option<usize>], mut b1: usize, mut b2: usize) -> usize {
    while b1 != b2 {
        while b1 > b2 {
            b1 = idom[b1].expect("rpo pred must have idom");
        }
        while b2 > b1 {
            b2 = idom[b2].expect("rpo pred must have idom");
        }
    }
    b1
}

/// Cooper-Harvey-Kennedy iterative immediate dominators over RPO indices `0..n`.
fn compute_idom(m: &Module) -> Vec<Option<usize>> {
    let n = m.blocks.len();
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if n == 0 {
        return idom;
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        for s in block_succs(m, b) {
            if s < n {
                preds[s].push(b);
            }
        }
    }
    idom[0] = Some(0);
    let mut dom_changed = true;
    while dom_changed {
        dom_changed = false;
        for b in 1..n {
            let Some(&first) = preds[b].iter().find(|&&p| idom[p].is_some()) else {
                continue;
            };
            let mut new_idom = first;
            for &p in &preds[b] {
                if p != first && idom[p].is_some() {
                    new_idom = intersect_idom(&idom, p, new_idom);
                }
            }
            if idom[b] != Some(new_idom) {
                idom[b] = Some(new_idom);
                dom_changed = true;
            }
        }
    }
    idom
}

fn dominates(idom: &[Option<usize>], dom: usize, mut node: usize) -> bool {
    while node > dom {
        let Some(p) = idom[node] else { return false };
        if p == node {
            break;
        }
        node = p;
    }
    node == dom
}

/// Eliminates trivial PHIs (`phi [v, ...], [self, ...] -> v`) and deduplicates
/// identical PHIs within the same block until reaching a fixed point.
pub(super) fn simplify_phis(m: &mut Module) -> bool {
    let idom = compute_idom(m);
    let mut inst_block: Vec<Option<usize>> = vec![None; m.insts.len()];
    for (bi, blk) in m.blocks.iter().enumerate() {
        for &id in blk {
            inst_block[id as usize] = Some(bi);
        }
    }
    let mut repl: Vec<Option<Value>> = vec![None; m.insts.len()];
    let mut any_changed = false;

    loop {
        let mut round_changed = false;

        for b in 0..m.blocks.len() {
            let mut seen: HashMap<(Ty, Vec<(Option<Value>, u32)>), Value> = HashMap::new();
            let old_insts = std::mem::take(&mut m.blocks[b]);
            let mut keep = Vec::with_capacity(old_insts.len());
            let mut in_phi_prefix = true;

            for id in old_insts {
                if !in_phi_prefix || !matches!(m.insts[id as usize].op, Op::Phi(..)) {
                    in_phi_prefix = false;
                    if rewrite_op_values(&mut m.insts[id as usize].op, &repl) {
                        round_changed = true;
                    }
                    keep.push(id);
                    continue;
                }
                let phi_ty = m.insts[id as usize].ty;
                let mut inc = match &mut m.insts[id as usize].op {
                    Op::Phi(v) => std::mem::take(v),
                    _ => unreachable!(),
                };
                for (v, _) in inc.iter_mut() {
                    let nv = resolve_val(*v, &repl);
                    if nv != *v {
                        *v = nv;
                        round_changed = true;
                    }
                }

                let self_val = Value::Inst(id);

                // Trivial PHI check: ignore self-references and undefs when looking for a unique value.
                let mut unique: Option<Value> = None;
                let mut non_trivial = false;
                for &(v, _) in &inc {
                    if v == self_val || is_undef(m, v) {
                        continue;
                    }
                    match unique {
                        None => unique = Some(v),
                        Some(u) if u == v => {}
                        Some(_) => {
                            non_trivial = true;
                            break;
                        }
                    }
                }
                if !non_trivial {
                    let can_replace = match unique {
                        Some(Value::Inst(ui)) => match inst_block.get(ui as usize).copied().flatten() {
                            Some(db) if db != b => dominates(&idom, db, b),
                            Some(_) => matches!(m.insts[ui as usize].op, Op::Phi(..)),
                            None => false,
                        },
                        _ => true,
                    };
                    if can_replace {
                        let rep = unique.unwrap_or_else(|| m.undef(phi_ty));
                        repl[id as usize] = Some(rep);
                        round_changed = true;
                        continue;
                    }
                }

                // Duplicate-PHI check within block `b` (normalizing self-ref to `None`).
                let mut key_edges: Vec<(Option<Value>, u32)> = inc
                    .iter()
                    .map(|&(v, pb)| (if v == self_val { None } else { Some(v) }, pb.0))
                    .collect();
                key_edges.sort_unstable_by_key(|&(v, pb)| (pb, v));
                let key = (phi_ty, key_edges);
                if let Some(&prev) = seen.get(&key) {
                    repl[id as usize] = Some(prev);
                    round_changed = true;
                    continue;
                }
                seen.insert(key, self_val);
                m.insts[id as usize].op = Op::Phi(inc);
                keep.push(id);
            }
            m.blocks[b] = keep;
        }

        if !round_changed {
            break;
        }
        any_changed = true;
        // Propagate replacements into PHIs whose producers came later in the same block or backedges.
        for b in 0..m.blocks.len() {
            for &id in &m.blocks[b] {
                if !matches!(m.insts[id as usize].op, Op::Phi(..)) {
                    break;
                }
                rewrite_op_values(&mut m.insts[id as usize].op, &repl);
            }
        }
    }

    if any_changed {
        rewrite_md_values(m, &repl);
    }
    any_changed
}

// --------------------------------------------------- InstCombine & Folding

fn eval_int_binop(op: BinOp, w: u32, ua: u64, ub: u64) -> Option<u64> {
    let mask = mask_for(w);
    let ua = ua & mask;
    let ub = ub & mask;
    let r = match op {
        BinOp::Add => ua.wrapping_add(ub),
        BinOp::Sub => ua.wrapping_sub(ub),
        BinOp::Mul => ua.wrapping_mul(ub),
        BinOp::And => ua & ub,
        BinOp::Or => ua | ub,
        BinOp::Xor => ua ^ ub,
        BinOp::UDiv => {
            if ub == 0 {
                return None;
            }
            ua / ub
        }
        BinOp::URem => {
            if ub == 0 {
                return None;
            }
            ua % ub
        }
        BinOp::SDiv => {
            let (sa, sb) = (sext_i64(ua, w), sext_i64(ub, w));
            let smin = sext_i64(1u64 << (w - 1), w);
            if sb == 0 || (sb == -1 && sa == smin) {
                return None;
            }
            (sa / sb) as u64
        }
        BinOp::SRem => {
            let (sa, sb) = (sext_i64(ua, w), sext_i64(ub, w));
            let smin = sext_i64(1u64 << (w - 1), w);
            if sb == 0 || (sb == -1 && sa == smin) {
                return None;
            }
            (sa % sb) as u64
        }
        BinOp::Shl => {
            if ub >= w as u64 {
                return None;
            }
            ua.wrapping_shl(ub as u32)
        }
        BinOp::LShr => {
            if ub >= w as u64 {
                return None;
            }
            ua >> (ub as u32)
        }
        BinOp::AShr => {
            if ub >= w as u64 {
                return None;
            }
            (sext_i64(ua, w) >> (ub as u32)) as u64
        }
    };
    Some(r & mask)
}

fn eval_icmp(p: Pred, w: u32, ua: u64, ub: u64) -> Option<bool> {
    let mask = mask_for(w);
    let ua = ua & mask;
    let ub = ub & mask;
    let (sa, sb) = (sext_i64(ua, w), sext_i64(ub, w));
    Some(match p {
        Pred::Eq => ua == ub,
        Pred::Ne => ua != ub,
        Pred::UGt => ua > ub,
        Pred::UGe => ua >= ub,
        Pred::ULt => ua < ub,
        Pred::ULe => ua <= ub,
        Pred::SGt => sa > sb,
        Pred::SGe => sa >= sb,
        Pred::SLt => sa < sb,
        Pred::SLe => sa <= sb,
        _ => return None,
    })
}

fn make_not_op(m: &mut Module, x: Value) -> Op {
    if let Value::Inst(xi) = x {
        if let Op::Cmp(p, a, b) = m.insts[xi as usize].op {
            return Op::Cmp(invert_pred(p), a, b);
        }
    }
    let t = m.i1c(true);
    Op::Bin(BinOp::Xor, x, t, 0)
}

fn unwrap_bitcast_src(m: &mut Module, v: Value, dst: Ty) -> Option<Value> {
    if m.value_ty(v) == dst {
        return Some(v);
    }
    if let Some(c) = const_bitcast(m, v, dst) {
        return Some(c);
    }
    if let Value::Inst(id) = v {
        if let Op::Cast(CastOp::BitCast, inner, _) = m.insts[id as usize].op {
            if m.value_ty(inner) == dst {
                return Some(inner);
            }
        }
    }
    None
}

fn simplify_int_bin(m: &mut Module, id: u32, ty: Ty, w: u32) -> Result<Value, Option<Op>> {
    let Op::Bin(op, mut a, mut b, _flags) = m.insts[id as usize].op else {
        return Err(None);
    };
    if matches!(op, BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor)
        && const_int_u64(m, a).is_some()
        && const_int_u64(m, b).is_none()
    {
        std::mem::swap(&mut a, &mut b);
        return simplify_rewritten_bin(m, ty, w, op, a, b, true);
    }
    simplify_rewritten_bin(m, ty, w, op, a, b, false)
}

fn simplify_rewritten_bin(
    m: &mut Module, ty: Ty, w: u32, op: BinOp, a: Value, b: Value, swapped: bool,
) -> Result<Value, Option<Op>> {
    let mask = mask_for(w);
    let ca = const_int_u64(m, a);
    let cb = const_int_u64(m, b);
    if let (Some(ua), Some(ub)) = (ca, cb) {
        if let Some(r) = eval_int_binop(op, w, ua, ub) {
            return Ok(m.const_int(ty, r));
        }
    }

    // Reassociate `(inner_op x, c1) outer_op c2` when `b = c2`.
    if let (Value::Inst(ai), Some(c2)) = (a, cb) {
        if let Op::Bin(op1, x, c1_val, _) = m.insts[ai as usize].op {
            if let Some(c1) = const_int_u64(m, c1_val) {
                let reassociated: Option<(BinOp, u64)> = match (op1, op) {
                    (BinOp::Add, BinOp::Add) => Some((BinOp::Add, c1.wrapping_add(c2) & mask)),
                    (BinOp::Sub, BinOp::Add) => Some((BinOp::Add, c2.wrapping_sub(c1) & mask)),
                    (BinOp::Add, BinOp::Sub) => Some((BinOp::Add, c1.wrapping_sub(c2) & mask)),
                    (BinOp::Sub, BinOp::Sub) => Some((BinOp::Add, 0u64.wrapping_sub(c1.wrapping_add(c2)) & mask)),
                    (BinOp::Mul, BinOp::Mul) => Some((BinOp::Mul, c1.wrapping_mul(c2) & mask)),
                    (BinOp::And, BinOp::And) => Some((BinOp::And, c1 & c2)),
                    (BinOp::Or, BinOp::Or) => Some((BinOp::Or, c1 | c2)),
                    (BinOp::Xor, BinOp::Xor) => Some((BinOp::Xor, c1 ^ c2)),
                    (BinOp::Shl, BinOp::Shl)
                    | (BinOp::LShr, BinOp::LShr)
                    | (BinOp::AShr, BinOp::AShr)
                        if (c1 as u128 + c2 as u128) < w as u128 =>
                    {
                        Some((op, c1 + c2))
                    }
                    _ => None,
                };
                if let Some((new_op, nc)) = reassociated {
                    let nc_val = m.const_int(ty, nc);
                    // Apply immediate identity check (e.g. `(x ^ c) ^ c -> x`).
                    if let Ok(folded) = simplify_rewritten_bin(m, ty, w, new_op, x, nc_val, true) {
                        return Ok(folded);
                    }
                    return Err(Some(Op::Bin(new_op, x, nc_val, 0)));
                }
            }
        }
    }

    match op {
        BinOp::Add => {
            if cb == Some(0) {
                return Ok(a);
            }
            if a == b && w == 1 {
                return Ok(m.const_int(ty, 0));
            }
        }
        BinOp::Sub => {
            if cb == Some(0) {
                return Ok(a);
            }
            if a == b {
                return Ok(m.const_int(ty, 0));
            }
            if ca == Some(0) && w == 1 {
                return Ok(b);
            }
            if let Some(c) = cb {
                let neg = (0u64.wrapping_sub(c)) & mask;
                let neg_val = m.const_int(ty, neg);
                return Err(Some(Op::Bin(BinOp::Add, a, neg_val, 0)));
            }
        }
        BinOp::Mul => {
            if cb == Some(0) {
                return Ok(m.const_int(ty, 0));
            }
            if cb == Some(1) {
                return Ok(a);
            }
            if a == b && w == 1 {
                return Ok(a);
            }
        }
        BinOp::UDiv | BinOp::SDiv => {
            if cb == Some(1) {
                return Ok(a);
            }
        }
        BinOp::URem | BinOp::SRem => {
            if cb == Some(1) {
                return Ok(m.const_int(ty, 0));
            }
        }
        BinOp::Shl | BinOp::LShr | BinOp::AShr => {
            if cb == Some(0) {
                return Ok(a);
            }
        }
        BinOp::And => {
            if cb == Some(0) {
                return Ok(m.const_int(ty, 0));
            }
            if cb == Some(mask) || a == b {
                return Ok(a);
            }
            // `x & ~x == 0`
            let is_comp = |lhs: Value, rhs: Value| -> bool {
                let Value::Inst(ri) = rhs else { return false };
                if let Op::Bin(BinOp::Xor, x, y, _) = m.insts[ri as usize].op {
                    (x == lhs && const_int_u64(m, y) == Some(mask))
                        || (y == lhs && const_int_u64(m, x) == Some(mask))
                } else {
                    false
                }
            };
            if is_comp(a, b) || is_comp(b, a) {
                return Ok(m.const_int(ty, 0));
            }
        }
        BinOp::Or => {
            if cb == Some(0) || a == b {
                return Ok(a);
            }
            if cb == Some(mask) {
                return Ok(m.const_int(ty, mask));
            }
            let is_comp = |lhs: Value, rhs: Value| -> bool {
                let Value::Inst(ri) = rhs else { return false };
                if let Op::Bin(BinOp::Xor, x, y, _) = m.insts[ri as usize].op {
                    (x == lhs && const_int_u64(m, y) == Some(mask))
                        || (y == lhs && const_int_u64(m, x) == Some(mask))
                } else {
                    false
                }
            };
            if is_comp(a, b) || is_comp(b, a) {
                return Ok(m.const_int(ty, mask));
            }
        }
        BinOp::Xor => {
            if cb == Some(0) {
                return Ok(a);
            }
            if a == b {
                return Ok(m.const_int(ty, 0));
            }
            // `(x ^ y) ^ y -> x`
            if let Value::Inst(ai) = a {
                if let Op::Bin(BinOp::Xor, u, v, _) = m.insts[ai as usize].op {
                    if v == b {
                        return Ok(u);
                    }
                    if u == b {
                        return Ok(v);
                    }
                }
            }
            if let Value::Inst(bi) = b {
                if let Op::Bin(BinOp::Xor, u, v, _) = m.insts[bi as usize].op {
                    if v == a {
                        return Ok(u);
                    }
                    if u == a {
                        return Ok(v);
                    }
                }
            }
            // `xor(cmp(p, u, v), true) -> cmp(invert(p), u, v)`
            if cb == Some(1) && w == 1 {
                if let Value::Inst(ai) = a {
                    if let Op::Cmp(p, u, v) = m.insts[ai as usize].op {
                        return Err(Some(Op::Cmp(invert_pred(p), u, v)));
                    }
                }
            }
        }
    }

    if swapped {
        Err(Some(Op::Bin(op, a, b, 0)))
    } else {
        Err(None)
    }
}

fn simplify_cast(m: &mut Module, id: u32) -> Result<Value, Option<Op>> {
    let Op::Cast(op, v, dst) = m.insts[id as usize].op else {
        return Err(None);
    };
    let src = m.value_ty(v);
    if src == dst && matches!(op, CastOp::Trunc | CastOp::ZExt | CastOp::SExt | CastOp::FPTrunc | CastOp::FPExt | CastOp::BitCast) {
        return Ok(v);
    }

    match op {
        CastOp::Trunc | CastOp::ZExt => {
            if let (Some(uv), Some(_)) = (const_int_u64(m, v), m.int_width(dst)) {
                return Ok(m.const_int(dst, uv));
            }
        }
        CastOp::SExt => {
            if let (Some(uv), Some(sw), Some(_)) = (const_int_u64(m, v), m.int_width(src), m.int_width(dst)) {
                let sv = sext_i64(uv, sw);
                return Ok(m.const_int(dst, sv as u64));
            }
        }
        CastOp::BitCast => {
            if let Some(c) = const_bitcast(m, v, dst) {
                return Ok(c);
            }
        }
        _ => {}
    }

    if let Value::Inst(vi) = v {
        match (op, m.insts[vi as usize].op.clone()) {
            (CastOp::Trunc, Op::Cast(CastOp::Trunc, x, _)) => return Err(Some(Op::Cast(CastOp::Trunc, x, dst))),
            (CastOp::ZExt, Op::Cast(CastOp::ZExt, x, _)) => return Err(Some(Op::Cast(CastOp::ZExt, x, dst))),
            (CastOp::SExt, Op::Cast(CastOp::SExt, x, _)) => return Err(Some(Op::Cast(CastOp::SExt, x, dst))),
            (CastOp::SExt, Op::Cast(CastOp::ZExt, x, _)) => {
                if let (Some(wx), Some(wsrc)) = (m.int_width(m.value_ty(x)), m.int_width(src)) {
                    if wx < wsrc {
                        return Err(Some(Op::Cast(CastOp::ZExt, x, dst)));
                    }
                }
            }
            (CastOp::Trunc, Op::Cast(CastOp::ZExt, x, _)) => {
                if let (Some(wx), Some(wd)) = (m.int_width(m.value_ty(x)), m.int_width(dst)) {
                    if wx == wd {
                        return Ok(x);
                    } else if wx > wd {
                        return Err(Some(Op::Cast(CastOp::Trunc, x, dst)));
                    } else {
                        return Err(Some(Op::Cast(CastOp::ZExt, x, dst)));
                    }
                }
            }
            (CastOp::Trunc, Op::Cast(CastOp::SExt, x, _)) => {
                if let (Some(wx), Some(wd)) = (m.int_width(m.value_ty(x)), m.int_width(dst)) {
                    if wx == wd {
                        return Ok(x);
                    } else if wx > wd {
                        return Err(Some(Op::Cast(CastOp::Trunc, x, dst)));
                    } else {
                        return Err(Some(Op::Cast(CastOp::SExt, x, dst)));
                    }
                }
            }
            (CastOp::BitCast, Op::Cast(CastOp::BitCast, x, _)) => {
                let xt = m.value_ty(x);
                if xt == dst {
                    return Ok(x);
                }
                if scalar_bits(m, xt).is_some() && scalar_bits(m, xt) == scalar_bits(m, dst) {
                    return Err(Some(Op::Cast(CastOp::BitCast, x, dst)));
                }
            }
            (CastOp::BitCast, Op::Select(c, a, b)) => {
                if scalar_bits(m, src).is_some() && scalar_bits(m, src) == scalar_bits(m, dst) {
                    if let (Some(ua), Some(ub)) = (unwrap_bitcast_src(m, a, dst), unwrap_bitcast_src(m, b, dst)) {
                        if ua == ub {
                            return Ok(ua);
                        }
                        return Err(Some(Op::Select(c, ua, ub)));
                    }
                }
            }
            _ => {}
        }
    }

    Err(None)
}

fn simplify_cmp(m: &mut Module, id: u32) -> Result<Value, Option<Op>> {
    let Op::Cmp(mut p, mut a, mut b) = m.insts[id as usize].op else {
        return Err(None);
    };
    let ty = m.value_ty(a);
    let Some(w) = m.int_width(ty) else {
        return Err(None);
    };
    let mut swapped = false;
    if const_int_u64(m, a).is_some() && const_int_u64(m, b).is_none() {
        std::mem::swap(&mut a, &mut b);
        p = swap_pred(p);
        swapped = true;
    }

    if let (Some(ua), Some(ub)) = (const_int_u64(m, a), const_int_u64(m, b)) {
        if let Some(res) = eval_icmp(p, w, ua, ub) {
            return Ok(m.i1c(res));
        }
    }
    if a == b {
        if let Some(res) = eval_icmp(p, w, 0, 0) {
            return Ok(m.i1c(res));
        }
    }

    let mask = mask_for(w);
    if let Some(cb) = const_int_u64(m, b) {
        if cb == 0 && p == Pred::ULt {
            return Ok(m.i1c(false));
        }
        if cb == 0 && p == Pred::UGe {
            return Ok(m.i1c(true));
        }
        if cb == mask && p == Pred::UGt {
            return Ok(m.i1c(false));
        }
        if cb == mask && p == Pred::ULe {
            return Ok(m.i1c(true));
        }
        if cb == 0 && p == Pred::ULe {
            return Err(Some(Op::Cmp(Pred::Eq, a, b)));
        }
        if cb == 0 && p == Pred::UGt {
            return Err(Some(Op::Cmp(Pred::Ne, a, b)));
        }
        if cb == mask && p == Pred::UGe {
            return Err(Some(Op::Cmp(Pred::Eq, a, b)));
        }
        if cb == mask && p == Pred::ULt {
            return Err(Some(Op::Cmp(Pred::Ne, a, b)));
        }

        if w == 1 {
            return match (p, cb) {
                (Pred::Eq, 1) | (Pred::Ne, 0) => Ok(a),
                (Pred::Eq, 0) | (Pred::Ne, 1) => Err(Some(make_not_op(m, a))),
                _ => {
                    if swapped {
                        Err(Some(Op::Cmp(p, a, b)))
                    } else {
                        Err(None)
                    }
                }
            };
        }

        // Collapse `(zext i1 x) cmp k` and `(sext i1 x) cmp k` (`set_scc` / `scc_bit` idioms)
        // and `select(c, const, const) cmp k`.
        if let Value::Inst(ai) = a {
            match m.insts[ai as usize].op {
                Op::Cast(CastOp::ZExt, inner, _) if m.int_width(m.value_ty(inner)) == Some(1) => {
                    if let (Some(f0), Some(f1)) = (eval_icmp(p, w, 0, cb), eval_icmp(p, w, 1, cb)) {
                        if f0 == f1 {
                            return Ok(m.i1c(f0));
                        }
                        return if f1 { Ok(inner) } else { Err(Some(make_not_op(m, inner))) };
                    }
                }
                Op::Cast(CastOp::SExt, inner, _) if m.int_width(m.value_ty(inner)) == Some(1) => {
                    if let (Some(f0), Some(f1)) = (eval_icmp(p, w, 0, cb), eval_icmp(p, w, mask, cb)) {
                        if f0 == f1 {
                            return Ok(m.i1c(f0));
                        }
                        return if f1 { Ok(inner) } else { Err(Some(make_not_op(m, inner))) };
                    }
                }
                Op::Select(cond, tc, fc) => {
                    if let (Some(ct), Some(cf)) = (const_int_u64(m, tc), const_int_u64(m, fc)) {
                        if let (Some(bt), Some(bf)) = (eval_icmp(p, w, ct, cb), eval_icmp(p, w, cf, cb)) {
                            if bt == bf {
                                return Ok(m.i1c(bt));
                            }
                            return if bt { Ok(cond) } else { Err(Some(make_not_op(m, cond))) };
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if swapped {
        Err(Some(Op::Cmp(p, a, b)))
    } else {
        Err(None)
    }
}

fn simplify_select(m: &mut Module, id: u32) -> Result<Value, Option<Op>> {
    let Op::Select(c, mut t, mut f) = m.insts[id as usize].op else {
        return Err(None);
    };
    if let Some(cv) = const_int_u64(m, c) {
        return Ok(if cv & 1 != 0 { t } else { f });
    }
    if t == f {
        return Ok(t);
    }
    if let Some(inner) = match_not_i1(m, c) {
        return Err(Some(Op::Select(inner, f, t)));
    }

    let ty = m.insts[id as usize].ty;
    if m.int_width(ty) == Some(1) {
        let ct = const_int_u64(m, t);
        let cf = const_int_u64(m, f);
        if ct == Some(1) && cf == Some(0) {
            return Ok(c);
        }
        if ct == Some(0) && cf == Some(1) {
            return Err(Some(make_not_op(m, c)));
        }
        if (t == c && cf == Some(0)) || (ct == Some(1) && f == c) {
            return Ok(c);
        }
        if t == c && cf == Some(1) {
            return Ok(m.i1c(true));
        }
        if ct == Some(0) && f == c {
            return Ok(m.i1c(false));
        }
    }

    // `select c, (select c, t1, _), f -> select c, t1, f` (and analogously in `f`).
    let mut folded_arm = false;
    if let Value::Inst(ti) = t {
        if let Op::Select(c2, t2, _) = m.insts[ti as usize].op {
            if c2 == c {
                t = t2;
                folded_arm = true;
            }
        }
    }
    if let Value::Inst(fi) = f {
        if let Op::Select(c2, _, f2) = m.insts[fi as usize].op {
            if c2 == c {
                f = f2;
                folded_arm = true;
            }
        }
    }
    if folded_arm {
        if t == f {
            return Ok(t);
        }
        return Err(Some(Op::Select(c, t, f)));
    }

    Err(None)
}

fn simplify_pure_dxop_call(m: &mut Module, id: u32) -> Option<Value> {
    let Op::Call(f, args) = &m.insts[id as usize].op else { return None };
    let name = &m.funcs[*f as usize].name;
    let opc = args.first().and_then(|&v| const_int_u64(m, v))?;

    if is_op_name(name, "dx.op.unaryBits") && args.len() == 2 {
        let uv = const_int_u64(m, args[1])?;
        let sw = m.int_width(m.value_ty(args[1]))?;
        let mask = mask_for(sw);
        let u = uv & mask;
        let res = match opc {
            31 => u.count_ones() as u64,
            32 => {
                if u == 0 {
                    u32::MAX as u64
                } else {
                    u.trailing_zeros() as u64
                }
            }
            33 => {
                if u == 0 {
                    u32::MAX as u64
                } else {
                    (u.leading_zeros() - (64 - sw)) as u64
                }
            }
            34 => {
                let sign = (u >> (sw - 1)) & 1 != 0;
                let bits = if sign { (!u) & mask } else { u };
                if bits == 0 {
                    u32::MAX as u64
                } else {
                    (bits.leading_zeros() - (64 - sw)) as u64
                }
            }
            _ => return None,
        };
        return Some(m.i32c(res as u32));
    }

    if is_op_name(name, "dx.op.unary") && opc == 30 && args.len() == 2 {
        let ty = m.insts[id as usize].ty;
        let sw = m.int_width(ty)?;
        let uv = const_int_u64(m, args[1])?;
        let rev = (uv & mask_for(sw)).reverse_bits() >> (64 - sw);
        return Some(m.const_int(ty, rev));
    }

    if is_op_name(name, "dx.op.binary") && (37..=40).contains(&opc) && args.len() == 3 {
        let (a, b) = (args[1], args[2]);
        if a == b {
            return Some(a);
        }
        let ty = m.insts[id as usize].ty;
        let sw = m.int_width(ty)?;
        let (ua, ub) = (const_int_u64(m, a)?, const_int_u64(m, b)?);
        let r = match opc {
            37 => (sext_i64(ua, sw).max(sext_i64(ub, sw))) as u64,
            38 => (sext_i64(ua, sw).min(sext_i64(ub, sw))) as u64,
            39 => ua.max(ub),
            40 => ua.min(ub),
            _ => unreachable!(),
        };
        return Some(m.const_int(ty, r));
    }

    None
}

/// RPO constant folding, cast simplification, integer identities/reassociation,
/// comparison and select simplifications.
pub(super) fn simplify_insts(m: &mut Module) -> bool {
    let mut repl: Vec<Option<Value>> = vec![None; m.insts.len()];
    let mut changed = false;

    for b in 0..m.blocks.len() {
        let mut keep = Vec::with_capacity(m.blocks[b].len());
        let old_insts = std::mem::take(&mut m.blocks[b]);
        for id in old_insts {
            if rewrite_op_values(&mut m.insts[id as usize].op, &repl) {
                changed = true;
            }
            if matches!(m.insts[id as usize].op, Op::Phi(..)) {
                keep.push(id);
                continue;
            }
            let ty = m.insts[id as usize].ty;
            let simplified = match &m.insts[id as usize].op {
                Op::Bin(..) => {
                    if let Some(w) = m.int_width(ty) {
                        simplify_int_bin(m, id, ty, w)
                    } else {
                        Err(None)
                    }
                }
                Op::Cast(..) => simplify_cast(m, id),
                Op::Cmp(..) => simplify_cmp(m, id),
                Op::Select(..) => simplify_select(m, id),
                Op::Call(..) => match simplify_pure_dxop_call(m, id) {
                    Some(v) => Ok(v),
                    None => Err(None),
                },
                _ => Err(None),
            };
            match simplified {
                Ok(v) if v != Value::Inst(id) => {
                    repl[id as usize] = Some(v);
                    changed = true;
                }
                Err(Some(new_op)) => {
                    m.insts[id as usize].op = new_op;
                    changed = true;
                    keep.push(id);
                }
                _ => keep.push(id),
            }
        }
        m.blocks[b] = keep;
    }

    if changed {
        for b in 0..m.blocks.len() {
            for &id in &m.blocks[b] {
                if !matches!(m.insts[id as usize].op, Op::Phi(..)) {
                    break;
                }
                rewrite_op_values(&mut m.insts[id as usize].op, &repl);
            }
        }
        rewrite_md_values(m, &repl);
    }

    changed
}

// --------------------------------------------------------- Dominator CSE

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CseKey {
    Bin(u8, Value, Value, u64),
    Cast(u8, Value, Ty),
    Cmp(u8, Value, Value),
    Select(Value, Value, Value),
    Extract(Value, u32),
    Gep(Value, Vec<Value>),
    Call(u32, Vec<Value>),
}

fn cse_key(m: &Module, op: &Op) -> Option<CseKey> {
    match op {
        Op::Bin(bin_op, a, b, flags) => {
            let (mut x, mut y) = (*a, *b);
            if matches!(bin_op, BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor) && x > y {
                std::mem::swap(&mut x, &mut y);
            }
            Some(CseKey::Bin(*bin_op as u8, x, y, *flags))
        }
        Op::Cast(cop, v, ty) => Some(CseKey::Cast(*cop as u8, *v, *ty)),
        Op::Cmp(p, a, b) => {
            let (mut p, mut x, mut y) = (*p, *a, *b);
            if x > y {
                std::mem::swap(&mut x, &mut y);
                p = swap_pred(p);
            }
            Some(CseKey::Cmp(p as u8, x, y))
        }
        Op::Select(c, t, f) => Some(CseKey::Select(*c, *t, *f)),
        Op::Extract(a, idx) => Some(CseKey::Extract(*a, *idx)),
        Op::Gep(p, idx) => Some(CseKey::Gep(*p, idx.clone())),
        Op::Call(f, args) => {
            let fd = &m.funcs[*f as usize];
            if fd.attrs != FnAttrs::ReadNone {
                return None;
            }
            // Implicit quad derivatives (`dx.op.unary` opcodes 83..=86) depend on
            // CFG position even though declared `readnone`. Never CSE them.
            if is_op_name(&fd.name, "dx.op.unary") {
                let opc = args.first().and_then(|&v| const_int_u64(m, v));
                if !matches!(opc, Some(0..=30)) {
                    return None;
                }
            }
            let mut norm_args = args.clone();
            if is_op_name(&fd.name, "dx.op.binary")
                && norm_args.len() == 3
                && matches!(norm_args.first().and_then(|&v| const_int_u64(m, v)), Some(35..=40))
                && norm_args[1] > norm_args[2]
            {
                norm_args.swap(1, 2);
            }
            Some(CseKey::Call(*f, norm_args))
        }
        _ => None,
    }
}

/// Dominator-tree scoped CSE assuming blocks `0..n` are already in RPO.
pub(super) fn cse(m: &mut Module) -> bool {
    let n = m.blocks.len();
    if n == 0 {
        return false;
    }

    let idom = compute_idom(m);
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 1..n {
        if let Some(p) = idom[b] {
            children[p].push(b);
        }
    }

    enum DomFrame {
        Enter(usize),
        Exit(usize),
    }

    let mut table: HashMap<CseKey, Value> = HashMap::new();
    let mut undo: Vec<CseKey> = Vec::new();
    let mut repl: Vec<Option<Value>> = vec![None; m.insts.len()];
    let mut changed = false;
    let mut stack = vec![DomFrame::Enter(0)];

    while let Some(frame) = stack.pop() {
        match frame {
            DomFrame::Enter(b) => {
                let mark = undo.len();
                stack.push(DomFrame::Exit(mark));

                let old_insts = std::mem::take(&mut m.blocks[b]);
                let mut keep = Vec::with_capacity(old_insts.len());
                for id in old_insts {
                    if !matches!(m.insts[id as usize].op, Op::Phi(..))
                        && rewrite_op_values(&mut m.insts[id as usize].op, &repl)
                    {
                        changed = true;
                    }
                    if let Some(key) = cse_key(m, &m.insts[id as usize].op) {
                        if let Some(&prev) = table.get(&key) {
                            repl[id as usize] = Some(prev);
                            changed = true;
                            continue;
                        }
                        table.insert(key.clone(), Value::Inst(id));
                        undo.push(key);
                    }
                    keep.push(id);
                }
                m.blocks[b] = keep;

                for &c in children[b].iter().rev() {
                    stack.push(DomFrame::Enter(c));
                }
            }
            DomFrame::Exit(mark) => {
                while undo.len() > mark {
                    if let Some(k) = undo.pop() {
                        table.remove(&k);
                    }
                }
            }
        }
    }

    if changed {
        for b in 0..n {
            for &id in &m.blocks[b] {
                if !matches!(m.insts[id as usize].op, Op::Phi(..)) {
                    break;
                }
                rewrite_op_values(&mut m.insts[id as usize].op, &repl);
            }
        }
        rewrite_md_values(m, &repl);
    }

    changed
}

// ------------------------------------------------------------------- DCE

fn is_pure_wave_quad(name: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "dx.op.waveIsFirstLane",
        "dx.op.waveAnyTrue",
        "dx.op.waveAllTrue",
        "dx.op.waveActiveAllEqual",
        "dx.op.waveActiveBallot",
        "dx.op.waveReadLaneAt",
        "dx.op.waveReadLaneFirst",
        "dx.op.waveActiveOp",
        "dx.op.waveActiveBit",
        "dx.op.wavePrefixOp",
        "dx.op.waveAllOp",
        "dx.op.quadReadLaneAt",
        "dx.op.quadOp",
        "dx.op.waveMatch",
        "dx.op.waveMultiPrefixOp",
        "dx.op.waveMultiPrefixBitCount",
    ];
    PREFIXES.iter().any(|base| is_op_name(name, base))
}

fn is_dce_root(m: &Module, op: &Op) -> bool {
    match op {
        Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) | Op::Br(_) | Op::CondBr(..) | Op::Switch(..) | Op::Ret | Op::Unreachable => {
            true
        }
        Op::Call(f, _) => {
            let fd = &m.funcs[*f as usize];
            match fd.attrs {
                FnAttrs::ReadNone | FnAttrs::ReadOnly => false,
                FnAttrs::NoUnwind => !is_pure_wave_quad(&fd.name),
                FnAttrs::NoDuplicate | FnAttrs::None => true,
            }
        }
        _ => false,
    }
}

/// Iterative mark-and-sweep DCE over all block instructions.
pub(super) fn dce(m: &mut Module) -> bool {
    let mut live = vec![false; m.insts.len()];
    let mut work = Vec::new();

    for b in &m.blocks {
        for &id in b {
            if is_dce_root(m, &m.insts[id as usize].op) {
                live[id as usize] = true;
                work.push(id);
            }
        }
    }
    for md in &m.md {
        if let MdDef::Val(_, Value::Inst(id)) = *md {
            if !live[id as usize] {
                live[id as usize] = true;
                work.push(id);
            }
        }
    }

    while let Some(id) = work.pop() {
        for_each_op_value(&m.insts[id as usize].op, |v| {
            if let Value::Inst(dep) = v {
                if !live[dep as usize] {
                    live[dep as usize] = true;
                    work.push(dep);
                }
            }
        });
    }

    let mut changed = false;
    for b in &mut m.blocks {
        let orig = b.len();
        b.retain(|&id| live[id as usize]);
        if b.len() != orig {
            changed = true;
        }
    }
    changed
}

// ---------------------------------------------------- Module compaction

/// Compacts live instructions, constants, declarations, and globals densely while
/// preserving constant aggregate dependency order.
pub(super) fn compact_module(m: &mut Module, mut entry_func: Option<&mut u32>) {
    let mut live_inst = vec![false; m.insts.len()];
    let mut live_const = vec![false; m.consts.len()];
    let mut live_func = vec![false; m.funcs.len()];
    let mut live_global = vec![false; m.globals.len()];
    let mut const_work = Vec::new();
    let mut global_work = Vec::new();

    let mark_val = |v: Value, lc: &mut [bool], lf: &mut [bool], lg: &mut [bool], cw: &mut Vec<u32>, gw: &mut Vec<u32>| match v {
        Value::Const(c) => {
            if !lc[c as usize] {
                lc[c as usize] = true;
                cw.push(c);
            }
        }
        Value::Func(f) => lf[f as usize] = true,
        Value::Global(g) => {
            if !lg[g as usize] {
                lg[g as usize] = true;
                gw.push(g);
            }
        }
        Value::Inst(_) => {}
    };

    for (i, fd) in m.funcs.iter().enumerate() {
        if fd.defined {
            live_func[i] = true;
        }
    }
    if let Some(ref ef) = entry_func {
        live_func[**ef as usize] = true;
    }

    for b in &m.blocks {
        for &id in b {
            live_inst[id as usize] = true;
            if let Op::Call(f, _) = m.insts[id as usize].op {
                live_func[f as usize] = true;
            }
            for_each_op_value(&m.insts[id as usize].op, |v| {
                mark_val(v, &mut live_const, &mut live_func, &mut live_global, &mut const_work, &mut global_work);
            });
        }
    }
    for md in &m.md {
        if let MdDef::Val(_, v) = *md {
            mark_val(v, &mut live_const, &mut live_func, &mut live_global, &mut const_work, &mut global_work);
        }
    }

    while !const_work.is_empty() || !global_work.is_empty() {
        while let Some(c) = const_work.pop() {
            if let ConstDef::Aggregate(_, elems) = &m.consts[c as usize] {
                for &v in elems {
                    mark_val(v, &mut live_const, &mut live_func, &mut live_global, &mut const_work, &mut global_work);
                }
            }
        }
        while let Some(g) = global_work.pop() {
            if let Some(v) = m.globals[g as usize].init {
                mark_val(v, &mut live_const, &mut live_func, &mut live_global, &mut const_work, &mut global_work);
            }
        }
    }

    let inst_map = build_remap(&live_inst);
    let const_map = build_remap(&live_const);
    let func_map = build_remap(&live_func);
    let global_map = build_remap(&live_global);

    let remap_val = |v: Value| -> Value {
        match v {
            Value::Inst(i) => Value::Inst(inst_map[i as usize].expect("live inst")),
            Value::Const(c) => Value::Const(const_map[c as usize].expect("live const")),
            Value::Func(f) => Value::Func(func_map[f as usize].expect("live func")),
            Value::Global(g) => Value::Global(global_map[g as usize].expect("live global")),
        }
    };

    let mut new_insts = Vec::with_capacity(live_inst.iter().filter(|&&x| x).count());
    let old_insts = std::mem::take(&mut m.insts);
    for (i, mut inst) in old_insts.into_iter().enumerate() {
        if live_inst[i] {
            remap_op(&mut inst.op, &remap_val, &func_map);
            new_insts.push(inst);
        }
    }
    m.insts = new_insts;
    for b in &mut m.blocks {
        for id in b.iter_mut() {
            *id = inst_map[*id as usize].expect("live inst");
        }
    }

    let mut new_consts = Vec::with_capacity(live_const.iter().filter(|&&x| x).count());
    m.const_ids.clear();
    let old_consts = std::mem::take(&mut m.consts);
    for (i, mut c) in old_consts.into_iter().enumerate() {
        if live_const[i] {
            if let ConstDef::Aggregate(_, elems) = &mut c {
                for v in elems {
                    *v = remap_val(*v);
                }
            }
            let nid = new_consts.len() as u32;
            m.const_ids.insert(c.clone(), nid);
            new_consts.push(c);
        }
    }
    m.consts = new_consts;

    let mut new_funcs = Vec::with_capacity(live_func.iter().filter(|&&x| x).count());
    m.func_ids.clear();
    let old_funcs = std::mem::take(&mut m.funcs);
    for (i, fd) in old_funcs.into_iter().enumerate() {
        if live_func[i] {
            let nid = new_funcs.len() as u32;
            m.func_ids.insert(fd.name.clone(), nid);
            new_funcs.push(fd);
        }
    }
    m.funcs = new_funcs;
    if let Some(ef) = entry_func.as_deref_mut() {
        *ef = func_map[*ef as usize].expect("entry func live");
    }

    let mut new_globals = Vec::with_capacity(live_global.iter().filter(|&&x| x).count());
    let old_globals = std::mem::take(&mut m.globals);
    for (i, mut g) in old_globals.into_iter().enumerate() {
        if live_global[i] {
            g.init = g.init.map(&remap_val);
            new_globals.push(g);
        }
    }
    m.globals = new_globals;

    for md in &mut m.md {
        if let MdDef::Val(_, v) = md {
            *v = remap_val(*v);
        }
    }
}

fn build_remap(live: &[bool]) -> Vec<Option<u32>> {
    let mut map = Vec::with_capacity(live.len());
    let mut next = 0u32;
    for &l in live {
        if l {
            map.push(Some(next));
            next += 1;
        } else {
            map.push(None);
        }
    }
    map
}

fn remap_op(op: &mut Op, remap_val: &dyn Fn(Value) -> Value, func_map: &[Option<u32>]) {
    if let Op::Call(f, _) = op {
        *f = func_map[*f as usize].expect("live call target");
    }
    visit_op_values_mut(op, &mut |v| *v = remap_val(*v));
}

// ---------------------------------------------------------- Feature sync

/// Recomputes `s.features` against the remaining live instructions and static
/// resource bindings after optimization and DCE.
pub(super) fn sync_features(s: &mut Shader) {
    let orig = s.features.clone();
    let (i64_ty, f64_ty) = s.m.existing_i64_f64();

    let mut has_fma64 = false;
    let mut has_wave_or_quad = false;
    let mut has_res_heap = false;
    let mut has_smp_heap = false;
    let mut has_i64_atomic_gs = false;
    let mut has_i64_atomic_res = false;
    let mut has_tiled = false;
    let mut has_typed_uav_load = false;
    let mut has_raw_struct_annotate = false;

    for &id in s.m.blocks.iter().flatten() {
        match &s.m.insts[id as usize].op {
            Op::AtomicRmw(_, _, v) | Op::CmpXchg(_, v, _) => {
                if Some(s.m.value_ty(*v)) == i64_ty {
                    has_i64_atomic_gs = true;
                }
            }
            Op::Call(f, args) => {
                let name = &s.m.funcs[*f as usize].name;
                if name.starts_with("dx.op.wave") || name.starts_with("dx.op.quad") {
                    has_wave_or_quad = true;
                }
                if name == "dx.op.tertiary.f64" && args.first().and_then(|&v| const_int_u64(&s.m, v)) == Some(47) {
                    has_fma64 = true;
                }
                if is_op_name(name, "dx.op.createHandleFromHeap") {
                    match args.get(2).and_then(|&v| const_int_u64(&s.m, v)) {
                        Some(0) => has_res_heap = true,
                        Some(1) => has_smp_heap = true,
                        _ => {
                            has_res_heap = true;
                            has_smp_heap = true;
                        }
                    }
                }
                if (is_op_name(name, "dx.op.atomicBinOp") || is_op_name(name, "dx.op.atomicCompareExchange"))
                    && name.ends_with(".i64")
                {
                    has_i64_atomic_res = true;
                }
                if is_op_name(name, "dx.op.checkAccessFullyMapped") {
                    has_tiled = true;
                }
                if is_op_name(name, "dx.op.sample")
                    || is_op_name(name, "dx.op.sampleBias")
                    || is_op_name(name, "dx.op.sampleGrad")
                    || is_op_name(name, "dx.op.sampleCmp")
                {
                    if let Some(&clamp) = args.last() {
                        if !is_undef(&s.m, clamp) {
                            has_tiled = true;
                        }
                    }
                }
                if is_op_name(name, "dx.op.textureLoad") {
                    has_typed_uav_load = true;
                }
                if is_op_name(name, "dx.op.annotateHandle") {
                    if let Some(&Value::Const(c)) = args.get(2) {
                        if let ConstDef::Aggregate(_, elems) = &s.m.consts[c as usize] {
                            if let Some(p0) = elems.first().and_then(|&e| const_int_u64(&s.m, e)) {
                                let k = (p0 & 0xFF) as u32;
                                if k == kind::RAW_BUFFER || k == kind::STRUCTURED_BUFFER {
                                    has_raw_struct_annotate = true;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut out = BTreeSet::new();
    let mut set = |cond: bool, feat: Feature| {
        if cond {
            out.insert(feat);
        }
    };
    set(i64_ty.is_some_and(|t| s.m.insts_use_type(t)), Feature::Int64Ops);
    set(f64_ty.is_some_and(|t| s.m.insts_use_type(t)), Feature::Doubles);
    set(has_fma64 || s.m.uses_double_extensions(), Feature::DoubleExtensions);
    set(has_wave_or_quad, Feature::WaveOps);
    set(has_res_heap, Feature::ResourceHeapIndexing);
    set(has_smp_heap, Feature::SamplerHeapIndexing);
    set(has_i64_atomic_gs, Feature::AtomicInt64GroupShared);
    set(orig.contains(&Feature::AtomicInt64Typed) && has_i64_atomic_res, Feature::AtomicInt64Typed);
    set(orig.contains(&Feature::AtomicInt64Heap) && has_i64_atomic_res && has_res_heap, Feature::AtomicInt64Heap);
    set(orig.contains(&Feature::TiledResources) && has_tiled, Feature::TiledResources);
    set(orig.contains(&Feature::TypedUavLoadAdditionalFormats) && has_typed_uav_load, Feature::TypedUavLoadAdditionalFormats);
    set(
        s.bindings.iter().any(|b| b.kind == kind::RAW_BUFFER || b.kind == kind::STRUCTURED_BUFFER)
            || has_raw_struct_annotate,
        Feature::RawAndStructuredBuffers,
    );
    set(orig.contains(&Feature::NativeLowPrecision), Feature::NativeLowPrecision);
    s.features = out;
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::dxil::container::Stage;
    use crate::gpu::dxil::ops::{Binary, BitCount, Deriv, WaveOp};
    use crate::gpu::dxil::shader::Shader;
    use crate::gpu::dxil::validate;
    use crate::gpu::recomp::runner;

    fn validate_if_supported(dxil: &[u8]) {
        if validate::validator_supported() {
            if let Err(e) = validate::validate(dxil) {
                let dis = validate::disassemble(dxil).unwrap_or_else(|d| d);
                panic!("DXIL failed validation: {e}\n--- disassembly ---\n{dis}");
            }
        }
    }

    #[test]
    fn trivial_and_chained_phis_collapse_and_unlock_constant_folding() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(1, 0);
        let i32 = s.m.i32();
        let entry = Block(0);
        let loop_b = s.m.new_block();
        let exit = s.m.new_block();

        s.m.set_block(entry);
        let init_exec = s.m.i32c(0xFFFF_FFFF);
        s.m.br(loop_b);

        s.m.set_block(loop_b);
        let p_exec = s.m.phi(i32);
        let p_dup1 = s.m.phi(i32);
        let p_dup2 = s.m.phi(i32);
        s.m.add_incoming(p_exec, init_exec, entry);
        s.m.add_incoming(p_exec, p_exec, loop_b);
        let c10 = s.m.i32c(10);
        s.m.add_incoming(p_dup1, c10, entry);
        s.m.add_incoming(p_dup1, p_dup1, loop_b);
        s.m.add_incoming(p_dup2, c10, entry);
        s.m.add_incoming(p_dup2, p_dup2, loop_b);

        // `((p_exec & 1) != 0) ? (p_dup1 + p_dup2) : 99` should fold completely to `20`.
        let one = s.m.i32c(1);
        let zero = s.m.i32c(0);
        let bit = s.m.bin(BinOp::And, p_exec, one);
        let cond = s.m.cmp(Pred::Ne, bit, zero);
        let sum = s.m.bin(BinOp::Add, p_dup1, p_dup2);
        let fallback = s.m.i32c(99);
        let sel = s.m.select(cond, sum, fallback);
        s.raw_store(uav, zero, &[sel]);
        let f = s.m.i1c(false);
        s.m.cond_br(f, loop_b, exit);

        s.m.set_block(exit);
        s.m.ret();

        s.optimize();
        let st = module_stats(&s.m);
        assert_eq!(st.phis, 0, "all trivial/duplicate phis must collapse");
        assert_eq!(st.blocks, 1, "loop + exit must merge into entry block");

        let dxil = s.finish([1, 1, 1], Some(32));
        validate_if_supported(&dxil);
    }

    #[test]
    fn bitcast_and_select_and_scc_idioms_simplify() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(1, 0);
        let tid = s.flattened_thread_id_in_group();
        let i32 = s.m.i32();
        let f32 = s.m.f32();
        let i64 = s.m.i64();

        let zero = s.m.i32c(0);
        let c5 = s.m.i32c(5);
        let cmp = s.m.cmp(Pred::SLt, tid, c5);
        // `set_scc` + `scc_bit` idiom: zext(cmp) != 0, then xor(..., true), then select swap.
        let scc = s.m.cast(CastOp::ZExt, cmp, i32);
        let nz = s.m.cmp(Pred::Ne, scc, zero);
        let t1 = s.m.i1c(true);
        let n_nz = s.m.bin(BinOp::Xor, nz, t1);
        let f_two = s.m.f32c(2.0f32.to_bits());
        let f_tid = s.m.cast(CastOp::UIToFP, tid, f32);
        let sel_f = s.m.select(n_nz, f_two, f_tid);
        let bc_i = s.m.bitcast(sel_f, i32);
        let bc_roundtrip = s.m.bitcast(bc_i, f32);
        let tid64 = s.m.cast(CastOp::ZExt, tid, i64);
        let same_i = s.m.cast(CastOp::Trunc, tid64, i32);
        let x0 = s.m.bin(BinOp::Xor, same_i, same_i);
        let off = s.m.bin(BinOp::Add, x0, zero);
        let out_i = s.m.bitcast(bc_roundtrip, i32);
        let sext_i = s.m.cast(CastOp::SExt, n_nz, i32);
        let _sext_nz = s.m.cmp(Pred::Ne, sext_i, zero);
        s.raw_store(uav, off, &[out_i]);
        s.m.ret();

        s.optimize();
        let has_cast = s.m.blocks.iter().flatten().any(|&id| {
            matches!(s.m.insts[id as usize].op, Op::Cast(CastOp::ZExt | CastOp::SExt | CastOp::Trunc, ..))
        });
        assert!(!has_cast, "redundant zext/sext/trunc roundtrips should disappear");

        let dxil = s.finish([32, 1, 1], Some(32));
        validate_if_supported(&dxil);
    }

    #[test]
    fn dominator_cse_deduplicates_pure_ops_and_preserves_derivatives_and_wave_ops() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(1, 0);
        let b1 = s.m.new_block();
        let else_b = s.m.new_block();

        let c0 = s.m.i32c(0);
        let ld = s.raw_load(uav, c0, 3);
        let (a, b) = (ld[0], ld[1]);
        let _add1 = s.m.bin(BinOp::Add, a, b);
        let f32 = s.m.f32();
        let f_a = s.m.bitcast(a, f32);
        let dx1 = s.deriv(Deriv::CoarseX, f_a);
        let sum1 = s.wave_active_op(a, WaveOp::Sum, false);
        let c4 = s.m.i32c(4);
        let c1 = s.m.cmp(Pred::UGt, a, c4);
        s.m.cond_br(c1, b1, else_b);

        s.m.set_block(else_b);
        s.m.ret();

        s.m.set_block(b1);
        // Commutative duplicate dominated by entry -> CSE must merge into `add1`.
        let add2 = s.m.bin(BinOp::Add, b, a);
        // Derivatives and WaveOps must NOT be CSE'd across entry -> child!
        let dx2 = s.deriv(Deriv::CoarseX, f_a);
        let sum2 = s.wave_active_op(a, WaveOp::Sum, false);
        let dx_sum = s.m.fbin(BinOp::Add, dx1, dx2, 0);
        let i32 = s.m.i32();
        let dx_i = s.m.bitcast(dx_sum, i32);
        let t1 = s.m.bin(BinOp::Add, add2, sum1);
        let t2 = s.m.bin(BinOp::Add, t1, sum2);
        let out = s.m.bin(BinOp::Add, t2, dx_i);
        s.raw_store(uav, c0, &[out]);
        s.m.ret();

        s.optimize();
        let extract_add_count = s
            .m
            .blocks
            .iter()
            .flatten()
            .filter(|&&id| match s.m.insts[id as usize].op {
                Op::Bin(BinOp::Add, Value::Inst(x), Value::Inst(y), _) => {
                    matches!(s.m.insts[x as usize].op, Op::Extract(..))
                        && matches!(s.m.insts[y as usize].op, Op::Extract(..))
                }
                _ => false,
            })
            .count();
        assert_eq!(extract_add_count, 1, "commutative add across dominated blocks must CSE");
        let deriv_count = s
            .m
            .blocks
            .iter()
            .flatten()
            .filter(|&&id| match &s.m.insts[id as usize].op {
                Op::Call(f, args) => {
                    is_op_name(&s.m.funcs[*f as usize].name, "dx.op.unary")
                        && args.first().and_then(|&v| const_int_u64(&s.m, v)) == Some(Deriv::CoarseX as u64)
                }
                _ => false,
            })
            .count();
        let wave_count = s
            .m
            .blocks
            .iter()
            .flatten()
            .filter(|&&id| match &s.m.insts[id as usize].op {
                Op::Call(f, _) => is_op_name(&s.m.funcs[*f as usize].name, "dx.op.waveActiveOp"),
                _ => false,
            })
            .count();
        assert_eq!(deriv_count, 2, "implicit derivatives must never be CSE'd");
        assert_eq!(wave_count, 2, "waveActiveOp must never be CSE'd");

        let dxil = s.finish([4, 4, 1], Some(32));
        validate_if_supported(&dxil);
    }

    #[test]
    fn dce_eliminates_dead_phi_cycles_and_unused_loads_and_preserves_side_effects() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(1, 0);
        let i32 = s.m.i32();
        let entry = Block(0);
        let loop_b = s.m.new_block();
        let exit = s.m.new_block();

        s.m.set_block(entry);
        let c0 = s.m.i32c(0);
        let c1 = s.m.i32c(1);
        let dead_load = s.raw_load(uav, c0, 1)[0];
        let _dead_wave = s.wave_active_op(c1, WaveOp::Sum, false);
        s.m.br(loop_b);

        s.m.set_block(loop_b);
        let live_ctr = s.m.phi(i32);
        let dead_cycle = s.m.phi(i32);
        s.m.add_incoming(live_ctr, c0, entry);
        let next_ctr = s.m.bin(BinOp::Add, live_ctr, c1);
        s.m.add_incoming(live_ctr, next_ctr, loop_b);

        s.m.add_incoming(dead_cycle, dead_load, entry);
        let dead_next = s.m.bin(BinOp::Add, dead_cycle, c1);
        s.m.add_incoming(dead_cycle, dead_next, loop_b);

        let c8 = s.m.i32c(8);
        let done = s.m.cmp(Pred::Eq, next_ctr, c8);
        s.m.cond_br(done, exit, loop_b);

        s.m.set_block(exit);
        s.raw_store(uav, c0, &[next_ctr]);
        s.m.ret();

        s.optimize();
        let st = module_stats(&s.m);
        assert_eq!(st.phis, 1, "dead phi cycle must be eliminated while live phi stays");
        let load_count = s
            .m
            .blocks
            .iter()
            .flatten()
            .filter(|&&id| matches!(&s.m.insts[id as usize].op, Op::Call(f, _) if is_op_name(&s.m.funcs[*f as usize].name, "dx.op.rawBufferLoad")))
            .count();
        let store_count = s
            .m
            .blocks
            .iter()
            .flatten()
            .filter(|&&id| matches!(&s.m.insts[id as usize].op, Op::Call(f, _) if is_op_name(&s.m.funcs[*f as usize].name, "dx.op.rawBufferStore")))
            .count();
        assert_eq!(load_count, 0, "unused ReadOnly load must be eliminated");
        assert_eq!(store_count, 1, "side-effecting store must remain");
        assert!(!s.features.contains(&Feature::WaveOps), "dead wave op feature must clear");

        let dxil = s.finish([1, 1, 1], Some(32));
        validate_if_supported(&dxil);
    }

    #[test]
    fn cfg_rpo_reordering_fixes_exit_before_body_dominance() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(1, 0);
        // Allocate exit BEFORE body (mirrors `kernel.rs` layout before RPO fix).
        let exit = s.m.new_block();
        let body = s.m.new_block();

        s.m.set_block(Block(0));
        let c0 = s.m.i32c(0);
        let tid = s.flattened_thread_id_in_group();
        let c16 = s.m.i32c(16);
        let cond = s.m.cmp(Pred::ULt, tid, c16);
        s.m.cond_br(cond, body, exit);

        s.m.set_block(body);
        let v = s.m.bin(BinOp::Add, tid, c16);
        let two = s.m.i32c(2);
        let off = s.m.bin(BinOp::Shl, tid, two);
        s.raw_store(uav, off, &[v]);
        s.m.br(exit);

        s.m.set_block(exit);
        let i32 = s.m.i32();
        let phi_out = s.m.phi(i32);
        s.m.add_incoming(phi_out, c0, Block(0));
        s.m.add_incoming(phi_out, c16, body);
        let after = s.m.bin(BinOp::Add, phi_out, tid);
        let base_off = s.m.i32c(256);
        let off2 = s.m.bin(BinOp::Add, base_off, off);
        s.raw_store(uav, off2, &[after]);
        s.m.ret();

        s.optimize();
        let dxil = s.finish([32, 1, 1], Some(32));
        validate_if_supported(&dxil);

        if let Ok(true) = runner::wave32_supported() {
            let mem = vec![0u32; 128];
            let r = runner::run(&dxil, &runner::KernelState::default(), &mem).expect("GPU run");
            for l in 0..32u32 {
                if l < 16 {
                    assert_eq!(r.raw[l as usize], l + 16);
                    assert_eq!(r.raw[64 + l as usize], 16 + l);
                } else {
                    assert_eq!(r.raw[64 + l as usize], l);
                }
            }
        }
    }

    #[test]
    fn compaction_and_feature_sync_drop_dead_consts_funcs_and_stale_features() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(1, 0);
        let dead_b = s.m.new_block();

        s.m.set_block(Block(0));
        let c0 = s.m.i32c(0);
        let out = s.m.i32c(42);
        s.raw_store(uav, c0, &[out]);
        s.m.ret();

        // Dead block referencing i64, f64, fma, wave ops, heap handle, and unused constants.
        s.m.set_block(dead_b);
        let i64t = s.m.i64();
        let i64c = s.m.const_int(i64t, 0x1234_5678_9abc_def0);
        let f64t = s.m.f64();
        let d0 = s.m.const_float(f64t, 1.5f64.to_bits());
        let d1 = s.m.cast(CastOp::SIToFP, i64c, f64t);
        let fma = s.tertiary(super::super::ops::Tertiary::Fma, d0, d1, d0);
        let _d_div = s.m.fbin(BinOp::SDiv, fma, d0, 0);
        let _wv = s.wave_active_op(c0, WaveOp::Sum, false);
        let _hb = s.heap_raw_buffer(c0, true, false);
        let _cnt = s.bit_count(BitCount::Countbits, i64c);
        s.m.ret();

        let before = module_stats(&s.m);
        s.optimize();
        let after = module_stats(&s.m);

        assert!(after.consts < before.consts, "dead constants must compact out");
        assert!(after.funcs < before.funcs, "dead intrinsic decls must compact out");
        assert!(!s.features.contains(&Feature::Int64Ops));
        assert!(!s.features.contains(&Feature::Doubles));
        assert!(!s.features.contains(&Feature::DoubleExtensions));
        assert!(!s.features.contains(&Feature::WaveOps));
        assert!(!s.features.contains(&Feature::ResourceHeapIndexing));
        assert!(s.features.contains(&Feature::RawAndStructuredBuffers));

        let dxil = s.finish([1, 1, 1], Some(32));
        validate_if_supported(&dxil);
    }

    #[test]
    fn gpu_differential_alu_loops_and_bitops_match_unoptimized() {
        let supported = match runner::wave32_supported() {
            Ok(true) => true,
            Ok(false) => {
                runner::skip_test("no wave32");
                false
            }
            Err(e) => {
                runner::skip_test(&e);
                false
            }
        };

        let build = |do_opt: bool| -> (Stats, Vec<u8>) {
            let mut s = Shader::new(Stage::Compute, (6, 6));
            let res = s.raw_uav(1, 0);
            let mem = s.raw_uav(2, 0);
            let tid = s.flattened_thread_id_in_group();
            let i32 = s.m.i32();
            let f32 = s.m.f32();
            let i64 = s.m.i64();

            let entry = Block(0);
            let loop_b = s.m.new_block();
            let exit = s.m.new_block();

            s.m.set_block(entry);
            let two = s.m.i32c(2);
            let off = s.m.bin(BinOp::Shl, tid, two);
            let in_v = s.raw_load(mem, off, 1)[0];
            let c0 = s.m.i32c(0);
            let c1 = s.m.i32c(1);
            let exec_all = s.m.i32c(0xFFFF_FFFF);
            s.m.br(loop_b);

            s.m.set_block(loop_b);
            let p_exec = s.m.phi(i32);
            let p_idx = s.m.phi(i32);
            let p_acc = s.m.phi(i32);
            let p_dead = s.m.phi(i32);
            s.m.add_incoming(p_exec, exec_all, entry);
            s.m.add_incoming(p_exec, p_exec, loop_b);
            s.m.add_incoming(p_idx, c0, entry);
            s.m.add_incoming(p_acc, c0, entry);
            s.m.add_incoming(p_dead, c0, entry);

            let exec_bit = s.m.bin(BinOp::And, p_exec, c1);
            let act = s.m.cmp(Pred::Ne, exec_bit, c0);
            let scc = s.m.cast(CastOp::ZExt, act, i32);
            let act2 = s.m.cmp(Pred::Ne, scc, c0);
            // Redundant `f32 <-> i32` bitcasts and CSE candidates.
            let x_f = s.m.bitcast(p_acc, f32);
            let one_f = s.m.f32c(1.0f32.to_bits());
            let add_f = s.m.fbin(BinOp::Add, x_f, one_f, 0);
            let add_i = s.m.bitcast(add_f, i32);
            let mix1 = s.m.bin(BinOp::Xor, in_v, c1);
            let mix2 = s.m.bin(BinOp::Xor, c1, in_v);
            let zext_mix = s.m.cast(CastOp::ZExt, mix2, i64);
            let trunc_zext = s.m.cast(CastOp::Trunc, zext_mix, i32);
            let diff = s.m.bin(BinOp::Sub, mix1, trunc_zext);
            let step = s.m.bin(BinOp::Add, add_i, diff);
            let next_acc = s.m.select(act2, step, p_acc);
            let next_idx = s.m.bin(BinOp::Add, p_idx, c1);
            let dead_next = s.m.bin(BinOp::Add, p_dead, c1);
            s.m.add_incoming(p_idx, next_idx, loop_b);
            s.m.add_incoming(p_acc, next_acc, loop_b);
            s.m.add_incoming(p_dead, dead_next, loop_b);

            let c4 = s.m.i32c(4);
            let go = s.m.cmp(Pred::ULt, next_idx, c4);
            s.m.cond_br(go, loop_b, exit);

            s.m.set_block(exit);
            let c_fshi = s.m.i32c(0x4000_0000);
            let const_bits = s.bit_count(BitCount::FirstbitSHi, c_fshi);
            let c7 = s.m.i32c(7);
            let c19 = s.m.i32c(19);
            let cst = s.binary(Binary::UMax, c7, c19);
            let rhs = s.m.bin(BinOp::Add, const_bits, cst);
            let out = s.m.bin(BinOp::Add, p_acc, rhs);
            s.raw_store(res, off, &[out]);
            s.m.ret();

            if do_opt {
                s.optimize();
            }
            let st = module_stats(&s.m);
            let dxil = s.finish([32, 1, 1], Some(32));
            (st, dxil)
        };

        let (raw_st, raw_dxil) = build(false);
        let (opt_st, opt_dxil) = build(true);
        assert!(opt_st.insts < raw_st.insts);
        assert!(opt_st.phis < raw_st.phis);
        assert!(opt_dxil.len() < raw_dxil.len());
        validate_if_supported(&opt_dxil);

        if supported {
            let mut mem = vec![0u32; 64];
            for i in 0..32u32 {
                mem[i as usize] = i.wrapping_mul(0x9e37_79b9);
            }
            let raw_r = runner::run(&raw_dxil, &runner::KernelState::default(), &mem).expect("raw GPU run");
            let opt_r = runner::run(&opt_dxil, &runner::KernelState::default(), &mem).expect("opt GPU run");
            assert_eq!(&opt_r.raw[..32], &raw_r.raw[..32]);
        }
    }

    #[test]
    fn gpu_differential_divergent_wave_ops_not_cse_or_hoisted() {
        let supported = match runner::wave32_supported() {
            Ok(true) => true,
            Ok(false) => {
                runner::skip_test("no wave32");
                false
            }
            Err(e) => {
                runner::skip_test(&e);
                false
            }
        };

        let build = |do_opt: bool| -> Vec<u8> {
            let mut s = Shader::new(Stage::Compute, (6, 6));
            let res = s.raw_uav(1, 0);
            let tid = s.flattened_thread_id_in_group();
            let two = s.m.i32c(2);
            let off = s.m.bin(BinOp::Shl, tid, two);
            let c1 = s.m.i32c(1);
            let c16 = s.m.i32c(16);
            let then_b = s.m.new_block();
            let else_b = s.m.new_block();
            let join = s.m.new_block();

            s.m.set_block(Block(0));
            let in_high = s.m.cmp(Pred::UGe, tid, c16);
            let base_sum = s.wave_active_op(c1, WaveOp::Sum, false);
            s.m.cond_br(in_high, then_b, else_b);

            s.m.set_block(then_b);
            let then_sum = s.wave_active_op(c1, WaveOp::Sum, false);
            s.m.br(join);

            s.m.set_block(else_b);
            let else_sum = s.wave_active_op(c1, WaveOp::Sum, false);
            s.m.br(join);

            s.m.set_block(join);
            let i32 = s.m.i32();
            let br_sum = s.m.phi(i32);
            s.m.add_incoming(br_sum, then_sum, then_b);
            s.m.add_incoming(br_sum, else_sum, else_b);
            let c8 = s.m.i32c(8);
            let hi_bits = s.m.bin(BinOp::Shl, base_sum, c8);
            let packed = s.m.bin(BinOp::Add, hi_bits, br_sum);
            s.raw_store(res, off, &[packed]);
            s.m.ret();

            if do_opt {
                s.optimize();
            }
            s.finish([32, 1, 1], Some(32))
        };

        let raw_dxil = build(false);
        let opt_dxil = build(true);
        validate_if_supported(&opt_dxil);

        if supported {
            let mem = vec![0u32; 64];
            let raw_r = runner::run(&raw_dxil, &runner::KernelState::default(), &mem).expect("raw GPU run");
            let opt_r = runner::run(&opt_dxil, &runner::KernelState::default(), &mem).expect("opt GPU run");
            assert_eq!(&opt_r.raw[..32], &raw_r.raw[..32]);
            // Verify waveActiveOp inside each branch returned 16 (not 32 from hoisted base_sum).
            assert_eq!(opt_r.raw[0], (32 << 8) | 16);
            assert_eq!(opt_r.raw[16], (32 << 8) | 16);
        }
    }

    #[test]
    fn phi_with_undef_and_non_dominating_backedge_is_not_collapsed() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(1, 0);
        let i32 = s.m.i32();
        let entry = Block(0);
        let loop_b = s.m.new_block();
        let exit = s.m.new_block();

        s.m.set_block(entry);
        let c0 = s.m.i32c(0);
        let c1 = s.m.i32c(1);
        let c10 = s.m.i32c(10);
        let undef_i = s.m.undef(i32);
        s.m.br(loop_b);

        s.m.set_block(loop_b);
        let p = s.m.phi(i32);
        s.m.add_incoming(p, undef_i, entry);
        let next = s.m.bin(BinOp::Add, p, c1);
        s.m.add_incoming(p, next, loop_b);
        let go = s.m.cmp(Pred::ULt, next, c10);
        s.m.cond_br(go, loop_b, exit);

        s.m.set_block(exit);
        s.raw_store(uav, c0, &[next]);
        s.m.ret();

        s.optimize();
        assert!(module_stats(&s.m).phis >= 1, "phi with undef entry and non-dominating backedge must stay");
        let dxil = s.finish([1, 1, 1], Some(32));
        validate_if_supported(&dxil);
    }
}
