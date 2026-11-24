//! Resource tracking (decision 0003): where each descriptor a program uses
//! comes from, in terms of the root signature.
//!
//! Root parameters arrive in user SGPRs ([`RootParam`]): a descriptor table
//! as the 32-bit heap address of its range (the code sets the high half to
//! 0), a root descriptor as a 64-bit buffer address, root constants as
//! themselves. A root signature with dynamic resources (SM 6.6) adds the
//! 32-bit base addresses of both heaps; a descriptor is then loaded from
//! one of them at an index the code computes (from a root constant, say). The code loads table descriptors from heap memory with
//! `s_load_dword*`, and builds a buffer descriptor around a root
//! descriptor's address itself. Each descriptor operand and each scalar
//! memory address is traced through the SSA values back to those
//! parameters, by data flow: copies, phis that agree, and additions.

use std::collections::HashMap;

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// A multi-term, scaled, masked, or phi-joined byte offset relative to `base`:
/// the program computes `value = base + rel_offset` (modulo 2^32), plus an
/// optional `extra` term from the instruction's offset register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelOffset {
    pub value: Value,
    pub base: Value,
    pub extra: Option<Value>,
}

/// A byte address: a root parameter's, plus a constant and optional computed
/// offset (either a single value in `dynamic`, or a general relative
/// expression in `rel` for multi-term, scaled, masked, or phi-joined pointers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    /// Index into the layout, or [`CODE`].
    pub param: usize,
    /// First dword within `param` holding the low half of a 64-bit address
    /// (always 0 for `Descriptor`/`Table`/heap parameters; the low constant
    /// dword index for a 64-bit address passed in a `RootType::Constants`
    /// pair).
    pub param_dword: u32,
    /// Added modulo 2^32 (with carry into the high half of a root
    /// descriptor's address).
    pub offset: u32,
    pub dynamic: Option<Value>,
    pub rel: Option<RelOffset>,
}

/// [`Address::param`] of the program's own code: a constant table the
/// compiler appends after the code and reads through `s_getpc_b64`; the
/// offset is then the byte offset in the code.
pub const CODE: usize = usize::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Dwords `first..` of what the scalar load at `load` reads (see
    /// [`Resources::scalar`] for its address).
    Loaded { load: Site, first: u8 },
    /// A buffer descriptor built around an address: `bits` OR-ed into
    /// dword 1 (stride, swizzle), dwords 2 and 3 as given.
    Built {
        address: Address,
        bits: u32,
        words: [u32; 2],
    },
    /// Fixed in the code (an immediate sampler, say).
    Constant(Vec<u32>),
    /// Chosen by control flow: one source per predecessor of `block`, in
    /// [`crate::ir::Block::preds`] order.
    Phi { block: usize, args: Vec<Source> },
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } => args.iter().flat_map(Source::leaves).collect(),
            s => vec![s],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub site: Site,
    pub role: Role,
    pub source: Source,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    /// Every scalar memory access by address (`s_load_*`, `s_store_*`), and
    /// every global memory access (`global_*`, the dynamic term per lane).
    pub scalar: Vec<(Site, Address)>,
    pub descriptors: Vec<Descriptor>,
    /// User SGPRs and spilled user-data loads holding root parameters: value,
    /// layout index, dword within the parameter.
    pub params: Vec<(Value, usize, u32)>,
    /// `s_load_*` sites that read spilled user-data dwords (`16+`) through
    /// the spill-table pointer in user SGPRs `0..2`.
    pub spill_loads: Vec<Site>,
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace.
pub fn track(p: &Program, layout: &[RootParam], stage: HwStage) -> Result<Resources, ShaderError> {
    let t = Tracer::new(p, layout, stage)?;
    let mut params: Vec<(Value, usize, u32)> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| {
            let (param, dword) = t.param_of(reg)?;
            Some((v, param, dword))
        })
        .collect();
    let mut spilled: Vec<(Value, usize, u32)> = t
        .spilled
        .iter()
        .map(|(&v, &(param, dword))| (v, param, dword))
        .collect();
    spilled.sort_unstable_by_key(|&(v, _, _)| v.0);
    params.extend(spilled);
    let mut out = Resources {
        params,
        spill_loads: t.spill_loads.clone(),
        ..Resources::default()
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op).map_err(fail)?;
                out.descriptors.push(Descriptor { site, role, source });
                Ok(())
            };
            match &inst.fields {
                Fields::Image { rsrc, sampler, .. } => {
                    descriptor(rsrc, Role::Resource)?;
                    if let Some(s) = sampler {
                        descriptor(s, Role::Sampler)?;
                    }
                }
                Fields::Buffer { rsrc, .. } => descriptor(rsrc, Role::Resource)?,
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    descriptor(base, Role::Resource)?;
                }
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } if inst.name().contains("_load_") || inst.name().contains("_store_") => {
                    if t.spill_loads.contains(&site) {
                        continue;
                    }
                    let address = t.smem_address(inst, base, *offset, soffset).map_err(fail)?;
                    out.scalar.push((site, address));
                }
                Fields::Flat {
                    addr,
                    saddr,
                    offset,
                    ..
                } if inst.name().starts_with("global_") => {
                    let address = t
                        .flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                _ => {}
            }
        }
    }
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Const(u32),
    Value(Value),
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term, and whether the expression needs general relative
/// lowering (`multi`: multiple dynamic terms, shifts, subtractions, or phis).
#[derive(Clone, Copy)]
struct Sum {
    base: Option<Value>,
    offset: u32,
    dynamic: Option<Value>,
    multi: bool,
    cyclic: bool,
}

impl Sum {
    fn plain(offset: u32) -> Self {
        Self {
            base: None,
            offset,
            dynamic: None,
            multi: false,
            cyclic: false,
        }
    }

    fn dynamic(v: Value) -> Self {
        Self {
            base: None,
            offset: 0,
            dynamic: Some(v),
            multi: false,
            cyclic: false,
        }
    }

    fn base(v: Value) -> Self {
        Self {
            base: Some(v),
            offset: 0,
            dynamic: None,
            multi: false,
            cyclic: false,
        }
    }

    fn cyclic() -> Self {
        Self {
            base: None,
            offset: 0,
            dynamic: None,
            multi: false,
            cyclic: true,
        }
    }
}

struct Tracer<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    stage: HwStage,
    /// Defining instruction per value (phis and inputs have none).
    defs: Vec<Option<Site>>,
    /// Block and index of each phi value.
    phis: Vec<Option<(usize, usize)>>,
    /// Register per input value.
    inputs: Vec<Option<Reg>>,
    origin: Vec<Term>,
    /// Values loaded from a user-data spill table (`user_data >= 16`):
    /// layout index and dword within the parameter.
    spilled: HashMap<Value, (usize, u32)>,
    /// Values loaded from the spill table keyed by `(param, dword)` so both
    /// 64-bit `RootType::Descriptor` high halves (dword 1) and 64-bit
    /// `RootType::Constants` address pairs (`dword + 1`) can be verified.
    spill_dwords: HashMap<(usize, u32), Vec<Value>>,
    /// `s_load_*` sites that read the user-data spill table.
    spill_loads: Vec<Site>,
}

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;

impl<'a> Tracer<'a> {
    fn new(p: &'a Program, layout: &'a [RootParam], stage: HwStage) -> Result<Self, ShaderError> {
        let all = || {
            p.inputs
                .iter()
                .map(|i| i.1)
                .chain(p.blocks.iter().flat_map(|b| {
                    b.phis
                        .iter()
                        .map(|phi| phi.value)
                        .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1)))
                }))
        };
        let n = all().map(|v| v.0 as usize + 1).max().unwrap_or(0);
        let mut defs = vec![None; n];
        let mut phis = vec![None; n];
        let mut inputs = vec![None; n];
        for &(reg, v) in &p.inputs {
            inputs[v.0 as usize] = Some(reg);
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phis[phi.value.0 as usize] = Some((b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some((b, i));
                }
            }
        }
        let mut t = Tracer {
            p,
            layout,
            stage,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            spilled: HashMap::new(),
            spill_dwords: HashMap::new(),
            spill_loads: Vec::new(),
        };
        t.origin = t.propagate_copies(n)?;
        t.collect_spill_loads();
        Ok(t)
    }

    /// Optimistic copy propagation: a move's result is its source's
    /// origin, a phi's the one origin its arguments agree on.
    fn propagate_copies(&self, n: usize) -> Result<Vec<Term>, ShaderError> {
        let mut o: Vec<Option<Term>> = vec![None; n];
        for &(_, v) in &self.p.inputs {
            o[v.0 as usize] = Some(Term::Value(v));
        }
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for &b in &order {
                let block = &self.p.blocks[b];
                for phi in &block.phis {
                    let own = Term::Value(phi.value);
                    let mut args = phi
                        .args
                        .iter()
                        .filter_map(|a| o[a.0 as usize])
                        .filter(|&t| t != own);
                    let new = args
                        .next()
                        .map(|first| if args.all(|t| t == first) { first } else { own });
                    if new.is_some() && new != o[phi.value.0 as usize] {
                        o[phi.value.0 as usize] = new;
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    for (k, &(_, v)) in inst.writes.iter().enumerate() {
                        let new = match copy_source(inst, k).or_else(|| self.restored(inst)) {
                            Some(Moved::Const(c)) => Some(Term::Const(c)),
                            Some(Moved::Value(s)) => o[s.0 as usize],
                            None => Some(Term::Value(v)),
                        };
                        if new.is_some() && new != o[v.0 as usize] {
                            o[v.0 as usize] = new;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(o
                    .into_iter()
                    .enumerate()
                    .map(|(v, t)| t.unwrap_or(Term::Value(Value(v as u32))))
                    .collect());
            }
        }
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "copies that do not settle",
        })
    }

    fn inst(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        self.defs[v.0 as usize].map(|s| self.inst(s))
    }

    fn origin(&self, v: Value) -> Term {
        self.origin[v.0 as usize]
    }

    /// The origin of a one-dword operand as `inst` reads it.
    fn term(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code))?)),
            Operand::Const(c) => inline(c).map(Term::Const),
            Operand::Literal(x) => Some(Term::Const(x)),
            _ => None,
        }
    }

    /// Layout index and dword of the root parameter `reg` holds on entry.
    fn param_of(&self, reg: Reg) -> Option<(usize, u32)> {
        let Reg::S(code) = reg else { return None };
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let first = self.stage.user_sgpr(p.user_data);
            let dword = u32::from(code).checked_sub(first.into())?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// Layout index and dword of spilled user-data dword `ud`.
    fn param_of_user_data(&self, ud: u32) -> Option<(usize, u32)> {
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let dword = ud.checked_sub(u32::from(p.user_data))?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// Layout index and dword of root parameter value `v` (either an entry
    /// user SGPR or a value loaded from the user-data spill table).
    fn param_of_value(&self, v: Value) -> Option<(usize, u32)> {
        if let Some(reg) = self.inputs[v.0 as usize] {
            return self.param_of(reg);
        }
        self.spilled.get(&v).copied()
    }

    /// Recognizes `s_load_dword*` reads from the user-data spill table pointed
    /// to by user SGPRs `0..2` when the root layout places parameters at
    /// `user_data >= 16` (leaving `0..2` free for the 64-bit table pointer):
    /// byte `4 * k` holds user-data dword `16 + k`.
    fn collect_spill_loads(&mut self) {
        let has_spill = !self.layout.is_empty()
            && self.layout.iter().any(|rp| rp.user_data >= 16)
            && self.layout.iter().all(|rp| rp.user_data >= 2);
        if !has_spill {
            return;
        }
        let lo_reg = Reg::S(self.stage.user_sgpr(0));
        let hi_reg = Reg::S(self.stage.user_sgpr(1));
        let Some(spill_lo) = self.p.inputs.iter().find(|i| i.0 == lo_reg).map(|i| i.1) else {
            return;
        };
        let Some(spill_hi) = self.p.inputs.iter().find(|i| i.0 == hi_reg).map(|i| i.1) else {
            return;
        };
        for (b, block) in self.p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                let Fields::Smem {
                    data: Some(Operand::Scalar { code: dst, .. }),
                    base: Operand::Scalar { code, .. },
                    offset,
                    soffset,
                    ..
                } = &inst.fields
                else {
                    continue;
                };
                let Some(count) = smem_load_dwords(inst.name()) else {
                    continue;
                };
                let (Some(lo), Some(hi)) = (
                    inst.read(Reg::S(*code)).map(|v| self.origin(v)),
                    inst.read(Reg::S(code + 1)).map(|v| self.origin(v)),
                ) else {
                    continue;
                };
                let Some(lo_off) = self.spill_lo_offset(lo, spill_lo) else {
                    continue;
                };
                if !self.is_spill_hi(hi, spill_hi) {
                    continue;
                }
                let Some(Term::Const(soff)) = self.term(inst, soffset) else {
                    continue;
                };
                let byte_off = lo_off.wrapping_add(*offset as u32).wrapping_add(soff);
                if !byte_off.is_multiple_of(4) {
                    continue;
                }
                let first_ud = 16u32.saturating_add(byte_off / 4);
                for k in 0..count {
                    if let Some((param, dword)) = self.param_of_user_data(first_ud + k)
                        && let Some(v) = inst.write(Reg::S(*dst + k as u8))
                    {
                        self.spilled.insert(v, (param, dword));
                        self.spill_dwords.entry((param, dword)).or_default().push(v);
                    }
                }
                self.spill_loads.push((b, i));
            }
        }
    }

    /// Constant byte offset added to `spill_lo`, if `t` traces strictly to
    /// `spill_lo + const`.
    fn spill_lo_offset(&self, t: Term, spill_lo: Value) -> Option<u32> {
        let Term::Value(v) = t else { return None };
        if v == spill_lo {
            return Some(0);
        }
        let (_, [a, b]) = self.sop2(v, &["s_add_u32", "s_add_i32"])?;
        match (a, b) {
            (x, Term::Const(c)) | (Term::Const(c), x) => {
                self.spill_lo_offset(x, spill_lo).map(|o| o.wrapping_add(c))
            }
            _ => None,
        }
    }

    /// Whether `hi` is the spill-table high dword (optionally masked to a
    /// 40/48-bit address via `s_and_b32` / `s_bfe_u32` or carried by 0).
    fn is_spill_hi(&self, hi: Term, spill_hi: Value) -> bool {
        let Term::Value(v) = hi else { return false };
        if v == spill_hi {
            return true;
        }
        if let Some((_, [a, b])) = self.sop2(v, &["s_and_b32"]) {
            return matches!(b, Term::Const(_)) && self.is_spill_hi(a, spill_hi)
                || matches!(a, Term::Const(_)) && self.is_spill_hi(b, spill_hi);
        }
        if let Some((_, [a, b])) = self.sop2(v, &["s_bfe_u32"]) {
            return matches!(b, Term::Const(_)) && self.is_spill_hi(a, spill_hi);
        }
        if let Some((_, [a, b])) = self.sop2(v, &["s_addc_u32"]) {
            return (b == Term::Const(0) && self.is_spill_hi(a, spill_hi))
                || (a == Term::Const(0) && self.is_spill_hi(b, spill_hi));
        }
        false
    }

    /// The SOP2 operands of `v` if an `s_add` with one of `names` wrote it.
    fn sop2(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) || inst.writes.first()?.1 != v {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        Some((
            inst,
            [
                self.term(inst, src.first()?)?,
                self.term(inst, src.get(1)?)?,
            ],
        ))
    }

    /// Whether layout entry `param` holds a pointer/address (never root
    /// constants, so a constant dword 0 added to a root pointer is treated as
    /// a dynamic offset rather than a colliding second pointer base).
    fn is_pointer_param(&self, param: usize) -> bool {
        matches!(
            self.layout[param].root,
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap | RootType::Descriptor
        )
    }

    fn sum(&self, t: Term) -> Sum {
        self.sum_inner(t, &mut Vec::new())
    }

    fn combine_add(x: Sum, y: Sum, v: Value) -> Sum {
        if x.cyclic || y.cyclic {
            return Sum::cyclic();
        }
        if x.base.is_some() && y.base.is_some() {
            return Sum::dynamic(v);
        }
        let multi = x.multi || y.multi || (x.dynamic.is_some() && y.dynamic.is_some());
        Sum {
            base: x.base.or(y.base),
            offset: x.offset.wrapping_add(y.offset),
            dynamic: if multi {
                Some(v)
            } else {
                x.dynamic.or(y.dynamic)
            },
            multi,
            cyclic: false,
        }
    }

    fn sum_inner(&self, t: Term, visiting: &mut Vec<Value>) -> Sum {
        let v = match t {
            Term::Const(c) => return Sum::plain(c),
            Term::Value(v) => v,
        };
        if self
            .param_of_value(v)
            .is_some_and(|(p, d)| d == 0 && self.is_pointer_param(p))
        {
            return Sum::base(v);
        }
        if self.inputs[v.0 as usize].is_some() {
            return Sum::dynamic(v);
        }
        if self.getpc(v).is_some() {
            return Sum::base(v);
        }
        if visiting.contains(&v) {
            return Sum::cyclic();
        }
        if let Some((b, pi)) = self.phis[v.0 as usize] {
            visiting.push(v);
            let mut agreed: Option<Value> = None;
            let mut ok = true;
            for &arg in &self.p.blocks[b].phis[pi].args {
                let s = self.sum_inner(self.origin(arg), visiting);
                if s.cyclic {
                    continue;
                }
                match (agreed, s.base) {
                    (None, Some(base)) => agreed = Some(base),
                    (Some(prev), Some(base)) if prev == base => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            visiting.pop();
            if ok && let Some(base) = agreed {
                return Sum {
                    base: Some(base),
                    offset: 0,
                    dynamic: Some(v),
                    multi: true,
                    cyclic: false,
                };
            }
            return Sum::dynamic(v);
        }
        let Some(inst) = self.def(v) else {
            return Sum::dynamic(v);
        };
        visiting.push(v);
        let out = self.sum_inst(v, inst, visiting);
        visiting.pop();
        out
    }

    fn sum_inst(&self, v: Value, inst: &'a Inst, visiting: &mut Vec<Value>) -> Sum {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { dst, src, imm } if inst.writes.first().is_some_and(|w| w.1 == v) => {
                match name {
                    "s_mov_b32" => {
                        if let Some(t) = src.first().and_then(|o| self.term(inst, o)) {
                            return self.sum_inner(t, visiting);
                        }
                    }
                    "s_add_u32" | "s_add_i32" => {
                        if let (Some(a), Some(b)) = (
                            src.first().and_then(|o| self.term(inst, o)),
                            src.get(1).and_then(|o| self.term(inst, o)),
                        ) {
                            let (x, y) = (
                                self.sum_inner(a, visiting),
                                self.sum_inner(b, visiting),
                            );
                            return Self::combine_add(x, y, v);
                        }
                    }
                    "s_addk_i32" => {
                        if let (Some(d), Some(k)) = (dst.as_ref(), *imm)
                            && let Some(a) = self.term(inst, d)
                        {
                            let x = self.sum_inner(a, visiting);
                            let off = i32::from(k as i16) as u32;
                            return Self::combine_add(x, Sum::plain(off), v);
                        }
                    }
                    "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32"
                    | "s_lshl4_add_u32" => {
                        let shift = match name {
                            "s_lshl1_add_u32" => 1u32,
                            "s_lshl2_add_u32" => 2,
                            "s_lshl3_add_u32" => 3,
                            _ => 4,
                        };
                        if let (Some(a), Some(b)) = (
                            src.first().and_then(|o| self.term(inst, o)),
                            src.get(1).and_then(|o| self.term(inst, o)),
                        ) {
                            let sb = self.sum_inner(b, visiting);
                            if sb.cyclic {
                                return Sum::cyclic();
                            }
                            if let Term::Const(c) = a {
                                return Self::combine_add(sb, Sum::plain(c.wrapping_shl(shift)), v);
                            }
                            let sa = self.sum_inner(a, visiting);
                            if sa.cyclic {
                                return Sum::cyclic();
                            }
                            if sb.base.is_some() && sa.base.is_none() {
                                return Sum {
                                    base: sb.base,
                                    offset: sb.offset,
                                    dynamic: Some(v),
                                    multi: true,
                                    cyclic: false,
                                };
                            }
                        }
                    }
                    "s_sub_u32" | "s_sub_i32" => {
                        if let (Some(a), Some(b)) = (
                            src.first().and_then(|o| self.term(inst, o)),
                            src.get(1).and_then(|o| self.term(inst, o)),
                        ) {
                            let (sa, sb) = (
                                self.sum_inner(a, visiting),
                                self.sum_inner(b, visiting),
                            );
                            if sa.cyclic || sb.cyclic {
                                return Sum::cyclic();
                            }
                            if sa.base.is_some() && sb.base.is_none() {
                                if let Term::Const(c) = b {
                                    return Sum {
                                        base: sa.base,
                                        offset: sa.offset.wrapping_sub(c),
                                        dynamic: sa.dynamic,
                                        multi: sa.multi,
                                        cyclic: false,
                                    };
                                }
                                return Sum {
                                    base: sa.base,
                                    offset: sa.offset,
                                    dynamic: Some(v),
                                    multi: true,
                                    cyclic: false,
                                };
                            }
                        }
                    }
                    "s_or_b32" => {
                        if let (Some(a), Some(b)) = (
                            src.first().and_then(|o| self.term(inst, o)),
                            src.get(1).and_then(|o| self.term(inst, o)),
                        ) {
                            let (x, c) = match (a, b) {
                                (x, Term::Const(c)) | (Term::Const(c), x) if c < 256 => (x, c),
                                _ => return Sum::dynamic(v),
                            };
                            let sx = self.sum_inner(x, visiting);
                            if sx.cyclic {
                                return Sum::cyclic();
                            }
                            if sx.base.is_some() {
                                let _ = c;
                                return Sum {
                                    base: sx.base,
                                    offset: sx.offset,
                                    dynamic: Some(v),
                                    multi: true,
                                    cyclic: false,
                                };
                            }
                        }
                    }
                    _ => {}
                }
            }
            Fields::Valu(f)
                if f.abs == 0 && f.neg == 0 && f.sdwa.is_none() && f.dpp.is_none() =>
            {
                let writes_v = match f.dst {
                    Some(Operand::Vector { reg, .. }) => inst.write(Reg::V(reg)) == Some(v),
                    Some(Operand::Scalar { code, .. }) => inst.write(Reg::S(code)) == Some(v),
                    _ => false,
                };
                if !writes_v {
                    return Sum::dynamic(v);
                }
                match name {
                    "v_mov_b32" | "v_readfirstlane_b32" | "v_readlane_b32" => {
                        if let Some(a) = f.src.first().and_then(|o| self.vterm(inst, o)) {
                            let sa = self.sum_inner(a, visiting);
                            if sa.cyclic {
                                return Sum::cyclic();
                            }
                            if sa.base.is_some() && name != "v_mov_b32" {
                                return Sum {
                                    base: sa.base,
                                    offset: sa.offset,
                                    dynamic: Some(v),
                                    multi: true,
                                    cyclic: false,
                                };
                            }
                            return sa;
                        }
                    }
                    "v_add_co_u32" | "v_add_nc_u32" | "v_add_nc_i32" => {
                        if let (Some(a), Some(b)) = (
                            f.src.first().and_then(|o| self.vterm(inst, o)),
                            f.src.get(1).and_then(|o| self.vterm(inst, o)),
                        ) {
                            let (x, y) = (
                                self.sum_inner(a, visiting),
                                self.sum_inner(b, visiting),
                            );
                            return Self::combine_add(x, y, v);
                        }
                    }
                    "v_add3_u32" => {
                        if let (Some(a), Some(b), Some(c)) = (
                            f.src.first().and_then(|o| self.vterm(inst, o)),
                            f.src.get(1).and_then(|o| self.vterm(inst, o)),
                            f.src.get(2).and_then(|o| self.vterm(inst, o)),
                        ) {
                            let ab = Self::combine_add(
                                self.sum_inner(a, visiting),
                                self.sum_inner(b, visiting),
                                v,
                            );
                            let mut abc =
                                Self::combine_add(ab, self.sum_inner(c, visiting), v);
                            if abc.base.is_some() {
                                abc.multi = true;
                                abc.dynamic = Some(v);
                            }
                            return abc;
                        }
                    }
                    "v_lshl_add_u32" => {
                        if let (Some(a), Some(sh), Some(b)) = (
                            f.src.first().and_then(|o| self.vterm(inst, o)),
                            f.src.get(1).and_then(|o| self.vterm(inst, o)),
                            f.src.get(2).and_then(|o| self.vterm(inst, o)),
                        ) {
                            let sb = self.sum_inner(b, visiting);
                            if sb.cyclic {
                                return Sum::cyclic();
                            }
                            if let (Term::Const(cv), Term::Const(sv)) = (a, sh) {
                                return Self::combine_add(sb, Sum::plain(cv.wrapping_shl(sv & 31)), v);
                            }
                            let sa = self.sum_inner(a, visiting);
                            let ss = self.sum_inner(sh, visiting);
                            if sa.cyclic || ss.cyclic {
                                return Sum::cyclic();
                            }
                            if sb.base.is_some() && sa.base.is_none() && ss.base.is_none() {
                                return Sum {
                                    base: sb.base,
                                    offset: sb.offset,
                                    dynamic: Some(v),
                                    multi: true,
                                    cyclic: false,
                                };
                            }
                        }
                    }
                    "v_sub_co_u32" | "v_sub_nc_u32" | "v_subrev_co_u32" | "v_subrev_nc_u32" => {
                        if let (Some(s0), Some(s1)) = (
                            f.src.first().and_then(|o| self.vterm(inst, o)),
                            f.src.get(1).and_then(|o| self.vterm(inst, o)),
                        ) {
                            let (a, b) = if name.starts_with("v_subrev_") {
                                (s1, s0)
                            } else {
                                (s0, s1)
                            };
                            let (sa, sb) = (
                                self.sum_inner(a, visiting),
                                self.sum_inner(b, visiting),
                            );
                            if sa.cyclic || sb.cyclic {
                                return Sum::cyclic();
                            }
                            if sa.base.is_some() && sb.base.is_none() {
                                if let Term::Const(c) = b {
                                    return Sum {
                                        base: sa.base,
                                        offset: sa.offset.wrapping_sub(c),
                                        dynamic: sa.dynamic,
                                        multi: sa.multi,
                                        cyclic: false,
                                    };
                                }
                                return Sum {
                                    base: sa.base,
                                    offset: sa.offset,
                                    dynamic: Some(v),
                                    multi: true,
                                    cyclic: false,
                                };
                            }
                        }
                    }
                    "v_or_b32" => {
                        if let (Some(a), Some(b)) = (
                            f.src.first().and_then(|o| self.vterm(inst, o)),
                            f.src.get(1).and_then(|o| self.vterm(inst, o)),
                        ) {
                            let x = match (a, b) {
                                (x, Term::Const(c)) | (Term::Const(c), x) if c < 256 => x,
                                _ => return Sum::dynamic(v),
                            };
                            let sx = self.sum_inner(x, visiting);
                            if sx.cyclic {
                                return Sum::cyclic();
                            }
                            if sx.base.is_some() {
                                return Sum {
                                    base: sx.base,
                                    offset: sx.offset,
                                    dynamic: Some(v),
                                    multi: true,
                                    cyclic: false,
                                };
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Sum::dynamic(v)
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
    }

    /// The origin of a VALU operand.
    fn vterm(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(self.vorigin(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// A VGPR value's origin through `v_mov_b32` copies.
    fn vorigin(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.vterm(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A global memory address: an SGPR pair plus a VGPR offset, or a VGPR
    /// pair.
    fn flat_address(
        &self,
        inst: &Inst,
        addr: Option<&Operand>,
        saddr: Option<&Operand>,
        offset: i16,
    ) -> Result<Address, &'static str> {
        let Some(&Operand::Vector { reg, .. }) = addr else {
            return Err("a global address not in VGPRs");
        };
        let off = offset as i32 as u32;
        match saddr {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("an address not in SGPRs");
                };
                let v = inst.read(Reg::V(reg)).ok_or("a global address")?;
                let mut extra = self.sum(self.vorigin(v));
                if extra.base.is_some() {
                    return Err("an offset from a root parameter");
                }
                extra.offset = extra.offset.wrapping_add(off);
                self.address(lo, hi, extra)
            }
            _ => {
                let word = |k| inst.read(Reg::V(reg + k)).map(|v| self.vorigin(v));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("a global address");
                };
                self.address(lo, hi, Sum::plain(off))
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register).
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        let s = self.sum(lo);
        let base = s.base.ok_or_else(|| self.why_no_base(s.dynamic))?;
        let use_rel = s.multi || (s.dynamic.is_some() && extra.dynamic.is_some());
        let rel = if use_rel {
            let Term::Value(lo_val) = lo else {
                unreachable!("a sum with a base comes from a Value");
            };
            Some(RelOffset {
                value: lo_val,
                base,
                extra: extra.dynamic,
            })
        } else {
            None
        };
        let has_dynamic = s.dynamic.is_some() || extra.dynamic.is_some() || rel.is_some();
        let dynamic = s.dynamic.or(extra.dynamic);
        // When `rel` is used, `lo_val` already incorporates `s.offset`; keep
        // only `extra.offset` in `Address::offset` to avoid double-counting.
        let byte_offset = if rel.is_some() {
            extra.offset
        } else {
            s.offset.wrapping_add(extra.offset)
        };
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset.
            let high = pc.writes.get(1).map(|w| w.1);
            if !self.high_half(hi, high)
                && !high.is_some_and(|h| self.carried(hi, Term::Value(h)))
            {
                return Err("an address with an untracked high half");
            }
            return Ok(Address {
                param: CODE,
                param_dword: 0,
                offset: (pc.machine.offset as u32 + 4).wrapping_add(byte_offset),
                dynamic,
                rel,
            });
        }
        let (param, param_dword) = self
            .param_of_value(base)
            .ok_or("an address not from a root parameter")?;
        let matches_high_dword = |high_dword: u32| -> Result<bool, &'static str> {
            // shortcut: with a computed term, a high half that adds
            // anything to the base's is taken as the carry of that term
            // (offsets below 4 GB).
            let matches_high = |h: Value| {
                self.high_half(hi, Some(h))
                    || (has_dynamic && self.carried(hi, Term::Value(h)))
            };
            let rp = self.layout[param];
            let ud = u32::from(rp.user_data).saturating_add(high_dword);
            let mut found_any = false;
            if ud < 16 {
                let high = Reg::S(self.stage.user_sgpr(ud as u8));
                if let Some(h) = self.p.inputs.iter().find(|i| i.0 == high).map(|i| i.1) {
                    found_any = true;
                    if matches_high(h) {
                        return Ok(true);
                    }
                }
            }
            if let Some(highs) = self.spill_dwords.get(&(param, high_dword)) {
                found_any = true;
                if highs.iter().copied().any(matches_high) {
                    return Ok(true);
                }
            }
            if !found_any {
                return Err("a root descriptor without its high half");
            }
            Ok(false)
        };
        let high_ok = match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => {
                self.high_half(hi, None)
                    || (has_dynamic && self.carried(hi, Term::Const(0)))
            }
            RootType::Descriptor => {
                if param_dword != 0 {
                    return Err("a root descriptor without its low half");
                }
                matches_high_dword(1)?
            }
            // Ray-tracing builders and compute shaders can pass 64-bit GPU
            // virtual addresses in adjacent 32-bit root constants
            // (`dword` = lo, `dword + 1` = hi).
            RootType::Constants => {
                let Some(hi_dword) = param_dword.checked_add(1) else {
                    return Err("an address from a root parameter that is no address");
                };
                if hi_dword >= self.layout[param].sgprs {
                    return Err("an address from a root parameter that is no address");
                }
                matches_high_dword(hi_dword)?
            }
            _ => return Err("an address from a root parameter that is no address"),
        };
        if !high_ok {
            return Err("an address with an untracked high half");
        }
        Ok(Address {
            param,
            param_dword,
            offset: byte_offset,
            dynamic,
            rel,
        })
    }

    /// Strips VA-preserving masks (`s_and_b32`/`v_and_b32` preserving at least
    /// the low byte of the high VA dword, or `s_bfe_u32`) and plain copies.
    fn strip_va_mask(&self, mut t: Term) -> Term {
        for _ in 0..8 {
            let Term::Value(v) = t else { break };
            if let Some((_, [a, Term::Const(m)] | [Term::Const(m), a])) =
                self.sop2(v, &["s_and_b32"])
                && m & 0xFF == 0xFF
            {
                t = a;
                continue;
            }
            if let Some((_, [a, Term::Const(_)])) = self.sop2(v, &["s_bfe_u32"]) {
                t = a;
                continue;
            }
            if let Some(inst) = self.def(v)
                && let Fields::Valu(f) = &inst.fields
                && f.abs == 0
                && f.neg == 0
                && f.sdwa.is_none()
                && f.dpp.is_none()
            {
                if matches!(
                    inst.name(),
                    "v_mov_b32" | "v_readfirstlane_b32" | "v_readlane_b32"
                ) && let Some(a) = f.src.first().and_then(|o| self.vterm(inst, o))
                {
                    t = a;
                    continue;
                }
                if inst.name() == "v_and_b32"
                    && let (Some(a), Some(b)) = (
                        f.src.first().and_then(|o| self.vterm(inst, o)),
                        f.src.get(1).and_then(|o| self.vterm(inst, o)),
                    )
                {
                    match (a, b) {
                        (x, Term::Const(m)) | (Term::Const(m), x) if m & 0xFF == 0xFF => {
                            t = x;
                            continue;
                        }
                        _ => {}
                    }
                }
            }
            break;
        }
        t
    }

    /// `hi` is `high` (0 for a table), possibly plus a carry or joined by phi.
    fn high_half(&self, hi: Term, high: Option<Value>) -> bool {
        let want = high.map_or(Term::Const(0), Term::Value);
        self.high_half_inner(hi, want, &mut Vec::new())
    }

    fn high_half_inner(&self, hi: Term, want: Term, visiting: &mut Vec<Value>) -> bool {
        let hi = self.strip_va_mask(hi);
        if hi == want {
            return true;
        }
        let Term::Value(v) = hi else { return false };
        if visiting.contains(&v) {
            return true;
        }
        if let Some((b, pi)) = self.phis[v.0 as usize] {
            visiting.push(v);
            let mut non_cyclic = 0usize;
            let mut ok = true;
            for &arg in &self.p.blocks[b].phis[pi].args {
                let a = self.strip_va_mask(self.origin(arg));
                if let Term::Value(av) = a
                    && visiting.contains(&av)
                {
                    continue;
                }
                non_cyclic += 1;
                if !self.high_half_inner(a, want, visiting) {
                    ok = false;
                    break;
                }
            }
            visiting.pop();
            return ok && non_cyclic > 0;
        }
        visiting.push(v);
        let ok = if let Some((_, [a, b])) = self.sop2(v, &["s_addc_u32"]) {
            (self.high_half_inner(a, want, visiting) && b == Term::Const(0))
                || (self.high_half_inner(b, want, visiting) && a == Term::Const(0))
        } else if let Some(inst) = self.def(v)
            && let Fields::Valu(f) = &inst.fields
            && inst.name() == "v_add_co_ci_u32"
            && let (Some(a), Some(b)) = (
                f.src.first().and_then(|o| self.vterm(inst, o)),
                f.src.get(1).and_then(|o| self.vterm(inst, o)),
            )
        {
            (self.high_half_inner(a, want, visiting) && b == Term::Const(0))
                || (self.high_half_inner(b, want, visiting) && a == Term::Const(0))
        } else {
            false
        };
        visiting.pop();
        ok
    }

    /// `hi` adds something (with carry) to `want`: `s_addc_u32`,
    /// `v_add_co_ci_u32`, or a phi over such values.
    fn carried(&self, hi: Term, want: Term) -> bool {
        self.carried_inner(hi, want, &mut Vec::new())
    }

    fn carried_inner(&self, hi: Term, want: Term, visiting: &mut Vec<Value>) -> bool {
        let hi = self.strip_va_mask(hi);
        if hi == want {
            return true;
        }
        let Term::Value(v) = hi else { return false };
        if visiting.contains(&v) {
            return true;
        }
        if let Some((b, pi)) = self.phis[v.0 as usize] {
            visiting.push(v);
            let mut non_cyclic = 0usize;
            let mut ok = true;
            for &arg in &self.p.blocks[b].phis[pi].args {
                let a = self.strip_va_mask(self.origin(arg));
                if let Term::Value(av) = a
                    && visiting.contains(&av)
                {
                    continue;
                }
                non_cyclic += 1;
                if !self.carried_inner(a, want, visiting) {
                    ok = false;
                    break;
                }
            }
            visiting.pop();
            return ok && non_cyclic > 0;
        }
        visiting.push(v);
        let ok = if let Some((_, [a, b])) = self.sop2(v, &["s_addc_u32"]) {
            self.carried_inner(a, want, visiting) || self.carried_inner(b, want, visiting)
        } else if let Some(inst) = self.def(v)
            && let Fields::Valu(f) = &inst.fields
            && inst.name() == "v_add_co_ci_u32"
        {
            f.src[..2.min(f.src.len())]
                .iter()
                .any(|o| self.vterm(inst, o).is_some_and(|t| self.carried_inner(t, want, visiting)))
        } else {
            false
        };
        visiting.pop();
        ok
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if self.inputs[v.0 as usize].is_some() {
            return "an address in an SGPR that is no root parameter";
        }
        if self
            .leaves(v)
            .any(|l| self.def(l).is_some_and(|i| i.name() == "s_getpc_b64"))
        {
            return "an address relative to the program counter";
        }
        match self.def(v).map(|i| i.name()) {
            Some(n) if n.starts_with("s_load_") => "an address loaded from memory",
            None => "an address from a phi",
            _ => "an address not from a root parameter",
        }
    }

    /// The terms of the additions that compute `v`.
    fn leaves(&self, v: Value) -> Box<dyn Iterator<Item = Value> + '_> {
        match self.sop2(v, &["s_add_u32", "s_add_i32"]) {
            Some((_, terms)) => Box::new(terms.into_iter().flat_map(move |t| match t {
                Term::Value(x) => self.leaves(x),
                Term::Const(_) => Box::new(std::iter::empty()),
            })),
            None => Box::new(std::iter::once(v)),
        }
    }

    fn smem_address(
        &self,
        inst: &Inst,
        base: &Operand,
        offset: i32,
        soffset: &Operand,
    ) -> Result<Address, &'static str> {
        let Operand::Scalar { code, .. } = *base else {
            return Err("an address not in SGPRs");
        };
        let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
        let (Some(lo), Some(hi)) = (word(0), word(1)) else {
            return Err("an address not in SGPRs");
        };
        let mut extra = self.sum(
            self.term(inst, soffset)
                .ok_or("an unknown offset operand")?,
        );
        if extra.base.is_some() {
            return Err("an offset from a root parameter");
        }
        extra.offset = extra.offset.wrapping_add(offset as u32);
        self.address(lo, hi, extra)
    }

    fn descriptor(&self, inst: &Inst, op: &Operand) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        self.classify(&words, &mut Vec::new())
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved.
    fn classify(&self, words: &[Term], joins: &mut Vec<usize>) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(words, joins)? {
            return Ok(source);
        }
        if let Some(consts) = words
            .iter()
            .map(|w| match *w {
                Term::Const(c) => Some(c),
                Term::Value(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            return Ok(Source::Constant(consts));
        }
        if let Some(source) = self.loaded(words) {
            return Ok(source);
        }
        if let [lo, hi, Term::Const(w2), Term::Const(w3)] = *words {
            let (mut hi, mut bits) = (hi, 0);
            while let Term::Value(v) = hi
                && let Some((x, more)) = self.set_bits(v)
            {
                (hi, bits) = (x, bits | more);
            }
            let address = self.address(lo, hi, Sum::plain(0))?;
            return Ok(Source::Built {
                address,
                bits,
                words: [w2, w3],
            });
        }
        let loaded = |w: &Term| {
            matches!(*w, Term::Value(v)
                if self.def(v).is_some_and(|i| i.name().starts_with("s_load_")))
        };
        Err(if words.iter().any(loaded) {
            "a descriptor changed after loading"
        } else {
            "a descriptor neither loaded nor built"
        })
    }

    /// Words that phis of one block join, traced per predecessor (words
    /// defined before the join hold on every edge).
    fn phi(&self, words: &[Term], joins: &mut Vec<usize>) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        // Each level multiplies the sources by the predecessor count.
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let args = (0..self.p.blocks[block].preds.len())
            .map(|j| {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => self.origin(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.classify(&edge, joins)
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Phi { block, args: args? }))
    }

    /// `v` as some other dword with constant bits set (or VA-preserving high
    /// mask applied).
    fn set_bits(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(bits)] | [Term::Const(bits), x])) =
            self.sop2(v, &["s_or_b32"])
        {
            return Some((x, bits));
        }
        if let Some((_, [x, Term::Const(mask)] | [Term::Const(mask), x])) =
            self.sop2(v, &["s_and_b32"])
            && mask & 0xFF == 0xFF
        {
            return Some((x, 0));
        }
        let inst = self.def(v)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        match (inst.name(), self.term(inst, src.first()?)?) {
            ("s_bitset1_b32", Term::Const(bit)) => {
                Some((self.origin(inst.read(Reg::S(*code))?), 1 << (bit & 31)))
            }
            _ => None,
        }
    }

    /// Consecutive dwords of one scalar load.
    fn loaded(&self, words: &[Term]) -> Option<Source> {
        let Term::Value(v0) = words[0] else {
            return None;
        };
        let load = self.defs[v0.0 as usize]?;
        if self.spill_loads.contains(&load) {
            return None;
        }
        let inst = self.inst(load);
        if !inst.name().starts_with("s_load_") {
            return None;
        }
        let first = inst.writes.iter().position(|w| w.1 == v0)?;
        let same = words.iter().enumerate().all(|(k, w)| {
            inst.writes
                .get(first + k)
                .is_some_and(|x| *w == Term::Value(x.1))
        });
        same.then_some(Source::Loaded {
            load,
            first: first as u8,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Moved {
    Const(u32),
    Value(Value),
}

impl Tracer<'_> {
    /// For `v_readlane_b32` of a constant lane: the SGPR spilled to that
    /// lane, found by following the VGPR's `v_writelane_b32` writes.
    fn restored(&self, inst: &Inst) -> Option<Moved> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readlane_b32" {
            return None;
        }
        let [Operand::Vector { reg, .. }, lane] = v.src[..] else {
            return None;
        };
        self.spilled(inst.read(Reg::V(reg))?, lane_of(lane)?, &mut Vec::new())
            .ok()
            .flatten()
    }

    /// What lane `lane` of VGPR value `v` holds: `Ok(None)` for a phi in
    /// `seen` (already being resolved), `Err` if not a spilled SGPR.
    fn spilled(&self, v: Value, lane: u32, seen: &mut Vec<Value>) -> Result<Option<Moved>, ()> {
        if let Some((b, i)) = self.phis[v.0 as usize] {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled(a, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            return Ok(found);
        }
        let inst = self.def(v).ok_or(())?;
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        if inst.name() != "v_writelane_b32" {
            return Err(());
        }
        if lane_of(*at).ok_or(())? != lane {
            return self.spilled(inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => {
                inst.read(Reg::S(code)).map(Moved::Value)
            }
            Operand::Literal(x) => Some(Moved::Const(x)),
            Operand::Const(c) => inline(c).map(Moved::Const),
            _ => None,
        }
        .map(Some)
        .ok_or(())
    }
}

/// A lane select given as a constant.
fn lane_of(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// What dword `k` of a move's result copies.
fn copy_source(inst: &Inst, k: usize) -> Option<Moved> {
    let Fields::Sop { src, imm, .. } = &inst.fields else {
        return None;
    };
    if inst.name() == "s_movk_i32" {
        return Some(Moved::Const(i32::from((*imm)? as i16) as u32));
    }
    let src = src.first()?;
    match (inst.name(), *src) {
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { code, .. }) if code != NULL => {
            inst.read(Reg::S(code + k as u8)).map(Moved::Value)
        }
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { .. }) => Some(Moved::Const(0)),
        ("s_mov_b32", Operand::Literal(x)) => Some(Moved::Const(x)),
        ("s_mov_b32", Operand::Const(c)) => inline(c).map(Moved::Const),
        // shortcut: only the one-operand ways the compiler makes a constant
        // that has no inline form; fold more if descriptors need it.
        ("s_brev_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(x.reverse_bits())),
        ("s_not_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(!x)),
        // 64-bit inline integers are sign-extended.
        ("s_mov_b64", Operand::Const(c @ 128..=208)) => {
            let low = inline(c)?;
            Some(Moved::Const(if k == 0 {
                low
            } else {
                ((low as i32) >> 31) as u32
            }))
        }
        _ => None,
    }
}

/// The 32-bit value of an inline constant operand.
pub(crate) fn inline(code: u8) -> Option<u32> {
    let float = |f: f32| Some(f.to_bits());
    match code {
        128..=192 => Some(u32::from(code) - 128),
        193..=208 => Some((192 - i32::from(code)) as u32),
        240 => float(0.5),
        241 => float(-0.5),
        242 => float(1.0),
        243 => float(-1.0),
        244 => float(2.0),
        245 => float(-2.0),
        246 => float(4.0),
        247 => float(-4.0),
        248 => Some(0x3E22_F983),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        track(&crate::ir::build(&code, false, 256).unwrap(), layout, stage).unwrap()
    }

    fn param(root: RootType, user_data: u8, sgprs: u32) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    fn at(param: usize, offset: u32) -> Address {
        Address {
            param,
            param_dword: 0,
            offset,
            dynamic: None,
            rel: None,
        }
    }

    #[test]
    fn table_descriptors_are_loaded_from_their_range() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE89_0380, // s_mov_b32 s9, 0
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF408_0204,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
                0xF40C_0001,
                0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
                0xF080_0F08,
                0x0040_0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
                S_ENDPGM,
            ],
            &[param(Table, 8, 1), param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0)), ((0, 4), at(1, 0))]);
        let roles: Vec<_> = r.descriptors.iter().map(|d| (d.role, &d.source)).collect();
        assert_eq!(
            roles,
            [
                (
                    Role::Resource,
                    &Source::Loaded {
                        load: (0, 4),
                        first: 0
                    }
                ),
                (
                    Role::Sampler,
                    &Source::Loaded {
                        load: (0, 3),
                        first: 0
                    }
                ),
            ]
        );
    }

    #[test]
    fn root_descriptors_are_built_in_code() {
        let r = track_words(
            &[
                0x8801_FF0B,
                0x0010_0000, // s_or_b32 s1, s11, 0x100000
                0xBE80_030A, // s_mov_b32 s0, s10
                0xBE82_03C1, // s_mov_b32 s2, -1
                0xBE83_03FF,
                0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
                0xF430_0000,
                0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
                S_ENDPGM,
            ],
            &[param(Descriptor, 10, 2)],
            HwStage::Vs,
        );
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        assert_eq!(
            r.params.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
    }

    #[test]
    fn heap_indices_from_root_constants_in_a_merged_stage() {
        // User data starts at s8: the constant (index 6) is s14, the heap
        // (index 10) s18.
        let r = track_words(
            &[
                0x8F08_850E, // s_lshl_b32 s8, s14, 5
                0xBE93_0380, // s_mov_b32 s19, 0
                0xF408_0209,
                0x1000_0000, // s_load_dwordx4 s[8:11], s[18:19], s8
                0xE000_2000,
                0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
                S_ENDPGM,
            ],
            &[param(Constants, 6, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ms,
        );
        let index = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        let shift = r.scalar[0].1.dynamic.unwrap();
        assert_eq!(r.scalar[0].1.param, 1);
        // The computed term is the shifted constant, not the constant itself.
        assert_ne!(shift, index);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 2),
                first: 0
            }
        );
    }

    #[test]
    fn spilled_sgprs_are_followed_through_vgpr_lanes() {
        let r = track_words(
            &[
                0xD761_0000,
                0x0001_0609, // v_writelane_b32 v0, s9, 3
                0xD760_0002,
                0x0001_0700, // v_readlane_b32 s2, v0, 3
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0001,
                0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0x20))]);
    }

    #[test]
    fn descriptors_joined_by_control_flow_are_traced_per_edge() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0003, // s_cbranch_scc1 -> 0x1c
                0xF408_0001,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
                0xBF82_0002, // s_branch -> 0x24
                0xF408_0001,
                0xFA00_0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
                0xF420_0100,
                0xFA00_0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Phi { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        let offsets: Vec<_> = args
            .iter()
            .map(|a| match a {
                Source::Loaded { load, .. } => {
                    r.scalar.iter().find(|s| s.0 == *load).unwrap().1.offset
                }
                a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets.len(), 2);
        assert!(
            offsets.contains(&0) && offsets.contains(&0x10),
            "{offsets:?}"
        );
    }

    #[test]
    fn untraceable_addresses_fail_with_a_reason() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 0,
                what: "an address in an SGPR that is no root parameter"
            }
        );
    }

    #[test]
    fn spill_table_loads_map_to_root_parameters_and_addresses() {
        // Compute shader whose root layout starts at user_data 16 (spilled),
        // with s[0:1] holding the spill-table pointer (high dword masked to
        // 40 bits via s_and_b32 s1, s1, 0xff).
        let r = track_words(
            &[
                0x8701_FF01,
                0x0000_00FF, // s_and_b32 s1, s1, 0xff
                0xF408_0100,
                0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], null (ud 16..20)
                0x7E00_0280, // v_mov_b32 v0, 0
                0x7E02_0206, // v_mov_b32 v1, s6 (root constant from ud 18)
                0xDC70_8000,
                0x0004_0100, // global_store_dword v0, v1, s[4:5] (root desc at ud 16..18)
                S_ENDPGM,
            ],
            &[param(Descriptor, 16, 2), param(Constants, 18, 2)],
            HwStage::Cs,
        );
        assert_eq!(r.spill_loads, [(0, 1)]);
        assert_eq!(r.scalar, [((0, 4), at(0, 0))]);
        assert!(r.params.iter().any(|&(_, p, d)| p == 0 && d == 0));
        assert!(r.params.iter().any(|&(_, p, d)| p == 0 && d == 1));
        assert!(r.params.iter().any(|&(_, p, d)| p == 1 && d == 0));
        assert!(r.params.iter().any(|&(_, p, d)| p == 1 && d == 1));
    }

    #[test]
    fn root_constant_dword0_does_not_collide_with_pointer_base() {
        // s[0:1] = RootType::Descriptor, s2 = RootType::Constants dword 0.
        // s4 = s0 + s2; s5 = s1 + carry; s_load_dword s6, s[4:5], 0
        let r = track_words(
            &[
                0x8004_0200, // s_add_u32 s4, s0, s2
                0x8205_8001, // s_addc_u32 s5, s1, 0
                0xF400_0182,
                0xFA00_0000, // s_load_dword s6, s[4:5], 0
                S_ENDPGM,
            ],
            &[param(Descriptor, 0, 2), param(Constants, 2, 1)],
            HwStage::Cs,
        );
        assert_eq!(r.scalar.len(), 1);
        assert_eq!(r.scalar[0].1.param, 0);
        assert!(r.scalar[0].1.dynamic.is_some());
    }

    #[test]
    fn scaled_multi_term_address_uses_rel_offset() {
        // s[0:1] = RootType::Descriptor, s2..s3 = RootType::Constants.
        // s4 = (s3 << 2) + s0; s4 = s4 + s2; s5 = s1 + carry; s_load_dword s6, s[4:5], 0x10
        let r = track_words(
            &[
                0x9784_0003, // s_lshl2_add_u32 s4, s3, s0
                0x8004_0204, // s_add_u32 s4, s4, s2
                0x8205_8001, // s_addc_u32 s5, s1, 0
                0xF400_0182,
                0xFA00_0010, // s_load_dword s6, s[4:5], 0x10
                S_ENDPGM,
            ],
            &[param(Descriptor, 0, 2), param(Constants, 2, 2)],
            HwStage::Cs,
        );
        assert_eq!(r.scalar.len(), 1);
        assert_eq!(r.scalar[0].1.param, 0);
        assert_eq!(r.scalar[0].1.offset, 0x10);
        assert!(r.scalar[0].1.rel.is_some());
    }
}

fn smem_load_dwords(name: &str) -> Option<u32> {
    match name.strip_prefix("s_load_dword")? {
        "" => Some(1),
        "x2" => Some(2),
        "x4" => Some(4),
        "x8" => Some(8),
        "x16" => Some(16),
        _ => None,
    }
}
