//! How far past its root address a shader can reach through each root UAV.
//!
//! A root view carries no size, so the host would otherwise take the whole
//! rest of the title buffer as read and written. The bound comes from a
//! range analysis of the built DXIL. Each access through a root UAV handle
//! gets the largest byte offset it can use. That offset is a constant plus
//! terms in the dispatch's group counts (thread ids, group ids, lanes and
//! constants combined by adds, constant multiplies and shifts, masks,
//! remainders and the like), plus terms in user-data dwords, which the host
//! knows when it binds the shader. Anything read from memory makes the
//! slot's extent unknown.

use std::collections::HashMap;

use super::RootUavExtent;
use super::bitcode::{Bin, Cast, Def, Module, V, pred};

/// How many user-data dwords one bound can name.
pub const USER_TERMS: usize = 4;

/// `c + Σ g[i] * (groups[i] - 1) + Σ k * user[dword]`: an upper bound in the
/// dispatch's group counts and the user-data dwords the shader gets at bind
/// time, with non-negative coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Affine {
    pub c: u64,
    pub g: [u64; 3],
    /// `(dword, k)`: user-data dword `dword` times `k` (unused: `k` 0).
    pub u: [(u8, u64); USER_TERMS],
}

impl Affine {
    pub const fn constant(c: u64) -> Self {
        Affine {
            c,
            g: [0; 3],
            u: [(0, 0); USER_TERMS],
        }
    }

    fn user(dword: u8) -> Self {
        let mut a = Affine::constant(0);
        a.u[0] = (dword, 1);
        a
    }

    pub fn is_constant(&self) -> bool {
        self.g == [0; 3] && self.u.iter().all(|t| t.1 == 0)
    }

    /// Its value for a dispatch of `groups` with user-data dwords `user`
    /// (`None` past `u64`, for a term in groups when there is no dispatch,
    /// or for a dword `user` does not know).
    pub fn eval(&self, groups: Option<[u32; 3]>, user: impl Fn(u8) -> Option<u32>) -> Option<u64> {
        let mut x = self.c;
        if self.g != [0; 3] {
            for (g, n) in self.g.iter().zip(groups?) {
                x = x.checked_add(g.checked_mul(u64::from(n.saturating_sub(1)))?)?;
            }
        }
        for &(dword, k) in &self.u {
            if k != 0 {
                x = x.checked_add(k.checked_mul(u64::from(user(dword)?))?)?;
            }
        }
        Some(x)
    }

    /// `f` of each pair of coefficients (0 where one side lacks a term).
    fn zip(self, o: Affine, f: impl Fn(u64, u64) -> Option<u64>) -> Option<Affine> {
        let mut u = [(0, 0); USER_TERMS];
        let mut n = 0;
        let mut put = |dword: u8, k: u64| -> Option<()> {
            if k != 0 {
                *u.get_mut(n)? = (dword, k);
                n += 1;
            }
            Some(())
        };
        let of = |a: &Affine, dword: u8| {
            a.u.iter()
                .find(|t| t.1 != 0 && t.0 == dword)
                .map_or(0, |t| t.1)
        };
        for &(dword, k) in self.u.iter().filter(|t| t.1 != 0) {
            put(dword, f(k, of(&o, dword))?)?;
        }
        for &(dword, k) in o.u.iter().filter(|t| t.1 != 0) {
            if of(&self, dword) == 0 {
                put(dword, f(0, k)?)?;
            }
        }
        Some(Affine {
            c: f(self.c, o.c)?,
            g: [
                f(self.g[0], o.g[0])?,
                f(self.g[1], o.g[1])?,
                f(self.g[2], o.g[2])?,
            ],
            u,
        })
    }

    fn add(self, o: Affine) -> Option<Affine> {
        self.zip(o, u64::checked_add)
    }

    fn scale(self, k: u64) -> Option<Affine> {
        self.zip(Affine::constant(0), |a, _| a.checked_mul(k))
    }

    /// Componentwise: at least either bound for every dispatch (`None` if
    /// they name too many dwords together).
    fn max(self, o: Affine) -> Option<Affine> {
        self.zip(o, |a, b| Some(a.max(b)))
    }

    /// Floor of the constant, ceiling of the coefficients: at least the
    /// quotient of the bound by `d`.
    fn div(self, d: u64) -> Affine {
        let mut a = self;
        a.c /= d;
        a.g = a.g.map(|g| g.div_ceil(d));
        for t in &mut a.u {
            t.1 = t.1.div_ceil(d);
        }
        a
    }
}

/// The values an integer can take: at least `lo`, at most `hi`. `peak`
/// bounds every intermediate result the value was computed from; the bounds
/// hold if it stays below 2^32 (no 32-bit operation wrapped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Range {
    lo: u64,
    hi: Affine,
    peak: Affine,
}

impl Range {
    fn exact(x: u64) -> Range {
        Range {
            lo: x,
            hi: Affine::constant(x),
            peak: Affine::constant(x),
        }
    }

    /// At most `hi`, whatever the inputs were (a mask, a remainder).
    fn below(hi: u64) -> Range {
        Range {
            lo: 0,
            hi: Affine::constant(hi),
            peak: Affine::constant(hi),
        }
    }

    fn constant(&self) -> Option<u64> {
        (self.hi.is_constant() && self.lo == self.hi.c).then_some(self.lo)
    }

    /// The smaller of two values: a bound of either side holds, with that
    /// side's peak.
    fn min(a: Range, b: Range) -> Range {
        let pick_b = match (a.hi.is_constant(), b.hi.is_constant()) {
            (true, true) => b.hi.c < a.hi.c,
            (false, true) => true,
            _ => false,
        };
        let r = if pick_b { b } else { a };
        Range {
            lo: a.lo.min(b.lo),
            ..r
        }
    }

    /// The smaller of two values when either may be unbounded.
    fn min_of(a: Option<Range>, b: Option<Range>) -> Option<Range> {
        match (a, b) {
            (Some(a), Some(b)) => Some(Range::min(a, b)),
            (Some(r), None) | (None, Some(r)) => Some(Range { lo: 0, ..r }),
            (None, None) => None,
        }
    }

    /// Either value.
    fn either(a: Range, b: Range) -> Option<Range> {
        Some(Range {
            lo: a.lo.min(b.lo),
            hi: a.hi.max(b.hi)?,
            peak: a.peak.max(b.peak)?,
        })
    }
}

/// What a compute shader's ids can be.
pub(crate) struct Ids<'a> {
    /// The thread-group size (compute and mesh).
    pub threads: Option<[u32; 3]>,
    /// The largest host lane index plus one.
    pub wave: u32,
    /// Values that are group id component `.1` of the whole dispatch (a
    /// split dispatch's offset already added).
    pub group_ids: &'a [(V, u8)],
    /// The user-data constant buffer's handle: its dwords are known when
    /// the shader is bound.
    pub user_data: Option<V>,
}

const DEPTH: u32 = 400;

struct Ranges<'a> {
    m: &'a Module,
    ids: &'a Ids<'a>,
    memo: HashMap<V, Option<Range>>,
    /// Phis being evaluated: reaching one again is a loop.
    open: Vec<V>,
}

impl Ranges<'_> {
    fn range(&mut self, v: V, depth: u32) -> Option<Range> {
        if let Some(&r) = self.memo.get(&v) {
            return r;
        }
        if depth == 0 || self.open.contains(&v) {
            return None;
        }
        let r = self.compute(v, depth - 1);
        self.memo.insert(v, r);
        r
    }

    fn compute(&mut self, v: V, depth: u32) -> Option<Range> {
        if let Some(&(_, c)) = self.ids.group_ids.iter().find(|g| g.0 == v) {
            return Some(group(c));
        }
        match self.m.def(v) {
            Def::Int(x) => Some(Range::exact(x)),
            Def::Call(name, args) => self.call(name, args, depth),
            Def::Bin(op, a, b) => {
                let ra = self.range(a, depth);
                let rb = self.range(b, depth);
                bin(op, ra, rb, self.m.int_bits(self.m.type_of(v))?)
            }
            Def::Cast(op, a, bits) => {
                let from = self.m.int_bits(self.m.type_of(a))?;
                if from == 1 && matches!(op, Cast::ZExt) {
                    return Some(Range::below(1));
                }
                let r = self.range(a, depth)?;
                match op {
                    Cast::ZExt => Some(r),
                    Cast::SExt => (r.hi.is_constant() && r.hi.c < 1 << (from - 1)).then_some(r),
                    Cast::Trunc if bits >= 32 => Some(r),
                    Cast::Trunc if bits > 0 => Some(mask(r, (1 << bits) - 1)),
                    _ => None,
                }
            }
            Def::Cmp(..) => Some(Range::below(1)),
            Def::Select(c, t, f) => {
                let rt = self.range(t, depth);
                let rf = self.range(f, depth);
                // `x < y ? x : y` and its mirror forms are a minimum.
                if let Def::Cmp(p, x, y) = self.m.def(c) {
                    let less = matches!(p, pred::IULT | pred::IULE);
                    let more = matches!(p, pred::IUGT | pred::IUGE);
                    if (less && (t, f) == (x, y)) || (more && (t, f) == (y, x)) {
                        return Range::min_of(rt, rf);
                    }
                }
                Range::either(rt?, rf?)
            }
            Def::Phi(incoming) => {
                self.open.push(v);
                let mut r: Option<Range> = None;
                let mut known = true;
                for &(x, _) in incoming {
                    match self.range(x, depth) {
                        Some(x) => match r.map_or(Some(x), |r| Range::either(r, x)) {
                            Some(x) => r = Some(x),
                            None => {
                                known = false;
                                break;
                            }
                        },
                        None => {
                            known = false;
                            break;
                        }
                    }
                }
                self.open.pop();
                if known { r } else { None }
            }
            Def::Extract(agg, k) => {
                // A user-data dword: `cbufferLoadLegacy(user data, row)`.
                let Def::Call("dx.op.cbufferLoadLegacy.i32", &[_, h, row]) = self.m.def(agg) else {
                    return None;
                };
                let Def::Int(row) = self.m.def(row) else {
                    return None;
                };
                if Some(h) != self.ids.user_data || k > 3 {
                    return None;
                }
                let dword = u8::try_from(row.checked_mul(4)? + u64::from(k)).ok()?;
                let hi = Affine::user(dword);
                Some(Range {
                    lo: 0,
                    hi,
                    peak: hi,
                })
            }
            Def::Other => None,
        }
    }

    fn call(&mut self, name: &str, args: &[V], depth: u32) -> Option<Range> {
        name.strip_prefix("dx.op.")?;
        let Def::Int(opcode) = self.m.def(*args.first()?) else {
            return None;
        };
        let component = |k: usize| match args.get(k).map(|&a| self.m.def(a)) {
            Some(Def::Int(c)) if c < 3 => Some(c as usize),
            _ => None,
        };
        let threads = self.ids.threads;
        match opcode {
            // threadId: group id * size + thread id in the group.
            93 => {
                let c = component(1)?;
                let t = u64::from(threads?[c]);
                let mut hi = Affine::constant(t - 1);
                hi.g[c] = t;
                Some(Range {
                    lo: 0,
                    hi,
                    peak: hi,
                })
            }
            94 => Some(group(component(1)? as u8)),
            95 => Some(Range::below(u64::from(threads?[component(1)?]) - 1)),
            96 => {
                let [x, y, z] = threads?.map(u64::from);
                Some(Range::below(x * y * z - 1))
            }
            // waveGetLaneIndex
            111 => Some(Range::below(u64::from(self.ids.wave) - 1)),
            // waveReadLaneAt, waveReadLaneFirst: another lane's value.
            117 | 118 => self.range(*args.get(1)?, depth),
            // UMax, UMin
            39 | 40 => {
                let a = self.range(*args.get(1)?, depth);
                let b = self.range(*args.get(2)?, depth);
                if opcode == 40 {
                    return Range::min_of(a, b);
                }
                let (a, b) = (a?, b?);
                Some(Range {
                    lo: a.lo.max(b.lo),
                    ..Range::either(a, b)?
                })
            }
            _ => None,
        }
    }
}

fn group(c: u8) -> Range {
    let mut hi = Affine::default();
    hi.g[usize::from(c)] = 1;
    Range {
        lo: 0,
        hi,
        peak: hi,
    }
}

/// `x op k` at `bits` bits (as LLVM computes it: shifts past the width
/// and division by zero are not folded).
fn fold(op: Bin, x: u64, k: u64, bits: u32) -> Option<u64> {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    };
    let sign = |v: u64| -> i64 {
        if bits >= 64 {
            v as i64
        } else {
            ((v << (64 - bits)) as i64) >> (64 - bits)
        }
    };
    let shift = (k < u64::from(bits)).then_some(k as u32);
    let r = match op {
        Bin::Add => x.wrapping_add(k),
        Bin::Sub => x.wrapping_sub(k),
        Bin::Mul => x.wrapping_mul(k),
        Bin::UDiv => x.checked_div(k)?,
        Bin::URem => x.checked_rem(k)?,
        Bin::SDiv => sign(x).checked_div(sign(k))? as u64,
        Bin::Shl => x << shift?,
        Bin::LShr => x >> shift?,
        Bin::AShr => (sign(x) >> shift?) as u64,
        Bin::And => x & k,
        Bin::Or => x | k,
        Bin::Xor => x ^ k,
    };
    Some(r & mask)
}

/// `r & m` for a constant `m` and a value that may be unbounded.
fn mask_of(r: Option<Range>, m: u64) -> Range {
    r.map_or(Range::below(m), |r| mask(r, m))
}

/// `r & m` for a constant `m`: at most `m` whatever `r` is, and at most `r`.
fn mask(r: Range, m: u64) -> Range {
    if r.hi.is_constant() && r.hi.c <= m {
        return Range { lo: 0, ..r };
    }
    if r.hi.is_constant() || m <= r.hi.c || m < 1 << 16 {
        return Range::below(m);
    }
    Range { lo: 0, ..r }
}

fn bin(op: Bin, a: Option<Range>, b: Option<Range>, bits: u32) -> Option<Range> {
    let k = b.and_then(|b| b.constant());
    if let (Some(x), Some(k)) = (a.and_then(|a| a.constant()), k)
        && let Some(x) = fold(op, x, k, bits)
    {
        return Some(Range::exact(x));
    }
    match op {
        // Bounded whatever the other operand is.
        Bin::URem => {
            let k = k.filter(|&k| k > 0)?;
            if let Some(a) = a
                && a.hi.is_constant()
                && a.hi.c < k
            {
                return Some(a);
            }
            return Some(Range::below(k - 1));
        }
        Bin::And => {
            return match (a.and_then(|a| a.constant()), k) {
                (_, Some(k)) => Some(mask_of(a, k)),
                (Some(k), _) => Some(mask_of(b, k)),
                _ => Range::min_of(a, b).map(|r| Range { lo: 0, ..r }),
            };
        }
        _ => {}
    }
    let a = a?;
    let grow = |lo: u64, hi: Affine, peak: Option<Affine>| {
        Some(Range {
            lo,
            hi,
            peak: peak?.max(hi)?,
        })
    };
    match op {
        Bin::Add => {
            let b = b?;
            grow(a.lo.checked_add(b.lo)?, a.hi.add(b.hi)?, a.peak.max(b.peak))
        }
        Bin::Sub => {
            // Never below zero: the result is at most `a`.
            let k = k?;
            if a.lo < k {
                return None;
            }
            let mut hi = a.hi;
            hi.c = hi.c.checked_sub(k).unwrap_or(hi.c);
            Some(Range {
                lo: a.lo - k,
                hi,
                peak: a.peak,
            })
        }
        Bin::Mul => {
            // Both non-negative: at most the bound of one side times the
            // other's, when one of them is a constant.
            let b = b?;
            let (x, y) = if b.hi.is_constant() { (a, b) } else { (b, a) };
            if !y.hi.is_constant() {
                return None;
            }
            grow(
                x.lo.checked_mul(y.lo)?,
                x.hi.scale(y.hi.c)?,
                x.peak.max(y.peak),
            )
        }
        Bin::Shl => {
            // By at most a constant amount below the width.
            let b = b?;
            if !b.hi.is_constant() || b.hi.c >= u64::from(bits) {
                return None;
            }
            let f = 1u64.checked_shl(b.hi.c as u32)?;
            let lo =
                a.lo.checked_shl(b.lo as u32)
                    .filter(|&lo| lo >> b.lo == a.lo)?;
            grow(lo, a.hi.scale(f)?, a.peak.max(b.peak))
        }
        Bin::LShr | Bin::UDiv => {
            let d = match op {
                Bin::LShr => 1u64.checked_shl(k.filter(|&k| k < 64)? as u32)?,
                _ => k.filter(|&k| k > 0)?,
            };
            Some(Range {
                lo: a.lo / d,
                hi: a.hi.div(d),
                peak: a.peak,
            })
        }
        Bin::AShr => {
            // A non-negative value shifts as with LShr.
            let k = k.filter(|&k| k < 64)?;
            (a.hi.is_constant() && a.hi.c < 1 << (bits - 1)).then(|| Range {
                lo: a.lo >> k,
                hi: Affine::constant(a.hi.c >> k),
                peak: a.peak,
            })
        }
        Bin::URem | Bin::And => unreachable!("bounded above"),
        Bin::Or | Bin::Xor => {
            // Both at most the sum; constant bounds also at most the bits
            // the larger one spans.
            let b = b?;
            let mut hi = a.hi.add(b.hi)?;
            if hi.is_constant() {
                let top = a.hi.c.max(b.hi.c);
                let span = u64::MAX >> top.leading_zeros().min(63);
                hi.c = hi.c.min(span.max(top));
            }
            let lo = if op == Bin::Or { a.lo.max(b.lo) } else { 0 };
            grow(lo, hi, a.peak.max(b.peak))
        }
        Bin::SDiv => None,
    }
}

/// A DXIL access through a raw buffer handle: its byte offset and how many
/// bytes it reaches from there.
fn access(m: &Module, name: &str, args: &[V]) -> Option<(V, V, u64)> {
    let op = name.strip_prefix("dx.op.")?;
    let (kind, overload) = op.split_once('.')?;
    let size = match overload {
        "i16" | "f16" => 2,
        "i32" | "f32" => 4,
        "i64" | "f64" => 8,
        _ => return None,
    };
    let lanes = |mask: V| match m.def(mask) {
        Def::Int(x) if x & 0xf != 0 => Some(u64::from(64 - (x & 0xf).leading_zeros())),
        _ => None,
    };
    let at = |k: usize| args.get(k).copied();
    match kind {
        // opcode, handle, offset, (unused), 4 values, mask, alignment
        "rawBufferStore" => Some((at(1)?, at(2)?, size * lanes(at(8)?)?)),
        // opcode, handle, offset, (unused), mask, alignment
        "rawBufferLoad" => Some((at(1)?, at(2)?, size * lanes(at(4)?)?)),
        // opcode, handle, operation, 3 offsets, value
        "atomicBinOp" => Some((at(1)?, at(3)?, size)),
        // opcode, handle, 3 offsets, compared, new
        "atomicCompareExchange" => Some((at(1)?, at(2)?, size)),
        _ => None,
    }
}

/// The extent of each root UAV in `handles` (slot, handle) that every
/// access bounds; slots missing from the result are unbounded.
pub(crate) fn root_uav_extents(m: &Module, handles: &[(u8, V)], ids: &Ids) -> Vec<RootUavExtent> {
    let mut ranges = Ranges {
        m,
        ids,
        memo: HashMap::new(),
        open: Vec::new(),
    };
    // Per handle: the bound so far, or `None` once one access is unbounded.
    let mut found: HashMap<V, Option<(Affine, Affine)>> = HashMap::new();
    let mut calls = Vec::new();
    m.each_inst(|v, reads| {
        for &h in reads {
            if handles.iter().any(|x| x.1 == h) {
                calls.push((v, h));
            }
        }
    });
    for (v, h) in calls {
        let entry = found
            .entry(h)
            .or_insert(Some((Affine::default(), Affine::default())));
        let Some((end, peak)) = *entry else {
            continue;
        };
        let bound = match m.def(v) {
            Def::Call(name, args) => access(m, name, args)
                .filter(|&(handle, ..)| handle == h)
                .and_then(|(_, offset, bytes)| {
                    let r = ranges.range(offset, DEPTH)?;
                    let reach = r.hi.add(Affine::constant(bytes))?;
                    Some((end.max(reach)?, peak.max(r.peak)?))
                }),
            _ => None,
        };
        *entry = bound;
    }
    let mut out: Vec<RootUavExtent> = handles
        .iter()
        .filter_map(|&(slot, h)| {
            let (end, peak) = (*found.get(&h)?)?;
            Some(RootUavExtent { slot, end, peak })
        })
        .collect();
    out.sort_by_key(|e| e.slot);
    out
}

#[cfg(test)]
mod tests {
    use super::super::bitcode::Attr;
    use super::super::shader::Shader;
    use super::*;

    fn store(s: &mut Shader, h: V, offset: V, mask: u8) {
        let (iu, x) = (s.m.undef(s.t.i32), s.m.i32(7));
        let (mask, align) = (s.m.i8(mask), s.m.i32(4));
        let void = s.t.void;
        let e = s.entry;
        s.op(
            e,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[h, offset, iu, x, x, x, x, mask, align],
        );
    }

    fn id(s: &mut Shader, opcode: u32, c: u32) -> V {
        let (e, i32t, c) = (s.entry, s.t.i32, s.m.i32(c));
        let name = match opcode {
            93 => "threadId.i32",
            94 => "groupId.i32",
            _ => "threadIdInGroup.i32",
        };
        s.op(e, name, Attr::ReadNone, i32t, opcode, &[c])
    }

    fn extents(s: &Shader, group_ids: &[(V, u8)]) -> Vec<RootUavExtent> {
        extents_with(s, group_ids, None)
    }

    fn extents_with(s: &Shader, group_ids: &[(V, u8)], user_data: Option<V>) -> Vec<RootUavExtent> {
        let ids = Ids {
            threads: Some([64, 1, 1]),
            wave: 64,
            group_ids,
            user_data,
        };
        root_uav_extents(&s.m, &s.root_uavs, &ids)
    }

    #[test]
    fn constant_offsets_bound_the_view() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(2);
        let at = s.m.i32(0x40);
        store(&mut s, h, at, 0b11);
        let at = s.m.i32(0x10);
        store(&mut s, h, at, 0b1);
        let e = extents(&s, &[]);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].slot, 2);
        assert_eq!(e[0].len(None, |_| None), Some(0x48));
    }

    #[test]
    fn thread_ids_scale_with_the_dispatch() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(0);
        // 16 * threadId.x + 8, four dwords.
        let t = id(&mut s, 93, 0);
        let (k4, k8) = (s.m.i32(4), s.m.i32(8));
        let e0 = s.entry;
        let x = s.m.bin(e0, Bin::Shl, t, k4);
        let x = s.m.bin(e0, Bin::Add, x, k8);
        store(&mut s, h, x, 0xf);
        let e = extents(&s, &[]);
        // Threads 0..64*groups: the last writes 16 bytes at 16 * (64g - 1) + 8.
        assert_eq!(e[0].len(Some([1, 1, 1]), |_| None), Some(16 * 63 + 8 + 16));
        assert_eq!(
            e[0].len(Some([10, 3, 1]), |_| None),
            Some(16 * (640 - 1) + 8 + 16)
        );
        // A draw has no group counts.
        assert_eq!(e[0].len(None, |_| None), None);
    }

    #[test]
    fn group_ids_with_a_split_offset_and_local_ids() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(1);
        let g = id(&mut s, 94, 1);
        let user = s.m.undef(s.t.i32);
        let e0 = s.entry;
        let g = s.m.bin(e0, Bin::Add, g, user);
        let local = id(&mut s, 95, 0);
        let (k256, k4) = (s.m.i32(256), s.m.i32(4));
        let row = s.m.bin(e0, Bin::Mul, g, k256);
        let lane = s.m.bin(e0, Bin::Mul, local, k4);
        let x = s.m.bin(e0, Bin::Add, row, lane);
        store(&mut s, h, x, 1);
        let e = extents(&s, &[(g, 1)]);
        assert_eq!(
            e[0].len(Some([1, 5, 1]), |_| None),
            Some(256 * 4 + 63 * 4 + 4)
        );
    }

    #[test]
    fn masks_minima_and_remainders_bound_unknown_values() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(0);
        let unknown = s.m.undef(s.t.i32);
        let e0 = s.entry;
        let (m, k, r) = (s.m.i32(0xffc), s.m.i32(100), s.m.i32(7));
        let a = s.m.bin(e0, Bin::And, unknown, m);
        store(&mut s, h, a, 1);
        let lt = s.m.cmp(e0, pred::IULT, unknown, k);
        let b = s.m.select(e0, lt, unknown, k);
        store(&mut s, h, b, 1);
        let c = s.m.bin(e0, Bin::URem, unknown, r);
        store(&mut s, h, c, 1);
        assert_eq!(extents(&s, &[])[0].len(None, |_| None), Some(0xffc + 4));
    }

    #[test]
    fn unknown_offsets_or_other_uses_leave_the_view_unbounded() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(0);
        let at = s.m.i32(0);
        store(&mut s, h, at, 1);
        let unknown = s.m.undef(s.t.i32);
        store(&mut s, h, unknown, 1);
        let h3 = s.root_uav_handle(3);
        let at = s.m.i32(4);
        store(&mut s, h3, at, 1);
        let e = extents(&s, &[]);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].slot, 3);
    }

    #[test]
    fn user_data_dwords_are_read_at_bind_time() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(1);
        let user = s.user_data_handle();
        let e0 = s.entry;
        // base (dword 6) + 8 * threadId.x, two dwords.
        let row = s.m.i32(1);
        let cbuf_ret = s.t.cbuf_ret;
        let r = s.op(
            e0,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            cbuf_ret,
            59,
            &[user, row],
        );
        let base = s.m.extract(e0, r, 2);
        let t = id(&mut s, 93, 0);
        let k8 = s.m.i32(8);
        let x = s.m.bin(e0, Bin::Mul, t, k8);
        let x = s.m.bin(e0, Bin::Add, base, x);
        store(&mut s, h, x, 0b11);
        // Another constant buffer's dwords are not the user data's.
        let e = extents_with(&s, &[], None);
        assert!(e.is_empty());
        let e = extents_with(&s, &[], Some(user));
        let user = |dword| (dword == 6).then_some(0x1000);
        assert_eq!(e[0].len(Some([2, 1, 1]), user), Some(0x1000 + 8 * 127 + 8));
        assert_eq!(e[0].len(Some([2, 1, 1]), |_| None), None);
        // A base near 4 GiB would wrap.
        assert_eq!(e[0].len(Some([2, 1, 1]), |_| Some(u32::MAX - 16)), None);
    }

    #[test]
    fn products_with_a_bounded_side_scale_the_other() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(0);
        let user = s.user_data_handle();
        let e0 = s.entry;
        let row = s.m.i32(0);
        let cbuf_ret = s.t.cbuf_ret;
        let r = s.op(
            e0,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            cbuf_ret,
            59,
            &[user, row],
        );
        let stride = s.m.extract(e0, r, 1);
        // threadIdInGroup.x * stride (dword 1) << (lane & 3), one dword.
        let t = id(&mut s, 95, 0);
        let x = s.m.bin(e0, Bin::Mul, t, stride);
        let lane = id(&mut s, 95, 0);
        let three = s.m.i32(3);
        let sh = s.m.bin(e0, Bin::And, lane, three);
        let x = s.m.bin(e0, Bin::Shl, x, sh);
        store(&mut s, h, x, 1);
        let e = extents_with(&s, &[], Some(user));
        let user = |dword| (dword == 1).then_some(32);
        assert_eq!(e[0].len(None, user), Some(63 * 32 * 8 + 4));
    }

    #[test]
    fn wrapping_offsets_are_unbounded_at_run_time() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(0);
        let t = id(&mut s, 93, 0);
        let k = s.m.i32(1 << 20);
        let e0 = s.entry;
        let x = s.m.bin(e0, Bin::Mul, t, k);
        store(&mut s, h, x, 1);
        let e = extents(&s, &[]);
        // 64 threads: at most 63 MiB; 4096 groups would wrap 32 bits.
        assert!(e[0].len(Some([1, 1, 1]), |_| None).is_some());
        assert_eq!(e[0].len(Some([4096, 1, 1]), |_| None), None);
    }

    #[test]
    fn loops_are_unbounded() {
        let mut s = Shader::new(false);
        let h = s.root_uav_handle(0);
        let e0 = s.entry;
        let body = s.m.new_block();
        let i32t = s.t.i32;
        let phi = s.m.phi(body, i32t);
        let four = s.m.i32(4);
        let next = s.m.bin(body, Bin::Add, phi, four);
        let zero = s.m.i32(0);
        s.m.set_phi(phi, vec![(zero, e0), (next, body)]);
        store(&mut s, h, phi, 1);
        assert!(extents(&s, &[]).is_empty());
    }
}
