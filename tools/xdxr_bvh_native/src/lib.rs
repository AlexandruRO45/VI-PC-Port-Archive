//! Optional native traversal for `tools/xdxr_bvh.py`: `trace` (TLAS or BLAS closest hit) and `brute_force`.
//!
//! Every function here mirrors the Python reference step by step, with the same float64 operations in the same
//! order, Python's `min`/`max` semantics (the first argument wins a tie or a NaN comparison) and the same memory
//! reads (each read looks up the first region that holds all of it, as `Memory.read` does). Whatever the Python
//! path would raise on (a read outside every buffer, an instance root that is not the BLAS header's, a triangle
//! node in a top-level tree) is returned as a refusal instead, and the caller reruns the Python path so its own
//! exception and message come out. A leaf with internal geometry flag bits (QUAD_PRIM, FLIP_TRI1) is refused too:
//! how the traversal treats those bits is Python's to define.

const INVALID: u32 = 0xFFFF_FFFF;
const UNIT: i128 = 64;
const TRI1: u32 = 1;
const TRI3: u32 = 3;
const BOX16: u32 = 4;
const BOX32: u32 = 5;
const INSTANCE: u32 = 6;
const FLAG_BOTTOM_LEVEL: u32 = 0x8000_0000;
const INST_FLAGS_MASK: u32 = 0xC000_0000;
const LEAF_INTERNAL_FLAGS: u32 = 0xC0;
const REMAP_IDENTITY: u32 = 1 | (2 << 2);
const FLT_MAX: f64 = f32::MAX as f64;
const INF: f64 = f64::INFINITY;
/// Node visits per call before the native path gives up (a cyclic tree never ends in Python either; the cap only
/// keeps the native side from running out of memory first).
const MAX_STEPS: u64 = 1 << 26;

/// Why the native path stopped; any of them makes the caller rerun the Python path.
#[derive(Debug, PartialEq)]
pub enum Refusal {
    /// A read Python's `Memory.read` would raise IndexError on.
    Read,
    /// The instance root check of `trace` (ValueError in Python).
    Root,
    /// A triangle node reached in a top-level tree (TypeError in Python).
    TriangleInTlas,
    /// A leaf whose geometry flags carry QUAD_PRIM or FLIP_TRI1.
    LeafFlags,
    /// Too many steps.
    Steps,
    /// An allocation failed (the Python path then runs, or raises its own MemoryError).
    Memory,
}

type R<T> = Result<T, Refusal>;

/// A Vec with room for `n` items, or a refusal instead of an abort when the allocation fails.
fn vec_with<T>(n: usize) -> R<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Refusal::Memory)?;
    Ok(v)
}

/// `stack.push`, refusing instead of aborting when the stack cannot grow.
fn push(stack: &mut Vec<u32>, p: u32) -> R<()> {
    stack.try_reserve(1).map_err(|_| Refusal::Memory)?;
    stack.push(p);
    Ok(())
}

pub struct Memory<'a> {
    regions: Vec<(i128, &'a [u8])>,
}

impl<'a> Memory<'a> {
    pub fn new(regions: &[(i64, &'a [u8])]) -> R<Self> {
        let mut out = vec_with(regions.len())?;
        out.extend(regions.iter().map(|&(b, d)| (b as i128, d)));
        Ok(Memory { regions: out })
    }

    fn read(&self, va: i128, n: usize) -> R<&'a [u8]> {
        for &(base, data) in &self.regions {
            if base <= va && va + n as i128 <= base + data.len() as i128 {
                let s = (va - base) as usize;
                return Ok(&data[s..s + n]);
            }
        }
        Err(Refusal::Read)
    }

    fn u32s<const N: usize>(&self, va: i128) -> R<[u32; N]> {
        let b = self.read(va, 4 * N)?;
        let mut out = [0u32; N];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        Ok(out)
    }

    fn u32(&self, va: i128) -> R<u32> {
        Ok(self.u32s::<1>(va)?[0])
    }

    fn f32s<const N: usize>(&self, va: i128) -> R<[f64; N]> {
        let w = self.u32s::<N>(va)?;
        let mut out = [0f64; N];
        for i in 0..N {
            out[i] = f32::from_bits(w[i]) as f64;
        }
        Ok(out)
    }
}

/// Python's `min(a, b)`: `a` unless `b < a`.
#[inline]
fn pmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python's `max(a, b)`: `a` unless `b > a`.
#[inline]
fn pmax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

fn ptr_offset(p: u32) -> i128 {
    (p >> 3) as i128 * UNIT
}

/// `struct.unpack("<e")` of the half bits, as float64.
pub fn half_value(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f64;
    let mag = if exp == 0 {
        frac * (2f64).powi(-24)
    } else if exp == 31 {
        if frac == 0.0 {
            INF
        } else {
            f64::NAN
        }
    } else {
        (1024.0 + frac) * (2f64).powi(exp - 25)
    };
    sign * mag
}

/// `decode_box16`.
pub fn decode_box16(w: [u32; 3]) -> ([f64; 3], [f64; 3]) {
    let h = [w[0] & 0xFFFF, w[0] >> 16, w[1] & 0xFFFF, w[1] >> 16, w[2] & 0xFFFF, w[2] >> 16];
    let mut v = [0f64; 6];
    for i in 0..6 {
        v[i] = pmin(pmax(half_value(h[i] as u16), -FLT_MAX), FLT_MAX);
    }
    ([v[0], v[1], v[2]], [v[3], v[4], v[5]])
}

/// `_box_test`.
pub fn box_test(extent: f64, org: &[f64; 3], d: &[f64; 3], inv: &[f64; 3], lo: &[f64; 3], hi: &[f64; 3]) -> f64 {
    if (0..3).any(|c| lo[c] > hi[c]) {
        return INF;
    }
    let mut tnear = 0.0;
    let mut tfar = extent;
    for c in 0..3 {
        if d[c] == 0.0 {
            if org[c] < lo[c] || org[c] > hi[c] {
                return INF;
            }
        } else {
            let t1 = (lo[c] - org[c]) * inv[c];
            let t2 = (hi[c] - org[c]) * inv[c];
            tnear = pmax(tnear, pmin(t1, t2));
            tfar = pmin(tfar, pmax(t1, t2));
            if tfar < tnear {
                return INF;
            }
        }
    }
    tnear
}

fn tie_miss(ay: f64, by: f64, cy: f64, winding: bool) -> bool {
    if by == 0.0 && cy == 0.0 {
        return ay > 0.0;
    }
    let below = by < 0.0 || (by == 0.0 && cy > 0.0);
    below != winding
}

/// `triangle_test`: (t_num, t_den, i_num, j_num), or (inf, 1, 0, 0) on a miss.
pub fn triangle_test(v: &[[f64; 3]; 3], org: &[f64; 3], d: &[f64; 3], remap: u32) -> (f64, f64, f64, f64) {
    let mag = [d[0].abs(), d[1].abs(), d[2].abs()];
    let mut m = mag[0];
    if mag[1] > m {
        m = mag[1];
    }
    if mag[2] > m {
        m = mag[2];
    }
    let perm = if mag[0] == m {
        [1, 2, 0]
    } else if mag[1] == m {
        [2, 0, 1]
    } else {
        [0, 1, 2]
    };
    let mut rel = [[0f64; 3]; 3];
    for (k, p) in v.iter().enumerate() {
        let diff = [p[0] - org[0], p[1] - org[1], p[2] - org[2]];
        rel[k] = [diff[perm[0]], diff[perm[1]], diff[perm[2]]];
    }
    let dd = [d[perm[0]], d[perm[1]], d[perm[2]]];
    let s = |p: &[f64; 3]| [p[0] * dd[2] - dd[0] * p[2], p[1] * dd[2] - dd[1] * p[2], p[2]];
    let a = s(&rel[0]);
    let b = s(&rel[1]);
    let c = s(&rel[2]);
    let u = c[0] * b[1] - c[1] * b[0];
    let vv = a[0] * c[1] - a[1] * c[0];
    let w = b[0] * a[1] - b[1] * a[0];
    let t_num = u * a[2] + vv * b[2] + w * c[2];
    let t_den = (u + vv + w) * dd[2];
    let mut missed = (u < 0.0 || vv < 0.0 || w < 0.0) && (u > 0.0 || vv > 0.0 || w > 0.0);
    missed |= t_den == 0.0;
    missed |= t_num * (1.0f64).copysign(t_den) <= 0.0;
    missed |= t_num.is_nan();
    let winding = t_den > 0.0;
    if u == 0.0 {
        missed |= tie_miss(a[1], b[1], c[1], winding);
    }
    if vv == 0.0 {
        missed |= tie_miss(b[1], c[1], a[1], winding);
    }
    if w == 0.0 {
        missed |= tie_miss(c[1], a[1], b[1], winding);
    }
    if missed {
        return (INF, 1.0, 0.0, 0.0);
    }
    let e = [u, vv, w];
    let i_sel = remap & 3;
    let j_sel = (remap >> 2) & 3;
    let i_num = (if i_sel < 3 { e[i_sel as usize] } else { u }) * dd[2];
    let j_num = (if j_sel < 3 { e[j_sel as usize] } else { u }) * dd[2];
    (t_num, t_den, i_num, j_num)
}

fn inverse_dir(d: &[f64; 3]) -> [f64; 3] {
    let mut out = [0f64; 3];
    for c in 0..3 {
        out[c] = if d[c] != 0.0 { 1.0 / d[c] } else { FLT_MAX.copysign(d[c]) };
    }
    out
}

const TRI_VERTS: [[i128; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];

enum Step {
    Tri((f64, f64, f64, f64)),
    Box([u32; 4]),
    Other,
}

/// `intersect_node` with sort=True.
fn intersect_node(mem: &Memory, bvh_va: i128, p: u32, org: &[f64; 3], d: &[f64; 3], inv: &[f64; 3], extent: f64)
                  -> R<Step> {
    let va = bvh_va + ptr_offset(p);
    let kind = p & 7;
    if kind <= TRI3 {
        let slots = TRI_VERTS[kind as usize];
        let mut verts = [[0f64; 3]; 3];
        for (k, s) in slots.iter().enumerate() {
            verts[k] = mem.f32s::<3>(va + 12 * s)?;
        }
        let remap = (mem.u32(va + 60)? >> (8 * kind)) & 0xFF;
        return Ok(Step::Tri(triangle_test(&verts, org, d, remap)));
    }
    if kind != BOX16 && kind != BOX32 {
        return Ok(Step::Other);
    }
    let mut kids = mem.u32s::<4>(va)?;
    let mut dist = [0f64; 4];
    for i in 0..4 {
        let (lo, hi) = if kind == BOX16 {
            decode_box16(mem.u32s::<3>(va + 16 + 12 * i as i128)?)
        } else {
            let b = mem.f32s::<6>(va + 16 + 24 * i as i128)?;
            ([b[0], b[1], b[2]], [b[3], b[4], b[5]])
        };
        dist[i] = box_test(extent, org, d, inv, &lo, &hi);
    }
    for i in 0..4 {
        if !(dist[i] != INF && dist[i] <= extent) {
            kids[i] = INVALID;
        }
    }
    for (a, b) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
        if dist[a] > dist[b] {
            dist.swap(a, b);
            kids.swap(a, b);
        }
    }
    Ok(Step::Box(kids))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f64,
    pub instance: Option<u32>,
    pub geometry: u32,
    pub primitive: u32,
    pub leaf: u32,
    pub bary: (f64, f64),
    pub front: bool,
}

struct Tracer<'m, 'a> {
    mem: &'m Memory<'a>,
    steps: u64,
}

impl Tracer<'_, '_> {
    fn step(&mut self) -> R<()> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return Err(Refusal::Steps);
        }
        Ok(())
    }

    fn trace_blas(&mut self, bvh_va: i128, org: &[f64; 3], d: &[f64; 3], tmax: f64, mut hit: Option<Hit>,
                  inst: Option<u32>) -> R<Option<Hit>> {
        let mem = self.mem;
        let inv = inverse_dir(d);
        let mut stack = vec_with(64)?;
        stack.push(mem.u32(bvh_va + 48)?);
        while let Some(p) = stack.pop() {
            if p == INVALID {
                continue;
            }
            self.step()?;
            let extent = hit.map_or(tmax, |h| h.t);
            match intersect_node(mem, bvh_va, p, org, d, &inv, extent)? {
                Step::Other => {}
                Step::Tri((t_num, t_den, i_num, j_num)) => {
                    let leaf = bvh_va + ptr_offset(p);
                    if mem.u32(leaf + 48)? & LEAF_INTERNAL_FLAGS != 0 {
                        return Err(Refusal::LeafFlags);
                    }
                    if t_num == INF {
                        continue;
                    }
                    let t = t_num / t_den;
                    if t <= hit.map_or(tmax, |h| h.t) {
                        let [gif, base, prim2] = mem.u32s::<3>(leaf + 48)?;
                        let prim = if p & 7 == TRI1 { prim2 } else { base };
                        hit = Some(Hit { t, instance: inst, geometry: gif >> 8, primitive: prim, leaf: p,
                                         bary: (i_num / t_den, j_num / t_den), front: t_den > 0.0 });
                    }
                }
                Step::Box(kids) => {
                    for &k in kids.iter().rev() {
                        if k != INVALID {
                            push(&mut stack, k)?;
                        }
                    }
                }
            }
        }
        Ok(hit)
    }

    fn trace(&mut self, bvh_va: i128, org: &[f64; 3], d: &[f64; 3], tmax: f64, mask: u32, bias: i128)
             -> R<Option<Hit>> {
        let mem = self.mem;
        let flags = mem.u32(bvh_va)?;
        if flags & FLAG_BOTTOM_LEVEL != 0 {
            return self.trace_blas(bvh_va, org, d, tmax, None, None);
        }
        let mut hit: Option<Hit> = None;
        let inv = inverse_dir(d);
        let mut stack = vec_with(64)?;
        stack.push(mem.u32(bvh_va + 48)?);
        while let Some(p) = stack.pop() {
            if p == INVALID {
                continue;
            }
            if p & 7 == INSTANCE {
                let rec = bvh_va + ptr_offset(p);
                let [id_mask, _, internal, blas] = mem.u32s::<4>(rec + 48)?;
                if (id_mask >> 24) & mask == 0 || blas == 0 {
                    continue;
                }
                let w = mem.f32s::<12>(rec + 64)?;
                let mut lo = [0f64; 3];
                let mut ld = [0f64; 3];
                for r in 0..3 {
                    let row = &w[4 * r..4 * r + 4];
                    lo[r] = row[0] * org[0] + row[1] * org[1] + row[2] * org[2] + row[3];
                    ld[r] = row[0] * d[0] + row[1] * d[1] + row[2] * d[2];
                }
                let blas_va = ((blas as i128) << 8) + bias;
                let root = internal & !INST_FLAGS_MASK;
                if root != mem.u32(blas_va + 48)? {
                    return Err(Refusal::Root);
                }
                let index = mem.u32(rec + 112)?;
                hit = self.trace_blas(blas_va, &lo, &ld, tmax, hit, Some(index))?;
                continue;
            }
            self.step()?;
            let extent = hit.map_or(tmax, |h| h.t);
            match intersect_node(mem, bvh_va, p, org, d, &inv, extent)? {
                Step::Other => {}
                // Python pushes the four numbers of the result as pointers and fails on the next pop.
                Step::Tri(_) => return Err(Refusal::TriangleInTlas),
                Step::Box(kids) => {
                    for &k in kids.iter().rev() {
                        if k != INVALID {
                            push(&mut stack, k)?;
                        }
                    }
                }
            }
        }
        Ok(hit)
    }
}

/// `trace`: the closest hit and the number of steps counted the way `stats["steps"]` counts them.
pub fn trace(mem: &Memory, bvh_va: i64, org: &[f64; 3], d: &[f64; 3], tmax: f64, mask: u32, bias: i64)
             -> R<(Option<Hit>, u64)> {
    let mut tracer = Tracer { mem, steps: 0 };
    let hit = tracer.trace(bvh_va as i128, org, d, tmax, mask & 0xFF, bias as i128)?;
    Ok((hit, tracer.steps))
}

/// One instance of `brute_force`, already transformed by the caller: the object-space ray and its triangles'
/// coordinates as float32 values (9 per triangle), as `f32()` rounds them.
pub struct BruteInstance<'a> {
    pub org: [f64; 3],
    pub dir: [f64; 3],
    pub tris: &'a [f32],
}


/// `brute_force`: (instance position, primitive, t, bary) of the closest hit.
pub fn brute_force(instances: &[BruteInstance], tmax: f64) -> Option<(usize, usize, f64, (f64, f64))> {
    let mut best: Option<(usize, usize, f64, (f64, f64))> = None;
    for (k, inst) in instances.iter().enumerate() {
        for (prim, t) in inst.tris.chunks_exact(9).enumerate() {
            let v = [[t[0] as f64, t[1] as f64, t[2] as f64], [t[3] as f64, t[4] as f64, t[5] as f64],
                     [t[6] as f64, t[7] as f64, t[8] as f64]];
            let (t_num, t_den, i_num, j_num) = triangle_test(&v, &inst.org, &inst.dir, REMAP_IDENTITY);
            if t_num == INF {
                continue;
            }
            let tt = t_num / t_den;
            if tt <= best.map_or(tmax, |b| b.2) {
                best = Some((k, prim, tt, (i_num / t_den, j_num / t_den)));
            }
        }
    }
    best
}

// ---- C ABI ----

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// Runs an entry point's body so that a panic returns a refusal instead of unwinding into the caller.
fn guarded(body: impl FnOnce() -> i32) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_or(-2)
}

unsafe fn trace_entry(bases: *const i64, ptrs: *const *const u8, lens: *const usize, n_regions: usize, bvh_va: i64,
                      ray: *const f64, mask: u32, bias: i64, out_f: *mut f64, out_i: *mut u64) -> i32 {
    if ray.is_null() || out_f.is_null() || out_i.is_null() {
        return -1;
    }
    let bases = slice(bases, n_regions);
    let ptrs = slice(ptrs, n_regions);
    let lens = slice(lens, n_regions);
    if bases.len() != n_regions || ptrs.len() != n_regions || lens.len() != n_regions {
        return -1;
    }
    let mut regions = match vec_with(n_regions) {
        Ok(v) => v,
        Err(e) => return 1 + e as i32,
    };
    for i in 0..n_regions {
        if lens[i] != 0 && ptrs[i].is_null() {
            return -1;
        }
        regions.push((bases[i], slice(ptrs[i], lens[i])));
    }
    let mem = match Memory::new(&regions) {
        Ok(m) => m,
        Err(e) => return 1 + e as i32,
    };
    let r = slice(ray, 7);
    let org = [r[0], r[1], r[2]];
    let d = [r[3], r[4], r[5]];
    match trace(&mem, bvh_va, &org, &d, r[6], mask, bias) {
        Ok((hit, steps)) => {
            let of = std::slice::from_raw_parts_mut(out_f, 3);
            let oi = std::slice::from_raw_parts_mut(out_i, 8);
            oi.fill(0);
            of.fill(0.0);
            oi[7] = steps;
            if let Some(h) = hit {
                of[0] = h.t;
                of[1] = h.bary.0;
                of[2] = h.bary.1;
                oi[0] = 1;
                oi[1] = h.instance.is_some() as u64;
                oi[2] = h.instance.unwrap_or(0) as u64;
                oi[3] = h.geometry as u64;
                oi[4] = h.primitive as u64;
                oi[5] = h.leaf as u64;
                oi[6] = h.front as u64;
            }
            0
        }
        Err(e) => 1 + e as i32,
    }
}

/// Trace one ray. `ray` is org[3], dir[3], tmax. On success (0) `out_f` holds t, bary0, bary1 and `out_i` holds
/// found, has_instance, instance, geometry, primitive, leaf, front, steps. Any other return value is a refusal
/// (negative: bad arguments or a caught panic; positive: an input the Python path must handle).
///
/// # Safety
/// `bases`, `ptrs` and `lens` point to `n_regions` entries, each `ptrs[i]` to `lens[i]` readable bytes; `ray` to 7
/// doubles, `out_f` to 3 and `out_i` to 8 writable slots.
#[no_mangle]
pub unsafe extern "C" fn xdxr_trace_v1(bases: *const i64, ptrs: *const *const u8, lens: *const usize,
                                       n_regions: usize, bvh_va: i64, ray: *const f64, mask: u32, bias: i64,
                                       out_f: *mut f64, out_i: *mut u64) -> i32 {
    guarded(|| trace_entry(bases, ptrs, lens, n_regions, bvh_va, ray, mask, bias, out_f, out_i))
}

unsafe fn brute_force_entry(n_inst: usize, rays: *const f64, counts: *const usize, tris: *const *const f32,
                            tmax: f64, out_f: *mut f64, out_i: *mut u64) -> i32 {
    if out_f.is_null() || out_i.is_null() {
        return -1;
    }
    let n_rays = match n_inst.checked_mul(6) {
        Some(n) => n,
        None => return -1,
    };
    let rays = slice(rays, n_rays);
    let counts = slice(counts, n_inst);
    let ptrs = slice(tris, n_inst);
    if rays.len() != n_rays || counts.len() != n_inst || ptrs.len() != n_inst {
        return -1;
    }
    let mut instances = match vec_with(n_inst) {
        Ok(v) => v,
        Err(e) => return 1 + e as i32,
    };
    for k in 0..n_inst {
        let n = match counts[k].checked_mul(9) {
            Some(n) => n,
            None => return -1,
        };
        if n != 0 && ptrs[k].is_null() {
            return -1;
        }
        let r = &rays[6 * k..6 * k + 6];
        instances.push(BruteInstance { org: [r[0], r[1], r[2]], dir: [r[3], r[4], r[5]], tris: slice(ptrs[k], n) });
    }
    let of = std::slice::from_raw_parts_mut(out_f, 3);
    let oi = std::slice::from_raw_parts_mut(out_i, 3);
    of.fill(0.0);
    oi.fill(0);
    if let Some((k, prim, t, bary)) = brute_force(&instances, tmax) {
        of[0] = t;
        of[1] = bary.0;
        of[2] = bary.1;
        oi[0] = 1;
        oi[1] = k as u64;
        oi[2] = prim as u64;
    }
    0
}

/// Brute force over `n_inst` transformed instances. `rays` holds org[3], dir[3] per instance, `counts` the
/// triangle count of each and `tris` a pointer per instance to its float32 coordinates (9 per triangle). On success
/// (0) `out_f` holds t, bary0, bary1 and `out_i` found, instance position, primitive.
///
/// # Safety
/// `rays` holds 6 * `n_inst` doubles, `counts` and `tris` `n_inst` entries, each `tris[k]` 9 * `counts[k]` floats
/// (or is null for none); `out_f` and `out_i` hold 3 writable slots each.
#[no_mangle]
pub unsafe extern "C" fn xdxr_brute_force_v1(n_inst: usize, rays: *const f64, counts: *const usize,
                                             tris: *const *const f32, tmax: f64, out_f: *mut f64,
                                             out_i: *mut u64) -> i32 {
    guarded(|| brute_force_entry(n_inst, rays, counts, tris, tmax, out_f, out_i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_min_max_keep_the_first_argument_on_nan() {
        assert!(pmin(f64::NAN, 1.0).is_nan());
        assert_eq!(pmin(1.0, f64::NAN), 1.0);
        assert!(pmax(f64::NAN, 1.0).is_nan());
        assert_eq!(pmax(1.0, f64::NAN), 1.0);
        assert_eq!(pmin(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(pmax(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn half_values() {
        assert_eq!(half_value(0x3C00), 1.0);
        assert_eq!(half_value(0xC000), -2.0);
        assert_eq!(half_value(0x7BFF), 65504.0);
        assert_eq!(half_value(0x0001), 2f64.powi(-24));
        assert_eq!(half_value(0x8000).to_bits(), (-0.0f64).to_bits());
        assert_eq!(half_value(0x7C00), INF);
        assert!(half_value(0x7E00).is_nan());
        for bits in 0..=0xFFFFu32 {
            let v = half_value(bits as u16);
            if !v.is_nan() {
                assert_eq!(v as f32 as f64, v, "half {bits:#x} is a float32 value");
            }
        }
        let (lo, hi) = decode_box16([0x7C00_FC00, 0, 0]);
        assert_eq!(lo[0], -FLT_MAX);
        assert_eq!(lo[1], FLT_MAX);
        assert_eq!(hi, [0.0; 3]);
    }

    #[test]
    fn triangle_hit_and_miss() {
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (t_num, t_den, i, j) = triangle_test(&tri, &[0.25, 0.25, 1.0], &[0.0, 0.0, -1.0], REMAP_IDENTITY);
        assert_eq!(t_num / t_den, 1.0);
        assert_eq!((i / t_den, j / t_den), (0.25, 0.25));
        let miss = triangle_test(&tri, &[2.0, 2.0, 1.0], &[0.0, 0.0, -1.0], REMAP_IDENTITY);
        assert_eq!(miss, (INF, 1.0, 0.0, 0.0));
    }

    #[test]
    fn box_test_misses_an_empty_box_and_clips_to_the_extent() {
        let org = [0.0, 0.0, -5.0];
        let d = [0.0, 0.0, 1.0];
        let inv = inverse_dir(&d);
        assert_eq!(box_test(INF, &org, &d, &inv, &[1.0; 3], &[0.0; 3]), INF);
        assert_eq!(box_test(INF, &org, &d, &inv, &[-1.0; 3], &[1.0; 3]), 4.0);
        assert_eq!(box_test(3.0, &org, &d, &inv, &[-1.0; 3], &[1.0; 3]), INF);
    }

    #[test]
    fn reads_outside_every_region_refuse() {
        let data = [0u8; 64];
        let regions = [(0x1000, &data[..])];
        let mem = Memory::new(&regions).unwrap();
        assert_eq!(trace(&mem, 0x2000, &[0.0; 3], &[1.0, 0.0, 0.0], INF, 0xFF, 0), Err(Refusal::Read));
        assert!(mem.read(0x1000 + 60, 4).is_ok());
        assert_eq!(mem.read(0x1000 + 61, 4), Err(Refusal::Read));
    }

    #[test]
    fn a_panic_is_a_refusal() {
        assert_eq!(guarded(|| panic!("caught")), -2);
        assert_eq!(guarded(|| 0), 0);
    }

    #[test]
    fn null_arguments_refuse() {
        unsafe {
            assert_eq!(xdxr_trace_v1(std::ptr::null(), std::ptr::null(), std::ptr::null(), 0, 0, std::ptr::null(),
                                     0xFF, 0, std::ptr::null_mut(), std::ptr::null_mut()), -1);
            let mut of = [0f64; 3];
            let mut oi = [0u64; 3];
            let counts = [1usize];
            let ptrs = [std::ptr::null::<f32>()];
            let rays = [0f64; 6];
            assert_eq!(xdxr_brute_force_v1(1, rays.as_ptr(), counts.as_ptr(), ptrs.as_ptr(), INF, of.as_mut_ptr(),
                                           oi.as_mut_ptr()), -1);
            assert_eq!(xdxr_brute_force_v1(0, std::ptr::null(), std::ptr::null(), std::ptr::null(), INF,
                                           of.as_mut_ptr(), oi.as_mut_ptr()), 0);
            assert_eq!(oi[0], 0);
        }
    }

    #[test]
    fn brute_force_keeps_the_last_of_equal_hits() {
        let tri = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let both: Vec<f32> = tri.iter().chain(tri.iter()).copied().collect();
        let inst = [BruteInstance { org: [0.25, 0.25, 1.0], dir: [0.0, 0.0, -1.0], tris: &both }];
        let (k, prim, t, _) = brute_force(&inst, INF).unwrap();
        assert_eq!((k, prim, t), (0, 1, 1.0));
        assert!(brute_force(&inst, 0.5).is_none());
    }
}
