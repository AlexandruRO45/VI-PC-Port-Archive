//! Optional native kernels for the `--deep` walks of `tools/navmesh_decode.py`.
//!
//! Each entry point mirrors one pure-Python function of that tool and returns the same result through a fixed
//! array of `u64` words. A return value of `0` means the kernel ran and filled `out`; any other value is a refusal
//! (null pointer, an out-of-bounds range the Python caller has already ruled out, a counter it cannot represent),
//! and the caller then runs the Python reference instead. No entry point allocates memory the caller has to free.
//!
//! The kernels read the inflated resource only. They never panic on any input: every read is bounds-checked.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

/// Offsets are small integers chosen by the file, so a multiplicative hash is enough; SipHash is not needed.
#[derive(Default)]
struct OffsetHasher(u64);

impl Hasher for OffsetHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &c in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(c)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0 ^ v)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(29);
    }
}

type OffsetSet = HashSet<u64, BuildHasherDefault<OffsetHasher>>;

const BASE: u64 = 0x5000_0000;
const GRID_GROUP: u64 = 272;
const GRID_INNER: u32 = 0x0000_FF00;

/// Calls returned without filling `out`.
const REFUSED: i32 = 1;

fn u32_at(b: &[u8], o: u64) -> Option<u32> {
    let o = usize::try_from(o).ok()?;
    let s = b.get(o..o.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], o: u64) -> Option<u64> {
    let o = usize::try_from(o).ok()?;
    let s = b.get(o..o.checked_add(8)?)?;
    let mut w = [0u8; 8];
    w.copy_from_slice(s);
    Some(u64::from_le_bytes(w))
}

fn range(b: &[u8], lo: u64, hi: u64) -> Option<&[u8]> {
    if lo > hi {
        return None;
    }
    b.get(usize::try_from(lo).ok()?..usize::try_from(hi).ok()?)
}

/// `_zero(b, lo, hi)`: true when `0 <= lo <= hi <= len(b)` and every byte in between is zero.
fn zero(b: &[u8], lo: u64, hi: u64) -> bool {
    match range(b, lo, hi) {
        Some(s) => s.iter().all(|&c| c == 0),
        None => false,
    }
}

/// `offset_of(value)`: a zero-extended pointer in `[0x50000000, 0x60000000)` minus the base.
fn offset_of(value: u64) -> Option<u64> {
    if value >> 32 != 0 || !(BASE..BASE + 0x1000_0000).contains(&value) {
        return None;
    }
    Some(value - BASE)
}

/// Maximum of `n >= 1` little-endian u16 values at `o`; the caller has checked the range.
fn u16_max(b: &[u8], o: u64, n: u64) -> Option<u16> {
    let s = range(b, o, o.checked_add(n.checked_mul(2)?)?)?;
    s.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .max()
}

/// A slice from a raw pointer, or `None` for a null pointer with a non-zero length.
///
/// # Safety
/// `ptr` must point to `len` readable bytes that stay alive and unchanged for the call.
unsafe fn input<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(std::slice::from_raw_parts(ptr, len))
}

/// # Safety
/// `ptr` must point to `n` writable `u64` words.
unsafe fn output<'a>(ptr: *mut u64, n: usize) -> Option<&'a mut [u64]> {
    if ptr.is_null() {
        return None;
    }
    let out = std::slice::from_raw_parts_mut(ptr, n);
    out.fill(0);
    Some(out)
}

// ------------------------------------------------------------------------------------------------- quadtree

/// `(ok, nodes, leaves, max_depth)` of `_walk_quadtree`, visiting nodes in the same stack order.
fn quadtree(b: &[u8], root: u64, n_polys: u64, max_depth_limit: u64, max_nodes: u64) -> [u64; 4] {
    let len = b.len() as u64;
    let mut stack: Vec<(u64, u64)> = vec![(root, 0)];
    let mut seen = OffsetSet::default();
    let mut seen_leaves = OffsetSet::default();
    let (mut nodes, mut leaves, mut max_depth) = (0u64, 0u64, 0u64);
    while let Some((o, depth)) = stack.pop() {
        if seen.contains(&o) || depth > max_depth_limit || nodes >= max_nodes || o + 0x60 > len {
            return [0, nodes, leaves, max_depth];
        }
        seen.insert(o);
        nodes += 1;
        max_depth = max_depth.max(depth);
        if !zero(b, o + 0x54, o + 0x60) {
            return [0, nodes, leaves, max_depth];
        }
        let word = |k: u64| u64_at(b, o + 0x2C + 8 * k).unwrap_or(0);
        let leaf_raw = word(0);
        if leaf_raw != 0 {
            let lo = match offset_of(leaf_raw) {
                Some(lo) if lo + 0x20 <= len => lo,
                _ => return [0, nodes, leaves, max_depth],
            };
            if !seen_leaves.contains(&lo) {
                let field = |k: u64| u64_at(b, lo + 8 * k).unwrap_or(0);
                let (z0, pp, z1, np) = (field(0), field(1), field(2), field(3));
                if z0 != 0 || z1 != 0 || np > n_polys {
                    return [0, nodes, leaves, max_depth];
                }
                if np != 0 {
                    let fits = match offset_of(pp) {
                        Some(po) if po + 2 * np <= len => {
                            matches!(u16_max(b, po, np), Some(m) if u64::from(m) < n_polys)
                        }
                        _ => false,
                    };
                    if !fits {
                        return [0, nodes, leaves, max_depth];
                    }
                }
                seen_leaves.insert(lo);
            }
            leaves += 1;
        }
        for k in 1..5 {
            let cp = word(k);
            if cp != 0 {
                match offset_of(cp) {
                    Some(co) => stack.push((co, depth + 1)),
                    None => return [0, nodes, leaves, max_depth],
                }
            }
        }
    }
    [1, nodes, leaves, max_depth]
}

/// `_walk_quadtree(b, root, n_polys)` for a root the caller has already decoded. `out` takes four words:
/// ok, nodes, leaves, max_depth.
///
/// # Safety
/// `data` must point to `len` readable bytes and `out` to four writable `u64` words.
#[no_mangle]
pub unsafe extern "C" fn navmesh_quadtree_v1(
    data: *const u8,
    len: usize,
    root: u64,
    n_polys: u64,
    max_depth: u64,
    max_nodes: u64,
    out: *mut u64,
) -> i32 {
    let (Some(b), Some(out)) = (input(data, len), output(out, 4)) else {
        return REFUSED;
    };
    if root >= 0x1000_0000
        || n_polys > u64::from(u32::MAX)
        || max_depth > 1 << 20
        || max_nodes > 1 << 32
    {
        return REFUSED;
    }
    out.copy_from_slice(&quadtree(b, root, n_polys, max_depth, max_nodes));
    0
}

// --------------------------------------------------------------------------------------------- polygon chain

/// `_poly_chain(b, blocks, owner, pool)`: `blocks` holds `nblocks` (offset, count) pairs. `out` takes two words:
/// ok, has_water. `water_mask` is the bit of the u32 at polygon +0x18 that marks water.
///
/// # Safety
/// `data` must point to `len` readable bytes, `blocks` to `2 * nblocks` readable `u64` words and `out` to two
/// writable `u64` words.
#[no_mangle]
pub unsafe extern "C" fn navmesh_poly_chain_v1(
    data: *const u8,
    len: usize,
    blocks: *const u64,
    nblocks: usize,
    owner: u64,
    pool: u64,
    water_mask: u32,
    out: *mut u64,
) -> i32 {
    let (Some(b), Some(out)) = (input(data, len), output(out, 2)) else {
        return REFUSED;
    };
    if nblocks != 0 && blocks.is_null() {
        return REFUSED;
    }
    let pairs: &[u64] = if nblocks == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(blocks, 2 * nblocks)
    };
    // Every block must lie inside the buffer, as split_array guarantees; anything else is the reference's job.
    let mut spans = Vec::with_capacity(nblocks);
    for pair in pairs.chunks_exact(2) {
        let (o, n) = (pair[0], pair[1]);
        match n
            .checked_mul(44)
            .and_then(|size| o.checked_add(size))
            .and_then(|end| range(b, o, end))
        {
            Some(s) => spans.push(s),
            None => return REFUSED,
        }
    }
    let mut next: u64 = 0;
    let mut has_water = false;
    for s in spans {
        for poly in s.chunks_exact(44) {
            let mesh = u32::from_le_bytes([poly[0], poly[1], poly[2], poly[3]]);
            let packed = u32::from_le_bytes([poly[4], poly[5], poly[6], poly[7]]);
            let pf18 = u32::from_le_bytes([poly[24], poly[25], poly[26], poly[27]]);
            if u64::from(mesh) != owner || u64::from(packed >> 4) != next {
                out[1] = u64::from(has_water);
                return 0;
            }
            if pf18 & water_mask != 0 {
                has_water = true;
            }
            next += u64::from(packed & 0xF);
        }
    }
    out[0] = u64::from(next == pool);
    out[1] = u64::from(has_water);
    0
}

// --------------------------------------------------------------------------------------------------- volumes

/// Ordered counter: values in first-seen order, as a Python `Counter` built by `+= 1` keeps them.
struct Ordered {
    items: Vec<(u64, u64)>,
}

impl Ordered {
    fn new() -> Self {
        Ordered { items: Vec::new() }
    }

    fn add(&mut self, value: u64, count: u64) {
        match self.items.iter_mut().find(|(v, _)| *v == value) {
            Some((_, c)) => *c += count,
            None => self.items.push((value, count)),
        }
    }

    /// Write `len, (value, count)*` into `out`; `None` when it does not fit.
    fn write(&self, out: &mut [u64]) -> Option<()> {
        let need = 1 + 2 * self.items.len();
        if need > out.len() {
            return None;
        }
        out[0] = self.items.len() as u64;
        for (i, (v, c)) in self.items.iter().enumerate() {
            out[1 + 2 * i] = *v;
            out[2 + 2 * i] = *c;
        }
        Some(())
    }
}

const VOLUME_WORDS: usize = 16;

fn volumes(b: &[u8], vol: u64, dc: u64, out: &mut [u64]) -> Option<()> {
    let mut flags = Ordered::new();
    let mut vertex_counts = Ordered::new();
    for p in 0..dc / 2 {
        let o0 = vol + 2 * p * 96;
        let o1 = o0 + 96;
        let head0 = range(b, o0, o0 + 0x60)?;
        let head1 = range(b, o1, o1 + 0x60)?;
        if head0[..20] != head1[..20] {
            return Some(());
        }
        for k in 0..3 {
            let at = 8 + 4 * k;
            let v = f32::from_le_bytes([head0[at], head0[at + 1], head0[at + 2], head0[at + 3]]);
            if !v.is_finite() {
                return Some(());
            }
        }
        let (f0, vc0, f1, vc1) = (head0[0x14], head0[0x15], head1[0x14], head1[0x15]);
        if !matches!(f0, 0 | 4) || f1 != 8 || !matches!(vc0, 4 | 8) || vc1 != 4 {
            return Some(());
        }
        let tail = |o: u64, vc: u8| {
            zero(b, o + 0x16 + 4 * u64::from(vc), o + 0x56) && zero(b, o + 0x5A, o + 0x60)
        };
        if !(tail(o0, vc0) && tail(o1, vc1)) {
            return Some(());
        }
        flags.add(u64::from(f0), 1);
        flags.add(u64::from(f1), 1);
        vertex_counts.add(u64::from(vc0), 1);
        vertex_counts.add(u64::from(vc1), 1);
    }
    out[0] = 1;
    out[1] = dc / 2;
    flags.write(&mut out[2..9])?;
    vertex_counts.write(&mut out[9..16])?;
    Some(())
}

/// The loop of `_walk_volumes(b, vol, dc, stats)` once the caller has checked `vol`, `dc` and the bounds. `out`
/// takes 16 words: ok, pairs, then the flag and vertex-count counters as `len, (value, count)*` (3 slots each).
/// On ok == 0 the counters are empty and the caller leaves its stats alone.
///
/// # Safety
/// `data` must point to `len` readable bytes and `out` to 16 writable `u64` words.
#[no_mangle]
pub unsafe extern "C" fn navmesh_volumes_v1(
    data: *const u8,
    len: usize,
    vol: u64,
    dc: u64,
    out: *mut u64,
) -> i32 {
    let (Some(b), Some(out)) = (input(data, len), output(out, VOLUME_WORDS)) else {
        return REFUSED;
    };
    if dc == 0 || dc % 2 != 0 || vol >= 0x1000_0000 || dc > u64::from(u32::MAX) {
        return REFUSED;
    }
    if vol + 96 * dc > b.len() as u64 {
        return REFUSED;
    }
    match volumes(b, vol, dc, out) {
        Some(()) => 0,
        None => REFUSED,
    }
}

// ------------------------------------------------------------------------------------------------------ grid

const GRID_WORDS: usize = 64;
const DEPTH_SLOTS: usize = 22; // a u64 key has at most 64 bits: depth (64 - 1) / 3 == 21
const KIND_FF00: u64 = 0;
const KIND_LOW_01: u64 = 1;
const KIND_LOW_03: u64 = 2;
const KIND_OTHER: u64 = 3;

/// Every occupied slot as (key, value), in table order; `None` when the table runs past the buffer.
fn grid_slots(b: &[u8], groups: u64) -> Option<Vec<(u64, u64)>> {
    let table = range(b, 0x60, 0x60 + GRID_GROUP * groups)?;
    let mut slots = Vec::new();
    for group in table.chunks_exact(GRID_GROUP as usize) {
        for (k, &c) in group[..16].iter().enumerate() {
            if c < 0x80 {
                let at = 16 + 16 * k;
                let mut kw = [0u8; 8];
                let mut vw = [0u8; 8];
                kw.copy_from_slice(&group[at..at + 8]);
                vw.copy_from_slice(&group[at + 8..at + 16]);
                slots.push((u64::from_le_bytes(kw), u64::from_le_bytes(vw)));
            }
        }
    }
    Some(slots)
}

/// `_grid_values`: every occupied slot's value is a distinct index below `entries`, and every index is used.
fn grid_values(slots: &[(u64, u64)], entries: u64) -> bool {
    let mut seen = vec![false; entries as usize];
    for &(_, value) in slots {
        if value >= entries || seen[value as usize] {
            return false;
        }
        seen[value as usize] = true;
    }
    seen.iter().all(|&s| s)
}

fn depth_of(key: u64) -> u64 {
    (63 - u64::from(key.leading_zeros())) / 3
}

/// `_grid_octree` over keys that are known to be distinct, given sorted.
fn grid_octree(sorted: &[u64]) -> bool {
    if sorted.binary_search(&1).is_err() {
        return false;
    }
    sorted.iter().all(|&key| {
        key != 0
            && (63 - key.leading_zeros()) % 3 == 0
            && (key == 1 || sorted.binary_search(&(key >> 3)).is_ok())
    })
}

/// The `--deep` part of `decode_grid` once `occupied_equals_entries` holds. `out` takes 64 words:
///
/// * 0: values_bijective (0/1)
/// * 1: keys_octree (0 not computed, 1 false, 2 true)
/// * 2: inner_payload_ff00 (0 not computed, 1 false, 2 true)
/// * 3: inner node count (only when word 2 is 2)
/// * 4..49: key depths as `len, (depth, count)*`, first-seen order
/// * 49..58: leaf payload kinds as `len, (kind, count)*` (0 ff00, 1 low_byte_01, 2 low_byte_03, 3 other)
///
/// # Safety
/// `data` must point to `len` readable bytes and `out` to 64 writable `u64` words.
#[no_mangle]
pub unsafe extern "C" fn navmesh_grid_deep_v1(
    data: *const u8,
    len: usize,
    groups: u64,
    entries: u64,
    payload_ok: i32,
    payload: u64,
    out: *mut u64,
) -> i32 {
    let (Some(b), Some(out)) = (input(data, len), output(out, GRID_WORDS)) else {
        return REFUSED;
    };
    if groups > u64::from(u32::MAX) || entries > u64::from(u32::MAX) {
        return REFUSED;
    }
    let Some(slots) = grid_slots(b, groups) else {
        return REFUSED;
    };
    // The caller only asks once the occupied count equals `entries`.
    if slots.len() as u64 != entries {
        return REFUSED;
    }
    if !grid_values(&slots, entries) {
        return 0;
    }
    out[0] = 1;
    if payload_ok == 0 {
        return 0;
    }
    let mut sorted: Vec<u64> = slots.iter().map(|&(key, _)| key).collect();
    sorted.sort_unstable();
    let distinct = sorted.windows(2).all(|w| w[0] != w[1]);
    if !(distinct && grid_octree(&sorted)) {
        out[1] = 1;
        return 0;
    }
    out[1] = 2;

    let mut parents: Vec<u64> = sorted
        .iter()
        .filter(|&&k| k != 1)
        .map(|&k| k >> 3)
        .collect();
    parents.sort_unstable();
    parents.dedup();
    let mut words = Vec::with_capacity(slots.len());
    for &(_, value) in &slots {
        match payload.checked_add(4 * value).and_then(|at| u32_at(b, at)) {
            Some(w) => words.push(w),
            None => return REFUSED,
        }
    }
    let inner = |key: u64| parents.binary_search(&key).is_ok();
    if slots
        .iter()
        .zip(&words)
        .any(|(&(key, _), &w)| inner(key) && w != GRID_INNER)
    {
        out[2] = 1;
        return 0;
    }
    let mut depths = Ordered::new();
    let mut kinds = Ordered::new();
    for (&(key, _), &w) in slots.iter().zip(&words) {
        depths.add(depth_of(key), 1);
        if inner(key) {
            continue;
        }
        let kind = if w == GRID_INNER {
            KIND_FF00
        } else {
            match w & 0xFF {
                0x01 => KIND_LOW_01,
                0x03 => KIND_LOW_03,
                _ => KIND_OTHER,
            }
        };
        kinds.add(kind, 1);
    }
    out[2] = 2;
    out[3] = parents.len() as u64;
    if depths.write(&mut out[4..4 + 1 + 2 * DEPTH_SLOTS]).is_none()
        || kinds.write(&mut out[49..58]).is_none()
    {
        out.fill(0);
        return REFUSED;
    }
    0
}

/// ABI version of this library; the Python adapter refuses any other value.
#[no_mangle]
pub extern "C" fn navmesh_native_abi() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u64(b: &mut [u8], o: usize, v: u64) {
        b[o..o + 8].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn offsets_and_zero_ranges() {
        assert_eq!(offset_of(BASE), Some(0));
        assert_eq!(offset_of(BASE - 1), None);
        assert_eq!(offset_of(BASE + 0x1000_0000), None);
        assert_eq!(offset_of(1 << 32 | BASE), None);
        let b = [0u8, 0, 1];
        assert!(zero(&b, 0, 2));
        assert!(!zero(&b, 0, 3));
        assert!(!zero(&b, 2, 1));
        assert!(!zero(&b, 0, 4));
        assert_eq!(u32_at(&b, 0), None);
        assert_eq!(u64_at(&b, 0), None);
        assert_eq!(u16_max(&b, 0, 1), Some(0));
        assert_eq!(u16_max(&b, 1, 1), Some(256));
        assert_eq!(u16_max(&b, 2, 1), None);
    }

    #[test]
    fn quadtree_root_leaf_and_cycle() {
        let mut b = vec![0u8; 0x200];
        // Root at 0 with one child at 0x60; the child points to a leaf at 0xC0 listing polygon 0 at 0xE0.
        put_u64(&mut b, 0x34, BASE + 0x60);
        put_u64(&mut b, 0x60 + 0x2C, BASE + 0xC0);
        put_u64(&mut b, 0xC0 + 8, BASE + 0xE0);
        put_u64(&mut b, 0xC0 + 24, 1);
        assert_eq!(quadtree(&b, 0, 1, 32, 1 << 20), [1, 2, 1, 1]);
        assert_eq!(quadtree(&b, 0, 0, 32, 1 << 20), [0, 2, 0, 1]);
        put_u64(&mut b, 0x60 + 0x34, BASE);
        assert_eq!(quadtree(&b, 0, 1, 32, 1 << 20), [0, 2, 1, 1]);
    }

    #[test]
    fn grid_octree_rules() {
        assert!(grid_octree(&[1, 8, 9, 64]));
        assert!(!grid_octree(&[8, 9]));
        assert!(!grid_octree(&[1, 8, 16]));
        assert!(!grid_octree(&[0, 1, 8]));
        assert!(!grid_octree(&[1, 8, 72]));
        assert_eq!(depth_of(1), 0);
        assert_eq!(depth_of(8), 1);
        assert_eq!(depth_of(1 << 63), 21);
    }

    #[test]
    fn ordered_counter_keeps_first_seen_order() {
        let mut c = Ordered::new();
        c.add(8, 1);
        c.add(4, 1);
        c.add(8, 1);
        let mut out = [0u64; 7];
        c.write(&mut out).unwrap();
        assert_eq!(out, [2, 8, 2, 4, 1, 0, 0]);
        let mut small = [0u64; 2];
        assert!(c.write(&mut small).is_none());
    }
}
