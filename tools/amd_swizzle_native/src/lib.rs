//! Optional native kernels for `tools/amd_swizzle.py`: the element offset grid of `offsets` and the element
//! gather of `detile`, with the same integer steps as the numpy code.
//!
//! An address bit `i` of the in-block offset is `parity(x & xmask[i]) ^ parity(y & ymask[i])`, so the offset is
//! the XOR of a function of x alone and a function of y alone; the block term is added on top. Each byte offset
//! is `(row[y] ^ col[x]) + (y / bh) * blocks_x * block_bytes + (x / bw) * block_bytes`, and `detile` reads
//! element `offset / bpe` (numpy's `_grid(..., shift)` and `_grid(..., 0) // bpe` both reduce to that).
//!
//! Return codes: 0 done, 1 an element lies outside the source (`detile` returns None), negative = refused (bad
//! parameters or an offset that would not fit in an int64); the Python code then runs instead.
//! No dependencies, standard library only.

use std::thread;

pub const ABI: u32 = 1;
const REFUSE_PARAMS: i32 = -1;
const REFUSE_OVERFLOW: i32 = -2;
/// Grids with fewer elements than this run on one thread.
const PARALLEL_MIN: usize = 1 << 15;
/// Elements per extra thread: below this, spawning a thread costs more than it saves.
const PER_THREAD: usize = 1 << 17;

#[no_mangle]
pub extern "C" fn amd_swizzle_native_abi() -> u32 {
    ABI
}

/// In-block address contribution of one axis, for coordinates `0..n` (numpy `_axis_offsets` with shift 0).
pub fn axis(masks: &[u64], n: usize) -> Vec<u64> {
    let mut out = vec![0u64; n];
    for (c, slot) in out.iter_mut().enumerate() {
        let c = c as u64;
        let mut acc = 0u64;
        for (bit, &m) in masks.iter().enumerate() {
            if m != 0 {
                acc |= (((c & m).count_ones() & 1) as u64) << bit;
            }
        }
        *slot = acc;
    }
    out
}

/// The per-axis tables of one grid: XOR parts and additive block parts.
pub struct Grid {
    pub wb: usize,
    pub hb: usize,
    pub row: Vec<u64>,
    pub col: Vec<u64>,
    pub row_block: Vec<u64>,
    pub col_block: Vec<u64>,
}

impl Grid {
    /// Builds the tables, or a refusal code when the parameters are invalid or an offset would not fit in an
    /// int64 (numpy would wrap there; the Python code keeps that behaviour).
    pub fn new(
        xm: &[u64],
        ym: &[u64],
        wb: usize,
        hb: usize,
        bw: u64,
        bh: u64,
        block_bytes: u64,
    ) -> Result<Grid, i32> {
        if xm.len() != ym.len() || xm.len() > 63 || bw == 0 || bh == 0 {
            return Err(REFUSE_PARAMS);
        }
        if xm.iter().chain(ym.iter()).any(|&m| m >= 1 << 63) {
            return Err(REFUSE_PARAMS);
        }
        let blocks_x = (wb as u64).div_ceil(bw);
        let row_unit = blocks_x.checked_mul(block_bytes).ok_or(REFUSE_OVERFLOW)?;
        if wb > 0 && hb > 0 {
            // Bound every offset before allocating: the XOR part stays below 2^(highest used address bit + 1).
            let used = xm.iter().zip(ym).rposition(|(&a, &b)| a != 0 || b != 0);
            let xor_max = used.map_or(0, |bit| u64::MAX >> (63 - bit));
            let total = ((hb as u64 - 1) / bh)
                .checked_mul(row_unit)
                .and_then(|v| v.checked_add(((wb as u64 - 1) / bw).checked_mul(block_bytes)?))
                .and_then(|v| v.checked_add(xor_max))
                .ok_or(REFUSE_OVERFLOW)?;
            if total > i64::MAX as u64 {
                return Err(REFUSE_OVERFLOW);
            }
        }
        let col = axis(xm, wb);
        let row = axis(ym, hb);
        let row_block = (0..hb as u64).map(|y| (y / bh) * row_unit).collect();
        let col_block = (0..wb as u64).map(|x| (x / bw) * block_bytes).collect();
        Ok(Grid { wb, hb, row, col, row_block, col_block })
    }

    #[inline(always)]
    fn offset(&self, y: usize, x: usize) -> u64 {
        (self.row[y] ^ self.col[x]) + self.row_block[y] + self.col_block[x]
    }
}

/// Splits `rows` rows of `row_len` items of `out` over threads by grid elements (`elems` in all);
/// `work(first_row, chunk)` returns false to stop.
fn for_row_chunks<T: Send>(
    out: &mut [T],
    rows: usize,
    row_len: usize,
    elems: usize,
    work: &(dyn Fn(usize, &mut [T]) -> bool + Sync),
) -> bool {
    let threads = if elems < PARALLEL_MIN {
        1
    } else {
        let cpus = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        cpus.clamp(1, 16).min(rows).min(elems / PER_THREAD + 1)
    };
    if threads <= 1 {
        return work(0, out);
    }
    let per = rows.div_ceil(threads);
    thread::scope(|s| {
        let mut handles = Vec::new();
        let mut rest = out;
        let mut first = 0;
        while first < rows {
            let n = per.min(rows - first);
            let (chunk, tail) = rest.split_at_mut(n * row_len);
            rest = tail;
            let start = first;
            handles.push(s.spawn(move || work(start, chunk)));
            first += n;
        }
        handles.into_iter().fold(true, |ok, h| h.join().unwrap_or(false) && ok)
    })
}

/// Fills `out` (hb * wb, row-major) with every byte offset (numpy `offsets`).
pub fn fill_offsets(g: &Grid, out: &mut [i64]) {
    let wb = g.wb;
    if wb == 0 || g.hb == 0 {
        return;
    }
    for_row_chunks(out, g.hb, wb, g.hb * wb, &|y0, chunk| {
        for (r, line) in chunk.chunks_exact_mut(wb).enumerate() {
            let y = y0 + r;
            for (x, slot) in line.iter_mut().enumerate() {
                *slot = g.offset(y, x) as i64;
            }
        }
        true
    });
}

#[inline(always)]
fn gather_rows<const E: usize>(
    g: &Grid,
    src: &[u8],
    n_elems: u64,
    div: u64,
    shift: Option<u32>,
    y0: usize,
    chunk: &mut [u8],
) -> bool {
    let wb = g.wb;
    for (r, line) in chunk.chunks_exact_mut(wb * E).enumerate() {
        let y = y0 + r;
        let (ry, rby) = (g.row[y], g.row_block[y]);
        for (x, dst) in line.chunks_exact_mut(E).enumerate() {
            let off = (ry ^ g.col[x]) + rby + g.col_block[x];
            let idx = match shift {
                Some(s) => off >> s,
                None => off / div,
            };
            if idx >= n_elems {
                return false;
            }
            let at = idx as usize * E;
            dst.copy_from_slice(&src[at..at + E]);
        }
    }
    true
}

fn gather_dyn(g: &Grid, src: &[u8], n_elems: u64, bpe: usize, y0: usize, chunk: &mut [u8]) -> bool {
    let wb = g.wb;
    for (r, line) in chunk.chunks_exact_mut(wb * bpe).enumerate() {
        let y = y0 + r;
        for (x, dst) in line.chunks_exact_mut(bpe).enumerate() {
            let idx = g.offset(y, x) / bpe as u64;
            if idx >= n_elems {
                return false;
            }
            let at = idx as usize * bpe;
            dst.copy_from_slice(&src[at..at + bpe]);
        }
    }
    true
}

/// Gathers element `offset / bpe` of `src` for every grid cell into `out` (hb * wb * bpe bytes). Returns false
/// when an element lies outside `src` (its first `n_elems * bpe` bytes); `out` is then unspecified.
pub fn detile_into(g: &Grid, src: &[u8], bpe: usize, out: &mut [u8]) -> bool {
    if g.wb == 0 || g.hb == 0 {
        return true;
    }
    let n_elems = (src.len() / bpe) as u64;
    let shift = if bpe.is_power_of_two() { Some(bpe.trailing_zeros()) } else { None };
    let div = bpe as u64;
    let row_len = g.wb * bpe;
    // Rows are handed out as runs of `row_len` bytes, so the per-element copies below never cross a row.
    let work = |y0: usize, chunk: &mut [u8]| -> bool {
        match bpe {
            1 => gather_rows::<1>(g, src, n_elems, div, shift, y0, chunk),
            2 => gather_rows::<2>(g, src, n_elems, div, shift, y0, chunk),
            4 => gather_rows::<4>(g, src, n_elems, div, shift, y0, chunk),
            8 => gather_rows::<8>(g, src, n_elems, div, shift, y0, chunk),
            16 => gather_rows::<16>(g, src, n_elems, div, shift, y0, chunk),
            _ => gather_dyn(g, src, n_elems, bpe, y0, chunk),
        }
    };
    for_row_chunks(out, g.hb, row_len, g.hb * g.wb, &work)
}

/// # Safety
/// `xm` and `ym` point to `nbits` u64 each; `out` to `hb * wb` writable i64.
#[no_mangle]
pub unsafe extern "C" fn amd_swizzle_offsets(
    xm: *const u64,
    ym: *const u64,
    nbits: usize,
    wb: usize,
    hb: usize,
    bw: u64,
    bh: u64,
    block_bytes: u64,
    out: *mut i64,
) -> i32 {
    if nbits > 63 || (nbits > 0 && (xm.is_null() || ym.is_null())) {
        return REFUSE_PARAMS;
    }
    let Some(cells) = wb.checked_mul(hb) else { return REFUSE_PARAMS };
    let (xm, ym) = if nbits == 0 {
        (&[][..], &[][..])
    } else {
        (std::slice::from_raw_parts(xm, nbits), std::slice::from_raw_parts(ym, nbits))
    };
    let g = match Grid::new(xm, ym, wb, hb, bw, bh, block_bytes) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if cells == 0 {
        return 0;
    }
    if out.is_null() {
        return REFUSE_PARAMS;
    }
    fill_offsets(&g, std::slice::from_raw_parts_mut(out, cells));
    0
}

/// # Safety
/// `src` points to `src_len` readable bytes; `xm`/`ym` to `nbits` u64 each; `out` to `hb * wb * bpe` writable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn amd_swizzle_detile(
    src: *const u8,
    src_len: usize,
    bpe: usize,
    xm: *const u64,
    ym: *const u64,
    nbits: usize,
    wb: usize,
    hb: usize,
    bw: u64,
    bh: u64,
    block_bytes: u64,
    out: *mut u8,
) -> i32 {
    if bpe == 0 || nbits > 63 || (nbits > 0 && (xm.is_null() || ym.is_null())) {
        return REFUSE_PARAMS;
    }
    let Some(bytes) = wb.checked_mul(hb).and_then(|c| c.checked_mul(bpe)) else { return REFUSE_PARAMS };
    let (xm, ym) = if nbits == 0 {
        (&[][..], &[][..])
    } else {
        (std::slice::from_raw_parts(xm, nbits), std::slice::from_raw_parts(ym, nbits))
    };
    let g = match Grid::new(xm, ym, wb, hb, bw, bh, block_bytes) {
        Ok(g) => g,
        Err(code) => return code,
    };
    if bytes == 0 {
        return 0;
    }
    if out.is_null() || (src_len > 0 && src.is_null()) {
        return REFUSE_PARAMS;
    }
    let src = if src_len == 0 { &[][..] } else { std::slice::from_raw_parts(src, src_len) };
    let out = std::slice::from_raw_parts_mut(out, bytes);
    if detile_into(&g, src, bpe, out) {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-bit reference over the whole grid, as tests/test_amd_swizzle_detile.py's `reference_offsets`.
    fn reference(eq: &[(u64, u64)], wb: usize, hb: usize, bw: u64, bh: u64, block: u64) -> Vec<u64> {
        let blocks_x = (wb as u64).div_ceil(bw);
        let mut out = Vec::new();
        for y in 0..hb as u64 {
            for x in 0..wb as u64 {
                let mut off = 0u64;
                for (bit, &(mx, my)) in eq.iter().enumerate() {
                    let p = ((x & mx).count_ones() ^ (y & my).count_ones()) & 1;
                    off |= (p as u64) << bit;
                }
                out.push(off + ((y / bh) * blocks_x + x / bw) * block);
            }
        }
        out
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn block_shape(bpe: u64, block: u64) -> (u64, u64) {
        let elems = block / bpe;
        let lb = 63 - elems.leading_zeros() as u64;
        let bw = 1u64 << lb.div_ceil(2);
        (bw, elems / bw)
    }

    fn split(eq: &[(u64, u64)]) -> (Vec<u64>, Vec<u64>) {
        (eq.iter().map(|e| e.0).collect(), eq.iter().map(|e| e.1).collect())
    }

    #[test]
    fn offsets_and_detile_match_the_per_bit_reference() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..300 {
            let bits = [8u32, 12, 16][case % 3];
            let bpe = [1u64, 2, 3, 4, 8, 12, 16][(rng.next() % 7) as usize];
            let block = 1u64 << bits;
            let eq: Vec<(u64, u64)> = (0..bits).map(|_| (rng.next() & 0xFFFF, rng.next() & 0xFFFF)).collect();
            let wb = (rng.next() % 300) as usize;
            let hb = (rng.next() % 300) as usize;
            let (bw, bh) = block_shape(bpe, block);
            let want = reference(&eq, wb, hb, bw, bh, block);
            let (xm, ym) = split(&eq);
            let g = Grid::new(&xm, &ym, wb, hb, bw, bh, block).unwrap();
            let mut got = vec![0i64; wb * hb];
            fill_offsets(&g, &mut got);
            assert!(got.iter().zip(&want).all(|(&a, &b)| a as u64 == b), "case {case}");
            let max = want.iter().copied().max().unwrap_or(0) / bpe;
            let n = (max + 1) as usize * bpe as usize + (rng.next() % 5) as usize;
            let src: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            let mut out = vec![0u8; wb * hb * bpe as usize];
            assert!(detile_into(&g, &src, bpe as usize, &mut out), "case {case}");
            for (i, &off) in want.iter().enumerate() {
                let at = (off / bpe) as usize * bpe as usize;
                let e = bpe as usize;
                assert_eq!(&out[i * e..i * e + e], &src[at..at + e], "case {case} elem {i}");
            }
            if wb * hb > 0 {
                let short = (max as usize) * bpe as usize + bpe as usize - 1;
                assert!(!detile_into(&g, &src[..short], bpe as usize, &mut out), "case {case} short");
            }
        }
    }

    #[test]
    fn large_grid_uses_threads_and_matches() {
        let eq: Vec<(u64, u64)> = (0..16).map(|i| if i % 2 == 0 { (1 << (i / 2), 1 << 7) } else { (1 << 5, 1 << (i / 2)) }).collect();
        let (xm, ym) = split(&eq);
        let (bw, bh) = block_shape(4, 65536);
        let (wb, hb) = (700, 333);
        let g = Grid::new(&xm, &ym, wb, hb, bw, bh, 65536).unwrap();
        let want = reference(&eq, wb, hb, bw, bh, 65536);
        let mut got = vec![0i64; wb * hb];
        fill_offsets(&g, &mut got);
        assert!(got.iter().zip(&want).all(|(&a, &b)| a as u64 == b));
        let n = (want.iter().copied().max().unwrap() / 4 * 4) as usize + 4;
        let src: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        let mut out = vec![0u8; wb * hb * 4];
        assert!(detile_into(&g, &src, 4, &mut out));
        for (i, &off) in want.iter().enumerate() {
            let at = (off / 4 * 4) as usize;
            assert_eq!(&out[i * 4..i * 4 + 4], &src[at..at + 4]);
        }
        assert!(!detile_into(&g, &src[..n - 1], 4, &mut out));
    }

    #[test]
    fn refuses_what_numpy_would_wrap() {
        assert_eq!(Grid::new(&[1], &[1], 4, 4, 0, 1, 256).err(), Some(REFUSE_PARAMS));
        assert_eq!(Grid::new(&[1 << 63], &[0], 4, 4, 1, 1, 256).err(), Some(REFUSE_PARAMS));
        assert_eq!(Grid::new(&[1], &[1], 1 << 20, 1 << 20, 1, 1, 1 << 40).err(), Some(REFUSE_OVERFLOW));
        assert!(Grid::new(&[], &[], 0, 0, 1, 1, 256).is_ok());
    }

    #[test]
    fn ffi_entry_points() {
        unsafe {
            assert_eq!(amd_swizzle_native_abi(), ABI);
            let (xm, ym) = ([1u64, 0], [0u64, 1]);
            let mut off = [0i64; 4];
            assert_eq!(amd_swizzle_offsets(xm.as_ptr(), ym.as_ptr(), 2, 2, 2, 2, 2, 4, off.as_mut_ptr()), 0);
            assert_eq!(off, [0, 1, 2, 3]);
            let src = [10u8, 11, 12, 13];
            let mut out = [0u8; 4];
            let r = amd_swizzle_detile(src.as_ptr(), 4, 1, xm.as_ptr(), ym.as_ptr(), 2, 2, 2, 2, 2, 4, out.as_mut_ptr());
            assert_eq!((r, out), (0, [10, 11, 12, 13]));
            let r = amd_swizzle_detile(src.as_ptr(), 3, 1, xm.as_ptr(), ym.as_ptr(), 2, 2, 2, 2, 2, 4, out.as_mut_ptr());
            assert_eq!(r, 1);
            assert_eq!(amd_swizzle_detile(src.as_ptr(), 4, 0, xm.as_ptr(), ym.as_ptr(), 2, 2, 2, 2, 2, 4, out.as_mut_ptr()), REFUSE_PARAMS);
            let r = amd_swizzle_detile(std::ptr::null(), 0, 4, std::ptr::null(), std::ptr::null(), 0, 0, 7, 1, 1, 4, std::ptr::null_mut());
            assert_eq!(r, 0);
        }
    }
}
