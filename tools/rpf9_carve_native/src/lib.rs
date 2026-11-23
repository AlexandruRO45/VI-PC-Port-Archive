//! Optional native candidate scan for `tools/rpf9_carve.py`.
//!
//! Standard library only. `rpf9_next` does what one turn of the Python loop does until it finds a stream it keeps:
//! the `ZHDR` header search, the Bink 2 header check and the zlib inflate. The inflate accepts exactly the streams
//! `zlib.decompressobj()` (window bits 15) inflates to its end inside the scan window, with the same output and the
//! same consumed length. Every stream zlib rejects or that runs past the window is rejected here as well; the
//! Python loop treats both the same way (`pos = s + 1`), so which of zlib's errors it would have been does not
//! matter. The rules followed are zlib's `inflate.c` and `inftrees.c` with their default build (no
//! `INFLATE_STRICT`): a code-length code must be complete, a literal/length or distance code may only be
//! incomplete when it is one code of length 1, the distance code may be empty, a distance may not reach before
//! the start of the stream, and the Adler-32 trailer must match.
//!
//! Python keeps the bookkeeping (`ModelSystemPages`, the index rows, the file writes) and stays the reference:
//! a negative return means "refused", and the caller runs its Python code for that step instead.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

const FAST_BITS: u32 = 12;
const FAST_MASK: u64 = (1 << FAST_BITS) - 1;
const MAX_BITS: usize = 15;

const LBASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227,
    258,
];
const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
    6145, 8193, 12289, 16385, 24577,
];
const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// LSB-first bit reader over one candidate's bytes (from its header to the end of the scan window).
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Bits { data, pos: 0, buf: 0, cnt: 0 }
    }

    /// Top the buffer up to at least 56 bits, or with every byte left.
    #[inline]
    fn refill(&mut self) {
        if self.cnt > 56 {
            return;
        }
        if self.pos + 8 <= self.data.len() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&self.data[self.pos..self.pos + 8]);
            let take = (63 - self.cnt) / 8;
            let v = u64::from_le_bytes(w) & ((1u64 << (take * 8)) - 1);
            self.buf |= v << self.cnt;
            self.pos += take as usize;
            self.cnt += take * 8;
        } else {
            while self.cnt <= 56 && self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << self.cnt;
                self.pos += 1;
                self.cnt += 8;
            }
        }
    }

    /// The next `n` bits (n <= 32), or None when the input ends first.
    #[inline]
    fn bits(&mut self, n: u32) -> Option<u32> {
        if self.cnt < n {
            self.refill();
            if self.cnt < n {
                return None;
            }
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.cnt -= n;
        Some(v)
    }

    #[inline]
    fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.cnt -= n;
    }

    /// Drop the bits up to the next byte boundary.
    fn align(&mut self) {
        let r = self.cnt % 8;
        self.consume(r);
    }

    /// Bytes read so far, not counting whole bytes still in the buffer (call after `align`).
    fn consumed(&self) -> usize {
        self.pos - (self.cnt / 8) as usize
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Kind {
    Codes,
    Lens,
    Dists,
}

/// Canonical Huffman decoder: a `FAST_BITS` lookup table, then puff's count/offset walk for longer codes.
struct Huff {
    fast: Vec<u16>, // (symbol << 4) | length; 0 = not in the table
    count: [u16; MAX_BITS + 1],
    symbol: Vec<u16>,
}

impl Huff {
    fn new() -> Self {
        Huff { fast: vec![0; 1 << FAST_BITS], count: [0; MAX_BITS + 1], symbol: Vec::with_capacity(320) }
    }

    /// Build from code lengths with zlib's `inflate_table` rules; false where zlib reports an invalid set.
    fn build(&mut self, lens: &[u8], kind: Kind) -> bool {
        self.count = [0; MAX_BITS + 1];
        for &l in lens {
            self.count[l as usize] += 1;
        }
        let mut max = MAX_BITS;
        while max >= 1 && self.count[max] == 0 {
            max -= 1;
        }
        self.fast.iter_mut().for_each(|e| *e = 0);
        self.symbol.clear();
        if max == 0 {
            // No codes at all. zlib accepts this table; for distances every lookup is then an invalid code,
            // and an all-zero code-length code only ever gives length 0, so the block always fails later
            // (no end-of-block code). Both cases reject the stream, as here.
            self.count[0] = 0;
            return kind == Kind::Dists;
        }
        let mut left: i32 = 1;
        for len in 1..=MAX_BITS {
            left <<= 1;
            left -= self.count[len] as i32;
            if left < 0 {
                return false; // over-subscribed
            }
        }
        if left > 0 && (kind == Kind::Codes || max != 1) {
            return false; // incomplete
        }
        self.count[0] = 0;
        let mut offs = [0u16; MAX_BITS + 2];
        for len in 1..=MAX_BITS {
            offs[len + 1] = offs[len] + self.count[len];
        }
        self.symbol.resize(lens.len(), 0);
        let mut next = offs;
        for (sym, &l) in lens.iter().enumerate() {
            if l != 0 {
                self.symbol[next[l as usize] as usize] = sym as u16;
                next[l as usize] += 1;
            }
        }
        // Canonical codes, reversed into stream bit order, for the codes short enough for the table.
        let mut code: u32 = 0;
        let mut idx = 0usize;
        for len in 1..=MAX_BITS {
            for _ in 0..self.count[len] {
                if len as u32 <= FAST_BITS {
                    let mut r = 0u32;
                    for b in 0..len {
                        r |= ((code >> b) & 1) << (len - 1 - b);
                    }
                    let e = (self.symbol[idx] << 4) | len as u16;
                    let mut k = r as usize;
                    while k < (1 << FAST_BITS) {
                        self.fast[k] = e;
                        k += 1 << len;
                    }
                }
                code += 1;
                idx += 1;
            }
            code <<= 1;
        }
        true
    }

    /// The next symbol, or None for an invalid code or the end of the input.
    #[inline]
    fn decode(&self, b: &mut Bits) -> Option<u16> {
        if b.cnt < FAST_BITS {
            b.refill();
        }
        if b.cnt >= FAST_BITS {
            let e = self.fast[(b.buf & FAST_MASK) as usize];
            if e != 0 {
                b.consume((e & 15) as u32);
                return Some(e >> 4);
            }
        }
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=MAX_BITS {
            code |= b.bits(1)? as i32;
            let count = self.count[len] as i32;
            if code - first < count {
                return Some(self.symbol[(index + code - first) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

pub struct Ctx {
    out: Vec<u8>,
    lit: Huff,
    dist: Huff,
    codes: Huff,
    fixed_lit: Huff,
    fixed_dist: Huff,
    lens: [u8; 320],
}

impl Ctx {
    fn new() -> Self {
        let mut fl = [0u8; 288];
        fl[..144].fill(8);
        fl[144..256].fill(9);
        fl[256..280].fill(7);
        fl[280..].fill(8);
        let mut fixed_lit = Huff::new();
        fixed_lit.build(&fl, Kind::Lens);
        let mut fixed_dist = Huff::new();
        fixed_dist.build(&[5u8; 32], Kind::Dists);
        Ctx {
            out: Vec::new(),
            lit: Huff::new(),
            dist: Huff::new(),
            codes: Huff::new(),
            fixed_lit,
            fixed_dist,
            lens: [0; 320],
        }
    }
}

/// Room for `n` more bytes in `out`, grown geometrically with `try_reserve_exact`: None when the allocation
/// fails, so a huge output is refused (and Python decides) instead of aborting the process.
#[inline]
fn room(out: &mut Vec<u8>, n: usize) -> Option<()> {
    if out.capacity() - out.len() < n {
        let want = n.max(out.len()).max(1 << 16);
        out.try_reserve_exact(want).ok()?;
    }
    Some(())
}

/// Append the `len` bytes that start `d` bytes back (1 <= d <= out.len(), len <= 258).
#[inline]
fn copy_match(out: &mut Vec<u8>, d: usize, len: usize) -> Option<()> {
    room(out, len + 16)?;
    let n = out.len();
    let start = n - d;
    if d >= 16 {
        // 16-byte pieces may write up to 15 bytes past `len`, inside the room reserved above and then cut off
        // by `set_len`; a piece never reads bytes it writes, since the source is at least 16 bytes behind.
        let p = out.as_mut_ptr();
        let mut i = 0;
        while i < len {
            // SAFETY: start + i + 16 <= n + i + 16 <= n + len + 15 < capacity, and the ranges do not overlap.
            unsafe { std::ptr::copy_nonoverlapping(p.add(start + i), p.add(n + i), 16) };
            i += 16;
        }
        // SAFETY: the first `len` bytes past `n` were all written above.
        unsafe { out.set_len(n + len) };
    } else {
        // An overlapping match repeats its last `d` bytes: copy what is already there, which doubles each time.
        let mut done = 0;
        while done < len {
            let k = (len - done).min(out.len() - start - done);
            out.extend_from_within(start + done..start + done + k);
            done += k;
        }
    }
    Some(())
}

/// Literal/length and distance symbols of one compressed block into `out`; None on any error.
fn codes(out: &mut Vec<u8>, b: &mut Bits, lit: &Huff, dist: &Huff) -> Option<()> {
    loop {
        room(out, 258)?;
        if b.cnt < 48 {
            b.refill();
        }
        // With 48 bits in hand a whole literal or match fits (15 + 5 + 15 + 13 bits): read the table
        // entries and extra bits without the end-of-input checks. Anything else takes the checked path.
        if b.cnt >= 48 {
            let e = lit.fast[(b.buf & FAST_MASK) as usize];
            if e != 0 && (e >> 4) < 256 {
                b.consume((e & 15) as u32);
                out.push((e >> 4) as u8);
                let e = lit.fast[(b.buf & FAST_MASK) as usize];
                if e != 0 && (e >> 4) < 256 && (e & 15) as u32 <= b.cnt {
                    b.consume((e & 15) as u32);
                    out.push((e >> 4) as u8);
                }
                continue;
            }
        }
        let sym = lit.decode(b)? as usize;
        if sym < 256 {
            out.push(sym as u8);
            continue;
        }
        if sym == 256 {
            return Some(());
        }
        let li = sym - 257;
        if li >= 29 {
            return None; // 286 and 287 of the fixed code
        }
        let len = LBASE[li] as usize + b.bits(LEXT[li] as u32)? as usize;
        let ds = dist.decode(b)? as usize;
        if ds >= 30 {
            return None; // 30 and 31 of the fixed code
        }
        let d = DBASE[ds] as usize + b.bits(DEXT[ds] as u32)? as usize;
        if d > out.len() {
            return None; // invalid distance too far back
        }
        copy_match(out, d, len)?;
    }
}

/// Inflate the zlib stream at the start of `data` into `ctx.out`; Some(bytes consumed) when it ends cleanly
/// inside `data`, None where zlib would raise an error or stop short of the end of the stream.
fn inflate_zlib(ctx: &mut Ctx, data: &[u8]) -> Option<usize> {
    ctx.out.clear();
    if data.len() < 2 {
        return None;
    }
    let (cmf, flg) = (data[0] as u32, data[1] as u32);
    if ((cmf << 8) | flg) % 31 != 0 || cmf & 15 != 8 || (cmf >> 4) + 8 > 15 || flg & 0x20 != 0 {
        return None; // header check, method, window size, preset dictionary (Z_NEED_DICT is an error here)
    }
    let mut b = Bits::new(&data[2..]);
    let Ctx { out, lit, dist, codes: cl, fixed_lit, fixed_dist, lens } = ctx;
    loop {
        let last = b.bits(1)?;
        match b.bits(2)? {
            0 => {
                b.align();
                let n = b.bits(16)?;
                let nn = b.bits(16)?;
                if n != nn ^ 0xFFFF {
                    return None;
                }
                let mut n = n as usize;
                room(out, n)?;
                while n > 0 && b.cnt >= 8 {
                    out.push(b.bits(8)? as u8);
                    n -= 1;
                }
                if n > 0 {
                    // The buffer is empty here, so the reader's position is the next unread byte.
                    let at = b.pos;
                    if at + n > b.data.len() {
                        return None;
                    }
                    out.extend_from_slice(&b.data[at..at + n]);
                    b.pos += n;
                }
            }
            1 => codes(out, &mut b, fixed_lit, fixed_dist)?,
            2 => {
                let nlen = b.bits(5)? as usize + 257;
                let ndist = b.bits(5)? as usize + 1;
                let ncode = b.bits(4)? as usize + 4;
                if nlen > 286 || ndist > 30 {
                    return None;
                }
                let mut cll = [0u8; 19];
                for &i in CL_ORDER.iter().take(ncode) {
                    cll[i] = b.bits(3)? as u8;
                }
                if !cl.build(&cll, Kind::Codes) {
                    return None;
                }
                let total = nlen + ndist;
                let mut have = 0usize;
                while have < total {
                    let sym = cl.decode(&mut b)?;
                    if sym < 16 {
                        lens[have] = sym as u8;
                        have += 1;
                        continue;
                    }
                    let (val, copy) = match sym {
                        16 => {
                            if have == 0 {
                                return None;
                            }
                            (lens[have - 1], 3 + b.bits(2)? as usize)
                        }
                        17 => (0, 3 + b.bits(3)? as usize),
                        _ => (0, 11 + b.bits(7)? as usize),
                    };
                    if have + copy > total {
                        return None;
                    }
                    lens[have..have + copy].fill(val);
                    have += copy;
                }
                if lens[256] == 0 {
                    return None; // missing end-of-block
                }
                if !lit.build(&lens[..nlen], Kind::Lens) || !dist.build(&lens[nlen..total], Kind::Dists) {
                    return None;
                }
                codes(out, &mut b, lit, dist)?;
            }
            _ => return None,
        }
        if last == 1 {
            break;
        }
    }
    b.align();
    let mut check = 0u32;
    for _ in 0..4 {
        check = (check << 8) | b.bits(8)?;
    }
    if check != adler32(out) {
        return None;
    }
    Some(2 + b.consumed())
}

#[inline]
fn u32_at(buf: &[u8], at: usize) -> u64 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as u64
}

/// Next offset in buf[from..end] that `ZHDR` (`\x78[\x01\x5e\x9c\xda]|KB2[a-z]`) matches, the match inside end.
fn find_header(buf: &[u8], from: usize, end: usize) -> Option<usize> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let mut s = from;
    loop {
        // Skip 8 bytes at a time while none of them is 0x78 or 'K'.
        while s + 8 <= end {
            let mut w = [0u8; 8];
            w.copy_from_slice(&buf[s..s + 8]);
            let v = u64::from_le_bytes(w);
            let x = v ^ (0x78 * LO);
            let k = v ^ (0x4B * LO);
            if ((x.wrapping_sub(LO) & !x) | (k.wrapping_sub(LO) & !k)) & HI != 0 {
                break;
            }
            s += 8;
        }
        let stop = (s + 8).min(end);
        while s < stop {
            match buf[s] {
                0x78 if s + 2 <= end && matches!(buf[s + 1], 0x01 | 0x5E | 0x9C | 0xDA) => return Some(s),
                b'K' if s + 4 <= end && buf[s + 1] == b'B' => {
                    if buf[s + 2] == b'2' && buf[s + 3].is_ascii_lowercase() {
                        return Some(s);
                    }
                }
                _ => {}
            }
            s += 1;
        }
        if s >= end {
            return None;
        }
    }
}

/// A new scan context (reused buffers); free it with `rpf9_ctx_free`.
#[no_mangle]
pub extern "C" fn rpf9_ctx_new() -> *mut Ctx {
    catch_unwind(|| Box::into_raw(Box::new(Ctx::new()))).unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `ctx` comes from `rpf9_ctx_new` and is not used afterwards (null is ignored).
#[no_mangle]
pub unsafe extern "C" fn rpf9_ctx_free(ctx: *mut Ctx) {
    if !ctx.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(ctx))));
    }
}

/// Interface version the Python loader checks.
#[no_mangle]
pub extern "C" fn rpf9_abi() -> u32 {
    1
}

/// From `pos`, the next stream the Python loop keeps.
///
/// Returns 0 when no header is left in buf[pos..end], 1 for a zlib stream (`res` = offset, consumed bytes,
/// inflated size, pointer to the inflated bytes, valid until the next call on `ctx`), 2 for a Bink 2 video
/// (`res` = offset, size) and a negative value to refuse (the caller runs the Python step instead).
/// A zlib stream is kept when its inflated size is at least the Python loop's `minimum_size`: 1 when it is
/// adjacent to `last` (the next 16-byte boundary, zero bytes between) and `small_ok` is set, else 64.
///
/// # Safety
/// `buf` points to `size` readable bytes, `ctx` comes from `rpf9_ctx_new`, `res` to 4 writable u64.
#[no_mangle]
pub unsafe extern "C" fn rpf9_next(
    ctx: *mut Ctx,
    buf: *const u8,
    size: usize,
    pos: usize,
    end: usize,
    last: usize,
    small_ok: i32,
    res: *mut u64,
) -> i32 {
    if ctx.is_null() || buf.is_null() || res.is_null() || end > size || pos > end || last > pos {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| next_hit(&mut *ctx, slice::from_raw_parts(buf, size), pos, end, last,
                                             small_ok != 0, slice::from_raw_parts_mut(res, 4))))
    .unwrap_or(-2)
}

fn next_hit(ctx: &mut Ctx, buf: &[u8], pos: usize, end: usize, last: usize, small_ok: bool, res: &mut [u64]) -> i32 {
    let mut p = pos;
    while p < end {
        let s = match find_header(buf, p, end) {
            Some(s) => s,
            None => return 0,
        };
        if buf[s] == b'K' {
            if s + 0x1C > end {
                p = s + 1;
                continue;
            }
            let blen = u32_at(buf, s + 4) + 8;
            let (w, h) = (u32_at(buf, s + 0x14), u32_at(buf, s + 0x18));
            if !(64 <= blen && blen <= (end - s) as u64 && (16..=8192).contains(&w) && (16..=8192).contains(&h)) {
                p = s + 1;
                continue;
            }
            res[0] = s as u64;
            res[1] = blen;
            res[2] = 0;
            res[3] = 0;
            return 2;
        }
        if let Some(used) = inflate_zlib(ctx, &buf[s..end]) {
            let adjacent = s == ((last + 15) & !15) && buf[last..s].iter().all(|&x| x == 0);
            let minimum = if adjacent && small_ok { 1 } else { 64 };
            if ctx.out.len() >= minimum {
                res[0] = s as u64;
                res[1] = used as u64;
                res[2] = ctx.out.len() as u64;
                res[3] = ctx.out.as_ptr() as u64;
                return 1;
            }
        }
        p = s + 1;
    }
    0
}

/// 1 when buf[a..b] is all zero bytes, 0 when not, -1 for a range outside the buffer.
///
/// # Safety
/// `buf` points to `size` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn rpf9_all_zero(buf: *const u8, size: usize, a: usize, b: usize) -> i32 {
    if buf.is_null() || a > b || b > size {
        return -1;
    }
    let s = slice::from_raw_parts(buf, size);
    catch_unwind(|| all_zero(&s[a..b])).unwrap_or(-2)
}

fn all_zero(s: &[u8]) -> i32 {
    let mut chunks = s.chunks_exact(8);
    let mut acc = 0u64;
    for c in &mut chunks {
        let mut w = [0u8; 8];
        w.copy_from_slice(c);
        acc |= u64::from_le_bytes(w);
        if acc != 0 {
            return 0;
        }
    }
    (chunks.remainder().iter().all(|&x| x == 0) && acc == 0) as i32
}

/// Byte counts of buf[a..b]: `counts[v]` for each value and `order` the values in order of first appearance
/// (`collections.Counter` order). Returns how many distinct values there are, or -1 for a bad range.
///
/// # Safety
/// `buf` points to `size` readable bytes, `counts` to 256 writable u64, `order` to 256 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn rpf9_byte_counts(
    buf: *const u8,
    size: usize,
    a: usize,
    b: usize,
    counts: *mut u64,
    order: *mut u8,
) -> i32 {
    if buf.is_null() || counts.is_null() || order.is_null() || a > b || b > size {
        return -1;
    }
    let s = &slice::from_raw_parts(buf, size)[a..b];
    let counts = slice::from_raw_parts_mut(counts, 256);
    let order = slice::from_raw_parts_mut(order, 256);
    catch_unwind(AssertUnwindSafe(|| byte_counts(s, counts, order))).unwrap_or(-2)
}

fn byte_counts(s: &[u8], counts: &mut [u64], order: &mut [u8]) -> i32 {
    let mut c = [0u64; 256];
    let mut first = [usize::MAX; 256];
    // Four sub-histograms break the store-to-load dependency of runs of one value.
    let mut sub = [[0u64; 256]; 4];
    let mut q = s.chunks_exact(4);
    for w in &mut q {
        sub[0][w[0] as usize] += 1;
        sub[1][w[1] as usize] += 1;
        sub[2][w[2] as usize] += 1;
        sub[3][w[3] as usize] += 1;
    }
    for &x in q.remainder() {
        sub[0][x as usize] += 1;
    }
    for v in 0..256 {
        c[v] = sub[0][v] + sub[1][v] + sub[2][v] + sub[3][v];
    }
    let mut missing = c.iter().filter(|&&n| n != 0).count();
    let distinct = missing;
    for (i, &x) in s.iter().enumerate() {
        if missing == 0 {
            break;
        }
        if first[x as usize] == usize::MAX {
            first[x as usize] = i;
            missing -= 1;
        }
    }
    let mut vals: Vec<usize> = (0..256).filter(|&v| c[v] != 0).collect();
    vals.sort_by_key(|&v| first[v]);
    counts.copy_from_slice(&c);
    for (i, &v) in vals.iter().enumerate() {
        order[i] = v as u8;
    }
    distinct as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    // zlib.compress(b"hello hello hello hello, synthetic test line " * 3, 9): a dynamic-Huffman stream.
    const DYN: [u8; 42] = [
        120, 218, 203, 72, 205, 201, 201, 87, 200, 64, 39, 117, 20, 138, 43, 243, 74, 50, 82, 75, 50, 147, 21, 74,
        82, 139, 75, 20, 114, 50, 243, 82, 49, 149, 81, 77, 49, 0, 76, 175, 49, 222,
    ];

    fn plain() -> Vec<u8> {
        b"hello hello hello hello, synthetic test line ".repeat(3)
    }

    fn stored(payload: &[u8]) -> Vec<u8> {
        let mut v = vec![0x78, 0x01, 0x01];
        let n = payload.len() as u16;
        v.extend_from_slice(&n.to_le_bytes());
        v.extend_from_slice(&(!n).to_le_bytes());
        v.extend_from_slice(payload);
        v.extend_from_slice(&adler32(payload).to_be_bytes());
        v
    }

    #[test]
    fn match_copy_matches_a_byte_loop() {
        for d in 1..=40usize {
            for len in [3usize, 4, 7, 15, 16, 17, 31, 32, 33, 100, 257, 258] {
                let mut out: Vec<u8> = (0..50u8).map(|i| i.wrapping_mul(37)).collect();
                let mut want = out.clone();
                for _ in 0..len {
                    let v = want[want.len() - d];
                    want.push(v);
                }
                copy_match(&mut out, d, len).unwrap();
                assert_eq!(out, want, "d {d} len {len}");
            }
        }
    }

    #[test]
    fn adler_known_value() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn dynamic_stream_inflates() {
        let mut ctx = Ctx::new();
        let mut data = DYN.to_vec();
        data.extend_from_slice(b"trailing");
        assert_eq!(inflate_zlib(&mut ctx, &data), Some(DYN.len()));
        assert_eq!(ctx.out, plain());
    }

    #[test]
    fn truncated_or_corrupt_stream_is_rejected() {
        let mut ctx = Ctx::new();
        for cut in 0..DYN.len() {
            assert_eq!(inflate_zlib(&mut ctx, &DYN[..cut]), None, "cut {cut}");
        }
        let mut bad = DYN.to_vec();
        *bad.last_mut().unwrap() ^= 1; // Adler-32
        assert_eq!(inflate_zlib(&mut ctx, &bad), None);
    }

    #[test]
    fn stored_block_and_its_length_check() {
        let mut ctx = Ctx::new();
        let payload: Vec<u8> = (0..300u32).map(|i| (i * 7) as u8).collect();
        let s = stored(&payload);
        assert_eq!(inflate_zlib(&mut ctx, &s), Some(s.len()));
        assert_eq!(ctx.out, payload);
        let mut bad = s.clone();
        bad[5] ^= 1; // NLEN
        assert_eq!(inflate_zlib(&mut ctx, &bad), None);
    }

    #[test]
    fn header_rules() {
        let mut ctx = Ctx::new();
        let mut s = stored(b"abc");
        s[1] = 0x20; // FDICT set, check bits still valid: 0x7820 % 31 == 0
        assert_eq!((0x78u32 << 8 | s[1] as u32) % 31, 0);
        assert_eq!(inflate_zlib(&mut ctx, &s), None);
        let mut s = stored(b"abc");
        s[0] = 0x88; // window size 2^16
        s[1] = 0x1C; // 0x881C % 31 == 0
        assert_eq!((0x88u32 << 8 | 0x1C) % 31, 0);
        assert_eq!(inflate_zlib(&mut ctx, &s), None);
    }

    #[test]
    fn distance_before_the_start_is_rejected() {
        // Fixed block: length 3 (code 257 = 0000001), distance 1 (code 0 = 00000) with no output yet.
        let mut ctx = Ctx::new();
        // bits LSB first: BFINAL=1, BTYPE=01, then 7-bit code 0000001 (MSB first), 5-bit 00000, then EOB.
        let data = [0x78, 0x01, 0x03, 0x02, 0x00, 0x00, 0, 0, 0, 0];
        assert_eq!(inflate_zlib(&mut ctx, &data), None);
    }

    #[test]
    fn header_search_matches_the_regex() {
        let buf = b"..x\x78\x9d\x78\x9cKB2AKB2b";
        assert_eq!(find_header(buf, 0, buf.len()), Some(5));
        assert_eq!(find_header(buf, 6, buf.len()), Some(11));
        assert_eq!(find_header(buf, 0, 6), None); // 78 9c would end past `end`
        assert_eq!(find_header(buf, 12, buf.len()), None);
    }

    #[test]
    fn byte_counts_in_first_seen_order() {
        let data = b"bbaacb";
        let mut counts = [0u64; 256];
        let mut order = [0u8; 256];
        let n = unsafe { rpf9_byte_counts(data.as_ptr(), data.len(), 0, 6, counts.as_mut_ptr(), order.as_mut_ptr()) };
        assert_eq!(n, 3);
        assert_eq!(&order[..3], b"bac");
        assert_eq!((counts[b'a' as usize], counts[b'b' as usize], counts[b'c' as usize]), (2, 3, 1));
    }

    #[test]
    fn all_zero_ranges() {
        let mut data = vec![0u8; 100];
        assert_eq!(unsafe { rpf9_all_zero(data.as_ptr(), 100, 0, 100) }, 1);
        data[97] = 1;
        assert_eq!(unsafe { rpf9_all_zero(data.as_ptr(), 100, 0, 100) }, 0);
        assert_eq!(unsafe { rpf9_all_zero(data.as_ptr(), 100, 0, 97) }, 1);
        assert_eq!(unsafe { rpf9_all_zero(data.as_ptr(), 100, 0, 101) }, -1);
    }
}
