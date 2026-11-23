//! Native kernels for `tools/ttpf_mips.py`.
//!
//! Line-by-line ports of `detile` / `detile_volume_slice` (the element copy), the BC1-BC7 block
//! decoders behind `decode_pixels`, and `block_means`. Every integer step and every float
//! operation is the one the Python code performs, in the same order, so the bytes and the IEEE
//! double bits are identical. The tile bit tables, the BPTC partition tables and the BC6H header
//! plans are not copied here: `ttpf_mips.py` passes the ones it uses itself (`PATTERNS_*`,
//! `bcpack_tables_bptc.load_bc7()`, `bcpack._BC6H_INFO` / `bcpack._bc6h_plan`).
//!
//! Every entry point checks its lengths and returns -1, writing nothing it promised, when
//! something is out of range; the caller then runs the Python code, which keeps its own
//! behaviour (and its own errors) for those inputs. `tests/test_ttpf_native.py` compares the two.

use std::slice;

// --- detile ---------------------------------------------------------------------------------

/// Parameters of one `detile` / `detile_volume_slice` copy, as computed by the Python code.
pub struct Detile<'a> {
    /// Element bytes.
    pub eb: u64,
    /// Elements per row and rows.
    pub w: u64,
    pub h: u64,
    /// Tile width and height in elements, and tile bytes.
    pub bw: u64,
    pub bh: u64,
    pub tile: u64,
    /// Element origin inside the tile grid (`tile_x`, `tile_y`; 0 for volumes).
    pub tx: u64,
    pub ty: u64,
    /// Bytes per row of tiles (`tiles_x * tile`).
    pub row_stride: u64,
    /// Added to every row base before the Y bits (`z_base + z_in_tile` for volumes, else 0).
    pub base: u64,
    /// `(mask, shift)` pairs: offset bit `mask` is set when coordinate bit `shift` is set.
    pub x_bits: &'a [(u64, u32)],
    pub y_bits: &'a [(u64, u32)],
}

fn set_bits(mut o: u64, coord: u64, bits: &[(u64, u32)]) -> u64 {
    for &(mask, shift) in bits {
        if shift < 64 && (coord >> shift) & 1 != 0 {
            o |= mask;
        }
    }
    o
}

impl Detile<'_> {
    /// `x_off_of(x)` of the Python code.
    fn x_off(&self, x: u64) -> Option<u64> {
        let o = (x / self.bw).checked_mul(self.tile)?;
        Some(set_bits(o, x.checked_add(self.tx)?, self.x_bits))
    }

    /// `y_base_of(y)` of the Python code.
    fn y_base(&self, y: u64) -> Option<u64> {
        let o = (y / self.bh).checked_mul(self.row_stride)?.checked_add(self.base)?;
        Some(set_bits(o, y.checked_add(self.ty)?, self.y_bits))
    }

    /// Copy every element of `src` into `out` (`w * h * eb` bytes, block-linear). Returns
    /// false, leaving `out` unspecified, when an element would fall outside `src`: the Python
    /// code has its own (truncating) behaviour there, so the caller runs it instead.
    pub fn run(&self, src: &[u8], out: &mut [u8]) -> bool {
        if self.bw == 0 || self.bh == 0 || self.eb == 0 {
            return false;
        }
        let need = match self.w.checked_mul(self.h).and_then(|n| n.checked_mul(self.eb)) {
            Some(n) => n,
            None => return false,
        };
        if out.len() as u64 != need {
            return false;
        }
        if need == 0 {
            return true;
        }
        let mut x_offs = Vec::with_capacity(self.w as usize);
        for x in 0..self.w {
            match self.x_off(x) {
                Some(o) => x_offs.push(o),
                None => return false,
            }
        }
        let mut y_bases = Vec::with_capacity(self.h as usize);
        for y in 0..self.h {
            match self.y_base(y) {
                Some(o) => y_bases.push(o),
                None => return false,
            }
        }
        let max_x = *x_offs.iter().max().unwrap_or(&0);
        let max_y = *y_bases.iter().max().unwrap_or(&0);
        match max_y.checked_add(max_x).and_then(|n| n.checked_add(self.eb)) {
            Some(end) if end <= src.len() as u64 => {}
            _ => return false,
        }
        // Coalesce contiguous X offsets into runs, as `_coalesce_x_runs` does.
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for &xo in &x_offs {
            match runs.last_mut() {
                Some((start, len)) if *start + *len == xo => *len += self.eb,
                _ => runs.push((xo, self.eb)),
            }
        }
        let mut at = 0usize;
        for &yb in &y_bases {
            for &(xo, len) in &runs {
                let off = (yb + xo) as usize;
                let len = len as usize;
                out[at..at + len].copy_from_slice(&src[off..off + len]);
                at += len;
            }
        }
        true
    }
}

// --- BC decoders -----------------------------------------------------------------------------

const W2: [i64; 4] = [0, 21, 43, 64];
const W3: [i64; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [i64; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn weights(bits: u32) -> &'static [i64] {
    match bits {
        2 => &W2,
        3 => &W3,
        _ => &W4,
    }
}

/// `(blk >> pos) & ((1 << n) - 1)` on the 128-bit block, 0 past its end as for a Python int.
fn field(blk: u128, pos: u32, n: u32) -> u32 {
    if pos >= 128 || n == 0 {
        return 0;
    }
    ((blk >> pos) & ((1u128 << n) - 1)) as u32
}

fn block128(data: &[u8], at: usize) -> u128 {
    let mut b = [0u8; 16];
    b.copy_from_slice(&data[at..at + 16]);
    u128::from_le_bytes(b)
}

/// The BPTC partition and anchor tables of `bcpack_tables_bptc.load_bc7()`.
pub struct Bptc<'a> {
    /// 64 x 16 subset indices.
    pub part2: &'a [u8],
    pub part3: &'a [u8],
    pub anchor2: &'a [u8],
    pub anchor3a: &'a [u8],
    pub anchor3b: &'a [u8],
}

/// `(subsets, part_bits, rot_bits, isel_bits, cb, ab, ep_pbits, shared_pbits, ib, ib2)`.
const BC7_MODES: [[u32; 10]; 8] = [
    [3, 4, 0, 0, 4, 0, 1, 0, 3, 0],
    [2, 6, 0, 0, 6, 0, 0, 1, 3, 0],
    [3, 6, 0, 0, 5, 0, 0, 0, 2, 0],
    [2, 6, 0, 0, 7, 0, 1, 0, 2, 0],
    [1, 0, 2, 1, 5, 6, 0, 0, 2, 3],
    [1, 0, 2, 0, 7, 8, 0, 0, 2, 2],
    [1, 0, 0, 0, 7, 7, 1, 0, 4, 0],
    [2, 6, 0, 0, 5, 5, 1, 0, 2, 0],
];

/// `decode_bc7_block`: 16 (r, g, b) in 0..255 as doubles.
pub fn decode_bc7(data: &[u8], at: usize, t: &Bptc) -> [[f64; 3]; 16] {
    let b0 = data[at];
    if b0 == 0 {
        return [[0.0; 3]; 16];
    }
    let mode = b0.trailing_zeros();
    let [ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2] = BC7_MODES[mode as usize];
    let blk = block128(data, at);
    let mut pos = mode + 1;
    let part = field(blk, pos, pb) as usize;
    pos += pb;
    let rot = field(blk, pos, rb);
    pos += rb;
    let isel = if isb != 0 { field(blk, pos, 1) } else { 0 };
    pos += isb;

    let n_ep = (2 * ns) as usize;
    let mut raw = [[0u32; 6]; 4];
    for row in raw.iter_mut().take(3) {
        for v in row.iter_mut().take(n_ep) {
            *v = field(blk, pos, cb);
            pos += cb;
        }
    }
    if ab != 0 {
        for v in raw[3].iter_mut().take(n_ep) {
            *v = field(blk, pos, ab);
            pos += ab;
        }
    }

    let mut c_bits = cb;
    let mut a_bits = ab;
    let mut pbits = [0u32; 6];
    let has_pbits = epb != 0 || spb != 0;
    if epb != 0 {
        for (k, p) in pbits.iter_mut().enumerate().take(n_ep) {
            *p = field(blk, pos + k as u32, 1);
        }
        pos += n_ep as u32;
        c_bits += 1;
        if ab != 0 {
            a_bits += 1;
        }
    } else if spb != 0 {
        let mut sp = [0u32; 3];
        for (s, v) in sp.iter_mut().enumerate().take(ns as usize) {
            *v = field(blk, pos + s as u32, 1);
        }
        pos += ns;
        for (k, p) in pbits.iter_mut().enumerate().take(n_ep) {
            *p = sp[k >> 1];
        }
        c_bits += 1;
    }

    let mut ep = [[255u32; 6]; 4];
    let (c_sh, c_rep) = (8 - c_bits, 2 * c_bits - 8);
    for ch in 0..3 {
        for k in 0..n_ep {
            let v = if has_pbits { (raw[ch][k] << 1) | pbits[k] } else { raw[ch][k] };
            ep[ch][k] = (v << c_sh) | (v >> c_rep);
        }
    }
    if ab != 0 {
        let (a_sh, a_rep) = (8 - a_bits, 2 * a_bits - 8);
        for k in 0..n_ep {
            let v = if has_pbits && epb != 0 { (raw[3][k] << 1) | pbits[k] } else { raw[3][k] };
            ep[3][k] = (v << a_sh) | (v >> a_rep);
        }
    }

    let (sm, anchors): (Option<&[u8]>, [usize; 3]) = match ns {
        1 => (None, [0, 0, 0]),
        2 => (Some(&t.part2[part * 16..part * 16 + 16]), [0, t.anchor2[part] as usize, 0]),
        _ => (Some(&t.part3[part * 16..part * 16 + 16]),
              [0, t.anchor3a[part] as usize, t.anchor3b[part] as usize]),
    };
    let sub_of = |p: usize| sm.map_or(0, |s| s[p] as usize);

    let mut idx1 = [0u32; 16];
    for (p, v) in idx1.iter_mut().enumerate() {
        if p == anchors[sub_of(p)] {
            *v = field(blk, pos, ib - 1);
            pos += ib - 1;
        } else {
            *v = field(blk, pos, ib);
            pos += ib;
        }
    }
    let (c_idx, c_wt, a_idx, a_wt);
    if ib2 != 0 {
        let mut idx2 = [0u32; 16];
        for (p, v) in idx2.iter_mut().enumerate() {
            let n = if p == 0 { ib2 - 1 } else { ib2 };
            *v = field(blk, pos, n);
            pos += n;
        }
        if isel != 0 {
            (c_idx, c_wt, a_idx, a_wt) = (idx2, weights(ib2), idx1, weights(ib));
        } else {
            (c_idx, c_wt, a_idx, a_wt) = (idx1, weights(ib), idx2, weights(ib2));
        }
    } else {
        (c_idx, c_wt, a_idx, a_wt) = (idx1, weights(ib), idx1, weights(ib));
    }

    let mut out = [[0.0f64; 3]; 16];
    for (p, o) in out.iter_mut().enumerate() {
        let s2 = 2 * sub_of(p);
        let wc = c_wt[c_idx[p] as usize];
        let iwc = 64 - wc;
        let lerp = |e: &[u32; 6], iw: i64, w: i64| (iw * e[s2] as i64 + w * e[s2 + 1] as i64 + 32) >> 6;
        let mut rgb = [lerp(&ep[0], iwc, wc), lerp(&ep[1], iwc, wc), lerp(&ep[2], iwc, wc)];
        if rot != 0 {
            let wa = a_wt[a_idx[p] as usize];
            rgb[rot as usize - 1] = lerp(&ep[3], 64 - wa, wa);
        }
        *o = [rgb[0] as f64, rgb[1] as f64, rgb[2] as f64];
    }
    out
}

/// One BC6H mode of `bcpack._BC6H_INFO` with its `bcpack._bc6h_plan` bit plan.
#[derive(Clone)]
pub struct Bc6hMode {
    pub trans: bool,
    pub base: u32,
    pub deltas: [u32; 3],
    pub reg: u32,
    pub hdr: u32,
    /// `(channel, endpoint_slot, bit_index, stream_pos)`; channel -1 is the partition field.
    pub plan: Vec<(i32, i32, u32, u32)>,
}

/// Stride, in `i32`, of one mode in the flat table `ttpf_mips.py` passes: valid, trans, base,
/// three deltas, reg, hdr, plan length, then `BC6H_PLAN_MAX` plan items of four values.
pub const BC6H_PLAN_MAX: usize = 128;
pub const BC6H_STRIDE: usize = 9 + 4 * BC6H_PLAN_MAX;

/// Parse the 32-mode flat table; None when a value is out of the range this port handles.
pub fn bc6h_modes(flat: &[i32]) -> Option<Vec<Option<Bc6hMode>>> {
    if flat.len() != 32 * BC6H_STRIDE {
        return None;
    }
    let mut modes = Vec::with_capacity(32);
    for m in flat.chunks(BC6H_STRIDE) {
        if m[0] == 0 {
            modes.push(None);
            continue;
        }
        let n = m[8];
        let small = |v: i32| (0..=16).contains(&v);
        if !(0..=BC6H_PLAN_MAX as i32).contains(&n) || !small(m[2]) || m[2] == 0
            || !small(m[3]) || !small(m[4]) || !small(m[5]) || !(1..=2).contains(&m[6])
            || !(0..=128).contains(&m[7]) {
            return None;
        }
        let mut plan = Vec::with_capacity(n as usize);
        for i in 0..n as usize {
            let it = &m[9 + 4 * i..13 + 4 * i];
            let (c, q, b, p) = (it[0], it[1], it[2], it[3]);
            let ok = if c < 0 { (0..5).contains(&b) } else { c < 3 && (0..4).contains(&q) && (0..32).contains(&b) };
            if !ok || !(0..=4096).contains(&p) {
                return None;
            }
            plan.push((c, q, b as u32, p as u32));
        }
        modes.push(Some(Bc6hMode {
            trans: m[1] != 0,
            base: m[2] as u32,
            deltas: [m[3] as u32, m[4] as u32, m[5] as u32],
            reg: m[6] as u32,
            hdr: m[7] as u32,
            plan,
        }));
    }
    Some(modes)
}

fn sext(v: i64, bits: u32) -> i64 {
    let sign = 1i64 << (bits - 1);
    (v ^ sign) - sign
}

/// `_bc6h_unquantize`.
fn bc6h_unquantize(comp: i64, bits: u32, signed: bool) -> i64 {
    if !signed {
        if bits >= 15 {
            return comp;
        }
        if comp == 0 {
            return 0;
        }
        if comp == (1i64 << bits) - 1 {
            return 0xFFFF;
        }
        return ((comp << 16) + 0x8000) >> bits;
    }
    if bits >= 16 {
        return if comp < -32767 { -32767 } else { comp };
    }
    let neg = comp < 0;
    let c = if neg { -comp } else { comp };
    let unq = if c == 0 {
        0
    } else if c >= (1i64 << (bits - 1)) - 1 {
        0x7FFF
    } else {
        ((c << 15) + 0x4000) >> (bits - 1)
    };
    if neg { -unq } else { unq }
}

/// IEEE 754 binary16 to double, exactly as `struct.unpack("<e")`.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = ((h as u64) >> 15) << 63;
    let exp = ((h >> 10) & 31) as u64;
    let mant = (h & 1023) as u64;
    let bits = match exp {
        // Subnormal: mant * 2**-24 is exact in a double.
        0 => {
            let mag = mant as f64 * (1.0 / 16_777_216.0);
            return if sign != 0 { -mag } else { mag };
        }
        // Infinity, or a quiet NaN with the half payload at the top of the double mantissa.
        31 => 0x7FF0_0000_0000_0000 | (mant << 42) | if mant != 0 { 1 << 51 } else { 0 },
        _ => ((exp + 1008) << 52) | (mant << 42),
    };
    f64::from_bits(sign | bits)
}

/// `_bc6h_finish` followed by the float16 lookup.
fn bc6h_finish(val: i64, signed: bool) -> Option<f64> {
    if !signed {
        let h = (val * 31) >> 6;
        if h < 0 {
            return None; // Python would index the table from its end; not reachable
        }
        return Some(f16_to_f64(h.min(0x7BFF) as u16));
    }
    let neg = val < 0;
    let v = (if neg { -val } else { val }).min(0x7FFF);
    let h = ((v * 31) >> 5).min(0x7BFF) as u16;
    Some(f16_to_f64(if neg { h | 0x8000 } else { h }))
}

/// `decode_bc6h_block`: 16 (r, g, b) doubles. None when the port meets a value it does not
/// cover (a table index past its end), so the caller keeps the Python result.
pub fn decode_bc6h(data: &[u8], at: usize, signed: bool, modes: &[Option<Bc6hMode>], t: &Bptc)
                   -> Option<[[f64; 3]; 16]> {
    let b0 = data[at];
    let code = if (b0 & 3) < 2 { b0 & 3 } else { b0 & 31 };
    let m = match &modes[code as usize] {
        Some(m) => m,
        None => return Some([[0.0; 3]; 16]),
    };
    let blk = block128(data, at);
    let mut f = [[0i64; 4]; 3];
    let mut part = 0usize;
    for &(c, q, bit, pos) in &m.plan {
        if field(blk, pos, 1) != 0 {
            if c < 0 {
                part |= 1 << bit;
            } else {
                f[c as usize][q as usize] |= 1 << bit;
            }
        }
    }
    let base = m.base;
    let mask_base = (1i64 << base) - 1;
    let n_ep = if m.reg == 2 { 4 } else { 2 };
    let mut eps = [[0i64; 4]; 3];
    for c in 0..3 {
        let dc = m.deltas[c];
        let e0 = if signed { sext(f[c][0], base) } else { f[c][0] };
        eps[c][0] = bc6h_unquantize(e0, base, signed);
        for q in 1..n_ep {
            let mut eq = if signed || m.trans {
                if dc == 0 {
                    return None;
                }
                sext(f[c][q], dc)
            } else {
                f[c][q]
            };
            if m.trans {
                eq = (e0 + eq) & mask_base;
                if signed {
                    eq = sext(eq, base);
                }
            }
            eps[c][q] = bc6h_unquantize(eq, base, signed);
        }
    }

    let mut pos = m.hdr;
    let mut out = [[0.0f64; 3]; 16];
    if m.reg == 1 {
        for (p, o) in out.iter_mut().enumerate() {
            let nb = if p == 0 { 3 } else { 4 };
            let w = W4[field(blk, pos, nb) as usize];
            pos += nb;
            let iw = 64 - w;
            for c in 0..3 {
                o[c] = bc6h_finish((iw * eps[c][0] + w * eps[c][1] + 32) >> 6, signed)?;
            }
        }
    } else {
        if part >= 64 {
            return None;
        }
        let sm = &t.part2[part * 16..part * 16 + 16];
        let anchor1 = t.anchor2[part] as usize;
        for (p, o) in out.iter_mut().enumerate() {
            let nb = if p == 0 || p == anchor1 { 2 } else { 3 };
            let w = W3[field(blk, pos, nb) as usize];
            pos += nb;
            let iw = 64 - w;
            let s2 = sm[p] as usize * 2;
            if s2 + 1 >= n_ep {
                return None;
            }
            for c in 0..3 {
                o[c] = bc6h_finish((iw * eps[c][s2] + w * eps[c][s2 + 1] + 32) >> 6, signed)?;
            }
        }
    }
    Some(out)
}

const fn levels(max: u32) -> [f64; 64] {
    let mut t = [0.0f64; 64];
    let mut i = 0;
    while i <= max as usize {
        t[i] = i as f64 * 255.0 / max as f64;
        i += 1;
    }
    t
}

/// `_R565`, `_G565` (`i * 255.0 / max`).
const R565: [f64; 64] = levels(31);
const G565: [f64; 64] = levels(63);

/// `_bc1_palette`.
fn bc1_palette(c0: u16, c1: u16, four: bool) -> [[f64; 3]; 4] {
    let r0 = R565[((c0 >> 11) & 31) as usize];
    let g0 = G565[((c0 >> 5) & 63) as usize];
    let b0 = R565[(c0 & 31) as usize];
    let r1 = R565[((c1 >> 11) & 31) as usize];
    let g1 = G565[((c1 >> 5) & 63) as usize];
    let b1 = R565[(c1 & 31) as usize];
    if four || c0 > c1 {
        [[r0, g0, b0], [r1, g1, b1],
         [(2.0 * r0 + r1) / 3.0, (2.0 * g0 + g1) / 3.0, (2.0 * b0 + b1) / 3.0],
         [(r0 + 2.0 * r1) / 3.0, (g0 + 2.0 * g1) / 3.0, (b0 + 2.0 * b1) / 3.0]]
    } else {
        [[r0, g0, b0], [r1, g1, b1], [(r0 + r1) / 2.0, (g0 + g1) / 2.0, (b0 + b1) / 2.0],
         [0.0, 0.0, 0.0]]
    }
}

/// `_bc4_palette` (one grey level per entry).
fn bc4_palette(e0: u8, e1: u8) -> [f64; 8] {
    let (e0, e1) = (e0 as i32, e1 as i32);
    let mut v = [e0, e1, 0, 0, 0, 0, 0, 255];
    if e0 > e1 {
        for i in 1..7 {
            v[i as usize + 1] = ((7 - i) * e0 + i * e1) / 7;
        }
    } else {
        for i in 1..5 {
            v[i as usize + 1] = ((5 - i) * e0 + i * e1) / 5;
        }
    }
    v.map(|x| x as f64)
}

fn bc4_block(data: &[u8], at: usize) -> [f64; 16] {
    let pal = bc4_palette(data[at], data[at + 1]);
    let mut bits = 0u64;
    for i in 0..6 {
        bits |= (data[at + 2 + i] as u64) << (8 * i);
    }
    let mut out = [0.0; 16];
    for (p, o) in out.iter_mut().enumerate() {
        *o = pal[((bits >> (3 * p)) & 7) as usize];
    }
    out
}

/// The block families of `decode_pixels`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// BC1 (71, 72): 8-byte blocks, three-colour mode when c0 <= c1.
    Bc1,
    /// BC2 / BC3 colour (74, 75, 77, 78): the BC1 block after 8 alpha bytes, always four colours.
    Bc23,
    Bc4,
    Bc5,
    Bc6hUf,
    Bc6hSf,
    Bc7,
}

impl Kind {
    pub fn from_code(code: u32) -> Option<Kind> {
        Some(match code {
            0 => Kind::Bc1,
            1 => Kind::Bc23,
            2 => Kind::Bc4,
            3 => Kind::Bc5,
            4 => Kind::Bc6hUf,
            5 => Kind::Bc6hSf,
            6 => Kind::Bc7,
            _ => return None,
        })
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Kind::Bc1 | Kind::Bc4 => 8,
            _ => 16,
        }
    }
}

/// Decode a `blocks_w` x `blocks_h` grid into `out` (`blocks_w * 4` pixels per row, three
/// doubles per pixel), as `decode_pixels` orders it. False when a block meets a value the port
/// does not cover; `out` is then unspecified.
pub fn decode_surface(kind: Kind, data: &[u8], blocks_w: usize, blocks_h: usize, t: &Bptc,
                      modes: &[Option<Bc6hMode>], out: &mut [f64]) -> bool {
    decode_rows(kind, data, blocks_w, 0, blocks_h, t, modes, out)
}

/// [`decode_surface`] for the block rows `by0..by1` only; `out` holds just those rows (the
/// pixels of block row `by0` come first), so a caller can decode a large surface in bands.
#[allow(clippy::too_many_arguments)]
pub fn decode_rows(kind: Kind, data: &[u8], blocks_w: usize, by0: usize, by1: usize, t: &Bptc,
                   modes: &[Option<Bc6hMode>], out: &mut [f64]) -> bool {
    let row = blocks_w * 4 * 3;
    let stride = kind.block_bytes();
    let mut cache: Option<(u32, [[f64; 3]; 4])> = None;
    for by in by0..by1 {
        for bx in 0..blocks_w {
            let at = (by * blocks_w + bx) * stride;
            let px: [[f64; 3]; 16] = match kind {
                Kind::Bc1 | Kind::Bc23 => {
                    let at = if kind == Kind::Bc23 { at + 8 } else { at };
                    let c0 = u16::from_le_bytes([data[at], data[at + 1]]);
                    let c1 = u16::from_le_bytes([data[at + 2], data[at + 3]]);
                    let idx = u32::from_le_bytes([data[at + 4], data[at + 5], data[at + 6], data[at + 7]]);
                    let key = (c0 as u32) << 16 | c1 as u32;
                    let pal = match cache {
                        Some((k, p)) if k == key => p,
                        _ => {
                            let p = bc1_palette(c0, c1, kind == Kind::Bc23);
                            cache = Some((key, p));
                            p
                        }
                    };
                    let mut o = [[0.0; 3]; 16];
                    for (p, v) in o.iter_mut().enumerate() {
                        *v = pal[((idx >> (2 * p)) & 3) as usize];
                    }
                    o
                }
                Kind::Bc4 => bc4_block(data, at).map(|v| [v, v, v]),
                Kind::Bc5 => {
                    let a = bc4_block(data, at);
                    let b = bc4_block(data, at + 8);
                    let mut o = [[0.0; 3]; 16];
                    for (p, v) in o.iter_mut().enumerate() {
                        *v = [a[p], b[p], 0.0];
                    }
                    o
                }
                Kind::Bc6hUf | Kind::Bc6hSf => match decode_bc6h(data, at, kind == Kind::Bc6hSf, modes, t) {
                    Some(o) => o,
                    None => return false,
                },
                Kind::Bc7 => decode_bc7(data, at, t),
            };
            for y in 0..4 {
                let dst = ((by - by0) * 4 + y) * row + bx * 4 * 3;
                for x in 0..4 {
                    out[dst + 3 * x..dst + 3 * x + 3].copy_from_slice(&px[y * 4 + x]);
                }
            }
        }
    }
    true
}

// --- block_means -----------------------------------------------------------------------------

/// `block_means` for `block >= 2` on a flat (r, g, b) pixel buffer with `rx` visible columns
/// and `ry` visible rows of the `blocks_w * block`-wide grid. The caller has checked that the
/// buffer holds every visible pixel. Sums run in raster order inside each block from 0.0, then
/// divide by the visible pixel count (1 for an empty block), as the Python code does.
pub fn block_means(px: &[f64], blocks_w: usize, blocks_h: usize, rx: usize, ry: usize, block: usize,
                   out: &mut [f64]) {
    let width = blocks_w * block;
    let mut sums = vec![[0.0f64; 3]; blocks_w];
    for by in 0..blocks_h {
        let rows_in = block.min(ry.saturating_sub(by * block));
        for s in sums.iter_mut() {
            *s = [0.0; 3];
        }
        for y in 0..rows_in {
            let start = (by * block + y) * width;
            for (bx, s) in sums.iter_mut().enumerate() {
                let x0 = bx * block;
                let x1 = (x0 + block).min(rx);
                for col in x0..x1.max(x0) {
                    let p = &px[3 * (start + col)..3 * (start + col) + 3];
                    s[0] += p[0];
                    s[1] += p[1];
                    s[2] += p[2];
                }
            }
        }
        for (bx, s) in sums.iter().enumerate() {
            let cols = block.min(rx.saturating_sub(bx * block));
            let n = rows_in * cols;
            let n = if n == 0 { 1.0 } else { n as f64 };
            let o = 3 * (by * blocks_w + bx);
            out[o] = s[0] / n;
            out[o + 1] = s[1] / n;
            out[o + 2] = s[2] / n;
        }
    }
}

// --- C entry points --------------------------------------------------------------------------

/// Version of the C interface; `ttpf_mips.py` refuses a library that reports another one.
#[no_mangle]
pub extern "C" fn ttpf_native_abi() -> u32 {
    1
}

/// `detile` / `detile_volume_slice`. `params` is `eb, w, h, bw, bh, tile, tx, ty, row_stride,
/// base` (see [`Detile`]); `bits` holds `nx` X then `ny` Y `(mask, shift)` pairs as
/// 2 * (nx + ny) `u64`. Returns 0, or -1 (the caller runs Python) for a null pointer, a length
/// that does not match, or an element outside `src`.
///
/// # Safety
/// `src` must be valid for `src_len` bytes, `out` for `out_len` writable bytes and `bits` for
/// `2 * (nx + ny)` `u64` values, and `params` for 10.
#[no_mangle]
pub unsafe extern "C" fn ttpf_detile(src: *const u8, src_len: usize, out: *mut u8, out_len: usize,
                                     params: *const u64, bits: *const u64, nx: usize, ny: usize)
                                     -> isize {
    if src.is_null() || out.is_null() || params.is_null() || (bits.is_null() && nx + ny > 0)
        || nx > 64 || ny > 64 {
        return -1;
    }
    // SAFETY: the caller guarantees the lengths (see the safety section); params has 10 values.
    let (src, out, p) = unsafe {
        (slice::from_raw_parts(src, src_len), slice::from_raw_parts_mut(out, out_len),
         slice::from_raw_parts(params, 10))
    };
    let mut pairs = Vec::with_capacity(nx + ny);
    if nx + ny > 0 {
        // SAFETY: as above.
        let b = unsafe { slice::from_raw_parts(bits, 2 * (nx + ny)) };
        for i in 0..nx + ny {
            if b[2 * i + 1] >= 64 {
                return -1;
            }
            pairs.push((b[2 * i], b[2 * i + 1] as u32));
        }
    }
    let d = Detile {
        eb: p[0], w: p[1], h: p[2], bw: p[3], bh: p[4], tile: p[5], tx: p[6], ty: p[7],
        row_stride: p[8], base: p[9], x_bits: &pairs[..nx], y_bits: &pairs[nx..],
    };
    if d.run(src, out) { 0 } else { -1 }
}

/// Blocks decoded per band by [`DecodeArgs::decode_distinct`] (about 6 MiB of doubles).
const BAND_BLOCKS: usize = 1 << 14;

/// `n` zeroed doubles, or None when the allocation fails (a surface of a few hundred megapixels
/// would otherwise abort the process).
fn zeroed(n: usize) -> Option<Vec<f64>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0.0);
    Some(v)
}

/// The decoder inputs shared by [`ttpf_decode_bc`] and [`ttpf_decode_means`].
struct DecodeArgs<'a> {
    kind: Kind,
    data: &'a [u8],
    blocks_w: usize,
    blocks_h: usize,
    bptc: Bptc<'a>,
    modes: Vec<Option<Bc6hMode>>,
}

/// Check and borrow the decoder inputs. `tables` is 64 x 16 `part2`, 64 x 16 `part3`, then
/// `anchor2`, `anchor3a`, `anchor3b` (64 bytes each): 2240 bytes. `bc6h` is the flat mode table
/// described at [`BC6H_STRIDE`] (only read for BC6H).
///
/// # Safety
/// `data` must be valid for `data_len` bytes, `tables` for 2240 and `bc6h` for `bc6h_len` values.
#[allow(clippy::too_many_arguments)]
unsafe fn decode_args<'a>(kind: u32, data: *const u8, data_len: usize, blocks_w: usize, blocks_h: usize,
                          tables: *const u8, bc6h: *const i32, bc6h_len: usize) -> Option<DecodeArgs<'a>> {
    let kind = Kind::from_code(kind)?;
    if data.is_null() || tables.is_null() || blocks_w == 0 || blocks_h == 0 {
        return None;
    }
    let blocks = blocks_w.checked_mul(blocks_h)?;
    let need_in = blocks.checked_mul(kind.block_bytes())?;
    blocks.checked_mul(48)?;
    if data_len < need_in {
        return None;
    }
    // SAFETY: the caller guarantees the lengths (see the safety section).
    let (data, tables) = unsafe { (slice::from_raw_parts(data, data_len), slice::from_raw_parts(tables, 2240)) };
    if tables[..2048].iter().any(|&s| s > 2) {
        return None;
    }
    let modes = if matches!(kind, Kind::Bc6hUf | Kind::Bc6hSf) {
        if bc6h.is_null() {
            return None;
        }
        // SAFETY: as above.
        bc6h_modes(unsafe { slice::from_raw_parts(bc6h, bc6h_len) })?
    } else {
        Vec::new()
    };
    Some(DecodeArgs {
        kind,
        data,
        blocks_w,
        blocks_h,
        bptc: Bptc {
            part2: &tables[0..1024],
            part3: &tables[1024..2048],
            anchor2: &tables[2048..2112],
            anchor3a: &tables[2112..2176],
            anchor3b: &tables[2176..2240],
        },
        modes,
    })
}

impl DecodeArgs<'_> {
    /// The whole surface, or None when a block is not covered or the buffer cannot be allocated
    /// (an allocation failure must reach the caller as -1, not abort the Python process).
    fn decode(&self) -> Option<Vec<f64>> {
        let mut px = zeroed(self.blocks_w.checked_mul(self.blocks_h)?.checked_mul(48)?)?;
        decode_surface(self.kind, self.data, self.blocks_w, self.blocks_h, &self.bptc, &self.modes, &mut px)
            .then_some(px)
    }

    /// The surface as [`Distinct`] tables, decoded in bands of block rows so that only a band of
    /// doubles is held at once (the same tables as `distinct(&self.decode()?)`).
    fn decode_distinct(&self) -> Option<Distinct> {
        self.decode_distinct_in((BAND_BLOCKS / self.blocks_w).max(1))
    }

    /// [`Self::decode_distinct`] with `band` block rows per band.
    fn decode_distinct_in(&self, band: usize) -> Option<Distinct> {
        let band = band.clamp(1, self.blocks_h);
        let mut buf = zeroed(self.blocks_w.checked_mul(band)?.checked_mul(48)?)?;
        let mut b = DistinctBuilder::new(self.blocks_w.checked_mul(self.blocks_h)?.checked_mul(16)?)?;
        let mut by = 0;
        while by < self.blocks_h {
            let end = (by + band).min(self.blocks_h);
            let part = &mut buf[..(end - by) * self.blocks_w * 48];
            if !decode_rows(self.kind, self.data, self.blocks_w, by, end, &self.bptc, &self.modes, part) {
                return None;
            }
            b.push(part)?;
            by = end;
        }
        Some(b.finish())
    }
}

/// A multiply-rotate hasher with a SplitMix64 finish for the bit-pattern keys of [`Distinct`]
/// (std's SipHash is several times slower here; the keys are local decoder output).
#[derive(Default)]
struct FastHash(u64);

impl std::hash::Hasher for FastHash {
    fn finish(&self) -> u64 {
        // Decoded doubles often have all-zero low bits; mix them into every bit.
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
    fn write(&mut self, bytes: &[u8]) {
        // `[u64; 3]` keys arrive here as one 24-byte slice.
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.write_u64(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]));
        }
        for &b in chunks.remainder() {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type FastMap<K> = std::collections::HashMap<K, u32, std::hash::BuildHasherDefault<FastHash>>;

/// The decoded pixels as two tables of distinct bit patterns, so the caller can share one Python
/// float per distinct channel value and one tuple per distinct (r, g, b).
#[derive(Default)]
pub struct Distinct {
    /// Distinct channel values, in order of first appearance.
    pub scalars: Vec<f64>,
    /// Distinct (r, g, b), as indices into `scalars`, in order of first appearance.
    pub triples: Vec<[u32; 3]>,
    /// Index into `triples` of every pixel.
    pub pixels: Vec<u32>,
}

/// Builds [`Distinct`] from pixel buffers pushed in raster order.
struct DistinctBuilder {
    d: Distinct,
    scalars: FastMap<u64>,
    triples: FastMap<[u64; 3]>,
    last: Option<([u64; 3], u32)>,
}

impl DistinctBuilder {
    /// None when the `n` pixel indices cannot be allocated.
    fn new(n: usize) -> Option<Self> {
        let mut d = Distinct::default();
        d.pixels.try_reserve_exact(n).ok()?;
        Some(DistinctBuilder {
            d,
            scalars: FastMap::default(),
            triples: FastMap::with_capacity_and_hasher(n.min(1 << 16), Default::default()),
            last: None,
        })
    }

    /// Add the next pixels (three doubles each). None when a table cannot grow.
    fn push(&mut self, px: &[f64]) -> Option<()> {
        for p in px.chunks_exact(3) {
            let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
            // Neighbouring pixels often repeat; skip the table for them.
            if let Some((k, i)) = self.last {
                if k == key {
                    self.d.pixels.push(i);
                    continue;
                }
            }
            let next = self.d.triples.len() as u32;
            // Reserve first so that growing a table fails softly instead of aborting.
            self.triples.try_reserve(1).ok()?;
            let i = *self.triples.entry(key).or_insert(next);
            if i == next {
                self.d.triples.try_reserve(1).ok()?;
                let mut t = [0u32; 3];
                for (c, k) in key.iter().enumerate() {
                    self.scalars.try_reserve(1).ok()?;
                    self.d.scalars.try_reserve(1).ok()?;
                    let d = &mut self.d;
                    t[c] = *self.scalars.entry(*k).or_insert_with(|| {
                        d.scalars.push(p[c]);
                        (d.scalars.len() - 1) as u32
                    });
                }
                self.d.triples.push(t);
            }
            self.last = Some((key, i));
            self.d.pixels.push(i);
        }
        Some(())
    }

    fn finish(self) -> Distinct {
        self.d
    }
}

pub fn distinct(px: &[f64]) -> Distinct {
    let mut b = DistinctBuilder::new(px.len() / 3).expect("pixel index buffer");
    b.push(px).expect("distinct tables");
    b.finish()
}

/// `decode_pixels` for the BC kinds of [`Kind::from_code`], as [`Distinct`] tables.
///
/// `counts` (two writable values) always receives the number of distinct triples and scalars.
/// When `triples` holds at least that many triples (`3 * triple_cap` values), `scalars` that
/// many doubles (`scalar_cap`), and every triple index fits `width` bytes (1, 2 or 4),
/// `pixels` receives `blocks_w * blocks_h * 16` little-endian indices of `width` bytes and the
/// tables are filled: the result is 0. Otherwise nothing else is written and the result is 1, so
/// a first call with null buffers asks for the sizes. -1 means a bad pointer or length, or a
/// block the port does not cover.
///
/// # Safety
/// `data` must be valid for `data_len` bytes, `tables` for 2240, `bc6h` for `bc6h_len` values,
/// `counts` for 2 writable values and, when non-null, `pixels` for `n * width` bytes, `triples`
/// for `3 * triple_cap` and `scalars` for `scalar_cap` writable values.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ttpf_decode_bc(kind: u32, data: *const u8, data_len: usize, blocks_w: usize,
                                        blocks_h: usize, tables: *const u8, bc6h: *const i32,
                                        bc6h_len: usize, pixels: *mut u8, width: usize, triples: *mut u32,
                                        triple_cap: usize, scalars: *mut f64, scalar_cap: usize,
                                        counts: *mut usize) -> isize {
    if counts.is_null() || !matches!(width, 1 | 2 | 4) {
        return -1;
    }
    // SAFETY: forwarded from the caller.
    let args = match unsafe { decode_args(kind, data, data_len, blocks_w, blocks_h, tables, bc6h, bc6h_len) } {
        Some(a) => a,
        None => return -1,
    };
    let d = match args.decode_distinct() {
        Some(d) => d,
        None => return -1,
    };
    // SAFETY: the caller guarantees the length (see the safety section).
    let counts = unsafe { slice::from_raw_parts_mut(counts, 2) };
    counts[0] = d.triples.len();
    counts[1] = d.scalars.len();
    if pixels.is_null() || triples.is_null() || scalars.is_null() || d.triples.len() > triple_cap
        || d.scalars.len() > scalar_cap || (width < 4 && d.triples.len() > 1 << (8 * width)) {
        return 1;
    }
    let n = d.pixels.len();
    // SAFETY: the caller guarantees the lengths (see the safety section).
    let (pixels, triples, scalars) = unsafe {
        (slice::from_raw_parts_mut(pixels, n * width), slice::from_raw_parts_mut(triples, 3 * triple_cap),
         slice::from_raw_parts_mut(scalars, scalar_cap))
    };
    for (dst, &i) in pixels.chunks_exact_mut(width).zip(d.pixels.iter()) {
        dst.copy_from_slice(&i.to_le_bytes()[..width]);
    }
    for (dst, t) in triples.chunks_exact_mut(3).zip(d.triples.iter()) {
        dst.copy_from_slice(t);
    }
    scalars[..d.scalars.len()].copy_from_slice(&d.scalars);
    0
}

/// `block_means(decode_pixels(...), blocks_w, blocks_h, ...)` for `block >= 2`: decodes the
/// `dec_w` x `dec_h` block grid as [`ttpf_decode_bc`] does, then averages it with the caller's
/// grid, `rx` visible columns and `ry` visible rows. `out` receives `blocks_w * blocks_h * 3`
/// doubles. Returns 0, or -1 for a bad pointer or length, a block the port does not cover, or a
/// visible pixel outside the decoded ones.
///
/// # Safety
/// As [`ttpf_decode_bc`] for the inputs; `out` must be valid for `out_len` writable doubles.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ttpf_decode_means(kind: u32, data: *const u8, data_len: usize, dec_w: usize,
                                           dec_h: usize, tables: *const u8, bc6h: *const i32,
                                           bc6h_len: usize, blocks_w: usize, blocks_h: usize, rx: usize,
                                           ry: usize, block: usize, out: *mut f64, out_len: usize) -> isize {
    if out.is_null() || block < 2 || rx == 0 || ry == 0 {
        return -1;
    }
    let width = match blocks_w.checked_mul(block) {
        Some(w) => w,
        None => return -1,
    };
    let n_px = match dec_w.checked_mul(dec_h).and_then(|n| n.checked_mul(16)) {
        Some(n) => n,
        None => return -1,
    };
    let need_out = blocks_w.checked_mul(blocks_h).and_then(|n| n.checked_mul(3));
    let last = (ry - 1).checked_mul(width).and_then(|n| n.checked_add(rx));
    if need_out != Some(out_len) || rx > width || blocks_h.checked_mul(block).is_none_or(|n| ry > n)
        || last.is_none_or(|n| n > n_px) {
        return -1;
    }
    // SAFETY: forwarded from the caller.
    let args = match unsafe { decode_args(kind, data, data_len, dec_w, dec_h, tables, bc6h, bc6h_len) } {
        Some(a) => a,
        None => return -1,
    };
    let px = match args.decode() {
        Some(p) => p,
        None => return -1,
    };
    // SAFETY: the caller guarantees the length (see the safety section).
    let out = unsafe { slice::from_raw_parts_mut(out, out_len) };
    block_means(&px, blocks_w, blocks_h, rx, ry, block, out);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banded_distinct_equals_whole_surface_distinct() {
        // BC4 blocks with varied endpoints and indices; every band size must give the same tables.
        let (bw, bh) = (5usize, 7usize);
        let mut data = vec![0u8; bw * bh * 8];
        let mut x = 0x1234_5678u32;
        for b in data.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        let tables = vec![0u8; 2240];
        let args = DecodeArgs {
            kind: Kind::Bc4,
            data: &data,
            blocks_w: bw,
            blocks_h: bh,
            bptc: Bptc {
                part2: &tables[0..1024],
                part3: &tables[1024..2048],
                anchor2: &tables[2048..2112],
                anchor3a: &tables[2112..2176],
                anchor3b: &tables[2176..2240],
            },
            modes: Vec::new(),
        };
        let whole = distinct(&args.decode().unwrap());
        for band in [1, 2, 3, 7, 100] {
            let d = args.decode_distinct_in(band).unwrap();
            assert_eq!((d.scalars.len(), d.triples.len()), (whole.scalars.len(), whole.triples.len()), "band {band}");
            assert_eq!(d.pixels, whole.pixels, "band {band}");
            assert_eq!(d.triples, whole.triples, "band {band}");
            let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&d.scalars), bits(&whole.scalars), "band {band}");
        }
    }

    #[test]
    fn zeroed_reports_an_impossible_allocation() {
        assert_eq!(zeroed(3), Some(vec![0.0; 3]));
        assert!(zeroed(usize::MAX / 16).is_none());
    }

    #[test]
    fn f16_matches_known_values() {
        assert_eq!(f16_to_f64(0x3C00), 1.0);
        assert_eq!(f16_to_f64(0xC000), -2.0);
        assert_eq!(f16_to_f64(0x7BFF), 65504.0);
        assert_eq!(f16_to_f64(0x0001), f64::powi(2.0, -24));
        assert_eq!(f16_to_f64(0x8000).to_bits(), (-0.0f64).to_bits());
        assert!(f16_to_f64(0x7E00).is_nan());
        assert_eq!(f16_to_f64(0xFC00), f64::NEG_INFINITY);
    }

    #[test]
    fn bc1_three_colour_mode_has_black() {
        let pal = bc1_palette(0x0000, 0xFFFF, false);
        assert_eq!(pal[3], [0.0, 0.0, 0.0]);
        assert_eq!(pal[2], [127.5, 127.5, 127.5]);
        let pal4 = bc1_palette(0x0000, 0xFFFF, true);
        assert_eq!(pal4[2], [85.0, 85.0, 85.0]);
    }

    #[test]
    fn bc4_palettes() {
        assert_eq!(bc4_palette(255, 0), [255.0, 0.0, 218.0, 182.0, 145.0, 109.0, 72.0, 36.0]);
        assert_eq!(bc4_palette(0, 255), [0.0, 255.0, 51.0, 102.0, 153.0, 204.0, 0.0, 255.0]);
    }

    #[test]
    fn bc7_reserved_mode_is_black() {
        let t = Bptc { part2: &[0; 1024], part3: &[0; 1024], anchor2: &[15; 64], anchor3a: &[3; 64],
                       anchor3b: &[15; 64] };
        let mut data = [0xFFu8; 16];
        data[0] = 0;
        assert_eq!(decode_bc7(&data, 0, &t), [[0.0; 3]; 16]);
    }

    #[test]
    fn block_means_crops_and_averages() {
        // 8x4 pixel grid of two 4x4 blocks; only 5 columns and 3 rows visible.
        let mut px = vec![0.0; 8 * 4 * 3];
        for (i, v) in px.iter_mut().enumerate() {
            *v = (i / 3) as f64;
        }
        let mut out = [0.0; 6];
        block_means(&px, 2, 1, 5, 3, 4, &mut out);
        // Block 0: rows 0-2, cols 0-3 -> mean of (0..4) + (8..12) + (16..20) = 10.
        assert_eq!(out[0], 9.5);
        // Block 1: rows 0-2, col 4 -> 4, 12, 20.
        assert_eq!(out[3], 12.0);
    }

    #[test]
    fn detile_identity_pattern_and_bounds() {
        // A 4x2 grid of 1-byte elements, one tile of 8 bytes laid out linearly.
        let x_bits = [(1u64, 0u32), (2, 1)];
        let y_bits = [(4u64, 0u32)];
        let d = Detile { eb: 1, w: 4, h: 2, bw: 4, bh: 2, tile: 8, tx: 0, ty: 0, row_stride: 8, base: 0,
                         x_bits: &x_bits, y_bits: &y_bits };
        let src: Vec<u8> = (0..8).collect();
        let mut out = [0u8; 8];
        assert!(d.run(&src, &mut out));
        assert_eq!(out, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(!d.run(&src[..7], &mut out));
    }

    #[test]
    fn f16_matches_the_reference_formula() {
        for h in 0..=u16::MAX {
            let exp = ((h >> 10) & 31) as i32;
            let mant = (h & 1023) as f64;
            let mag = match exp {
                0 => mant * f64::powi(2.0, -24),
                31 => continue,
                _ => (1024.0 + mant) * f64::powi(2.0, exp - 25),
            };
            let want = if h & 0x8000 != 0 { -mag } else { mag };
            assert_eq!(f16_to_f64(h).to_bits(), want.to_bits(), "{h:#06x}");
        }
    }

    #[test]
    fn ffi_rejects_bad_lengths() {
        let data = [0u8; 16];
        let tables = [0u8; 2240];
        let nob = std::ptr::null();
        let mut pixels = [9u8; 16];
        let mut triples = [0u32; 3];
        let mut scalars = [1.0f64; 1];
        let mut counts = [0usize; 2];
        let mut means = [0.0f64; 3];
        // SAFETY: every pointer is valid for the lengths passed.
        unsafe {
            let mut call = |kind, len, width, cap| {
                ttpf_decode_bc(kind, data.as_ptr(), len, 1, 1, tables.as_ptr(), nob, 0, pixels.as_mut_ptr(), width,
                               triples.as_mut_ptr(), cap, scalars.as_mut_ptr(), cap, counts.as_mut_ptr())
            };
            // One all-black BC1 block: one distinct pixel and one distinct channel value.
            assert_eq!(call(0, 8, 1, 0), 1);
            assert_eq!(call(0, 8, 1, 1), 0);
            assert_eq!(call(0, 7, 1, 1), -1);
            assert_eq!(call(0, 8, 3, 1), -1);
            // BC6H without its mode table, and an unknown kind.
            assert_eq!(call(4, 16, 1, 1), -1);
            assert_eq!(call(9, 16, 1, 1), -1);
            assert_eq!(counts, [1, 1]);
            assert_eq!(scalars[0], 0.0);
            assert_eq!(pixels, [0u8; 16]);
            assert_eq!(ttpf_decode_means(0, data.as_ptr(), 8, 1, 1, tables.as_ptr(), nob, 0, 1, 1, 4, 4, 4,
                                         means.as_mut_ptr(), 3), 0);
            assert_eq!(means, [0.0; 3]);
            // A visible row past the decoded grid.
            assert_eq!(ttpf_decode_means(0, data.as_ptr(), 8, 1, 1, tables.as_ptr(), nob, 0, 1, 2, 4, 5, 4,
                                         means.as_mut_ptr(), 3), -1);
        }
    }

    #[test]
    fn distinct_keeps_signed_zero_apart() {
        let d = distinct(&[0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.0]);
        assert_eq!(d.pixels, [0, 1, 0, 2]);
        assert_eq!(d.triples, [[0, 0, 0], [1, 0, 0], [2, 0, 1]]);
        assert_eq!(d.scalars.len(), 3);
    }
}
