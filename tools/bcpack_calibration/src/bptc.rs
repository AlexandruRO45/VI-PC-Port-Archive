// Strict BC6H and BC7 block rebuild, ported from `_bc6h_blocks`, `_bc7_check`
// and `_bc7_rebuild` in tools/bcpack.py. Python builds the tables below from its
// own (`_bc6h_native_lut`, `_bc6h_native_history`, `_bc7_native_lut`,
// `_bc7_native_history`) and passes them in, so the plans, partitions and
// signature table are never written down twice. Anything the strict path would
// not return as it is (an unknown or inferred signature, a history entry that
// is not confirmed to the bits a block reads, an anchor index or a p-bit that
// does not agree, a block that does not end on bit 128) returns None: Python
// then decodes the chunk itself, so its errors and provisional entries stay
// unchanged. Refusing is always safe; only accepted output must be identical.

fn unzig(v: i64) -> i64 {
    if v & 1 != 0 {
        -((v + 1) >> 1)
    } else {
        v >> 1
    }
}

/// `_inter(a, b, n)`: bit k of `a` goes to bit 2k, bit k of `b` to bit 2k + 1.
fn inter(a: u16, b: u16, n: u32) -> i64 {
    let mut d = 0i64;
    for k in 0..n {
        d |= ((((a >> k) & 1) as i64) << (2 * k)) | ((((b >> k) & 1) as i64) << (2 * k + 1));
    }
    d
}

/// `_DEINT[odd][s]`: the four even (odd = 0) or odd (odd = 1) bits of an 8-bit symbol.
fn deint(s: u16, odd: u16) -> i64 {
    (0..4).map(|k| (((s >> (2 * k + odd)) & 1) as i64) << k).sum()
}

/// The four index deltas of one row: two pixels from `sa`, two from `sb`.
fn row_deltas(sa: u16, sb: u16, y: usize) -> Option<[i64; 4]> {
    if sa > 255 || sb > 255 {
        return None;
    }
    Some(if y < 3 {
        [deint(sa, 1), deint(sa, 0), deint(sb, 1), deint(sb, 0)]
    } else {
        [deint(sa, 0), deint(sa, 1), deint(sb, 0), deint(sb, 1)]
    })
}

/// ORs `value & (2^bits - 1)` in at `*at`; None past bit 128 (Python would overflow `to_bytes(16)`).
fn put(acc: &mut u128, at: &mut u32, value: i64, bits: u32) -> Option<()> {
    if *at + bits > 128 {
        return None;
    }
    if bits > 0 {
        *acc |= ((value as u128) & ((1u128 << bits) - 1)) << *at;
    }
    *at += bits;
    Some(())
}

// BC6H -----------------------------------------------------------------------

/// `lut`: `BC7_ANCHOR2[:32]`, then for each mode 1..=14 `BC6H_MODE` bytes:
/// code, transformed, base bits, three delta bits, regions, plan length,
/// header bits, three zero bytes, then 128 plan items of four bytes
/// (source: 0 partition / 1 + channel, endpoint slot, bit index, stream bit).
pub const BC6H_MODE: usize = 12 + 128 * 4;
pub const BC6H_LUT: usize = 32 + 14 * BC6H_MODE;
/// `builtin`: endpoint history [256][3][4] (low 16, 10, 6, 6 bits), index
/// history [256][16] (low 4 bits), endpoint and index class-3 flags [256] each,
/// endpoint confirmed bits [256][3][4] and index confirmed bits [256][16].
pub const BC6H_BUILTIN: usize = 256 * 12 + 256 * 16 + 256 + 256 + 256 * 12 + 256 * 16;

const BC6H_NONE_E: [[i64; 4]; 3] = [[0x8000, 0x200, 0x20, 0x20]; 3];
const BC6H_NONE_I: [i64; 16] = [8; 16];

/// `_bc6h_stream_w`: BCPack's base endpoint in D3D11 bit order.
fn stream_w(mode: usize, w: i64) -> i64 {
    match mode {
        13 => (w & 0x3FF) | (((w >> 10) & 1) << 11) | (((w >> 11) & 1) << 10),
        14 => {
            let mut out = w & 0x3FF;
            for b in 10..16 {
                out |= ((w >> b) & 1) << (25 - b);
            }
            out
        }
        _ => w,
    }
}

/// `sym` is lane-major, `n` symbols per lane: two per block.
pub fn bc6h(sym: &[u16], n: usize, lut: &[u8], builtin: &[i32], out: &mut [u8]) -> Option<()> {
    if n % 2 != 0 || sym.len() != 16 * n || lut.len() != BC6H_LUT || builtin.len() != BC6H_BUILTIN || out.len() != n * 8 {
        return None;
    }
    let lane = |l: usize| &sym[l * n..(l + 1) * n];
    let anchor2 = &lut[..32];
    let (eh, rest) = builtin.split_at(256 * 12);
    let (ih, rest) = rest.split_at(256 * 16);
    let (eok, rest) = rest.split_at(256);
    let (iok, rest) = rest.split_at(256);
    let (ep_cb, ix_cb) = rest.split_at(256 * 12);
    let blocks = n / 2;
    let mut st_e = vec![[[0i64; 4]; 3]; blocks];
    let mut st_i = vec![[0i64; 16]; blocks];
    let (l0, l1, l15) = (lane(0), lane(1), lane(15));
    for i in 0..blocks {
        let (a, b) = (2 * i, 2 * i + 1);
        let m_sym = l15[a] as usize;
        if m_sym > 13 {
            return None;
        }
        let mode = m_sym + 1;
        let info = &lut[32 + m_sym * BC6H_MODE..][..BC6H_MODE];
        let (code, trans, base) = (info[0] as u128, info[1], info[2] as u32);
        let deltas = [info[3] as u32, info[4] as u32, info[5] as u32];
        let (reg, items, header) = (info[6], info[7] as usize, info[8] as u32);
        if !(1..=16).contains(&base) || deltas.iter().any(|d| !(1..=10).contains(d)) || !(1..=2).contains(&reg) || items > 128 {
            return None;
        }
        let part = (l15[b] as usize).min(31);

        let di = inter(l1[a], l1[b], 4) as usize;
        let de = ((di as i64 + unzig(inter(l0[a], l0[b], 4))) & 255) as usize;
        let eref = if de == 0 {
            BC6H_NONE_E
        } else if i >= de {
            st_e[i - de]
        } else {
            // Strict mode needs a class-3 entry confirmed to every bit this block reads.
            let entry = 256 + i - de;
            if eok[entry] != 1 {
                return None;
            }
            let mut r = [[0i64; 4]; 3];
            for c in 0..3 {
                let dc = deltas[c] as i32;
                let need = if reg == 2 { [base as i32, dc, dc, dc] } else { [base as i32, dc, 0, 0] };
                for q in 0..4 {
                    if ep_cb[entry * 12 + c * 4 + q] < need[q] {
                        return None;
                    }
                    r[c][q] = eh[entry * 12 + c * 4 + q] as i64;
                }
            }
            r
        };
        let ibits: u32 = if reg == 2 { 3 } else { 4 };
        let iref = if di == 0 {
            BC6H_NONE_I
        } else if i >= di {
            st_i[i - di]
        } else {
            let entry = 256 + i - di;
            if iok[entry] != 1 || ix_cb[entry * 16..entry * 16 + 16].iter().any(|&b| b < ibits as i32) {
                return None;
            }
            let mut r = [0i64; 16];
            for p in 0..16 {
                r[p] = ih[entry * 16 + p] as i64;
            }
            r
        };

        let mw = (1i64 << base) - 1;
        let w_shift = 16 - base;
        let mut f = [[0i64; 4]; 3];
        for c in 0..3 {
            let dc = deltas[c];
            let md = (1i64 << dc) - 1;
            let sh = if dc < 6 { 6 - dc } else { 0 };
            let dz0 = unzig(inter(lane(2 + c)[a], lane(2 + c)[b], 8));
            let dz1 = unzig(inter(lane(5 + c)[a], lane(5 + c)[b], 5));
            let dz2 = unzig(lane(8 + c)[a] as i64);
            let dz3 = unzig(lane(8 + c)[b] as i64);
            let [r0, r1, r2, r3] = eref[c];
            let w = ((r0 >> w_shift) + dz0) & mw;
            let x = ((r1 >> (10 - dc)) + dz1) & md;
            let y = ((r2 >> sh) + dz2) & md;
            let z = ((r3 >> sh) + dz3) & md;
            let x10 = x << (10 - dc);
            if reg == 2 {
                st_e[i][c] = [w << w_shift, x10, y << sh, z << sh];
                f[c] = [stream_w(mode, w), x, y, z];
            } else {
                let v = (w & md) ^ if trans == 1 { 1 << (dc - 1) } else { 0 };
                let low_y = if dc >= 6 { v >> (dc - 6) } else { v << sh };
                st_e[i][c] = [w << w_shift, x10, low_y, x10 >> 4];
                f[c] = [stream_w(mode, w), x, 0, 0];
            }
        }

        let imask = (1i64 << ibits) - 1;
        let ishift = 4 - ibits;
        let mut ix = [0i64; 16];
        for y in 0..4 {
            let d = row_deltas(lane(11 + y)[a], lane(11 + y)[b], y)?;
            for off in 0..4 {
                let p = y * 4 + off;
                ix[p] = ((iref[p] >> ishift) + unzig(d[off])) & imask;
                st_i[i][p] = ix[p] << ishift;
            }
        }

        // An anchor index with its top bit set is an error in strict mode.
        let second = if reg == 2 { anchor2[part] as usize } else { 16 };
        if ix[0] & (1 << (ibits - 1)) != 0 || (reg == 2 && (second >= 16 || ix[second] & 4 != 0)) {
            return None;
        }

        let mut acc = code;
        for item in info[12..12 + 4 * items].chunks_exact(4) {
            let (src, q, bit, pos) = (item[0] as usize, item[1] as usize, item[2] as u32, item[3] as u32);
            if pos >= 128 || bit >= 16 || q >= 4 || src > 3 {
                return None;
            }
            let value = if src == 0 { part as i64 } else { f[src - 1][q] };
            if (value >> bit) & 1 != 0 {
                acc |= 1u128 << pos;
            }
        }
        let mut at = header;
        for p in 0..16 {
            let nb = if p == 0 || p == second { ibits - 1 } else { ibits };
            put(&mut acc, &mut at, ix[p], nb)?;
        }
        out[16 * i..16 * i + 16].copy_from_slice(&acc.to_le_bytes());
    }
    Some(())
}

// BC7 ------------------------------------------------------------------------

/// `lut`: 32 x 32 signature cells of (mode, partition, rotation, index
/// selection), mode 255 where strict mode refuses the pair (outside the table
/// or inferred); `partition2` and `partition3` (64 x 16 each); `anchor2`,
/// `anchor3a` and `anchor3b` (64 each); `_B7_NS`, `_B7_CB`, `_B7_AB`,
/// `_B7_PK`, `_B7_IB` and `_B7_PB` (8 each); `_B7_LANE` (3 x 4, 255 for -1).
pub const BC7_LUT: usize = 1024 * 4 + 64 * 16 * 2 + 64 * 3 + 8 * 6 + 12;
/// `builtin`: history endpoints [256][24] (low 8 bits), history indices
/// [256][32] (low 4 bits), then the depth Python checks against for each slot,
/// min(known bits, confirmed bits) clamped to 0..=9, for endpoints and indices.
pub const BC7_BUILTIN: usize = 256 * 24 + 256 * 32 + 256 * 24 + 256 * 32;

pub fn bc7(sym: &[u16], n: usize, lut: &[u8], builtin: &[i32], out: &mut [u8]) -> Option<()> {
    if n % 2 != 0 || sym.len() != 16 * n || lut.len() != BC7_LUT || builtin.len() != BC7_BUILTIN || out.len() != n * 8 {
        return None;
    }
    let lane = |l: usize| &sym[l * n..(l + 1) * n];
    let (cells, rest) = lut.split_at(1024 * 4);
    let (p2, rest) = rest.split_at(64 * 16);
    let (p3, rest) = rest.split_at(64 * 16);
    let (a2, rest) = rest.split_at(64);
    let (a3a, rest) = rest.split_at(64);
    let (a3b, rest) = rest.split_at(64);
    let (consts, lane_map) = rest.split_at(48);
    let (ns_t, cb_t, ab_t) = (&consts[0..8], &consts[8..16], &consts[16..24]);
    let (pk_t, ib_t, pb_t) = (&consts[24..32], &consts[32..40], &consts[40..48]);
    let (pre_e, rest) = builtin.split_at(256 * 24);
    let (pre_i, rest) = rest.split_at(256 * 32);
    let (depth_e, depth_i) = rest.split_at(256 * 24);
    let blocks = n / 2;
    let (l0, l1, l15) = (lane(0), lane(1), lane(15));

    // `_bc7_check`: every signature is in the table and confirmed.
    let mut cfg = vec![(0usize, 0usize, 0usize, 0i64); blocks];
    for i in 0..blocks {
        let (sa, sb) = (l15[2 * i] as usize, l15[2 * i + 1] as usize);
        if sa >= 32 || sb >= 32 {
            return None;
        }
        let cell = &cells[(sa * 32 + sb) * 4..][..4];
        let (mode, part, rot) = (cell[0] as usize, cell[1] as usize, cell[2] as usize);
        if mode > 7 || part >= 64 || rot > 3 {
            return None;
        }
        let (ns, ib, pk) = (ns_t[mode] as usize, ib_t[mode] as u32, pk_t[mode]);
        if !(1..=3).contains(&ns) || !(1..=4).contains(&ib) || pk > 2 || cb_t[mode] > 8 || ab_t[mode] > 8 || pb_t[mode] > 6 {
            return None;
        }
        cfg[i] = (mode, part, rot, cell[3] as i64);
    }
    let field_bits = |mode: usize, rot: usize| -> [u32; 4] {
        let (cb, ab) = (cb_t[mode] as u32, ab_t[mode] as u32);
        let mut w = [cb, cb, cb, ab];
        if rot > 0 {
            w[rot - 1] = ab;
            w[3] = cb;
        }
        let pk = pk_t[mode] != 0;
        let mut bits = [0u32; 4];
        for c in 0..4 {
            bits[c] = if w[c] > 0 { w[c] + (pk && c == 0) as u32 } else { 0 };
        }
        bits
    };
    // Every built-in history slot a block reads is known and confirmed to the bits it reads.
    for i in 0..blocks.min(256) {
        let (mode, _, rot, _) = cfg[i];
        let di = inter(l1[2 * i], l1[2 * i + 1], 4) as usize;
        let de = ((di as i64 + unzig(inter(l0[2 * i], l0[2 * i + 1], 4))) & 255) as usize;
        if de > i {
            let entry = 256 + i - de;
            let bits = field_bits(mode, rot);
            for sub in 0..ns_t[mode] as usize {
                for e in 0..2 {
                    for c in 0..4 {
                        if bits[c] > 0 && depth_e[entry * 24 + sub * 8 + e * 4 + c] < bits[c] as i32 {
                            return None;
                        }
                    }
                }
            }
        }
        if di > i {
            let entry = 256 + i - di;
            let row = &depth_i[entry * 32..entry * 32 + 32];
            if row[..16].iter().any(|&d| d < ib_t[mode] as i32) || (matches!(mode, 4 | 5) && row[16..].iter().any(|&d| d < 2)) {
                return None;
            }
        }
    }

    // `_bc7_rebuild`, with every non-zero status a refusal.
    let mut eps = vec![[0i64; 24]; blocks];
    let mut ixs = vec![[0i64; 32]; blocks];
    for i in 0..blocks {
        let (a, b) = (2 * i, 2 * i + 1);
        let (mode, part, rot, isel) = cfg[i];
        let di = inter(l1[a], l1[b], 4) as usize;
        let de = ((di as i64 + unzig(inter(l0[a], l0[b], 4))) & 255) as usize;
        let rep: [i64; 24] = if de == 0 {
            [0; 24]
        } else if i >= de {
            eps[i - de]
        } else {
            let e = 256 + i - de;
            core::array::from_fn(|k| pre_e[e * 24 + k] as i64)
        };
        let rix: [i64; 32] = if di == 0 {
            [0; 32]
        } else if i >= di {
            ixs[i - di]
        } else {
            let e = 256 + i - di;
            core::array::from_fn(|k| pre_i[e * 32 + k] as i64)
        };

        let ns = ns_t[mode] as usize;
        let (cb, ab, pk) = (cb_t[mode] as u32, ab_t[mode] as u32, pk_t[mode]);
        let mut w = [cb, cb, cb, ab];
        if rot > 0 {
            w[rot - 1] = ab;
            w[3] = cb;
        }
        let mut val = [[[0i64; 4]; 2]; 3];
        let mut sep = [0i64; 24];
        let mut pb = [[0i64; 2]; 3];
        for sub in 0..ns {
            for c in 0..4 {
                let lane_idx = lane_map[sub * 4 + c] as usize;
                let wc = w[c];
                if wc == 0 || lane_idx == 255 {
                    continue;
                }
                let bits = wc + (pk != 0 && c == 0) as u32;
                if bits > 8 || lane_idx >= 16 {
                    return None;
                }
                let shift = 8 - bits;
                let mask = (1i64 << bits) - 1;
                let ln = lane(lane_idx);
                let v0 = ((rep[sub * 8 + c] >> shift) + unzig(ln[a] as i64)) & mask;
                let v1 = ((rep[sub * 8 + 4 + c] >> shift) + unzig(ln[b] as i64)) & mask;
                val[sub][0][c] = v0;
                val[sub][1][c] = v1;
                sep[sub * 8 + c] = v0 << shift;
                sep[sub * 8 + 4 + c] = v1 << shift;
            }
            if pk == 1 {
                pb[sub] = [val[sub][0][0] & 1, val[sub][1][0] & 1];
            } else if pk == 2 {
                let p0 = val[sub][0][0] & 1;
                pb[sub] = [p0, p0];
                if val[sub][1][0] & 1 != p0 {
                    return None; // shared p-bit mismatch (_B7_PBIT)
                }
            }
        }
        for sub in ns..3 {
            sep.copy_within(0..8, sub * 8);
        }
        eps[i] = sep;

        let b1 = ib_t[mode] as u32;
        let b1_shift = 4 - b1;
        let b1_mask = (1i64 << b1) - 1;
        let two = matches!(mode, 4 | 5);
        let mut s1 = [0i64; 16];
        let mut s2 = [0i64; 16];
        let mut six = [0i64; 32];
        for y in 0..4 {
            let d = row_deltas(lane(11 + y)[a], lane(11 + y)[b], y)?;
            for off in 0..4 {
                let p = y * 4 + off;
                s1[p] = ((rix[p] >> b1_shift) + unzig(d[off])) & b1_mask;
                six[p] = s1[p] << b1_shift;
            }
            if two {
                let r2 = [6, 7, 8, 10][y];
                let d = row_deltas(lane(r2)[a], lane(r2)[b], y)?;
                for off in 0..4 {
                    let p = y * 4 + off;
                    s2[p] = ((rix[16 + p] >> 2) + unzig(d[off])) & 3;
                    six[16 + p] = s2[p] << 2;
                }
            }
        }
        ixs[i] = six;

        let o = &mut out[16 * i..16 * i + 16];
        if mode == 0 && part == 0 && sep.iter().all(|&v| v == 0) && s1.iter().all(|&v| v == 0) {
            o.fill(0);
            continue;
        }

        let subset = |p: usize| -> usize {
            match ns {
                2 => p2[part * 16 + p] as usize,
                3 => p3[part * 16 + p] as usize,
                _ => 0,
            }
        };
        let anchors = [0usize, if ns == 2 { a2[part] as usize } else { a3a[part] as usize }, a3b[part] as usize];
        let (pri, ib, sec, ib2) = match mode {
            4 => (&s2, 2u32, Some(&s1), 3u32),
            5 => (&s1, 2, Some(&s2), 2),
            _ => (&s1, b1, None, 0),
        };

        let mut acc = 1u128 << mode;
        let mut pos = mode as u32 + 1;
        put(&mut acc, &mut pos, part as i64, pb_t[mode] as u32)?;
        if two {
            put(&mut acc, &mut pos, rot as i64, 2)?;
        }
        if mode == 4 {
            put(&mut acc, &mut pos, isel, 1)?;
        }
        for c in 0..3 {
            let tc = if rot > 0 && c == rot - 1 { 3 } else { c };
            let drop = pk != 0 && tc == 0;
            for sub in 0..ns {
                for e in 0..2 {
                    let v = val[sub][e][tc];
                    put(&mut acc, &mut pos, if drop { v >> 1 } else { v }, cb)?;
                }
            }
        }
        if ab > 0 {
            let tc = if rot > 0 { rot - 1 } else { 3 };
            let drop = pk != 0 && tc == 0;
            for sub in 0..ns {
                for e in 0..2 {
                    let v = val[sub][e][tc];
                    put(&mut acc, &mut pos, if drop { v >> 1 } else { v }, ab)?;
                }
            }
        }
        if pk == 1 {
            for sub in 0..ns {
                put(&mut acc, &mut pos, pb[sub][0], 1)?;
                put(&mut acc, &mut pos, pb[sub][1], 1)?;
            }
        } else if pk == 2 {
            for sub in 0..ns {
                put(&mut acc, &mut pos, pb[sub][0], 1)?;
            }
        }
        for p in 0..16 {
            let s = subset(p);
            if s >= ns {
                return None;
            }
            let pv = pri[p];
            if p == anchors[s] {
                if pv >> (ib - 1) != 0 {
                    return None; // anchor index with its top bit set (_B7_ANCHOR)
                }
                put(&mut acc, &mut pos, pv, ib - 1)?;
            } else {
                put(&mut acc, &mut pos, pv, ib)?;
            }
        }
        if let Some(sec) = sec {
            for p in 0..16 {
                let sv = sec[p];
                if p == 0 {
                    if sv >> (ib2 - 1) != 0 {
                        return None;
                    }
                    put(&mut acc, &mut pos, sv, ib2 - 1)?;
                } else {
                    put(&mut acc, &mut pos, sv, ib2)?;
                }
            }
        }
        if pos != 128 {
            return None; // _B7_SIGNATURE
        }
        o.copy_from_slice(&acc.to_le_bytes());
    }
    Some(())
}
