//! Optional native kernels for `tools/rsc8_audio.py`: the AWC (`ADAT`) bank header parse and the word loops of
//! the `0x5a`, `0xd9` and `0xbd` companion-table decoders.
//!
//! The Python functions stay the specification; every kernel computes exactly what its reference computes and
//! returns one of the status codes below. `tools/_audio_native.py` is the only caller.

use std::collections::HashMap;
use std::slice;

const ABI: u32 = 1;

/// The kernel filled its outputs.
const DONE: i32 = 0;
/// The reference would return its "malformed" result (`None` for a header, `size_ok = False` for a table).
const INVALID: i32 = 1;
/// The kernel cannot answer (an output buffer is too small or an argument is out of range); the caller falls
/// back to the reference.
const DECLINE: i32 = 2;

const ADAT_MAGIC: &[u8; 4] = b"ADAT";

#[no_mangle]
pub extern "C" fn audio_native_abi() -> u32 {
    ABI
}

/// # Safety
/// `ptr` is valid for `len` bytes, or `len` is zero.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        slice::from_raw_parts(ptr, len)
    }
}

/// # Safety
/// `ptr` is valid for `len` elements, or `len` is zero.
unsafe fn out_slice<'a, T>(ptr: *mut T, len: usize) -> Option<&'a mut [T]> {
    if len == 0 {
        Some(&mut [])
    } else if ptr.is_null() {
        None
    } else {
        Some(slice::from_raw_parts_mut(ptr, len))
    }
}

#[inline]
fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

#[inline]
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

#[inline]
fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(w)
}

// ---------------------------------------------------------------------------------------------------------
// parse_adat_header_bytes
// ---------------------------------------------------------------------------------------------------------

/// Result of a header parse into caller-owned buffers.
struct Header {
    flags: u32,
    count: u32,
    hdr_sz: u32,
    end: u64,
    tags: usize,
}

/// `streams` gets `(stream_id, tag_count)` per stream, `tags` gets `(type, offset, size)` per tag, in header order.
fn adat_header(
    hb: &[u8],
    bank_offset: u64,
    file_size: u64,
    max_streams: u64,
    max_header: u64,
    streams: &mut [u32],
    tags: &mut [u32],
) -> Result<Header, i32> {
    if hb.len() < 16 || &hb[..4] != ADAT_MAGIC {
        return Err(INVALID);
    }
    let flags = u32_at(hb, 4);
    let count = u32_at(hb, 8);
    let hdr_sz = u32_at(hb, 12);
    let (cnt, hsz) = (u64::from(count), u64::from(hdr_sz));
    if cnt > max_streams
        || hsz < 16
        || hsz > max_header
        || hsz > hb.len() as u64
        || u128::from(bank_offset) + u128::from(hsz) > u128::from(file_size)
    {
        return Err(INVALID);
    }
    let mut pos: u64 = 16;
    if flags & 0x0040_0000 != 0 {
        pos += 4;
    }
    if flags & 0x0001_0000 != 0 {
        pos += cnt * 2;
    }
    if flags & 0x0020_0000 != 0 {
        pos += (cnt + 1) * 4;
    }
    if count == 0 {
        if pos != hsz {
            return Err(INVALID);
        }
    } else if pos + cnt * 4 > hsz {
        return Err(INVALID);
    }
    // From here every read is below hdr_sz <= hb.len().
    let n = count as usize;
    if streams.len() < 2 * n {
        return Err(DECLINE);
    }
    let mut at = pos as usize;
    let mut total: u64 = 0;
    for i in 0..n {
        let w = u32_at(hb, at);
        at += 4;
        let tc = w >> 29;
        streams[2 * i] = w & 0x1FFF_FFFF;
        streams[2 * i + 1] = tc;
        total += u64::from(tc);
    }
    if at as u64 + total * 8 > hsz {
        return Err(INVALID);
    }
    let total = total as usize;
    if tags.len() < 3 * total {
        return Err(DECLINE);
    }
    let mut max_end = hsz;
    for t in 0..total {
        let q = u64_at(hb, at);
        at += 8;
        let tt = (q >> 56) & 0xFF;
        let tsz = (q >> 28) & 0x0FFF_FFFF;
        let toff = q & 0x0FFF_FFFF;
        if u128::from(bank_offset) + u128::from(toff) + u128::from(tsz) > u128::from(file_size) {
            return Err(INVALID);
        }
        if toff + tsz > max_end {
            max_end = toff + tsz;
        }
        tags[3 * t] = tt as u32;
        tags[3 * t + 1] = toff as u32;
        tags[3 * t + 2] = tsz as u32;
    }
    let end = bank_offset.checked_add(max_end).ok_or(DECLINE)?;
    Ok(Header { flags, count, hdr_sz, end, tags: total })
}

/// `head` gets `flags, count, header_size, end, tag_count`.
///
/// # Safety
/// Every pointer is valid for its length (`streams_cap`/`tags_cap` u32 words, five u64 words for `head`).
#[no_mangle]
pub unsafe extern "C" fn audio_adat_header_v1(
    hb: *const u8,
    hb_len: usize,
    bank_offset: u64,
    file_size: u64,
    max_streams: u64,
    max_header: u64,
    streams: *mut u32,
    streams_cap: usize,
    tags: *mut u32,
    tags_cap: usize,
    head: *mut u64,
) -> i32 {
    let (Some(st), Some(tg), Some(hd)) =
        (out_slice(streams, streams_cap), out_slice(tags, tags_cap), out_slice(head, 5))
    else {
        return DECLINE;
    };
    match adat_header(bytes(hb, hb_len), bank_offset, file_size, max_streams, max_header, st, tg) {
        Ok(h) => {
            hd.copy_from_slice(&[
                u64::from(h.flags),
                u64::from(h.count),
                u64::from(h.hdr_sz),
                h.end,
                h.tags as u64,
            ]);
            DONE
        }
        Err(code) => code,
    }
}

// ---------------------------------------------------------------------------------------------------------
// decode_grid20_table
// ---------------------------------------------------------------------------------------------------------

/// The entry loop: `(first, last, increasing, p_multiple_of_32)`, `first`/`last` `None` without entries.
fn grid20(raw: &[u8], n: usize) -> (Option<u32>, Option<u32>, bool, bool) {
    let mut first = None;
    let mut prev: Option<u32> = None;
    let mut p_ok = true;
    let mut increasing = true;
    for i in 0..n {
        let at = 4 + 20 * i;
        let position = u32_at(raw, at);
        if u16_at(raw, at + 8) % 32 != 0 {
            p_ok = false;
        }
        match prev {
            None => first = Some(position),
            Some(p) if position <= p => increasing = false,
            Some(_) => {}
        }
        prev = Some(position);
    }
    (first, prev, increasing, p_ok)
}

/// `out` gets `has_entries, position_first, position_last, position_increasing, p_multiple_of_32`.
///
/// # Safety
/// `raw` is valid for `len` bytes and `out` for five u64 words.
#[no_mangle]
pub unsafe extern "C" fn audio_grid20_v1(raw: *const u8, len: usize, out: *mut u64) -> i32 {
    let Some(out) = out_slice(out, 5) else {
        return DECLINE;
    };
    let raw = bytes(raw, len);
    if raw.len() < 8 || (raw.len() - 8) % 20 != 0 {
        return INVALID;
    }
    let (first, last, increasing, p_ok) = grid20(raw, (raw.len() - 8) / 20);
    out.copy_from_slice(&[
        u64::from(first.is_some()),
        u64::from(first.unwrap_or(0)),
        u64::from(last.unwrap_or(0)),
        u64::from(increasing),
        u64::from(p_ok),
    ]);
    DONE
}

// ---------------------------------------------------------------------------------------------------------
// decode_index_runs
// ---------------------------------------------------------------------------------------------------------

struct Runs {
    groups: u64,
    declared: u32,
    marker_ok: bool,
    hashes: Vec<u32>,
    values: u64,
    ascending: bool,
    index_max: Option<u32>,
}

fn index_runs(raw: &[u8], max_groups: u64) -> Option<Runs> {
    let nw = raw.len() / 4;
    let val = |i: usize| u32_at(raw, 4 * i);
    let declared = val(0);
    let mut at = 1usize;
    let mut groups = 0u64;
    let mut hashes = Vec::new();
    let mut values = 0u64;
    let mut index_max: Option<u32> = None;
    let mut ascending = true;
    let mut marker_ok = true;
    while at < nw {
        if at + 3 > nw || groups >= max_groups {
            return None;
        }
        let (marker, length, h) = (val(at), val(at + 1) as usize, val(at + 2));
        if marker != 2 {
            marker_ok = false;
        }
        if length > nw - at - 3 {
            return None;
        }
        hashes.push(h);
        let start = at + 3;
        if length > 0 {
            let mut prev = val(start);
            let mut run_max = prev;
            for i in start + 1..start + length {
                let v = val(i);
                if prev >= v {
                    ascending = false;
                }
                if v > run_max {
                    run_max = v;
                }
                prev = v;
            }
            index_max = Some(index_max.map_or(run_max, |m| m.max(run_max)));
        }
        values += length as u64;
        at = start + length;
        groups += 1;
    }
    hashes.sort_unstable();
    hashes.dedup();
    Some(Runs { groups, declared, marker_ok, hashes, values, ascending, index_max })
}

/// `out` gets `groups, group_count_matches, marker_ok, hash_count, values, indices_ascending, has_max,
/// index_max`; `hashes` the sorted distinct hashes.
///
/// # Safety
/// `raw` is valid for `len` bytes, `hashes` for `hashes_cap` u32 words and `out` for eight u64 words.
#[no_mangle]
pub unsafe extern "C" fn audio_index_runs_v1(
    raw: *const u8,
    len: usize,
    max_groups: u64,
    hashes: *mut u32,
    hashes_cap: usize,
    out: *mut u64,
) -> i32 {
    let (Some(hs), Some(out)) = (out_slice(hashes, hashes_cap), out_slice(out, 8)) else {
        return DECLINE;
    };
    let raw = bytes(raw, len);
    if raw.len() < 4 || raw.len() % 4 != 0 {
        return INVALID;
    }
    let Some(r) = index_runs(raw, max_groups) else {
        return INVALID;
    };
    if r.hashes.len() > hs.len() {
        return DECLINE;
    }
    hs[..r.hashes.len()].copy_from_slice(&r.hashes);
    out.copy_from_slice(&[
        r.groups,
        u64::from(r.groups == u64::from(r.declared)),
        u64::from(r.marker_ok),
        r.hashes.len() as u64,
        r.values,
        u64::from(r.ascending),
        u64::from(r.index_max.is_some()),
        u64::from(r.index_max.unwrap_or(0)),
    ]);
    DONE
}

// ---------------------------------------------------------------------------------------------------------
// decode_marker_records
// ---------------------------------------------------------------------------------------------------------

/// `(names in first-seen order, their counts, offsets_ascending, offset_max, extra_max)` of `n` records.
fn markers(raw: &[u8], n: usize) -> (Vec<u32>, Vec<u64>, bool, u32, u32) {
    let mut index: HashMap<u32, usize> = HashMap::new();
    let mut names = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    let mut ascending = true;
    let mut prev: Option<u32> = None;
    let mut offset_max = 0u32;
    let mut extra_max = 0u32;
    let mut last_name: Option<(u32, usize)> = None;
    for i in 0..n {
        let at = 16 * i;
        let name = u32_at(raw, at);
        let offset = u32_at(raw, at + 8);
        let extra = u32_at(raw, at + 12);
        // Runs of one name are common, so the last slot is checked before the map.
        let slot = match last_name {
            Some((nm, s)) if nm == name => s,
            _ => *index.entry(name).or_insert_with(|| {
                names.push(name);
                counts.push(0);
                names.len() - 1
            }),
        };
        counts[slot] += 1;
        last_name = Some((name, slot));
        if matches!(prev, Some(p) if offset < p) {
            ascending = false;
        }
        prev = Some(offset);
        offset_max = offset_max.max(offset);
        extra_max = extra_max.max(extra);
    }
    (names, counts, ascending, offset_max, extra_max)
}

/// `out` gets `name_count, offsets_ascending, offset_max, extra_max`; `names`/`counts` the names in first-seen
/// order and how often each occurs.
///
/// # Safety
/// `raw` is valid for `len` bytes, `names`/`counts` for `cap` words each and `out` for four u64 words.
#[no_mangle]
pub unsafe extern "C" fn audio_markers_v1(
    raw: *const u8,
    len: usize,
    names: *mut u32,
    counts: *mut u64,
    cap: usize,
    out: *mut u64,
) -> i32 {
    let (Some(nm), Some(ct), Some(out)) = (out_slice(names, cap), out_slice(counts, cap), out_slice(out, 4))
    else {
        return DECLINE;
    };
    let raw = bytes(raw, len);
    if raw.is_empty() || raw.len() % 16 != 0 {
        return INVALID;
    }
    let (n_names, n_counts, ascending, offset_max, extra_max) = markers(raw, raw.len() / 16);
    if n_names.len() > nm.len() {
        return DECLINE;
    }
    nm[..n_names.len()].copy_from_slice(&n_names);
    ct[..n_counts.len()].copy_from_slice(&n_counts);
    out.copy_from_slice(&[
        n_names.len() as u64,
        u64::from(ascending),
        u64::from(offset_max),
        u64::from(extra_max),
    ]);
    DONE
}

/// 1 when the value word of a whole 16-byte record equals `a` or `b`, else 0 (a trailing partial record is
/// ignored, as `marker_records()` drops it).
///
/// # Safety
/// `raw` is valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn audio_marker_has_value_v1(raw: *const u8, len: usize, a: u32, b: u32) -> i32 {
    let raw = bytes(raw, len);
    i32::from(raw.chunks_exact(16).any(|r| {
        let v = u32_at(r, 4);
        v == a || v == b
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adat(flags: u32, streams: &[(u32, Vec<(u8, u32, u32)>)], hdr_extra: usize) -> Vec<u8> {
        let cnt = streams.len() as u32;
        let mut tables = 0usize;
        if flags & 0x0040_0000 != 0 {
            tables += 4;
        }
        if flags & 0x0001_0000 != 0 {
            tables += 2 * cnt as usize;
        }
        if flags & 0x0020_0000 != 0 {
            tables += 4 * (cnt as usize + 1);
        }
        let ntags: usize = streams.iter().map(|s| s.1.len()).sum();
        let hdr = 16 + tables + 4 * cnt as usize + 8 * ntags + hdr_extra;
        let mut b = Vec::new();
        b.extend_from_slice(ADAT_MAGIC);
        b.extend_from_slice(&flags.to_le_bytes());
        b.extend_from_slice(&cnt.to_le_bytes());
        b.extend_from_slice(&(hdr as u32).to_le_bytes());
        b.resize(16 + tables, 0);
        for (sid, tags) in streams {
            b.extend_from_slice(&(((tags.len() as u32) << 29) | sid).to_le_bytes());
        }
        for (_, tags) in streams {
            for &(tt, off, sz) in tags {
                let q = (u64::from(tt) << 56) | (u64::from(sz) << 28) | u64::from(off);
                b.extend_from_slice(&q.to_le_bytes());
            }
        }
        b.resize(hdr, 0);
        b.resize(hdr + 4096, 0);
        b
    }

    #[test]
    fn header_round_trip() {
        let b = adat(0x0040_0000, &[(7, vec![(0x55, 100, 50), (0x76, 200, 32)]), (9, vec![])], 0);
        let mut st = [0u32; 4];
        let mut tg = [0u32; 6];
        let h = adat_header(&b, 0, b.len() as u64, 200_000, 1 << 25, &mut st, &mut tg).ok().unwrap();
        assert_eq!((h.count, h.tags), (2, 2));
        assert_eq!(st, [7, 2, 9, 0]);
        assert_eq!(tg, [0x55, 100, 50, 0x76, 200, 32]);
        assert_eq!(h.end, 232);
        // A tag past the file end is malformed; a short output buffer declines.
        assert_eq!(adat_header(&b, 0, 231, 200_000, 1 << 25, &mut st, &mut tg).err(), Some(INVALID));
        assert_eq!(adat_header(&b, 0, b.len() as u64, 200_000, 1 << 25, &mut st, &mut tg[..3]).err(), Some(DECLINE));
        assert_eq!(adat_header(&b[..15], 0, 15, 200_000, 1 << 25, &mut st, &mut tg).err(), Some(INVALID));
    }

    #[test]
    fn zero_stream_bank_must_fill_header() {
        let ok = adat(0x0040_0000, &[], 0);
        let mut none: [u32; 0] = [];
        let mut none2: [u32; 0] = [];
        assert!(adat_header(&ok, 0, ok.len() as u64, 200_000, 1 << 25, &mut none, &mut none2).is_ok());
        let bad = adat(0x0040_0000, &[], 4);
        assert_eq!(
            adat_header(&bad, 0, bad.len() as u64, 200_000, 1 << 25, &mut none, &mut none2).err(),
            Some(INVALID)
        );
    }

    #[test]
    fn grid20_checks() {
        let mut raw = vec![0u8; 8 + 40];
        raw[4..8].copy_from_slice(&5u32.to_le_bytes());
        raw[24..28].copy_from_slice(&5u32.to_le_bytes());
        raw[32..34].copy_from_slice(&33u16.to_le_bytes());
        assert_eq!(grid20(&raw, 2), (Some(5), Some(5), false, false));
        assert_eq!(grid20(&raw, 0), (None, None, true, true));
    }

    fn words(w: &[u32]) -> Vec<u8> {
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn index_runs_checks() {
        let r = index_runs(&words(&[2, 2, 3, 9, 1, 2, 4, 2, 2, 8, 5, 5]), 1 << 20).unwrap();
        assert_eq!((r.groups, r.values, r.ascending, r.index_max), (2, 5, false, Some(5)));
        assert_eq!(r.hashes, vec![8, 9]);
        assert!(index_runs(&words(&[1, 2, 9, 9]), 1 << 20).is_none());
        assert!(index_runs(&words(&[1, 2, 0, 9]), 0).is_none());
        let empty = index_runs(&words(&[0]), 1 << 20).unwrap();
        assert_eq!((empty.groups, empty.index_max), (0, None));
    }

    #[test]
    fn marker_checks() {
        let raw = words(&[5, 0, 10, 1, 6, 0, 9, 3, 5, 0, 12, 0]);
        let (names, counts, asc, omax, emax) = markers(&raw, 3);
        assert_eq!((names, counts, asc, omax, emax), (vec![5, 6], vec![2, 1], false, 12, 3));
        assert_eq!(unsafe { audio_marker_has_value_v1(raw.as_ptr(), raw.len(), 1, 0) }, 1);
        assert_eq!(unsafe { audio_marker_has_value_v1(raw.as_ptr(), raw.len(), 1, 2) }, 0);
    }
}
