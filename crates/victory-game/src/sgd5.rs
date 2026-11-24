//! `SGD5`: the engine's shader file. Each record holds one shader container,
//! deflate-packed. The header tables are not decoded yet, so records are
//! found by their marker and cross-checked by their sizes:
//!
//! ```text
//! u8 name length (with NUL), name, NUL, u8 kind,
//! u32 n, u32 0x000DDCE7, u32 n, u32 unpacked size, raw deflate (n - 12 bytes)
//! ```

pub const MAGIC: &[u8; 4] = b"SGD5";
const MARKER: [u8; 4] = 0x000D_DCE7_u32.to_le_bytes();
/// Containers are at most a few MiB; refuse to allocate for absurd sizes.
const MAX_UNPACKED: usize = 64 << 20;
/// Bytes a record's packed stream may hold beyond what deflate needs (see
/// [`max_packed`]): the stream's end is not checked, so allow some padding.
const PACKED_SLACK: usize = 4096;
/// Packed bytes read, and unpacked bytes made, to tell whether a record's
/// bytes are a deflate stream (see [`starts_stream`]).
const PROBE: usize = 4096;

/// The most packed bytes a container of `unpacked` bytes can take. Deflate
/// never spends more than 9 bits a byte (fixed codes), and stored blocks only
/// 5 bytes per 65535, so a size beyond this is not a record's: it is bytes
/// that look like a marker (in a container stored between records, say).
fn max_packed(unpacked: usize) -> usize {
    unpacked + unpacked / 8 + PACKED_SLACK
}

/// The fewest packed bytes a container of `unpacked` bytes can take. The
/// smallest deflate stream (one empty fixed-code block) is 2 bytes, and no
/// stream does better than one 258-byte match per 2 bits (1-bit length and
/// distance codes, no extra bits): 1032 bytes a byte. Fewer packed bytes is
/// not a record's, and taking it for one would make its span (12 bytes and
/// up) swallow the header of a real record right after it.
fn min_packed(unpacked: usize) -> usize {
    (unpacked / 1032).max(2)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    /// Offset of the marker in the file.
    pub offset: usize,
    /// Like `PS_<name>_<hash>_Wrapped`; `None` if the bytes before the
    /// record do not read as one.
    pub name: Option<&'a str>,
    /// Record type, meaning unknown (0x32 most shaders, 0x41 mesh shaders
    /// and some material families, 0x3c compute, 0x40, 0x42).
    pub kind: u8,
    packed: &'a [u8],
    unpacked_size: usize,
}

impl Record<'_> {
    /// The container, or `None` if the stream is corrupt.
    pub fn unpack(&self) -> Option<Vec<u8>> {
        miniz_oxide::inflate::decompress_to_vec_with_limit(self.packed, self.unpacked_size)
            .ok()
            .filter(|out| out.len() == self.unpacked_size)
    }

    /// Whether the packed bytes are a deflate stream of the stated size (or
    /// its start: the stream's end is not checked).
    fn is_stream(&self) -> bool {
        starts_stream(self.packed, self.unpacked_size)
    }

    /// The file bytes from the marker to the end of the packed stream.
    pub fn span(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + 12 + self.packed.len()
    }
}

/// Whether `offset` lies inside one of `records`, taken in the order
/// [`records`] yields them (file order, disjoint). Bytes there belong to a
/// record: a container stored as is inside its packed stream (deflate stored
/// blocks) is that record's container, not another one.
pub fn in_record(records: &[Record<'_>], offset: usize) -> bool {
    let i = records.partition_point(|r| r.span().end <= offset);
    records.get(i).is_some_and(|r| r.span().contains(&offset))
}

/// An SGD5 file, or a file holding one (an archive: records are found by
/// their marker, so no index is needed).
pub fn is_sgd5(file: &[u8]) -> bool {
    memchr::memmem::find(file, MAGIC).is_some()
}

/// Every record in an SGD5 file, in file order. The search starts at the
/// first magic: bytes before it (an archive's other files) belong to no SGD5
/// file, whatever markers they happen to hold. It resumes after each
/// record's packed bytes: a marker inside them (stored deflate blocks carry a
/// container's bytes as they are) is part of that record, and so is any part
/// of the next record's header or name, which must follow them. A record
/// whose packed bytes run past the end of the file (a truncated file) is not
/// yielded, and ends the search: every byte after its marker is its own.
/// Sizes that no deflate stream of the stated unpacked size could have (too
/// many packed bytes, or too few) do not form a record, and neither do bytes
/// running past the end that do not start a deflate stream, nor bytes that
/// are no deflate stream laid over a record that is one (whole or cut off),
/// so such bytes neither end the search nor hide records. A corrupt record
/// that covers no other record is still yielded (its [`Record::unpack`] is
/// `None`).
pub fn records(file: &[u8]) -> impl Iterator<Item = Record<'_>> {
    let finder = memchr::memmem::Finder::new(&MARKER);
    // Where the next record may start: after the magic, then after each
    // record's packed bytes. No magic: past the end, so nothing is found.
    let mut floor = memchr::memmem::find(file, MAGIC).map_or(file.len(), |m| m + MAGIC.len());
    // A marker needs its name's NUL, the kind and `n` (6 bytes) before it.
    let mut from = floor + 6;
    let mut streams = Streams::default();
    std::iter::from_fn(move || {
        while let Some(found) = finder.find(file.get(from..)?) {
            let at = from + found;
            match record(&file[floor..], at - floor) {
                Found::Record(r) => {
                    let r = Record { offset: at, ..r };
                    if covers_a_record(file, &r, &mut streams) && !r.is_stream() {
                        // Bytes that only look like a marker, laid over a
                        // real record: the real one wins.
                        from = at + 1;
                        continue;
                    }
                    floor = r.span().end;
                    from = floor + 6;
                    return Some(r);
                }
                Found::Truncated => {
                    from = file.len() + 1;
                    return None;
                }
                Found::Nothing => from = at + 1,
            }
        }
        None
    })
}

enum Found<'a> {
    Record(Record<'a>),
    /// A record header whose packed bytes run past the end of the file.
    Truncated,
    Nothing,
}

/// The record whose marker is at `at`, reading nothing before `file`.
fn record(file: &[u8], at: usize) -> Found<'_> {
    let u32_at = |o: usize| {
        file.get(o..o.checked_add(4)?)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
    };
    let Some(nul) = at.checked_sub(6) else {
        return Found::Nothing;
    };
    let (Some(n), Some(n2), Some(unpacked_size)) = (u32_at(at - 4), u32_at(at + 4), u32_at(at + 8))
    else {
        return Found::Nothing;
    };
    if n2 != n
        || n < 12
        || unpacked_size > MAX_UNPACKED
        || n - 12 > max_packed(unpacked_size)
        || n - 12 < min_packed(unpacked_size)
    {
        return Found::Nothing;
    }
    let Some(packed) = at.checked_add(n).and_then(|end| file.get(at + 12..end)) else {
        // Bytes that only look like a marker (in a container stored between
        // records, say) may have sizes that fit and still run past the end.
        // Only a record whose bytes so far are a deflate stream is one.
        return if starts_stream(file.get(at + 12..).unwrap_or_default(), unpacked_size) {
            Found::Truncated
        } else {
            Found::Nothing
        };
    };
    Found::Record(Record {
        offset: at,
        name: (file[nul] == 0).then(|| name_before(file, nul)).flatten(),
        kind: file[at - 5],
        packed,
        unpacked_size,
    })
}

/// Whether a record whose packed bytes are a deflate stream (or, cut off by
/// the end of the file, start one) starts (its header, 6 bytes before its
/// marker) inside `r`'s span. Such a record inside a real one is that
/// record's (stored deflate blocks); but if `r`'s own bytes are no deflate
/// stream, `r` is bytes that look like a marker (in a container stored
/// between records, say) and the record it covers is real. A cut-off record
/// counts too: were `r` taken for a record, the search would go on inside
/// the cut-off one and find the records it stores.
fn covers_a_record(file: &[u8], r: &Record<'_>, streams: &mut Streams) -> bool {
    let span = r.span();
    // Its header must be after `r`'s marker, and a marker from
    // `span.end + 6` on has its header after the span.
    let hi = file.len().min(span.end + 6 + MARKER.len() - 1);
    streams.any_in(file, r.offset + 7, hi)
}

/// Which markers start a record whose packed bytes are (or, cut off, start)
/// a deflate stream, as
/// far as [`records`] has asked. Its questions move forward through the
/// file, and markers in bytes that only look like records may each cover
/// all the rest of it: asking each of them anew rescans and reinflates the
/// same bytes, so a crafted file of `n` such markers takes `n²` work and
/// more. Each marker is looked at once instead.
#[derive(Default)]
struct Streams {
    /// No marker at an offset in `from..to` starts such a record.
    from: usize,
    to: usize,
    /// The marker at `to` does (and was the first found from `from` on).
    hit: bool,
}

impl Streams {
    /// Whether a marker that lies wholly in `lo..hi` starts such a record
    /// (its header read from `lo` on).
    fn any_in(&mut self, file: &[u8], lo: usize, hi: usize) -> bool {
        if lo < self.from || lo > self.to {
            *self = Self {
                from: lo,
                to: lo,
                hit: false,
            };
        }
        if !self.hit {
            let window = file.get(self.to..hi).unwrap_or_default();
            match memchr::memmem::find_iter(window, &MARKER)
                .map(|m| self.to + m)
                .find(|&at| match record(&file[at - 6..], 6) {
                    Found::Record(inner) => inner.is_stream(),
                    // Cut off, its bytes so far a deflate stream.
                    Found::Truncated => true,
                    Found::Nothing => false,
                }) {
                Some(at) => (self.to, self.hit) = (at, true),
                // Markers at offsets up to `hi - 4` have been looked at.
                None => self.to = self.to.max(hi.saturating_sub(MARKER.len() - 1)),
            }
        }
        self.hit && self.to + MARKER.len() <= hi
    }
}

/// Whether `bytes` are the start of a deflate stream of `unpacked` bytes:
/// they run out before the stream ends, or the stream ends having produced
/// exactly that many bytes.
///
/// Only the first [`PROBE`] bytes are read, and at most [`PROBE`] bytes
/// inflated: [`records`] asks this of every marker that might be a record,
/// and in bytes that only look like records each such marker's bytes may run
/// on through all the ones after it (stored blocks that each end where the
/// next marker's begin), so reading them all makes a file of `n` markers
/// take `n²` work. Real records are told from noise long before that.
fn starts_stream(bytes: &[u8], unpacked: usize) -> bool {
    use miniz_oxide::inflate::{TINFLStatus, decompress_to_vec_with_limit};
    let limit = unpacked.min(PROBE);
    let probe = &bytes[..bytes.len().min(PROBE)];
    #[cfg(test)]
    tests::WORK.with(|w| {
        let (checks, read) = w.get();
        w.set((checks + 1, read + probe.len()));
    });
    match decompress_to_vec_with_limit(probe, limit) {
        Ok(out) => out.len() == unpacked,
        Err(e) => {
            e.status == TINFLStatus::FailedCannotMakeProgress
                // Inflated as far as the probe goes, with more to come.
                || (e.status == TINFLStatus::HasMoreOutput && limit < unpacked)
        }
    }
}

/// The printable name that ends at `nul` and follows its length byte. A
/// length byte of 1 (no characters) is not a name: any `01 00` before the
/// kind would read as one.
fn name_before(file: &[u8], nul: usize) -> Option<&str> {
    let run = file[..nul]
        .iter()
        .rposition(|b| !b.is_ascii_graphic())
        .map_or(0, |p| p + 1);
    // From 32 characters on, the length byte is printable too, and so may be
    // the bytes before it (the previous record's packed tail, say) and the
    // name's own characters: any of them that equals its distance to the NUL
    // reads as a length byte as well. Shader names are identifiers (letters,
    // digits, `_`, not starting with a digit), and a length byte from 33 to
    // 47 (names of 32 to 46 characters) is punctuation and from 48 to 57 a
    // digit, so the earliest length byte whose name is an identifier wins
    // over a stray byte before it; failing that, the earliest.
    let mut lengths = (run.saturating_sub(1)..nul).filter(|&l| usize::from(file[l]) == nul - l);
    let first = lengths.clone().next();
    lengths
        .find(|&l| {
            let name = &file[l + 1..nul];
            !name.first().is_some_and(u8::is_ascii_digit)
                && name.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_')
        })
        .or(first)
        .and_then(|l| std::str::from_utf8(&file[l + 1..nul]).ok())
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// Stream checks made, and packed bytes they read, on this thread
        /// (each test runs on its own): the work [`records`] does, counted
        /// rather than timed, so a slow or busy machine cannot fail a test
        /// and a fast one cannot hide a regression.
        pub(super) static WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    /// The records in `file`, and the stream checks and packed bytes read
    /// to find them.
    fn records_counting(file: &[u8]) -> (Vec<Record<'_>>, usize, usize) {
        WORK.with(|w| w.set((0, 0)));
        let found = records(file).collect();
        let (checks, read) = WORK.with(Cell::get);
        (found, checks, read)
    }

    fn record_bytes(name: &str, kind: u8, payload: &[u8]) -> Vec<u8> {
        let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let n = (packed.len() + 12) as u32;
        let mut out = vec![0xF4, 0x01, 0, 0, name.len() as u8 + 1];
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&[0, kind]);
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&packed);
        out
    }

    #[test]
    fn reads_records() {
        let long = format!("PS_{}_Wrapped", "x".repeat(40));
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&[1, 2, 3, 4]);
        file.extend(record_bytes(
            "VS_Blit_0011aabb_Wrapped",
            0x32,
            b"first container",
        ));
        // A marker whose sizes disagree is not a record.
        file.extend_from_slice(&[9, 0, 0, 0, 0xE7, 0xDC, 0x0D, 0, 8, 0, 0, 0]);
        file.extend(record_bytes(&long, 0x41, &[7; 300]));

        assert!(is_sgd5(&file));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, Some("VS_Blit_0011aabb_Wrapped"));
        assert_eq!(found[0].kind, 0x32);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"first container"[..]));
        assert_eq!(found[1].name, Some(long.as_str()));
        assert_eq!(found[1].unpack(), Some(vec![7; 300]));

        // The same file inside an archive.
        let mut archive = vec![0x5A; 0x1000];
        archive.extend_from_slice(&file);
        assert!(is_sgd5(&archive));
        assert_eq!(records(&archive).count(), 2);
        assert!(!is_sgd5(&archive[..0x1000]));
    }

    #[test]
    fn marker_inside_a_record_is_not_another_record() {
        let inner = record_bytes("PS_inner", 0x32, b"inner container");
        // Level 0: stored blocks, so the inner record's bytes appear as they are.
        let packed = miniz_oxide::deflate::compress_to_vec(&inner, 0);
        assert!(memchr::memmem::find(&packed, &MARKER).is_some());
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&[0xF4, 0x01, 0, 0, 9]);
        file.extend_from_slice(b"PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, inner.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"next"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
        assert_eq!(found[0].unpack(), Some(inner));
    }

    #[test]
    fn offsets_inside_records() {
        // A container stored as is (level 0) inside a record's packed bytes.
        let container = b"DXBC stored as is";
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x05PS_a\0\x32");
        let packed = miniz_oxide::deflate::compress_to_vec(container, 0);
        let n = (packed.len() + 12) as u32;
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        let between = file.len();
        file.extend_from_slice(b"DXBC between records");
        file.extend(record_bytes("PS_b", 0x32, b"second"));
        let after = file.len();
        file.extend_from_slice(b"DXBC after");

        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        let stored = memchr::memmem::find(&file, b"DXBC").unwrap();
        assert!(stored < between);
        assert!(in_record(&found, stored));
        assert!(!in_record(&found, 0), "the magic");
        assert!(!in_record(&found, between));
        assert!(!in_record(&found, after));
        for r in &found {
            let span = r.span();
            assert!(in_record(&found, span.start));
            assert!(in_record(&found, span.end - 1));
            assert!(!in_record(&found, span.start - 1), "the header");
        }
        assert!(!in_record(&[], stored));
    }

    #[test]
    fn header_inside_a_record_is_not_another_record() {
        // The record's packed bytes end (after the deflate stream, which
        // ignores them) with what would be the next header: NUL, kind, `n`.
        let mut packed = miniz_oxide::deflate::compress_to_vec(b"first", 6);
        let tail = miniz_oxide::deflate::compress_to_vec(b"phantom", 6);
        let tail_n = (tail.len() + 12) as u32;
        packed.extend_from_slice(&[0, 0x32]);
        packed.extend_from_slice(&tail_n.to_le_bytes());
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_first\0\x32");
        for v in [n, 0x000D_DCE7, n, 5] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        // Right after it, a marker with no header of its own.
        for v in [0x000D_DCE7, tail_n, 7] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&tail);

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first")]);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"first"[..]));
    }

    #[test]
    fn records_before_the_magic_are_not_read() {
        // An archive whose other file happens to hold a well-formed record.
        let mut archive = record_bytes("PS_elsewhere", 0x32, b"not ours");
        assert_eq!(records(&archive).count(), 0, "no magic, no records");
        archive.extend_from_slice(MAGIC);
        archive.extend(record_bytes("PS_ours", 0x32, b"ours"));
        let names: Vec<_> = records(&archive).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_ours")]);
    }

    #[test]
    fn truncated_file_yields_only_whole_records() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_a", 0x32, &[5; 200]));
        let first_end = file.len();
        file.extend(record_bytes("PS_b", 0x41, b"second"));
        for cut in 0..=file.len() {
            let part = &file[..cut];
            let found: Vec<_> = records(part).collect();
            let expected = usize::from(cut >= first_end) + usize::from(cut == file.len());
            assert_eq!(found.len(), expected, "cut at {cut}");
            for r in found {
                assert!(
                    r.unpack().is_some(),
                    "cut at {cut}: {:?} does not unpack",
                    r.name
                );
            }
        }
    }

    #[test]
    fn truncated_record_hides_the_records_it_stores() {
        // A record whose packed bytes (stored blocks) hold a whole record,
        // cut off after that inner record but before its own end.
        let inner = record_bytes("PS_inner", 0x32, b"inner container");
        let mut payload = inner.clone();
        payload.extend_from_slice(&[0x55; 64]);
        let packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let stored = file.len();
        file.extend_from_slice(&packed);
        let inner_end = stored + memchr::memmem::find(&packed, &inner).unwrap() + inner.len();
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first"), Some("PS_outer")]);

        // Before: [PS_first, PS_inner].
        let names: Vec<_> = records(&file[..inner_end]).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first")]);
    }

    #[test]
    fn marker_without_a_stream_over_a_cut_off_record_hides_nothing() {
        // Bytes that read as a marker with sizes that fit, then bytes that
        // are no deflate stream (block type 3 is reserved), laid over the
        // header of a record whose packed bytes (stored blocks) hold a whole
        // record; the file then cut off after that inner record.
        let inner = record_bytes("PS_inner", 0x32, b"inner container");
        let mut payload = inner.clone();
        payload.extend_from_slice(&[0x55; 64]);
        let packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        let n = (packed.len() + 12) as u32;
        let mut outer = b"\x09PS_outer\0\x32".to_vec();
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            outer.extend_from_slice(&v.to_le_bytes());
        }
        let stored = outer.len();
        outer.extend_from_slice(&packed);

        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        file.extend_from_slice(b"DXBC");
        // 16 bytes of no stream, then the outer record's header and the
        // start of its packed bytes.
        let phantom = 16 + stored + 8;
        let m = (12 + phantom) as u32;
        for v in [m, 0x000D_DCE7, m, 64] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&[0x07; 16]);
        let outer_at = file.len();
        file.extend_from_slice(&outer);
        let inner_end =
            outer_at + stored + memchr::memmem::find(&packed, &inner).unwrap() + inner.len();
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first"), Some("PS_outer")]);

        // Cut off: the phantom covers a record whose bytes so far are a
        // deflate stream, so it is not the record. Before: [PS_first, None,
        // PS_inner] (the phantom was yielded and the search went on inside
        // the outer record, exposing the record it stores).
        for cut in [inner_end, inner_end + 10] {
            let names: Vec<_> = records(&file[..cut]).map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_first")], "cut at {cut}");
        }
    }

    #[test]
    fn marker_with_impossible_sizes_does_not_end_the_search() {
        // Bytes between records that read as a marker whose packed size runs
        // past the end of the file, but is far more than a deflate stream of
        // 16 bytes could take.
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"DXBC");
        for v in [u32::MAX, 0x000D_DCE7, u32::MAX, 16] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        // Before: [] (the phantom was a truncated record and ended the search).
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_after")]);
    }

    #[test]
    fn marker_running_past_the_end_without_a_stream_does_not_end_the_search() {
        // Bytes between records that read as a marker with sizes that fit
        // (64 KiB both ways) and run past the end of the file, followed by
        // bytes that are no deflate stream (block type 3 is reserved).
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        file.extend_from_slice(b"DXBC");
        for v in [0x1_0000_u32, 0x000D_DCE7, 0x1_0000, 0x1_0000] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&[0x07; 16]);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        // Before: [PS_first] (the phantom was a truncated record and ended
        // the search).
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first"), Some("PS_after")]);
    }

    #[test]
    fn truncated_stream_is_told_from_noise() {
        let payload: Vec<u8> = (0..4000u32).map(|i| (i * 7 % 251) as u8).collect();
        for level in [0, 1, 6, 10] {
            let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
            for cut in 0..packed.len() {
                assert!(
                    starts_stream(&packed[..cut], payload.len()),
                    "level {level}, cut at {cut}"
                );
            }
            assert!(starts_stream(&packed, payload.len()), "level {level}");
            // A whole stream of another size is not this record's.
            assert!(!starts_stream(&packed, payload.len() + 1), "level {level}");
            assert!(!starts_stream(&packed, payload.len() - 1), "level {level}");
        }
        // Reserved block type, and a stored block whose length check fails.
        assert!(!starts_stream(&[0x07, 0, 0], 100));
        assert!(!starts_stream(&[0x01, 0x05, 0x00, 0x00, 0x00, 1, 2, 3], 5));
    }

    #[test]
    fn packed_size_bound_holds_for_every_level() {
        // Incompressible bytes, at every level miniz writes: the bound is
        // never reached, so no real record is refused.
        let mut x = 0x1234_5678_u32;
        let noise: Vec<u8> = (0..200_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        for len in [0, 1, 100, 65_535, 65_536, noise.len()] {
            for level in 0..=10 {
                let packed = miniz_oxide::deflate::compress_to_vec(&noise[..len], level);
                assert!(packed.len() <= max_packed(len), "len {len} level {level}");
            }
        }
    }

    #[test]
    fn empty_container_unpacks() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_empty", 0x32, b""));
        let r = records(&file).next().unwrap();
        assert_eq!(r.unpack(), Some(Vec::new()));
    }

    /// Every single-byte change to a small file: no panic, and the records
    /// found are in file order, inside the file and do not overlap.
    #[test]
    fn damaged_file_yields_disjoint_records() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_a", 0x32, &[5; 40]));
        file.extend(record_bytes("VS_b", 0x41, b"second"));
        for at in 0..file.len() {
            for value in [0x00, 0x01, 0x0C, 0x0D, 0x7F, 0xDC, 0xE7, 0xFF] {
                let mut damaged = file.clone();
                damaged[at] = value;
                let mut end = 0;
                for r in records(&damaged) {
                    let start = r.offset - 6;
                    assert!(start >= end, "byte {at} = {value:#x}: records overlap");
                    end = r.span().end;
                    assert!(end <= damaged.len());
                    let _ = r.unpack();
                }
            }
        }
    }

    /// Seeded random files: gaps of noise and of near-miss markers between
    /// records, some records stored as is (level 0) and holding records of
    /// their own. Exactly the planted records are found, each with its
    /// container, and [`in_record`] agrees with a scan of their spans.
    #[test]
    fn random_files_yield_exactly_the_planted_records() {
        let mut x = 0x9E37_79B9_u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for seed in 0..200 {
            let mut file = MAGIC.to_vec();
            // (offset of the marker, name, container)
            let mut planted: Vec<(usize, String, Vec<u8>)> = Vec::new();
            for i in 0..1 + next() % 6 {
                // A gap: noise, then sometimes a marker whose sizes disagree
                // or that no deflate stream could have.
                for _ in 0..next() % 40 {
                    file.push(next() as u8);
                }
                match next() % 3 {
                    0 => {
                        for v in [20_u32, 0x000D_DCE7, 21, 8] {
                            file.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    1 => {
                        for v in [u32::MAX, 0x000D_DCE7, u32::MAX, 16] {
                            file.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    _ => {}
                }
                // A name's length byte follows the gap; a long name's is
                // printable, so noise right before it is kept non-printable.
                file.push(0x01);
                let name = format!("PS_{seed}_{i}_{}", "w".repeat(next() as usize % 50));
                let mut payload: Vec<u8> = (0..next() % 300).map(|_| next() as u8).collect();
                let level = if next() % 2 == 0 {
                    // Stored as is, holding a record and a container's magic.
                    payload.extend(record_bytes("PS_stored", 0x32, b"stored"));
                    payload.extend_from_slice(b"DXBC");
                    0
                } else {
                    6
                };
                let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
                let n = (packed.len() + 12) as u32;
                file.push(name.len() as u8 + 1);
                file.extend_from_slice(name.as_bytes());
                file.extend_from_slice(&[0, 0x32]);
                planted.push((file.len() + 4, name, payload.clone()));
                for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
            }
            for _ in 0..next() % 40 {
                file.push(next() as u8);
            }

            let found: Vec<_> = records(&file).collect();
            let got: Vec<_> = found
                .iter()
                .map(|r| (r.offset, r.name.map(str::to_owned), r.unpack()))
                .collect();
            let want: Vec<_> = planted
                .iter()
                .map(|(at, name, payload)| (*at, Some(name.clone()), Some(payload.clone())))
                .collect();
            assert_eq!(got, want, "seed {seed}");
            for offset in 0..=file.len() {
                let linear = found.iter().any(|r| r.span().contains(&offset));
                assert_eq!(
                    in_record(&found, offset),
                    linear,
                    "seed {seed}, offset {offset}"
                );
            }
        }
    }

    #[test]
    fn empty_name_is_no_name() {
        // A record whose name would be empty: `01 00` before the kind.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("", 0x32, b"nameless"));
        file.extend(record_bytes("PS_named", 0x32, b"named"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [None, Some("PS_named")]);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"nameless"[..]));
    }

    #[test]
    fn printable_bytes_before_a_long_name_are_not_part_of_it() {
        // From 32 characters on, a name's length byte is printable, so the
        // bytes before it (the previous record's packed tail, say) run on
        // into the name. One that equals its own distance to the NUL reads
        // as a length byte too, and must not win over the real one.
        let name = "PS_DeferredLightingComposite_0011aabb_Wrapped";
        let len = name.len() as u8 + 1;
        for before in [vec![len + 1], vec![len + 2, b'A'], vec![b'x', len + 1]] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(&[0, 0]);
            file.extend_from_slice(&before);
            let header = record_bytes(name, 0x32, b"container");
            // Drop the helper's 4 bytes before the length byte.
            file.extend_from_slice(&header[4..]);
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some(name)], "bytes before: {before:?}");
        }
        // A character of the name that equals its distance to the NUL (the
        // `4`, 52) is not its length byte either.
        for long in [
            format!("PS_{}_Wrapped", "x".repeat(100)),
            format!("PS_4_0_{}", "w".repeat(48)),
        ] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes(&long, 0x32, b"long"));
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some(&long[..])]);
        }
    }

    #[test]
    fn a_name_does_not_start_with_its_length_byte() {
        // A name of 47 to 56 characters has a digit for its length byte, so a
        // stray byte right before it that equals its distance to the NUL
        // reads as a length byte whose name (the digit, then the name) is all
        // letters, digits and `_`. Names are identifiers, which do not start
        // with a digit.
        for name in [
            format!("PS_{}", "w".repeat(44)),
            format!(
                "PS_DeferredLightingComposite_0011aabb_{}_Wrapped",
                "x".repeat(3)
            ),
            format!("PS_{}", "w".repeat(53)),
        ] {
            let len = name.len() as u8 + 1;
            assert!(len.is_ascii_digit(), "{name}");
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(&[0, 0, len + 1]);
            let header = record_bytes(&name, 0x32, b"container");
            file.extend_from_slice(&header[4..]);
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some(&name[..])]);
        }
    }

    #[test]
    fn archive_with_two_sgd5_files_yields_both() {
        // Two SGD5 files one after the other in an archive, with another
        // file's bytes between them: the second magic sits in a gap.
        let mut archive = b"other file".to_vec();
        archive.extend_from_slice(MAGIC);
        archive.extend(record_bytes("PS_one", 0x32, b"first file"));
        archive.extend_from_slice(b"another file between");
        archive.extend_from_slice(MAGIC);
        archive.extend(record_bytes("PS_two", 0x41, b"second file"));
        let found: Vec<_> = records(&archive).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_one"), Some("PS_two")]);
        assert_eq!(found[1].kind, 0x41);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"second file"[..]));
    }

    #[test]
    fn packed_size_lower_bound_holds_for_every_level() {
        // The most compressible bytes, at every level miniz writes: no real
        // record has fewer packed bytes than the bound.
        let zeros = vec![0_u8; 4 << 20];
        for len in [0, 1, 2, 258, 259, 1032, 65_536, 1 << 20, zeros.len()] {
            for level in 0..=10 {
                let packed = miniz_oxide::deflate::compress_to_vec(&zeros[..len], level);
                assert!(packed.len() >= min_packed(len), "len {len} level {level}");
            }
        }
    }

    #[test]
    fn marker_with_too_few_packed_bytes_does_not_hide_the_next_record() {
        // Bytes between records that read as a marker with no packed bytes
        // (or one) for a 1 MiB container, right before a real record: taken
        // for a record, its span would cover the real record's header.
        for packed_len in [0_u32, 1, 2, 3] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes("PS_first", 0x32, b"first"));
            file.extend_from_slice(b"DXBC");
            let n = 12 + packed_len;
            for v in [n, 0x000D_DCE7, n, 1 << 20] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            // The next record from its length byte on: with packed bytes,
            // the phantom's span would start over it.
            let mut next = record_bytes("PS_after", 0x32, b"after");
            next.drain(..4);
            file.extend(next);
            // Before: [PS_first, None, None] for a phantom of 1 to 3 packed
            // bytes (its span swallowed PS_after's length byte, so PS_after
            // lost its name), and [PS_first, None, PS_after] for none (a
            // record that never unpacks).
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(
                names,
                [Some("PS_first"), Some("PS_after")],
                "{packed_len} packed bytes"
            );
        }
    }

    #[test]
    fn smallest_containers_are_records() {
        // The fewest packed bytes deflate writes: every small container at
        // every level is still found.
        for len in [0_usize, 1, 2, 3] {
            for level in 0..=10 {
                let payload = vec![0x41; len];
                let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
                let n = (packed.len() + 12) as u32;
                let mut file = MAGIC.to_vec();
                file.extend_from_slice(b"\x05PS_s\0\x32");
                for v in [n, 0x000D_DCE7, n, len as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
                let found: Vec<_> = records(&file).collect();
                assert_eq!(found.len(), 1, "len {len} level {level}");
                assert_eq!(found[0].unpack(), Some(payload), "len {len} level {level}");
            }
        }
    }

    #[test]
    fn marker_without_a_stream_does_not_hide_the_record_it_covers() {
        // Bytes between records that read as a marker with sizes that fit
        // and stay inside the file, followed by bytes that are no deflate
        // stream (block type 3 is reserved) and then a real record: taken
        // for a record, its span covers the real record's header.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        let after = record_bytes("PS_after", 0x32, b"after");
        // PS_after's header (6 bytes before its marker) starts 16 + 13 bytes
        // into the phantom's packed bytes: from 30 on they cover it.
        for span in [30, 35, after.len(), after.len() + 31] {
            let mut file = file.clone();
            file.extend_from_slice(b"DXBC");
            let n = (12 + span) as u32;
            for v in [n, 0x000D_DCE7, n, 64] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&[0x07; 16]);
            file.extend(&after);
            file.extend_from_slice(&[0x5A; 16]);
            // Before: [PS_first, None] (the phantom hid PS_after).
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_first"), Some("PS_after")], "span {span}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn marker_without_a_stream_ending_before_a_header_is_a_record_too() {
        // The phantom of the test above, its span ending just short of
        // PS_after's header (29 packed bytes: 16 of noise, then PS_after's
        // 13 bytes before its NUL) or of its name (20: noise and the 4 bytes
        // before the length byte). It covers no record, so it is yielded
        // (and does not unpack), and PS_after follows it; a name the phantom
        // swallowed, wholly or in part, is not read, as the name lies before
        // where PS_after may start.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        let after = record_bytes("PS_after", 0x32, b"after");
        for (span, name) in [(20, Some("PS_after")), (21, None), (25, None), (29, None)] {
            let mut file = file.clone();
            file.extend_from_slice(b"DXBC");
            let phantom = file.len() + 4;
            let n = (12 + span) as u32;
            for v in [n, 0x000D_DCE7, n, 64] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&[0x07; 16]);
            let start = file.len();
            file.extend(&after);
            file.extend_from_slice(&[0x5A; 16]);
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 3, "span {span}");
            assert_eq!(found[1].offset, phantom, "span {span}");
            assert_eq!(found[1].name, None, "span {span}");
            assert_eq!(found[1].unpack(), None, "span {span}");
            assert_eq!(found[2].offset, start + 19, "span {span}");
            assert_eq!(found[2].name, name, "span {span}");
            assert_eq!(found[2].unpack().as_deref(), Some(&b"after"[..]));
            assert!(found[1].span().end <= found[2].offset - 6, "span {span}");
        }
    }

    #[test]
    fn corrupt_record_is_still_a_record() {
        // A real record whose stream is damaged covers no other record: it
        // is yielded (and does not unpack), and so is the record after it.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("CS_x", 0x3c, &[1; 64]));
        file[31] = 0xFF;
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("CS_x"), Some("PS_after")]);
        assert_eq!(found[0].unpack(), None);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
    }

    #[test]
    fn stream_of_another_size_does_not_unpack() {
        // A record whose stream makes one byte more, or one fewer, than its
        // stated size covers no other record: it is yielded, does not
        // unpack, and the record after it is found.
        let packed = miniz_oxide::deflate::compress_to_vec(b"twelve bytes", 6);
        let n = (packed.len() + 12) as u32;
        for stated in [11_u32, 13] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_a\0\x32");
            for v in [n, 0x000D_DCE7, n, stated] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_a"), Some("PS_after")], "stated {stated}");
            assert_eq!(found[0].unpack(), None, "stated {stated}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn packed_bytes_up_to_the_bound_are_a_record() {
        // A real stream padded out (its end is not checked) to exactly the
        // most packed bytes the stated size allows is a record and unpacks;
        // one byte more is not a record (it never hides the next one).
        for payload in [Vec::new(), vec![0x41; 64], (0..=255).collect()] {
            let packed = miniz_oxide::deflate::compress_to_vec(&payload, 6);
            let bound = max_packed(payload.len());
            for (len, want) in [
                (bound, vec![Some("PS_a"), Some("PS_after")]),
                (bound + 1, vec![Some("PS_after")]),
            ] {
                let mut padded = packed.clone();
                padded.resize(len, 0);
                let n = (padded.len() + 12) as u32;
                let mut file = MAGIC.to_vec();
                file.extend_from_slice(b"\x05PS_a\0\x32");
                for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&padded);
                file.extend(record_bytes("PS_after", 0x32, b"after"));
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                let at = format!("{} bytes, {len} packed", payload.len());
                assert_eq!(names, want, "{at}");
                if len == bound {
                    assert_eq!(found[0].unpack(), Some(payload.clone()), "{at}");
                }
                let last = found.last().expect("PS_after");
                assert_eq!(last.unpack().as_deref(), Some(&b"after"[..]), "{at}");
            }
        }
    }

    #[test]
    fn marker_in_padding_belongs_to_the_record_even_if_padding_is_cut_short() {
        let payload = b"outer";
        let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let inner = record_bytes("PS_inner", 0x41, b"inner");
        let n = (12 + packed.len() + inner.len() + 1) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        file.extend_from_slice(&n.to_le_bytes());
        let marker = file.len();
        for v in [0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend_from_slice(&inner);
        let cut = file.len();
        file.push(0);
        let end = file.len();
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].span(), marker..end);
        assert_eq!(found[0].unpack().as_deref(), Some(payload.as_slice()));
        assert_eq!(found[1].name, Some("PS_after"));
        assert_eq!(records(&file[..cut]).count(), 0);

        // The first stream ended after five bytes. If the outer header
        // claims six, the complete record in its padding wins instead.
        file[marker + 8..marker + 12].copy_from_slice(&6u32.to_le_bytes());
        for (bytes, names) in [
            (&file[..], vec![Some("PS_inner"), Some("PS_after")]),
            (&file[..cut], vec![Some("PS_inner")]),
        ] {
            let found: Vec<_> = records(bytes).map(|r| r.name).collect();
            assert_eq!(found, names);
        }
    }

    #[test]
    fn marker_stating_more_than_the_cap_is_not_a_record() {
        // A record stored as is (level 0) inside bytes whose marker states
        // an unpacked size: up to the cap they are a record, which holds
        // the stored one; past it they are not, and the stored record is
        // found in their place. Enough packed bytes that the cap's lower
        // bound on them is met.
        let inner = record_bytes("PS_inner", 0x32, b"inner");
        let mut payload = vec![0x5A; 100];
        payload.extend_from_slice(&inner);
        payload.resize(min_packed(MAX_UNPACKED + 1) + 64, 0x5A);
        let packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        let n = (packed.len() + 12) as u32;
        for (stated, want) in [
            (MAX_UNPACKED, Some("PS_outer")),
            (MAX_UNPACKED + 1, Some("PS_inner")),
        ] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x09PS_outer\0\x32");
            for v in [n, 0x000D_DCE7, n, stated as u32] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [want], "stated {stated}");
        }
    }

    #[test]
    fn nested_markers_without_streams_take_linear_work() {
        // Bytes that each read as a marker whose span runs to the end of the
        // file and covers every one after it, then a real record: each is
        // skipped in favour of the record it covers. Before, every one of
        // them rescanned (and reinflated) the rest of the file: these 8000
        // (150 KB) took 43 s in a test build, and now take milliseconds.
        let real = record_bytes("PS_real", 0x32, b"real");
        let count = 8000;
        let end = MAGIC.len() + count * 19 + real.len();
        let mut file = MAGIC.to_vec();
        for _ in 0..count {
            let n = (end - (file.len() + 6)) as u32;
            file.extend_from_slice(&[0, 0x32]);
            for v in [n, 0x000D_DCE7, n, 1 << 20] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.push(0x07); // reserved block type: no deflate stream
        }
        file.extend(&real);
        assert_eq!(file.len(), end);
        let (found, checks, _) = records_counting(&file);
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_real")]);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"real"[..]));
        // Each marker is checked for a stream at most twice (as a record,
        // and as one another covers), not once for every marker before it.
        assert!(checks <= 2 * (count + 1), "{checks} stream checks");
    }

    #[test]
    fn corrupt_stream_does_not_unpack() {
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("CS_x", 0x3c, &[1; 64]));
        file[31] = 0xFF; // first packed byte: reserved block type
        let r = records(&file).next().unwrap();
        assert_eq!(r.unpack(), None);
    }

    /// `count` markers between the magic and a real record, each followed by
    /// a stored block that ends where the next marker's stored block starts,
    /// so the deflate stream after every marker runs through all the ones
    /// after it, to a reserved block (no deflate stream) just before the
    /// real record. `n(at)` is each marker's `n`, given its offset.
    fn chained_markers(count: usize, n: impl Fn(usize) -> u32, real: &[u8]) -> Vec<u8> {
        const PAD: usize = 41;
        let len = (PAD + 18) as u16;
        let mut file = MAGIC.to_vec();
        for _ in 0..count {
            file.extend_from_slice(&[0, 0x32]);
            let n = n(file.len() + 4);
            for v in [n, 0x000D_DCE7, n, n] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.push(0x00); // stored, not final
            file.extend_from_slice(&len.to_le_bytes());
            file.extend_from_slice(&(!len).to_le_bytes());
            file.extend_from_slice(&[0x5A; PAD]);
        }
        file.extend_from_slice(&[0x5A; 18]);
        file.push(0x07); // reserved block type
        file.extend_from_slice(real);
        file
    }

    #[test]
    fn chained_markers_past_the_end_take_linear_work() {
        // Each marker's packed size runs past the end of the file, so its
        // bytes are checked for the start of a deflate stream. Before, each
        // check inflated the rest of the chain: these 32000 (2 MB) took 9 s
        // in a test build. The chain is a valid deflate stream for far more
        // than the probe reads, so the first marker is now taken for a
        // truncated record, as a cut-off file's last record would be.
        let real = record_bytes("PS_real", 0x32, b"real");
        let count = 32_000;
        let file = chained_markers(count, |_| MAX_UNPACKED as u32, &real);
        let (found, checks, read) = records_counting(&file);
        assert!(found.is_empty());
        // Each check reads a probe, not the rest of the chain.
        assert!(checks <= 2 * count, "{checks} stream checks");
        assert!(read <= checks * PROBE, "{read} bytes read");
    }

    #[test]
    fn chained_markers_inside_the_file_take_linear_work() {
        // Each marker's packed bytes end at the end of the file, so the
        // first covers all the others and each of them is checked for a
        // deflate stream. Before, each check inflated the rest of the chain
        // (9 s for these 32000 in a test build). The first is yielded as a
        // corrupt record whose span is all of them.
        let count = 32_000;
        let end = chained_markers(count, |_| 0, &[]).len();
        let file = chained_markers(count, |at| (end - at) as u32, &[]);
        let (found, checks, read) = records_counting(&file);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].span(), MAGIC.len() + 6..file.len());
        assert_eq!(found[0].unpack(), None);
        assert!(checks <= 2 * count, "{checks} stream checks");
        assert!(read <= checks * PROBE, "{read} bytes read");
    }

    /// Seeded files, some records stored as is (level 0, holding a record of
    /// their own), the bytes then damaged (changed, inserted, removed, or a
    /// marker or a small `n` put in), and every prefix of each: a file cut
    /// short (still being written or copied, say) yields no container the
    /// whole file does not, each from the same record in every field, so
    /// reading it early never reports a container that later turns out to be
    /// bytes of another record. Records only appear as the file grows:
    /// nothing it adds can hide one that unpacks.
    #[test]
    fn cut_off_files_yield_only_records_of_the_whole_file() {
        let mut x = 0x2545_F491_u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for seed in 0..300 {
            let mut file = MAGIC.to_vec();
            for i in 0..1 + next() % 4 {
                let mut payload: Vec<u8> = (0..next() % 120).map(|_| (next() % 4) as u8).collect();
                let level = [0, 1, 6, 10][next() as usize % 4];
                if level == 0 {
                    // Stored as is, holding a record: cut off after it, the
                    // outer record must not expose it.
                    payload.extend(record_bytes("PS_stored", 0x32, b"stored"));
                    payload.extend((0..next() % 40).map(|_| next() as u8));
                }
                let packed = miniz_oxide::deflate::compress_to_vec(&payload, level);
                let n = (packed.len() + 12) as u32;
                let name = format!("PS_{i}");
                file.extend_from_slice(&[0x01, name.len() as u8 + 1]);
                file.extend_from_slice(name.as_bytes());
                file.extend_from_slice(&[0, 0x32]);
                for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
            }
            for _ in 0..1 + next() % 5 {
                let at = next() as usize % (file.len() + 1);
                match next() % 5 {
                    0 if at < file.len() => file[at] = next() as u8,
                    1 => file.insert(at, next() as u8),
                    2 if at < file.len() => _ = file.remove(at),
                    3 => _ = file.splice(at..at, MARKER),
                    _ => _ = file.splice(at..at, (next() % 600).to_le_bytes()),
                }
            }

            let whole: Vec<_> = records(&file).collect();
            for cut in 0..file.len() {
                let part = &file[..cut];
                for r in records(part) {
                    assert!(r.span().end <= cut, "seed {seed}, cut at {cut}");
                    if r.unpack().is_none() {
                        // Corrupt bytes over corrupt bytes: which is the
                        // record is not known, and neither yields a container.
                        continue;
                    }
                    assert!(
                        whole.contains(&r),
                        "seed {seed}, cut at {cut}: {r:?} is not a record of the whole file"
                    );
                }
            }
        }
    }

    /// A bare record header (NUL, kind, `n`, no name) with its marker.
    fn bare_record(nul: u8, kind: u8, payload: &[u8]) -> Vec<u8> {
        let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let n = (packed.len() + 12) as u32;
        let mut out = vec![nul, kind];
        out.extend_from_slice(&n.to_le_bytes());
        for v in [0x000D_DCE7, n, payload.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&packed);
        out
    }

    #[test]
    fn header_right_after_the_magic_is_a_record() {
        // The header's 6 bytes start at the magic's end: the earliest
        // marker the search reads. Nothing before them reads as a name.
        let mut file = MAGIC.to_vec();
        file.extend(bare_record(0, 0x3c, b"compute"));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].offset, MAGIC.len() + 6);
        assert_eq!(found[0].name, None);
        assert_eq!(found[0].kind, 0x3c);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"compute"[..]));
    }

    #[test]
    fn header_reaching_into_the_magic_is_not_a_record() {
        // One header byte short: its NUL would be the magic's last byte, so
        // the marker (one byte earlier than the earliest) is not looked at,
        // whatever the sizes say. A real record after it is still found.
        let mut file = MAGIC.to_vec();
        let mut cut = bare_record(0, 0x32, b"inside the magic");
        cut.remove(0);
        file.extend(&cut);
        assert_eq!(records(&file).count(), 0);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let names: Vec<_> = records(&file).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_after")]);
    }

    #[test]
    fn record_without_a_nul_before_its_kind_has_no_name() {
        // The NUL is not checked to find a record, only to read its name.
        for nul in [0x01, 0x41, 0xFF] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_x");
            file.extend(bare_record(nul, 0x32, b"no name"));
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 1, "nul {nul:#x}");
            assert_eq!(found[0].name, None, "nul {nul:#x}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"no name"[..]));
        }
    }

    #[test]
    fn header_right_after_a_record_is_a_record_and_one_byte_earlier_is_not() {
        // Back to back: the next header starts where the packed bytes end.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_first", 0x32, b"first"));
        let end = file.len();
        let mut back_to_back = file.clone();
        back_to_back.extend(bare_record(0, 0x41, b"second"));
        let found: Vec<_> = records(&back_to_back).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].span().end, end);
        assert_eq!(found[1].offset, end + 6);
        assert_eq!(found[1].kind, 0x41);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"second"[..]));

        // Its NUL laid over the first record's last packed byte: that byte
        // is the first record's, so the second is not read.
        let mut overlap = file.clone();
        let mut second = bare_record(0, 0x41, b"second");
        second.remove(0);
        overlap.extend(&second);
        let names: Vec<_> = records(&overlap).map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_first")]);
    }

    #[test]
    fn record_stating_no_packed_bytes_is_not_a_record() {
        // `n` of 12 (header only) and 13 hold fewer packed bytes than the
        // smallest deflate stream, even for an empty container.
        for n in [12_u32, 13] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_e\0\x32");
            for v in [n, 0x000D_DCE7, n, 0] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&[0x03, 0x00][..(n - 12) as usize]);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let names: Vec<_> = records(&file).map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_after")], "n {n}");
        }
    }

    #[test]
    fn packed_bytes_at_the_lower_bound_are_a_record_and_one_fewer_is_not() {
        // A 10320-byte container needs at least 10 packed bytes: 10 bytes
        // that are no deflate stream (block type 3 is reserved) and cover no
        // record are a corrupt record; 9 are not a record at all. Either
        // way the real record right after them is found.
        let unpacked = 1032 * 10;
        assert_eq!(min_packed(unpacked), 10);
        for (packed_len, corrupt) in [(10_u32, true), (9, false)] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_l\0\x32");
            let n = 12 + packed_len;
            for v in [n, 0x000D_DCE7, n, unpacked as u32] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&vec![0x07; packed_len as usize]);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if corrupt {
                assert_eq!(names, [Some("PS_l"), Some("PS_after")]);
                assert_eq!(found[0].unpack(), None);
            } else {
                assert_eq!(names, [Some("PS_after")]);
            }
            let last = found.last().expect("PS_after");
            assert_eq!(
                last.unpack().as_deref(),
                Some(&b"after"[..]),
                "{packed_len}"
            );
        }
    }

    #[test]
    fn marker_at_the_end_without_its_sizes_is_not_a_record() {
        // A marker in the file's last bytes, with its header before it but
        // none, some or all of its second `n` and unpacked size cut off: not
        // a record, not a truncated one, and no out-of-bounds read. The
        // record before it is still yielded.
        let mut whole = MAGIC.to_vec();
        whole.extend(record_bytes("PS_first", 0x32, b"first"));
        let first_end = whole.len();
        let mut tail = vec![0, 0x32];
        for v in [64_u32, 0x000D_DCE7, 64, 64] {
            tail.extend_from_slice(&v.to_le_bytes());
        }
        // Header and marker (10 bytes), then up to 7 of the 8 size bytes.
        for keep in 10..tail.len() {
            let mut file = whole.clone();
            file.extend_from_slice(&tail[..keep]);
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 1, "{keep} bytes");
            assert_eq!(found[0].span().end, first_end, "{keep} bytes");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"first"[..]));
        }
    }

    #[test]
    fn sizes_that_disagree_are_not_a_record() {
        // The two copies of `n` must match: one more or one less in either
        // copy is bytes that only look like a record, whatever stream
        // follows, and the real record after them is still found.
        let packed = miniz_oxide::deflate::compress_to_vec(b"phantom", 6);
        let n = (packed.len() + 12) as u32;
        for (first, second) in [(n, n + 1), (n, n - 1), (n + 1, n), (n - 1, n)] {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_d\0\x32");
            for v in [first, 0x000D_DCE7, second, 7] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_after")], "n {first} and {second}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn record_whose_size_reads_as_a_marker_is_one_record() {
        // `n` of 0x000DDCE7 (908519): both copies of it, before and after
        // the real marker, are marker bytes too. Neither is another record
        // (the first's sizes are its name and the real marker, the second's
        // lie inside its own header), and the record unpacks.
        let n = 0x000D_DCE7_u32;
        let packed_len = n as usize - 12;
        // Stored blocks: a few bytes per block on top of the container.
        let mut payload = vec![0x5A_u8; packed_len];
        let mut packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        payload.truncate(2 * packed_len - packed.len());
        packed = miniz_oxide::deflate::compress_to_vec(&payload, 0);
        assert!(packed.len() <= packed_len);
        // The stream's end is not checked: pad it out to `n`.
        packed.resize(packed_len, 0);
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x05PS_m\0\x32");
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_m"), Some("PS_after")]);
        assert_eq!(found[0].offset, MAGIC.len() + 11);
        assert_eq!(found[0].span().len(), n as usize);
        assert_eq!(found[0].unpack().as_deref(), Some(&payload[..]));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
    }

    #[test]
    fn files_too_short_for_a_header_yield_nothing() {
        // Nothing, part of the magic, the magic alone, and the magic with
        // fewer bytes than a header and marker: no records, no truncated
        // one, and no out-of-bounds read, whatever those bytes are.
        assert_eq!(records(b"").count(), 0);
        for cut in 1..MAGIC.len() {
            assert!(!is_sgd5(&MAGIC[..cut]));
            assert_eq!(records(&MAGIC[..cut]).count(), 0, "{cut} bytes");
        }
        let record = bare_record(0, 0x32, b"cut");
        for keep in 0..10 {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(&record[..keep]);
            assert!(is_sgd5(&file));
            assert_eq!(records(&file).count(), 0, "{keep} bytes");
        }
        // Marker bytes right after the magic have no header before them.
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&MARKER);
        file.extend_from_slice(&record[6..]);
        assert_eq!(records(&file).count(), 0);
    }

    #[test]
    fn magic_inside_a_record_does_not_restart_the_search() {
        // A whole SGD5 file stored as is (level 0) in a record's packed
        // bytes: its magic and records are that record's container, and the
        // search goes on after the packed bytes, not from the inner magic.
        let mut inner = MAGIC.to_vec();
        inner.extend(record_bytes("PS_inner", 0x32, b"inner container"));
        let packed = miniz_oxide::deflate::compress_to_vec(&inner, 0);
        assert!(memchr::memmem::find(&packed, MAGIC).is_some());
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, inner.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
        assert_eq!(found[0].unpack(), Some(inner.clone()));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        // The stored file, read on its own, is one record.
        let nested: Vec<_> = records(&inner).map(|r| r.name).collect();
        assert_eq!(nested, [Some("PS_inner")]);
    }

    #[test]
    fn container_longer_than_the_probe_is_one_record_whole_or_cut_off() {
        // Only the first PROBE packed bytes are read to tell a stream from
        // noise. A container stored as is (level 0), several probes long,
        // holding a record past the probe: whole, it is one record and the
        // one it stores is its own; cut off past the probe, it is a cut-off
        // record, so the search ends and the stored record is not yielded.
        let mut container: Vec<u8> = (0..3 * PROBE as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let stored_at = container.len();
        container.extend(record_bytes("PS_inner", 0x32, b"inner container"));
        let packed = miniz_oxide::deflate::compress_to_vec(&container, 0);
        let marker = memchr::memmem::find(&packed, &MARKER).unwrap();
        assert!(marker > 2 * PROBE, "stored past the probe: {marker}");
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_before", 0x32, b"before"));
        let outer = file.len() + 15;
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [Some("PS_before"), Some("PS_outer"), Some("PS_after")]
        );
        assert_eq!(found[1].offset, outer);
        assert_eq!(found[1].span().len(), n as usize);
        assert_eq!(found[1].unpack(), Some(container.clone()));
        assert!(in_record(&found, outer + 12 + marker));
        assert!(stored_at > PROBE);

        // Cut off anywhere past the probe, up to the stored record's end.
        let packed_at = outer + 12;
        for cut in [PROBE + 1, 2 * PROBE, marker, marker + 12, packed.len() - 1] {
            let names: Vec<_> = records(&file[..packed_at + cut]).map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_before")], "cut {cut} packed bytes in");
        }
    }

    #[test]
    fn search_stays_ended_once_it_ends() {
        // Past the last record, or at a cut-off one, the iterator keeps
        // returning `None` (no rescan from the start, no out-of-bounds
        // read), and a new search over the same bytes finds the same.
        let mut whole = MAGIC.to_vec();
        whole.extend(record_bytes("PS_a", 0x32, b"first"));
        whole.extend(record_bytes("PS_b", 0x41, b"second"));
        let cut = whole.len() - 1;
        for file in [&whole[..], &whole[..cut]] {
            let want = if file.len() == whole.len() { 2 } else { 1 };
            let mut it = records(file);
            for _ in 0..want {
                assert!(it.next().is_some(), "{} bytes", file.len());
            }
            for _ in 0..3 {
                assert_eq!(it.next(), None, "{} bytes", file.len());
            }
            assert_eq!(records(file).count(), want, "{} bytes", file.len());
        }
    }

    #[test]
    fn stream_is_checked_only_as_far_as_the_probe() {
        // At most PROBE bytes are inflated, from at most PROBE packed bytes.
        // Within both, a stream's size is checked exactly; past either,
        // only as far as the probe goes, and such a record is still one
        // (it covers no other), with `unpack` saying it is the wrong size.
        let payload =
            |size: usize| -> Vec<u8> { (0..size as u32).map(|i| (i * 13 % 251) as u8).collect() };
        let deflated = |size: usize| miniz_oxide::deflate::compress_to_vec(&payload(size), 6);

        // Exactly PROBE bytes, packed well within the probe: exact.
        let packed = deflated(PROBE);
        assert!(packed.len() < PROBE);
        assert!(starts_stream(&packed, PROBE));
        assert!(!starts_stream(&packed, PROBE - 1));
        assert!(!starts_stream(&packed, PROBE + 1));

        // One byte more than PROBE: a stated size past it is not checked.
        let packed = deflated(PROBE + 1);
        assert!(starts_stream(&packed, PROBE + 1));
        assert!(!starts_stream(&packed, PROBE), "the limit is reached");
        assert!(starts_stream(&packed, PROBE + 2), "past the probe");

        // Stored as is (level 0), fewer bytes than PROBE take more packed
        // bytes than it: the probe ends inside the stored block.
        let stored = miniz_oxide::deflate::compress_to_vec(&payload(PROBE - 1), 0);
        assert!(stored.len() > PROBE);
        assert!(starts_stream(&stored, PROBE - 1));
        assert!(starts_stream(&stored, PROBE - 2), "cut by the probe");
        // Within the probe, stored bytes past the stated size still show.
        assert!(!starts_stream(&stored[..PROBE / 2], PROBE / 4));

        for (packed, stated) in [(deflated(PROBE + 1), PROBE + 2), (stored, PROBE - 2)] {
            let n = (packed.len() + 12) as u32;
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_p\0\x32");
            for v in [n, 0x000D_DCE7, n, stated as u32] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_p"), Some("PS_after")], "stated {stated}");
            assert!(found[0].is_stream(), "stated {stated}");
            assert_eq!(found[0].unpack(), None, "stated {stated}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn marker_straddling_a_records_end_is_neither_a_record_nor_hides_the_next() {
        // A record's packed bytes (past the end of its deflate stream, which
        // ignores them) end with the first bytes of a marker, and the next
        // record's header supplies the rest: 3 bytes and its NUL, or 2 bytes
        // and its NUL and kind. With all 4 in the packed bytes, the marker
        // reads its sizes from the next record's header. Whichever, the
        // marker lies in the first record, and both records are found.
        let first = miniz_oxide::deflate::compress_to_vec(b"first", 6);
        for (tail, nul, kind) in [
            (&MARKER[..3], 0x00, 0x32),
            (&MARKER[..2], MARKER[2], MARKER[3]),
            (&MARKER[..], 0x00, 0x41),
        ] {
            let mut packed = first.clone();
            packed.extend_from_slice(tail);
            let n = (packed.len() + 12) as u32;
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x09PS_first\0\x32");
            for v in [n, 0x000D_DCE7, n, 5] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            let end = file.len();
            file.extend(bare_record(nul, kind, b"second"));
            let straddling = end - tail.len();
            assert_eq!(
                file[straddling..straddling + 4],
                MARKER,
                "{} bytes",
                tail.len()
            );

            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 2, "{} bytes", tail.len());
            assert_eq!(found[0].name, Some("PS_first"));
            assert_eq!(found[0].span().end, end);
            assert_eq!(found[0].unpack().as_deref(), Some(&b"first"[..]));
            assert!(in_record(&found, straddling));
            assert_eq!(found[1].offset, end + 6);
            assert_eq!(found[1].kind, kind);
            assert_eq!(found[1].name, None, "{} bytes", tail.len());
            assert_eq!(found[1].unpack().as_deref(), Some(&b"second"[..]));
        }
    }

    #[test]
    fn size_shorter_than_its_own_header_is_not_a_record() {
        // `n` counts the 12 bytes from the marker on, so 0 to 11 cannot be a
        // record's, whatever follows (here a real stream of the stated
        // size): no record, no underflow, and the record after is found.
        let packed = miniz_oxide::deflate::compress_to_vec(b"short", 6);
        for n in 0_u32..12 {
            let mut file = MAGIC.to_vec();
            file.extend_from_slice(b"\x05PS_s\0\x32");
            for v in [n, 0x000D_DCE7, n, 5] {
                file.extend_from_slice(&v.to_le_bytes());
            }
            file.extend_from_slice(&packed);
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_after")], "n {n}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn longest_name_a_length_byte_states_is_read_whole() {
        // A length byte of 0xFF: 254 characters and the NUL, the most one
        // byte can state. Read whole, with no byte before it taken for part
        // of the name, whether the record follows the magic or another
        // record's packed bytes.
        let name = format!("PS_{}", "n".repeat(251));
        assert_eq!(name.len() + 1, 0xFF);
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes(&name, 0x32, b"long"));
        file.extend(record_bytes(&name, 0x41, b"again"));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 2);
        for (r, kind, body) in [(&found[0], 0x32, &b"long"[..]), (&found[1], 0x41, b"again")] {
            assert_eq!(r.name, Some(name.as_str()));
            assert_eq!(r.kind, kind);
            assert_eq!(r.unpack().as_deref(), Some(body));
        }
    }

    #[test]
    fn length_byte_that_misstates_the_name_gives_no_name() {
        // `PS_wrong` and its NUL take 9 bytes. A length byte of 0, one off
        // either way, or the most a byte can state is not its distance to
        // the NUL, and no character of the name is either: no name. The
        // record is still found, and so is a named one after it.
        for len in [0_u8, 8, 10, 0xFF] {
            let mut header = record_bytes("PS_wrong", 0x32, b"misnamed");
            header[4] = len;
            let mut file = MAGIC.to_vec();
            file.extend(&header);
            file.extend(record_bytes("PS_after", 0x41, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [None, Some("PS_after")], "length byte {len}");
            assert_eq!(found[0].kind, 0x32);
            assert_eq!(found[0].unpack().as_deref(), Some(&b"misnamed"[..]));
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn printable_name_that_is_no_identifier_is_still_read() {
        // Punctuation is printable: with no identifier among the length
        // bytes, the earliest one wins, so the whole name is read.
        for name in ["PS-x.y@1", "PS_a+b", "!"] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes(name, 0x32, b"odd"));
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 1, "{name}");
            assert_eq!(found[0].name, Some(name));
            assert_eq!(found[0].unpack().as_deref(), Some(&b"odd"[..]));
        }
    }

    #[test]
    fn name_with_unprintable_bytes_is_no_name() {
        // Bytes that are not printable ASCII (UTF-8 past ASCII, a space, a
        // control byte) cut the name off from its length byte: no name,
        // rather than the tail after them. The record is still found.
        for name in ["PS_\u{e9}t\u{e9}", "PS a", "PS_\tb"] {
            let mut file = MAGIC.to_vec();
            file.extend(record_bytes(name, 0x32, b"unprintable"));
            file.extend(record_bytes("PS_after", 0x32, b"after"));
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [None, Some("PS_after")], "{name:?}");
            assert_eq!(found[0].unpack().as_deref(), Some(&b"unprintable"[..]));
        }
    }

    #[test]
    fn a_name_is_not_read_from_before_the_magic_or_from_the_previous_record() {
        // A record whose name has no length byte of its own. The byte that
        // would state it lies before the magic, or in the previous record's
        // padding past its stream: both belong to something else, so the
        // name is not read across them. Read whole, it would be
        // `SGD5PS_x` or `ABPS_x`.
        let tail = |name: &[u8], payload: &[u8]| {
            let packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
            let n = (packed.len() + 12) as u32;
            let mut out = name.to_vec();
            out.extend_from_slice(&[0, 0x32]);
            for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&packed);
            out
        };
        let mut file = vec![0x09];
        file.extend_from_slice(MAGIC);
        file.extend(tail(b"PS_x", b"after the magic"));
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, None);
        assert_eq!(found[0].unpack().as_deref(), Some(&b"after the magic"[..]));

        // The previous record, padded past its stream with `07 'A' 'B'`.
        let payload = b"padded";
        let mut packed = miniz_oxide::deflate::compress_to_vec(payload, 6);
        packed.extend_from_slice(b"\x07AB");
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x0aPS_padded\0\x32");
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(tail(b"PS_x", b"after a record"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_padded"), None]);
        assert_eq!(found[0].unpack().as_deref(), Some(&payload[..]));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after a record"[..]));
    }

    #[test]
    fn every_kind_byte_is_reported_as_it_is() {
        // The kind is not checked to find a record: any byte, even one that
        // reads as a NUL, a length byte or part of the marker, is reported
        // as it is, and the name before it is still read.
        let kinds = [0x00, 0x01, 0x09, 0x0D, 0x32, 0x7F, 0xDC, 0xE7, 0xFF];
        let mut file = MAGIC.to_vec();
        for &kind in &kinds {
            file.extend(record_bytes("PS_kind", kind, &[kind; 40]));
        }
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), kinds.len());
        for (r, &kind) in found.iter().zip(&kinds) {
            assert_eq!(r.kind, kind);
            assert_eq!(r.name, Some("PS_kind"), "kind {kind:#x}");
            assert_eq!(r.unpack(), Some(vec![kind; 40]), "kind {kind:#x}");
        }
    }

    #[test]
    fn every_name_length_a_byte_can_state_is_read_whole() {
        // Names of 1 to 254 characters: the length byte is a control byte,
        // then punctuation, a digit, a letter, and past 0x7F unprintable.
        // Whichever it is, the name it states is read whole, and the record
        // after it is found with its own name.
        let mut file = MAGIC.to_vec();
        let names: Vec<String> = (1..=254)
            .map(|len| {
                "PS_"
                    .chars()
                    .chain(std::iter::repeat('n'))
                    .take(len)
                    .collect()
            })
            .collect();
        for (i, name) in names.iter().enumerate() {
            file.extend(record_bytes(name, 0x32, &[i as u8; 3]));
        }
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), names.len());
        for (i, (r, name)) in found.iter().zip(&names).enumerate() {
            assert_eq!(r.name, Some(name.as_str()), "{} characters", name.len());
            assert_eq!(r.unpack(), Some(vec![i as u8; 3]));
        }
    }

    #[test]
    fn magic_in_a_name_is_part_of_the_name() {
        // `SGD5` in a record's name, after the file's own magic: the search
        // started at the first magic, so the name is read whole and the
        // records on both sides of it are found.
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_before", 0x32, b"before"));
        file.extend(record_bytes("PS_SGD5_Wrapped", 0x41, b"named SGD5"));
        file.extend(record_bytes("SGD5", 0x32, b"just the magic"));
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [
                Some("PS_before"),
                Some("PS_SGD5_Wrapped"),
                Some("SGD5"),
                Some("PS_after")
            ]
        );
        assert_eq!(found[1].kind, 0x41);
        assert_eq!(found[1].unpack().as_deref(), Some(&b"named SGD5"[..]));
        assert_eq!(found[2].unpack().as_deref(), Some(&b"just the magic"[..]));
    }

    #[test]
    fn records_in_an_archive_are_the_bare_files_shifted() {
        // The same SGD5 file after 0 to 40 bytes of another file (zeros,
        // printable bytes that read as length bytes, bytes of the marker):
        // the same records, with every offset and span moved by as many
        // bytes, and the same names, kinds and containers.
        let mut bare = MAGIC.to_vec();
        bare.extend(record_bytes("VS_Blit_0011aabb_Wrapped", 0x32, b"first"));
        bare.extend(bare_record(0, 0x3c, b"nameless"));
        bare.extend(record_bytes(
            &format!("PS_{}", "x".repeat(40)),
            0x41,
            &[7; 300],
        ));
        let expected: Vec<_> = records(&bare).collect();
        assert_eq!(expected.len(), 3);
        let fills: [&[u8]; 4] = [&[0], b"\x1aPS_x\0", &MARKER, &[0xE7, 0x2A, 0x0D]];
        for fill in fills {
            for skip in 0..=40 {
                let mut archive: Vec<u8> = fill.iter().copied().cycle().take(skip).collect();
                archive.extend_from_slice(&bare);
                let found: Vec<_> = records(&archive).collect();
                assert_eq!(found.len(), expected.len(), "{skip} bytes of {fill:?}");
                for (r, e) in found.iter().zip(&expected) {
                    assert_eq!(r.offset, e.offset + skip);
                    assert_eq!(r.span(), e.span().start + skip..e.span().end + skip);
                    assert_eq!((r.name, r.kind), (e.name, e.kind));
                    assert_eq!(r.unpack(), e.unpack());
                }
            }
        }
    }

    #[test]
    fn containers_packed_at_every_level_are_records() {
        // The same containers packed at every level (0, stored blocks, to
        // 10), sized around the probe and a stored block's 65535 bytes: all
        // are records, with their names, and unpack whole.
        let sizes = [0, 1, PROBE - 1, PROBE, PROBE + 1, 65_535, 65_536, 70_000];
        let mut file = MAGIC.to_vec();
        let mut want = Vec::new();
        for (i, &size) in sizes.iter().enumerate() {
            let container: Vec<u8> = (0..size as u32)
                .map(|j| (j.wrapping_mul(2_654_435_761) >> (i + 9)) as u8)
                .collect();
            for level in 0..=10u8 {
                let packed = miniz_oxide::deflate::compress_to_vec(&container, level);
                let n = (packed.len() + 12) as u32;
                let name = format!("PS_{size}_{level}");
                file.push(name.len() as u8 + 1);
                file.extend_from_slice(name.as_bytes());
                file.extend_from_slice(&[0, 0x32]);
                for v in [n, 0x000D_DCE7, n, size as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
                want.push((name, container.clone()));
            }
        }
        let found: Vec<_> = records(&file).collect();
        assert_eq!(found.len(), want.len());
        for (r, (name, container)) in found.iter().zip(&want) {
            assert_eq!(r.name, Some(name.as_str()));
            assert_eq!(r.unpack().as_ref(), Some(container), "{name}");
        }
    }

    #[test]
    fn whole_sgd5_file_stored_in_a_record_is_that_records() {
        // A record whose container is, stored as is (level 0), a whole SGD5
        // file: its magic and records are the outer record's bytes, so the
        // search neither restarts at that magic nor yields the records; each
        // lies inside the outer record, whose container holds them all.
        let mut inner = MAGIC.to_vec();
        inner.extend(record_bytes("PS_inner_a", 0x32, b"first inside"));
        inner.extend(bare_record(0, 0x3c, b"nameless inside"));
        inner.extend(record_bytes("PS_inner_c", 0x41, &[9; 200]));
        let inside: Vec<_> = records(&inner).map(|r| (r.offset, r.name)).collect();
        assert_eq!(inside.len(), 3);

        let packed = miniz_oxide::deflate::compress_to_vec(&inner, 0);
        let stored = memchr::memmem::find(&packed, &inner).expect("stored as is");
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend(record_bytes("PS_before", 0x32, b"before"));
        // The marker follows the name, NUL, kind and `n`.
        let outer = file.len() + 13 + 4;
        file.extend_from_slice(b"\x0bPS_archive\0\x32");
        for v in [n, 0x000D_DCE7, n, inner.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [Some("PS_before"), Some("PS_archive"), Some("PS_after")]
        );
        assert_eq!(found[1].offset, outer);
        let container = found[1].unpack().expect("the stored file");
        assert_eq!(container, inner);
        let base = outer + 12 + stored;
        assert!(in_record(&found, base), "the stored magic");
        for &(offset, _) in &inside {
            assert!(in_record(&found, base + offset));
        }
        let again: Vec<_> = records(&container).map(|r| (r.offset, r.name)).collect();
        assert_eq!(again, inside);
    }

    #[test]
    fn each_record_is_yielded_before_the_bytes_after_it_are_read() {
        // Two real records, then 32000 chained markers whose stated sizes
        // run past the end. The search is lazy: the first record is yielded
        // having checked no stream beyond its own span's end, and the second
        // likewise, however much of the file is left; only asking for a
        // third walks the chain.
        let mut head = MAGIC.to_vec();
        head.extend(record_bytes("PS_first", 0x32, b"first"));
        head.extend(record_bytes("PS_second", 0x41, &[3; 500]));
        let chain = chained_markers(32_000, |_| MAX_UNPACKED as u32, &[]);
        let mut file = head.clone();
        file.extend_from_slice(&chain[MAGIC.len()..]);

        WORK.with(|w| w.set((0, 0)));
        let mut it = records(&file);
        let first = it.next().expect("PS_first");
        assert_eq!(first.name, Some("PS_first"));
        let second = it.next().expect("PS_second");
        assert_eq!(second.name, Some("PS_second"));
        assert_eq!(second.span().end, head.len());
        let (checks, read) = WORK.with(Cell::get);
        assert!(checks <= 4, "{checks} stream checks for two records");
        assert!(read <= 4 * PROBE, "{read} bytes read for two records");
        assert_eq!(first.unpack().as_deref(), Some(&b"first"[..]));
        assert_eq!(second.unpack(), Some(vec![3; 500]));

        // The chain itself is a truncated record: the search ends there,
        // having now checked its first marker for a stream.
        assert_eq!(it.next(), None);
        let (after, _) = WORK.with(Cell::get);
        assert!(after > checks, "the chain was looked at");
    }

    #[test]
    fn zlib_wrapped_container_is_a_record_that_does_not_unpack() {
        // Packed bytes are raw deflate. The same container with a zlib
        // header and checksum around it is not a stream of the stated size;
        // covering no other record, it is still yielded (with its name and
        // span over all its bytes) and does not unpack, and the raw record
        // after it is found and unpacks.
        let container = b"DXBC wrapped in zlib".repeat(8);
        let wrapped = miniz_oxide::deflate::compress_to_vec_zlib(&container, 6);
        assert_eq!(wrapped[0], 0x78, "zlib header");
        let n = (wrapped.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x08PS_zlib\0\x32");
        let marker = file.len() + 4;
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&wrapped);
        let raw = file.len();
        file.extend(record_bytes("PS_raw", 0x32, &container));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_zlib"), Some("PS_raw")]);
        assert_eq!(found[0].span(), marker..raw);
        assert_eq!(found[0].unpack(), None);
        assert_eq!(found[1].unpack(), Some(container));
    }

    #[test]
    fn stream_split_into_blocks_with_empty_ones_between_is_a_record() {
        // Deflate allows any number of blocks, stored ones of length 0
        // among them (a sync flush writes `00 00 00 FF FF`). A stream of
        // stored blocks built by hand, empty ones between them, ended by a
        // fixed-code block from miniz, is a record: it starts a stream
        // however it is cut, unpacks whole, and the record after it is
        // found.
        fn stored(data: &[u8]) -> Vec<u8> {
            let len = data.len() as u16;
            let mut out = vec![0]; // BFINAL 0, stored
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(data);
            out
        }
        let container = b"DXBC in blocks, then a fixed-code tail".to_vec();
        let (head, tail) = container.split_at(13);
        let mut packed = Vec::new();
        packed.extend(stored(&[]));
        packed.extend(stored(&head[..5]));
        packed.extend(stored(&[]));
        packed.extend(stored(&head[5..]));
        packed.extend(stored(&[]));
        // miniz's stream for the tail ends it: a block after a fixed-code
        // one starts mid-byte, so only stored blocks are put together here.
        let fixed = miniz_oxide::deflate::compress_to_vec(tail, 1);
        assert_eq!(fixed[0] & 0b111, 0b011, "one final fixed-code block");
        packed.extend(fixed);
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&packed).ok(),
            Some(container.clone())
        );
        for cut in 0..packed.len() {
            assert!(
                starts_stream(&packed[..cut], container.len()),
                "cut at {cut}"
            );
        }
        assert!(!starts_stream(&packed, container.len() + 1));

        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x0aPS_blocks\0\x32");
        let marker = file.len() + 4;
        for v in [n, 0x000D_DCE7, n, container.len() as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(&packed);
        let after = file.len();
        file.extend(record_bytes("PS_after", 0x32, b"after"));

        let found: Vec<_> = records(&file).collect();
        let names: Vec<_> = found.iter().map(|r| r.name).collect();
        assert_eq!(names, [Some("PS_blocks"), Some("PS_after")]);
        assert_eq!(found[0].span(), marker..after);
        assert_eq!(found[0].unpack(), Some(container));
        assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
    }

    #[test]
    fn stream_broken_after_a_valid_block_is_noise() {
        // Packed bytes that start a valid stored block and then break (a
        // stored block whose NLEN is not LEN's complement, or a block of
        // the reserved type 3) are no deflate stream: alone, such a record
        // covers no other, so it is yielded and does not unpack; laid over
        // a real record stored in its first block, the real one wins. The
        // record after it is found either way.
        fn stored(data: &[u8]) -> Vec<u8> {
            let len = data.len() as u16;
            let mut out = vec![0]; // BFINAL 0, stored
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(data);
            out
        }
        let bad_nlen = [1, 4, 0, 4, 0, b'x', b'y', b'z', b'w'];
        let reserved = [0b111, 0, 0, 0];
        let inner = record_bytes("PS_inner", 0x41, b"stored inside");
        for (what, tail) in [
            ("bad NLEN", &bad_nlen[..]),
            ("reserved type", &reserved[..]),
        ] {
            for (covering, first) in [(false, &b"DXBC head"[..]), (true, &inner[..])] {
                let mut packed = stored(first);
                packed.extend_from_slice(tail);
                let stated = first.len() + 4;
                assert!(!starts_stream(&packed, stated), "{what}");
                assert!(
                    starts_stream(&packed[..1 + 4 + first.len()], stated),
                    "{what}"
                );

                let n = (packed.len() + 12) as u32;
                let mut file = MAGIC.to_vec();
                file.extend_from_slice(b"\x0aPS_broken\0\x32");
                let marker = file.len() + 4;
                for v in [n, 0x000D_DCE7, n, stated as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(&packed);
                let after = file.len();
                file.extend(record_bytes("PS_after", 0x32, b"after"));

                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                if covering {
                    assert_eq!(names, [Some("PS_inner"), Some("PS_after")], "{what}");
                    let at = marker + 12 + 5 + 4 + 15;
                    assert_eq!(found[0].offset, at, "{what}");
                    assert_eq!(found[0].kind, 0x41, "{what}");
                    assert_eq!(found[0].unpack().as_deref(), Some(&b"stored inside"[..]));
                    assert!(!in_record(&found, marker), "{what}");
                } else {
                    assert_eq!(names, [Some("PS_broken"), Some("PS_after")], "{what}");
                    assert_eq!(found[0].span(), marker..after, "{what}");
                    assert_eq!(found[0].unpack(), None, "{what}");
                }
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            }
        }
    }

    /// A non-final stored deflate block holding `data` as is.
    fn stored_block(data: &[u8]) -> Vec<u8> {
        let len = u16::try_from(data.len()).expect("one stored block");
        let mut out = vec![0]; // BFINAL 0, stored
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    /// A file of the magic, a record named `PS_outer` with these packed
    /// bytes stating `unpacked` bytes, and `PS_after`; and the outer
    /// record's marker offset.
    fn outer_then_after(packed: &[u8], unpacked: usize) -> (Vec<u8>, usize) {
        let n = (packed.len() + 12) as u32;
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(b"\x09PS_outer\0\x32");
        let marker = file.len() + 4;
        for v in [n, 0x000D_DCE7, n, unpacked as u32] {
            file.extend_from_slice(&v.to_le_bytes());
        }
        file.extend_from_slice(packed);
        file.extend(record_bytes("PS_after", 0x32, b"after"));
        (file, marker)
    }

    #[test]
    fn record_split_across_stored_blocks_is_inside_the_one_storing_it() {
        // A container holding a record, stored as is in two stored blocks
        // split at every byte of that record: a block header (5 bytes) may
        // cut its header, its marker, its sizes or its packed bytes. Where
        // the marker survives whole, the record it starts is no stream
        // (its packed bytes are cut) or is the outer one's own: either way
        // the outer record is the one yielded, it unpacks to the container
        // whole, and the record after it is found. Offsets past every
        // record, the end of the file among them, are in none.
        let inner = record_bytes("PS_inner", 0x41, &[9; 64]);
        let mut container = b"DXBC head ".to_vec();
        let inner_at = container.len();
        container.extend_from_slice(&inner);
        container.extend_from_slice(b" DXBC tail");
        for split in inner_at..=inner_at + inner.len() {
            let (head, tail) = container.split_at(split);
            let mut packed = stored_block(head);
            packed.extend(miniz_oxide::deflate::compress_to_vec(tail, 0));
            assert!(starts_stream(&packed, container.len()), "split {split}");

            let (file, marker) = outer_then_after(&packed, container.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "split {split}");
            assert_eq!(found[0].offset, marker, "split {split}");
            assert_eq!(found[0].unpack(), Some(container.clone()), "split {split}");
            assert_eq!(found[1].span().end, file.len(), "split {split}");
            if let Some(at) = memchr::memmem::find(&file[marker + 1..], &MARKER) {
                let at = marker + 1 + at;
                if at < found[0].span().end {
                    assert!(in_record(&found, at), "split {split}");
                }
            }
            for past in [file.len(), file.len() + 1, usize::MAX] {
                assert!(!in_record(&found, past), "split {split}, offset {past}");
            }
        }
    }

    #[test]
    fn stream_broken_past_the_probe_still_hides_the_record_it_stores() {
        // Only the first PROBE packed bytes tell a stream from noise. A
        // record whose stored block (more than a probe long, holding a
        // record past the probe) is followed by a broken block reads as a
        // stream: it wins over the record it stores, is yielded with its
        // whole span, and does not unpack. Broken within the probe, the
        // same bytes are noise and the stored record wins instead.
        let inner = record_bytes("PS_inner", 0x32, b"stored past the probe");
        let bad_nlen = [1, 4, 0, 4, 0, b'x', b'y', b'z', b'w'];
        for lead in [PROBE + 64, PROBE / 4] {
            let mut first: Vec<u8> = (0..lead as u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
                .collect();
            first.extend_from_slice(&inner);
            assert!(memchr::memmem::find(&first[..lead], &MARKER).is_none());
            let mut packed = stored_block(&first);
            packed.extend_from_slice(&bad_nlen);
            let stated = first.len() + 4;
            let broken_at = packed.len() - bad_nlen.len();
            assert_eq!(
                starts_stream(&packed, stated),
                broken_at >= PROBE,
                "lead {lead}"
            );

            let (file, marker) = outer_then_after(&packed, stated);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if broken_at >= PROBE {
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "lead {lead}");
                assert_eq!(found[0].span().end, marker + 12 + packed.len());
                assert_eq!(found[0].unpack(), None, "lead {lead}");
                let stored = marker + 12 + 5 + lead + 19;
                assert_eq!(&file[stored..stored + 4], &MARKER);
                assert!(in_record(&found, stored), "lead {lead}");
            } else {
                assert_eq!(names, [Some("PS_inner"), Some("PS_after")], "lead {lead}");
                assert_eq!(
                    found[0].unpack().as_deref(),
                    Some(&b"stored past the probe"[..])
                );
                assert!(!in_record(&found, marker), "lead {lead}");
            }
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
        }
    }

    #[test]
    fn stated_size_cutting_a_stream_short_ends_the_record_there() {
        // A record's stated size, not its stream, says where it ends. Cut
        // short by that size at every byte, a stream packed at any level is
        // still the start of one: the record is yielded with the span its
        // size states, does not unpack, and the record after it is found in
        // the bytes the stream would have gone on through.
        let container: Vec<u8> = (0..200_u32).map(|i| (i * 7 % 61) as u8).collect();
        for level in [0, 1, 6, 10] {
            let whole = miniz_oxide::deflate::compress_to_vec(&container, level);
            for keep in min_packed(container.len())..=whole.len() {
                let (mut file, marker) = outer_then_after(&whole[..keep], container.len());
                file.splice(
                    marker + 12 + keep..marker + 12 + keep,
                    whole[keep..].to_vec(),
                );
                let what = format!("level {level}, {keep} of {} bytes", whole.len());
                let found: Vec<_> = records(&file).collect();
                let names: Vec<_> = found.iter().map(|r| r.name).collect();
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "{what}");
                assert_eq!(found[0].span(), marker..marker + 12 + keep, "{what}");
                assert_eq!(
                    found[0].unpack(),
                    (keep == whole.len()).then(|| container.clone()),
                    "{what}"
                );
                assert_eq!(found[1].span().end, file.len(), "{what}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            }
        }
    }

    #[test]
    fn record_stored_past_where_a_stated_size_cuts_its_block_is_the_next_record() {
        // A record stored as is in a container whose record states fewer
        // packed bytes than its stored block holds: where the stated end
        // falls at or before the stored record's header, the stored record
        // is the next one, read whole from the bytes after the cut; where it
        // falls inside that header or later, the stored record starts inside
        // the outer one, which is a stream (cut short) and wins.
        let inner = record_bytes("PS_inner", 0x41, b"stored inside");
        let mut container = b"DXBC head ".to_vec();
        let inner_at = container.len();
        container.extend_from_slice(&inner);
        container.extend_from_slice(b" DXBC tail");
        let whole = stored_block(&container);
        // The inner header (NUL, kind, n) is the 6 bytes before its marker.
        let inner_marker = 5 + inner_at + memchr::memmem::find(&inner, &MARKER).unwrap();
        for keep in min_packed(container.len())..whole.len() {
            let (mut file, marker) = outer_then_after(&whole[..keep], container.len());
            let end = marker + 12 + keep;
            file.splice(end..end, whole[keep..].to_vec());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(found[0].name, Some("PS_outer"), "keep {keep}");
            assert_eq!(found[0].span(), marker..end, "keep {keep}");
            assert_eq!(found[0].unpack(), None, "keep {keep}");
            let stored = marker + 12 + inner_marker;
            assert_eq!(&file[stored..stored + 4], &MARKER, "keep {keep}");
            if keep <= inner_marker - 6 {
                assert_eq!(names.len(), 3, "keep {keep}");
                assert_eq!(found[1].offset, stored, "keep {keep}");
                assert_eq!(found[1].kind, 0x41, "keep {keep}");
                assert_eq!(found[1].unpack().as_deref(), Some(&b"stored inside"[..]));
                assert!(!in_record(&found, stored - 1), "keep {keep}");
                if keep < inner_marker - 6 - 13 {
                    // Its length byte and name are past the cut too.
                    assert_eq!(found[1].name, Some("PS_inner"), "keep {keep}");
                }
            } else {
                assert_eq!(names, [Some("PS_outer"), Some("PS_after")], "keep {keep}");
                assert!(!found.iter().any(|r| r.offset == stored), "keep {keep}");
            }
            assert_eq!(found.last().unwrap().name, Some("PS_after"), "keep {keep}");
        }
    }

    #[test]
    fn stream_broken_where_the_probe_ends_is_noise_only_if_the_probe_sees_the_break() {
        // A stored block's header is 5 bytes; a broken one (its NLEN not
        // the complement of its LEN) is told only once all 5 are read. Put
        // just inside the probe, the break makes the outer record noise and
        // the record it stores wins; one byte later, the probe ends inside
        // the broken header, which reads as a stream cut off, and the outer
        // record wins over the one it stores.
        let inner = record_bytes("PS_inner", 0x32, b"stored early");
        let bad_header = [1, 4, 0, 4, 0];
        for broken_at in [PROBE - 6, PROBE - 5, PROBE - 4, PROBE - 1, PROBE] {
            let mut first = b"DXBC head ".to_vec();
            let inner_at = first.len();
            first.extend_from_slice(&inner);
            first.resize(broken_at - 5, b'.');
            let mut packed = stored_block(&first);
            assert_eq!(packed.len(), broken_at);
            packed.extend_from_slice(&bad_header);
            packed.extend_from_slice(b"wxyz");
            let stated = first.len() + 4;
            let seen = broken_at + bad_header.len() <= PROBE;
            assert_eq!(
                starts_stream(&packed, stated),
                !seen,
                "break at {broken_at}"
            );

            let (file, marker) = outer_then_after(&packed, stated);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            let stored =
                marker + 12 + 5 + inner_at + memchr::memmem::find(&inner, &MARKER).unwrap();
            assert_eq!(&file[stored..stored + 4], &MARKER);
            if seen {
                assert_eq!(
                    names,
                    [Some("PS_inner"), Some("PS_after")],
                    "break at {broken_at}"
                );
                assert_eq!(found[0].offset, stored, "break at {broken_at}");
                assert_eq!(found[0].unpack().as_deref(), Some(&b"stored early"[..]));
                assert!(!in_record(&found, marker), "break at {broken_at}");
            } else {
                assert_eq!(
                    names,
                    [Some("PS_outer"), Some("PS_after")],
                    "break at {broken_at}"
                );
                assert_eq!(found[0].span(), marker..marker + 12 + packed.len());
                assert_eq!(found[0].unpack(), None, "break at {broken_at}");
                assert!(in_record(&found, stored), "break at {broken_at}");
            }
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "break at {broken_at}");
        }
    }

    #[test]
    fn stream_of_another_size_over_a_stored_record_is_noise() {
        // A container holding a record, stored as is in one final stored
        // block, under a record stating one byte more or one fewer than the
        // container: its bytes are no stream of that size, so it is noise
        // and the record it stores wins. Stating the container's size
        // exactly, it is a stream and wins over the record it stores.
        let inner = record_bytes("PS_inner", 0x41, b"stored inside");
        let mut container = b"DXBC head ".to_vec();
        let inner_at = container.len();
        container.extend_from_slice(&inner);
        container.extend_from_slice(b" DXBC tail");
        let packed = miniz_oxide::deflate::compress_to_vec(&container, 0);
        for stated in [container.len() - 1, container.len(), container.len() + 1] {
            let (file, marker) = outer_then_after(&packed, stated);
            let stored =
                marker + 12 + 5 + inner_at + memchr::memmem::find(&inner, &MARKER).unwrap();
            assert_eq!(&file[stored..stored + 4], &MARKER, "stated {stated}");
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if stated == container.len() {
                assert_eq!(
                    names,
                    [Some("PS_outer"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].unpack(), Some(container.clone()));
                assert!(in_record(&found, stored), "stated {stated}");
            } else {
                assert_eq!(
                    names,
                    [Some("PS_inner"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].offset, stored, "stated {stated}");
                assert_eq!(found[0].kind, 0x41, "stated {stated}");
                assert_eq!(found[0].unpack().as_deref(), Some(&b"stored inside"[..]));
                assert!(!in_record(&found, marker), "stated {stated}");
            }
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "stated {stated}");
        }
    }

    #[test]
    fn two_records_stored_in_one_container_are_both_its_or_both_yielded() {
        // A container holding two records back to back, stored as is in one
        // final stored block. Stating the container's size, the outer record
        // is a stream and wins over both: neither is yielded, both markers
        // are inside it. Stating one byte more or fewer, it is noise laid
        // over the first stored record, which wins; the search goes on
        // after it and finds the second, then the record after the outer.
        let first = record_bytes("PS_first", 0x41, b"stored first");
        let second = record_bytes("PS_second", 0x3c, b"stored second");
        let mut container = b"DXBC head ".to_vec();
        let first_at = container.len();
        container.extend_from_slice(&first);
        let second_at = container.len();
        container.extend_from_slice(&second);
        container.extend_from_slice(b" DXBC tail");
        let packed = miniz_oxide::deflate::compress_to_vec(&container, 0);
        let marker_in = |r: &[u8]| memchr::memmem::find(r, &MARKER).unwrap();
        for stated in [container.len() - 1, container.len(), container.len() + 1] {
            let (file, marker) = outer_then_after(&packed, stated);
            let stored = [
                marker + 12 + 5 + first_at + marker_in(&first),
                marker + 12 + 5 + second_at + marker_in(&second),
            ];
            for at in stored {
                assert_eq!(&file[at..at + 4], &MARKER, "stated {stated}");
            }
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            if stated == container.len() {
                assert_eq!(
                    names,
                    [Some("PS_outer"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].unpack(), Some(container.clone()));
                for at in stored {
                    assert!(in_record(&found, at), "stated {stated}");
                }
            } else {
                assert_eq!(
                    names,
                    [Some("PS_first"), Some("PS_second"), Some("PS_after")],
                    "stated {stated}"
                );
                assert_eq!(found[0].offset, stored[0], "stated {stated}");
                assert_eq!(found[1].offset, stored[1], "stated {stated}");
                assert_eq!(found[0].kind, 0x41, "stated {stated}");
                assert_eq!(found[1].kind, 0x3c, "stated {stated}");
                // The second starts right where the first one's span ends.
                assert_eq!(found[0].span().end + marker_in(&second), found[1].offset);
                assert_eq!(found[0].unpack().as_deref(), Some(&b"stored first"[..]));
                assert_eq!(found[1].unpack().as_deref(), Some(&b"stored second"[..]));
                assert!(!in_record(&found, marker), "stated {stated}");
            }
            let last = found.last().unwrap();
            assert_eq!(last.unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(last.span().end, file.len(), "stated {stated}");
        }
    }

    /// Bits written least significant first, as deflate packs them.
    #[derive(Default)]
    struct Bits {
        out: Vec<u8>,
        used: u32,
    }

    impl Bits {
        /// `count` bits of `value`, its low bit first (block headers,
        /// extra bits).
        fn put(&mut self, value: u32, count: u32) {
            for i in 0..count {
                if self.used.is_multiple_of(8) {
                    self.out.push(0);
                }
                let bit = (value >> i) & 1;
                *self.out.last_mut().unwrap() |= (bit as u8) << (self.used % 8);
                self.used += 1;
            }
        }

        /// A Huffman code of `count` bits, its high bit first.
        fn code(&mut self, code: u32, count: u32) {
            for i in (0..count).rev() {
                self.put((code >> i) & 1, 1);
            }
        }
    }

    /// One final fixed-code block: the literal `A`, then a copy of 3 bytes
    /// from `distance` (1 to 4) bytes back, then the end of the block.
    fn literal_then_copy(distance: u32) -> Vec<u8> {
        let mut bits = Bits::default();
        bits.put(1, 1); // BFINAL
        bits.put(1, 2); // fixed codes
        bits.code(0x30 + u32::from(b'A'), 8); // literals 0-143: 8 bits from 0x30
        bits.code(1, 7); // length code 257: 3 bytes
        bits.code(distance - 1, 5); // distance codes 0-3: 1-4, no extra bits
        bits.code(0, 7); // end of block (256)
        bits.out
    }

    #[test]
    fn copy_from_before_the_start_of_the_container_is_noise() {
        // A copy may only reach back over bytes already unpacked. One byte
        // back from the first literal is a stream of `AAAA`; two bytes back
        // reaches before the container's start, and the stream breaks
        // there: such a record covers no other, so it is yielded and does
        // not unpack, and the record after it is found either way.
        let good = literal_then_copy(1);
        assert_eq!(
            miniz_oxide::inflate::decompress_to_vec(&good)
                .ok()
                .as_deref(),
            Some(&b"AAAA"[..])
        );
        for distance in [2, 3, 4] {
            let bad = literal_then_copy(distance);
            assert_eq!(bad.len(), good.len());
            assert!(miniz_oxide::inflate::decompress_to_vec(&bad).is_err());
            assert!(!starts_stream(&bad, 4), "distance {distance}");
        }
        assert!(starts_stream(&good, 4));
        assert!(!starts_stream(&good, 3));
        assert!(!starts_stream(&good, 5));

        for distance in [1, 2, 4] {
            let (file, marker) = outer_then_after(&literal_then_copy(distance), 4);
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(
                names,
                [Some("PS_outer"), Some("PS_after")],
                "distance {distance}"
            );
            assert_eq!(found[0].offset, marker, "distance {distance}");
            let want = (distance == 1).then(|| b"AAAA".to_vec());
            assert_eq!(found[0].unpack(), want, "distance {distance}");
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));
            assert_eq!(found[1].span().end, file.len(), "distance {distance}");
        }
    }

    #[test]
    fn stream_without_a_final_block_is_a_record_that_does_not_unpack() {
        // Packed bytes of non-final stored blocks only (a stream left after
        // a sync flush, never finished) hold the whole container but no
        // end: inflating them runs out of bytes, as a cut-off stream does,
        // so they are a record, but one that does not unpack. Its span is
        // what its sizes state, and the record after it is found.
        let container = b"DXBC never finished".to_vec();
        let (head, tail) = container.split_at(7);
        for packed in [
            stored_block(&container),
            [stored_block(head), stored_block(tail)].concat(),
            [stored_block(&container), stored_block(&[])].concat(),
        ] {
            assert!(miniz_oxide::inflate::decompress_to_vec(&packed).is_err());
            assert!(starts_stream(&packed, container.len()));

            let (file, marker) = outer_then_after(&packed, container.len());
            let found: Vec<_> = records(&file).collect();
            let names: Vec<_> = found.iter().map(|r| r.name).collect();
            assert_eq!(names, [Some("PS_outer"), Some("PS_after")]);
            assert_eq!(found[0].span(), marker..marker + 12 + packed.len());
            assert_eq!(found[0].unpack(), None);
            assert_eq!(found[1].unpack().as_deref(), Some(&b"after"[..]));

            // Ended by an empty final stored block, the same bytes unpack.
            let mut ended = packed.clone();
            ended.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
            let (file, _) = outer_then_after(&ended, container.len());
            let found: Vec<_> = records(&file).collect();
            assert_eq!(found.len(), 2);
            assert_eq!(found[0].unpack(), Some(container.clone()));
        }
    }
}
