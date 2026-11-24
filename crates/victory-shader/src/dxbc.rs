//! The DXBC container: a header, then a table of FourCC-tagged parts.
//!
//! ```text
//! 0x00  "DXBC"
//! 0x04  digest[16]
//! 0x14  version major u16, minor u16
//! 0x18  total size u32
//! 0x1C  part count u32
//! 0x20  part offsets u32[count]
//! part: fourcc[4], size u32, data[size]
//! ```

use crate::ShaderError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const DXIL: FourCC = FourCC(*b"DXIL");
    pub const STAT: FourCC = FourCC(*b"STAT");
    pub const RTS0: FourCC = FourCC(*b"RTS0");
    pub const PSV0: FourCC = FourCC(*b"PSV0");
    pub const ISG1: FourCC = FourCC(*b"ISG1");
    pub const OSG1: FourCC = FourCC(*b"OSG1");
    /// Per-primitive outputs of a mesh shader.
    pub const PSG1: FourCC = FourCC(*b"PSG1");
    pub const SFI0: FourCC = FourCC(*b"SFI0");
    pub const HASH: FourCC = FourCC(*b"HASH");
    pub const ILDB: FourCC = FourCC(*b"ILDB");
    pub const ILDN: FourCC = FourCC(*b"ILDN");

    /// The parts DXC knows. Its validator refuses a container that lists one
    /// of them twice ("More than one ... part"); other FourCCs are a title's
    /// own and may repeat.
    const KNOWN: [FourCC; 11] = [
        Self::DXIL,
        Self::STAT,
        Self::RTS0,
        Self::PSV0,
        Self::ISG1,
        Self::OSG1,
        Self::PSG1,
        Self::SFI0,
        Self::HASH,
        Self::ILDB,
        Self::ILDN,
    ];
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &b in &self.0 {
            let c = if b.is_ascii_graphic() { b as char } else { '.' };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub fourcc: FourCC,
    /// Offset of the part header within the container.
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Container<'a> {
    pub digest: [u8; 16],
    pub version: (u16, u16),
    pub parts: Vec<Part<'a>>,
}

const HEADER_SIZE: usize = 0x20;
const MAX_PARTS: usize = 256;

impl<'a> Container<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < HEADER_SIZE || &data[0..4] != b"DXBC" {
            return Err(ShaderError::NotDxbc);
        }
        let digest = data[4..20].try_into().expect("16 bytes");
        let version = (u16_at(data, 0x14)?, u16_at(data, 0x16)?);
        let total = u32_at(data, 0x18)? as usize;
        if total > data.len() || total < HEADER_SIZE {
            return Err(ShaderError::Malformed("container size disagrees with data"));
        }
        let data = &data[..total];
        let count = u32_at(data, 0x1C)? as usize;
        if count > MAX_PARTS {
            return Err(ShaderError::Malformed("implausible part count"));
        }
        // Parts follow the offset table; one inside it would read the
        // header or the offsets as a part.
        let table_end = HEADER_SIZE + 4 * count;
        let mut parts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = u32_at(data, HEADER_SIZE + 4 * i)? as usize;
            if offset < table_end {
                return Err(ShaderError::Malformed("part overlaps the container header"));
            }
            let fourcc = FourCC(
                data.get(offset..offset + 4)
                    .ok_or(ShaderError::Malformed("part header out of bounds"))?
                    .try_into()
                    .expect("4 bytes"),
            );
            let size = u32_at(data, offset + 4)? as usize;
            let body = data
                .get(offset + 8..offset + 8 + size)
                .ok_or(ShaderError::Malformed("part data out of bounds"))?;
            // `part` returns the first of a FourCC, so a second `ISG1` or
            // `DXIL` was silently ignored while the host, the validator or a
            // tool reading the container could take the other one.
            if FourCC::KNOWN.contains(&fourcc) && parts.iter().any(|p: &Part| p.fourcc == fourcc) {
                return Err(ShaderError::Malformed("a part listed twice"));
            }
            // A part whose bytes another part also claims reads that part's
            // header or data as its own; DXC lays parts end to end.
            let end = offset + 8 + size;
            if parts
                .iter()
                .any(|p: &Part| offset < p.offset + 8 + p.data.len() && p.offset < end)
            {
                return Err(ShaderError::Malformed("two parts overlap"));
            }
            parts.push(Part {
                fourcc,
                offset,
                data: body,
            });
        }
        Ok(Container {
            digest,
            version,
            parts,
        })
    }

    pub fn part(&self, fourcc: FourCC) -> Option<&Part<'a>> {
        self.parts.iter().find(|p| p.fourcc == fourcc)
    }
}

pub(crate) fn u16_at(data: &[u8], at: usize) -> Result<u16, ShaderError> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("2 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u64_at(data: &[u8], at: usize) -> Result<u64, ShaderError> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

/// Builds a container from parts (test helper).
#[cfg(test)]
pub fn build(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let table = HEADER_SIZE + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_parts() {
        let bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4]), (FourCC::HASH, &[9; 20])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        assert_eq!(c.part(FourCC::HASH).unwrap().data.len(), 20);
        assert!(c.part(FourCC::DXIL).is_none());
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Container::parse(b"nope").unwrap_err(), ShaderError::NotDxbc);
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let last = bytes.len() - 1;
        bytes[0x20] = 0xFF; // corrupt the first part offset
        assert!(Container::parse(&bytes).is_err());
        bytes.truncate(last);
        assert!(Container::parse(&bytes).is_err());
    }

    fn malformed(bytes: &[u8]) -> &'static str {
        match Container::parse(bytes).unwrap_err() {
            ShaderError::Malformed(why) => why,
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn rejects_sizes_that_disagree_with_the_data() {
        let bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let mut larger = bytes.clone();
        larger[0x18..0x1C].copy_from_slice(&(bytes.len() as u32 + 1).to_le_bytes());
        assert_eq!(malformed(&larger), "container size disagrees with data");
        let mut smaller = bytes.clone();
        smaller[0x18..0x1C].copy_from_slice(&(HEADER_SIZE as u32 - 1).to_le_bytes());
        assert_eq!(malformed(&smaller), "container size disagrees with data");
        // Bytes past the stated size are not part of the container.
        let mut padded = bytes.clone();
        padded.extend_from_slice(&[0xAA; 16]);
        assert_eq!(Container::parse(&padded).unwrap().parts.len(), 1);
    }

    #[test]
    fn rejects_implausible_part_counts() {
        let mut bytes = build(&[]);
        bytes[0x1C..0x20].copy_from_slice(&(MAX_PARTS as u32 + 1).to_le_bytes());
        assert_eq!(malformed(&bytes), "implausible part count");
        // A count whose offset table runs past the container.
        bytes[0x1C..0x20].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(malformed(&bytes), "truncated");
    }

    #[test]
    fn rejects_parts_out_of_bounds() {
        let bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let mut header = bytes.clone();
        header[0x20..0x24].copy_from_slice(&(bytes.len() as u32 - 2).to_le_bytes());
        assert_eq!(malformed(&header), "part header out of bounds");
        let mut size = bytes.clone();
        size[0x28..0x2C].copy_from_slice(&9u32.to_le_bytes());
        assert_eq!(malformed(&size), "part data out of bounds");
    }

    #[test]
    fn rejects_parts_inside_the_header() {
        let bytes = build(&[(FourCC::SFI0, &[0; 8]), (FourCC::HASH, &[0; 4])]);
        // Offset 0 would read "DXBC" as a part; the offset table's own
        // second entry is no part either.
        for offset in [0u32, 0x10, HEADER_SIZE as u32 + 4] {
            let mut bad = bytes.clone();
            bad[0x20..0x24].copy_from_slice(&offset.to_le_bytes());
            assert_eq!(
                malformed(&bad),
                "part overlaps the container header",
                "offset {offset:#x}"
            );
        }
        // The first byte after the table is where parts start.
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts[0].offset, HEADER_SIZE + 8);
    }

    #[test]
    fn overlapping_parts_are_refused() {
        // A title's own FourCC first, so the same offset twice is not
        // refused as a known part listed twice.
        let own = FourCC(*b"PRIV");
        let bytes = build(&[(own, &[0; 8]), (FourCC::HASH, &[0; 4])]);
        let first = HEADER_SIZE as u32 + 8;
        let second = first + 16;
        // The second part starting inside the first's header, inside its
        // data, or on its very offset.
        for offset in [first, first + 4, first + 8] {
            let mut bad = bytes.clone();
            bad[0x24..0x28].copy_from_slice(&offset.to_le_bytes());
            assert_eq!(malformed(&bad), "two parts overlap", "offset {offset:#x}");
        }
        // The first part's size growing over the second's header.
        let mut grown = bytes.clone();
        grown[first as usize + 4..first as usize + 8].copy_from_slice(&9u32.to_le_bytes());
        assert_eq!(malformed(&grown), "two parts overlap");
        // Listed in reverse order, end to end, the parts still parse.
        let mut swapped = bytes.clone();
        swapped[0x20..0x24].copy_from_slice(&second.to_le_bytes());
        swapped[0x24..0x28].copy_from_slice(&first.to_le_bytes());
        let c = Container::parse(&swapped).unwrap();
        assert_eq!(c.parts[0].fourcc, FourCC::HASH);
        assert_eq!(c.parts[1].fourcc, own);
        // A gap between parts is not an overlap.
        let mut gap = bytes.clone();
        gap[first as usize + 4..first as usize + 8].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(Container::parse(&gap).unwrap().parts[0].data.len(), 4);
    }

    #[test]
    fn a_known_part_listed_twice_is_refused() {
        for fourcc in FourCC::KNOWN {
            let bytes = build(&[
                (fourcc, &[1; 4]),
                (FourCC::SFI0, &[0; 8]),
                (fourcc, &[2; 4]),
            ]);
            assert_eq!(malformed(&bytes), "a part listed twice", "{fourcc}");
        }
        // Each known part once, and a title's own part twice, still parse.
        let once: Vec<_> = FourCC::KNOWN.iter().map(|&f| (f, &[0u8; 4][..])).collect();
        assert_eq!(Container::parse(&build(&once)).unwrap().parts.len(), 11);
        let own = FourCC(*b"PRIV");
        let twice = build(&[(own, &[1; 4]), (FourCC::DXIL, &[0; 4]), (own, &[2; 4])]);
        let c = Container::parse(&twice).unwrap();
        assert_eq!(c.parts.len(), 3);
        assert_eq!(c.part(own).unwrap().data, &[1; 4]);
    }
}
