#![deny(clippy::all)]

use crate::dds::DxgiFormat;

pub struct TtpfPack {
    pub version: u32,
    pub entries: Vec<TtpfEntry>,
}

pub struct TtpfEntry {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub slices: u32,
    pub dxgi_format: DxgiFormat,
    pub bcpack_codec: u8,
    pub tile_mode: u32,
    pub d0: u32,
    pub data_offset: u64,
    pub mip_offsets: Vec<u32>,
    pub mip_sizes: Vec<u32>,
}

impl TtpfPack {
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 {
            return Err("too short".into());
        }
        if &data[0..4] != b"RDS5" {
            return Err(format!(
                "bad magic {:?}",
                &data[0..4.min(data.len())]
            ));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let entry_count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let _header_size = u32::from_le_bytes(data[12..16].try_into().unwrap());

        let mut entries = Vec::with_capacity(entry_count);
        let mut off = 16usize;

        for _ in 0..entry_count {
            if off + 64 > data.len() {
                return Err("entry header truncated".into());
            }

            let r = |o: usize| u32::from_le_bytes(data[off + o..off + o + 4].try_into().unwrap());
            let r64 = |o: usize| u64::from_le_bytes(data[off + o..off + o + 8].try_into().unwrap());

            let dxgi_raw = r(0);
            let width = r(4);
            let height = r(8);
            let mip_count = r(12) & 0xFF;
            let slices = (r(12) >> 8) & 0xFF;
            let tile_mode = r(16) & 0xF;
            let d0 = r(16);
            let bcpack_codec = data[off + 20];
            let data_offset = r64(24);

            let mip_table_off = off + 32;
            let max_mips = mip_count.min(16) as usize;
            let mut mip_offsets = vec![0u32; max_mips];
            let mut mip_sizes = vec![0u32; max_mips];
            for m in 0..max_mips {
                if mip_table_off + m * 8 + 8 <= data.len() {
                    mip_offsets[m] = u32::from_le_bytes(
                        data[mip_table_off + m * 8..mip_table_off + m * 8 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    mip_sizes[m] = u32::from_le_bytes(
                        data[mip_table_off + m * 8 + 4..mip_table_off + m * 8 + 8]
                            .try_into()
                            .unwrap(),
                    );
                }
            }

            entries.push(TtpfEntry {
                width: width.max(1),
                height: height.max(1),
                mip_count: mip_count.max(1),
                slices: slices.max(1),
                dxgi_format: DxgiFormat::from_u32(dxgi_raw),
                bcpack_codec,
                tile_mode,
                d0,
                data_offset,
                mip_offsets,
                mip_sizes,
            });

            off += 64 + max_mips * 8;
        }

        Ok(Self { version, entries })
    }
}

impl TtpfEntry {
    pub fn mip_chunk<'a>(&self, pack: &'a [u8], mip: u32) -> Option<&'a [u8]> {
        let m = mip as usize;
        if m >= self.mip_offsets.len() {
            return None;
        }
        let base = self.data_offset as usize;
        let start = base + self.mip_offsets[m] as usize;
        let size = self.mip_sizes[m] as usize;
        pack.get(start..start + size)
    }

    pub fn bytes_per_block(&self) -> u32 {
        match self.dxgi_format {
            DxgiFormat::BC1Unorm => 8,
            DxgiFormat::BC3Unorm | DxgiFormat::BC7Unorm => 16,
            DxgiFormat::Unknown => 16,
        }
    }

    pub fn block_dim(&self) -> u32 {
        4
    }
}
