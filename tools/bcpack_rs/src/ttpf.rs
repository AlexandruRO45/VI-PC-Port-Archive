#![deny(clippy::all)]

use crate::dds::DxgiFormat;

pub struct TtpfEntry {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub dxgi_format: DxgiFormat,
    pub bcpack_codec: u8,
    pub tile_mode: u32,
    pub d0: u32,
    pub slices: u32,
    data_offset: u64,
    data_size: u64,
}

pub struct TtpfPack {
    pub version: u32,
    pub entries: Vec<TtpfEntry>,
}

impl TtpfEntry {
    pub fn bytes_per_block(&self) -> u32 {
        match self.dxgi_format {
            DxgiFormat::BC1Unorm => 8,
            DxgiFormat::BC3Unorm | DxgiFormat::BC7Unorm => 16,
        }
    }

    pub fn block_dim(&self) -> u32 {
        4
    }

    pub fn mip_chunk<'a>(&self, pack: &'a [u8], _mip: u32) -> Option<&'a [u8]> {
        let start = self.data_offset as usize;
        let end = start + self.data_size as usize;
        pack.get(start..end)
    }
}

impl TtpfPack {
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 {
            return Err("TTPF too short".into());
        }
        let magic = &data[0..4];
        if magic != b"RDS5" && magic != b"TTPF" {
            return Err(format!("bad magic: {:?}", magic));
        }
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let entry_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        let mut entries = Vec::with_capacity(entry_count);
        let mut off = 16usize;

        for _ in 0..entry_count {
            if off + 48 > data.len() {
                break;
            }
            let width = u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
            let height = u32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
            let mip_count = u32::from_le_bytes([data[off+8], data[off+9], data[off+10], data[off+11]]);
            let fmt_raw = u32::from_le_bytes([data[off+12], data[off+13], data[off+14], data[off+15]]);
            let codec = data[off+16];
            let tile_mode = u32::from_le_bytes([data[off+20], data[off+21], data[off+22], data[off+23]]);
            let d0 = u32::from_le_bytes([data[off+24], data[off+25], data[off+26], data[off+27]]);
            let slices = u32::from_le_bytes([data[off+28], data[off+29], data[off+30], data[off+31]]);
            let data_offset = u64::from_le_bytes([
                data[off+32], data[off+33], data[off+34], data[off+35],
                data[off+36], data[off+37], data[off+38], data[off+39],
            ]);
            let data_size = u64::from_le_bytes([
                data[off+40], data[off+41], data[off+42], data[off+43],
                data[off+44], data[off+45], data[off+46], data[off+47],
            ]);
            let dxgi_format = match fmt_raw {
                71 => DxgiFormat::BC1Unorm,
                98 => DxgiFormat::BC7Unorm,
                _ => DxgiFormat::BC3Unorm,
            };
            entries.push(TtpfEntry {
                width, height, mip_count, dxgi_format,
                bcpack_codec: codec, tile_mode, d0, slices,
                data_offset, data_size,
            });
            off += 48;
        }

        Ok(TtpfPack { version, entries })
    }
}
