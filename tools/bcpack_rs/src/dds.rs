#![deny(clippy::all)]

#[derive(Debug, Clone, Copy)]
pub enum DxgiFormat {
    BC1Unorm = 71,
    BC3Unorm = 77,
    BC7Unorm = 98,
    Unknown = 0,
}

impl DxgiFormat {
    pub fn from_u32(v: u32) -> Self {
        match v {
            71 => Self::BC1Unorm,
            77 => Self::BC3Unorm,
            98 => Self::BC7Unorm,
            _ => Self::Unknown,
        }
    }
}

pub fn write_dds(width: u32, height: u32, fmt: DxgiFormat, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(148 + data.len());

    out.extend_from_slice(b"DDS ");

    let mut header = [0u8; 124];
    write_u32(&mut header, 0, 124);
    write_u32(&mut header, 4, 0x00001007);
    write_u32(&mut header, 8, height);
    write_u32(&mut header, 12, width);
    write_u32(&mut header, 16, data.len() as u32);
    write_u32(&mut header, 20, 1);
    write_u32(&mut header, 24, 1);

    write_u32(&mut header, 72, 32);
    write_u32(&mut header, 76, 0x00000004);
    header[80..84].copy_from_slice(b"DX10");

    write_u32(&mut header, 108, 0x00401008);

    out.extend_from_slice(&header);

    let mut dx10 = [0u8; 20];
    write_u32(&mut dx10, 0, fmt as u32);
    write_u32(&mut dx10, 4, 3);
    write_u32(&mut dx10, 8, 0);
    write_u32(&mut dx10, 12, 1);
    write_u32(&mut dx10, 16, 0);
    out.extend_from_slice(&dx10);

    out.extend_from_slice(data);
    out
}

fn write_u32(buf: &mut [u8], off: usize, val: u32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}
