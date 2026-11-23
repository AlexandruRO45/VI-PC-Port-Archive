#![deny(clippy::all)]

#[derive(Debug, Clone, Copy)]
pub enum DxgiFormat {
    BC1Unorm = 71,
    BC3Unorm = 77,
    BC7Unorm = 98,
}

pub fn write_dds(width: u32, height: u32, fmt: DxgiFormat, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(148 + data.len());
    out.extend_from_slice(b"DDS ");
    let pitch = match fmt {
        DxgiFormat::BC1Unorm => ((width + 3) / 4) * 8,
        DxgiFormat::BC3Unorm | DxgiFormat::BC7Unorm => ((width + 3) / 4) * 16,
    };
    push_u32(&mut out, 124);
    push_u32(&mut out, 0x1 | 0x2 | 0x4 | 0x1000);
    push_u32(&mut out, height);
    push_u32(&mut out, width);
    push_u32(&mut out, pitch);
    push_u32(&mut out, 1);
    push_u32(&mut out, 1);
    for _ in 0..11 { push_u32(&mut out, 0); }
    push_u32(&mut out, 32);
    push_u32(&mut out, 0x4);
    out.extend_from_slice(b"DX10");
    for _ in 0..5 { push_u32(&mut out, 0); }
    push_u32(&mut out, 0x1000);
    push_u32(&mut out, 0);
    push_u32(&mut out, 0);
    push_u32(&mut out, fmt as u32);
    push_u32(&mut out, 3);
    push_u32(&mut out, 0);
    push_u32(&mut out, 1);
    push_u32(&mut out, 0);
    out.extend_from_slice(data);
    out
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
