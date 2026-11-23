#![deny(clippy::all)]

use crate::rans::decode_stream;

pub const CODEC_BC1: u8 = 0x04;
pub const CODEC_BC3: u8 = 0x0C;
pub const CODEC_BC7: u8 = 0x1C;

pub fn bcpack_decode(chunk: &[u8], codec: u8, width: u32, height: u32) -> Result<Vec<u8>, String> {
    match codec {
        CODEC_BC1 => decode_bc1(chunk, width, height),
        CODEC_BC3 => decode_bc3(chunk, width, height),
        CODEC_BC7 => decode_bc7(chunk, width, height),
        other => Err(format!("unsupported BCPack codec 0x{other:02x}")),
    }
}

fn block_count(width: u32, height: u32) -> usize {
    let bw = width.div_ceil(4) as usize;
    let bh = height.div_ceil(4) as usize;
    bw * bh
}

fn decode_bc1(chunk: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let n = block_count(width, height);

    if chunk.len() < 4 {
        return Err("BC1 chunk too short for header".into());
    }
    let sym_width = chunk[0] as u32;
    let q = chunk[1] as u32;
    let _reserved = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
    let payload = &chunk[4..];

    let n_syms = n * 4;
    let syms = decode_stream(payload, sym_width, q, n_syms)?;

    let mut out = vec![0u8; n * 8];
    for (blk, s) in syms.chunks(4).enumerate() {
        let base = blk * 8;
        if base + 8 > out.len() {
            break;
        }
        let ep0 = s[0];
        let ep1 = s[1];
        let idx0 = s[2];
        let idx1 = s[3];
        out[base..base + 2].copy_from_slice(&ep0.to_le_bytes());
        out[base + 2..base + 4].copy_from_slice(&ep1.to_le_bytes());
        out[base + 4..base + 6].copy_from_slice(&idx0.to_le_bytes());
        out[base + 6..base + 8].copy_from_slice(&idx1.to_le_bytes());
    }
    Ok(out)
}

fn decode_bc3(chunk: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let n = block_count(width, height);

    if chunk.len() < 4 {
        return Err("BC3 chunk too short for header".into());
    }
    let sym_width = chunk[0] as u32;
    let q = chunk[1] as u32;
    let _flags = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
    let payload = &chunk[4..];

    let n_syms = n * 8;
    let syms = decode_stream(payload, sym_width, q, n_syms)?;

    let mut out = vec![0u8; n * 16];
    for (blk, s) in syms.chunks(8).enumerate() {
        let base = blk * 16;
        if base + 16 > out.len() {
            break;
        }
        out[base] = s[0] as u8;
        out[base + 1] = s[1] as u8;
        let alpha_bits: u64 = (s[2] as u64)
            | ((s[3] as u64) << 16)
            | ((s[4] as u64) << 32);
        out[base + 2..base + 8].copy_from_slice(&alpha_bits.to_le_bytes()[..6]);
        let ep0 = s[5];
        let ep1 = s[6];
        let color_idx = s[7];
        out[base + 8..base + 10].copy_from_slice(&ep0.to_le_bytes());
        out[base + 10..base + 12].copy_from_slice(&ep1.to_le_bytes());
        out[base + 12..base + 14].copy_from_slice(&color_idx.to_le_bytes());
    }
    Ok(out)
}

fn decode_bc7(chunk: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let n = block_count(width, height);

    if chunk.len() < 4 {
        return Err("BC7 chunk too short for header".into());
    }
    let sym_width = chunk[0] as u32;
    let q = chunk[1] as u32;
    let _flags = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
    let payload = &chunk[4..];

    let n_syms = n * 8;
    let syms = decode_stream(payload, sym_width, q, n_syms)?;

    let mut out = vec![0u8; n * 16];
    for (blk, s) in syms.chunks(8).enumerate() {
        let base = blk * 16;
        if base + 16 > out.len() {
            break;
        }
        let lo: u64 = (s[0] as u64)
            | ((s[1] as u64) << 16)
            | ((s[2] as u64) << 32)
            | ((s[3] as u64) << 48);
        let hi: u64 = (s[4] as u64)
            | ((s[5] as u64) << 16)
            | ((s[6] as u64) << 32)
            | ((s[7] as u64) << 48);
        out[base..base + 8].copy_from_slice(&lo.to_le_bytes());
        out[base + 8..base + 16].copy_from_slice(&hi.to_le_bytes());
    }
    Ok(out)
}
