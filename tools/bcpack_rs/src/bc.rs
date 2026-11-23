#![deny(clippy::all)]

use crate::rans::RansDecoder;
use crate::tables::{BC3_HISTORY_CONFIRMED, BC7_HISTORY_CONFIRMED};

pub fn bcpack_decode(chunk: &[u8], codec: u8, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let w_blocks = ((width + 3) / 4).max(1) as usize;
    let h_blocks = ((height + 3) / 4).max(1) as usize;
    let n_blocks = w_blocks * h_blocks;

    match codec {
        0x04 => decode_bc1(chunk, n_blocks),
        0x0C => decode_bc3(chunk, n_blocks),
        0x1C => decode_bc7(chunk, n_blocks),
        other => Err(format!("unsupported BCPack codec 0x{other:02x}")),
    }
}

fn decode_bc1(chunk: &[u8], n_blocks: usize) -> Result<Vec<u8>, String> {
    let q = chunk.first().copied().unwrap_or(0) & 0xF;
    let mut dec = RansDecoder::new(&chunk[1..], q)?;
    let mut out = vec![0u8; n_blocks * 8];
    dec.decode_block(&mut out);
    Ok(out)
}

fn decode_bc3(chunk: &[u8], n_blocks: usize) -> Result<Vec<u8>, String> {
    let q = chunk.first().copied().unwrap_or(0) & 0xF;
    let hist_len = BC3_HISTORY_CONFIRMED.len().min(8);
    let mut dec = RansDecoder::new(&chunk[1..], q)?;
    let mut out = vec![0u8; n_blocks * 16];
    dec.decode_block(&mut out);
    let _ = hist_len;
    Ok(out)
}

fn decode_bc7(chunk: &[u8], n_blocks: usize) -> Result<Vec<u8>, String> {
    let q = chunk.first().copied().unwrap_or(0) & 0xF;
    let hist_len = BC7_HISTORY_CONFIRMED.len().min(8);
    let mut dec = RansDecoder::new(&chunk[1..], q)?;
    let mut out = vec![0u8; n_blocks * 16];
    dec.decode_block(&mut out);
    let _ = hist_len;
    Ok(out)
}
