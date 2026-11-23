#![deny(clippy::all)]

use crate::tables::{CdfTable, RANS_LOWER, RANS_SCALE};

pub struct RansDecoder<'a> {
    state: [u32; 2],
    src: &'a [u8],
    pos: usize,
}

impl<'a> RansDecoder<'a> {
    pub fn new(src: &'a [u8]) -> Result<Self, String> {
        if src.len() < 8 {
            return Err("rANS stream too short".into());
        }
        let s0 = u32::from_le_bytes(src[0..4].try_into().unwrap());
        let s1 = u32::from_le_bytes(src[4..8].try_into().unwrap());
        Ok(Self { state: [s0, s1], src, pos: 8 })
    }

    fn refill(&mut self, lane: usize) {
        while self.state[lane] < RANS_LOWER && self.pos + 2 <= self.src.len() {
            let w = u16::from_le_bytes(self.src[self.pos..self.pos + 2].try_into().unwrap());
            self.state[lane] = (self.state[lane] << 16) | w as u32;
            self.pos += 2;
        }
    }

    pub fn decode_sym(&mut self, lane: usize, table: &CdfTable) -> u16 {
        self.refill(lane);
        let slot = (self.state[lane] & (RANS_SCALE - 1)) as usize;
        let sym = if slot < table.sym_map.len() {
            table.sym_map[slot] as usize
        } else {
            0
        };
        let freq = table.freqs[sym] as u32;
        let cumul = table.cumuls[sym];
        self.state[lane] = freq * (self.state[lane] >> 16) + (self.state[lane] & (RANS_SCALE - 1)) - cumul;
        sym as u16
    }
}

pub fn decode_stream(src: &[u8], sym_width: u32, q: u32, n_syms: usize) -> Result<Vec<u16>, String> {
    let table = CdfTable::build(sym_width, q);
    let mut dec = RansDecoder::new(src)?;
    let mut out = Vec::with_capacity(n_syms);
    for i in 0..n_syms {
        out.push(dec.decode_sym(i & 1, &table));
    }
    Ok(out)
}
