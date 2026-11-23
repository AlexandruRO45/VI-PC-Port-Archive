#![deny(clippy::all)]

use crate::tables::prob_table;

pub struct RansDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    state0: u32,
    state1: u32,
    q: u8,
}

impl<'a> RansDecoder<'a> {
    pub fn new(data: &'a [u8], q: u8) -> Result<Self, String> {
        if data.len() < 8 {
            return Err(format!("rANS stream too short: {}", data.len()));
        }
        let state0 = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let state1 = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        Ok(Self { data, pos: 8, state0, state1, q })
    }

    fn renorm_state(mut s: u32, data: &[u8], pos: &mut usize) -> u32 {
        let mut p = *pos;
        while s < (1 << 23) && p + 1 < data.len() {
            let lo = data[p] as u32;
            let hi = data[p + 1] as u32;
            s = (s << 16) | (hi << 8) | lo;
            p += 2;
        }
        let pos_slot = std::slice::from_mut(pos);
        pos_slot[0] = p;
        s
    }

    fn advance(&self, state: u32, sym: u8) -> u32 {
        let freq = prob_table(self.q, sym).max(1) as u32;
        let bias = (sym as u32 * (1u32 << self.q)) >> 8;
        (state / 256) * freq + (state & 0xFF) as u32 - bias
    }

    pub fn decode_sym(&mut self) -> u8 {
        let sym0 = (self.state0 & 0xFF) as u8;
        let next0 = self.advance(self.state0, sym0);
        self.state0 = Self::renorm_state(next0, self.data, &mut self.pos);

        let sym1 = (self.state1 & 0xFF) as u8;
        let next1 = self.advance(self.state1, sym1);
        self.state1 = Self::renorm_state(next1, self.data, &mut self.pos);

        sym0
    }

    pub fn decode_block(&mut self, out: &mut [u8]) {
        let len = out.len();
        let decoded: Vec<u8> = (0..len).map(|_| self.decode_sym()).collect();
        out.copy_from_slice(&decoded);
    }
}
