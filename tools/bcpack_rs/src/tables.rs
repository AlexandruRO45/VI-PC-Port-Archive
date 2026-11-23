#![deny(clippy::all)]

pub const RANS_SCALE: u32 = 65536;
pub const RANS_LOWER: u32 = 1 << 23;

pub fn prob_table(symbol_width: u32, q: u32) -> &'static [u16] {
    let idx = (q * 10 + symbol_width.saturating_sub(1)) as usize;
    PROB_FLAT.get(idx * 256..).and_then(|s| s.get(..256)).unwrap_or(&UNIFORM_256)
}

static UNIFORM_256: [u16; 256] = {
    [256u16; 256]
};

static PROB_FLAT: [u16; 0] = [];

pub fn prob_entry_for_sym(sym_width: u32, q: u32, sym: u32) -> (u32, u32) {
    let n_sym = 1u32 << sym_width;
    if q == 0 {
        let freq = RANS_SCALE / n_sym;
        let cumul = sym * freq;
        return (freq, cumul);
    }

    let alpha = q as f64 / 15.0;
    let base = ((1.0 - alpha) / n_sym as f64) + alpha * (-((sym as f64 + 0.5) / n_sym as f64).ln() / (n_sym as f64).ln());
    let freq = ((base * RANS_SCALE as f64).round() as u32).max(1).min(RANS_SCALE - sym);
    let cumul = (0..sym).map(|s| prob_entry_for_sym(sym_width, q, s).0).sum::<u32>();
    (freq, cumul)
}

pub struct CdfTable {
    pub freqs: Vec<u16>,
    pub cumuls: Vec<u32>,
    pub sym_map: Vec<u16>,
}

impl CdfTable {
    pub fn build(sym_width: u32, q: u32) -> Self {
        let n = 1usize << sym_width;
        let mut freqs = vec![0u16; n];
        let mut cumuls = vec![0u32; n + 1];
        let mut total = 0u32;

        if q == 0 {
            let f = (RANS_SCALE / n as u32) as u16;
            for i in 0..n {
                freqs[i] = f;
                cumuls[i] = i as u32 * f as u32;
                total += f as u32;
            }
            cumuls[n] = total;
        } else {
            let alpha = q as f64 / 15.0;
            let mut rem = RANS_SCALE;
            for i in 0..n {
                let t = n - i;
                let ideal = if t == 1 {
                    rem
                } else {
                    let w = 1.0 - alpha * (i as f64 / (n - 1) as f64);
                    ((rem as f64 * w / t as f64).round() as u32).max(1).min(rem - (t as u32 - 1))
                };
                freqs[i] = ideal as u16;
                cumuls[i] = total;
                total += ideal;
                rem -= ideal;
            }
            cumuls[n] = total;
        }

        let mut sym_map = vec![0u16; RANS_SCALE as usize];
        for sym in 0..n {
            let lo = cumuls[sym] as usize;
            let hi = cumuls[sym + 1] as usize;
            for c in lo..hi {
                if c < sym_map.len() {
                    sym_map[c] = sym as u16;
                }
            }
        }

        Self { freqs, cumuls, sym_map }
    }
}

pub const BC3_HISTORY_CONFIRMED: &[(u16, u16)] = &[
    (0x0000, 0x0000),
    (0x00ff, 0x00ff),
    (0xff00, 0xff00),
    (0xffff, 0xffff),
    (0x0000, 0xffff),
    (0xffff, 0x0000),
];

pub const BC7_HISTORY_CONFIRMED: &[u32] = &[
    0x00000000,
    0xffffffff,
    0x00ff00ff,
    0xff00ff00,
];
