#![deny(clippy::all)]

pub const BC3_HISTORY_CONFIRMED: &[(u8, u8)] = &[
    (0x00, 0x00),
    (0xFF, 0xFF),
    (0x00, 0xFF),
    (0xFF, 0x00),
    (0x49, 0x92),
    (0x92, 0x49),
];

pub const BC7_HISTORY_CONFIRMED: &[u32] = &[
    0x00000002,
    0x00000040,
    0x00000080,
    0x000000C0,
];

pub fn prob_table(q: u8, sym: u8) -> u16 {
    if q == 0 {
        1
    } else {
        ((sym as u32 * (1u32 << q) + 128) >> 8) as u16
    }
}
