use std::collections::HashSet;

use super::mode_at_blocks;
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Program};

/// MODE bit 6 enables half/double input denormals. Leave it unknown at
/// entry and if the incoming paths or an SGPR write do not determine it.
fn flush_half_input_after(inst: &Inst, flush: Option<bool>) -> Option<bool> {
    match (inst.name(), &inst.fields) {
        ("s_denorm_mode", Fields::Sopp { imm }) => Some(imm & 4 == 0),
        (
            "s_setreg_imm32_b32" | "s_setreg_b32",
            Fields::Sop {
                src, imm: Some(h), ..
            },
        ) => {
            let (id, off, size) = (h & 0x3F, h >> 6 & 0x1F, (h >> 11 & 0x1F) + 1);
            if id != 1 || off > 6 || off + size <= 6 {
                return flush;
            }
            match src.first() {
                Some(&Operand::Literal(x)) => Some(x >> (6 - off) & 1 == 0),
                _ => None,
            }
        }
        _ => flush,
    }
}

pub(super) fn flush_half_input_sites(p: &Program) -> HashSet<usize> {
    let entries = mode_at_blocks(p, None, flush_half_input_after);
    let mut sites = HashSet::new();
    for (block, mut flush) in p.blocks.iter().zip(entries) {
        for inst in &block.insts {
            if flush == Some(true)
                && matches!(inst.name(), "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16")
            {
                sites.insert(inst.machine.offset);
            }
            flush = flush_half_input_after(inst, flush);
        }
    }
    sites
}
