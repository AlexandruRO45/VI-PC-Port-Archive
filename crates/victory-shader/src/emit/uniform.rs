//! Which VGPR values are the same in every lane of a console wave, for a console wave64
//! program on a host of 32-lane waves: such a value read with `v_readfirstlane` needs no
//! exchange between the two host waves.

use std::collections::HashSet;

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Program, Reg, Value};
use crate::structure::Structured;

/// VGPR values that are the same in every lane holding them (`all` is `lower.rs`'s `all_lane_blocks`):
/// written by an instruction every arriving lane runs (no guard), lane by lane, from scalars,
/// constants and such values; or loaded from an address made of them. The compiler keeps
/// wave-uniform data in VGPRs and fetches it with `v_readfirstlane`.
/// shortcut: a value joined by a phi counts only where every predecessor is a block all lanes
/// run (no lane branch can have split the arriving lanes); loop-carried values never do.
pub(super) fn lane_uniform(p: &Program, st: &Structured, all: &[bool], _threads: Option<[u32; 3]>) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut same: HashSet<Value> = HashSet::new();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        let whole = block.preds.iter().all(|&q| all.get(q).copied().unwrap_or(false));
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) && whole && phi.args.iter().all(|a| same.contains(a)) {
                same.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            let ops: Vec<Operand> = match &inst.fields {
                Fields::Valu(v)
                    if v.dpp.is_none()
                        && v.sdwa.is_none()
                        && !name.starts_with("v_mbcnt")
                        && !name.starts_with("v_cmpx")
                        && !name.contains("lane") =>
                {
                    v.src.clone()
                }
                _ => continue,
            };
            if st.guards[b][i].is_some() {
                continue; // lanes outside EXEC keep another value
            }
            let alike = ops.iter().all(|op| match *op {
                Operand::Vector { reg, .. } => {
                    // every dword of the operand the instruction reads from that register on
                    let read: Vec<Value> =
                        inst.reads.iter().filter(|r| matches!(r.0, Reg::V(x) if x >= reg)).map(|r| r.1).collect();
                    !read.is_empty() && read.iter().all(|x| same.contains(x))
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                Operand::Scalar { code, .. } if code != NULL => {
                    inst.read(Reg::S(code)).is_none_or(|x| !is_mask(x) || same.contains(&x))
                }
                _ => true,
            });
            if alike {
                same.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
            }
        }
    }
    same
}

#[cfg(test)]
mod tests {
    use crate::ir::Reg;
    use crate::{ir, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// A write under a lane guard leaves the other lanes their old value: the result is alike
    /// in every lane only if the old value was (ryotdev's correction to the proof).
    #[test]
    fn a_guarded_uniform_write_can_keep_different_values_in_each_lane() {
        let check = |initial: u32, write: &[u32]| {
            let mut words = vec![
                initial,     // v_mov_b32 v1, v0 (thread ID) or 0
                0x7E04_0280, // v_mov_b32 v2, 0
                0x7D88_0400, // v_cmp_gt_u32 vcc, s0, v2
                0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            ];
            words.extend_from_slice(write);
            words.extend_from_slice(&[
                0xBEFE_040C, // s_mov_b64 exec, s[12:13]
                0x7E0E_0501, // v_readfirstlane_b32 s7, v1
                S_ENDPGM,
            ]);
            let code: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = ir::build(&code, true, 8).unwrap();
            let st = structure::structure(&p).unwrap();
            let all = super::super::lower::all_lane_blocks(&st).unwrap();
            let uniform = super::lane_uniform(&p, &st, &all, Some([64, 1, 1]));
            let read = p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.name() == "v_readfirstlane_b32")
                .unwrap();
            uniform.contains(&read.read(Reg::V(1)).unwrap())
        };

        let writes: &[(&str, &[u32])] = &[
            ("v_mov_b32 v1, 1", &[0x7E02_0281]),
            ("v_lshrrev_b32 v1, 6, v0", &[0x2C02_0086]),
            ("ds_read_b32 v1, v2", &[0xD8D8_0000, 0x0100_0002]),
        ];
        let mut divergent = Vec::new();
        for &(name, write) in writes {
            assert!(check(0x7E02_0280, write), "{name}: old and new values are uniform");
            if check(0x7E02_0300, write) {
                divergent.push(name);
            }
        }
        assert!(divergent.is_empty(), "when s0 is zero, lanes keep their IDs: {divergent:?}");
    }
}
