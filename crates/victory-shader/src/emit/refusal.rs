use std::collections::{HashMap, HashSet, VecDeque};

use crate::io::{Interp, Io};
use crate::ir::{Inst, Program, Value};
use crate::native::RootParam;
use crate::resource::{Resources, Site};

enum Definition<'a> {
    Instruction(Site, &'a Inst),
    Phi(&'a [Value]),
}

/// Candidate producers of a refused value, summarized without code bytes.
/// The error offset can be the eventual consumer rather than the producer.
pub fn refusal_sources(
    program: &Program,
    io: &Io,
    resources: &Resources,
    layout: &[RootParam],
    offset: usize,
    reason: &str,
) -> Option<Vec<String>> {
    let interpolation = match reason {
        "the first half of an interpolation" => true,
        "a descriptor read only as data" | "a descriptor's address or format read as data" => false,
        _ => return None,
    };
    let producer = |inst: &Inst| {
        if interpolation {
            inst.name().starts_with("v_interp_p1")
        } else {
            inst.name().starts_with("s_load_dword")
        }
    };
    let mut defs = HashMap::new();
    let mut queue = VecDeque::new();
    let mut sources = Vec::new();
    for (b, block) in program.blocks.iter().enumerate() {
        for phi in &block.phis {
            defs.insert(phi.value, Definition::Phi(&phi.args));
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            for &(_, value) in &inst.writes {
                defs.insert(value, Definition::Instruction(site, inst));
            }
            if inst.machine.offset == offset {
                if producer(inst) {
                    sources.push((site, inst));
                }
                queue.extend(inst.reads.iter().map(|&(_, value)| value));
            }
        }
    }

    let mut seen = HashSet::new();
    while let Some(value) = queue.pop_front() {
        if sources.len() >= 8 || seen.len() >= 4096 {
            break;
        }
        if !seen.insert(value) {
            continue;
        }
        match defs.get(&value) {
            Some(&Definition::Instruction(site, inst)) if producer(inst) => {
                if !sources.iter().any(|&(s, _)| s == site) {
                    sources.push((site, inst));
                }
            }
            Some(Definition::Instruction(_, inst)) => {
                queue.extend(inst.reads.iter().map(|&(_, value)| value));
            }
            Some(Definition::Phi(args)) => queue.extend(args.iter().copied()),
            None => {}
        }
    }

    let details = sources
        .into_iter()
        .take(8)
        .map(|(site, inst)| {
            let opcode = inst.name();
            let at = inst.machine.offset;
            if interpolation {
                let folded = io.interps.iter().any(|(_, interp)| {
                    matches!(interp, Interp::Interpolated { first, .. } if first.contains(&site))
                });
                format!("{opcode} at {at:#x} folded={folded}")
            } else if let Some((_, address)) = resources.scalar.iter().find(|(s, _)| *s == site) {
                let root = layout.get(address.param).map(|p| (p.root, p.slot));
                format!(
                    "{opcode} at {at:#x} root={root:?} byte-offset={:#x} indexed={}",
                    address.offset,
                    address.dynamic.is_some(),
                )
            } else {
                format!("{opcode} at {at:#x} address=untracked")
            }
        })
        .collect();
    Some(details)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RootType, resource::Address};

    fn program(words: &[u32]) -> Program {
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::ir::build(&bytes, false, 32).unwrap()
    }

    #[test]
    fn follows_interpolation_through_a_move_to_the_consumer() {
        let p = program(&[
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0x7E06_0302, // v_mov_b32 v3, v2
            0xF800_180F, // exp mrt0, v3, v3, v3, v3
            0x0303_0303,
            0xBF81_0000,
        ]);
        assert_eq!(
            refusal_sources(
                &p,
                &Io::default(),
                &Resources::default(),
                &[],
                8,
                "the first half of an interpolation",
            )
            .unwrap(),
            ["v_interp_p1_f32 at 0x0 folded=false"]
        );
    }

    #[test]
    fn reports_only_descriptor_loads_that_feed_the_failed_operation() {
        let p = program(&[
            0xF400_0400,
            0xFA00_0008, // s_load_dword s16, s[0:1], 8
            0xF400_0440,
            0xFA00_0048, // s_load_dword s17, s[0:1], 72
            0x7E00_0210, // v_mov_b32 v0, s16
            0xF800_180F,
            0, // exp mrt0, v0, v0, v0, v0
            0xBF81_0000,
        ]);
        let res = Resources {
            scalar: vec![
                (
                    (0, 0),
                    Address {
                        param: 0,
                        offset: 8,
                        dynamic: None,
                    },
                ),
                (
                    (0, 1),
                    Address {
                        param: 0,
                        offset: 72,
                        dynamic: None,
                    },
                ),
            ],
            ..Resources::default()
        };
        let layout = [RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 2,
        }];
        assert_eq!(
            refusal_sources(
                &p,
                &Io::default(),
                &res,
                &layout,
                20,
                "a descriptor read only as data"
            )
            .unwrap(),
            ["s_load_dword at 0x0 root=Some((Table, 2)) byte-offset=0x8 indexed=false"]
        );
    }
}
