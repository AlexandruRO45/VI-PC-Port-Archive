//! HLSL code generation and hoisting for portable multi-wave Wave64 lowering.

use super::super::{hi, low_mask, name, r, wave_sum, Emitter, Enc, Insn, Opnd, OP_WAVE_REDUCE};
use super::MaskOp;

fn lane_bit_extract(lane: &str, lo: &str, hi_expr: &str) -> String {
    format!("(((({lane} & 63u) < 32u) ? {lo} : {hi_expr}) >> ({lane} & 31u))")
}

fn mask_reg_name(op: &Opnd) -> &'static str {
    if matches!(*op, Opnd::VccLo | Opnd::VccHi) {
        "vcc_lo"
    } else {
        "exec_lo"
    }
}

fn dpp_partner_lane(ins: &Insn, id: &str) -> Result<String, String> {
    let control = ins.raw[1] >> 8;
    Ok(match (ins.aux.imm, control) {
        (0, _) => format!("{id} ^ 1u"),
        (1, 0x936012) => format!("({id} & ~3u) | (({id} ^ 2u) & 2u)"),
        (2, 0x000924) => format!("({id} & ~7u) | (({id} ^ 4u) & 4u)"),
        (1, 0xb3e21a) => format!("{id} ^ 2u"),
        (2, 0x688fac) => format!("{id} ^ 4u"),
        (1 | 2, _) => format!("({id} & ~7u) | (({control:#x}u >> (3u * ({id} & 7u))) & 7u)"),
        (3, _) => format!("{id} ^ 15u"),
        _ => return Err("invalid integer sum step".into()),
    })
}

fn bit_combine(m: &str, a: &str, b: &str) -> Result<String, String> {
    let opname = m.trim_start_matches("s_").trim_end_matches("_b64");
    Ok(match opname {
        "and" => format!("({a} & {b})"),
        "or" => format!("({a} | {b})"),
        "xor" => format!("({a} ^ {b})"),
        "andn2" => format!("({a} & ~{b})"),
        "orn2" => format!("({a} | ~{b})"),
        "nand" => format!("(~({a} & {b}))"),
        "nor" => format!("(~({a} | {b}))"),
        "xnor" => format!("(~({a} ^ {b}))"),
        _ => return Err("unsupported 64-bit ballot logic op".into()),
    })
}

/// Emit hoisted barrier-bearing portable Wave64 computation BEFORE an enclosing forward skip guard.
pub(in crate::hlsl) fn hoist_before(
    em: &mut Emitter<'_>,
    idx: usize,
    ins: &Insn,
) -> Result<Option<String>, String> {
    let Some(plan) = em.portable_w64.as_ref().cloned() else {
        return Ok(None);
    };
    if !plan.hoisted_ops.contains(&idx) {
        return Ok(None);
    }
    let lane = plan.lane_expr();
    let m = ins.mnemonic.as_str();
    let s = &ins.src;
    let is_ballot_cmp = plan.ballot_compares.contains(&idx)
        || em.packed.as_ref().is_some_and(|p| p.ballots.contains(&idx));
    if is_ballot_cmp {
        for o in s {
            em.touch(o);
        }
        let c = em.compare_cond(ins)?;
        return Ok(Some(format!(
            "uint2 w64_b{idx} = tb_w64_ballot({lane}, ((exec_lo & 1u) != 0u) && ({c}));"
        )));
    }
    if plan.ballot_mask_copies.contains(&idx)
        || plan.ballot_word_reads.contains(&idx)
        || matches!(
            m,
            "tb_capture_allocation_mask" | "tb_allocation_prefix_lo" | "tb_allocation_prefix_hi"
        )
        || (m == "s_wqm_b64" && s == &[Opnd::ExecLo])
    {
        let reg = if m == "s_mov_b64" || plan.ballot_word_reads.contains(&idx) {
            mask_reg_name(&s[0])
        } else {
            "exec_lo"
        };
        return Ok(Some(format!(
            "uint2 w64_b{idx} = tb_w64_ballot({lane}, ({reg} & 1u) != 0u);"
        )));
    }
    if plan.ballot_logic_mask.contains(&idx) {
        let mut lines = Vec::new();
        for (k, op) in s.iter().enumerate() {
            if matches!(op, Opnd::ExecLo | Opnd::VccLo) {
                lines.push(format!(
                    "uint2 w64_b{idx}_{k} = tb_w64_ballot({lane}, ({} & 1u) != 0u);",
                    mask_reg_name(op)
                ));
            }
        }
        if !lines.is_empty() {
            return Ok(Some(lines.join("\n    ")));
        }
    }
    if matches!(m, "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32") {
        if let Some(op @ (Opnd::ExecLo | Opnd::ExecHi | Opnd::VccLo | Opnd::VccHi)) = s.first() {
            return Ok(Some(format!(
                "uint2 w64_b{idx} = tb_w64_ballot({lane}, ({} & 1u) != 0u);",
                mask_reg_name(op)
            )));
        }
    }
    if let Some(&op) = plan.mask_votes.get(&idx) {
        let code = match op {
            MaskOp::AnyTrue(reg) | MaskOp::NoneTrue(reg) => {
                em.touch(&reg);
                format!("bool w64_v{idx} = tb_w64_any({lane}, ({} & 1u) != 0u);", r(&reg)?)
            }
            MaskOp::AllTrue(reg) | MaskOp::NotAllTrue(reg) => {
                em.touch(&reg);
                format!("bool w64_v{idx} = tb_w64_all({lane}, ({} & 1u) != 0u);", r(&reg)?)
            }
            MaskOp::AllEqual(a, b) => {
                em.touch(&a);
                em.touch(&b);
                format!(
                    "bool w64_v{idx} = tb_w64_all({lane}, (({} & 1u) == ({} & 1u)));",
                    r(&a)?,
                    r(&b)?
                )
            }
            MaskOp::AnyDiff(a, b) => {
                em.touch(&a);
                em.touch(&b);
                format!(
                    "bool w64_v{idx} = tb_w64_any({lane}, (({} & 1u) != ({} & 1u)));",
                    r(&a)?,
                    r(&b)?
                )
            }
            MaskOp::AllEqualBallot(bal, msk) => {
                em.touch(&bal);
                em.touch(&hi(&bal));
                em.touch(&msk);
                let bit = lane_bit_extract(lane, &r(&bal)?, &r(&hi(&bal))?);
                format!("bool w64_v{idx} = tb_w64_all({lane}, (({bit} & 1u) == ({} & 1u)));", r(&msk)?)
            }
            MaskOp::AnyDiffBallot(bal, msk) => {
                em.touch(&bal);
                em.touch(&hi(&bal));
                em.touch(&msk);
                let bit = lane_bit_extract(lane, &r(&bal)?, &r(&hi(&bal))?);
                format!("bool w64_v{idx} = tb_w64_any({lane}, (({bit} & 1u) != ({} & 1u)));", r(&msk)?)
            }
            MaskOp::CountBits(reg) => {
                em.touch(&reg);
                format!("uint w64_u{idx} = tb_w64_count_bits({lane}, ({} & 1u) != 0u);", r(&reg)?)
            }
            MaskOp::FirstActive(reg) => {
                em.touch(&reg);
                format!("uint w64_u{idx} = tb_w64_first_active({lane}, ({} & 1u) != 0u);", r(&reg)?)
            }
            MaskOp::PostAnyTrue(_) => return Ok(None),
        };
        return Ok(Some(code));
    }
    if ins.enc == Enc::Sopp
        && ((ins.op == 8 && em.reduction_execz_guards.contains(&idx))
            || (matches!(ins.op, 6..=9) && em.lockstep.is_some()))
    {
        let reg = if matches!(ins.op, 6 | 7) { "vcc_lo" } else { "exec_lo" };
        return Ok(Some(format!(
            "bool w64_br{idx} = tb_w64_any({lane}, ({reg} & 1u) != 0u);"
        )));
    }
    if ins.enc == Enc::Vop2 && ins.op == wave_sum::DPP_REDUCE {
        em.touch(&s[0]);
        let partner = dpp_partner_lane(ins, lane)?;
        return Ok(Some(format!(
            "uint w64_r{idx} = tb_reduction_read({lane}, {}, {partner});",
            r(&s[0])?
        )));
    }
    if (ins.enc == Enc::Vop3 && ins.op == wave_sum::READ_LANE) || m == super::super::GROUP_READ {
        em.touch(&s[0]);
        em.touch(&s[1]);
        let sel = if matches!(s[1], Opnd::Sgpr(_)) {
            format!("({} & 63u)", r(&s[1])?)
        } else {
            r(&s[1])?
        };
        return Ok(Some(format!(
            "uint w64_r{idx} = tb_reduction_read({lane}, {}, {sel});",
            r(&s[0])?
        )));
    }
    if m == "v_readfirstlane_b32" {
        em.touch(&s[0]);
        return Ok(Some(format!(
            "uint w64_r{idx} = tb_w64_read_first_lane({lane}, {}, (exec_lo & 1u) != 0u);",
            r(&s[0])?
        )));
    }
    if m == "v_permlanex16_b32" {
        for o in s.iter().take(3) {
            em.touch(o);
        }
        if plan.reduction_permlane.contains(&idx) {
            return Ok(Some(format!(
                "uint w64_r{idx} = tb_reduction_read({lane}, {}, ({lane} & ~15u) ^ 16u);",
                r(&s[0])?
            )));
        }
        let (val, lo, hi_s) = (r(&s[0])?, r(&s[1])?, r(&s[2])?);
        return Ok(Some(format!(
            "uint w64_p{idx}lane = ({lane}) & 63u; uint w64_p{idx}sel = (w64_p{idx}lane & 8u) ? {hi_s} : {lo};\n    \
             uint w64_p{idx}src = ((w64_p{idx}lane & ~15u) ^ 16u) | ((w64_p{idx}sel >> (4u * (w64_p{idx}lane & 7u))) & 15u);\n    \
             uint w64_p{idx}val = tb_reduction_read({lane}, {val}, w64_p{idx}src);\n    \
             uint2 w64_p{idx}act = tb_w64_ballot({lane}, (exec_lo & 1u) != 0u);"
        )));
    }
    if ins.enc == Enc::Vop3 && ins.op == OP_WAVE_REDUCE {
        em.touch(&s[0]);
        em.touch(&s[1]);
        let maximum = m == "tb_wave_max_f32";
        return Ok(Some(format!(
            "uint w64_r{idx} = tb_w64_extrema_f32({lane}, (({} & 1u) != 0u) ? asfloat({}) : asfloat({:#x}u), {maximum});",
            r(&s[1])?,
            r(&s[0])?,
            ins.aux.imm as u32
        )));
    }
    if m == "ds_swizzle_b32" && em.packed.as_ref().is_some_and(|p| p.scan_width == Some(64)) {
        em.touch(&s[0]);
        let off = ins.aux.imm as u32;
        let src_lane = format!(
            "((({lane}) & 63u) & ~31u) | ((((({lane}) & 31u) & {}u) | {}u) ^ {}u)",
            off & 31,
            (off >> 5) & 31,
            (off >> 10) & 31
        );
        return Ok(Some(format!(
            "uint w64_r{idx} = tb_reduction_read({lane}, {}, {src_lane});",
            r(&s[0])?
        )));
    }
    Ok(None)
}

/// Emit hoisted post-instruction barrier votes AFTER `emit_insn` when enclosed by a forward skip.
pub(in crate::hlsl) fn hoist_after(
    em: &mut Emitter<'_>,
    idx: usize,
    ins: &Insn,
) -> Result<Option<(String, String)>, String> {
    let Some(plan) = em.portable_w64.as_ref().cloned() else {
        return Ok(None);
    };
    if !plan.hoisted_ops.contains(&idx) {
        return Ok(None);
    }
    let lane = plan.lane_expr();
    let mut barriers = Vec::new();
    let mut applies = Vec::new();
    if let Some(vote) = em.lockstep.as_ref().and_then(|v| v.get(&idx)).cloned() {
        let mask = r(&vote.mask)?;
        let not = if vote.none { "!" } else { "" };
        barriers.push(format!("bool w64_lv{idx} = tb_w64_any({lane}, ({mask} & 1u) != 0u);"));
        applies.push(format!("scc = uint({not}w64_lv{idx});"));
    } else if let Some(MaskOp::PostAnyTrue(mask)) = plan.mask_votes.get(&idx).copied() {
        let m = r(&mask)?;
        barriers.push(format!("bool w64_pv{idx} = tb_w64_any({lane}, ({m} & 1u) != 0u);"));
        applies.push(format!("scc = uint(w64_pv{idx});"));
    }
    let is_ballot_cmp = plan.ballot_compares.contains(&idx)
        || em.packed.as_ref().is_some_and(|p| p.ballots.contains(&idx));
    if em.vote_defs.contains(&idx) && !is_ballot_cmp {
        let mask = r(&ins.dst[0])?;
        barriers.push(format!("uint w64_vd{idx} = tb_w64_count_bits({lane}, ({mask} & 1u) != 0u);"));
        applies.push(format!("tb_votes{idx} = w64_vd{idx};"));
    }
    if barriers.is_empty() {
        Ok(None)
    } else {
        Ok(Some((barriers.join("\n    "), applies.join("\n    "))))
    }
}

/// Emit portable Wave64 instruction body. Returns `Ok(true)` if handled.
pub(in crate::hlsl) fn emit(
    em: &mut Emitter<'_>,
    idx: usize,
    ins: &Insn,
    br: Option<usize>,
) -> Result<bool, String> {
    let Some(plan) = em.portable_w64.as_ref().cloned() else {
        return Ok(false);
    };
    let lane = plan.lane_expr();
    let hoisted = em.hoisted_w64 == Some(idx);
    let m = ins.mnemonic.as_str();
    let s = &ins.src;
    let d = ins.dst.first().cloned();

    // 1. 64-lane `v_cmp_*` ballot materialization.
    let is_ballot_cmp = plan.ballot_compares.contains(&idx)
        || em.packed.as_ref().is_some_and(|p| p.ballots.contains(&idx));
    if is_ballot_cmp {
        let dst = d.ok_or("missing ballot destination")?;
        if !hoisted {
            let c = em.compare_cond(ins)?;
            em.line(&format!(
                "uint2 w64_b{idx} = tb_w64_ballot({lane}, ((exec_lo & 1u) != 0u) && ({c}));"
            ));
        }
        em.ws(&dst, &format!("w64_b{idx}.x"))?;
        let upper = hi(&dst);
        em.touch(&upper);
        em.ws(&upper, &format!("w64_b{idx}.y"))?;
        return Ok(true);
    }

    // 2. Direct 32-bit ballot word moves from `exec_lo|exec_hi|vcc_lo|vcc_hi`.
    if plan.ballot_word_reads.contains(&idx) {
        let dst = d.ok_or("missing ballot word destination")?;
        if !hoisted {
            em.line(&format!(
                "uint2 w64_b{idx} = tb_w64_ballot({lane}, ({} & 1u) != 0u);",
                mask_reg_name(&s[0])
            ));
        }
        let word = if matches!(s[0], Opnd::ExecHi | Opnd::VccHi) { "y" } else { "x" };
        if m == "v_mov_b32" {
            em.wv(&dst, &format!("w64_b{idx}.{word}"))?;
        } else {
            em.ws(&dst, &format!("w64_b{idx}.{word}"))?;
        }
        return Ok(true);
    }

    // 3. 64-bit logic combining a `Ballot64` pair with `exec` or `vcc`.
    if plan.ballot_logic_mask.contains(&idx) {
        let dst = d.ok_or("missing ballot logic destination")?;
        if !hoisted {
            for (k, op) in s.iter().enumerate() {
                if matches!(op, Opnd::ExecLo | Opnd::VccLo) {
                    em.line(&format!(
                        "uint2 w64_b{idx}_{k} = tb_w64_ballot({lane}, ({} & 1u) != 0u);",
                        mask_reg_name(op)
                    ));
                }
            }
        }
        let mut op_words = |k: usize, op: &Opnd| -> Result<(String, String), String> {
            if matches!(*op, Opnd::ExecLo | Opnd::VccLo) {
                Ok((format!("w64_b{idx}_{k}.x"), format!("w64_b{idx}_{k}.y")))
            } else {
                em.touch(&hi(op));
                Ok((r(op)?, r(&hi(op))?))
            }
        };
        let (al, ah) = op_words(0, &s[0])?;
        let rhs = if m == "s_not_b64" { None } else { Some(op_words(1, &s[1])?) };
        let t = em.tmp;
        em.tmp += 1;
        if let Some((bl, bh)) = rhs {
            em.line(&format!(
                "uint t{t}l = {}; uint t{t}h = {};",
                bit_combine(m, &al, &bl)?,
                bit_combine(m, &ah, &bh)?
            ));
        } else {
            em.line(&format!("uint t{t}l = (~{al}); uint t{t}h = (~{ah});"));
        }
        em.ws(&dst, &format!("t{t}l"))?;
        let dh = hi(&dst);
        em.touch(&dh);
        em.ws(&dh, &format!("t{t}h"))?;
        em.line(&format!("scc = ((t{t}l | t{t}h) != 0u) ? 1u : 0u;"));
        return Ok(true);
    }

    // 4. EXEC/VCC ballot capture (`tb_capture_allocation_mask`, `s_mov_b64`, `saveexec_b64`).
    if m == "tb_capture_allocation_mask" || plan.ballot_mask_copies.contains(&idx) {
        let dst = d.ok_or("missing mask register")?;
        let upper = hi(&dst);
        em.touch(&upper);
        let reg = if m == "s_mov_b64" { mask_reg_name(&s[0]) } else { "exec_lo" };
        if !hoisted {
            em.line(&format!(
                "uint2 w64_b{idx} = tb_w64_ballot({lane}, ({reg} & 1u) != 0u);"
            ));
        }
        if m.contains("_saveexec_") {
            let src_bit = if plan.ballot_mask_reads.contains(&idx) {
                em.touch(&hi(&s[0]));
                format!("({} & 1u)", lane_bit_extract(lane, &r(&s[0])?, &r(&hi(&s[0]))?))
            } else {
                r(&s[0])?
            };
            let new_exec = match m.split('_').nth(1) {
                Some("and") => format!("{src_bit} & exec_lo"),
                Some("or") => format!("{src_bit} | exec_lo"),
                Some("xor") => format!("{src_bit} ^ exec_lo"),
                Some("nand") => format!("~({src_bit} & exec_lo)"),
                Some("andn2") => format!("{src_bit} & ~exec_lo"),
                Some("orn2") => format!("{src_bit} | ~exec_lo"),
                Some("andn1") => format!("~{src_bit} & exec_lo"),
                _ => return Err("unsupported saveexec in portable wave64".into()),
            };
            em.line(&format!("uint w64_e{idx} = {new_exec};"));
            em.ws(&dst, &format!("w64_b{idx}.x"))?;
            em.ws(&upper, &format!("w64_b{idx}.y"))?;
            em.line(&format!("exec_lo = w64_e{idx}; exec_hi = 0u; scc = (exec_lo != 0u);"));
        } else {
            em.ws(&dst, &format!("w64_b{idx}.x"))?;
            em.ws(&upper, &format!("w64_b{idx}.y"))?;
        }
        return Ok(true);
    }

    // 5. `tb_allocation_prefix_lo` / `tb_allocation_prefix_hi`.
    if matches!(m, "tb_allocation_prefix_lo" | "tb_allocation_prefix_hi") {
        let high = m.ends_with("_hi");
        if !hoisted {
            em.line(&format!("uint2 w64_b{idx} = tb_w64_ballot({lane}, (exec_lo & 1u) != 0u);"));
        }
        let lane_part = if high {
            format!("max(32u, ({lane}) & 63u) - 32u")
        } else {
            format!("({lane}) & 63u")
        };
        let word = if high { "y" } else { "x" };
        let prefix = format!("countbits(w64_b{idx}.{word} & {})", low_mask(&lane_part));
        em.wv(&d.ok_or("missing prefix destination")?, &format!("({} + {prefix})", r(&s[1])?))?;
        return Ok(true);
    }

    // 6. `v_mbcnt_lo_u32_b32` / `v_mbcnt_hi_u32_b32` (64-lane prefix count bits).
    if matches!(m, "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32") {
        let dst = d.ok_or("missing mbcnt destination")?;
        let high = m == "v_mbcnt_hi_u32_b32";
        let b = r(&s[1])?;
        if matches!(s[0], Opnd::ConstInt(-1) | Opnd::Literal(u32::MAX)) {
            let count = if high {
                format!("max(({lane}) & 63u, 32u) - 32u")
            } else {
                format!("min(({lane}) & 63u, 32u)")
            };
            em.wv(&dst, &format!("({b} + ({count}))"))?;
            return Ok(true);
        }
        let lane_part = if high {
            format!("max(32u, ({lane}) & 63u) - 32u")
        } else {
            format!("({lane}) & 63u")
        };
        let mask_word = if matches!(s[0], Opnd::ExecLo | Opnd::ExecHi | Opnd::VccLo | Opnd::VccHi) {
            if !hoisted {
                em.line(&format!(
                    "uint2 w64_b{idx} = tb_w64_ballot({lane}, ({} & 1u) != 0u);",
                    mask_reg_name(&s[0])
                ));
            }
            let word = if matches!(s[0], Opnd::ExecHi | Opnd::VccHi) { "y" } else { "x" };
            format!("w64_b{idx}.{word}")
        } else {
            r(&s[0])?
        };
        let t = em.tmp;
        em.tmp += 1;
        em.line(&format!("uint t{t} = countbits({mask_word} & {});", low_mask(&lane_part)));
        em.wv(&dst, &format!("({b} + t{t})"))?;
        return Ok(true);
    }

    // 7. Ballot-mask consumers (`s_mov_b64 exec, s[lo:hi]` and `s_*_saveexec_b64 dst, s[lo:hi]`).
    if plan.ballot_mask_reads.contains(&idx) && m != "v_cndmask_b32" {
        em.touch(&hi(&s[0]));
        let src_bit = format!("({} & 1u)", lane_bit_extract(lane, &r(&s[0])?, &r(&hi(&s[0]))?));
        if m == "s_mov_b64" && ins.dst == [Opnd::ExecLo] {
            em.line(&format!("exec_lo = {src_bit}; exec_hi = 0u;"));
            return Ok(true);
        }
        if m.contains("_saveexec_") {
            let dst = d.ok_or("missing saveexec dst")?;
            let new_exec = match m.split('_').nth(1) {
                Some("and") => format!("{src_bit} & exec_lo"),
                Some("or") => format!("{src_bit} | exec_lo"),
                Some("xor") => format!("{src_bit} ^ exec_lo"),
                Some("nand") => format!("~({src_bit} & exec_lo)"),
                Some("andn2") => format!("{src_bit} & ~exec_lo"),
                Some("orn2") => format!("{src_bit} | ~exec_lo"),
                Some("andn1") => format!("~{src_bit} & exec_lo"),
                _ => return Err("unsupported saveexec in portable wave64".into()),
            };
            let t = em.tmp;
            em.tmp += 1;
            em.line(&format!("uint t{t}l = {new_exec};"));
            em.ws(&dst, "exec_lo")?;
            let dh = hi(&dst);
            em.touch(&dh);
            em.ws(&dh, "exec_hi")?;
            em.line(&format!("exec_lo = t{t}l; exec_hi = 0u; scc = (exec_lo != 0u);"));
            return Ok(true);
        }
    }

    // 8. Per-lane mask votes / counts / first-active.
    if let Some(&op) = plan.mask_votes.get(&idx) {
        match op {
            MaskOp::AnyTrue(reg) => {
                let expr = if hoisted { format!("w64_v{idx}") } else { format!("tb_w64_any({lane}, ({} & 1u) != 0u)", r(&reg)?) };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::NoneTrue(reg) => {
                let expr = if hoisted { format!("!w64_v{idx}") } else { format!("!tb_w64_any({lane}, ({} & 1u) != 0u)", r(&reg)?) };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::AllTrue(reg) => {
                let expr = if hoisted { format!("w64_v{idx}") } else { format!("tb_w64_all({lane}, ({} & 1u) != 0u)", r(&reg)?) };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::NotAllTrue(reg) => {
                let expr = if hoisted { format!("!w64_v{idx}") } else { format!("!tb_w64_all({lane}, ({} & 1u) != 0u)", r(&reg)?) };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::AllEqual(a, b) => {
                let expr = if hoisted { format!("w64_v{idx}") } else { format!("tb_w64_all({lane}, (({} & 1u) == ({} & 1u)))", r(&a)?, r(&b)?) };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::AnyDiff(a, b) => {
                let expr = if hoisted { format!("w64_v{idx}") } else { format!("tb_w64_any({lane}, (({} & 1u) != ({} & 1u)))", r(&a)?, r(&b)?) };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::AllEqualBallot(bal, msk) => {
                let expr = if hoisted {
                    format!("w64_v{idx}")
                } else {
                    em.touch(&hi(&bal));
                    let bit = lane_bit_extract(lane, &r(&bal)?, &r(&hi(&bal))?);
                    format!("tb_w64_all({lane}, (({bit} & 1u) == ({} & 1u)))", r(&msk)?)
                };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::AnyDiffBallot(bal, msk) => {
                let expr = if hoisted {
                    format!("w64_v{idx}")
                } else {
                    em.touch(&hi(&bal));
                    let bit = lane_bit_extract(lane, &r(&bal)?, &r(&hi(&bal))?);
                    format!("tb_w64_any({lane}, (({bit} & 1u) != ({} & 1u)))", r(&msk)?)
                };
                em.line(&format!("scc = uint({expr});"));
                return Ok(true);
            }
            MaskOp::CountBits(reg) => {
                let dst = d.ok_or("missing bcnt destination")?;
                let expr = if hoisted { format!("w64_u{idx}") } else { format!("tb_w64_count_bits({lane}, ({} & 1u) != 0u)", r(&reg)?) };
                em.ws(&dst, &expr)?;
                em.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&dst)?));
                return Ok(true);
            }
            MaskOp::FirstActive(reg) => {
                let dst = d.ok_or("missing ff1 destination")?;
                let expr = if hoisted { format!("w64_u{idx}") } else { format!("tb_w64_first_active({lane}, ({} & 1u) != 0u)", r(&reg)?) };
                em.ws(&dst, &expr)?;
                return Ok(true);
            }
            MaskOp::PostAnyTrue(_) => {}
        }
    }

    // 9. Voted forward branches (`reduction_execz_guards` and `lockstep`).
    if ins.enc == Enc::Sopp
        && ((ins.op == 8 && em.reduction_execz_guards.contains(&idx))
            || (matches!(ins.op, 6..=9) && em.lockstep.is_some()))
    {
        if let Some(t) = br {
            let not = if matches!(ins.op, 6 | 8) { "!" } else { "" };
            let cond = if hoisted {
                format!("{not}w64_br{idx}")
            } else {
                let reg = if matches!(ins.op, 6 | 7) { "vcc_lo" } else { "exec_lo" };
                format!("{not}tb_w64_any({lane}, ({reg} & 1u) != 0u)")
            };
            em.line(&format!("if ({cond}) j{t} = true;"));
        }
        return Ok(true);
    }

    // 10. `v_readfirstlane_b32` and dynamic `GROUP_READ` (`v_readlane_b32`) across all 64 lanes.
    if m == super::super::GROUP_READ {
        let dst = d.ok_or("missing group read destination")?;
        let val = if hoisted {
            format!("w64_r{idx}")
        } else {
            let sel = if matches!(s[1], Opnd::Sgpr(_)) {
                format!("({} & 63u)", r(&s[1])?)
            } else {
                r(&s[1])?
            };
            format!("tb_reduction_read({lane}, {}, {sel})", r(&s[0])?)
        };
        em.ws(&dst, &val)?;
        return Ok(true);
    }
    if m == "v_readfirstlane_b32" {
        let dst = d.ok_or("missing readfirstlane destination")?;
        let val = if hoisted {
            format!("w64_r{idx}")
        } else {
            format!("tb_w64_read_first_lane({lane}, {}, (exec_lo & 1u) != 0u)", r(&s[0])?)
        };
        em.ws(&dst, &val)?;
        return Ok(true);
    }

    // 11. `v_permlanex16_b32`.
    if m == "v_permlanex16_b32" {
        let flags = (ins.raw[0] >> 11) & 0xf;
        if flags != 0 || ins.aux.mask != 0 || ins.aux.imm != 0 || ins.aux.imm2 != 0 {
            return Err("permlanex16 modifiers unsupported".into());
        }
        let dst = d.ok_or("missing permlanex16 destination")?;
        if plan.reduction_permlane.contains(&idx) {
            let val = if hoisted {
                format!("w64_r{idx}")
            } else {
                let t = em.tmp;
                em.tmp += 1;
                em.line(&format!(
                    "uint t{t} = tb_reduction_read({lane}, {}, ({lane} & ~15u) ^ 16u);",
                    r(&s[0])?
                ));
                format!("t{t}")
            };
            em.wv(&dst, &val)?;
            return Ok(true);
        }
        if !hoisted {
            let (val, lo, hi_s) = (r(&s[0])?, r(&s[1])?, r(&s[2])?);
            em.line(&format!("uint w64_p{idx}lane = ({lane}) & 63u; uint w64_p{idx}sel = (w64_p{idx}lane & 8u) ? {hi_s} : {lo};"));
            em.line(&format!("uint w64_p{idx}src = ((w64_p{idx}lane & ~15u) ^ 16u) | ((w64_p{idx}sel >> (4u * (w64_p{idx}lane & 7u))) & 15u);"));
            em.line(&format!("uint w64_p{idx}val = tb_reduction_read({lane}, {val}, w64_p{idx}src);"));
            em.line(&format!("uint2 w64_p{idx}act = tb_w64_ballot({lane}, (exec_lo & 1u) != 0u);"));
        }
        em.line(&format!(
            "if ((((w64_p{idx}src < 32u) ? w64_p{idx}act.x : w64_p{idx}act.y) >> (w64_p{idx}src & 31u)) & 1u) {{"
        ));
        em.wv(&dst, &format!("w64_p{idx}val"))?;
        em.line("}");
        return Ok(true);
    }

    // 12. `OP_WAVE_REDUCE` (`tb_wave_min_f32` / `tb_wave_max_f32`).
    if ins.enc == Enc::Vop3 && ins.op == OP_WAVE_REDUCE {
        let dst = d.ok_or("missing wave reduce destination")?;
        let val = if hoisted {
            format!("w64_r{idx}")
        } else {
            let maximum = m == "tb_wave_max_f32";
            format!(
                "tb_w64_extrema_f32({lane}, (({} & 1u) != 0u) ? asfloat({}) : asfloat({:#x}u), {maximum})",
                r(&s[1])?,
                r(&s[0])?,
                ins.aux.imm as u32
            )
        };
        em.ws(&dst, &val)?;
        return Ok(true);
    }

    // 13. `ds_swizzle_b32` in packed 64-lane scans.
    if m == "ds_swizzle_b32" && em.packed.as_ref().is_some_and(|p| p.scan_width == Some(64)) {
        let off = ins.aux.imm as u32;
        if ins.raw[0] & (1 << 17) != 0 || off & 0x8000 != 0 {
            return Err("unsupported compute swizzle mode".into());
        }
        let dst = d.ok_or("no swizzle destination")?;
        let val = if hoisted {
            format!("w64_r{idx}")
        } else {
            let t = em.tmp;
            em.tmp += 1;
            let src_lane = format!(
                "((({lane}) & 63u) & ~31u) | ((((({lane}) & 31u) & {}u) | {}u) ^ {}u)",
                off & 31,
                (off >> 5) & 31,
                (off >> 10) & 31
            );
            em.line(&format!("uint t{t} = tb_reduction_read({lane}, {}, {src_lane});", r(&s[0])?));
            format!("t{t}")
        };
        em.wv(&dst, &val)?;
        return Ok(true);
    }

    // 14. `s_wqm_b64` from EXEC.
    if m == "s_wqm_b64" && s == &[Opnd::ExecLo] {
        let dst = d.ok_or("missing wqm destination")?;
        if !hoisted {
            em.line(&format!("uint2 w64_b{idx} = tb_w64_ballot({lane}, (exec_lo & 1u) != 0u);"));
        }
        let t = em.tmp;
        em.tmp += 1;
        em.line(&format!("uint t{t}lane = ({lane}) & 63u;"));
        em.line(&format!(
            "uint t{t}l = (((t{t}lane < 32u ? w64_b{idx}.x : w64_b{idx}.y) >> (t{t}lane & 28u)) & 15u) != 0u ? 1u : 0u;"
        ));
        em.line(&format!("scc = uint((w64_b{idx}.x | w64_b{idx}.y) != 0u);"));
        em.ws(&dst, &format!("t{t}l"))?;
        let dh = hi(&dst);
        em.touch(&dh);
        em.ws(&dh, "0u")?;
        return Ok(true);
    }

    Ok(false)
}

/// Emit post-instruction portable Wave64 votes when NOT hoisted by `control_flow.rs`.
pub(in crate::hlsl) fn emit_post_votes(em: &mut Emitter<'_>, idx: usize, ins: &Insn) -> Result<bool, String> {
    let Some(plan) = em.portable_w64.as_ref().cloned() else {
        return Ok(false);
    };
    let lane = plan.lane_expr();
    let defer_to_hoist_after = em.in_active_guard && plan.hoisted_ops.contains(&idx);
    if !defer_to_hoist_after {
        if let Some(vote) = em.lockstep.as_ref().and_then(|v| v.get(&idx)).cloned() {
            let mask = r(&vote.mask)?;
            let not = if vote.none { "!" } else { "" };
            em.line(&format!("scc = uint({not}tb_w64_any({lane}, ({mask} & 1u) != 0u));"));
        } else if let Some(MaskOp::PostAnyTrue(mask)) = plan.mask_votes.get(&idx).copied() {
            let m = r(&mask)?;
            em.line(&format!("scc = uint(tb_w64_any({lane}, ({m} & 1u) != 0u));"));
        }
        if em.vote_defs.contains(&idx) {
            let is_ballot_cmp = plan.ballot_compares.contains(&idx)
                || em.packed.as_ref().is_some_and(|p| p.ballots.contains(&idx));
            if is_ballot_cmp {
                let lo = r(&ins.dst[0])?;
                let high = r(&hi(&ins.dst[0]))?;
                em.line(&format!("tb_votes{idx} = countbits({lo}) + countbits({high});"));
            } else {
                let mask = r(&ins.dst[0])?;
                em.line(&format!("tb_votes{idx} = tb_w64_count_bits({lane}, ({mask} & 1u) != 0u);"));
            }
        }
    } else if em.vote_defs.contains(&idx) {
        let is_ballot_cmp = plan.ballot_compares.contains(&idx)
            || em.packed.as_ref().is_some_and(|p| p.ballots.contains(&idx));
        if is_ballot_cmp {
            let lo = r(&ins.dst[0])?;
            let high = r(&hi(&ins.dst[0]))?;
            em.line(&format!("tb_votes{idx} = countbits({lo}) + countbits({high});"));
        }
    }
    Ok(true)
}

/// Portable Wave64 expression for `lockstep::branch_condition` on loop latches.
pub(in crate::hlsl) fn lockstep_branch_condition(op: u32, lane: &str) -> Option<String> {
    let (register, taken_when_none) = match op {
        6 => ("vcc_lo", true),
        7 => ("vcc_lo", false),
        8 => ("exec_lo", true),
        9 => ("exec_lo", false),
        _ => return None,
    };
    let not = if taken_when_none { "!" } else { "" };
    Some(format!("{not}tb_w64_any({lane}, ({register} & 1u) != 0u)"))
}
