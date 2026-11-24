//! Ordinary VOP3 integer forms with a 32-bit result: the signed no-carry
//! add/subtract and the 16x16+32 multiply-add. Like the newer 16-bit forms
//! they are initially limited to VGPR sources without modifiers; in
//! particular clamp (signed saturation) is refused.
use super::{int16::half, r, Enc, Insn, Opnd};

pub(super) fn handles(ins: &Insn) -> bool {
    ins.enc == Enc::Vop3 && matches!(ins.mnemonic.as_str(),
        "v_add_nc_i32" | "v_sub_nc_i32" | "v_mad_u32_u16" | "v_mad_i32_i16")
}

pub(super) fn expression(ins: &Insn) -> Result<Option<String>, String> {
    if !handles(ins) {
        return Ok(None);
    }
    let mad = ins.mnemonic.starts_with("v_mad_");
    let arity = if mad { 3 } else { 2 };
    if !ins.is_complete() || ins.raw.len() < 2 || ins.src.len() < arity
        || !ins.src[..arity].iter().all(|s| matches!(s, Opnd::Vgpr(_))) {
        return Err("unsupported ordinary 32-bit integer source or encoding".into());
    }
    let (w0, w1) = (ins.raw[0], ins.raw[1]);
    // RDNA2 ISA 6.4: op_sel bits 11 and 12 pick the 16-bit half of the two
    // multiplicands. The 32-bit addend and result take no half selection.
    let allowed_select = if mad { 0x1800 } else { 0 };
    if w0 & 0x8700 != 0 || w0 & 0x7800 & !allowed_select != 0 || w1 >> 27 != 0 {
        return Err("unsupported ordinary 32-bit integer modifiers".into());
    }
    let (a, b) = (r(&ins.src[0])?, r(&ins.src[1])?);
    Ok(Some(match ins.mnemonic.as_str() {
        // Without clamp these wrap exactly like the unsigned forms.
        "v_add_nc_i32" => format!("({a} + {b})"),
        "v_sub_nc_i32" => format!("({a} - {b})"),
        _ => {
            let x = half(&ins.src[0], (w0 >> 11) & 1)?;
            let y = half(&ins.src[1], (w0 >> 12) & 1)?;
            let c = r(&ins.src[2])?;
            if ins.mnemonic == "v_mad_u32_u16" {
                format!("({x} * {y} + {c})")
            } else {
                let s = |v: &str| format!("(asint({v} << 16) >> 16)");
                format!("asuint({} * {} + asint({c}))", s(&x), s(&y))
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna2::disassemble;

    fn native(a: u32, b: u32) -> Insn {
        let bytes: Vec<u8> = [a, b].iter().flat_map(|v| v.to_le_bytes()).collect();
        disassemble(&bytes).remove(0)
    }

    #[test]
    fn ordinary_32_bit_forms_accept_vgprs_without_modifiers() {
        // LLVM gfx1030 words for `v7, v4, v5[, v6]`.
        for (first, second, name, want) in [
            (0xd77f_0007u32, 0x0202_0b04u32, "v_add_nc_i32", "(v4 + v5)"),
            (0xd776_0007, 0x0202_0b04, "v_sub_nc_i32", "(v4 - v5)"),
            (0xd773_0007, 0x041a_0b04, "v_mad_u32_u16", "((v4 & 0xffffu) * (v5 & 0xffffu) + v6)"),
            (0xd775_0007, 0x041a_0b04, "v_mad_i32_i16", "asint(v6)"),
        ] {
            let ins = native(first, second);
            assert_eq!(ins.mnemonic, name);
            assert!(expression(&ins).unwrap().unwrap().contains(want), "{name}");
            // clamp, abs, a scalar source and neg remain refused.
            assert!(expression(&native(first | 0x8000, second)).is_err(), "{name} clamp");
            assert!(expression(&native(first | 0x100, second)).is_err(), "{name} abs");
            assert!(expression(&native(first, (second & !511) | 4)).is_err(), "{name} sgpr");
            assert!(expression(&native(first, second | 1 << 29)).is_err(), "{name} neg");
            // Destination or addend half selection has no meaning here.
            assert!(expression(&native(first | 0x4000, second)).is_err(), "{name} dst sel");
        }
        // op_sel:[1,1,0,0] selects the high multiplicand halves.
        let expr = expression(&native(0xd773_1807, 0x041a_0b04)).unwrap().unwrap();
        assert_eq!(expr, "((v4 >> 16) * (v5 >> 16) + v6)");
        assert!(expression(&native(0xd77f_0807, 0x0202_0b04)).is_err());
    }
}
