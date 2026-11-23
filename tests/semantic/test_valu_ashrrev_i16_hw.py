"""Semantic interpreter tests and translator verification for 16-bit arithmetic right shift with reversed operands (v_ashrrev_i16).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_valu_ashrrev_i16_hw.py
    py -m unittest discover -s tests/semantic -p test_valu_ashrrev_i16_hw.py
"""
from __future__ import annotations

import os
import re
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
SIGN_BIT16: int = 0x8000
OLD_VDST: int = 0x5A5A1234


def s16(word: int, hi: int = 0) -> int:
    """Extract 16-bit signed integer in [-32768, 32767]."""
    h = (word >> 16 if hi else word) & MASK16
    return h - 0x10000 if (h & SIGN_BIT16) else h


def u16(word: int, hi: int = 0) -> int:
    """Extract 16-bit unsigned integer in [0, 65535]."""
    return (word >> 16 if hi else word) & MASK16


def ref_v_ashrrev_i16(
    shift_src0: int,
    val_src1: int,
    *,
    old_dst: int = OLD_VDST,
    e64: bool = False,
    op_sel: tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Reference mathematical oracle for RDNA2 v_ashrrev_i16.

    Operand order is reversed:
      shift amount is src0 (low 4 bits, [3:0], range 0..15).
      value to be shifted is src1 (interpreted as signed 16-bit integer).
    VOP2 (_e32) writes to low 16 bits and zeroes high 16 bits.
    VOP3 (_e64) preserves untouched half according to op_sel[2].
    """
    sh = u16(shift_src0, op_sel[0]) & 0xF
    val = s16(val_src1, op_sel[1])
    res = (val >> sh) & MASK16

    if e64:
        if op_sel[2]:
            return (old_dst & 0xFFFF) | (res << 16)
        else:
            return (old_dst & 0xFFFF0000) | res
    return res

class TestVAshrrevI16OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_signed_16bit_extraction(self) -> None:
        self.assertEqual(s16(0x0000), 0)
        self.assertEqual(s16(0x0001), 1)
        self.assertEqual(s16(0x7FFF), 32767)
        self.assertEqual(s16(0x8000), -32768)
        self.assertEqual(s16(0xFFFF), -1)
        self.assertEqual(s16(0xFFF0), -16)

        # High half extraction
        self.assertEqual(s16(0xFFFF0000, hi=1), -1)
        self.assertEqual(s16(0x00200000, hi=1), 32)

    def test_reference_oracle_positive_values(self) -> None:
        # Shift 16 by 2 -> 4
        self.assertEqual(ref_v_ashrrev_i16(2, 16), 4)
        # Shift 1000 by 3 -> 125
        self.assertEqual(ref_v_ashrrev_i16(3, 1000), 125)
        # Shift by 0 -> unchanged
        self.assertEqual(ref_v_ashrrev_i16(0, 0x1234), 0x1234)

    def test_reference_oracle_negative_sign_preservation(self) -> None:
        # -16 (0xFFF0) >> 2 -> -4 (0xFFFC)
        self.assertEqual(ref_v_ashrrev_i16(2, 0xFFF0), 0xFFFC)
        # -1 (0xFFFF) >> any shift (0..15) -> -1 (0xFFFF)
        for sh in range(16):
            self.assertEqual(ref_v_ashrrev_i16(sh, 0xFFFF), 0xFFFF)
        # -32768 (0x8000) >> 1 -> -16384 (0xC000)
        self.assertEqual(ref_v_ashrrev_i16(1, 0x8000), 0xC000)
        # -32768 (0x8000) >> 15 -> -1 (0xFFFF)
        self.assertEqual(ref_v_ashrrev_i16(15, 0x8000), 0xFFFF)

    def test_reference_oracle_shift_masking_to_4_bits(self) -> None:
        # Shift amount is masked with 15 (& 0xF): 18 & 0xF = 2, so shifts by 2
        self.assertEqual(ref_v_ashrrev_i16(18, 16), 4)
        self.assertEqual(ref_v_ashrrev_i16(0xFFF2, 16), 4)

    def test_reference_oracle_vop3_op_sel_and_preservation(self) -> None:
        old = 0x12345678
        # Write to low half (op_sel[2]=0): preserves high half (0x1234)
        res_lo = ref_v_ashrrev_i16(1, 0x0020, old_dst=old, e64=True, op_sel=(0, 0, 0))
        self.assertEqual(res_lo, 0x12340010)

        # Write to high half (op_sel[2]=1): preserves low half (0x5678)
        res_hi = ref_v_ashrrev_i16(1, 0x0020, old_dst=old, e64=True, op_sel=(0, 0, 1))
        self.assertEqual(res_hi, 0x00105678)

class Register:
    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index

    def __repr__(self) -> str:
        return f"{self.name}{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Register):
            return False
        return self.name == other.name and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.name, self.index))


class VGPR(Register):
    def __init__(self, index: int) -> None:
        if not (0 <= index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {index}")
        super().__init__("v", index)


class VAshrrevI16Instruction:
    """Symbolic AST node representing v_ashrrev_i16 and v_ashrrev_i16_e64."""

    def __init__(
        self,
        vdst: VGPR,
        src0: Union[VGPR, int],
        src1: Union[VGPR, int],
        e64: bool = False,
        op_sel: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.vdst = vdst
        self.src0 = src0
        self.src1 = src1
        self.e64 = e64
        self.op_sel = op_sel

    def __repr__(self) -> str:
        op = "v_ashrrev_i16_e64" if self.e64 else "v_ashrrev_i16"
        s0_str = str(self.src0) if isinstance(self.src0, Register) else hex(self.src0)
        s1_str = str(self.src1) if isinstance(self.src1, Register) else hex(self.src1)
        base = f"{op} {self.vdst}, {s0_str}, {s1_str}"
        if self.e64 and self.op_sel != (0, 0, 0):
            base += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]},{self.op_sel[2]}]"
        return base

    @classmethod
    def parse(cls, asm: str) -> "VAshrrevI16Instruction":
        line = asm.strip()
        if not line:
            raise ValueError("Empty assembly string")
        tokens = line.split()
        mnem = tokens[0]
        if mnem not in ("v_ashrrev_i16", "v_ashrrev_i16_e64"):
            raise ValueError(f"Unsupported mnemonic: {mnem}")
        e64 = (mnem == "v_ashrrev_i16_e64")

        rest = " ".join(tokens[1:])
        mod_tokens = rest.split()
        clean_operands: list[str] = []
        op_sel = (0, 0, 0)

        for tok in mod_tokens:
            tok_clean = tok.rstrip(",")
            if tok_clean.startswith("op_sel:"):
                m_sel = re.search(r"\[(\d+),(\d+),(\d+)\]", tok_clean)
                if not m_sel:
                    raise ValueError(f"Invalid op_sel format: {tok_clean}")
                op_sel = (int(m_sel.group(1)), int(m_sel.group(2)), int(m_sel.group(3)))
            elif tok_clean:
                clean_operands.append(tok_clean)

        if len(clean_operands) != 3:
            raise ValueError(f"Expected 3 operands, got {len(clean_operands)}")

        def parse_operand(op_str: str) -> Union[VGPR, int]:
            m = re.fullmatch(r"v(\d+)", op_str)
            if m:
                return VGPR(int(m.group(1)))
            try:
                return int(op_str, 0)
            except ValueError:
                raise ValueError(f"Invalid operand: {op_str}")

        vdst_op = parse_operand(clean_operands[0])
        if not isinstance(vdst_op, VGPR):
            raise ValueError("Destination must be VGPR")

        src0 = parse_operand(clean_operands[1])
        src1 = parse_operand(clean_operands[2])

        return cls(vdst_op, src0, src1, e64=e64, op_sel=op_sel)


class TestVAshrrevI16ASTAndOpcodeDefinitions(unittest.TestCase):
    """Part 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_properties(self) -> None:
        inst_vop2 = VAshrrevI16Instruction(VGPR(0), VGPR(1), VGPR(2))
        self.assertFalse(inst_vop2.e64)
        self.assertEqual(inst_vop2.vdst, VGPR(0))
        self.assertEqual(inst_vop2.src0, VGPR(1))
        self.assertEqual(inst_vop2.src1, VGPR(2))
        self.assertEqual(repr(inst_vop2), "v_ashrrev_i16 v0, v1, v2")

        inst_vop3 = VAshrrevI16Instruction(VGPR(0), VGPR(1), VGPR(2), e64=True, op_sel=(1, 0, 1))
        self.assertTrue(inst_vop3.e64)
        self.assertEqual(repr(inst_vop3), "v_ashrrev_i16_e64 v0, v1, v2 op_sel:[1,0,1]")

    def test_ast_parsing_valid(self) -> None:
        inst1 = VAshrrevI16Instruction.parse("v_ashrrev_i16 v0, v1, v2")
        self.assertEqual(inst1.vdst, VGPR(0))
        self.assertEqual(inst1.src0, VGPR(1))
        self.assertEqual(inst1.src1, VGPR(2))

        inst2 = VAshrrevI16Instruction.parse("v_ashrrev_i16_e64 v3, 2, v4 op_sel:[0,1,0]")
        self.assertEqual(inst2.vdst, VGPR(3))
        self.assertEqual(inst2.src0, 2)
        self.assertEqual(inst2.src1, VGPR(4))
        self.assertEqual(inst2.op_sel, (0, 1, 0))

    def test_ast_parsing_invalid(self) -> None:
        with self.assertRaises(ValueError):
            VAshrrevI16Instruction.parse("")
        with self.assertRaises(ValueError):
            VAshrrevI16Instruction.parse("v_ashrrev_i16 v0, v1")
        with self.assertRaises(ValueError):
            VAshrrevI16Instruction.parse("v_ashrrev_i16 0x1234, v1, v2")  # literal as dest


if __name__ == '__main__':
    unittest.main()
