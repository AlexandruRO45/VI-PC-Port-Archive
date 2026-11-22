from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_u32_f16.

Instruction:
  v_cvt_norm_u32_f16 dst, src0 (VOP1 / VOP3)
  D.u32 = f16_to_unorm(S0.f16)

Architecture:
  - RDNA2 Vector ALU (GFX10) half-precision float to unsigned normalized 32-bit integer conversion.
  - S0: 16-bit IEEE 754 binary16 float (from VGPR halfword, SGPR halfword, or literal/inline const).
  - D:  32-bit unsigned normalized integer in [0, 4294967295] (full 32-bit VGPR).
  - Normalization:
      * Maps floating-point range [0.0, 1.0] to unsigned integer range [0, 4294967295].
      * Scaling factor is 4294967295.0 (2^32 - 1).
      * Rounding mode: round-to-nearest-even (IEEE 754-2008 roundTiesToEven).
      * Clamping: values <= 0.0 (or -inf) clamp to 0 (0x00000000).
                  values >= 1.0 (or +inf) clamp to 4294967295 (0xFFFFFFFF).
      * NaNs (both QNaN and SNaN) convert to 0 (0x00000000).
      * Signed zeros (+0.0, -0.0) convert to 0 (0x00000000).
  - Register model:
      * 32-bit vector registers (VGPRs: v0-v255) and scalar registers (SGPRs: s0-s105).
      * Overwrites the full 32-bit destination register.
      * Source can read either low half (.l [15:0]) or high half (.h [31:16]) via VOP3 op_sel.
  - Modifiers:
      * Input modifiers: abs (|S0|), neg (-S0).
      * Order of application: abs clears sign bit first, then neg inverts sign bit.
"""

from fractions import Fraction
import math
import struct
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary16 architectural bit layout
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
BIAS_F16: int = 15

# Special half-precision bit patterns
POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
QNAN_F16: int = 0x7E00
SNAN_F16: int = 0x7C01

# Standard reference floating-point constants in binary16
ONE_F16: int = 0x3C00            # 1.0
NEG_ONE_F16: int = 0xBC00        # -1.0
TWO_F16: int = 0x4000            # 2.0
NEG_TWO_F16: int = 0xC000        # -2.0
HALF_F16: int = 0x3800           # 0.5
NEG_HALF_F16: int = 0xB800       # -0.5
QUARTER_F16: int = 0x3400        # 0.25
NEG_QUARTER_F16: int = 0xB400    # -0.25
THREE_QUARTERS_F16: int = 0x3A00 # 0.75
FOUR_F16: int = 0x4400           # 4.0
NEG_FOUR_F16: int = 0xC400       # -4.0

MAX_FINITE_F16: int = 0x7BFF     # 65504.0
MIN_FINITE_NEG_F16: int = 0xFBFF # -65504.0
MIN_NORMAL_F16: int = 0x0400     # 2^-14 (~6.103515625e-5)
MAX_SUBNORMAL_F16: int = 0x03FF  # ~6.09755516e-5
MIN_SUBNORMAL_F16: int = 0x0001  # 2^-24 (~5.9604644775e-8)

# 32-bit unsigned normalized integer range constants
UNORM32_MAX: int = 4294967295    # 0xFFFFFFFF (+1.0)
UNORM32_MIN: int = 0             # 0x00000000 (0.0)
UNORM32_SCALE: float = 4294967295.0


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def bits_to_f16(bits: int) -> float:
    """Convert unsigned 16-bit bits into Python float via struct IEEE 754 half."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert Python float into unsigned 16-bit IEEE 754 half bits."""
    if math.isnan(val):
        return QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return NEG_INF_F16 if val < 0.0 else POS_INF_F16


def is_nan_f16(bits: int) -> bool:
    """Check if binary16 bits represent any NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Check if binary16 bits represent +/- infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Check if binary16 bits represent a subnormal number."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Check if binary16 bits represent +0.0 or -0.0."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits16: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 16-bit half float word.
    
    Order of operations: abs clears sign bit (bit 15) first, then neg inverts sign bit (bit 15).
    """
    val = bits16 & MASK16
    if abs_mod:
        val &= ~SIGN_MASK_F16
    if neg_mod:
        val ^= SIGN_MASK_F16
    return val


def f16_to_unorm32(src0_bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 f16_to_unorm32 conversion.
    
    Converts 16-bit float bits to unsigned normalized 32-bit integer bits.
    - NaNs convert to 0.
    - Negative values (<= 0.0) clamp to 0.
    - Values >= 1.0 (and +inf) clamp to 4294967295.
    - Scaled by 4294967295.0 and rounded with round-to-nearest-even.
    Returns unsigned 32-bit integer.
    """
    val_bits = apply_f16_modifiers(src0_bits, abs_mod, neg_mod)
    if is_nan_f16(val_bits):
        return 0
    if is_zero_f16(val_bits):
        return 0
    if is_inf_f16(val_bits):
        return 0 if (val_bits & SIGN_MASK_F16) else UNORM32_MAX

    val_float = bits_to_f16(val_bits)
    if val_float <= 0.0:
        return 0
    if val_float >= 1.0:
        return UNORM32_MAX

    scaled = val_float * UNORM32_SCALE
    rounded = int(round(scaled))
    if rounded > UNORM32_MAX:
        rounded = UNORM32_MAX
    elif rounded < 0:
        rounded = 0
    return u32(rounded)


def oracle_v_cvt_norm_u32_f16(
    src0_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
) -> int:
    """Bit-exact reference oracle for v_cvt_norm_u32_f16."""
    return f16_to_unorm32(src0_bits, abs_mod, neg_mod)


class GpuRegisterState:
    """Architectural register file state model for RDNA2 VALU execution."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx & 0xFF] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx & 0xFF] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs[idx % 106] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx % 106] = val & MASK32


class TestValuCvtNormU32F16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_oracle_zeros(self) -> None:
        self.assertEqual(oracle_v_cvt_norm_u32_f16(POS_ZERO_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_ZERO_F16), 0)

    def test_oracle_unit_boundaries(self) -> None:
        # +1.0 -> 4294967295 (0xFFFFFFFF)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16), 4294967295)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16), 0xFFFFFFFF)
        # -1.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_ONE_F16), 0)

    def test_oracle_half_values(self) -> None:
        # +0.5 * 4294967295 = 2147483647.5 -> rounds to 2147483648 (even) = 0x80000000
        self.assertEqual(oracle_v_cvt_norm_u32_f16(HALF_F16), 2147483648)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(HALF_F16), 0x80000000)
        # +0.25 * 4294967295 = 1073741823.75 -> rounds to 1073741824 = 0x40000000
        self.assertEqual(oracle_v_cvt_norm_u32_f16(QUARTER_F16), 1073741824)
        # -0.5 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_HALF_F16), 0)

    def test_oracle_clamping_extremes(self) -> None:
        # +2.0 clamps to +1.0 -> 4294967295
        self.assertEqual(oracle_v_cvt_norm_u32_f16(TWO_F16), 4294967295)
        # -2.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_TWO_F16), 0)
        # +4.0 clamps to 4294967295, -4.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(FOUR_F16), 4294967295)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_FOUR_F16), 0)
        # Infinities clamp to extremes
        self.assertEqual(oracle_v_cvt_norm_u32_f16(POS_INF_F16), 4294967295)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_INF_F16), 0)

    def test_oracle_nans(self) -> None:
        # Both QNaN and SNaN convert to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(QNAN_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(SNAN_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0x7E01), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0xFE00), 0)

    def test_oracle_modifiers(self) -> None:
        # abs(-1.0) = 1.0 -> 4294967295
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_ONE_F16, abs_mod=True), 4294967295)
        # neg(1.0) = -1.0 -> clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16, neg_mod=True), 0)
        # abs and neg: -|1.0| = -1.0 -> clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16, abs_mod=True, neg_mod=True), 0)

    def test_architectural_register_state(self) -> None:
        state = GpuRegisterState()
        state.set_vgpr(0, 0x12345678)
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        state.set_sgpr(5, 0x9ABCDEF0)
        self.assertEqual(state.get_sgpr(5), 0x9ABCDEF0)




# ---------------------------------------------------------------------------
# AST opcode definitions and typed expressions for v_cvt_norm_u32_f16
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum
import re
from typing import Union

class RegType(Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclass
class RegOperand:
    index: int
    reg_type: RegType
    is_hi: bool = False
    abs_mod: bool = False
    neg_mod: bool = False

    def validate(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range (0..255)")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range (0..105)")
        else:
            raise TypeError(f"Unknown register type: {self.reg_type}")


@dataclass
class ImmediateOperand:
    value: int
    is_f16: bool = True
    abs_mod: bool = False
    neg_mod: bool = False

    def validate(self) -> None:
        pass


@dataclass
class InstructionAST:
    opcode: str
    dst: RegOperand
    src0: Union[RegOperand, ImmediateOperand]
    op_sel: tuple[int, int] = (0, 0)  # (0, src0_hi)
    clamp: bool = False

    def validate(self) -> None:
        if self.opcode != "v_cvt_norm_u32_f16":
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if not isinstance(self.dst, RegOperand):
            raise TypeError("Destination must be a RegOperand")
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"VALU instruction destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination operand cannot have abs or neg modifiers")
        if self.dst.is_hi:
            raise ValueError("Destination is 32-bit dword and cannot have .h swizzle")
        self.dst.validate()
        self.src0.validate()


def _parse_reg_token(token: str, is_dst: bool = False) -> RegOperand:
    tok = token.strip()
    abs_mod = False
    neg_mod = False
    if tok.startswith("-"):
        neg_mod = True
        tok = tok[1:].strip()
    if tok.startswith("|") and tok.endswith("|"):
        abs_mod = True
        tok = tok[1:-1].strip()

    m = re.match(r"^([vs])(\d+)(\.[lh])?$", tok, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid register operand format: '{token}'")
    prefix = m.group(1).lower()
    idx = int(m.group(2))
    sub = m.group(3)
    is_hi = (sub.lower() == ".h") if sub else False
    reg_type = RegType.VGPR if prefix == "v" else RegType.SGPR

    if is_dst and reg_type != RegType.VGPR:
        raise TypeError(f"Destination register must be VGPR, got '{token}'")
    if is_dst and (abs_mod or neg_mod):
        raise ValueError(f"Destination cannot have modifiers: '{token}'")
    if is_dst and is_hi:
        raise ValueError(f"Destination is 32-bit and cannot be .h: '{token}'")

    op = RegOperand(index=idx, reg_type=reg_type, is_hi=is_hi, abs_mod=abs_mod, neg_mod=neg_mod)
    op.validate()
    return op


def _parse_src_token(token: str) -> Union[RegOperand, ImmediateOperand]:
    tok = token.strip()
    abs_mod = False
    neg_mod = False
    if tok.startswith("-"):
        neg_mod = True
        tok = tok[1:].strip()
    if tok.startswith("|") and tok.endswith("|"):
        abs_mod = True
        tok = tok[1:-1].strip()

    if tok.startswith("0x") or tok.startswith("0X"):
        try:
            val = int(tok, 16) & MASK16
            return ImmediateOperand(value=val, is_f16=True, abs_mod=abs_mod, neg_mod=neg_mod)
        except ValueError:
            raise ValueError(f"Invalid hex literal: '{token}'")
    try:
        val_f = float(tok)
        val_bits = f16_to_bits(val_f)
        return ImmediateOperand(value=val_bits, is_f16=True, abs_mod=abs_mod, neg_mod=neg_mod)
    except ValueError:
        pass

    m = re.match(r"^([vs])(\d+)(\.[lh])?$", tok, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid source operand: '{token}'")
    prefix = m.group(1).lower()
    idx = int(m.group(2))
    sub = m.group(3)
    is_hi = (sub.lower() == ".h") if sub else False
    reg_type = RegType.VGPR if prefix == "v" else RegType.SGPR
    op = RegOperand(index=idx, reg_type=reg_type, is_hi=is_hi, abs_mod=abs_mod, neg_mod=neg_mod)
    op.validate()
    return op


def parse_vop_cvt_norm_u32_f16(asm: str) -> InstructionAST:
    """Parse assembly text representation of v_cvt_norm_u32_f16 into AST."""
    line = asm.split(";")[0].split("//")[0].strip()
    if not line:
        raise ValueError("Empty assembly line")

    tokens = line.split(None, 1)
    mnemonic = tokens[0].lower()
    if mnemonic != "v_cvt_norm_u32_f16":
        raise ValueError(f"Invalid mnemonic '{mnemonic}', expected 'v_cvt_norm_u32_f16'")

    if len(tokens) < 2:
        raise ValueError(f"Missing operands for instruction '{asm}'")

    body = tokens[1].strip()

    clamp = False
    if re.search(r"\bclamp\b", body):
        clamp = True
        body = re.sub(r"\bclamp\b", " ", body)

    op_sel = None
    op_sel_match = re.search(r"op_sel:\s*\[\s*([0-1])\s*,\s*([0-1])\s*\]", body)
    if op_sel_match:
        op_sel = (int(op_sel_match.group(1)), int(op_sel_match.group(2)))
        body = body[:op_sel_match.start()] + " " + body[op_sel_match.end():]
    elif "op_sel:" in body:
        raise ValueError(f"Malformed op_sel syntax in '{asm}'")

    parts = [p.strip() for p in body.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Too few operands in '{asm}'")
    if len(parts) > 2:
        raise ValueError(f"Too many operands in '{asm}'")

    dst_op = _parse_reg_token(parts[0], is_dst=True)
    src_words = parts[1].strip().split()
    if not src_words:
        raise ValueError(f"Missing src0 operand in '{asm}'")
    if len(src_words) > 1:
        raise ValueError(f"Extraneous token in instruction: '{src_words[1]}'")

    src_op = _parse_src_token(src_words[0])

    if op_sel is None:
        op_sel = (0, 1 if (isinstance(src_op, RegOperand) and src_op.is_hi) else 0)

    ast = InstructionAST(opcode="v_cvt_norm_u32_f16", dst=dst_op, src0=src_op, op_sel=op_sel, clamp=clamp)
    ast.validate()
    return ast


class TestValuCvtNormU32F16AST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_creation_and_attributes(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.VGPR)
        src = RegOperand(index=1, reg_type=RegType.VGPR, is_hi=True, abs_mod=True)
        ast = InstructionAST(opcode="v_cvt_norm_u32_f16", dst=dst, src0=src, op_sel=(0, 1))
        ast.validate()
        self.assertEqual(ast.opcode, "v_cvt_norm_u32_f16")
        self.assertEqual(ast.dst.index, 0)
        self.assertTrue(ast.src0.is_hi)
        self.assertTrue(ast.src0.abs_mod)
        self.assertEqual(ast.op_sel, (0, 1))

    def test_ast_validation_rejects_sgpr_dst(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.SGPR)
        src = RegOperand(index=1, reg_type=RegType.VGPR)
        ast = InstructionAST(opcode="v_cvt_norm_u32_f16", dst=dst, src0=src)
        with self.assertRaises(TypeError):
            ast.validate()

    def test_ast_validation_rejects_dst_modifiers(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.VGPR, neg_mod=True)
        src = RegOperand(index=1, reg_type=RegType.VGPR)
        ast = InstructionAST(opcode="v_cvt_norm_u32_f16", dst=dst, src0=src)
        with self.assertRaises(ValueError):
            ast.validate()

    def test_ast_validation_rejects_out_of_bounds_registers(self) -> None:
        dst_oob = RegOperand(index=256, reg_type=RegType.VGPR)
        src = RegOperand(index=0, reg_type=RegType.VGPR)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_cvt_norm_u32_f16", dst=dst_oob, src0=src).validate()

        dst = RegOperand(index=0, reg_type=RegType.VGPR)
        src_oob = RegOperand(index=106, reg_type=RegType.SGPR)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_cvt_norm_u32_f16", dst=dst, src0=src_oob).validate()

    def test_parse_basic_asm(self) -> None:
        ast = parse_vop_cvt_norm_u32_f16("v_cvt_norm_u32_f16 v0, v1")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.dst.reg_type, RegType.VGPR)
        self.assertIsInstance(ast.src0, RegOperand)
        self.assertEqual(ast.src0.index, 1)
        self.assertEqual(ast.src0.reg_type, RegType.VGPR)
        self.assertFalse(ast.src0.is_hi)

    def test_parse_halfword_swizzles(self) -> None:
        ast = parse_vop_cvt_norm_u32_f16("v_cvt_norm_u32_f16 v2, v3.h")
        self.assertEqual(ast.dst.index, 2)
        self.assertEqual(ast.src0.index, 3)
        self.assertTrue(ast.src0.is_hi)
        self.assertEqual(ast.op_sel, (0, 1))

    def test_parse_modifiers(self) -> None:
        ast = parse_vop_cvt_norm_u32_f16("v_cvt_norm_u32_f16 v4, -|v5|")
        self.assertTrue(ast.src0.abs_mod)
        self.assertTrue(ast.src0.neg_mod)

    def test_parse_literals_and_sgpr(self) -> None:
        ast_s = parse_vop_cvt_norm_u32_f16("v_cvt_norm_u32_f16 v0, s4")
        self.assertEqual(ast_s.src0.reg_type, RegType.SGPR)
        self.assertEqual(ast_s.src0.index, 4)

        ast_lit = parse_vop_cvt_norm_u32_f16("v_cvt_norm_u32_f16 v0, 1.0")
        self.assertIsInstance(ast_lit.src0, ImmediateOperand)
        self.assertEqual(ast_lit.src0.value, ONE_F16)

        ast_hex = parse_vop_cvt_norm_u32_f16("v_cvt_norm_u32_f16 v0, 0x3C00")
        self.assertIsInstance(ast_hex.src0, ImmediateOperand)
        self.assertEqual(ast_hex.src0.value, ONE_F16)

    def test_parse_op_sel_and_clamp(self) -> None:
        ast = parse_vop_cvt_norm_u32_f16("v_cvt_norm_u32_f16 v0, v1 op_sel:[0, 1] clamp")
        self.assertEqual(ast.op_sel, (0, 1))
        self.assertTrue(ast.clamp)


# ---------------------------------------------------------------------------
# Deterministic scalar instruction interpreter for v_cvt_norm_u32_f16
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU instructions."""

    def __init__(self) -> None:
        self.num_vgprs = 256
        self.num_sgprs = 106
        self.vgprs: list[int] = [0] * self.num_vgprs
        self.sgprs: list[int] = [0] * self.num_sgprs

    def get_vgpr_u32(self, idx: int) -> int:
        return self.vgprs[idx & 0xFF] & MASK32

    def set_vgpr_u32(self, idx: int, val: int) -> int:
        self.vgprs[idx & 0xFF] = val & MASK32
        return self.vgprs[idx & 0xFF]

    def get_vgpr_half(self, idx: int, is_hi: bool) -> int:
        dword = self.get_vgpr_u32(idx)
        return (dword >> 16) & MASK16 if is_hi else dword & MASK16

    def get_sgpr_u32(self, idx: int) -> int:
        return self.sgprs[idx % self.num_sgprs] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> int:
        self.sgprs[idx % self.num_sgprs] = val & MASK32
        return self.sgprs[idx % self.num_sgprs]

    def get_sgpr_half(self, idx: int, is_hi: bool) -> int:
        dword = self.get_sgpr_u32(idx)
        return (dword >> 16) & MASK16 if is_hi else dword & MASK16

    def resolve_src0_half(self, src: Union[RegOperand, ImmediateOperand], src_hi: bool) -> int:
        """Fetch and apply modifiers to source halfword operand."""
        if isinstance(src, ImmediateOperand):
            val = src.value & MASK16
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
        elif isinstance(src, RegOperand):
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
            if src.reg_type == RegType.VGPR:
                val = self.get_vgpr_half(src.index, src_hi)
            else:
                val = self.get_sgpr_half(src.index, src_hi)
        else:
            raise TypeError(f"Unknown source operand type: {type(src)}")
        return apply_f16_modifiers(val, abs_mod, neg_mod)

    def execute(self, ast: InstructionAST) -> int:
        ast.validate()
        if ast.opcode == "v_cvt_norm_u32_f16":
            src0_hi = bool(ast.op_sel[1]) or (isinstance(ast.src0, RegOperand) and ast.src0.is_hi)
            src0_val = self.resolve_src0_half(ast.src0, src0_hi)
            res = oracle_v_cvt_norm_u32_f16(src0_val)
            return self.set_vgpr_u32(ast.dst.index, res)
        raise NotImplementedError(f"Opcode '{ast.opcode}' not implemented in scalar interpreter")

    def execute_asm(self, asm_line: str) -> int:
        ast = parse_vop_cvt_norm_u32_f16(asm_line)
        return self.execute(ast)


class TestValuCvtNormU32F16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_basic_scalar_execution(self) -> None:
        self.interp.set_vgpr_u32(1, ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        self.assertEqual(res, 4294967295)
        self.assertEqual(self.interp.get_vgpr_u32(0), 4294967295)

    def test_scalar_negative_execution(self) -> None:
        self.interp.set_vgpr_u32(1, NEG_ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        self.assertEqual(res, 0)
        self.assertEqual(self.interp.get_vgpr_u32(0), 0)

    def test_scalar_sgpr_source(self) -> None:
        self.interp.set_sgpr_u32(5, HALF_F16)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v2, s5")
        self.assertEqual(res, 2147483648)

    def test_scalar_literal_source(self) -> None:
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v3, 0.5")
        self.assertEqual(res, 2147483648)

    def test_scalar_op_sel_routing(self) -> None:
        # v4 holds high half = 1.0 (0x3C00), low half = -1.0 (0xBC00)
        self.interp.set_vgpr_u32(4, (ONE_F16 << 16) | NEG_ONE_F16)

        # Read high half (1.0)
        res_hi = self.interp.execute_asm("v_cvt_norm_u32_f16 v5, v4.h")
        self.assertEqual(res_hi, 4294967295)

        # Read low half (-1.0) -> clamps to 0
        res_lo = self.interp.execute_asm("v_cvt_norm_u32_f16 v6, v4.l")
        self.assertEqual(res_lo, 0)


# ---------------------------------------------------------------------------
# SIMD multi-lane divergence and 32/64-bit EXEC masking
# ---------------------------------------------------------------------------

class SIMDVALUInterpreter:
    """SIMD multi-lane execution model supporting Wave32 and Wave64 with EXEC masking."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size}. Must be 32 or 64.")
        self.wave_size = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.num_vgprs = 256
        self.num_sgprs = 106
        self.vgprs: list[list[int]] = [[0] * wave_size for _ in range(self.num_vgprs)]
        self.sgprs: list[int] = [0] * self.num_sgprs

    def get_vgpr_u32(self, reg: int, lane: int) -> int:
        return self.vgprs[reg & 0xFF][lane] & MASK32

    def set_vgpr_u32(self, reg: int, lane: int, val: int) -> int:
        self.vgprs[reg & 0xFF][lane] = val & MASK32
        return self.vgprs[reg & 0xFF][lane]

    def get_vgpr_half(self, reg: int, lane: int, is_hi: bool) -> int:
        dword = self.get_vgpr_u32(reg, lane)
        return (dword >> 16) & MASK16 if is_hi else dword & MASK16

    def get_sgpr_u32(self, idx: int) -> int:
        return self.sgprs[idx % self.num_sgprs] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> None:
        self.sgprs[idx % self.num_sgprs] = val & MASK32
        return self.sgprs[idx % self.num_sgprs]

    def get_sgpr_half(self, idx: int, is_hi: bool) -> int:
        dword = self.get_sgpr_u32(idx)
        return (dword >> 16) & MASK16 if is_hi else dword & MASK16

    def resolve_src0_half(self, src: Union[RegOperand, ImmediateOperand], lane: int, src_hi: bool) -> int:
        if isinstance(src, ImmediateOperand):
            val = src.value & MASK16
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
        elif isinstance(src, RegOperand):
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
            if src.reg_type == RegType.VGPR:
                val = self.get_vgpr_half(src.index, lane, src_hi)
            else:
                val = self.get_sgpr_half(src.index, src_hi)
        else:
            raise TypeError(f"Unknown source operand type: {type(src)}")
        return apply_f16_modifiers(val, abs_mod, neg_mod)

    def execute(self, ast: InstructionAST) -> list[int]:
        ast.validate()
        if ast.opcode == "v_cvt_norm_u32_f16":
            src0_hi = bool(ast.op_sel[1]) or (isinstance(ast.src0, RegOperand) and ast.src0.is_hi)
            results = []
            for lane in range(self.wave_size):
                lane_active = bool(self.exec_mask & (1 << lane))
                if lane_active:
                    src0_val = self.resolve_src0_half(ast.src0, lane, src0_hi)
                    res = oracle_v_cvt_norm_u32_f16(src0_val)
                    new_val = self.set_vgpr_u32(ast.dst.index, lane, res)
                    results.append(new_val)
                else:
                    results.append(self.get_vgpr_u32(ast.dst.index, lane))
            return results
        raise NotImplementedError(f"Opcode '{ast.opcode}' not implemented in SIMD interpreter")

    def execute_asm(self, asm_line: str) -> list[int]:
        ast = parse_vop_cvt_norm_u32_f16(asm_line)
        return self.execute(ast)


class TestValuCvtNormU32F16SIMD(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        for lane in range(32):
            simd.set_vgpr_u32(1, lane, ONE_F16)
        res = simd.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        self.assertEqual(len(res), 32)
        for lane in range(32):
            self.assertEqual(res[lane], 4294967295)
            self.assertEqual(simd.get_vgpr_u32(0, lane), 4294967295)

    def test_wave64_uniform_execution(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=64)
        for lane in range(64):
            simd.set_vgpr_u32(1, lane, NEG_ONE_F16)
        res = simd.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        self.assertEqual(len(res), 64)
        for lane in range(64):
            self.assertEqual(res[lane], 0)
            self.assertEqual(simd.get_vgpr_u32(0, lane), 0)

    def test_exec_mask_selective_divergence(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=64)
        simd.exec_mask = 0x55555555_55555555  # Even lanes active, odd lanes inactive
        for lane in range(64):
            simd.set_vgpr_u32(0, lane, 0xDEADBEEF)
            simd.set_vgpr_u32(1, lane, ONE_F16)

        res = simd.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(res[lane], 4294967295)
                self.assertEqual(simd.get_vgpr_u32(0, lane), 4294967295)
            else:
                self.assertEqual(res[lane], 0xDEADBEEF)
                self.assertEqual(simd.get_vgpr_u32(0, lane), 0xDEADBEEF)

    def test_lane_divergent_values(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        test_inputs = [
            (POS_ZERO_F16, 0),
            (HALF_F16, 2147483648),
            (ONE_F16, 4294967295),
            (NEG_ONE_F16, 0),
            (QNAN_F16, 0),
            (TWO_F16, 4294967295),
            (NEG_TWO_F16, 0),
            (POS_INF_F16, 4294967295),
        ]
        for lane in range(32):
            inp, _ = test_inputs[lane % len(test_inputs)]
            simd.set_vgpr_u32(1, lane, inp)

        res = simd.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        for lane in range(32):
            _, expected = test_inputs[lane % len(test_inputs)]
            self.assertEqual(res[lane], expected)

    def test_zero_exec_mask_no_modification(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        simd.exec_mask = 0
        for lane in range(32):
            simd.set_vgpr_u32(0, lane, 0x12345678)
            simd.set_vgpr_u32(1, lane, ONE_F16)
        res = simd.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        for lane in range(32):
            self.assertEqual(res[lane], 0x12345678)
            self.assertEqual(simd.get_vgpr_u32(0, lane), 0x12345678)


# ---------------------------------------------------------------------------
# Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuCvtNormU32F16Aliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_in_place_self_aliasing(self) -> None:
        # v0 = ONE_F16 in low half
        self.interp.set_vgpr_u32(0, ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v0, v0")
        self.assertEqual(res, 4294967295)
        self.assertEqual(self.interp.get_vgpr_u32(0), 4294967295)

    def test_in_place_self_aliasing_high_half(self) -> None:
        # v0 = (HALF_F16 << 16) | 0x1234
        self.interp.set_vgpr_u32(0, (HALF_F16 << 16) | 0x1234)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v0, v0.h")
        self.assertEqual(res, 2147483648)
        self.assertEqual(self.interp.get_vgpr_u32(0), 2147483648)

    def test_in_place_self_aliasing_with_abs_modifier(self) -> None:
        self.interp.set_vgpr_u32(0, NEG_ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v0, |v0|")
        self.assertEqual(res, 4294967295)
        self.assertEqual(self.interp.get_vgpr_u32(0), 4294967295)

    def test_in_place_self_aliasing_with_neg_modifier(self) -> None:
        self.interp.set_vgpr_u32(0, ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v0, -v0")
        self.assertEqual(res, 0)
        self.assertEqual(self.interp.get_vgpr_u32(0), 0)

    def test_distinct_registers_non_interference(self) -> None:
        self.interp.set_vgpr_u32(0, 0xCAFEBABE)
        self.interp.set_vgpr_u32(1, HALF_F16)
        res = self.interp.execute_asm("v_cvt_norm_u32_f16 v0, v1")
        self.assertEqual(res, 2147483648)
        self.assertEqual(self.interp.get_vgpr_u32(0), 2147483648)
        self.assertEqual(self.interp.get_vgpr_u32(1), HALF_F16)

    def test_multi_instruction_aliased_pipeline(self) -> None:
        self.interp.set_vgpr_u32(0, (HALF_F16 << 16) | ONE_F16)
        self.interp.set_vgpr_u32(1, NEG_TWO_F16)

        # Step 1: convert v0 low half into v2
        self.interp.execute_asm("v_cvt_norm_u32_f16 v2, v0.l")
        self.assertEqual(self.interp.get_vgpr_u32(2), 4294967295)

        # Step 2: convert v0 high half into v0 in-place
        self.interp.execute_asm("v_cvt_norm_u32_f16 v0, v0.h")
        self.assertEqual(self.interp.get_vgpr_u32(0), 2147483648)


# ---------------------------------------------------------------------------
# Condition codes, status flags, and carry/borrow generation
# ---------------------------------------------------------------------------

@dataclass
class FPStatusFlags:
    """IEEE 754 floating-point status exception flags."""
    invalid: bool = False
    div_by_zero: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False

    def merge(self, other: FPStatusFlags) -> None:
        self.invalid |= other.invalid
        self.div_by_zero |= other.div_by_zero
        self.overflow |= other.overflow
        self.underflow |= other.underflow
        self.inexact |= other.inexact

    def to_dict(self) -> dict[str, bool]:
        return {
            "invalid": self.invalid,
            "div_by_zero": self.div_by_zero,
            "overflow": self.overflow,
            "underflow": self.underflow,
            "inexact": self.inexact,
        }


@dataclass
class IntConditionCodes:
    """Condition codes and carry/borrow generation for 32-bit unsigned integer result."""
    zero: bool = False
    max_pos: bool = False
    overflow: bool = False
    carry: bool = False
    borrow: bool = False


def calc_cvt_norm_status_flags(src0_bits: int, res_bits: int) -> FPStatusFlags:
    """Calculate IEEE 754 exception flags for v_cvt_norm_u32_f16 conversion."""
    flags = FPStatusFlags()
    s0 = src0_bits & MASK16

    # NaNs set invalid flag
    if is_nan_f16(s0):
        flags.invalid = True
        return flags

    # Infinities set overflow and inexact
    if is_inf_f16(s0):
        flags.overflow = True
        flags.inexact = True
        return flags

    if is_zero_f16(s0):
        return flags

    val_f = bits_to_f16(s0)
    # Range check: input outside [0.0, 1.0] sets overflow & inexact
    if val_f > 1.0 or val_f < 0.0:
        flags.overflow = True
        flags.inexact = True
        return flags

    # Underflow: non-zero input float rounded to zero
    if (res_bits & MASK32) == 0 and not is_zero_f16(s0):
        flags.underflow = True
        flags.inexact = True
        return flags

    # Inexact check: if val_f * 4294967295.0 has fractional part
    scaled = val_f * UNORM32_SCALE
    if scaled != float(int(round(scaled))):
        flags.inexact = True

    return flags


def calc_cvt_norm_condition_codes(src0_bits: int, res_bits: int) -> IntConditionCodes:
    """Calculate integer condition codes and boundary carry/borrow flags."""
    is_zero = (res_bits & MASK32) == 0
    is_max = (res_bits & MASK32) == UNORM32_MAX

    s0 = src0_bits & MASK16
    overflow = False
    carry = False
    borrow = False

    if is_nan_f16(s0):
        pass
    elif is_inf_f16(s0):
        overflow = True
        if not (s0 & SIGN_MASK_F16):
            carry = True
        else:
            borrow = True
    else:
        val_f = bits_to_f16(s0)
        if val_f > 1.0:
            overflow = True
            carry = True
        elif val_f < 0.0:
            overflow = True
            borrow = True

    return IntConditionCodes(
        zero=is_zero,
        max_pos=is_max,
        overflow=overflow,
        carry=carry,
        borrow=borrow,
    )


class TestValuCvtNormU32F16StatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow generation."""

    def test_invalid_flag_on_nans(self) -> None:
        flags_qnan = calc_cvt_norm_status_flags(QNAN_F16, 0)
        self.assertTrue(flags_qnan.invalid)
        self.assertFalse(flags_qnan.overflow)

        flags_snan = calc_cvt_norm_status_flags(SNAN_F16, 0)
        self.assertTrue(flags_snan.invalid)

    def test_overflow_and_inexact_on_out_of_range(self) -> None:
        flags_pos = calc_cvt_norm_status_flags(TWO_F16, 4294967295)
        self.assertTrue(flags_pos.overflow)
        self.assertTrue(flags_pos.inexact)
        self.assertFalse(flags_pos.invalid)

        flags_neg = calc_cvt_norm_status_flags(NEG_TWO_F16, 0)
        self.assertTrue(flags_neg.overflow)
        self.assertTrue(flags_neg.inexact)

        flags_inf = calc_cvt_norm_status_flags(POS_INF_F16, 4294967295)
        self.assertTrue(flags_inf.overflow)
        self.assertTrue(flags_inf.inexact)

    def test_inexact_flag_on_fractional_scaled(self) -> None:
        # 1/3 ~ 0.333251953125 (0x3555)
        res = oracle_v_cvt_norm_u32_f16(0x3555)
        flags = calc_cvt_norm_status_flags(0x3555, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.overflow)

    def test_exact_conversions_no_flags(self) -> None:
        # 0.0 -> 0 (exact)
        flags_zero = calc_cvt_norm_status_flags(POS_ZERO_F16, 0)
        self.assertEqual(flags_zero.to_dict(), {
            "invalid": False, "div_by_zero": False, "overflow": False, "underflow": False, "inexact": False
        })

        # 1.0 -> 4294967295 (exact)
        flags_one = calc_cvt_norm_status_flags(ONE_F16, 4294967295)
        self.assertFalse(flags_one.overflow)
        self.assertFalse(flags_one.inexact)

    def test_condition_codes_zero_and_max(self) -> None:
        cc_zero = calc_cvt_norm_condition_codes(POS_ZERO_F16, 0)
        self.assertTrue(cc_zero.zero)
        self.assertFalse(cc_zero.max_pos)

        cc_max = calc_cvt_norm_condition_codes(ONE_F16, 4294967295)
        self.assertFalse(cc_max.zero)
        self.assertTrue(cc_max.max_pos)

    def test_condition_codes_saturation_and_carry_borrow(self) -> None:
        # +2.0 clamps to +1.0: sets max_pos, overflow, carry
        cc_pos = calc_cvt_norm_condition_codes(TWO_F16, 4294967295)
        self.assertTrue(cc_pos.max_pos)
        self.assertTrue(cc_pos.overflow)
        self.assertTrue(cc_pos.carry)
        self.assertFalse(cc_pos.borrow)

        # -2.0 clamps to 0: sets zero, overflow, borrow
        cc_neg = calc_cvt_norm_condition_codes(NEG_TWO_F16, 0)
        self.assertTrue(cc_neg.zero)
        self.assertTrue(cc_neg.overflow)
        self.assertTrue(cc_neg.borrow)
        self.assertFalse(cc_neg.carry)


# ---------------------------------------------------------------------------
# IEEE 754 edge cases, boundary extremes, subnormals, and NaN handling
# ---------------------------------------------------------------------------

class TestValuCvtNormU32F16IEEE754Boundaries(unittest.TestCase):
    """Comprehensive IEEE 754 boundary, subnormal, and NaN handling tests."""

    def test_signed_zeros(self) -> None:
        self.assertEqual(oracle_v_cvt_norm_u32_f16(POS_ZERO_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_ZERO_F16), 0)

    def test_infinities(self) -> None:
        # +inf clamps to +1.0 (4294967295 = 0xFFFFFFFF)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(POS_INF_F16), 4294967295)
        # -inf clamps to 0 (0x00000000)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_INF_F16), 0)

    def test_quiet_and_signaling_nans_with_payloads(self) -> None:
        nans = [
            QNAN_F16,
            SNAN_F16,
            0x7C01, 0x7C10, 0x7D55, 0x7CFF,  # SNaNs
            0x7E00, 0x7E01, 0x7EFF, 0x7FFF,  # QNaNs
            0xFC01, 0xFC10, 0xFD55, 0xFCFF,  # Negative SNaNs
            0xFE00, 0xFE01, 0xFEFF, 0xFFFF,  # Negative QNaNs
        ]
        for n in nans:
            res = oracle_v_cvt_norm_u32_f16(n)
            self.assertEqual(res, 0, f"Expected 0 for NaN {n:#06x}, got {res:#06x}")

    def test_finite_extremes_and_saturation(self) -> None:
        # Max positive finite half float (65504.0) -> clamps to 4294967295
        self.assertEqual(oracle_v_cvt_norm_u32_f16(MAX_FINITE_F16), 4294967295)
        # Min negative finite half float (-65504.0) -> clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(MIN_FINITE_NEG_F16), 0)

        # Values just above 1.0 (0x3C01) clamp to 4294967295
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0x3C01), 4294967295)
        # Negative values clamp to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0xBC01), 0)

    def test_subnormal_precision_and_scaling(self) -> None:
        # Smallest positive subnormal (2^-24) * 4294967295 = 255.99999994 -> rounds to 256
        self.assertEqual(oracle_v_cvt_norm_u32_f16(MIN_SUBNORMAL_F16), 256)
        # Smallest negative subnormal (-2^-24) clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0x8001), 0)

        # 0x0100 (2^-16) * 4294967295 = 65535.99998 -> rounds to 65536
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0x0100), 65536)

        # Largest subnormal (0x03FF) * 4294967295 = 261887.9999 -> rounds to 261888
        self.assertEqual(oracle_v_cvt_norm_u32_f16(MAX_SUBNORMAL_F16), 261888)

        # Minimum normal (0x0400 = 2^-14) * 4294967295 = 262143.9999 -> rounds to 262144
        self.assertEqual(oracle_v_cvt_norm_u32_f16(MIN_NORMAL_F16), 262144)

    def test_ties_to_even_half_precision(self) -> None:
        # 0.5 * 4294967295 = 2147483647.5 -> ties to even integer 2147483648 (0x80000000)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(HALF_F16), 2147483648)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(HALF_F16), 0x80000000)

    def test_boundary_values_near_unit(self) -> None:
        # 0x3BFF (0.99951171875) * 4294967295 = 4292870143.0004883 -> 4292870143 (0xFFDFFFFF)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0x3BFF), 4292870143)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0x3BFF), 0xFFDFFFFF)
        # 0x3C00 (1.0) -> 4294967295
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16), 4294967295)

if __name__ == "__main__":
    unittest.main()
