from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_i16_f16.

Instruction:
  v_cvt_norm_i16_f16 dst, src0 (VOP1 / VOP3)
  D.i16 = f16_to_snorm(S0.f16)

Architecture:
  - RDNA2 Vector ALU (GFX10) half-precision float to signed normalized 16-bit integer conversion.
  - S0: 16-bit IEEE 754 binary16 float (halfword from VGPR, SGPR, or inline constant).
  - D:  16-bit signed normalized integer in [-32767, 32767] (halfword in VGPR).
  - Normalization:
      * Maps floating-point range [-1.0, 1.0] to integer range [-32767, 32767].
      * Scaling factor is 32767.0 (2^(b-1) - 1 where b=16).
      * Rounding mode: round-to-nearest-even (IEEE 754-2008 roundTiesToEven).
      * Clamping: values > 1.0 (or +inf) clamp to 32767 (0x7FFF).
                  values < -1.0 (or -inf) clamp to -32767 (0x8001).
      * NaNs (both QNaN and SNaN) convert to 0 (0x0000).
      * Signed zeros (+0.0, -0.0) convert to 0 (0x0000).
  - Register model:
      * 32-bit vector registers (VGPRs: v0-v255) and scalar registers (SGPRs: s0-s105).
      * Halfword addressing: low half (.l [15:0]) or high half (.h [31:16]).
      * VOP1 destination write preserves the untouched 16-bit half of the destination 32-bit VGPR.
      * VOP3 op_sel allows selecting lo/hi half independently for destination and source.
  - Modifiers:
      * Input modifiers: abs (|S0|), neg (-S0).
      * Order of application: abs clears sign bit first, then neg inverts sign bit.
"""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
import re
import struct
import unittest
from typing import Union

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
FOUR_F16: int = 0x4400           # 4.0
NEG_FOUR_F16: int = 0xC400       # -4.0
THREE_F16: int = 0x4200          # 3.0

MAX_FINITE_F16: int = 0x7BFF     # 65504.0
MIN_FINITE_NEG_F16: int = 0xFBFF # -65504.0
MIN_NORMAL_F16: int = 0x0400     # 2^-14 (~6.103515625e-5)
MAX_SUBNORMAL_F16: int = 0x03FF  # ~6.09755516e-5
MIN_SUBNORMAL_F16: int = 0x0001  # 2^-24 (~5.9604644775e-8)

# 16-bit signed normalized integer range constants
SNORM16_MAX: int = 32767         # 0x7FFF (+1.0)
SNORM16_MIN: int = -32767        # 0x8001 (-1.0)
SNORM16_SCALE: float = 32767.0


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned halfword as signed two's complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & 0x8000) else val


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def split_halves(val32: int) -> tuple[int, int]:
    """Split 32-bit dword into (low_half, high_half)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def join_halves(lo: int, hi: int) -> int:
    """Join (low_half, high_half) into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


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


def f16_to_snorm16(src0_bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 f16_to_snorm conversion.
    
    Converts 16-bit half float bits to signed normalized 16-bit integer bits.
    - NaNs convert to 0.
    - Clamped to [-1.0, 1.0].
    - Values > 1.0 (and +inf) clamp to 32767.
    - Values < -1.0 (and -inf) clamp to -32767.
    - Scaled by 32767.0 and rounded with round-to-nearest-even.
    Returns unsigned 16-bit representation (two's complement for negatives).
    """
    val_bits = apply_f16_modifiers(src0_bits, abs_mod, neg_mod)
    if is_nan_f16(val_bits):
        return 0
    if is_zero_f16(val_bits):
        return 0
    if is_inf_f16(val_bits):
        return u16(SNORM16_MIN if (val_bits & SIGN_MASK_F16) else SNORM16_MAX)

    val_float = bits_to_f16(val_bits)
    if val_float >= 1.0:
        return u16(SNORM16_MAX)
    if val_float <= -1.0:
        return u16(SNORM16_MIN)

    # Scale and round-to-nearest-even
    scaled = val_float * SNORM16_SCALE
    rounded = int(round(scaled))
    # Safety clamp to [-32767, 32767]
    if rounded > SNORM16_MAX:
        rounded = SNORM16_MAX
    elif rounded < SNORM16_MIN:
        rounded = SNORM16_MIN
    return u16(rounded)


def oracle_v_cvt_norm_i16_f16(
    src0_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
) -> int:
    """Bit-exact reference oracle for v_cvt_norm_i16_f16."""
    return f16_to_snorm16(src0_bits, abs_mod, neg_mod)


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


# ---------------------------------------------------------------------------
# AST opcode definitions and typed expressions for v_cvt_norm_i16_f16
# ---------------------------------------------------------------------------

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
    op_sel: tuple[int, int] = (0, 0)  # (dst_hi, src0_hi)
    clamp: bool = False

    def validate(self) -> None:
        if self.opcode != "v_cvt_norm_i16_f16":
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if not isinstance(self.dst, RegOperand):
            raise TypeError("Destination must be a RegOperand")
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"VALU instruction destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination operand cannot have abs or neg modifiers")
        if not (isinstance(self.op_sel, (tuple, list)) and len(self.op_sel) == 2):
            raise ValueError("op_sel must be a 2-tuple")
        if self.op_sel[0] not in (0, 1) or self.op_sel[1] not in (0, 1):
            raise ValueError(f"op_sel values must be 0 or 1, got {self.op_sel}")
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


def parse_vop_cvt_norm_i16_f16(asm: str) -> InstructionAST:
    """Parse assembly text representation of v_cvt_norm_i16_f16 into AST."""
    line = asm.split(";")[0].split("//")[0].strip()
    if not line:
        raise ValueError("Empty assembly line")

    tokens = line.split(None, 1)
    mnemonic = tokens[0].lower()
    if mnemonic != "v_cvt_norm_i16_f16":
        raise ValueError(f"Invalid mnemonic '{mnemonic}', expected 'v_cvt_norm_i16_f16'")

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
        op_sel = (
            1 if dst_op.is_hi else 0,
            1 if (isinstance(src_op, RegOperand) and src_op.is_hi) else 0,
        )

    ast = InstructionAST(opcode="v_cvt_norm_i16_f16", dst=dst_op, src0=src_op, op_sel=op_sel, clamp=clamp)
    ast.validate()
    return ast


class TestValuCvtNormI16F16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_oracle_zeros(self) -> None:
        self.assertEqual(oracle_v_cvt_norm_i16_f16(POS_ZERO_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_ZERO_F16), 0)

    def test_oracle_unit_boundaries(self) -> None:
        # +1.0 -> 32767 (0x7FFF)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(ONE_F16), 32767)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(ONE_F16), 0x7FFF)
        # -1.0 -> -32767 (0x8001 in two's complement u16)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_ONE_F16), u16(-32767))
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_ONE_F16), 0x8001)

    def test_oracle_half_values(self) -> None:
        # +0.5 * 32767 = 16383.5 -> rounds to 16384 (even)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(HALF_F16), 16384)
        # -0.5 * 32767 = -16383.5 -> rounds to -16384 (even)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_HALF_F16), u16(-16384))

    def test_oracle_clamping_extremes(self) -> None:
        # +2.0 clamps to +1.0 -> 32767
        self.assertEqual(oracle_v_cvt_norm_i16_f16(TWO_F16), 32767)
        # -2.0 clamps to -1.0 -> -32767
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_TWO_F16), u16(-32767))
        # +4.0, -4.0 clamp
        self.assertEqual(oracle_v_cvt_norm_i16_f16(FOUR_F16), 32767)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_FOUR_F16), u16(-32767))
        # Infinities clamp to extremes
        self.assertEqual(oracle_v_cvt_norm_i16_f16(POS_INF_F16), 32767)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_INF_F16), u16(-32767))

    def test_oracle_nans(self) -> None:
        # Both QNaN and SNaN convert to 0
        self.assertEqual(oracle_v_cvt_norm_i16_f16(QNAN_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(SNAN_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(0x7E01), 0)
        self.assertEqual(oracle_v_cvt_norm_i16_f16(0xFE00), 0)

    def test_oracle_modifiers(self) -> None:
        # abs(-1.0) = 1.0 -> 32767
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_ONE_F16, abs_mod=True), 32767)
        # neg(1.0) = -1.0 -> -32767
        self.assertEqual(oracle_v_cvt_norm_i16_f16(ONE_F16, neg_mod=True), u16(-32767))
        # abs and neg: -|1.0| = -1.0
        self.assertEqual(oracle_v_cvt_norm_i16_f16(ONE_F16, abs_mod=True, neg_mod=True), u16(-32767))
        self.assertEqual(oracle_v_cvt_norm_i16_f16(NEG_ONE_F16, abs_mod=True, neg_mod=True), u16(-32767))

    def test_architectural_register_state(self) -> None:
        state = GpuRegisterState()
        state.set_vgpr(0, 0x12345678)
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        state.set_sgpr(5, 0x9ABCDEF0)
        self.assertEqual(state.get_sgpr(5), 0x9ABCDEF0)


class TestValuCvtNormI16F16AST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_creation_and_attributes(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.VGPR, is_hi=False)
        src = RegOperand(index=1, reg_type=RegType.VGPR, is_hi=True, abs_mod=True)
        ast = InstructionAST(opcode="v_cvt_norm_i16_f16", dst=dst, src0=src, op_sel=(0, 1))
        ast.validate()
        self.assertEqual(ast.opcode, "v_cvt_norm_i16_f16")
        self.assertEqual(ast.dst.index, 0)
        self.assertTrue(ast.src0.is_hi)
        self.assertTrue(ast.src0.abs_mod)
        self.assertEqual(ast.op_sel, (0, 1))

    def test_ast_validation_rejects_sgpr_dst(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.SGPR)
        src = RegOperand(index=1, reg_type=RegType.VGPR)
        ast = InstructionAST(opcode="v_cvt_norm_i16_f16", dst=dst, src0=src)
        with self.assertRaises(TypeError):
            ast.validate()

    def test_ast_validation_rejects_dst_modifiers(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.VGPR, neg_mod=True)
        src = RegOperand(index=1, reg_type=RegType.VGPR)
        ast = InstructionAST(opcode="v_cvt_norm_i16_f16", dst=dst, src0=src)
        with self.assertRaises(ValueError):
            ast.validate()

    def test_ast_validation_rejects_out_of_bounds_registers(self) -> None:
        dst_oob = RegOperand(index=256, reg_type=RegType.VGPR)
        src = RegOperand(index=0, reg_type=RegType.VGPR)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_cvt_norm_i16_f16", dst=dst_oob, src0=src).validate()

        dst = RegOperand(index=0, reg_type=RegType.VGPR)
        src_oob = RegOperand(index=106, reg_type=RegType.SGPR)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_cvt_norm_i16_f16", dst=dst, src0=src_oob).validate()

    def test_parse_basic_asm(self) -> None:
        ast = parse_vop_cvt_norm_i16_f16("v_cvt_norm_i16_f16 v0, v1")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.dst.reg_type, RegType.VGPR)
        self.assertFalse(ast.dst.is_hi)
        self.assertIsInstance(ast.src0, RegOperand)
        self.assertEqual(ast.src0.index, 1)
        self.assertEqual(ast.src0.reg_type, RegType.VGPR)
        self.assertFalse(ast.src0.is_hi)

    def test_parse_halfword_swizzles(self) -> None:
        ast = parse_vop_cvt_norm_i16_f16("v_cvt_norm_i16_f16 v2.h, v3.l")
        self.assertEqual(ast.dst.index, 2)
        self.assertTrue(ast.dst.is_hi)
        self.assertEqual(ast.src0.index, 3)
        self.assertFalse(ast.src0.is_hi)
        self.assertEqual(ast.op_sel, (1, 0))

    def test_parse_modifiers(self) -> None:
        ast = parse_vop_cvt_norm_i16_f16("v_cvt_norm_i16_f16 v4, -|v5|")
        self.assertTrue(ast.src0.abs_mod)
        self.assertTrue(ast.src0.neg_mod)

    def test_parse_literals_and_sgpr(self) -> None:
        ast_s = parse_vop_cvt_norm_i16_f16("v_cvt_norm_i16_f16 v0, s4")
        self.assertEqual(ast_s.src0.reg_type, RegType.SGPR)
        self.assertEqual(ast_s.src0.index, 4)

        ast_lit = parse_vop_cvt_norm_i16_f16("v_cvt_norm_i16_f16 v0, 1.0")
        self.assertIsInstance(ast_lit.src0, ImmediateOperand)
        self.assertEqual(ast_lit.src0.value, ONE_F16)

        ast_hex = parse_vop_cvt_norm_i16_f16("v_cvt_norm_i16_f16 v0, 0x3C00")
        self.assertIsInstance(ast_hex.src0, ImmediateOperand)
        self.assertEqual(ast_hex.src0.value, ONE_F16)

    def test_parse_op_sel_and_clamp(self) -> None:
        ast = parse_vop_cvt_norm_i16_f16("v_cvt_norm_i16_f16 v0, v1 op_sel:[1, 0] clamp")
        self.assertEqual(ast.op_sel, (1, 0))
        self.assertTrue(ast.clamp)


# ---------------------------------------------------------------------------
# Deterministic scalar instruction interpreter for v_cvt_norm_i16_f16
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

    def set_vgpr_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx & 0xFF] = val & MASK32

    def get_vgpr_half(self, idx: int, is_hi: bool) -> int:
        dword = self.get_vgpr_u32(idx)
        return (dword >> 16) & MASK16 if is_hi else dword & MASK16

    def set_vgpr_half(self, idx: int, is_hi: bool, half_val: int) -> int:
        """Write halfword into VGPR, preserving the untouched 16-bit half."""
        current = self.get_vgpr_u32(idx)
        h = half_val & MASK16
        if is_hi:
            new_dword = (current & 0x0000FFFF) | (h << 16)
        else:
            new_dword = (current & 0xFFFF0000) | h
        self.set_vgpr_u32(idx, new_dword)
        return new_dword

    def get_sgpr_u32(self, idx: int) -> int:
        return self.sgprs[idx % self.num_sgprs] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> None:
        self.sgprs[idx % self.num_sgprs] = val & MASK32

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
        if ast.opcode == "v_cvt_norm_i16_f16":
            dst_hi = bool(ast.op_sel[0]) or ast.dst.is_hi
            src0_hi = bool(ast.op_sel[1]) or (isinstance(ast.src0, RegOperand) and ast.src0.is_hi)
            src0_val = self.resolve_src0_half(ast.src0, src0_hi)
            res_half = oracle_v_cvt_norm_i16_f16(src0_val)
            return self.set_vgpr_half(ast.dst.index, dst_hi, res_half)
        raise NotImplementedError(f"Opcode '{ast.opcode}' not implemented in scalar interpreter")

    def execute_asm(self, asm_line: str) -> int:
        ast = parse_vop_cvt_norm_i16_f16(asm_line)
        return self.execute(ast)


class TestValuCvtNormI16F16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_basic_scalar_execution(self) -> None:
        self.interp.set_vgpr_u32(1, ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v0, v1")
        self.assertEqual(res & MASK16, 32767)
        self.assertEqual(self.interp.get_vgpr_half(0, False), 32767)

    def test_scalar_preserve_untouched_half(self) -> None:
        # Set v0 to 0xABCD1234
        self.interp.set_vgpr_u32(0, 0xABCD1234)
        self.interp.set_vgpr_u32(1, TWO_F16)

        # Write to low half v0.l
        res1 = self.interp.execute_asm("v_cvt_norm_i16_f16 v0.l, v1")
        self.assertEqual(res1 & MASK16, 32767)
        self.assertEqual(res1 >> 16, 0xABCD)
        self.assertEqual(self.interp.get_vgpr_u32(0), (0xABCD << 16) | 32767)

        # Write to high half v0.h with negative value
        self.interp.set_vgpr_u32(2, NEG_ONE_F16)
        res2 = self.interp.execute_asm("v_cvt_norm_i16_f16 v0.h, v2")
        self.assertEqual(res2 >> 16, 0x8001)
        self.assertEqual(res2 & MASK16, 32767)
        self.assertEqual(self.interp.get_vgpr_u32(0), (0x8001 << 16) | 32767)

    def test_scalar_sgpr_source(self) -> None:
        self.interp.set_sgpr_u32(5, NEG_ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v2, s5")
        self.assertEqual(res & MASK16, 0x8001)

    def test_scalar_literal_source(self) -> None:
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v3, 0.5")
        self.assertEqual(res & MASK16, 16384)

    def test_scalar_op_sel_routing(self) -> None:
        # v4 holds high half = 1.0 (0x3C00), low half = -1.0 (0xBC00)
        self.interp.set_vgpr_u32(4, (ONE_F16 << 16) | NEG_ONE_F16)

        # Read high half (1.0), write to low half of v5
        res_lo = self.interp.execute_asm("v_cvt_norm_i16_f16 v5, v4 op_sel:[0, 1]")
        self.assertEqual(res_lo & MASK16, 32767)

        # Read low half (-1.0), write to high half of v5
        res_hi = self.interp.execute_asm("v_cvt_norm_i16_f16 v5, v4 op_sel:[1, 0]")
        self.assertEqual(res_hi >> 16, 0x8001)
        self.assertEqual(res_hi & MASK16, 32767)


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

    def set_vgpr_u32(self, reg: int, lane: int, val: int) -> None:
        self.vgprs[reg & 0xFF][lane] = val & MASK32

    def get_vgpr_half(self, reg: int, lane: int, is_hi: bool) -> int:
        dword = self.get_vgpr_u32(reg, lane)
        return (dword >> 16) & MASK16 if is_hi else dword & MASK16

    def set_vgpr_half(self, reg: int, lane: int, is_hi: bool, half_val: int) -> int:
        """Write halfword into VGPR for specified lane, preserving the untouched 16-bit half."""
        current = self.get_vgpr_u32(reg, lane)
        h = half_val & MASK16
        if is_hi:
            new_dword = (current & 0x0000FFFF) | (h << 16)
        else:
            new_dword = (current & 0xFFFF0000) | h
        self.set_vgpr_u32(reg, lane, new_dword)
        return new_dword

    def get_sgpr_u32(self, idx: int) -> int:
        return self.sgprs[idx % self.num_sgprs] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> None:
        self.sgprs[idx % self.num_sgprs] = val & MASK32

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
        if ast.opcode == "v_cvt_norm_i16_f16":
            dst_hi = bool(ast.op_sel[0]) or ast.dst.is_hi
            src0_hi = bool(ast.op_sel[1]) or (isinstance(ast.src0, RegOperand) and ast.src0.is_hi)
            results = []
            for lane in range(self.wave_size):
                lane_active = bool(self.exec_mask & (1 << lane))
                if lane_active:
                    src0_val = self.resolve_src0_half(ast.src0, lane, src0_hi)
                    res_half = oracle_v_cvt_norm_i16_f16(src0_val)
                    new_dword = self.set_vgpr_half(ast.dst.index, lane, dst_hi, res_half)
                    results.append(new_dword)
                else:
                    results.append(self.get_vgpr_u32(ast.dst.index, lane))
            return results
        raise NotImplementedError(f"Opcode '{ast.opcode}' not implemented in SIMD interpreter")

    def execute_asm(self, asm_line: str) -> list[int]:
        ast = parse_vop_cvt_norm_i16_f16(asm_line)
        return self.execute(ast)


class TestValuCvtNormI16F16SIMD(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        for lane in range(32):
            simd.set_vgpr_u32(1, lane, ONE_F16)
        res = simd.execute_asm("v_cvt_norm_i16_f16 v0, v1")
        self.assertEqual(len(res), 32)
        for lane in range(32):
            self.assertEqual(res[lane] & MASK16, 32767)
            self.assertEqual(simd.get_vgpr_half(0, lane, False), 32767)

    def test_wave64_uniform_execution(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=64)
        for lane in range(64):
            simd.set_vgpr_u32(1, lane, NEG_ONE_F16)
        res = simd.execute_asm("v_cvt_norm_i16_f16 v0, v1")
        self.assertEqual(len(res), 64)
        for lane in range(64):
            self.assertEqual(res[lane] & MASK16, 0x8001)
            self.assertEqual(simd.get_vgpr_half(0, lane, False), 0x8001)

    def test_exec_mask_selective_divergence(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=64)
        simd.exec_mask = 0x55555555_55555555  # Even lanes active, odd lanes inactive
        for lane in range(64):
            simd.set_vgpr_u32(0, lane, 0xDEADBEEF)
            simd.set_vgpr_u32(1, lane, ONE_F16)

        res = simd.execute_asm("v_cvt_norm_i16_f16 v0.l, v1")
        for lane in range(64):
            if lane % 2 == 0:
                # Active: low half is 32767, high half preserved 0xDEAD
                self.assertEqual(res[lane] & MASK16, 32767)
                self.assertEqual(res[lane] >> 16, 0xDEAD)
                self.assertEqual(simd.get_vgpr_u32(0, lane), (0xDEAD << 16) | 32767)
            else:
                # Inactive: untouched
                self.assertEqual(res[lane], 0xDEADBEEF)
                self.assertEqual(simd.get_vgpr_u32(0, lane), 0xDEADBEEF)

    def test_lane_divergent_values(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        test_inputs = [
            (POS_ZERO_F16, 0),
            (HALF_F16, 16384),
            (ONE_F16, 32767),
            (NEG_ONE_F16, 0x8001),
            (QNAN_F16, 0),
            (TWO_F16, 32767),
            (NEG_TWO_F16, 0x8001),
            (POS_INF_F16, 32767),
        ]
        for lane in range(32):
            inp, _ = test_inputs[lane % len(test_inputs)]
            simd.set_vgpr_u32(1, lane, inp)

        res = simd.execute_asm("v_cvt_norm_i16_f16 v0, v1")
        for lane in range(32):
            _, expected = test_inputs[lane % len(test_inputs)]
            self.assertEqual(res[lane] & MASK16, expected)

    def test_zero_exec_mask_no_modification(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        simd.exec_mask = 0
        for lane in range(32):
            simd.set_vgpr_u32(0, lane, 0x12345678)
            simd.set_vgpr_u32(1, lane, ONE_F16)
        res = simd.execute_asm("v_cvt_norm_i16_f16 v0, v1")
        for lane in range(32):
            self.assertEqual(res[lane], 0x12345678)
            self.assertEqual(simd.get_vgpr_u32(0, lane), 0x12345678)


# ---------------------------------------------------------------------------
# Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuCvtNormI16F16Aliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_in_place_self_aliasing_low_half(self) -> None:
        # Set v0: high half = 0x5A5A, low half = ONE_F16 (1.0)
        self.interp.set_vgpr_u32(0, (0x5A5A << 16) | ONE_F16)
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v0.l, v0.l")
        # Low half should be 32767, high half preserved as 0x5A5A
        self.assertEqual(res & MASK16, 32767)
        self.assertEqual(res >> 16, 0x5A5A)
        self.assertEqual(self.interp.get_vgpr_u32(0), (0x5A5A << 16) | 32767)

    def test_in_place_self_aliasing_high_half(self) -> None:
        # Set v0: high half = NEG_ONE_F16 (-1.0), low half = 0xA5A5
        self.interp.set_vgpr_u32(0, (NEG_ONE_F16 << 16) | 0xA5A5)
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v0.h, v0.h")
        # High half should be 0x8001 (-32767), low half preserved as 0xA5A5
        self.assertEqual(res >> 16, 0x8001)
        self.assertEqual(res & MASK16, 0xA5A5)
        self.assertEqual(self.interp.get_vgpr_u32(0), (0x8001 << 16) | 0xA5A5)

    def test_cross_halfword_aliasing_low_to_high(self) -> None:
        # Set v0: high half = 0x1111, low half = HALF_F16 (0.5)
        self.interp.set_vgpr_u32(0, (0x1111 << 16) | HALF_F16)
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v0.h, v0.l")
        # High half should be 16384, low half remains HALF_F16
        self.assertEqual(res >> 16, 16384)
        self.assertEqual(res & MASK16, HALF_F16)
        self.assertEqual(self.interp.get_vgpr_u32(0), (16384 << 16) | HALF_F16)

    def test_cross_halfword_aliasing_high_to_low(self) -> None:
        # Set v0: high half = TWO_F16 (2.0), low half = 0x2222
        self.interp.set_vgpr_u32(0, (TWO_F16 << 16) | 0x2222)
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v0.l, v0.h")
        # Low half should be clamped to 32767, high half remains TWO_F16
        self.assertEqual(res & MASK16, 32767)
        self.assertEqual(res >> 16, TWO_F16)
        self.assertEqual(self.interp.get_vgpr_u32(0), (TWO_F16 << 16) | 32767)

    def test_multi_instruction_aliasing_pipeline(self) -> None:
        # v0 holds two f16 inputs: high = NEG_ONE_F16 (-1.0), low = ONE_F16 (1.0)
        self.interp.set_vgpr_u32(0, (NEG_ONE_F16 << 16) | ONE_F16)
        # Step 1: convert low half in-place
        self.interp.execute_asm("v_cvt_norm_i16_f16 v0.l, v0.l")
        self.assertEqual(self.interp.get_vgpr_half(0, False), 32767)
        self.assertEqual(self.interp.get_vgpr_half(0, True), NEG_ONE_F16)

        # Step 2: convert high half in-place
        self.interp.execute_asm("v_cvt_norm_i16_f16 v0.h, v0.h")
        self.assertEqual(self.interp.get_vgpr_half(0, True), 0x8001)
        self.assertEqual(self.interp.get_vgpr_half(0, False), 32767)
        self.assertEqual(self.interp.get_vgpr_u32(0), (0x8001 << 16) | 32767)

    def test_aliased_source_with_abs_modifier(self) -> None:
        # v0 low half = NEG_ONE_F16 (-1.0), high half = 0x3333
        self.interp.set_vgpr_u32(0, (0x3333 << 16) | NEG_ONE_F16)
        # Apply |src0|: |-1.0| = 1.0 -> 32767
        res = self.interp.execute_asm("v_cvt_norm_i16_f16 v0.l, |v0.l|")
        self.assertEqual(res & MASK16, 32767)
        self.assertEqual(res >> 16, 0x3333)


if __name__ == "__main__":
    unittest.main()
