from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_i32_f32.

Instruction:
  v_cvt_norm_i32_f32 dst, src0 (VOP1 / VOP3)
  D.i32 = f32_to_snorm(S0.f32)

Architecture:
  - RDNA2 Vector ALU (GFX10) single-precision float to signed normalized 32-bit integer conversion.
  - S0: 32-bit IEEE 754 binary32 float (from VGPR, SGPR, or literal/inline constant).
  - D:  32-bit signed normalized integer in [-2147483647, 2147483647] (in VGPR).
  - Normalization:
      * Maps floating-point range [-1.0, 1.0] to integer range [-2147483647, 2147483647].
      * Scaling factor is 2147483647.0 (2^(b-1) - 1 where b=32).
      * Rounding mode: round-to-nearest-even (IEEE 754-2008 roundTiesToEven).
      * Clamping: values > 1.0 (or +inf) clamp to 2147483647 (0x7FFFFFFF).
                  values < -1.0 (or -inf) clamp to -2147483647 (0x80000001).
      * NaNs (both QNaN and SNaN) convert to 0 (0x00000000).
      * Signed zeros (+0.0, -0.0) convert to 0 (0x00000000).
  - Register model:
      * 32-bit vector registers (VGPRs: v0-v255) and scalar registers (SGPRs: s0-s105).
      * Overwrites the full 32-bit destination register.
  - Modifiers:
      * Input modifiers: abs (|S0|), neg (-S0).
      * Order of application: abs clears sign bit first, then neg inverts sign bit.
"""

from fractions import Fraction
import math
import re
import struct
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary32 architectural bit layout
SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF
QNAN_BIT_F32: int = 0x00400000
BIAS_F32: int = 127

# Special single-precision bit patterns
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001

# Standard reference floating-point constants in binary32
ONE_F32: int = 0x3F800000            # 1.0f
NEG_ONE_F32: int = 0xBF800000        # -1.0f
TWO_F32: int = 0x40000000            # 2.0f
NEG_TWO_F32: int = 0xC0000000        # -2.0f
HALF_F32: int = 0x3F000000           # 0.5f
NEG_HALF_F32: int = 0xBF000000       # -0.5f
QUARTER_F32: int = 0x3E800000        # 0.25f
NEG_QUARTER_F32: int = 0xBE800000    # -0.25f
FOUR_F32: int = 0x40800000           # 4.0f
NEG_FOUR_F32: int = 0xC0800000       # -4.0f

MAX_FINITE_F32: int = 0x7F7FFFFF     # ~3.402823466e+38
MIN_FINITE_NEG_F32: int = 0xFF7FFFFF # ~-3.402823466e+38
MIN_NORMAL_F32: int = 0x00800000     # 2^-126 (~1.17549435e-38)
MIN_SUBNORMAL_F32: int = 0x00000001  # 2^-149 (~1.40129846e-45)
MAX_SUBNORMAL_F32: int = 0x007FFFFF  # ~1.17549421e-38

# 32-bit signed normalized integer range constants
SNORM32_MAX: int = 2147483647        # 0x7FFFFFFF (+1.0)
SNORM32_MIN: int = -2147483647       # 0x80000001 (-1.0)
SNORM32_SCALE: float = 2147483647.0


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


def bits_to_f32(bits: int) -> float:
    """Convert unsigned 32-bit bits into Python float via struct IEEE 754 single."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Convert Python float into unsigned 32-bit IEEE 754 single bits."""
    if math.isnan(val):
        return QNAN_F32
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return NEG_INF_F32 if val < 0.0 else POS_INF_F32


def is_nan_f32(bits: int) -> bool:
    """Check if binary32 bits represent any NaN."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) != 0)


def is_qnan_f32(bits: int) -> bool:
    """Check if binary32 bits represent a Quiet NaN."""
    return is_nan_f32(bits) and bool(bits & QNAN_BIT_F32)


def is_snan_f32(bits: int) -> bool:
    """Check if binary32 bits represent a Signaling NaN."""
    return is_nan_f32(bits) and not bool(bits & QNAN_BIT_F32)


def is_inf_f32(bits: int) -> bool:
    """Check if binary32 bits represent +/- infinity."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) == 0)


def is_subnormal_f32(bits: int) -> bool:
    """Check if binary32 bits represent a subnormal number."""
    return ((bits & EXP_MASK_F32) == 0) and ((bits & MANT_MASK_F32) != 0)


def is_zero_f32(bits: int) -> bool:
    """Check if binary32 bits represent +0.0 or -0.0."""
    return (bits & ~SIGN_MASK_F32) == 0


def apply_f32_modifiers(bits32: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 32-bit float word.
    
    Order of operations: abs clears sign bit (bit 31) first, then neg inverts sign bit (bit 31).
    """
    val = bits32 & MASK32
    if abs_mod:
        val &= ~SIGN_MASK_F32
    if neg_mod:
        val ^= SIGN_MASK_F32
    return val


def f32_to_snorm32(src0_bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 f32_to_snorm conversion.
    
    Converts 32-bit float bits to signed normalized 32-bit integer bits.
    - NaNs convert to 0.
    - Clamped to [-1.0, 1.0].
    - Values > 1.0 (and +inf) clamp to 2147483647.
    - Values < -1.0 (and -inf) clamp to -2147483647.
    - Scaled by 2147483647.0 and rounded with round-to-nearest-even.
    Returns unsigned 32-bit representation (two's complement for negatives).
    """
    val_bits = apply_f32_modifiers(src0_bits, abs_mod, neg_mod)
    if is_nan_f32(val_bits):
        return 0
    if is_zero_f32(val_bits):
        return 0
    if is_inf_f32(val_bits):
        return u32(SNORM32_MIN if (val_bits & SIGN_MASK_F32) else SNORM32_MAX)

    val_float = bits_to_f32(val_bits)
    if val_float >= 1.0:
        return u32(SNORM32_MAX)
    if val_float <= -1.0:
        return u32(SNORM32_MIN)

    # Scale and round-to-nearest-even
    scaled = val_float * SNORM32_SCALE
    rounded = int(round(scaled))
    # Safety clamp to [-2147483647, 2147483647]
    if rounded > SNORM32_MAX:
        rounded = SNORM32_MAX
    elif rounded < SNORM32_MIN:
        rounded = SNORM32_MIN
    return u32(rounded)


def oracle_v_cvt_norm_i32_f32(
    src0_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
) -> int:
    """Bit-exact reference oracle for v_cvt_norm_i32_f32."""
    return f32_to_snorm32(src0_bits, abs_mod, neg_mod)


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


class TestValuCvtNormI32F32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_oracle_zeros(self) -> None:
        self.assertEqual(oracle_v_cvt_norm_i32_f32(POS_ZERO_F32), 0)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_ZERO_F32), 0)

    def test_oracle_unit_boundaries(self) -> None:
        # +1.0 -> 2147483647 (0x7FFFFFFF)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(ONE_F32), 2147483647)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(ONE_F32), 0x7FFFFFFF)
        # -1.0 -> -2147483647 (0x80000001 in two's complement u32)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_ONE_F32), u32(-2147483647))
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_ONE_F32), 0x80000001)

    def test_oracle_half_values(self) -> None:
        # +0.5 * 2147483647 = 1073741823.5 -> rounds to 1073741824 (even) = 0x40000000
        self.assertEqual(oracle_v_cvt_norm_i32_f32(HALF_F32), 1073741824)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(HALF_F32), 0x40000000)
        # -0.5 * 2147483647 = -1073741823.5 -> rounds to -1073741824 (even) = 0xC0000000
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_HALF_F32), u32(-1073741824))
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_HALF_F32), 0xC0000000)

    def test_oracle_clamping_extremes(self) -> None:
        # +2.0 clamps to +1.0 -> 2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(TWO_F32), 2147483647)
        # -2.0 clamps to -1.0 -> -2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_TWO_F32), u32(-2147483647))
        # +4.0, -4.0 clamp
        self.assertEqual(oracle_v_cvt_norm_i32_f32(FOUR_F32), 2147483647)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_FOUR_F32), u32(-2147483647))
        # Infinities clamp to extremes
        self.assertEqual(oracle_v_cvt_norm_i32_f32(POS_INF_F32), 2147483647)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_INF_F32), u32(-2147483647))

    def test_oracle_nans(self) -> None:
        # Both QNaN and SNaN convert to 0
        self.assertEqual(oracle_v_cvt_norm_i32_f32(QNAN_F32), 0)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(SNAN_F32), 0)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(0x7FC00001), 0)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(0xFFC00000), 0)

    def test_oracle_modifiers(self) -> None:
        # abs(-1.0) = 1.0 -> 2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_ONE_F32, abs_mod=True), 2147483647)
        # neg(1.0) = -1.0 -> -2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(ONE_F32, neg_mod=True), u32(-2147483647))
        # abs and neg: -|1.0| = -1.0
        self.assertEqual(oracle_v_cvt_norm_i32_f32(ONE_F32, abs_mod=True, neg_mod=True), u32(-2147483647))
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_ONE_F32, abs_mod=True, neg_mod=True), u32(-2147483647))

    def test_architectural_register_state(self) -> None:
        state = GpuRegisterState()
        state.set_vgpr(0, 0x12345678)
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        state.set_sgpr(5, 0x9ABCDEF0)
        self.assertEqual(state.get_sgpr(5), 0x9ABCDEF0)


# ---------------------------------------------------------------------------
# AST opcode definitions and typed expressions for v_cvt_norm_i32_f32
# ---------------------------------------------------------------------------

class RegType(Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclass
class RegOperand:
    index: int
    reg_type: RegType
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
    is_f32: bool = True
    abs_mod: bool = False
    neg_mod: bool = False

    def validate(self) -> None:
        pass


@dataclass
class InstructionAST:
    opcode: str
    dst: RegOperand
    src0: Union[RegOperand, ImmediateOperand]
    clamp: bool = False

    def validate(self) -> None:
        if self.opcode != "v_cvt_norm_i32_f32":
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if not isinstance(self.dst, RegOperand):
            raise TypeError("Destination must be a RegOperand")
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"VALU instruction destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination operand cannot have abs or neg modifiers")
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

    m = re.match(r"^([vs])(\d+)$", tok, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid register operand format: '{token}'")
    prefix = m.group(1).lower()
    idx = int(m.group(2))
    reg_type = RegType.VGPR if prefix == "v" else RegType.SGPR

    if is_dst and reg_type != RegType.VGPR:
        raise TypeError(f"Destination register must be VGPR, got '{token}'")
    if is_dst and (abs_mod or neg_mod):
        raise ValueError(f"Destination cannot have modifiers: '{token}'")

    op = RegOperand(index=idx, reg_type=reg_type, abs_mod=abs_mod, neg_mod=neg_mod)
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
            val = int(tok, 16) & MASK32
            return ImmediateOperand(value=val, is_f32=True, abs_mod=abs_mod, neg_mod=neg_mod)
        except ValueError:
            raise ValueError(f"Invalid hex literal: '{token}'")
    try:
        val_f = float(tok)
        val_bits = f32_to_bits(val_f)
        return ImmediateOperand(value=val_bits, is_f32=True, abs_mod=abs_mod, neg_mod=neg_mod)
    except ValueError:
        pass

    m = re.match(r"^([vs])(\d+)$", tok, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid source operand: '{token}'")
    prefix = m.group(1).lower()
    idx = int(m.group(2))
    reg_type = RegType.VGPR if prefix == "v" else RegType.SGPR
    op = RegOperand(index=idx, reg_type=reg_type, abs_mod=abs_mod, neg_mod=neg_mod)
    op.validate()
    return op


def parse_vop_cvt_norm_i32_f32(asm: str) -> InstructionAST:
    """Parse assembly text representation of v_cvt_norm_i32_f32 into AST."""
    line = asm.split(";")[0].split("//")[0].strip()
    if not line:
        raise ValueError("Empty assembly line")

    tokens = line.split(None, 1)
    mnemonic = tokens[0].lower()
    if mnemonic != "v_cvt_norm_i32_f32":
        raise ValueError(f"Invalid mnemonic '{mnemonic}', expected 'v_cvt_norm_i32_f32'")

    if len(tokens) < 2:
        raise ValueError(f"Missing operands for instruction '{asm}'")

    body = tokens[1].strip()

    clamp = False
    if re.search(r"\bclamp\b", body):
        clamp = True
        body = re.sub(r"\bclamp\b", " ", body)

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

    ast = InstructionAST(opcode="v_cvt_norm_i32_f32", dst=dst_op, src0=src_op, clamp=clamp)
    ast.validate()
    return ast


class TestValuCvtNormI32F32AST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_creation_and_attributes(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.VGPR)
        src = RegOperand(index=1, reg_type=RegType.VGPR, abs_mod=True)
        ast = InstructionAST(opcode="v_cvt_norm_i32_f32", dst=dst, src0=src)
        ast.validate()
        self.assertEqual(ast.opcode, "v_cvt_norm_i32_f32")
        self.assertEqual(ast.dst.index, 0)
        self.assertTrue(ast.src0.abs_mod)

    def test_ast_validation_rejects_sgpr_dst(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.SGPR)
        src = RegOperand(index=1, reg_type=RegType.VGPR)
        ast = InstructionAST(opcode="v_cvt_norm_i32_f32", dst=dst, src0=src)
        with self.assertRaises(TypeError):
            ast.validate()

    def test_ast_validation_rejects_dst_modifiers(self) -> None:
        dst = RegOperand(index=0, reg_type=RegType.VGPR, neg_mod=True)
        src = RegOperand(index=1, reg_type=RegType.VGPR)
        ast = InstructionAST(opcode="v_cvt_norm_i32_f32", dst=dst, src0=src)
        with self.assertRaises(ValueError):
            ast.validate()

    def test_ast_validation_rejects_out_of_bounds_registers(self) -> None:
        dst_oob = RegOperand(index=256, reg_type=RegType.VGPR)
        src = RegOperand(index=0, reg_type=RegType.VGPR)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_cvt_norm_i32_f32", dst=dst_oob, src0=src).validate()

        dst = RegOperand(index=0, reg_type=RegType.VGPR)
        src_oob = RegOperand(index=106, reg_type=RegType.SGPR)
        with self.assertRaises(ValueError):
            InstructionAST(opcode="v_cvt_norm_i32_f32", dst=dst, src0=src_oob).validate()

    def test_parse_basic_asm(self) -> None:
        ast = parse_vop_cvt_norm_i32_f32("v_cvt_norm_i32_f32 v0, v1")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.dst.reg_type, RegType.VGPR)
        self.assertIsInstance(ast.src0, RegOperand)
        self.assertEqual(ast.src0.index, 1)
        self.assertEqual(ast.src0.reg_type, RegType.VGPR)

    def test_parse_modifiers(self) -> None:
        ast = parse_vop_cvt_norm_i32_f32("v_cvt_norm_i32_f32 v4, -|v5|")
        self.assertTrue(ast.src0.abs_mod)
        self.assertTrue(ast.src0.neg_mod)

    def test_parse_literals_and_sgpr(self) -> None:
        ast_s = parse_vop_cvt_norm_i32_f32("v_cvt_norm_i32_f32 v0, s4")
        self.assertEqual(ast_s.src0.reg_type, RegType.SGPR)
        self.assertEqual(ast_s.src0.index, 4)

        ast_lit = parse_vop_cvt_norm_i32_f32("v_cvt_norm_i32_f32 v0, 1.0")
        self.assertIsInstance(ast_lit.src0, ImmediateOperand)
        self.assertEqual(ast_lit.src0.value, ONE_F32)

        ast_hex = parse_vop_cvt_norm_i32_f32("v_cvt_norm_i32_f32 v0, 0x3F800000")
        self.assertIsInstance(ast_hex.src0, ImmediateOperand)
        self.assertEqual(ast_hex.src0.value, ONE_F32)

    def test_parse_clamp(self) -> None:
        ast = parse_vop_cvt_norm_i32_f32("v_cvt_norm_i32_f32 v0, v1 clamp")
        self.assertTrue(ast.clamp)


# ---------------------------------------------------------------------------
# Deterministic scalar instruction interpreter for v_cvt_norm_i32_f32
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

    def get_sgpr_u32(self, idx: int) -> int:
        return self.sgprs[idx % self.num_sgprs] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> int:
        self.sgprs[idx % self.num_sgprs] = val & MASK32
        return self.sgprs[idx % self.num_sgprs]

    def resolve_src0(self, src: Union[RegOperand, ImmediateOperand]) -> int:
        """Fetch and apply modifiers to source operand."""
        if isinstance(src, ImmediateOperand):
            val = src.value & MASK32
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
        elif isinstance(src, RegOperand):
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
            if src.reg_type == RegType.VGPR:
                val = self.get_vgpr_u32(src.index)
            else:
                val = self.get_sgpr_u32(src.index)
        else:
            raise TypeError(f"Unknown source operand type: {type(src)}")
        return apply_f32_modifiers(val, abs_mod, neg_mod)

    def execute(self, ast: InstructionAST) -> int:
        ast.validate()
        if ast.opcode == "v_cvt_norm_i32_f32":
            src0_val = self.resolve_src0(ast.src0)
            res = oracle_v_cvt_norm_i32_f32(src0_val)
            return self.set_vgpr_u32(ast.dst.index, res)
        raise NotImplementedError(f"Opcode '{ast.opcode}' not implemented in scalar interpreter")

    def execute_asm(self, asm_line: str) -> int:
        ast = parse_vop_cvt_norm_i32_f32(asm_line)
        return self.execute(ast)


class TestValuCvtNormI32F32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_basic_scalar_execution(self) -> None:
        self.interp.set_vgpr_u32(1, ONE_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        self.assertEqual(res, 2147483647)
        self.assertEqual(self.interp.get_vgpr_u32(0), 2147483647)

    def test_scalar_negative_execution(self) -> None:
        self.interp.set_vgpr_u32(1, NEG_ONE_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        self.assertEqual(res, 0x80000001)
        self.assertEqual(self.interp.get_vgpr_u32(0), 0x80000001)

    def test_scalar_sgpr_source(self) -> None:
        self.interp.set_sgpr_u32(5, HALF_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v2, s5")
        self.assertEqual(res, 1073741824)

    def test_scalar_literal_source(self) -> None:
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v3, 0.5")
        self.assertEqual(res, 1073741824)

    def test_scalar_modifiers_execution(self) -> None:
        self.interp.set_vgpr_u32(1, ONE_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, -v1")
        self.assertEqual(res, 0x80000001)


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

    def get_sgpr_u32(self, idx: int) -> int:
        return self.sgprs[idx % self.num_sgprs] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> int:
        self.sgprs[idx % self.num_sgprs] = val & MASK32
        return self.sgprs[idx % self.num_sgprs]

    def resolve_src0(self, src: Union[RegOperand, ImmediateOperand], lane: int) -> int:
        if isinstance(src, ImmediateOperand):
            val = src.value & MASK32
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
        elif isinstance(src, RegOperand):
            abs_mod = src.abs_mod
            neg_mod = src.neg_mod
            if src.reg_type == RegType.VGPR:
                val = self.get_vgpr_u32(src.index, lane)
            else:
                val = self.get_sgpr_u32(src.index)
        else:
            raise TypeError(f"Unknown source operand type: {type(src)}")
        return apply_f32_modifiers(val, abs_mod, neg_mod)

    def execute(self, ast: InstructionAST) -> list[int]:
        ast.validate()
        if ast.opcode == "v_cvt_norm_i32_f32":
            results = []
            for lane in range(self.wave_size):
                lane_active = bool(self.exec_mask & (1 << lane))
                if lane_active:
                    src0_val = self.resolve_src0(ast.src0, lane)
                    res = oracle_v_cvt_norm_i32_f32(src0_val)
                    new_val = self.set_vgpr_u32(ast.dst.index, lane, res)
                    results.append(new_val)
                else:
                    results.append(self.get_vgpr_u32(ast.dst.index, lane))
            return results
        raise NotImplementedError(f"Opcode '{ast.opcode}' not implemented in SIMD interpreter")

    def execute_asm(self, asm_line: str) -> list[int]:
        ast = parse_vop_cvt_norm_i32_f32(asm_line)
        return self.execute(ast)


class TestValuCvtNormI32F32SIMD(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        for lane in range(32):
            simd.set_vgpr_u32(1, lane, ONE_F32)
        res = simd.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        self.assertEqual(len(res), 32)
        for lane in range(32):
            self.assertEqual(res[lane], 2147483647)
            self.assertEqual(simd.get_vgpr_u32(0, lane), 2147483647)

    def test_wave64_uniform_execution(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=64)
        for lane in range(64):
            simd.set_vgpr_u32(1, lane, NEG_ONE_F32)
        res = simd.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        self.assertEqual(len(res), 64)
        for lane in range(64):
            self.assertEqual(res[lane], 0x80000001)
            self.assertEqual(simd.get_vgpr_u32(0, lane), 0x80000001)

    def test_exec_mask_selective_divergence(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=64)
        simd.exec_mask = 0x55555555_55555555  # Even lanes active, odd lanes inactive
        for lane in range(64):
            simd.set_vgpr_u32(0, lane, 0xDEADBEEF)
            simd.set_vgpr_u32(1, lane, ONE_F32)

        res = simd.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(res[lane], 2147483647)
                self.assertEqual(simd.get_vgpr_u32(0, lane), 2147483647)
            else:
                self.assertEqual(res[lane], 0xDEADBEEF)
                self.assertEqual(simd.get_vgpr_u32(0, lane), 0xDEADBEEF)

    def test_lane_divergent_values(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        test_inputs = [
            (POS_ZERO_F32, 0),
            (HALF_F32, 1073741824),
            (ONE_F32, 2147483647),
            (NEG_ONE_F32, 0x80000001),
            (QNAN_F32, 0),
            (TWO_F32, 2147483647),
            (NEG_TWO_F32, 0x80000001),
            (POS_INF_F32, 2147483647),
        ]
        for lane in range(32):
            inp, _ = test_inputs[lane % len(test_inputs)]
            simd.set_vgpr_u32(1, lane, inp)

        res = simd.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        for lane in range(32):
            _, expected = test_inputs[lane % len(test_inputs)]
            self.assertEqual(res[lane], expected)

    def test_zero_exec_mask_no_modification(self) -> None:
        simd = SIMDVALUInterpreter(wave_size=32)
        simd.exec_mask = 0
        for lane in range(32):
            simd.set_vgpr_u32(0, lane, 0x12345678)
            simd.set_vgpr_u32(1, lane, ONE_F32)
        res = simd.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        for lane in range(32):
            self.assertEqual(res[lane], 0x12345678)
            self.assertEqual(simd.get_vgpr_u32(0, lane), 0x12345678)


# ---------------------------------------------------------------------------
# Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuCvtNormI32F32Aliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_in_place_self_aliasing(self) -> None:
        self.interp.set_vgpr_u32(0, ONE_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, v0")
        self.assertEqual(res, 2147483647)
        self.assertEqual(self.interp.get_vgpr_u32(0), 2147483647)

    def test_in_place_self_aliasing_negative(self) -> None:
        self.interp.set_vgpr_u32(0, NEG_ONE_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, v0")
        self.assertEqual(res, 0x80000001)
        self.assertEqual(self.interp.get_vgpr_u32(0), 0x80000001)

    def test_in_place_self_aliasing_with_abs_modifier(self) -> None:
        self.interp.set_vgpr_u32(0, NEG_ONE_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, |v0|")
        self.assertEqual(res, 2147483647)
        self.assertEqual(self.interp.get_vgpr_u32(0), 2147483647)

    def test_in_place_self_aliasing_with_neg_modifier(self) -> None:
        self.interp.set_vgpr_u32(0, ONE_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, -v0")
        self.assertEqual(res, 0x80000001)
        self.assertEqual(self.interp.get_vgpr_u32(0), 0x80000001)

    def test_distinct_registers_non_interference(self) -> None:
        self.interp.set_vgpr_u32(0, 0xCAFEBABE)
        self.interp.set_vgpr_u32(1, HALF_F32)
        res = self.interp.execute_asm("v_cvt_norm_i32_f32 v0, v1")
        self.assertEqual(res, 1073741824)
        self.assertEqual(self.interp.get_vgpr_u32(0), 1073741824)
        self.assertEqual(self.interp.get_vgpr_u32(1), HALF_F32)

    def test_multi_instruction_aliased_pipeline(self) -> None:
        self.interp.set_vgpr_u32(0, HALF_F32)
        self.interp.set_vgpr_u32(1, NEG_TWO_F32)

        # Step 1: in-place convert v0
        self.interp.execute_asm("v_cvt_norm_i32_f32 v0, v0")
        self.assertEqual(self.interp.get_vgpr_u32(0), 1073741824)  # 0x40000000

        # Step 2: in-place convert v1
        self.interp.execute_asm("v_cvt_norm_i32_f32 v1, v1")
        self.assertEqual(self.interp.get_vgpr_u32(1), 0x80000001)

        # Step 3: use converted v0 as source for v2 (0x40000000 bitwise is 2.0f in f32)
        self.interp.execute_asm("v_cvt_norm_i32_f32 v2, v0")
        self.assertEqual(self.interp.get_vgpr_u32(2), 2147483647)


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
    """Condition codes and carry/borrow generation for 32-bit integer result."""
    zero: bool = False
    negative: bool = False
    max_pos: bool = False
    min_neg: bool = False
    overflow: bool = False
    carry: bool = False
    borrow: bool = False


def calc_cvt_norm_status_flags(src0_bits: int, res_bits: int) -> FPStatusFlags:
    """Calculate IEEE 754 exception flags for v_cvt_norm_i32_f32 conversion."""
    flags = FPStatusFlags()
    s0 = src0_bits & MASK32

    # NaNs set invalid flag
    if is_nan_f32(s0):
        flags.invalid = True
        return flags

    # Infinities set overflow and inexact
    if is_inf_f32(s0):
        flags.overflow = True
        flags.inexact = True
        return flags

    if is_zero_f32(s0):
        return flags

    val_f = bits_to_f32(s0)
    # Range check: input magnitude > 1.0 clamped sets overflow & inexact
    if val_f > 1.0 or val_f < -1.0:
        flags.overflow = True
        flags.inexact = True
        return flags

    # Underflow: non-zero input float rounded to zero
    if (res_bits & MASK32) == 0 and not is_zero_f32(s0):
        flags.underflow = True
        flags.inexact = True
        return flags

    # Inexact check: if val_f * 2147483647.0 has fractional part
    scaled = val_f * SNORM32_SCALE
    if scaled != float(int(round(scaled))):
        flags.inexact = True

    return flags


def calc_cvt_norm_condition_codes(src0_bits: int, res_bits: int) -> IntConditionCodes:
    """Calculate integer condition codes and boundary carry/borrow flags."""
    res_s32 = s32(res_bits)
    is_zero = (res_bits & MASK32) == 0
    is_neg = bool(res_bits & 0x80000000)
    is_max = res_s32 == SNORM32_MAX
    is_min = res_s32 == SNORM32_MIN

    s0 = src0_bits & MASK32
    overflow = False
    carry = False
    borrow = False

    if is_nan_f32(s0):
        pass
    elif is_inf_f32(s0):
        overflow = True
        if not (s0 & SIGN_MASK_F32):
            carry = True
        else:
            borrow = True
    else:
        val_f = bits_to_f32(s0)
        if val_f > 1.0:
            overflow = True
            carry = True  # positive saturation carry out
        elif val_f < -1.0:
            overflow = True
            borrow = True  # negative saturation borrow

    return IntConditionCodes(
        zero=is_zero,
        negative=is_neg,
        max_pos=is_max,
        min_neg=is_min,
        overflow=overflow,
        carry=carry,
        borrow=borrow,
    )


class TestValuCvtNormI32F32StatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow generation."""

    def test_invalid_flag_on_nans(self) -> None:
        flags_qnan = calc_cvt_norm_status_flags(QNAN_F32, 0)
        self.assertTrue(flags_qnan.invalid)
        self.assertFalse(flags_qnan.overflow)

        flags_snan = calc_cvt_norm_status_flags(SNAN_F32, 0)
        self.assertTrue(flags_snan.invalid)

    def test_overflow_and_inexact_on_out_of_range(self) -> None:
        flags_pos = calc_cvt_norm_status_flags(TWO_F32, 2147483647)
        self.assertTrue(flags_pos.overflow)
        self.assertTrue(flags_pos.inexact)
        self.assertFalse(flags_pos.invalid)

        flags_neg = calc_cvt_norm_status_flags(NEG_TWO_F32, u32(-2147483647))
        self.assertTrue(flags_neg.overflow)
        self.assertTrue(flags_neg.inexact)

        flags_inf = calc_cvt_norm_status_flags(POS_INF_F32, 2147483647)
        self.assertTrue(flags_inf.overflow)
        self.assertTrue(flags_inf.inexact)

    def test_underflow_and_inexact_on_subnormals(self) -> None:
        res = oracle_v_cvt_norm_i32_f32(MIN_SUBNORMAL_F32)
        self.assertEqual(res, 0)
        flags = calc_cvt_norm_status_flags(MIN_SUBNORMAL_F32, res)
        self.assertTrue(flags.underflow)
        self.assertTrue(flags.inexact)

    def test_inexact_flag_on_fractional_scaled(self) -> None:
        # 0.3f
        bits_0_3 = f32_to_bits(0.3)
        res = oracle_v_cvt_norm_i32_f32(bits_0_3)
        flags = calc_cvt_norm_status_flags(bits_0_3, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)

    def test_exact_conversions_no_flags(self) -> None:
        # 0.0 -> 0 (exact)
        flags_zero = calc_cvt_norm_status_flags(POS_ZERO_F32, 0)
        self.assertEqual(flags_zero.to_dict(), {
            "invalid": False, "div_by_zero": False, "overflow": False, "underflow": False, "inexact": False
        })

        # 1.0 -> 2147483647 (exact)
        flags_one = calc_cvt_norm_status_flags(ONE_F32, 2147483647)
        self.assertFalse(flags_one.overflow)
        self.assertFalse(flags_one.inexact)

        # -1.0 -> -2147483647 (exact)
        flags_neg_one = calc_cvt_norm_status_flags(NEG_ONE_F32, u32(-2147483647))
        self.assertFalse(flags_neg_one.overflow)
        self.assertFalse(flags_neg_one.inexact)

    def test_condition_codes_zero_and_negative(self) -> None:
        cc_zero = calc_cvt_norm_condition_codes(POS_ZERO_F32, 0)
        self.assertTrue(cc_zero.zero)
        self.assertFalse(cc_zero.negative)

        cc_neg = calc_cvt_norm_condition_codes(NEG_ONE_F32, u32(-2147483647))
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertTrue(cc_neg.min_neg)

    def test_condition_codes_saturation_and_carry_borrow(self) -> None:
        # +2.0 clamps to +1.0: sets max_pos, overflow, carry
        cc_pos = calc_cvt_norm_condition_codes(TWO_F32, 2147483647)
        self.assertTrue(cc_pos.max_pos)
        self.assertTrue(cc_pos.overflow)
        self.assertTrue(cc_pos.carry)
        self.assertFalse(cc_pos.borrow)

        # -2.0 clamps to -1.0: sets min_neg, overflow, borrow
        cc_neg = calc_cvt_norm_condition_codes(NEG_TWO_F32, u32(-2147483647))
        self.assertTrue(cc_neg.min_neg)
        self.assertTrue(cc_neg.overflow)
        self.assertTrue(cc_neg.borrow)
        self.assertFalse(cc_neg.carry)


# ---------------------------------------------------------------------------
# IEEE 754 boundary extremes, subnormals, and NaN handling
# ---------------------------------------------------------------------------

class TestValuCvtNormI32F32IEEE754Boundaries(unittest.TestCase):
    """Comprehensive IEEE 754 boundary, subnormal, and NaN handling tests."""

    def test_signed_zeros(self) -> None:
        self.assertEqual(oracle_v_cvt_norm_i32_f32(POS_ZERO_F32), 0)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_ZERO_F32), 0)

    def test_infinities(self) -> None:
        # +inf clamps to +1.0 (2147483647 = 0x7FFFFFFF)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(POS_INF_F32), 2147483647)
        # -inf clamps to -1.0 (-2147483647 = 0x80000001)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_INF_F32), u32(-2147483647))
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_INF_F32), 0x80000001)

    def test_quiet_and_signaling_nans_with_payloads(self) -> None:
        nans = [
            QNAN_F32,
            SNAN_F32,
            0x7F800001, 0x7F800010, 0x7FBFFFFF,  # SNaNs
            0x7FC00000, 0x7FC00001, 0x7FFFFFFF,  # QNaNs
            0xFF800001, 0xFF800010, 0xFFBFFFFF,  # Negative SNaNs
            0xFFC00000, 0xFFC00001, 0xFFFFFFFF,  # Negative QNaNs
        ]
        for n in nans:
            res = oracle_v_cvt_norm_i32_f32(n)
            self.assertEqual(res, 0, f"Expected 0 for NaN {n:#010x}, got {res:#010x}")

    def test_finite_extremes_and_saturation(self) -> None:
        # Max positive finite single float (~3.4e38) -> clamps to 2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(MAX_FINITE_F32), 2147483647)
        # Min negative finite single float (~-3.4e38) -> clamps to -2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(MIN_FINITE_NEG_F32), u32(-2147483647))

        # Values just above 1.0 (0x3F800001) clamp to 2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(0x3F800001), 2147483647)
        # Values just below -1.0 (0xBF800001) clamp to -2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(0xBF800001), u32(-2147483647))

    def test_subnormal_precision_and_underflow(self) -> None:
        # Smallest positive subnormal (2^-149) rounds to 0
        self.assertEqual(oracle_v_cvt_norm_i32_f32(MIN_SUBNORMAL_F32), 0)
        # Smallest negative subnormal (-2^-149) rounds to 0
        self.assertEqual(oracle_v_cvt_norm_i32_f32(0x80000001), 0)

        # Largest subnormal (0x007FFFFF ~ 1.1754942e-38) * 2147483647 -> 0
        self.assertEqual(oracle_v_cvt_norm_i32_f32(MAX_SUBNORMAL_F32), 0)

        # Minimum normal (0x00800000 = 2^-126) * 2147483647 -> 0
        self.assertEqual(oracle_v_cvt_norm_i32_f32(MIN_NORMAL_F32), 0)

    def test_ties_to_even_single_precision(self) -> None:
        # 0.5f * 2147483647 = 1073741823.5 -> ties to even integer 1073741824 (0x40000000)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(HALF_F32), 1073741824)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(HALF_F32), 0x40000000)
        # -0.5f * 2147483647 = -1073741823.5 -> ties to even integer -1073741824 (0xC0000000)
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_HALF_F32), u32(-1073741824))
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_HALF_F32), 0xC0000000)

    def test_boundary_values_near_unit(self) -> None:
        # 0x3F7FFFFF (1.0f - ulp) * 2147483647 = 2147483519.0 -> 2147483519
        self.assertEqual(oracle_v_cvt_norm_i32_f32(0x3F7FFFFF), 2147483519)
        # 0x3F800000 (1.0f) -> 2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(ONE_F32), 2147483647)
        # 0xBF800000 (-1.0f) -> -2147483647
        self.assertEqual(oracle_v_cvt_norm_i32_f32(NEG_ONE_F32), u32(-2147483647))


if __name__ == "__main__":
    unittest.main()
