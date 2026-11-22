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


if __name__ == "__main__":
    unittest.main()
