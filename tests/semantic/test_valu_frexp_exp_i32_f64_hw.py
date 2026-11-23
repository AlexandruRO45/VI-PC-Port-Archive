from __future__ import annotations

"""Hardware-verified semantic reference and translator regression suite for RDNA2 v_frexp_exp_i32_f64.

Tests v_frexp_exp_i32_f64 and v_frexp_exp_i32_f64_e64 (VOP1/VOP3) against IEEE 754-2008 double precision
exponent extraction semantics and translated SPIR-V shader executions via rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

# IEEE 754 binary64 constants
SIGN_MASK: int = 0x8000_0000_0000_0000
EXP_MASK: int = 0x7FF0_0000_0000_0000
FRAC_MASK: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT: int = 0x0008_0000_0000_0000
CANONICAL_QNAN: int = 0x7FF8_0000_0000_0000

POS_ZERO_BITS: int = 0x0000_0000_0000_0000
NEG_ZERO_BITS: int = 0x8000_0000_0000_0000
POS_INF_BITS: int = 0x7FF0_0000_0000_0000
NEG_INF_BITS: int = 0xFFF0_0000_0000_0000
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def f64_to_bits(val: float) -> int:
    """Convert Python float (IEEE 754 binary64) to 64-bit integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_f64_words(bits: int) -> Tuple[int, int]:
    """Split 64-bit qword into (lo_u32, hi_u32) for VGPR pair [v_lo:v_hi]."""
    return bits & MASK32, (bits >> 32) & MASK32


def combine_f64_words(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit qword."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def to_s32(val: int) -> int:
    """Convert unsigned 32-bit int to signed 32-bit int."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def to_u32(val: int) -> int:
    """Convert signed or unsigned int to unsigned 32-bit int."""
    return val & MASK32


def oracle_frexp_exp_i32_f64(
    src_bits: int,
    abs_src: bool = False,
    neg_src: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_frexp_exp_i32_f64.
    
    Extracts the unbiased exponent e such that x = m * 2^e where 0.5 <= |m| < 1.0.
    For zero, NaN, or infinity, returns 0.
    Returns 32-bit unsigned integer bit pattern (two's complement signed integer).
    """
    bits = src_bits & MASK64
    if abs_src:
        bits &= ~SIGN_MASK
    if neg_src:
        bits ^= SIGN_MASK

    val = bits_to_f64(bits)
    if math.isnan(val) or math.isinf(val) or val == 0.0:
        return 0

    _, exp = math.frexp(val)
    return to_u32(exp)


_TRANSLATE_ID: int = 0


def run_translated_frexp_exp_f64(
    src_lo: int,
    src_hi: int,
    vop3: bool = False,
    abs_src: bool = False,
    neg_src: bool = False,
) -> int:
    """Execute v_frexp_exp_i32_f64 through rdna2spirv translation and spirv_mini_interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    src_op = "v[2:3]"
    if abs_src and neg_src:
        src_op = "-|v[2:3]|"
    elif abs_src:
        src_op = "|v[2:3]|"
    elif neg_src:
        src_op = "-v[2:3]"

    if vop3 or abs_src or neg_src:
        inst = f"v_frexp_exp_i32_f64_e64 v0, {src_op}"
    else:
        inst = f"v_frexp_exp_i32_f64 v0, {src_op}"

    lines = [
        f"v_mov_b32 v2, {src_lo & MASK32:#x}",
        f"v_mov_b32 v3, {src_hi & MASK32:#x}",
        inst,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_frexp_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


class TestValuFrexpExpI32F64ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_frexp_exp_i32_f64."""

    def test_oracle_powers_of_two(self) -> None:
        cases = [
            (1.0, 1),
            (2.0, 2),
            (4.0, 3),
            (8.0, 4),
            (0.5, 0),
            (0.25, -1),
            (0.125, -2),
        ]
        for val, expected_exp in cases:
            bits = f64_to_bits(val)
            res_u32 = oracle_frexp_exp_i32_f64(bits)
            self.assertEqual(to_s32(res_u32), expected_exp)

    def test_oracle_matches_translated_execution(self) -> None:
        test_values = [0.0, 0.5, 1.0, 2.0, 3.141592653589793, 4.0, 8.0, 0.25, 1024.0, 1e-10]
        for val in test_values:
            inp_bits = f64_to_bits(val)
            lo, hi = split_f64_words(inp_bits)
            out_u32 = run_translated_frexp_exp_f64(lo, hi)
            expected_u32 = oracle_frexp_exp_i32_f64(inp_bits)
            self.assertEqual(out_u32, expected_u32)
            if val != 0.0:
                self.assertEqual(to_s32(out_u32), math.frexp(val)[1])

    def test_oracle_modifiers(self) -> None:
        neg_val = f64_to_bits(-4.0)
        res_abs = oracle_frexp_exp_i32_f64(neg_val, abs_src=True)
        self.assertEqual(to_s32(res_abs), 3)

        pos_val = f64_to_bits(4.0)
        res_neg = oracle_frexp_exp_i32_f64(pos_val, neg_src=True)
        self.assertEqual(to_s32(res_neg), 3)

    def test_oracle_special_values(self) -> None:
        self.assertEqual(to_s32(oracle_frexp_exp_i32_f64(POS_ZERO_BITS)), 0)
        self.assertEqual(to_s32(oracle_frexp_exp_i32_f64(NEG_ZERO_BITS)), 0)
        self.assertEqual(to_s32(oracle_frexp_exp_i32_f64(POS_INF_BITS)), 0)
        self.assertEqual(to_s32(oracle_frexp_exp_i32_f64(NEG_INF_BITS)), 0)
        self.assertEqual(to_s32(oracle_frexp_exp_i32_f64(CANONICAL_QNAN)), 0)


# ---------------------------------------------------------------------------
# Stage 2: Symbolic AST definitions and typed expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
        if self.kind == RegKind.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of range [0, 105]")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclasses.dataclass(frozen=True)
class RegisterPair:
    kind: RegKind
    lo: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.lo <= 254):
            raise ValueError(f"VGPR pair base {self.lo} out of range [0, 254]")
        if self.kind == RegKind.SGPR and not (0 <= self.lo <= 104):
            raise ValueError(f"SGPR pair base {self.lo} out of range [0, 104]")

    @property
    def hi(self) -> int:
        return self.lo + 1

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}[{self.lo}:{self.hi}]"


class Opcode(enum.Enum):
    V_FREXP_EXP_I32_F64 = "v_frexp_exp_i32_f64"
    V_FREXP_EXP_I32_F64_E64 = "v_frexp_exp_i32_f64_e64"


@dataclasses.dataclass
class VOP1Instruction:
    opcode: Opcode
    dst: Register
    src0: RegisterPair

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VOP1 destination must be VGPR")
        if self.opcode != Opcode.V_FREXP_EXP_I32_F64:
            raise ValueError(f"Invalid VOP1 opcode {self.opcode}")

    def to_asm(self) -> str:
        return f"{self.opcode.value} {self.dst}, {self.src0}"


@dataclasses.dataclass
class VOP3Instruction:
    opcode: Opcode
    dst: Register
    src0: RegisterPair
    abs_src0: bool = False
    neg_src0: bool = False

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VOP3 destination must be VGPR")
        if self.opcode != Opcode.V_FREXP_EXP_I32_F64_E64:
            raise ValueError(f"Invalid VOP3 opcode {self.opcode}")

    def to_asm(self) -> str:
        src = str(self.src0)
        if self.abs_src0 and self.neg_src0:
            src = f"-|{src}|"
        elif self.abs_src0:
            src = f"|{src}|"
        elif self.neg_src0:
            src = f"-{src}"
        return f"{self.opcode.value} {self.dst}, {src}"


class TestValuFrexpExpI32F64AST(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_frexp_exp_i32_f64."""

    def test_vop1_creation_and_disasm(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        self.assertEqual(inst.to_asm(), "v_frexp_exp_i32_f64 v0, v[2:3]")

    def test_vop3_modifiers_and_disasm(self) -> None:
        dst = Register(RegKind.VGPR, 4)
        src = RegisterPair(RegKind.VGPR, 6)
        inst = VOP3Instruction(Opcode.V_FREXP_EXP_I32_F64_E64, dst, src, abs_src0=True)
        self.assertEqual(inst.to_asm(), "v_frexp_exp_i32_f64_e64 v4, |v[6:7]|")

    def test_ast_emits_valid_spirv(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x3ff00000",
            inst.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_frexp", lines)
        self.assertIn("OpBitFieldUExtract", spv)
        self.assertIn("FindUMsb", spv)


if __name__ == "__main__":
    unittest.main()
