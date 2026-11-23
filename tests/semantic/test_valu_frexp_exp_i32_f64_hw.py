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


# ---------------------------------------------------------------------------
# Stage 3: Deterministic scalar instruction interpreter
# ---------------------------------------------------------------------------

class ScalarMachineState:
    """Architectural state for executing VALU scalar instructions."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_f64(self, pair: RegisterPair) -> int:
        if pair.kind == RegKind.VGPR:
            return combine_f64_words(self.vgprs[pair.lo], self.vgprs[pair.hi])
        return combine_f64_words(self.sgprs[pair.lo], self.sgprs[pair.hi])

    def write_u32(self, reg: Register, val: int) -> None:
        val &= MASK32
        if reg.kind == RegKind.VGPR:
            self.vgprs[reg.index] = val
        else:
            self.sgprs[reg.index] = val

    def write_f64(self, pair: RegisterPair, bits: int) -> None:
        lo, hi = split_f64_words(bits)
        if pair.kind == RegKind.VGPR:
            self.vgprs[pair.lo] = lo
            self.vgprs[pair.hi] = hi
        else:
            self.sgprs[pair.lo] = lo
            self.sgprs[pair.hi] = hi

    def step(self, inst: Union[VOP1Instruction, VOP3Instruction]) -> int:
        src_bits = self.read_f64(inst.src0)
        if isinstance(inst, VOP3Instruction):
            res_u32 = oracle_frexp_exp_i32_f64(
                src_bits,
                abs_src=inst.abs_src0,
                neg_src=inst.neg_src0,
            )
        else:
            res_u32 = oracle_frexp_exp_i32_f64(src_bits)
        self.write_u32(inst.dst, res_u32)
        return res_u32


class TestValuFrexpExpI32F64ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_frexp_exp_i32_f64."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_scalar_step_basic(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(16.0))
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        res_u32 = self.state.step(inst)
        self.assertEqual(to_s32(res_u32), 5)
        self.assertEqual(to_s32(self.state.vgprs[0]), 5)

    def test_scalar_step_sgpr_source(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.SGPR, 4)
        self.state.write_f64(src, f64_to_bits(0.25))
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        res_u32 = self.state.step(inst)
        self.assertEqual(to_s32(res_u32), -1)

    def test_scalar_step_vop3_abs(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(-128.0))
        inst = VOP3Instruction(Opcode.V_FREXP_EXP_I32_F64_E64, dst, src, abs_src0=True)
        res_u32 = self.state.step(inst)
        self.assertEqual(to_s32(res_u32), 8)

    def test_scalar_matches_translated_shader(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        for val in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 0.5, 0.25]:
            self.state.write_f64(src, f64_to_bits(val))
            inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
            scalar_res = self.state.step(inst)
            lo, hi = split_f64_words(f64_to_bits(val))
            trans_res = run_translated_frexp_exp_f64(lo, hi)
            self.assertEqual(scalar_res, trans_res)


# ---------------------------------------------------------------------------
# Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking
# ---------------------------------------------------------------------------

class SIMDWavefront:
    """Wavefront model simulating 32-lane or 64-lane SIMD execution."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"wave_size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]

    def read_f64(self, pair: RegisterPair, lane: int) -> int:
        return combine_f64_words(self.vgprs[pair.lo][lane], self.vgprs[pair.hi][lane])

    def write_u32(self, reg: Register, lane: int, val: int) -> None:
        self.vgprs[reg.index][lane] = val & MASK32

    def write_f64(self, pair: RegisterPair, lane: int, bits: int) -> None:
        lo, hi = split_f64_words(bits)
        self.vgprs[pair.lo][lane] = lo
        self.vgprs[pair.hi][lane] = hi

    def step(self, inst: Union[VOP1Instruction, VOP3Instruction]) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                src_bits = self.read_f64(inst.src0, lane)
                if isinstance(inst, VOP3Instruction):
                    res_u32 = oracle_frexp_exp_i32_f64(
                        src_bits,
                        abs_src=inst.abs_src0,
                        neg_src=inst.neg_src0,
                    )
                else:
                    res_u32 = oracle_frexp_exp_i32_f64(src_bits)
                self.write_u32(inst.dst, lane, res_u32)


class TestValuFrexpExpI32F64SIMD(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for v_frexp_exp_i32_f64."""

    def test_simd32_all_lanes_active(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        for lane in range(32):
            wave.write_f64(src, lane, f64_to_bits(float(1 << lane)))
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        wave.step(inst)
        for lane in range(32):
            res = to_s32(wave.vgprs[0][lane])
            self.assertEqual(res, lane + 1)

    def test_simd64_divergence_exec_mask(self) -> None:
        wave = SIMDWavefront(wave_size=64)
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        wave.exec_mask = 0xAAAA_AAAA_AAAA_AAAA  # odd lanes active
        for lane in range(64):
            wave.write_f64(src, lane, f64_to_bits(8.0))
            wave.write_u32(dst, lane, 0x12345678)
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        wave.step(inst)
        for lane in range(64):
            if lane % 2 == 1:
                self.assertEqual(to_s32(wave.vgprs[0][lane]), 4)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0x12345678)

    def test_simd_inactive_lanes_unmodified(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        dst = Register(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        wave.exec_mask = 0x0000_0001  # only lane 0 active
        for lane in range(32):
            wave.write_u32(dst, lane, 0xCAFE_BABE)
            wave.write_f64(src, lane, f64_to_bits(16.0))
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        wave.step(inst)
        self.assertEqual(to_s32(wave.vgprs[0][0]), 5)
        for lane in range(1, 32):
            self.assertEqual(wave.vgprs[0][lane], 0xCAFE_BABE)

    def test_simd_translator_active_mask_preservation(self) -> None:
        inst = VOP1Instruction(
            Opcode.V_FREXP_EXP_I32_F64,
            Register(RegKind.VGPR, 0),
            RegisterPair(RegKind.VGPR, 2),
        )
        lines = [inst.to_asm(), "s_endpgm"]
        spv = support.translate_lines("ps_simd_frexp", lines)
        self.assertIn("OpSelect", spv)


# ---------------------------------------------------------------------------
# Stage 5: Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuFrexpExpI32F64RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_frexp_exp_i32_f64."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_dst_aliases_src_low_word(self) -> None:
        # dst v2, src v[2:3]
        dst = Register(RegKind.VGPR, 2)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(32.0))
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        res_u32 = self.state.step(inst)
        self.assertEqual(to_s32(res_u32), 6)
        self.assertEqual(to_s32(self.state.vgprs[2]), 6)

    def test_dst_aliases_src_high_word(self) -> None:
        # dst v3, src v[2:3]
        dst = Register(RegKind.VGPR, 3)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(64.0))
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        res_u32 = self.state.step(inst)
        self.assertEqual(to_s32(res_u32), 7)
        self.assertEqual(to_s32(self.state.vgprs[3]), 7)

    def test_surrounding_registers_unmodified(self) -> None:
        self.state.vgprs[0] = 0xAAAAAAAA
        self.state.vgprs[1] = 0xBBBBBBBB
        self.state.vgprs[4] = 0xCCCCCCCC
        dst = Register(RegKind.VGPR, 1)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(16.0))
        inst = VOP1Instruction(Opcode.V_FREXP_EXP_I32_F64, dst, src)
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[0], 0xAAAAAAAA)
        self.assertEqual(to_s32(self.state.vgprs[1]), 5)
        self.assertEqual(self.state.vgprs[4], 0xCCCCCCCC)


# ---------------------------------------------------------------------------
# Stage 6: Condition codes, status flags, and exponent classification
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FrexpExpStatusFlags:
    """Classification flags for frexp exponent values."""
    is_zero: bool
    is_positive: bool
    is_negative: bool
    raw_exp_s32: int


def compute_frexp_status_flags(exp_u32: int) -> FrexpExpStatusFlags:
    s = to_s32(exp_u32)
    return FrexpExpStatusFlags(
        is_zero=(s == 0),
        is_positive=(s > 0),
        is_negative=(s < 0),
        raw_exp_s32=s,
    )


class TestValuFrexpExpI32F64ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes and status flags for v_frexp_exp_i32_f64."""

    def test_status_flags_zero(self) -> None:
        exp_0 = oracle_frexp_exp_i32_f64(f64_to_bits(0.5))
        flags = compute_frexp_status_flags(exp_0)
        self.assertTrue(flags.is_zero)
        self.assertFalse(flags.is_positive)
        self.assertFalse(flags.is_negative)

    def test_status_flags_positive(self) -> None:
        exp_pos = oracle_frexp_exp_i32_f64(f64_to_bits(2.0))
        flags = compute_frexp_status_flags(exp_pos)
        self.assertFalse(flags.is_zero)
        self.assertTrue(flags.is_positive)
        self.assertFalse(flags.is_negative)
        self.assertEqual(flags.raw_exp_s32, 2)

    def test_status_flags_negative(self) -> None:
        exp_neg = oracle_frexp_exp_i32_f64(f64_to_bits(0.25))
        flags = compute_frexp_status_flags(exp_neg)
        self.assertFalse(flags.is_zero)
        self.assertFalse(flags.is_positive)
        self.assertTrue(flags.is_negative)
        self.assertEqual(flags.raw_exp_s32, -1)


if __name__ == "__main__":
    unittest.main()
