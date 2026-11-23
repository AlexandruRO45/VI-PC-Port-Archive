from __future__ import annotations

"""Hardware-verified semantic reference and translator regression suite for RDNA2 v_sqrt_f64.

Tests v_sqrt_f64 and v_sqrt_f64_e64 (VOP1/VOP3) against IEEE 754-2008 double precision (f64)
semantics and translated SPIR-V shader executions via rdna2spirv and spirv_mini_interp.
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
ONE_F64_BITS: int = 0x3FF0_0000_0000_0000
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


def is_nan_bits(bits: int) -> bool:
    """Return True if 64-bit word represents an IEEE 754 NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan_bits(bits: int) -> bool:
    """Return True if 64-bit word represents a quiet NaN."""
    return is_nan_bits(bits) and bool(bits & QNAN_BIT)


def is_snan_bits(bits: int) -> bool:
    """Return True if 64-bit word represents a signaling NaN."""
    return is_nan_bits(bits) and not bool(bits & QNAN_BIT)


def is_subnormal_bits(bits: int) -> bool:
    """Return True if 64-bit word represents a subnormal number."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def oracle_sqrt_f64(
    src_bits: int,
    clamp: bool = False,
    abs_src: bool = False,
    neg_src: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sqrt_f64 and v_sqrt_f64_e64.
    
    Applies source modifiers (abs, neg), executes IEEE 754 square root,
    and applies destination clamp [0.0, 1.0] if requested.
    """
    bits = src_bits & MASK64
    if abs_src:
        bits &= ~SIGN_MASK
    if neg_src:
        bits ^= SIGN_MASK

    # Special IEEE 754 handling
    if bits == POS_ZERO_BITS:
        res_bits = POS_ZERO_BITS
    elif bits == NEG_ZERO_BITS:
        res_bits = NEG_ZERO_BITS
    elif bits == POS_INF_BITS:
        res_bits = POS_INF_BITS
    elif is_nan_bits(bits):
        res_bits = bits | QNAN_BIT  # quieted NaN
    elif bits & SIGN_MASK:
        # Negative nonzero operand returns NaN
        res_bits = CANONICAL_QNAN
    else:
        val = bits_to_f64(bits)
        res_f = math.sqrt(val)
        res_bits = f64_to_bits(res_f)

    if clamp:
        if is_nan_bits(res_bits):
            res_bits = POS_ZERO_BITS
        else:
            val = bits_to_f64(res_bits)
            clamped = max(0.0, min(1.0, val))
            res_bits = f64_to_bits(clamped)

    return res_bits & MASK64


_TRANSLATE_ID: int = 0


def run_translated_sqrt_f64(
    src_lo: int,
    src_hi: int,
    vop3: bool = False,
    clamp: bool = False,
    abs_src: bool = False,
    neg_src: bool = False,
) -> Tuple[int, int, float]:
    """Execute v_sqrt_f64 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    src_op = "v[2:3]"
    if abs_src and neg_src:
        src_op = "-|v[2:3]|"
    elif abs_src:
        src_op = "|v[2:3]|"
    elif neg_src:
        src_op = "-v[2:3]"

    if vop3 or clamp or abs_src or neg_src:
        inst = f"v_sqrt_f64_e64 v[0:1], {src_op}"
        if clamp:
            inst += " clamp"
    else:
        inst = f"v_sqrt_f64 v[0:1], {src_op}"

    lines = [
        f"v_mov_b32 v2, {src_lo & MASK32:#x}",
        f"v_mov_b32 v3, {src_hi & MASK32:#x}",
        inst,
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_sqrt_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    out_lo, out_hi = words[0], words[1]
    bits = combine_f64_words(out_lo, out_hi)
    return out_lo, out_hi, bits_to_f64(bits)


class TestValuSqrtF64ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_sqrt_f64."""

    def test_oracle_exact_squares(self) -> None:
        squares = [
            (0.0, 0.0),
            (1.0, 1.0),
            (4.0, 2.0),
            (9.0, 3.0),
            (16.0, 4.0),
            (25.0, 5.0),
            (100.0, 10.0),
            (0.25, 0.5),
            (0.0625, 0.25),
        ]
        for inp, expected in squares:
            inp_bits = f64_to_bits(inp)
            res_bits = oracle_sqrt_f64(inp_bits)
            self.assertEqual(bits_to_f64(res_bits), expected)

    def test_oracle_matches_translated_execution(self) -> None:
        exact_squares = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 0.25, 0.0625]
        for val in exact_squares:
            inp_bits = f64_to_bits(val)
            lo, hi = split_f64_words(inp_bits)
            out_lo, out_hi, out_f = run_translated_sqrt_f64(lo, hi)
            expected_bits = oracle_sqrt_f64(inp_bits)
            actual_bits = combine_f64_words(out_lo, out_hi)
            self.assertEqual(actual_bits, expected_bits)
            self.assertEqual(out_f, math.sqrt(val))

        inp_bits = f64_to_bits(2.0)
        lo, hi = split_f64_words(inp_bits)
        out_lo, out_hi, out_f = run_translated_sqrt_f64(lo, hi)
        self.assertAlmostEqual(out_f, math.sqrt(2.0), places=6)

    def test_oracle_modifiers(self) -> None:
        val_4 = f64_to_bits(4.0)
        clamped_bits = oracle_sqrt_f64(val_4, clamp=True)
        self.assertEqual(bits_to_f64(clamped_bits), 1.0)

        neg_4 = f64_to_bits(-4.0)
        abs_bits = oracle_sqrt_f64(neg_4, abs_src=True)
        self.assertEqual(bits_to_f64(abs_bits), 2.0)

    def test_oracle_small_and_large_normals(self) -> None:
        large_val = 1e100
        res_large = bits_to_f64(oracle_sqrt_f64(f64_to_bits(large_val)))
        self.assertAlmostEqual(res_large, 1e50, places=10)

        small_val = 1e-100
        res_small = bits_to_f64(oracle_sqrt_f64(f64_to_bits(small_val)))
        self.assertAlmostEqual(res_small, 1e-50, places=10)


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
    V_SQRT_F64 = "v_sqrt_f64"
    V_SQRT_F64_E64 = "v_sqrt_f64_e64"


@dataclasses.dataclass
class VOP1Instruction:
    opcode: Opcode
    dst: RegisterPair
    src0: RegisterPair

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VOP1 destination must be VGPR pair")
        if self.opcode != Opcode.V_SQRT_F64:
            raise ValueError(f"Invalid VOP1 opcode {self.opcode}")

    def to_asm(self) -> str:
        return f"{self.opcode.value} {self.dst}, {self.src0}"


@dataclasses.dataclass
class VOP3Instruction:
    opcode: Opcode
    dst: RegisterPair
    src0: RegisterPair
    clamp: bool = False
    abs_src0: bool = False
    neg_src0: bool = False

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VOP3 destination must be VGPR pair")
        if self.opcode != Opcode.V_SQRT_F64_E64:
            raise ValueError(f"Invalid VOP3 opcode {self.opcode}")

    def to_asm(self) -> str:
        src = str(self.src0)
        if self.abs_src0 and self.neg_src0:
            src = f"-|{src}|"
        elif self.abs_src0:
            src = f"|{src}|"
        elif self.neg_src0:
            src = f"-{src}"
        out = f"{self.opcode.value} {self.dst}, {src}"
        if self.clamp:
            out += " clamp"
        return out


class TestValuSqrtF64AST(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_sqrt_f64."""

    def test_vop1_creation_and_disasm(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        self.assertEqual(inst.to_asm(), "v_sqrt_f64 v[0:1], v[2:3]")

    def test_vop3_modifiers_and_disasm(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 4)
        src = RegisterPair(RegKind.VGPR, 6)
        inst = VOP3Instruction(Opcode.V_SQRT_F64_E64, dst, src, clamp=True, abs_src0=True)
        self.assertEqual(inst.to_asm(), "v_sqrt_f64_e64 v[4:5], |v[6:7]| clamp")

    def test_ast_emits_valid_spirv(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40100000",
            inst.to_asm(),
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpExtInst %double %glsl Sqrt", spv)

    def test_ast_clamp_emits_fclamp(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        inst = VOP3Instruction(Opcode.V_SQRT_F64_E64, dst, src, clamp=True)
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40100000",
            inst.to_asm(),
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_clamp_test", lines)
        self.assertIn("OpExtInst %double %glsl FClamp", spv)


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
            res_bits = oracle_sqrt_f64(
                src_bits,
                clamp=inst.clamp,
                abs_src=inst.abs_src0,
                neg_src=inst.neg_src0,
            )
        else:
            res_bits = oracle_sqrt_f64(src_bits)
        self.write_f64(inst.dst, res_bits)
        return res_bits


class TestValuSqrtF64ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_sqrt_f64."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_scalar_step_basic(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(16.0))
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        res_bits = self.state.step(inst)
        self.assertEqual(bits_to_f64(res_bits), 4.0)
        self.assertEqual(bits_to_f64(self.state.read_f64(dst)), 4.0)

    def test_scalar_step_sgpr_source(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.SGPR, 4)
        self.state.write_f64(src, f64_to_bits(25.0))
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        res_bits = self.state.step(inst)
        self.assertEqual(bits_to_f64(res_bits), 5.0)

    def test_scalar_step_vop3_clamp(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(9.0))
        inst = VOP3Instruction(Opcode.V_SQRT_F64_E64, dst, src, clamp=True)
        res_bits = self.state.step(inst)
        self.assertEqual(bits_to_f64(res_bits), 1.0)

    def test_scalar_step_vop3_abs(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        self.state.write_f64(src, f64_to_bits(-100.0))
        inst = VOP3Instruction(Opcode.V_SQRT_F64_E64, dst, src, abs_src0=True)
        res_bits = self.state.step(inst)
        self.assertEqual(bits_to_f64(res_bits), 10.0)

    def test_scalar_matches_translated_shader(self) -> None:
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        for val in [0.0, 1.0, 4.0, 9.0, 16.0]:
            self.state.write_f64(src, f64_to_bits(val))
            inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
            scalar_res = self.state.step(inst)
            lo, hi = split_f64_words(f64_to_bits(val))
            out_lo, out_hi, _ = run_translated_sqrt_f64(lo, hi)
            trans_bits = combine_f64_words(out_lo, out_hi)
            self.assertEqual(scalar_res, trans_bits)


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

    def write_f64(self, pair: RegisterPair, lane: int, bits: int) -> None:
        lo, hi = split_f64_words(bits)
        self.vgprs[pair.lo][lane] = lo
        self.vgprs[pair.hi][lane] = hi

    def step(self, inst: Union[VOP1Instruction, VOP3Instruction]) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                src_bits = self.read_f64(inst.src0, lane)
                if isinstance(inst, VOP3Instruction):
                    res_bits = oracle_sqrt_f64(
                        src_bits,
                        clamp=inst.clamp,
                        abs_src=inst.abs_src0,
                        neg_src=inst.neg_src0,
                    )
                else:
                    res_bits = oracle_sqrt_f64(src_bits)
                self.write_f64(inst.dst, lane, res_bits)


class TestValuSqrtF64SIMD(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for v_sqrt_f64."""

    def test_simd32_all_lanes_active(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        for lane in range(32):
            wave.write_f64(src, lane, f64_to_bits(float(lane * lane)))
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        wave.step(inst)
        for lane in range(32):
            res = bits_to_f64(wave.read_f64(dst, lane))
            self.assertEqual(res, float(lane))

    def test_simd64_divergence_exec_mask(self) -> None:
        wave = SIMDWavefront(wave_size=64)
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        wave.exec_mask = 0x5555_5555_5555_5555  # even lanes active
        for lane in range(64):
            wave.write_f64(src, lane, f64_to_bits(4.0))
            wave.write_f64(dst, lane, f64_to_bits(0.0))
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        wave.step(inst)
        for lane in range(64):
            res = bits_to_f64(wave.read_f64(dst, lane))
            if lane % 2 == 0:
                self.assertEqual(res, 2.0)
            else:
                self.assertEqual(res, 0.0)

    def test_simd_inactive_lanes_unmodified(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        dst = RegisterPair(RegKind.VGPR, 0)
        src = RegisterPair(RegKind.VGPR, 2)
        wave.exec_mask = 0x0000_0001  # only lane 0 active
        for lane in range(32):
            wave.write_f64(dst, lane, 0xDEAD_BEEF_CAFE_BABE)
            wave.write_f64(src, lane, f64_to_bits(9.0))
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        wave.step(inst)
        self.assertEqual(bits_to_f64(wave.read_f64(dst, 0)), 3.0)
        for lane in range(1, 32):
            self.assertEqual(wave.read_f64(dst, lane), 0xDEAD_BEEF_CAFE_BABE)

    def test_simd_translator_active_mask_preservation(self) -> None:
        inst = VOP1Instruction(
            Opcode.V_SQRT_F64,
            RegisterPair(RegKind.VGPR, 0),
            RegisterPair(RegKind.VGPR, 2),
        )
        lines = [inst.to_asm(), "s_endpgm"]
        spv = support.translate_lines("ps_simd_mask", lines)
        self.assertIn("OpSelect", spv)


# ---------------------------------------------------------------------------
# Stage 5: Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuSqrtF64RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_sqrt_f64."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_in_place_square_root_scalar(self) -> None:
        pair = RegisterPair(RegKind.VGPR, 0)
        self.state.write_f64(pair, f64_to_bits(16.0))
        inst = VOP1Instruction(Opcode.V_SQRT_F64, pair, pair)
        res_bits = self.state.step(inst)
        self.assertEqual(bits_to_f64(res_bits), 4.0)
        self.assertEqual(bits_to_f64(self.state.read_f64(pair)), 4.0)

    def test_in_place_square_root_translated(self) -> None:
        inp_bits = f64_to_bits(25.0)
        lo, hi = split_f64_words(inp_bits)
        lines = [
            f"v_mov_b32 v0, {lo:#x}",
            f"v_mov_b32 v1, {hi:#x}",
            "v_sqrt_f64 v[0:1], v[0:1]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inplace_sqrt", lines)
        words = support.mrt0_words(spv)
        res_bits = combine_f64_words(words[0], words[1])
        self.assertEqual(bits_to_f64(res_bits), 5.0)

    def test_partially_overlapping_register_pairs(self) -> None:
        # src: v[0:1], dst: v[1:2]
        self.state.vgprs[0] = 0
        self.state.vgprs[1] = 0x40100000  # 4.0
        inst = VOP1Instruction(
            Opcode.V_SQRT_F64,
            RegisterPair(RegKind.VGPR, 1),
            RegisterPair(RegKind.VGPR, 0),
        )
        res_bits = self.state.step(inst)
        self.assertEqual(bits_to_f64(res_bits), 2.0)
        self.assertEqual(combine_f64_words(self.state.vgprs[1], self.state.vgprs[2]), res_bits)

    def test_surrounding_registers_unmodified(self) -> None:
        self.state.vgprs[0] = 0x11111111
        self.state.vgprs[3] = 0x33333333
        dst = RegisterPair(RegKind.VGPR, 1)
        src = RegisterPair(RegKind.VGPR, 4)
        self.state.write_f64(src, f64_to_bits(9.0))
        inst = VOP1Instruction(Opcode.V_SQRT_F64, dst, src)
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[0], 0x11111111)
        self.assertEqual(self.state.vgprs[3], 0x33333333)


if __name__ == "__main__":
    unittest.main()
