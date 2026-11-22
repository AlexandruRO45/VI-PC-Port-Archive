"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_i32_f16.

Models IEEE 754-2008 / IEEE 754-2019 binary16 (half-precision float) to 32-bit signed normalized integer
(SNORM32) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_cvt_norm_i32_f16: D0.i32 = f16_to_snorm(S0.f16)

RDNA2 Floating-Point Conversion ISA Rules:
  1. Register Widths and Destination Semantics:
     - v_cvt_norm_i32_f16 operates on 16-bit half-precision floating-point source in S0[15:0].
     - Writes a signed normalized 32-bit integer (SNORM32) into D0[31:0].
     - Fully writes the 32-bit destination VGPR.
  2. Normalized Signed Integer Conversion & Rounding:
     - Scales normalized range [-1.0, 1.0] across 32-bit signed integer range [-2147483647, 2147483647]
       using scale factor 2147483647.0 (2^31 - 1).
     - Rounding mode: IEEE 754 round-to-nearest-even (RNE) is applied to the scaled value (val * 2147483647.0).
     - Midpoint ties (e.g. 0.5 * 2147483647.0 = 1073741823.5) round to the nearest even integer (1073741824 = 0x40000000).
  3. Boundary Extremes & Special Values:
     - Values <= -1.0 (including negative values <= -1.0 and -infinity) saturate to -2147483647.
     - Values >= 1.0 (including positive numbers >= 1.0 and +infinity) saturate to 2147483647 (0x7FFFFFFF).
     - Subnormal half-precision inputs round to 0 without faulting.
     - Signed zeros (+0.0 and -0.0) convert to 0.
  4. NaN Handling:
     - Any NaN input (quiet NaN or signaling NaN, positive or negative) converts to 0.
  5. VOP1 / VOP3 Encoding and Modifiers:
     - VOP1 (e32): Compact encoding with single source VGPR or SGPR or inline constant.
     - VOP3 (e64): Extended encoding supporting input modifiers:
       * abs (|x|): Clears sign bit before conversion.
       * neg (-x): Inverts sign bit before conversion.
       * clamp: Clamps converted result to [0, 2147483647].

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import math
import struct
import unittest
from enum import Enum, auto
from fractions import Fraction
from typing import Any, List, Optional, Tuple, Union

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary16 Bit Patterns
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
FRAC_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200

POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
QNAN_F16: int = 0x7E00
NEG_QNAN_F16: int = 0xFE00
SNAN_F16: int = 0x7C01
NEG_SNAN_F16: int = 0xFC01

ONE_F16: int = 0x3C00          # 1.0
NEG_ONE_F16: int = 0xBC00      # -1.0
HALF_F16: int = 0x3800         # 0.5
NEG_HALF_F16: int = 0xB800     # -0.5
QUARTER_F16: int = 0x3400      # 0.25
NEG_QUARTER_F16: int = 0xB400  # -0.25
THREE_QUARTERS_F16: int = 0x3A00  # 0.75
NEG_THREE_QUARTERS_F16: int = 0xBA00  # -0.75
TWO_F16: int = 0x4000          # 2.0
NEG_TWO_F16: int = 0xC000      # -2.0
TEN_F16: int = 0x4900          # 10.0
NEG_TEN_F16: int = 0xC900      # -10.0

MAX_NORMAL_F16: int = 0x7BFF   # 65504.0
MIN_NORMAL_F16: int = 0x0400   # 2^-14 ~= 6.1035e-5
MAX_SUBNORMAL_F16: int = 0x03FF # (1023/1024) * 2^-14
MIN_SUBNORMAL_F16: int = 0x0001 # 2^-24 ~= 5.9605e-8

# SNORM32 Constants
SNORM32_MAX: int = 0x7FFFFFFF          # 2147483647
SNORM32_MIN: int = -2147483647         # -2147483647 (two's complement 0x80000001)
SNORM32_MIN_RAW: int = 0x80000001
SNORM32_ZERO: int = 0x00000000
SNORM32_HALF: int = 0x40000000         # 1073741824
SNORM32_NEG_HALF_RAW: int = 0xC0000000 # -1073741824 in 32-bit two's complement


def f16_to_bits(val: float) -> int:
    """Pack Python float into 16-bit binary16 integer bit pattern."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF_F16 if val > 0 else NEG_INF_F16


def bits_to_f16(bits: int) -> float:
    """Unpack 16-bit integer bit pattern into Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 binary16 infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 subnormal value."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & FRAC_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits: int, abs_val: bool = False, neg: bool = False) -> int:
    """Applies VOP3 input modifiers: abs clears sign, neg inverts sign."""
    res = bits & MASK16
    if abs_val:
        res &= ~SIGN_MASK_F16
    if neg:
        res ^= SIGN_MASK_F16
    return res


def round_snorm32_ties_to_even(frac_val: Fraction) -> int:
    """Rounds an exact Fraction to nearest integer with ties to even."""
    sign = -1 if frac_val < 0 else 1
    pos_frac = abs(frac_val)
    n = pos_frac.numerator
    d = pos_frac.denominator
    q, r = divmod(n, d)
    rem_doubled = 2 * r
    if rem_doubled < d:
        mag = q
    elif rem_doubled > d:
        mag = q + 1
    else:
        # Exact tie: round to even integer
        mag = q if (q % 2 == 0) else q + 1
    return sign * mag


def ref_cvt_norm_i32_f16(
    src16: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_norm_i32_f16.

    Converts 16-bit half-precision floating-point value to 32-bit signed normalized integer.
    Returns 32-bit unsigned bit pattern (two's complement representation).
    """
    s_mod = apply_f16_modifiers(src16, abs_val=abs_val, neg=neg)

    # 1. NaN inputs convert to 0
    if is_nan_f16(s_mod):
        return SNORM32_ZERO

    # 2. Signed zeros convert to 0
    if is_zero_f16(s_mod):
        return SNORM32_ZERO

    # 3. Infinities
    if is_inf_f16(s_mod):
        is_neg = bool(s_mod & SIGN_MASK_F16)
        res_int = SNORM32_MIN if is_neg else SNORM32_MAX
        if clamp and res_int < 0:
            res_int = 0
        return res_int & MASK32

    # 4. Finite values
    f_val = bits_to_f16(s_mod)
    if f_val >= 1.0:
        res_int = SNORM32_MAX
    elif f_val <= -1.0:
        res_int = SNORM32_MIN
    else:
        scaled_frac = Fraction(f_val) * 2147483647
        res_int = round_snorm32_ties_to_even(scaled_frac)
        if res_int < SNORM32_MIN:
            res_int = SNORM32_MIN
        elif res_int > SNORM32_MAX:
            res_int = SNORM32_MAX

    if clamp and res_int < 0:
        res_int = 0

    return res_int & MASK32


class TestValuCvtNormI32F16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_zero_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(POS_ZERO_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_ZERO_F16), SNORM32_ZERO)

    def test_unit_and_boundary_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(ONE_F16), SNORM32_MAX)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_ONE_F16), SNORM32_MIN_RAW)
        self.assertEqual(ref_cvt_norm_i32_f16(HALF_F16), SNORM32_HALF)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_HALF_F16), SNORM32_NEG_HALF_RAW)
        self.assertEqual(ref_cvt_norm_i32_f16(TWO_F16), SNORM32_MAX)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_TWO_F16), SNORM32_MIN_RAW)

    def test_infinity_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(POS_INF_F16), SNORM32_MAX)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_INF_F16), SNORM32_MIN_RAW)

    def test_nan_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(QNAN_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_QNAN_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(SNAN_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_SNAN_F16), SNORM32_ZERO)

    def test_tie_breaking_nearest_even(self) -> None:
        # 0.5 * 2147483647 = 1073741823.5 -> 1073741824 (even)
        self.assertEqual(ref_cvt_norm_i32_f16(HALF_F16), 1073741824)
        self.assertEqual(ref_cvt_norm_i32_f16(HALF_F16) % 2, 0)

        # -0.5 * 2147483647 = -1073741823.5 -> -1073741824 (even)
        raw_neg = ref_cvt_norm_i32_f16(NEG_HALF_F16)
        self.assertEqual(raw_neg, (-1073741824) & MASK32)

    def test_input_modifiers(self) -> None:
        # abs clears negative sign
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_ONE_F16, abs_val=True), SNORM32_MAX)
        # neg inverts sign
        self.assertEqual(ref_cvt_norm_i32_f16(ONE_F16, neg=True), SNORM32_MIN_RAW)
        # clamp modifier clips negatives to 0
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_HALF_F16, clamp=True), SNORM32_ZERO)

# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class Opcode(Enum):
    V_CVT_NORM_I32_F16 = "v_cvt_norm_i32_f16"


class Encoding(Enum):
    VOP1 = "vop1"
    VOP3 = "vop3"


class RegKind(Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"


class SubWord(Enum):
    FULL = "full"
    LO = "lo"
    HI = "hi"


@dataclasses.dataclass(frozen=True)
class Register:
    """Typed architectural register descriptor with optional 16-bit sub-dword indexing."""
    kind: RegKind
    index: int
    subword: SubWord = SubWord.FULL

    def __post_init__(self) -> None:
        if self.index < 0 or self.index > 255:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")

    def name(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        base = f"{prefix}{self.index}"
        if self.subword == SubWord.LO:
            return f"{base}.l"
        elif self.subword == SubWord.HI:
            return f"{base}.h"
        return base


@dataclasses.dataclass(frozen=True)
class Immediate:
    """Immediate floating-point or integer constant operand."""
    value: Union[int, float]
    is_f16: bool = True

    def bits16(self) -> int:
        if isinstance(self.value, float):
            return f16_to_bits(self.value)
        return int(self.value) & MASK16


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST representation of a VALU conversion instruction."""
    opcode: Opcode
    dst: Register
    src0: Union[Register, Immediate]
    abs_src0: bool = False
    neg_src0: bool = False
    clamp: bool = False
    encoding: Encoding = Encoding.VOP1

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VALU destination must be a VGPR")
        if self.encoding == Encoding.VOP1 and (self.abs_src0 or self.neg_src0 or self.clamp):
            raise ValueError("VOP1 encoding does not support modifiers or clamp")

    def asm(self) -> str:
        suffix = "_e64" if self.encoding == Encoding.VOP3 else "_e32"
        s0_str = self.src0.name() if isinstance(self.src0, Register) else f"{self.src0.bits16():#06x}"
        if self.abs_src0:
            s0_str = f"|{s0_str}|"
        if self.neg_src0:
            s0_str = f"-{s0_str}"
        clamp_str = " clamp" if self.clamp else ""
        return f"{self.opcode.value}{suffix} {self.dst.name()}, {s0_str}{clamp_str}"


class TestValuCvtNormI32F16AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_and_encoding(self) -> None:
        self.assertEqual(Opcode.V_CVT_NORM_I32_F16.value, "v_cvt_norm_i32_f16")
        self.assertEqual(Encoding.VOP1.value, "vop1")
        self.assertEqual(Encoding.VOP3.value, "vop3")

    def test_register_descriptors(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v1_l = Register(RegKind.VGPR, 1, SubWord.LO)
        s2_h = Register(RegKind.SGPR, 2, SubWord.HI)
        self.assertEqual(v0.name(), "v0")
        self.assertEqual(v1_l.name(), "v1.l")
        self.assertEqual(s2_h.name(), "s2.h")

    def test_immediate_wrapping(self) -> None:
        imm = Immediate(0.5)
        self.assertEqual(imm.bits16(), HALF_F16)

    def test_instruction_construction_and_validation(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        self.assertEqual(inst.asm(), "v_cvt_norm_i32_f16_e32 v0, v1.l")

        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_NORM_I32_F16,
                dst=Register(RegKind.SGPR, 0),
                src0=Register(RegKind.VGPR, 1),
            )

# ---------------------------------------------------------------------------
# Deterministic Scalar VALU Interpreter
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Architectural scalar register file and execution engine for VALU conversion instructions."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.vcc: int = 0
        self.exec_mask: int = MASK64
        self.m0: int = 0
        self.status: int = 0

    def read_reg32(self, kind: RegKind, index: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[index] & MASK32
        return self.sgprs[index] & MASK32

    def write_reg32(self, kind: RegKind, index: int, val32: int) -> None:
        val32 &= MASK32
        if kind == RegKind.VGPR:
            self.vgprs[index] = val32
        else:
            self.sgprs[index] = val32

    def read_reg16(self, reg: Register) -> int:
        raw32 = self.read_reg32(reg.kind, reg.index)
        if reg.subword == SubWord.HI:
            return (raw32 >> 16) & MASK16
        return raw32 & MASK16

    def read_operand16(self, op: Union[Register, Immediate]) -> int:
        if isinstance(op, Immediate):
            return op.bits16()
        return self.read_reg16(op)

    def step(self, inst: Instruction) -> int:
        if inst.opcode != Opcode.V_CVT_NORM_I32_F16:
            raise ValueError(f"Unsupported opcode in VALU interpreter: {inst.opcode}")

        src_bits = self.read_operand16(inst.src0)
        res_snorm32 = ref_cvt_norm_i32_f16(
            src_bits,
            clamp=inst.clamp,
            neg=inst.neg_src0,
            abs_val=inst.abs_src0,
        )
        self.write_reg32(inst.dst.kind, inst.dst.index, res_snorm32)
        return res_snorm32


class TestValuCvtNormI32F16ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar VALU instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_basic_vgpr_execution(self) -> None:
        self.interp.write_reg32(RegKind.VGPR, 1, ONE_F16)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, SNORM32_MAX)
        self.assertEqual(self.interp.read_reg32(RegKind.VGPR, 0), SNORM32_MAX)

    def test_negative_vgpr_execution(self) -> None:
        self.interp.write_reg32(RegKind.VGPR, 1, NEG_ONE_F16)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, SNORM32_MIN_RAW)

    def test_sgpr_source_execution(self) -> None:
        self.interp.write_reg32(RegKind.SGPR, 4, HALF_F16)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.SGPR, 4, SubWord.LO),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, SNORM32_HALF)

    def test_immediate_source_execution(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0.25),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 536870912)

    def test_vcc_unmodified(self) -> None:
        canary = 0x1234567890ABCDEF
        self.interp.vcc = canary
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(1.0),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, canary)

# ---------------------------------------------------------------------------
# SIMD Wavefront Multi-Lane Divergence and EXEC Masking
# ---------------------------------------------------------------------------

class SIMDWavefrontInterpreter:
    """Multi-lane SIMD vector execution model supporting Wave32 and Wave64 execution with EXEC masking."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: List[int] = [0] * 106
        self.vcc: int = 0

    def set_exec_mask(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def read_lane_reg32(self, lane: int, kind: RegKind, index: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][index] & MASK32
        return self.sgprs[index] & MASK32

    def write_lane_reg32(self, lane: int, kind: RegKind, index: int, val32: int) -> None:
        val32 &= MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][index] = val32
        else:
            self.sgprs[index] = val32

    def read_lane_reg16(self, lane: int, reg: Register) -> int:
        raw32 = self.read_lane_reg32(lane, reg.kind, reg.index)
        if reg.subword == SubWord.HI:
            return (raw32 >> 16) & MASK16
        return raw32 & MASK16

    def read_lane_operand16(self, lane: int, op: Union[Register, Immediate]) -> int:
        if isinstance(op, Immediate):
            return op.bits16()
        return self.read_lane_reg16(lane, op)

    def step(self, inst: Instruction) -> List[Optional[int]]:
        if inst.opcode != Opcode.V_CVT_NORM_I32_F16:
            raise ValueError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")

        results: List[Optional[int]] = [None] * self.wave_size
        for lane in range(self.wave_size):
            if not (self.exec_mask & (1 << lane)):
                continue

            src_bits = self.read_lane_operand16(lane, inst.src0)
            res_snorm32 = ref_cvt_norm_i32_f16(
                src_bits,
                clamp=inst.clamp,
                neg=inst.neg_src0,
                abs_val=inst.abs_src0,
            )
            self.write_lane_reg32(lane, inst.dst.kind, inst.dst.index, res_snorm32)
            results[lane] = res_snorm32

        return results


class TestValuCvtNormI32F16SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F16)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)
        self.assertEqual(len(results), 32)
        self.assertTrue(all(r == SNORM32_MAX for r in results))

    def test_wave32_divergent_lanes(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        test_values = [
            0.0, 0.25, 0.5, 0.75, 1.0, 2.0, -0.0, -0.25,
            -0.5, -0.75, -1.0, -2.0, float("inf"), float("-inf"), float("nan"), 0.5,
            0.1, -0.1, 0.9, -0.9, 0.001, -0.001, 10.0, -10.0,
            0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 1.0, -1.0
        ]
        for lane, fval in enumerate(test_values):
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, f16_to_bits(fval))

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)

        for lane, fval in enumerate(test_values):
            expected = ref_cvt_norm_i32_f16(f16_to_bits(fval))
            self.assertEqual(results[lane], expected)
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), expected)

    def test_wave32_exec_mask_checkerboard(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xCAFEBABE
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, canary)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, HALF_F16)

        sim.set_exec_mask(0x55555555)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], SNORM32_HALF)
                self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), SNORM32_HALF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), canary)

    def test_wave64_halves(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=64)
        for lane in range(64):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, 0x11223344)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F16)

        sim.set_exec_mask(0xFFFFFFFF00000000)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_I32_F16,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), 0x11223344)
        for lane in range(32, 64):
            self.assertEqual(results[lane], SNORM32_MAX)
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), SNORM32_MAX)

if __name__ == "__main__":
    unittest.main()
