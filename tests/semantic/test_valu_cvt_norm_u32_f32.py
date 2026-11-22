"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_u32_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 (single-precision float) to 32-bit unsigned normalized integer
(UNORM32) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_cvt_norm_u32_f32: D0.u32 = f32_to_unorm(S0.f32)

RDNA2 Floating-Point Conversion ISA Rules:
  1. Register Widths and Destination Semantics:
     - v_cvt_norm_u32_f32 operates on a 32-bit single-precision floating-point source in S0[31:0].
     - Writes a 32-bit unsigned normalized integer (UNORM32) into D0[31:0].
     - Fully writes the 32-bit destination VGPR.
  2. Normalized Integer Conversion & Rounding:
     - Scales normalized range [0.0, 1.0] across full 32-bit integer range [0, 4294967295] (0xFFFFFFFF).
     - Rounding mode: IEEE 754 round-to-nearest-even (RNE) is applied to the scaled value (val * 4294967295.0).
     - Midpoint ties (e.g. 0.5 * 4294967295.0 = 2147483647.5) round to the nearest even integer (2147483648 = 0x80000000).
  3. Boundary Extremes & Special Values:
     - Values <= 0.0 (including +0.0, -0.0, negative finite values, and -infinity) saturate/clamp to 0.
     - Values >= 1.0 (including positive numbers > 1.0 and +infinity) saturate/clamp to 4294967295 (0xFFFFFFFF).
     - Subnormal single-precision inputs round to 0 without faulting.
  4. NaN Handling:
     - Any NaN input (quiet NaN or signaling NaN, positive or negative) converts to 0.
  5. VOP1 / VOP3 Encoding and Modifiers:
     - VOP1 (e32): Compact encoding with single source VGPR or SGPR or inline constant.
     - VOP3 (e64): Extended encoding supporting input modifiers:
       * abs (|x|): Clears sign bit before conversion.
       * neg (-x): Inverts sign bit before conversion.
       * neg(abs(x)): Inverts sign after abs (forcing negative, which subsequently yields 0).
       * clamp: Clamps converted result. For normalized conversion, output is already within [0, 0xFFFFFFFF].

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

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary32 Bit Patterns
SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF
QNAN_BIT_F32: int = 0x00400000

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
NEG_QNAN_F32: int = 0xFFC00000
SNAN_F32: int = 0x7F800001
NEG_SNAN_F32: int = 0xFF800001

ONE_F32: int = 0x3F800000          # 1.0f
NEG_ONE_F32: int = 0xBF800000      # -1.0f
HALF_F32: int = 0x3F000000         # 0.5f
NEG_HALF_F32: int = 0xBF000000     # -0.5f
QUARTER_F32: int = 0x3E800000      # 0.25f
THREE_QUARTERS_F32: int = 0x3F400000  # 0.75f
TWO_F32: int = 0x40000000          # 2.0f
NEG_TWO_F32: int = 0xC0000000      # -2.0f
TEN_F32: int = 0x41200000          # 10.0f
NEG_TEN_F32: int = 0xC1200000      # -10.0f

MAX_NORMAL_F32: int = 0x7F7FFFFF   # (2 - 2^-23) * 2^127 ~= 3.4028e38
MIN_NORMAL_F32: int = 0x00800000   # 2^-126 ~= 1.1755e-38
MAX_SUBNORMAL_F32: int = 0x007FFFFF # (1 - 2^-23) * 2^-126
MIN_SUBNORMAL_F32: int = 0x00000001 # 2^-149 ~= 1.4013e-45

# UNORM32 Constants
UNORM32_MIN: int = 0x00000000
UNORM32_MAX: int = 0xFFFFFFFF
UNORM32_HALF: int = 0x80000000
UNORM32_QUARTER: int = 0x40000000
UNORM32_THREE_QUARTERS: int = 0xC0000000


def f32_to_bits(val: float) -> int:
    """Pack Python float into 32-bit binary32 integer bit pattern."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def bits_to_f32(bits: int) -> float:
    """Unpack 32-bit integer bit pattern into Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 NaN."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) != 0)


def is_qnan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 Quiet NaN."""
    return is_nan_f32(bits) and bool(bits & QNAN_BIT_F32)


def is_snan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 Signaling NaN."""
    return is_nan_f32(bits) and not bool(bits & QNAN_BIT_F32)


def is_inf_f32(bits: int) -> bool:
    """Return True if bits represent IEEE 754 binary32 infinity."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) == 0)


def is_subnormal_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 subnormal value."""
    return ((bits & EXP_MASK_F32) == 0) and ((bits & MANT_MASK_F32) != 0)


def is_zero_f32(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F32) == 0


def apply_f32_modifiers(bits: int, abs_val: bool = False, neg: bool = False) -> int:
    """Applies VOP3 input modifiers: abs clears sign, neg inverts sign."""
    res = bits & MASK32
    if abs_val:
        res &= ~SIGN_MASK_F32
    if neg:
        res ^= SIGN_MASK_F32
    return res


def round_unorm32_ties_to_even(frac_val: Fraction) -> int:
    """Rounds an exact non-negative Fraction to nearest integer with ties to even."""
    n = frac_val.numerator
    d = frac_val.denominator
    q, r = divmod(n, d)
    rem_doubled = 2 * r
    if rem_doubled < d:
        return q
    elif rem_doubled > d:
        return q + 1
    else:
        # Exact tie: round to even integer
        return q if (q % 2 == 0) else q + 1


def ref_cvt_norm_u32_f32(
    src32: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_norm_u32_f32.

    Converts 32-bit single-precision floating-point value to 32-bit unsigned normalized integer.
    """
    s_mod = apply_f32_modifiers(src32, abs_val=abs_val, neg=neg)

    # 1. NaN inputs convert to 0
    if is_nan_f32(s_mod):
        return UNORM32_MIN

    # 2. Signed zeros convert to 0
    if is_zero_f32(s_mod):
        return UNORM32_MIN

    # 3. Negative values (including -inf) convert to 0
    if bool(s_mod & SIGN_MASK_F32):
        return UNORM32_MIN

    # 4. Positive infinity converts to 4294967295
    if is_inf_f32(s_mod):
        return UNORM32_MAX

    # 5. Finite non-negative floating-point values
    f_val = bits_to_f32(s_mod)
    if f_val <= 0.0:
        return UNORM32_MIN
    if f_val >= 1.0:
        return UNORM32_MAX

    # Exact scaling: f_val * 4294967295 with roundTiesToEven
    scaled_frac = Fraction(f_val) * 4294967295
    unorm_int = round_unorm32_ties_to_even(scaled_frac)

    # Clamp to [0, 4294967295]
    if unorm_int < UNORM32_MIN:
        unorm_int = UNORM32_MIN
    elif unorm_int > UNORM32_MAX:
        unorm_int = UNORM32_MAX

    return unorm_int


class TestValuCvtNormU32F32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_zero_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(POS_ZERO_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ZERO_F32), UNORM32_MIN)

    def test_unit_and_boundary_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(ONE_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(HALF_F32), UNORM32_HALF)
        self.assertEqual(ref_cvt_norm_u32_f32(TWO_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(TEN_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(MAX_NORMAL_F32), UNORM32_MAX)

    def test_negative_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_HALF_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_TWO_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_TEN_F32), UNORM32_MIN)

    def test_infinity_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(POS_INF_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_INF_F32), UNORM32_MIN)

    def test_nan_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(QNAN_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_QNAN_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(SNAN_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_SNAN_F32), UNORM32_MIN)
        # NaN with custom payload
        custom_nan = 0x7FC00123
        self.assertEqual(ref_cvt_norm_u32_f32(custom_nan), UNORM32_MIN)

    def test_tie_breaking_nearest_even(self) -> None:
        # 0.5 * 4294967295 = 2147483647.5 -> rounds to 2147483648 (even)
        self.assertEqual(ref_cvt_norm_u32_f32(HALF_F32), UNORM32_HALF)
        self.assertEqual(ref_cvt_norm_u32_f32(HALF_F32) % 2, 0)

        # 0.25 * 4294967295 = 1073741823.75 -> rounds to 1073741824
        self.assertEqual(ref_cvt_norm_u32_f32(QUARTER_F32), UNORM32_QUARTER)

        # 0.75 * 4294967295 = 3221225471.25 -> rounds to 3221225471
        self.assertEqual(ref_cvt_norm_u32_f32(THREE_QUARTERS_F32), 3221225471)

    def test_input_modifiers(self) -> None:
        # abs clears negative sign
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32, abs_val=True), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_HALF_F32, abs_val=True), UNORM32_HALF)

        # neg inverts positive sign to negative -> 0
        self.assertEqual(ref_cvt_norm_u32_f32(ONE_F32, neg=True), UNORM32_MIN)

        # neg inverts negative sign to positive -> valid unorm
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32, neg=True), UNORM32_MAX)

        # neg(abs(x)) forces negative -> 0
        self.assertEqual(ref_cvt_norm_u32_f32(ONE_F32, abs_val=True, neg=True), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32, abs_val=True, neg=True), UNORM32_MIN)

    def test_subnormals(self) -> None:
        # Subnormals in f32 are all < 2^-126, which scale to < 2^-94, rounding to 0
        self.assertEqual(ref_cvt_norm_u32_f32(MIN_SUBNORMAL_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(MAX_SUBNORMAL_F32), UNORM32_MIN)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class Opcode(Enum):
    V_CVT_NORM_U32_F32 = "v_cvt_norm_u32_f32"


class Encoding(Enum):
    VOP1 = "vop1"
    VOP3 = "vop3"


class RegKind(Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclasses.dataclass(frozen=True)
class Register:
    """Typed architectural 32-bit register descriptor."""
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.index > 255:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")

    def name(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclasses.dataclass(frozen=True)
class Immediate:
    """Immediate floating-point or integer constant operand."""
    value: Union[int, float]
    is_f32: bool = True

    def bits32(self) -> int:
        if isinstance(self.value, float):
            return f32_to_bits(self.value)
        return int(self.value) & MASK32


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
        s0_str = self.src0.name() if isinstance(self.src0, Register) else f"{self.src0.bits32():#010x}"
        if self.abs_src0:
            s0_str = f"|{s0_str}|"
        if self.neg_src0:
            s0_str = f"-{s0_str}"
        clamp_str = " clamp" if self.clamp else ""
        return f"{self.opcode.value}{suffix} {self.dst.name()}, {s0_str}{clamp_str}"


class TestValuCvtNormU32F32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_and_encoding_enums(self) -> None:
        self.assertEqual(Opcode.V_CVT_NORM_U32_F32.value, "v_cvt_norm_u32_f32")
        self.assertEqual(Encoding.VOP1.value, "vop1")
        self.assertEqual(Encoding.VOP3.value, "vop3")

    def test_register_descriptors(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        s5 = Register(RegKind.SGPR, 5)

        self.assertEqual(v0.name(), "v0")
        self.assertEqual(s5.name(), "s5")

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, -1)

    def test_immediate_wrapping(self) -> None:
        imm_f = Immediate(1.0)
        self.assertEqual(imm_f.bits32(), ONE_F32)

        imm_raw = Immediate(0x3F000000)
        self.assertEqual(imm_raw.bits32(), HALF_F32)

    def test_instruction_construction_and_asm(self) -> None:
        # VOP1 default
        inst_vop1 = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            encoding=Encoding.VOP1,
        )
        self.assertEqual(inst_vop1.asm(), "v_cvt_norm_u32_f32_e32 v0, v1")

        # VOP3 with modifiers and clamp
        inst_vop3 = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            abs_src0=True,
            neg_src0=True,
            clamp=True,
            encoding=Encoding.VOP3,
        )
        self.assertEqual(inst_vop3.asm(), "v_cvt_norm_u32_f32_e64 v0, -|v1| clamp")

    def test_instruction_validation(self) -> None:
        # Reject SGPR as VALU destination
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_NORM_U32_F32,
                dst=Register(RegKind.SGPR, 0),
                src0=Register(RegKind.VGPR, 1),
            )

        # Reject modifiers on VOP1
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_NORM_U32_F32,
                dst=Register(RegKind.VGPR, 0),
                src0=Register(RegKind.VGPR, 1),
                abs_src0=True,
                encoding=Encoding.VOP1,
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

    def read_operand32(self, op: Union[Register, Immediate]) -> int:
        if isinstance(op, Immediate):
            return op.bits32()
        return self.read_reg32(op.kind, op.index)

    def step(self, inst: Instruction) -> int:
        if inst.opcode != Opcode.V_CVT_NORM_U32_F32:
            raise ValueError(f"Unsupported opcode in VALU interpreter: {inst.opcode}")

        src_bits = self.read_operand32(inst.src0)
        res_unorm32 = ref_cvt_norm_u32_f32(
            src_bits,
            clamp=inst.clamp,
            neg=inst.neg_src0,
            abs_val=inst.abs_src0,
        )
        self.write_reg32(inst.dst.kind, inst.dst.index, res_unorm32)
        return res_unorm32


class TestValuCvtNormU32F32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar VALU instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_basic_vgpr_to_vgpr_execution(self) -> None:
        # v1 = 1.0f (0x3F800000)
        self.interp.write_reg32(RegKind.VGPR, 1, ONE_F32)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, UNORM32_MAX)
        self.assertEqual(self.interp.read_reg32(RegKind.VGPR, 0), UNORM32_MAX)

    def test_sgpr_source_execution(self) -> None:
        # s2 = 0.5f (0x3F000000)
        self.interp.write_reg32(RegKind.SGPR, 2, HALF_F32)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.SGPR, 2),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, UNORM32_HALF)

    def test_immediate_source_execution(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0.25),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, UNORM32_QUARTER)

    def test_vop3_modifiers_execution(self) -> None:
        # v1 = -1.0f
        self.interp.write_reg32(RegKind.VGPR, 1, NEG_ONE_F32)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            abs_src0=True,
            encoding=Encoding.VOP3,
        )
        res = self.interp.step(inst)
        self.assertEqual(res, UNORM32_MAX)

    def test_vcc_unmodified_during_step(self) -> None:
        canary = 0xDEADBEEFCAFE1234
        self.interp.vcc = canary
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
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

    def read_lane_operand32(self, lane: int, op: Union[Register, Immediate]) -> int:
        if isinstance(op, Immediate):
            return op.bits32()
        return self.read_lane_reg32(lane, op.kind, op.index)

    def step(self, inst: Instruction) -> List[Optional[int]]:
        if inst.opcode != Opcode.V_CVT_NORM_U32_F32:
            raise ValueError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")

        results: List[Optional[int]] = [None] * self.wave_size
        for lane in range(self.wave_size):
            if not (self.exec_mask & (1 << lane)):
                continue

            src_bits = self.read_lane_operand32(lane, inst.src0)
            res_unorm32 = ref_cvt_norm_u32_f32(
                src_bits,
                clamp=inst.clamp,
                neg=inst.neg_src0,
                abs_val=inst.abs_src0,
            )
            self.write_lane_reg32(lane, inst.dst.kind, inst.dst.index, res_unorm32)
            results[lane] = res_unorm32

        return results


class TestValuCvtNormU32F32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F32)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
        )
        results = sim.step(inst)
        self.assertEqual(len(results), 32)
        self.assertTrue(all(r == UNORM32_MAX for r in results))

    def test_wave32_divergent_lanes(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        test_values = [
            0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.75, 0.8, 0.9, 1.0, 1.5, 2.0, 10.0,
            -0.0, -0.1, -0.5, -1.0, -2.0, 0.001, 0.0001, 0.999,
            float("inf"), float("-inf"), float("nan"), 0.5,
            0.25, 0.75, 1.0, 0.0
        ]
        for lane, fval in enumerate(test_values):
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, f32_to_bits(fval))

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
        )
        results = sim.step(inst)

        for lane, fval in enumerate(test_values):
            expected = ref_cvt_norm_u32_f32(f32_to_bits(fval))
            self.assertEqual(results[lane], expected)
            lane_dst = sim.read_lane_reg32(lane, RegKind.VGPR, 0)
            self.assertEqual(lane_dst, expected)

    def test_wave32_exec_mask_checkerboard(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xAAAA5555
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, canary)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, HALF_F32)

        sim.set_exec_mask(0x55555555)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
        )
        results = sim.step(inst)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], UNORM32_HALF)
                self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), UNORM32_HALF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), canary)

    def test_wave32_all_inactive(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xCAFEBABE
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, canary)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F32)

        sim.set_exec_mask(0x00000000)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
        )
        results = sim.step(inst)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), canary)

    def test_wave64_execution_and_exec_halves(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=64)
        for lane in range(64):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, 0x11112222)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F32)

        sim.set_exec_mask(0xFFFFFFFF00000000)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
        )
        results = sim.step(inst)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), 0x11112222)

        for lane in range(32, 64):
            self.assertEqual(results[lane], UNORM32_MAX)
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), UNORM32_MAX)

    def test_wavefront_shared_sgpr_source(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        sim.write_lane_reg32(0, RegKind.SGPR, 4, HALF_F32)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U32_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.SGPR, 4),
        )
        results = sim.step(inst)
        self.assertTrue(all(r == UNORM32_HALF for r in results))


if __name__ == "__main__":
    unittest.main()
