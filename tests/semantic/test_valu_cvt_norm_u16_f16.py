"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_u16_f16.

Models IEEE 754-2008 / IEEE 754-2019 binary16 (half-precision float) to 16-bit unsigned normalized integer
(UNORM16) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_cvt_norm_u16_f16: D0.u16 = f16_to_unorm(S0.f16)

RDNA2 Floating-Point Conversion ISA Rules:
  1. Register Widths and Destination Semantics:
     - v_cvt_norm_u16_f16 operates on 16-bit half-precision floating-point source in S0[15:0].
     - Writes an unsigned normalized 16-bit integer (UNORM16) into D0[15:0].
     - On AMD RDNA2 / GFX10, 16-bit destination VGPR operations preserve the high 16 bits D0[31:16].
  2. Normalized Integer Conversion & Rounding:
     - Scales normalized range [0.0, 1.0] across full 16-bit integer range [0, 65535].
     - Rounding mode: IEEE 754 round-to-nearest-even (RNE) is applied to the scaled value (val * 65535.0).
     - Midpoint ties (e.g. 0.5 * 65535.0 = 32767.5) round to the nearest even integer (32768 = 0x8000).
  3. Boundary Extremes & Special Values:
     - Values <= 0.0 (including +0.0, -0.0, negative finite values, and -infinity) saturate/clamp to 0.
     - Values >= 1.0 (including positive numbers > 1.0 and +infinity) saturate/clamp to 65535 (0xFFFF).
     - Subnormal half-precision inputs round to 0 or appropriate integer without faulting.
  4. NaN Handling:
     - Any NaN input (quiet NaN or signaling NaN, positive or negative) converts to 0.
  5. VOP1 / VOP3 Encoding and Modifiers:
     - VOP1 (e32): Compact encoding with single source VGPR or SGPR or inline constant.
     - VOP3 (e64): Extended encoding supporting input modifiers:
       * abs (|x|): Clears sign bit before conversion.
       * neg (-x): Inverts sign bit before conversion.
       * neg(abs(x)): Inverts sign after abs (forcing negative, which subsequently yields 0).
       * clamp: Clamps converted result. For normalized conversion, output is already within [0, 65535].

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
THREE_QUARTERS_F16: int = 0x3A00  # 0.75
TWO_F16: int = 0x4000          # 2.0
NEG_TWO_F16: int = 0xC000      # -2.0
TEN_F16: int = 0x4900          # 10.0
NEG_TEN_F16: int = 0xC900      # -10.0

MAX_NORMAL_F16: int = 0x7BFF   # 65504.0
MIN_NORMAL_F16: int = 0x0400   # 2^-14 ~= 6.1035e-5
MAX_SUBNORMAL_F16: int = 0x03FF # (1023/1024) * 2^-14
MIN_SUBNORMAL_F16: int = 0x0001 # 2^-24 ~= 5.9605e-8

# UNORM16 Constants
UNORM16_MIN: int = 0x0000
UNORM16_MAX: int = 0xFFFF
UNORM16_HALF: int = 0x8000


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


def round_unorm16_ties_to_even(frac_val: Fraction) -> int:
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


def ref_cvt_norm_u16_f16(
    src16: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_norm_u16_f16.

    Converts 16-bit half-precision floating-point value to 16-bit unsigned normalized integer.
    """
    s_mod = apply_f16_modifiers(src16, abs_val=abs_val, neg=neg)

    # 1. NaN inputs convert to 0
    if is_nan_f16(s_mod):
        return UNORM16_MIN

    # 2. Signed zeros convert to 0
    if is_zero_f16(s_mod):
        return UNORM16_MIN

    # 3. Negative values (including -inf) convert to 0
    if bool(s_mod & SIGN_MASK_F16):
        return UNORM16_MIN

    # 4. Positive infinity converts to 65535
    if is_inf_f16(s_mod):
        return UNORM16_MAX

    # 5. Finite non-negative floating-point values
    f_val = bits_to_f16(s_mod)
    if f_val <= 0.0:
        return UNORM16_MIN
    if f_val >= 1.0:
        return UNORM16_MAX

    # Exact scaling: f_val * 65535 with roundTiesToEven
    scaled_frac = Fraction(f_val) * 65535
    unorm_int = round_unorm16_ties_to_even(scaled_frac)

    # Clamp to [0, 65535]
    if unorm_int < UNORM16_MIN:
        unorm_int = UNORM16_MIN
    elif unorm_int > UNORM16_MAX:
        unorm_int = UNORM16_MAX

    return unorm_int


class TestValuCvtNormU16F16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_zero_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(POS_ZERO_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ZERO_F16), UNORM16_MIN)

    def test_unit_and_boundary_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(ONE_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(HALF_F16), UNORM16_HALF)
        self.assertEqual(ref_cvt_norm_u16_f16(TWO_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(TEN_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(MAX_NORMAL_F16), UNORM16_MAX)

    def test_negative_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_HALF_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_TWO_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_TEN_F16), UNORM16_MIN)

    def test_infinity_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(POS_INF_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_INF_F16), UNORM16_MIN)

    def test_nan_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(QNAN_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_QNAN_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(SNAN_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_SNAN_F16), UNORM16_MIN)
        # NaN with custom payload
        custom_nan = 0x7E42
        self.assertEqual(ref_cvt_norm_u16_f16(custom_nan), UNORM16_MIN)

    def test_tie_breaking_nearest_even(self) -> None:
        # 0.5 * 65535 = 32767.5 -> rounds to 32768 (even)
        self.assertEqual(ref_cvt_norm_u16_f16(HALF_F16), 32768)
        self.assertEqual(ref_cvt_norm_u16_f16(HALF_F16) % 2, 0)

        # 0.25 * 65535 = 16383.75 -> rounds to 16384
        self.assertEqual(ref_cvt_norm_u16_f16(QUARTER_F16), 16384)

        # 0.75 * 65535 = 49151.25 -> rounds to 49151
        self.assertEqual(ref_cvt_norm_u16_f16(THREE_QUARTERS_F16), 49151)

    def test_input_modifiers(self) -> None:
        # abs clears negative sign
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16, abs_val=True), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_HALF_F16, abs_val=True), UNORM16_HALF)

        # neg inverts positive sign to negative -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(ONE_F16, neg=True), UNORM16_MIN)

        # neg inverts negative sign to positive -> valid unorm
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16, neg=True), UNORM16_MAX)

        # neg(abs(x)) forces negative -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(ONE_F16, abs_val=True, neg=True), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16, abs_val=True, neg=True), UNORM16_MIN)

    def test_subnormals(self) -> None:
        # Smallest subnormal 2^-24 * 65535 ~= 0.0039 rounds to 0
        self.assertEqual(ref_cvt_norm_u16_f16(MIN_SUBNORMAL_F16), UNORM16_MIN)
        # Largest subnormal ~6.0975e-5 * 65535 ~= 3.996 rounds to 4
        self.assertEqual(ref_cvt_norm_u16_f16(MAX_SUBNORMAL_F16), 4)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class Opcode(Enum):
    V_CVT_NORM_U16_F16 = "v_cvt_norm_u16_f16"


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


class TestValuCvtNormU16F16AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_and_encoding_enums(self) -> None:
        self.assertEqual(Opcode.V_CVT_NORM_U16_F16.value, "v_cvt_norm_u16_f16")
        self.assertEqual(Encoding.VOP1.value, "vop1")
        self.assertEqual(Encoding.VOP3.value, "vop3")

    def test_register_descriptors(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v0_l = Register(RegKind.VGPR, 0, SubWord.LO)
        v0_h = Register(RegKind.VGPR, 0, SubWord.HI)
        s5 = Register(RegKind.SGPR, 5)

        self.assertEqual(v0.name(), "v0")
        self.assertEqual(v0_l.name(), "v0.l")
        self.assertEqual(v0_h.name(), "v0.h")
        self.assertEqual(s5.name(), "s5")

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, -1)

    def test_immediate_wrapping(self) -> None:
        imm_f = Immediate(1.0)
        self.assertEqual(imm_f.bits16(), ONE_F16)

        imm_raw = Immediate(0x3800)
        self.assertEqual(imm_raw.bits16(), HALF_F16)

    def test_instruction_construction_and_asm(self) -> None:
        # VOP1 default
        inst_vop1 = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
            encoding=Encoding.VOP1,
        )
        self.assertEqual(inst_vop1.asm(), "v_cvt_norm_u16_f16_e32 v0.l, v1.l")

        # VOP3 with modifiers and clamp
        inst_vop3 = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
            abs_src0=True,
            neg_src0=True,
            clamp=True,
            encoding=Encoding.VOP3,
        )
        self.assertEqual(inst_vop3.asm(), "v_cvt_norm_u16_f16_e64 v0.l, -|v1.l| clamp")

    def test_instruction_validation(self) -> None:
        # Reject SGPR as VALU destination
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_NORM_U16_F16,
                dst=Register(RegKind.SGPR, 0),
                src0=Register(RegKind.VGPR, 1),
            )

        # Reject modifiers on VOP1
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_NORM_U16_F16,
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

    def read_reg16(self, reg: Register) -> int:
        raw32 = self.read_reg32(reg.kind, reg.index)
        if reg.subword == SubWord.HI:
            return (raw32 >> 16) & MASK16
        return raw32 & MASK16

    def write_reg16(self, reg: Register, val16: int) -> None:
        if reg.kind != RegKind.VGPR:
            raise ValueError("VALU destination register must be a VGPR")
        val16 &= MASK16
        old32 = self.read_reg32(reg.kind, reg.index)
        if reg.subword == SubWord.HI:
            new32 = (old32 & 0x0000FFFF) | (val16 << 16)
        else:
            # SubWord.LO or SubWord.FULL writes lower 16 bits and preserves D[31:16]
            new32 = (old32 & 0xFFFF0000) | val16
        self.write_reg32(reg.kind, reg.index, new32)

    def read_operand16(self, op: Union[Register, Immediate]) -> int:
        if isinstance(op, Immediate):
            return op.bits16()
        return self.read_reg16(op)

    def step(self, inst: Instruction) -> int:
        if inst.opcode != Opcode.V_CVT_NORM_U16_F16:
            raise ValueError(f"Unsupported opcode in VALU interpreter: {inst.opcode}")

        src_bits = self.read_operand16(inst.src0)
        res_unorm16 = ref_cvt_norm_u16_f16(
            src_bits,
            clamp=inst.clamp,
            neg=inst.neg_src0,
            abs_val=inst.abs_src0,
        )
        self.write_reg16(inst.dst, res_unorm16)
        return res_unorm16


class TestValuCvtNormU16F16ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar VALU instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_basic_vgpr_to_vgpr_execution(self) -> None:
        # v1 = 1.0 (0x3C00) in lower half
        self.interp.write_reg32(RegKind.VGPR, 1, ONE_F16)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, UNORM16_MAX)
        self.assertEqual(self.interp.read_reg16(Register(RegKind.VGPR, 0, SubWord.LO)), UNORM16_MAX)

    def test_sgpr_source_execution(self) -> None:
        # s2 = 0.5 (0x3800)
        self.interp.write_reg32(RegKind.SGPR, 2, HALF_F16)
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.SGPR, 2, SubWord.LO),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, UNORM16_HALF)

    def test_immediate_source_execution(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Immediate(0.25),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 16384)

    def test_destination_high16_preservation(self) -> None:
        # Preset v0 to 0x12345678. Writing v0.l must preserve 0x1234 in upper 16 bits.
        self.interp.write_reg32(RegKind.VGPR, 0, 0x12345678)
        self.interp.write_reg32(RegKind.VGPR, 1, ONE_F16)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        self.interp.step(inst)

        full_dst = self.interp.read_reg32(RegKind.VGPR, 0)
        self.assertEqual(full_dst >> 16, 0x1234)
        self.assertEqual(full_dst & 0xFFFF, UNORM16_MAX)

    def test_destination_high16_write_with_low16_preservation(self) -> None:
        # Preset v0 to 0x12345678. Writing v0.h must preserve 0x5678 in lower 16 bits.
        self.interp.write_reg32(RegKind.VGPR, 0, 0x12345678)
        self.interp.write_reg32(RegKind.VGPR, 1, HALF_F16)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.HI),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
            encoding=Encoding.VOP3,
        )
        self.interp.step(inst)

        full_dst = self.interp.read_reg32(RegKind.VGPR, 0)
        self.assertEqual(full_dst & 0xFFFF, 0x5678)
        self.assertEqual(full_dst >> 16, UNORM16_HALF)

    def test_source_high16_selection(self) -> None:
        # Pack two f16 in v1: lo=0.25 (0x3400), hi=0.5 (0x3800)
        packed = (HALF_F16 << 16) | QUARTER_F16
        self.interp.write_reg32(RegKind.VGPR, 1, packed)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.HI),
            encoding=Encoding.VOP3,
        )
        res = self.interp.step(inst)
        self.assertEqual(res, UNORM16_HALF)

    def test_vcc_unmodified_during_step(self) -> None:
        canary = 0xDEADBEEFCAFE1234
        self.interp.vcc = canary
        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
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

    def write_lane_reg16(self, lane: int, reg: Register, val16: int) -> None:
        if reg.kind != RegKind.VGPR:
            raise ValueError("VALU destination register must be a VGPR")
        val16 &= MASK16
        old32 = self.read_lane_reg32(lane, reg.kind, reg.index)
        if reg.subword == SubWord.HI:
            new32 = (old32 & 0x0000FFFF) | (val16 << 16)
        else:
            new32 = (old32 & 0xFFFF0000) | val16
        self.write_lane_reg32(lane, reg.kind, reg.index, new32)

    def read_lane_operand16(self, lane: int, op: Union[Register, Immediate]) -> int:
        if isinstance(op, Immediate):
            return op.bits16()
        return self.read_lane_reg16(lane, op)

    def step(self, inst: Instruction) -> List[Optional[int]]:
        if inst.opcode != Opcode.V_CVT_NORM_U16_F16:
            raise ValueError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")

        results: List[Optional[int]] = [None] * self.wave_size
        for lane in range(self.wave_size):
            if not (self.exec_mask & (1 << lane)):
                continue

            src_bits = self.read_lane_operand16(lane, inst.src0)
            res_unorm16 = ref_cvt_norm_u16_f16(
                src_bits,
                clamp=inst.clamp,
                neg=inst.neg_src0,
                abs_val=inst.abs_src0,
            )
            self.write_lane_reg16(lane, inst.dst, res_unorm16)
            results[lane] = res_unorm16

        return results


class TestValuCvtNormU16F16SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F16)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)
        self.assertEqual(len(results), 32)
        self.assertTrue(all(r == UNORM16_MAX for r in results))

    def test_wave32_divergent_lanes(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        # Setup divergent values across 32 lanes
        test_values = [
            0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
            0.7, 0.75, 0.8, 0.9, 1.0, 1.5, 2.0, 10.0,
            -0.0, -0.1, -0.5, -1.0, -2.0, 0.001, 0.0001, 0.999,
            float("inf"), float("-inf"), float("nan"), 0.5,
            0.25, 0.75, 1.0, 0.0
        ]
        for lane, fval in enumerate(test_values):
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, f16_to_bits(fval))

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)

        for lane, fval in enumerate(test_values):
            expected = ref_cvt_norm_u16_f16(f16_to_bits(fval))
            self.assertEqual(results[lane], expected)
            lane_dst = sim.read_lane_reg16(lane, Register(RegKind.VGPR, 0, SubWord.LO))
            self.assertEqual(lane_dst, expected)

    def test_wave32_exec_mask_checkerboard(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xAAAA5555
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, canary)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, HALF_F16)

        # Enable even lanes only
        sim.set_exec_mask(0x55555555)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], UNORM16_HALF)
                self.assertEqual(sim.read_lane_reg16(lane, Register(RegKind.VGPR, 0, SubWord.LO)), UNORM16_HALF)
                # Upper 16 bits preserved
                self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0) >> 16, 0xAAAA)
            else:
                self.assertIsNone(results[lane])
                # Completely untouched
                self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), canary)

    def test_wave32_all_inactive(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xCAFEBABE
        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, canary)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F16)

        sim.set_exec_mask(0x00000000)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), canary)

    def test_wave64_execution_and_exec_halves(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=64)
        for lane in range(64):
            sim.write_lane_reg32(lane, RegKind.VGPR, 0, 0x11112222)
            sim.write_lane_reg32(lane, RegKind.VGPR, 1, ONE_F16)

        # Upper 32 lanes active, lower 32 lanes inactive
        sim.set_exec_mask(0xFFFFFFFF00000000)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        results = sim.step(inst)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0), 0x11112222)

        for lane in range(32, 64):
            self.assertEqual(results[lane], UNORM16_MAX)
            self.assertEqual(sim.read_lane_reg16(lane, Register(RegKind.VGPR, 0, SubWord.LO)), UNORM16_MAX)
            self.assertEqual(sim.read_lane_reg32(lane, RegKind.VGPR, 0) >> 16, 0x1111)

    def test_wavefront_shared_sgpr_source(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        # Shared SGPR scalar value broadcast across all lanes
        sim.write_lane_reg32(0, RegKind.SGPR, 4, HALF_F16)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.SGPR, 4, SubWord.LO),
        )
        results = sim.step(inst)
        self.assertTrue(all(r == UNORM16_HALF for r in results))


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestValuCvtNormU16F16RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def test_inplace_low_half_aliasing(self) -> None:
        # v0.l = v_cvt_norm_u16_f16 v0.l
        interp = ScalarVALUInterpreter()
        canary_hi = 0x89AB
        initial_v0 = (canary_hi << 16) | HALF_F16
        interp.write_reg32(RegKind.VGPR, 0, initial_v0)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 0, SubWord.LO),
        )
        res = interp.step(inst)
        self.assertEqual(res, UNORM16_HALF)

        final_v0 = interp.read_reg32(RegKind.VGPR, 0)
        self.assertEqual(final_v0 >> 16, canary_hi)
        self.assertEqual(final_v0 & 0xFFFF, UNORM16_HALF)

    def test_inplace_high_half_aliasing(self) -> None:
        # v0.h = v_cvt_norm_u16_f16 v0.h
        interp = ScalarVALUInterpreter()
        canary_lo = 0xCDEF
        initial_v0 = (ONE_F16 << 16) | canary_lo
        interp.write_reg32(RegKind.VGPR, 0, initial_v0)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.HI),
            src0=Register(RegKind.VGPR, 0, SubWord.HI),
            encoding=Encoding.VOP3,
        )
        res = interp.step(inst)
        self.assertEqual(res, UNORM16_MAX)

        final_v0 = interp.read_reg32(RegKind.VGPR, 0)
        self.assertEqual(final_v0 & 0xFFFF, canary_lo)
        self.assertEqual(final_v0 >> 16, UNORM16_MAX)

    def test_cross_half_aliasing_hi_to_lo(self) -> None:
        # v0.l = v_cvt_norm_u16_f16 v0.h
        interp = ScalarVALUInterpreter()
        initial_v0 = (QUARTER_F16 << 16) | 0x1111
        interp.write_reg32(RegKind.VGPR, 0, initial_v0)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 0, SubWord.HI),
            encoding=Encoding.VOP3,
        )
        res = interp.step(inst)
        self.assertEqual(res, 16384)

        final_v0 = interp.read_reg32(RegKind.VGPR, 0)
        # High half must remain identical to source QUARTER_F16
        self.assertEqual(final_v0 >> 16, QUARTER_F16)
        self.assertEqual(final_v0 & 0xFFFF, 16384)

    def test_cross_half_aliasing_lo_to_hi(self) -> None:
        # v0.h = v_cvt_norm_u16_f16 v0.l
        interp = ScalarVALUInterpreter()
        initial_v0 = (0x9999 << 16) | THREE_QUARTERS_F16
        interp.write_reg32(RegKind.VGPR, 0, initial_v0)

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.HI),
            src0=Register(RegKind.VGPR, 0, SubWord.LO),
            encoding=Encoding.VOP3,
        )
        res = interp.step(inst)
        self.assertEqual(res, 49151)

        final_v0 = interp.read_reg32(RegKind.VGPR, 0)
        # Low half must remain identical to source THREE_QUARTERS_F16
        self.assertEqual(final_v0 & 0xFFFF, THREE_QUARTERS_F16)
        self.assertEqual(final_v0 >> 16, 49151)

    def test_sequential_aliasing_pipeline(self) -> None:
        # Step 1: v0.l = v_cvt_norm_u16_f16 v1.l (v1.l = 0.5) -> v0.l = 0x8000
        # Step 2: v1.l = v_cvt_norm_u16_f16 v1.h (v1.h = 1.0) -> v1.l = 0xFFFF
        # Step 3: v0.h = v_cvt_norm_u16_f16 v0.l (v0.l bits reinterpreted as f16 0x8000 = -0.0) -> v0.h = 0
        interp = ScalarVALUInterpreter()
        interp.write_reg32(RegKind.VGPR, 1, (ONE_F16 << 16) | HALF_F16)
        interp.write_reg32(RegKind.VGPR, 0, 0x12345678)

        inst1 = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.LO),
        )
        inst2 = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 1, SubWord.LO),
            src0=Register(RegKind.VGPR, 1, SubWord.HI),
            encoding=Encoding.VOP3,
        )
        inst3 = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.HI),
            src0=Register(RegKind.VGPR, 0, SubWord.LO),
            encoding=Encoding.VOP3,
        )

        r1 = interp.step(inst1)
        self.assertEqual(r1, UNORM16_HALF)
        self.assertEqual(interp.read_reg32(RegKind.VGPR, 0) >> 16, 0x1234)

        r2 = interp.step(inst2)
        self.assertEqual(r2, UNORM16_MAX)
        self.assertEqual(interp.read_reg32(RegKind.VGPR, 1) >> 16, ONE_F16)

        r3 = interp.step(inst3)
        # 0x8000 is -0.0 in f16, which converts to UNORM16_MIN (0)
        self.assertEqual(r3, UNORM16_MIN)
        self.assertEqual(interp.read_reg32(RegKind.VGPR, 0) & 0xFFFF, UNORM16_HALF)
        self.assertEqual(interp.read_reg32(RegKind.VGPR, 0) >> 16, UNORM16_MIN)

    def test_simd_multi_lane_inplace_aliasing(self) -> None:
        sim = SIMDWavefrontInterpreter(wave_size=32)
        canaries = [(0x1000 + i) for i in range(32)]
        inputs = [
            (HALF_F16 if (i % 2 == 0) else ONE_F16)
            for i in range(32)
        ]

        for lane in range(32):
            sim.write_lane_reg32(lane, RegKind.VGPR, 2, (canaries[lane] << 16) | inputs[lane])

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 2, SubWord.LO),
            src0=Register(RegKind.VGPR, 2, SubWord.LO),
        )
        results = sim.step(inst)

        for lane in range(32):
            expected = UNORM16_HALF if (lane % 2 == 0) else UNORM16_MAX
            self.assertEqual(results[lane], expected)
            final_word = sim.read_lane_reg32(lane, RegKind.VGPR, 2)
            self.assertEqual(final_word >> 16, canaries[lane])
            self.assertEqual(final_word & 0xFFFF, expected)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConditionCodes:
    """Architectural condition codes evaluating properties of converted UNORM16 results."""
    zero: bool
    max_val: bool
    even: bool
    negative: bool
    midpoint: bool


@dataclasses.dataclass(frozen=True)
class FPStatusFlags:
    """IEEE 754 and hardware status flags for f16 to unorm16 conversion."""
    invalid: bool       # e.g., sNaN operand
    denormal: bool      # e.g., subnormal f16 operand
    inexact: bool       # non-integer scaling requiring rounding
    overflow: bool      # finite input > 1.0 clamped to 65535
    underflow: bool     # positive finite input rounded to 0


def compute_condition_codes_u16(result: int) -> ConditionCodes:
    """Computes condition codes evaluating attributes of a 16-bit UNORM result."""
    u16 = result & MASK16
    return ConditionCodes(
        zero=(u16 == UNORM16_MIN),
        max_val=(u16 == UNORM16_MAX),
        even=(u16 % 2 == 0),
        negative=False,  # Unsigned values cannot be negative
        midpoint=(u16 == UNORM16_HALF),
    )


def compute_fp_status_flags_cvt_norm_u16_f16(src16: int, result16: int) -> FPStatusFlags:
    """Computes IEEE 754 / architectural exception status flags."""
    src16 &= MASK16
    result16 &= MASK16

    invalid = is_snan_f16(src16)
    denormal = is_subnormal_f16(src16)

    is_src_nan = is_nan_f16(src16)
    is_src_inf = is_inf_f16(src16)

    overflow = False
    underflow = False
    inexact = False

    if not is_src_nan and not is_src_inf:
        f_val = bits_to_f16(src16)
        if f_val > 1.0:
            overflow = True
            inexact = True
        elif 0.0 < f_val < 1.0:
            scaled = Fraction(f_val) * 65535
            if scaled.denominator != 1:
                inexact = True
            if result16 == UNORM16_MIN:
                underflow = True

    return FPStatusFlags(
        invalid=invalid,
        denormal=denormal,
        inexact=inexact,
        overflow=overflow,
        underflow=underflow,
    )


class TestValuCvtNormU16F16ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def test_condition_codes(self) -> None:
        # Zero result
        cc_zero = compute_condition_codes_u16(UNORM16_MIN)
        self.assertTrue(cc_zero.zero)
        self.assertTrue(cc_zero.even)
        self.assertFalse(cc_zero.max_val)
        self.assertFalse(cc_zero.negative)
        self.assertFalse(cc_zero.midpoint)

        # Max result
        cc_max = compute_condition_codes_u16(UNORM16_MAX)
        self.assertFalse(cc_max.zero)
        self.assertFalse(cc_max.even)
        self.assertTrue(cc_max.max_val)
        self.assertFalse(cc_max.negative)

        # Midpoint result (0x8000 = 32768)
        cc_mid = compute_condition_codes_u16(UNORM16_HALF)
        self.assertFalse(cc_mid.zero)
        self.assertTrue(cc_mid.even)
        self.assertTrue(cc_mid.midpoint)

    def test_status_flags_snan_invalid(self) -> None:
        res = ref_cvt_norm_u16_f16(SNAN_F16)
        flags = compute_fp_status_flags_cvt_norm_u16_f16(SNAN_F16, res)
        self.assertTrue(flags.invalid)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.overflow)

        # Quiet NaN should not set invalid flag
        res_qnan = ref_cvt_norm_u16_f16(QNAN_F16)
        flags_qnan = compute_fp_status_flags_cvt_norm_u16_f16(QNAN_F16, res_qnan)
        self.assertFalse(flags_qnan.invalid)

    def test_status_flags_denormal(self) -> None:
        res = ref_cvt_norm_u16_f16(MIN_SUBNORMAL_F16)
        flags = compute_fp_status_flags_cvt_norm_u16_f16(MIN_SUBNORMAL_F16, res)
        self.assertTrue(flags.denormal)
        self.assertFalse(flags.invalid)
        self.assertTrue(flags.underflow)

    def test_status_flags_overflow(self) -> None:
        # Input 2.0 exceeds 1.0
        res = ref_cvt_norm_u16_f16(TWO_F16)
        self.assertEqual(res, UNORM16_MAX)
        flags = compute_fp_status_flags_cvt_norm_u16_f16(TWO_F16, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_status_flags_inexact(self) -> None:
        # 1.0 * 65535 is exact integer 65535 (no inexact)
        res_one = ref_cvt_norm_u16_f16(ONE_F16)
        flags_one = compute_fp_status_flags_cvt_norm_u16_f16(ONE_F16, res_one)
        self.assertFalse(flags_one.inexact)

        # 0.5 * 65535 = 32767.5 is not an integer (inexact)
        res_half = ref_cvt_norm_u16_f16(HALF_F16)
        flags_half = compute_fp_status_flags_cvt_norm_u16_f16(HALF_F16, res_half)
        self.assertTrue(flags_half.inexact)

    def test_carry_borrow_vcc_isolation(self) -> None:
        # Floating point conversion instructions in RDNA2 never generate carry/borrow or alter VCC
        interp = ScalarVALUInterpreter()
        canary_vcc = 0xFEDCBA9876543210
        interp.vcc = canary_vcc

        inst = Instruction(
            opcode=Opcode.V_CVT_NORM_U16_F16,
            dst=Register(RegKind.VGPR, 0, SubWord.LO),
            src0=Immediate(1.0),
        )
        interp.step(inst)
        self.assertEqual(interp.vcc, canary_vcc)

        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.vcc = canary_vcc
        simd.step(inst)
        self.assertEqual(simd.vcc, canary_vcc)


# ---------------------------------------------------------------------------
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ---------------------------------------------------------------------------

class TestValuCvtNormU16F16IEEE754Boundaries(unittest.TestCase):
    """Exhaustive unit tests for IEEE 754 boundary extremes, subnormals, and NaNs."""

    def test_signed_zeros_and_modifiers(self) -> None:
        # +0.0 and -0.0 always produce UNORM16_MIN (0)
        self.assertEqual(ref_cvt_norm_u16_f16(POS_ZERO_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ZERO_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(POS_ZERO_F16, abs_val=True), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ZERO_F16, abs_val=True), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(POS_ZERO_F16, neg=True), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ZERO_F16, neg=True), UNORM16_MIN)

    def test_subnormal_scale_boundaries(self) -> None:
        # 0x0001 = 2^-24 -> scales to ~0.0039 -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(0x0001), 0)
        # 0x0002 = 2^-23 -> scales to ~0.0078 -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(0x0002), 0)
        # 0x0100 = 2^-16 -> scales to ~0.99998 -> 1
        self.assertEqual(ref_cvt_norm_u16_f16(0x0100), 1)
        # 0x0200 = 2^-15 -> scales to ~1.99996 -> 2
        self.assertEqual(ref_cvt_norm_u16_f16(0x0200), 2)
        # 0x03FF = max subnormal -> scales to ~3.996 -> 4
        self.assertEqual(ref_cvt_norm_u16_f16(MAX_SUBNORMAL_F16), 4)

    def test_negative_subnormals_saturation(self) -> None:
        # Negative subnormals without abs modifier clamp to 0
        neg_subnormals = [0x8001, 0x8002, 0x8100, 0x8200, 0x83FF]
        for ns in neg_subnormals:
            self.assertEqual(ref_cvt_norm_u16_f16(ns), UNORM16_MIN)

        # With abs modifier, negative subnormals mirror positive subnormals
        self.assertEqual(ref_cvt_norm_u16_f16(0x8001, abs_val=True), 0)
        self.assertEqual(ref_cvt_norm_u16_f16(0x8100, abs_val=True), 1)
        self.assertEqual(ref_cvt_norm_u16_f16(0x8200, abs_val=True), 2)
        self.assertEqual(ref_cvt_norm_u16_f16(0x83FF, abs_val=True), 4)

    def test_small_normals_boundary(self) -> None:
        # Smallest normal f16: 0x0400 = 2^-14 -> scales to ~3.9999 -> 4
        self.assertEqual(ref_cvt_norm_u16_f16(MIN_NORMAL_F16), 4)
        # 0x0800 = 2^-13 -> scales to ~7.9998 -> 8
        self.assertEqual(ref_cvt_norm_u16_f16(0x0800), 8)
        # 0x1000 = 2^-11 -> scales to ~31.9995 -> 32
        self.assertEqual(ref_cvt_norm_u16_f16(0x1000), 32)

    def test_unit_boundary_and_saturation(self) -> None:
        # Exact 1.0 (0x3C00) -> 65535
        self.assertEqual(ref_cvt_norm_u16_f16(ONE_F16), UNORM16_MAX)

        # 1.0 - epsilon: 0x3BFF (0.99951171875) -> 65503
        self.assertEqual(ref_cvt_norm_u16_f16(0x3BFF), 65503)

        # 1.0 + epsilon: 0x3C01 (1.0009765625) -> saturates to 65535
        self.assertEqual(ref_cvt_norm_u16_f16(0x3C01), UNORM16_MAX)

    def test_infinities_and_saturation(self) -> None:
        # +infinity -> 65535
        self.assertEqual(ref_cvt_norm_u16_f16(POS_INF_F16), UNORM16_MAX)
        # -infinity -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_INF_F16), UNORM16_MIN)

        # abs(-inf) -> +inf -> 65535
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_INF_F16, abs_val=True), UNORM16_MAX)
        # neg(+inf) -> -inf -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(POS_INF_F16, neg=True), UNORM16_MIN)
        # neg(-inf) -> +inf -> 65535
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_INF_F16, neg=True), UNORM16_MAX)

    def test_nan_exhaustive_handling(self) -> None:
        # All NaN representations must convert to 0 regardless of payload or sign
        nan_samples = [
            QNAN_F16,
            NEG_QNAN_F16,
            SNAN_F16,
            NEG_SNAN_F16,
            0x7C02,  # sNaN
            0x7C03,  # sNaN
            0x7E01,  # qNaN with payload
            0x7EFF,  # qNaN with payload
            0x7FFF,  # all-ones exponent and mantissa
            0xFFFF,  # negative all-ones
        ]
        for nan_val in nan_samples:
            self.assertEqual(ref_cvt_norm_u16_f16(nan_val), UNORM16_MIN)
            self.assertEqual(ref_cvt_norm_u16_f16(nan_val, abs_val=True), UNORM16_MIN)
            self.assertEqual(ref_cvt_norm_u16_f16(nan_val, neg=True), UNORM16_MIN)
            self.assertEqual(ref_cvt_norm_u16_f16(nan_val, clamp=True), UNORM16_MIN)

    def test_round_ties_to_even_halfway(self) -> None:
        # Halfway case: 0.5 * 65535 = 32767.5 -> 32768 (even)
        res_half = ref_cvt_norm_u16_f16(HALF_F16)
        self.assertEqual(res_half, 32768)
        self.assertEqual(res_half % 2, 0)


if __name__ == "__main__":
    unittest.main()
