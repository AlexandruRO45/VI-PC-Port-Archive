from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pknorm_i16_f64.

Models IEEE 754-2008 binary64 double-precision floating-point to signed normalized 16-bit
integer (SNORM16) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU)
architectural semantics:
  v_cvt_pknorm_i16_f64 D, S0, S1

Lowered Architectural Semantics:
  D[15:0]  = f64_to_snorm16(S0)
  D[31:16] = f64_to_snorm16(S1)

RDNA2 VALU Specification Rules:
  1. Register Layout:
     - Destination D is a single 32-bit VGPR (v[d]).
     - Sources S0 and S1 are 64-bit double-precision floats (binary64).
     - Each 64-bit source occupies an aligned pair of VGPRs (v[s:s+1]), SGPRs (s[s:s+1]),
       or a 64-bit literal constant.
     - D[15:0] is derived exclusively from S0; D[31:16] is derived exclusively from S1.
  2. Fixed-Point Signed Normalization (SNORM16):
     - Scale factor is 32767.0 (2^15 - 1 = 0x7FFF).
     - Input values outside [-1.0, +1.0] are clamped to [-1.0, +1.0].
     - Product (clamped * 32767.0) is rounded using IEEE 754 roundTiesToEven.
     - Representable integer range: [-32767, +32767].
     - Mapping:
         +1.0 -> +32767 (0x7FFF)
         -1.0 -> -32767 (0x8001 as unsigned 16-bit)
          0.0 ->  0     (0x0000)
         -0.0 ->  0     (0x0000)
  3. IEEE 754 Double-Precision Boundary Rules:
     - NaN: Converted to 0 (0x0000) per AMD VALU hardware specifications.
     - Infinities: +infinity clamps to +1.0 (0x7FFF); -infinity clamps to -1.0 (0x8001).
     - Subnormals: Extremely tiny magnitudes underflow to 0 (0x0000).
  4. VOP3 Modifiers:
     - Input modifiers:
         abs (|x|): Clears sign bit (bit 63) before conversion.
         neg (-x): Inverts sign bit (bit 63) before conversion.
         abs+neg (-|x|): Clears sign bit then inverts it, forcing negative sign.
     - Clamp modifier:
         Clamps normalized channels to non-negative range [0, 32767].

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

# Bitwise Masks & Architectural Constants
MASK64: int = 0xFFFFFFFFFFFFFFFF
MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
QNAN_BIT_F64: int = 0x0008000000000000
CANONICAL_QNAN_F64: int = 0x7FF8000000000000
CANONICAL_SNAN_F64: int = 0x7FF0000000000001

# IEEE 754 binary64 Bit Patterns
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000              # +1.0
NEG_ONE_F64: int = 0xBFF0000000000000              # -1.0
POS_HALF_F64: int = 0x3FE0000000000000             # +0.5
NEG_HALF_F64: int = 0xBFE0000000000000             # -0.5
POS_TWO_F64: int = 0x4000000000000000              # +2.0
NEG_TWO_F64: int = 0xC000000000000000              # -2.0
POS_INF_F64: int = 0x7FF0000000000000
NEG_INF_F64: int = 0xFFF0000000000000
MIN_SUBNORMAL_F64: int = 0x0000000000000001
MAX_SUBNORMAL_F64: int = 0x000FFFFFFFFFFFFF
MIN_NORMAL_F64: int = 0x0010000000000000
MAX_NORMAL_F64: int = 0x7FEFFFFFFFFFFFFF

# SNORM16 Integer Constants
SNORM16_MAX: int = 0x7FFF              # +32767
SNORM16_MIN: int = 0x8001              # -32767 (as uint16)
SNORM16_ZERO: int = 0x0000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned word as signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK_F64) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range."""
    return val & MASK16


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned word as signed two's-complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & 0x8000) else val


def split_dwords(val64: int) -> Tuple[int, int]:
    """Split 64-bit word into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into 64-bit word."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_bits(val: float) -> int:
    """Encode float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0 else NEG_INF_F64


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_nan_f64(bits: int) -> bool:
    """True if 64-bit integer represents an IEEE 754 double-precision NaN."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_qnan_f64(bits: int) -> bool:
    """True if bits represent a quiet double-precision NaN (bit 51 is 1)."""
    bits &= MASK64
    return is_nan_f64(bits) and bool(bits & QNAN_BIT_F64)


def is_snan_f64(bits: int) -> bool:
    """True if bits represent a signaling double-precision NaN (bit 51 is 0)."""
    bits &= MASK64
    return is_nan_f64(bits) and not bool(bits & QNAN_BIT_F64)


def is_inf_f64(bits: int) -> bool:
    """True if bits represent positive or negative infinity."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) == 0


def is_subnormal_f64(bits: int) -> bool:
    """True if bits represent an IEEE 754 double-precision subnormal number."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == 0 and (bits & MANT_MASK_F64) != 0


def apply_f64_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Applies VOP3 input modifiers to 64-bit float: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_mod:
        bits &= ~SIGN_MASK_F64
    if neg_mod:
        bits ^= SIGN_MASK_F64
    return bits & MASK64


def round_ties_to_even(val: float) -> int:
    """Round floating-point value to nearest integer using IEEE 754 roundTiesToEven."""
    return int(round(val))


def f64_to_snorm16(bits64: int, clamp_nonneg: bool = False) -> int:
    """Convert an IEEE 754 binary64 float to SNORM16 uint16.

    Semantics:
      - NaN converts to 0 (0x0000).
      - Clamped to [-1.0, +1.0].
      - Scaled by 32767.0.
      - Rounded with roundTiesToEven.
      - If clamp_nonneg is True, negative values clamp to 0.
    """
    bits64 &= MASK64
    if is_nan_f64(bits64):
        return 0

    val = bits_to_f64(bits64)
    if math.isinf(val):
        val = 1.0 if val > 0 else -1.0
    else:
        val = max(-1.0, min(1.0, val))

    scaled = val * 32767.0
    ival = round_ties_to_even(scaled)
    if clamp_nonneg and ival < 0:
        ival = 0

    return ival & MASK16


def unpack_pknorm_i16(packed_b32: int) -> Tuple[int, int]:
    """Unpack 32-bit packed word into (ch0_s16, ch1_s16) signed integers."""
    packed = packed_b32 & MASK32
    ch0 = s16(packed & MASK16)
    ch1 = s16((packed >> 16) & MASK16)
    return ch0, ch1


def snorm16_to_f64(val16: int) -> float:
    """Convert signed 16-bit normalized integer back to float in range [-1.0, 1.0]."""
    s_val = s16(val16)
    return max(-1.0, min(1.0, s_val / 32767.0))


def oracle_v_cvt_pknorm_i16_f64(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pknorm_i16_f64.

    Packs two double-precision float sources into a single 32-bit destination:
      dst[15:0]  = snorm16(src0)
      dst[31:16] = snorm16(src1)
    """
    s0_mod = apply_f64_modifiers(src0_bits, abs_mod=abs0, neg_mod=neg0)
    s1_mod = apply_f64_modifiers(src1_bits, abs_mod=abs1, neg_mod=neg1)

    ch0 = f64_to_snorm16(s0_mod, clamp_nonneg=clamp)
    ch1 = f64_to_snorm16(s1_mod, clamp_nonneg=clamp)

    return ((ch1 & MASK16) << 16) | (ch0 & MASK16)


# ==============================================================================
# Commit 1: Reference Mathematical Oracle and Architectural Model Tests
# ==============================================================================

class TestValuCvtPknormI16F64ModelAndOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_normalized_boundaries(self) -> None:
        """Verify conversions at normalized boundaries: +1.0, -1.0, +0.0, -0.0."""
        # +1.0 in both channels -> 0x7FFF7FFF
        res = oracle_v_cvt_pknorm_i16_f64(POS_ONE_F64, POS_ONE_F64)
        self.assertEqual(res, 0x7FFF7FFF)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 32767)
        self.assertEqual(ch1, 32767)

        # -1.0 in both channels -> 0x80018001
        res = oracle_v_cvt_pknorm_i16_f64(NEG_ONE_F64, NEG_ONE_F64)
        self.assertEqual(res, 0x80018001)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, -32767)
        self.assertEqual(ch1, -32767)

        # +0.0 and -0.0 -> 0x00000000
        res = oracle_v_cvt_pknorm_i16_f64(POS_ZERO_F64, NEG_ZERO_F64)
        self.assertEqual(res, 0x00000000)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 0)
        self.assertEqual(ch1, 0)

        # Asymmetric pairing: (+1.0, -1.0) -> 0x80017FFF
        res = oracle_v_cvt_pknorm_i16_f64(POS_ONE_F64, NEG_ONE_F64)
        self.assertEqual(res, 0x80017FFF)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 32767)
        self.assertEqual(ch1, -32767)

    def test_fractional_values(self) -> None:
        """Verify conversions for fractional values: +0.5, -0.5, +0.25, -0.25."""
        # +0.5: 0.5 * 32767 = 16383.5 -> roundTiesToEven -> 16384 (0x4000)
        res = oracle_v_cvt_pknorm_i16_f64(POS_HALF_F64, NEG_HALF_F64)
        self.assertEqual(res, 0xC0004000)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 16384)
        self.assertEqual(ch1, -16384)

        # +0.25: 0.25 * 32767 = 8191.75 -> 8192 (0x2000)
        pos_quarter = f64_to_bits(0.25)
        neg_quarter = f64_to_bits(-0.25)
        res_q = oracle_v_cvt_pknorm_i16_f64(pos_quarter, neg_quarter)
        ch0, ch1 = unpack_pknorm_i16(res_q)
        self.assertEqual(ch0, 8192)
        self.assertEqual(ch1, -8192)

    def test_round_ties_to_even(self) -> None:
        """Verify exact tie-breaking to nearest even integer."""
        self.assertEqual(round_ties_to_even(0.5), 0)
        self.assertEqual(round_ties_to_even(1.5), 2)
        self.assertEqual(round_ties_to_even(2.5), 2)
        self.assertEqual(round_ties_to_even(3.5), 4)

        self.assertEqual(f64_to_snorm16(POS_HALF_F64), 16384)
        self.assertEqual(s16(f64_to_snorm16(NEG_HALF_F64)), -16384)

    def test_out_of_range_clamping(self) -> None:
        """Verify inputs outside [-1.0, +1.0] are clamped."""
        res_pos = oracle_v_cvt_pknorm_i16_f64(POS_TWO_F64, f64_to_bits(100.0))
        self.assertEqual(res_pos, 0x7FFF7FFF)

        res_neg = oracle_v_cvt_pknorm_i16_f64(NEG_TWO_F64, f64_to_bits(-100.0))
        self.assertEqual(res_neg, 0x80018001)

    def test_nan_and_inf_handling(self) -> None:
        """Verify NaN converts to 0 and infinities clamp to boundaries."""
        res_nan = oracle_v_cvt_pknorm_i16_f64(CANONICAL_QNAN_F64, CANONICAL_SNAN_F64)
        self.assertEqual(res_nan, 0x00000000)

        res_inf = oracle_v_cvt_pknorm_i16_f64(POS_INF_F64, NEG_INF_F64)
        self.assertEqual(res_inf, 0x80017FFF)

    def test_input_modifiers(self) -> None:
        """Verify abs, neg, and abs+neg modifiers applied independently per channel."""
        res_abs = oracle_v_cvt_pknorm_i16_f64(NEG_ONE_F64, NEG_HALF_F64, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0x40007FFF)

        res_neg = oracle_v_cvt_pknorm_i16_f64(POS_ONE_F64, POS_HALF_F64, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0xC0008001)

        res_absneg = oracle_v_cvt_pknorm_i16_f64(NEG_ONE_F64, POS_HALF_F64, abs0=True, neg0=True, abs1=True, neg1=True)
        self.assertEqual(res_absneg, 0xC0008001)

    def test_clamp_modifier(self) -> None:
        """Verify clamp modifier forces negative channels to 0."""
        res_unclamped = oracle_v_cvt_pknorm_i16_f64(NEG_ONE_F64, POS_ONE_F64, clamp=False)
        self.assertEqual(res_unclamped, 0x7FFF8001)

        res_clamped = oracle_v_cvt_pknorm_i16_f64(NEG_ONE_F64, POS_ONE_F64, clamp=True)
        self.assertEqual(res_clamped, 0x7FFF0000)

    def test_unpack_and_roundtrip(self) -> None:
        """Verify unpacking and precision round-trip within 1 ULP of SNORM16."""
        for val in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
            f_bits = f64_to_bits(val)
            packed = oracle_v_cvt_pknorm_i16_f64(f_bits, f_bits)
            ch0, ch1 = unpack_pknorm_i16(packed)
            self.assertEqual(ch0, ch1)
            reconstructed = snorm16_to_f64(ch0)
            self.assertAlmostEqual(val, reconstructed, delta=1.0 / 32767.0)


# ==============================================================================
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg32:
    """Represents an architectural 32-bit register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Reg32 index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Reg64:
    """Represents an architectural aligned 64-bit register pair (e.g. v[0:1] or s[0:1])."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 254):
            raise ValueError(f"Reg64 index {self.index} out of range [0, 254]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Literal64:
    """Represents a 64-bit literal or inline float constant."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= 0xFFFFFFFFFFFFFFFF):
            raise ValueError(f"Literal64 bits 0x{self.bits:X} out of range [0, 0xFFFFFFFFFFFFFFFF]")

    @classmethod
    def from_float(cls, val: float) -> Literal64:
        return cls(f64_to_bits(val))

    @property
    def name(self) -> str:
        return f"0x{self.bits & MASK64:016X}"


@dataclass(frozen=True)
class VOP3Operand64:
    """Typed 64-bit expression operand with optional input modifiers (abs, neg)."""
    source: Union[Reg64, Literal64]
    abs_mod: bool = False
    neg_mod: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, (Reg64, Literal64)):
            raise TypeError(f"Operand source must be Reg64 or Literal64, got {type(self.source).__name__}")

    def disassemble(self) -> str:
        s = self.source.name
        if self.abs_mod and self.neg_mod:
            return f"-|{s}|"
        elif self.abs_mod:
            return f"|{s}|"
        elif self.neg_mod:
            return f"-{s}"
        return s


class Opcode(Enum):
    V_CVT_PKNORM_I16_F64 = "v_cvt_pknorm_i16_f64"


@dataclass(frozen=True)
class VCvtPknormI16F64Instruction:
    """Symbolic AST node representing a RDNA2 v_cvt_pknorm_i16_f64 instruction."""
    dst: Reg32
    src0: VOP3Operand64
    src1: VOP3Operand64
    clamp: bool = False
    opcode: Opcode = Opcode.V_CVT_PKNORM_I16_F64

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if not isinstance(self.src0, VOP3Operand64):
            raise TypeError(f"src0 must be VOP3Operand64, got {type(self.src0).__name__}")
        if not isinstance(self.src1, VOP3Operand64):
            raise TypeError(f"src1 must be VOP3Operand64, got {type(self.src1).__name__}")

    def disassemble(self) -> str:
        s = f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


class TestValuCvtPknormI16F64SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_types_and_bounds(self) -> None:
        """Verify register boundary indices and naming conventions."""
        v0 = Reg32(0, RegType.VGPR)
        v255 = Reg32(255, RegType.VGPR)
        vpair = Reg64(0, RegType.VGPR)
        spair = Reg64(2, RegType.SGPR)

        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(vpair.name, "v[0:1]")
        self.assertEqual(spair.name, "s[2:3]")

        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg64(255, RegType.VGPR)

    def test_literal_operand_creation(self) -> None:
        """Verify 64-bit literal constants creation from integers and floats."""
        lit_hex = Literal64(0x3FF0000000000000)
        lit_f = Literal64.from_float(1.0)
        self.assertEqual(lit_hex.bits, 0x3FF0000000000000)
        self.assertEqual(lit_f.bits, 0x3FF0000000000000)

        with self.assertRaises(ValueError):
            Literal64(-1)
        with self.assertRaises(ValueError):
            Literal64(0x1_0000_0000_0000_0000)

    def test_operand_modifiers_disassembly(self) -> None:
        """Verify operand formatting with abs, neg, and abs+neg modifiers."""
        vpair = Reg64(2, RegType.VGPR)
        op_plain = VOP3Operand64(vpair)
        op_abs = VOP3Operand64(vpair, abs_mod=True)
        op_neg = VOP3Operand64(vpair, neg_mod=True)
        op_absneg = VOP3Operand64(vpair, abs_mod=True, neg_mod=True)

        self.assertEqual(op_plain.disassemble(), "v[2:3]")
        self.assertEqual(op_abs.disassemble(), "|v[2:3]|")
        self.assertEqual(op_neg.disassemble(), "-v[2:3]")
        self.assertEqual(op_absneg.disassemble(), "-|v[2:3]|")

    def test_instruction_disassembly(self) -> None:
        """Verify complete instruction disassembly for standard and modified forms."""
        dst = Reg32(5, RegType.VGPR)
        s0 = VOP3Operand64(Reg64(0, RegType.VGPR))
        s1 = VOP3Operand64(Reg64(2, RegType.SGPR), neg_mod=True)

        instr = VCvtPknormI16F64Instruction(dst=dst, src0=s0, src1=s1)
        self.assertEqual(instr.disassemble(), "v_cvt_pknorm_i16_f64 v5, v[0:1], -s[2:3]")

        instr_clamped = VCvtPknormI16F64Instruction(dst=dst, src0=s0, src1=s1, clamp=True)
        self.assertEqual(instr_clamped.disassemble(), "v_cvt_pknorm_i16_f64 v5, v[0:1], -s[2:3] clamp")

    def test_ast_immutability_and_equality(self) -> None:
        """Verify frozen AST dataclasses ensure immutability and valid hashing."""
        v0 = Reg32(0, RegType.VGPR)
        s0 = VOP3Operand64(Reg64(2, RegType.VGPR))
        instr1 = VCvtPknormI16F64Instruction(v0, s0, s0)
        instr2 = VCvtPknormI16F64Instruction(v0, s0, s0)

        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))
        with self.assertRaises((dataclasses.FrozenInstanceError, TypeError)):
            instr1.clamp = True  # type: ignore

    def test_ast_rejection_sgpr_destination(self) -> None:
        """Verify VALU instructions reject SGPR destinations."""
        s0 = Reg32(0, RegType.SGPR)
        v0 = VOP3Operand64(Reg64(2, RegType.VGPR))
        with self.assertRaises(TypeError):
            VCvtPknormI16F64Instruction(dst=s0, src0=v0, src1=v0)

    def test_operand_type_rejection(self) -> None:
        """Verify invalid operand types raise TypeError."""
        with self.assertRaises(TypeError):
            VOP3Operand64("v[0:1]")  # type: ignore
        with self.assertRaises(TypeError):
            VCvtPknormI16F64Instruction(Reg32(0, RegType.VGPR), "bad", VOP3Operand64(Reg64(2, RegType.VGPR)))  # type: ignore


# ==============================================================================
# Commit 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront execution lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32

    def get_vgpr64(self, index: int) -> int:
        if not (0 <= index <= 254):
            raise IndexError(f"VGPR64 index {index} out of range [0, 254]")
        return join_dwords(self.vgprs[index], self.vgprs[index + 1])

    def set_vgpr64(self, index: int, val: int) -> None:
        if not (0 <= index <= 254):
            raise IndexError(f"VGPR64 index {index} out of range [0, 254]")
        lo, hi = split_dwords(val)
        self.vgprs[index] = lo
        self.vgprs[index + 1] = hi

    def get_sgpr64(self, index: int) -> int:
        if not (0 <= index <= 254):
            raise IndexError(f"SGPR64 index {index} out of range [0, 254]")
        return join_dwords(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr64(self, index: int, val: int) -> None:
        if not (0 <= index <= 254):
            raise IndexError(f"SGPR64 index {index} out of range [0, 254]")
        lo, hi = split_dwords(val)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_cvt_pknorm_i16_f64."""

    def resolve_operand(self, op: VOP3Operand64, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg64):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr64(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr64(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal64):
            return op.source.bits & MASK64
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VCvtPknormI16F64Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase via oracle
        result = oracle_v_cvt_pknorm_i16_f64(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
            clamp=instr.clamp,
        )

        # Writeback phase to 32-bit VGPR
        state.set_vgpr(instr.dst.index, result)
        return result


class TestValuCvtPknormI16F64ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_vgpr_sources_execution(self) -> None:
        """Verify execution with VGPR 64-bit pair sources."""
        self.state.set_vgpr64(0, POS_ONE_F64)
        self.state.set_vgpr64(2, NEG_ONE_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(4, RegType.VGPR),
            src0=VOP3Operand64(Reg64(0, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(2, RegType.VGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(self.state.get_vgpr(4), 0x80017FFF)

    def test_sgpr_sources_execution(self) -> None:
        """Verify execution reading from scalar 64-bit register file."""
        self.state.set_sgpr64(10, POS_HALF_F64)
        self.state.set_sgpr64(12, NEG_HALF_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(10, RegType.SGPR)),
            src1=VOP3Operand64(Reg64(12, RegType.SGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xC0004000)
        self.assertEqual(self.state.get_vgpr(0), 0xC0004000)
        # Verify SGPRs unchanged
        self.assertEqual(self.state.get_sgpr64(10), POS_HALF_F64)
        self.assertEqual(self.state.get_sgpr64(12), NEG_HALF_F64)

    def test_literal_sources_execution(self) -> None:
        """Verify execution with Literal64 inline constants."""
        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Literal64(POS_ONE_F64)),
            src1=VOP3Operand64(Literal64.from_float(0.0)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x00007FFF)
        self.assertEqual(self.state.get_vgpr(0), 0x00007FFF)

    def test_input_modifiers_in_interpreter(self) -> None:
        """Verify abs and neg input modifiers evaluated in interpreter."""
        self.state.set_vgpr64(0, NEG_ONE_F64)
        self.state.set_vgpr64(2, POS_HALF_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(5, RegType.VGPR),
            src0=VOP3Operand64(Reg64(0, RegType.VGPR), abs_mod=True),
            src1=VOP3Operand64(Reg64(2, RegType.VGPR), neg_mod=True),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xC0007FFF)
        self.assertEqual(self.state.get_vgpr(5), 0xC0007FFF)

    def test_clamp_modifier_in_interpreter(self) -> None:
        """Verify clamp modifier evaluated properly by interpreter."""
        self.state.set_vgpr64(0, NEG_ONE_F64)
        self.state.set_vgpr64(2, NEG_HALF_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(0, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(2, RegType.VGPR)),
            clamp=True,
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(self.state.get_vgpr(0), 0x00000000)

    def test_register_state_isolation(self) -> None:
        """Verify only destination VGPR is written and other registers remain unchanged."""
        for i in range(256):
            self.state.set_vgpr(i, 0x11111111 * (i % 7 + 1))
            self.state.set_sgpr(i, 0x22222222 * (i % 5 + 1))

        self.state.set_vgpr64(10, POS_ONE_F64)
        self.state.set_vgpr64(20, POS_ONE_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(30, RegType.VGPR),
            src0=VOP3Operand64(Reg64(10, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(20, RegType.VGPR)),
        )

        self.interp.step(instr, self.state)
        self.assertEqual(self.state.get_vgpr(30), 0x7FFF7FFF)

        for i in range(256):
            if i in (10, 11, 20, 21, 30):
                continue
            expected = 0x11111111 * (i % 7 + 1)
            self.assertEqual(self.state.get_vgpr(i), expected)

    def test_out_of_bounds_register_access(self) -> None:
        """Verify out of range register indices raise IndexError."""
        with self.assertRaises(IndexError):
            self.state.get_vgpr64(255)
        with self.assertRaises(IndexError):
            self.state.set_vgpr64(-1, 0)


# ==============================================================================
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64 execution."""
    num_lanes: int
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32

    def get_vgpr64(self, lane: int, index: int) -> int:
        return join_dwords(self.vgprs[lane][index], self.vgprs[lane][index + 1])

    def set_vgpr64(self, lane: int, index: int, val: int) -> None:
        lo, hi = split_dwords(val)
        self.vgprs[lane][index] = lo
        self.vgprs[lane][index + 1] = hi

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32

    def get_sgpr64(self, index: int) -> int:
        return join_dwords(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr64(self, index: int, val: int) -> None:
        lo, hi = split_dwords(val)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand64, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg64):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr64(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr64(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal64):
            return op.source.bits & MASK64
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VCvtPknormI16F64Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

        s0_vals = [
            self.resolve_lane_operand(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]
        s1_vals = [
            self.resolve_lane_operand(instr.src1, state, lane)
            for lane in range(state.num_lanes)
        ]

        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_cvt_pknorm_i16_f64(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                    abs1=instr.src1.abs_mod,
                    neg1=instr.src1.neg_mod,
                    clamp=instr.clamp,
                )
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestValuCvtPknormI16F64SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and Wave32/Wave64 EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_all_active(self) -> None:
        """Verify full 32-lane vector computation with unique values per lane."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            val0 = -1.0 + (2.0 * lane / 31.0)
            val1 = 1.0 - (2.0 * lane / 31.0)
            state.set_vgpr64(lane, 2, f64_to_bits(val0))
            state.set_vgpr64(lane, 4, f64_to_bits(val1))

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(2, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(4, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = oracle_v_cvt_pknorm_i16_f64(state.get_vgpr64(lane, 2), state.get_vgpr64(lane, 4))
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_all_active(self) -> None:
        """Verify full 64-lane vector execution in Wave64 mode."""
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        for lane in range(64):
            val0 = math.sin(lane * 0.1)
            val1 = math.cos(lane * 0.1)
            state.set_vgpr64(lane, 2, f64_to_bits(val0))
            state.set_vgpr64(lane, 4, f64_to_bits(val1))

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(2, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(4, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = oracle_v_cvt_pknorm_i16_f64(state.get_vgpr64(lane, 2), state.get_vgpr64(lane, 4))
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave32_even_odd_divergence(self) -> None:
        """Verify even lanes active and odd lanes masked out preserve state."""
        even_mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=even_mask)
        sentinel = 0xDEADBEEF

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr64(lane, 2, POS_ONE_F64)
            state.set_vgpr64(lane, 4, NEG_ONE_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(2, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(4, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x80017FFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0x80017FFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_exec_all_zero_mask(self) -> None:
        """Verify EXEC=0 skips execution across all lanes entirely."""
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        sentinel = 0x12345678

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr64(lane, 2, POS_ONE_F64)
            state.set_vgpr64(lane, 4, POS_ONE_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(2, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(4, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_shared_sgpr_broadcast_simd(self) -> None:
        """Verify uniform 64-bit SGPR operand is correctly broadcast across divergent lanes."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        state.set_sgpr64(4, POS_ONE_F64)

        for lane in range(32):
            val0 = -1.0 if lane < 16 else 0.5
            state.set_vgpr64(lane, 2, f64_to_bits(val0))

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(2, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(4, RegType.SGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            ch0_expected = 0x8001 if lane < 16 else 0x4000
            expected = (0x7FFF << 16) | ch0_expected
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)


# ==============================================================================
# Commit 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuCvtPknormI16F64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0_low(self) -> None:
        """Verify dst == src0.low (v0 writes to v0 while reading v[0:1])."""
        state = ScalarMachineState()
        state.set_vgpr64(0, POS_ONE_F64)
        state.set_vgpr64(2, NEG_ONE_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(0, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(2, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(state.get_vgpr(0), 0x80017FFF)
        # v1 (high dword of src0) should still hold its initial high bits
        _, hi_expected = split_dwords(POS_ONE_F64)
        self.assertEqual(state.get_vgpr(1), hi_expected)

    def test_dst_alias_src0_high(self) -> None:
        """Verify dst == src0.high (v1 writes to v1 while reading v[0:1])."""
        state = ScalarMachineState()
        state.set_vgpr64(0, POS_ONE_F64)
        state.set_vgpr64(2, NEG_ONE_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VOP3Operand64(Reg64(0, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(2, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(state.get_vgpr(1), 0x80017FFF)
        lo_expected, _ = split_dwords(POS_ONE_F64)
        self.assertEqual(state.get_vgpr(0), lo_expected)

    def test_full_self_aliasing(self) -> None:
        """Verify dst == src0.low == src1.low with identical inputs."""
        state = ScalarMachineState()
        state.set_vgpr64(0, POS_HALF_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(0, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(0, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x40004000)
        self.assertEqual(state.get_vgpr(0), 0x40004000)

    def test_src0_alias_src1_distinct_dst(self) -> None:
        """Verify src0 == src1 with distinct dst preserves source registers."""
        state = ScalarMachineState()
        state.set_vgpr64(2, NEG_ONE_F64)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(2, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(2, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x80018001)
        self.assertEqual(state.get_vgpr(0), 0x80018001)
        self.assertEqual(state.get_vgpr64(2), NEG_ONE_F64)

    def test_simd_aliasing_with_inactive_lanes(self) -> None:
        """Verify SIMD self-aliasing correctly preserves inactive lanes."""
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)
        initial_val = POS_ONE_F64

        for lane in range(32):
            state.set_vgpr64(lane, 0, initial_val)

        instr = VCvtPknormI16F64Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VOP3Operand64(Reg64(0, RegType.VGPR)),
            src1=VOP3Operand64(Reg64(0, RegType.VGPR)),
        )

        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            if lane < 16:
                self.assertEqual(results[lane], 0x7FFF7FFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0x7FFF7FFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr64(lane, 0), initial_val)


# ==============================================================================
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class CvtPknormI16F64StatusFlags:
    """Architectural condition codes and status flags for packed SNORM16 conversion from f64."""
    ch0_zero: bool
    ch1_zero: bool
    both_zero: bool
    ch0_negative: bool
    ch1_negative: bool
    ch0_clamped_max: bool
    ch0_clamped_min: bool
    ch1_clamped_max: bool
    ch1_clamped_min: bool
    overflow_or_clamped: bool
    nan_encountered: bool
    invalid_op: bool


def compute_status_flags(src0_bits: int, src1_bits: int, packed_result: int) -> CvtPknormI16F64StatusFlags:
    """Compute architectural condition codes and floating-point exception flags."""
    s0_nan = is_nan_f64(src0_bits)
    s1_nan = is_nan_f64(src1_bits)
    s0_snan = is_snan_f64(src0_bits)
    s1_snan = is_snan_f64(src1_bits)

    ch0_s16, ch1_s16 = unpack_pknorm_i16(packed_result)

    ch0_zero = (ch0_s16 == 0)
    ch1_zero = (ch1_s16 == 0)
    both_zero = ((packed_result & MASK32) == 0)

    ch0_negative = (ch0_s16 < 0)
    ch1_negative = (ch1_s16 < 0)

    ch0_clamped_max = (ch0_s16 == 32767)
    ch0_clamped_min = (ch0_s16 == -32767)
    ch1_clamped_max = (ch1_s16 == 32767)
    ch1_clamped_min = (ch1_s16 == -32767)

    f0 = bits_to_f64(src0_bits) if not s0_nan else 0.0
    f1 = bits_to_f64(src1_bits) if not s1_nan else 0.0

    s0_clamped = is_inf_f64(src0_bits) or (abs(f0) > 1.0)
    s1_clamped = is_inf_f64(src1_bits) or (abs(f1) > 1.0)
    overflow_or_clamped = s0_clamped or s1_clamped

    return CvtPknormI16F64StatusFlags(
        ch0_zero=ch0_zero,
        ch1_zero=ch1_zero,
        both_zero=both_zero,
        ch0_negative=ch0_negative,
        ch1_negative=ch1_negative,
        ch0_clamped_max=ch0_clamped_max,
        ch0_clamped_min=ch0_clamped_min,
        ch1_clamped_max=ch1_clamped_max,
        ch1_clamped_min=ch1_clamped_min,
        overflow_or_clamped=overflow_or_clamped,
        nan_encountered=s0_nan or s1_nan,
        invalid_op=s0_snan or s1_snan,
    )


class TestValuCvtPknormI16F64StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and exception detection."""

    def test_zero_flags(self) -> None:
        """Verify zero flags for zero inputs and mixed channels."""
        res_zero = oracle_v_cvt_pknorm_i16_f64(POS_ZERO_F64, NEG_ZERO_F64)
        flags_zero = compute_status_flags(POS_ZERO_F64, NEG_ZERO_F64, res_zero)
        self.assertTrue(flags_zero.ch0_zero)
        self.assertTrue(flags_zero.ch1_zero)
        self.assertTrue(flags_zero.both_zero)

        res_mixed = oracle_v_cvt_pknorm_i16_f64(POS_ONE_F64, POS_ZERO_F64)
        flags_mixed = compute_status_flags(POS_ONE_F64, POS_ZERO_F64, res_mixed)
        self.assertFalse(flags_mixed.ch0_zero)
        self.assertTrue(flags_mixed.ch1_zero)
        self.assertFalse(flags_mixed.both_zero)

    def test_negative_flags(self) -> None:
        """Verify negative flags properly distinguish signed channels."""
        res = oracle_v_cvt_pknorm_i16_f64(POS_HALF_F64, NEG_HALF_F64)
        flags = compute_status_flags(POS_HALF_F64, NEG_HALF_F64, res)
        self.assertFalse(flags.ch0_negative)
        self.assertTrue(flags.ch1_negative)

    def test_clamp_and_saturation_flags(self) -> None:
        """Verify clamped flags for boundary and out-of-range inputs."""
        res_pos = oracle_v_cvt_pknorm_i16_f64(POS_TWO_F64, POS_HALF_F64)
        flags_pos = compute_status_flags(POS_TWO_F64, POS_HALF_F64, res_pos)
        self.assertTrue(flags_pos.overflow_or_clamped)
        self.assertTrue(flags_pos.ch0_clamped_max)

        res_inf = oracle_v_cvt_pknorm_i16_f64(POS_INF_F64, NEG_INF_F64)
        flags_inf = compute_status_flags(POS_INF_F64, NEG_INF_F64, res_inf)
        self.assertTrue(flags_inf.overflow_or_clamped)
        self.assertTrue(flags_inf.ch0_clamped_max)
        self.assertTrue(flags_inf.ch1_clamped_min)

    def test_nan_and_invalid_op_flags(self) -> None:
        """Verify quiet NaN and signaling NaN exception detection."""
        res_qnan = oracle_v_cvt_pknorm_i16_f64(CANONICAL_QNAN_F64, POS_ONE_F64)
        flags_qnan = compute_status_flags(CANONICAL_QNAN_F64, POS_ONE_F64, res_qnan)
        self.assertTrue(flags_qnan.nan_encountered)
        self.assertFalse(flags_qnan.invalid_op)

        res_snan = oracle_v_cvt_pknorm_i16_f64(POS_ONE_F64, CANONICAL_SNAN_F64)
        flags_snan = compute_status_flags(POS_ONE_F64, CANONICAL_SNAN_F64, res_snan)
        self.assertTrue(flags_snan.nan_encountered)
        self.assertTrue(flags_snan.invalid_op)


if __name__ == "__main__":
    unittest.main()
