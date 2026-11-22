from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pknorm_i16_f32.

Models IEEE 754 binary32 floating-point to signed normalized 16-bit integer (SNORM16)
conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  v_cvt_pknorm_i16_f32 D, S0, S1

Lowered semantics:
  D[15:0]  = f32_to_snorm16(S0)
  D[31:16] = f32_to_snorm16(S1)

RDNA2 Architectural Specifications:
  1. Operand Layout:
     - Destination D is a single 32-bit vector register (VGPR).
     - Sources S0 and S1 are 32-bit single-precision floats (VGPR, SGPR, or literal constant).
     - Channel 0 (S0) occupies the lower 16 bits [15:0].
     - Channel 1 (S1) occupies the upper 16 bits [31:16].
  2. Fixed-Point Signed Normalization (SNORM16):
     - Normalized scale factor is 32767.0f (2^15 - 1 = 0x7FFF).
     - Input values outside [-1.0f, +1.0f] are clamped to [-1.0f, +1.0f].
     - Floating-point product (clamped * 32767.0f) is rounded using IEEE 754 roundTiesToEven.
     - Representable integer range: [-32767, +32767].
     - Mapping:
         +1.0f -> +32767 (0x7FFF)
         -1.0f -> -32767 (0x8001 in two's complement uint16)
          0.0f ->  0     (0x0000)
         -0.0f ->  0     (0x0000)
  3. IEEE 754 Boundary Rules:
     - NaN: Converted to 0 (0x0000) per DirectX / AMD VALU specification.
     - Infinities: +infinity clamps to +1.0f (0x7FFF); -infinity clamps to -1.0f (0x8001).
     - Subnormals: Tiny magnitudes underflow to 0 (0x0000).
  4. VOP3 Modifiers:
     - Input modifiers:
         abs (|x|): Clears sign bit (bit 31) before conversion.
         neg (-x): Inverts sign bit (bit 31) before conversion.
         abs+neg (-|x|): Clears sign bit then inverts it, guaranteeing <= 0.0f.
     - Clamp modifier:
         Clamps normalized channels to non-negative range [0, 32767].

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

# Bitwise Masks & Constants
MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF
QNAN_BIT_F32: int = 0x00400000

# Canonical Bit Patterns
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_ONE_F32: int = 0x3F800000          # +1.0f
NEG_ONE_F32: int = 0xBF800000          # -1.0f
POS_HALF_F32: int = 0x3F000000         # +0.5f
NEG_HALF_F32: int = 0xBF000000         # -0.5f
POS_TWO_F32: int = 0x40000000          # +2.0f
NEG_TWO_F32: int = 0xC0000000          # -2.0f
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
CANONICAL_QNAN_F32: int = 0x7FC00000
CANONICAL_SNAN_F32: int = 0x7F800001
MIN_SUBNORMAL_F32: int = 0x00000001
MAX_SUBNORMAL_F32: int = 0x007FFFFF
MAX_NORMAL_F32: int = 0x7F7FFFFF

# SNORM16 Integer Constants
SNORM16_MAX: int = 0x7FFF              # +32767
SNORM16_MIN: int = 0x8001              # -32767 (as uint16)
SNORM16_ZERO: int = 0x0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 65535]."""
    return val & MASK16


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned word as a signed two's-complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & 0x8000) else val


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def is_nan_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return (bits & EXP_MASK_F32) == EXP_MASK_F32 and (bits & MANT_MASK_F32) != 0


def is_qnan_f32(bits: int) -> bool:
    """True if bits represent a quiet single-precision NaN (bit 22 is 1)."""
    bits &= MASK32
    return is_nan_f32(bits) and bool(bits & QNAN_BIT_F32)


def is_snan_f32(bits: int) -> bool:
    """True if bits represent a signaling single-precision NaN (bit 22 is 0)."""
    bits &= MASK32
    return is_nan_f32(bits) and not bool(bits & QNAN_BIT_F32)


def is_inf_f32(bits: int) -> bool:
    """True if bits represent positive or negative infinity."""
    bits &= MASK32
    return (bits & EXP_MASK_F32) == EXP_MASK_F32 and (bits & MANT_MASK_F32) == 0


def is_subnormal_f32(bits: int) -> bool:
    """True if bits represent an IEEE 754 single-precision subnormal number."""
    bits &= MASK32
    return (bits & EXP_MASK_F32) == 0 and (bits & MANT_MASK_F32) != 0


def apply_f32_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Applies VOP3 input modifiers to 32-bit float: abs (|x|) first, then neg (-x)."""
    bits &= MASK32
    if abs_mod:
        bits &= ~SIGN_MASK_F32
    if neg_mod:
        bits ^= SIGN_MASK_F32
    return bits & MASK32


def round_ties_to_even(val: float) -> int:
    """Round floating-point value to nearest integer using IEEE 754 roundTiesToEven."""
    return int(round(val))


def f32_to_snorm16(bits: int, clamp_nonneg: bool = False) -> int:
    """Convert a single-precision float (as 32-bit integer) to SNORM16 uint16.

    Semantics:
      - NaN converts to 0 (0x0000).
      - Clamped to [-1.0f, +1.0f].
      - Scaled by 32767.0f.
      - Rounded with roundTiesToEven.
      - If clamp_nonneg is True, negative values clamp to 0.
    """
    bits &= MASK32
    if is_nan_f32(bits):
        return 0

    val = bits_to_f32(bits)
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


def snorm16_to_f32(val16: int) -> float:
    """Convert signed 16-bit normalized integer back to float in range [-1.0, 1.0]."""
    s_val = s16(val16)
    return max(-1.0, min(1.0, s_val / 32767.0))


def oracle_v_cvt_pknorm_i16_f32(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pknorm_i16_f32.

    Packs two single-precision float sources into a single 32-bit destination:
      dst[15:0]  = snorm16(src0)
      dst[31:16] = snorm16(src1)
    """
    s0_mod = apply_f32_modifiers(src0_bits, abs_mod=abs0, neg_mod=neg0)
    s1_mod = apply_f32_modifiers(src1_bits, abs_mod=abs1, neg_mod=neg1)

    ch0 = f32_to_snorm16(s0_mod, clamp_nonneg=clamp)
    ch1 = f32_to_snorm16(s1_mod, clamp_nonneg=clamp)

    return ((ch1 & MASK16) << 16) | (ch0 & MASK16)


# ==============================================================================
# Commit 1: Reference Mathematical Oracle and Architectural Model Tests
# ==============================================================================

class TestValuCvtPknormI16F32ModelAndOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_normalized_boundaries(self) -> None:
        """Verify conversions at normalized boundaries: +1.0, -1.0, +0.0, -0.0."""
        # +1.0f in both channels -> 0x7FFF7FFF
        res = oracle_v_cvt_pknorm_i16_f32(POS_ONE_F32, POS_ONE_F32)
        self.assertEqual(res, 0x7FFF7FFF)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 32767)
        self.assertEqual(ch1, 32767)

        # -1.0f in both channels -> 0x80018001
        res = oracle_v_cvt_pknorm_i16_f32(NEG_ONE_F32, NEG_ONE_F32)
        self.assertEqual(res, 0x80018001)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, -32767)
        self.assertEqual(ch1, -32767)

        # +0.0f and -0.0f -> 0x00000000
        res = oracle_v_cvt_pknorm_i16_f32(POS_ZERO_F32, NEG_ZERO_F32)
        self.assertEqual(res, 0x00000000)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 0)
        self.assertEqual(ch1, 0)

        # Asymmetric boundary pairing: (+1.0f, -1.0f) -> 0x80017FFF
        res = oracle_v_cvt_pknorm_i16_f32(POS_ONE_F32, NEG_ONE_F32)
        self.assertEqual(res, 0x80017FFF)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 32767)
        self.assertEqual(ch1, -32767)

    def test_fractional_values(self) -> None:
        """Verify conversions for fractional values: +0.5, -0.5, +0.25, -0.25."""
        # +0.5f: 0.5 * 32767 = 16383.5 -> roundTiesToEven -> 16384 (0x4000)
        # -0.5f: -0.5 * 32767 = -16383.5 -> roundTiesToEven -> -16384 (0xC000)
        res = oracle_v_cvt_pknorm_i16_f32(POS_HALF_F32, NEG_HALF_F32)
        self.assertEqual(res, 0xC0004000)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 16384)
        self.assertEqual(ch1, -16384)

        # +0.25f: 0.25 * 32767 = 8191.75 -> 8192 (0x2000)
        pos_quarter = f32_to_bits(0.25)
        neg_quarter = f32_to_bits(-0.25)
        res = oracle_v_cvt_pknorm_i16_f32(pos_quarter, neg_quarter)
        ch0, ch1 = unpack_pknorm_i16(res)
        self.assertEqual(ch0, 8192)
        self.assertEqual(ch1, -8192)

    def test_round_ties_to_even(self) -> None:
        """Verify exact tie-breaking to nearest even integer."""
        # round_ties_to_even on exact mathematical half-integers
        self.assertEqual(round_ties_to_even(0.5), 0)
        self.assertEqual(round_ties_to_even(1.5), 2)
        self.assertEqual(round_ties_to_even(2.5), 2)
        self.assertEqual(round_ties_to_even(3.5), 4)
        self.assertEqual(round_ties_to_even(-0.5), 0)
        self.assertEqual(round_ties_to_even(-1.5), -2)
        self.assertEqual(round_ties_to_even(-2.5), -2)

        # In f32, 0.5f is exact: 0.5 * 32767.0 = 16383.5 (half-way between 16383 and 16384).
        # Nearest even is 16384 (0x4000).
        self.assertEqual(f32_to_snorm16(POS_HALF_F32), 16384)
        self.assertEqual(s16(f32_to_snorm16(NEG_HALF_F32)), -16384)

    def test_out_of_range_clamping(self) -> None:
        """Verify inputs outside [-1.0f, +1.0f] are clamped to [-1.0f, +1.0f]."""
        # +2.0f and +100.0f clamp to +1.0f (0x7FFF)
        res = oracle_v_cvt_pknorm_i16_f32(POS_TWO_F32, f32_to_bits(100.0))
        self.assertEqual(res, 0x7FFF7FFF)

        # -2.0f and -100.0f clamp to -1.0f (0x8001)
        res = oracle_v_cvt_pknorm_i16_f32(NEG_TWO_F32, f32_to_bits(-100.0))
        self.assertEqual(res, 0x80018001)

    def test_nan_and_inf_handling(self) -> None:
        """Verify NaN converts to 0 and infinities clamp to boundaries."""
        # NaNs convert to 0
        res = oracle_v_cvt_pknorm_i16_f32(CANONICAL_QNAN_F32, CANONICAL_SNAN_F32)
        self.assertEqual(res, 0x00000000)

        # Infinities clamp to +32767 and -32767
        res = oracle_v_cvt_pknorm_i16_f32(POS_INF_F32, NEG_INF_F32)
        self.assertEqual(res, 0x80017FFF)

    def test_input_modifiers(self) -> None:
        """Verify abs, neg, and abs+neg modifiers applied independently per channel."""
        # abs on negative input yields positive normalized output
        res_abs = oracle_v_cvt_pknorm_i16_f32(NEG_ONE_F32, NEG_HALF_F32, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0x40007FFF)

        # neg on positive input yields negative normalized output
        res_neg = oracle_v_cvt_pknorm_i16_f32(POS_ONE_F32, POS_HALF_F32, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0xC0008001)

        # abs+neg guarantees negative normalized output even if originally negative
        res_absneg = oracle_v_cvt_pknorm_i16_f32(NEG_ONE_F32, POS_HALF_F32, abs0=True, neg0=True, abs1=True, neg1=True)
        self.assertEqual(res_absneg, 0xC0008001)

    def test_clamp_modifier(self) -> None:
        """Verify clamp modifier forces negative channels to 0."""
        # Without clamp: -1.0f yields -32767 (0x8001)
        res_unclamped = oracle_v_cvt_pknorm_i16_f32(NEG_ONE_F32, POS_ONE_F32, clamp=False)
        self.assertEqual(res_unclamped, 0x7FFF8001)

        # With clamp: -1.0f clamps to 0 (0x0000)
        res_clamped = oracle_v_cvt_pknorm_i16_f32(NEG_ONE_F32, POS_ONE_F32, clamp=True)
        self.assertEqual(res_clamped, 0x7FFF0000)

    def test_unpack_and_roundtrip(self) -> None:
        """Verify unpacking and precision round-trip within 1 ULP of SNORM16."""
        for val in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
            f_bits = f32_to_bits(val)
            packed = oracle_v_cvt_pknorm_i16_f32(f_bits, f_bits)
            ch0, ch1 = unpack_pknorm_i16(packed)
            self.assertEqual(ch0, ch1)
            reconstructed = snorm16_to_f32(ch0)
            self.assertAlmostEqual(val, reconstructed, delta=1.0 / 32767.0)


# ==============================================================================
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg:
    """Represents an architectural 32-bit register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit literal or inline float constant."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= 0xFFFFFFFF):
            raise ValueError(f"Literal32 bits 0x{self.bits:X} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def from_float(cls, val: float) -> Literal32:
        return cls(f32_to_bits(val))

    @property
    def name(self) -> str:
        return f"0x{self.bits & MASK32:08X}"


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand with optional input modifiers (abs, neg)."""
    source: Union[Reg, Literal32]
    abs_mod: bool = False
    neg_mod: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, (Reg, Literal32)):
            raise TypeError(f"Operand source must be Reg or Literal32, got {type(self.source).__name__}")

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
    V_CVT_PKNORM_I16_F32 = "v_cvt_pknorm_i16_f32"


@dataclass(frozen=True)
class VCvtPknormI16F32Instruction:
    """Symbolic AST node representing a RDNA2 v_cvt_pknorm_i16_f32 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    clamp: bool = False
    opcode: Opcode = Opcode.V_CVT_PKNORM_I16_F32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if not isinstance(self.src0, VOP3Operand):
            raise TypeError(f"src0 must be VOP3Operand, got {type(self.src0).__name__}")
        if not isinstance(self.src1, VOP3Operand):
            raise TypeError(f"src1 must be VOP3Operand, got {type(self.src1).__name__}")

    def disassemble(self) -> str:
        s = f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


class TestValuCvtPknormI16F32SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_types_and_bounds(self) -> None:
        """Verify register boundary indices and naming conventions."""
        v0 = Reg(0, RegType.VGPR)
        v255 = Reg(255, RegType.VGPR)
        s0 = Reg(0, RegType.SGPR)
        s255 = Reg(255, RegType.SGPR)

        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(s0.name, "s0")
        self.assertEqual(s255.name, "s255")

        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)

    def test_literal_operand_creation(self) -> None:
        """Verify 32-bit literal constants creation from integers and floats."""
        lit_hex = Literal32(0x3F800000)
        lit_f = Literal32.from_float(1.0)
        self.assertEqual(lit_hex.bits, 0x3F800000)
        self.assertEqual(lit_f.bits, 0x3F800000)
        self.assertEqual(lit_f.name, "0x3F800000")

        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_operand_modifiers_disassembly(self) -> None:
        """Verify operand formatting with abs, neg, and abs+neg modifiers."""
        v1 = Reg(1, RegType.VGPR)
        op_plain = VOP3Operand(v1)
        op_abs = VOP3Operand(v1, abs_mod=True)
        op_neg = VOP3Operand(v1, neg_mod=True)
        op_absneg = VOP3Operand(v1, abs_mod=True, neg_mod=True)

        self.assertEqual(op_plain.disassemble(), "v1")
        self.assertEqual(op_abs.disassemble(), "|v1|")
        self.assertEqual(op_neg.disassemble(), "-v1")
        self.assertEqual(op_absneg.disassemble(), "-|v1|")

    def test_instruction_disassembly(self) -> None:
        """Verify complete instruction disassembly for standard and modified forms."""
        dst = Reg(5, RegType.VGPR)
        s0 = VOP3Operand(Reg(1, RegType.VGPR))
        s1 = VOP3Operand(Reg(2, RegType.SGPR), neg_mod=True)

        instr = VCvtPknormI16F32Instruction(dst=dst, src0=s0, src1=s1)
        self.assertEqual(instr.disassemble(), "v_cvt_pknorm_i16_f32 v5, v1, -s2")

        instr_clamped = VCvtPknormI16F32Instruction(dst=dst, src0=s0, src1=s1, clamp=True)
        self.assertEqual(instr_clamped.disassemble(), "v_cvt_pknorm_i16_f32 v5, v1, -s2 clamp")

    def test_ast_immutability_and_equality(self) -> None:
        """Verify frozen AST dataclasses ensure immutability and valid hashing."""
        import dataclasses
        v0 = Reg(0, RegType.VGPR)
        v1 = Reg(1, RegType.VGPR)
        instr1 = VCvtPknormI16F32Instruction(v0, VOP3Operand(v1), VOP3Operand(v1))
        instr2 = VCvtPknormI16F32Instruction(v0, VOP3Operand(v1), VOP3Operand(v1))

        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))
        with self.assertRaises((dataclasses.FrozenInstanceError, TypeError)):
            instr1.clamp = True  # type: ignore

    def test_ast_rejection_sgpr_destination(self) -> None:
        """Verify VALU instructions reject SGPR destinations."""
        s0 = Reg(0, RegType.SGPR)
        v1 = Reg(1, RegType.VGPR)
        with self.assertRaises(TypeError):
            VCvtPknormI16F32Instruction(dst=s0, src0=VOP3Operand(v1), src1=VOP3Operand(v1))

    def test_operand_type_rejection(self) -> None:
        """Verify invalid operand types raise TypeError."""
        with self.assertRaises(TypeError):
            VOP3Operand("v1")  # type: ignore
        with self.assertRaises(TypeError):
            VCvtPknormI16F32Instruction(Reg(0, RegType.VGPR), "bad", VOP3Operand(Reg(1, RegType.VGPR)))  # type: ignore


if __name__ == "__main__":
    unittest.main()
