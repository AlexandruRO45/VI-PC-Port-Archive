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


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_cvt_pknorm_i16_f32."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VCvtPknormI16F32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase via oracle
        result = oracle_v_cvt_pknorm_i16_f32(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
            clamp=instr.clamp,
        )

        # Writeback phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestValuCvtPknormI16F32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_vgpr_sources_execution(self) -> None:
        """Verify execution with VGPR sources."""
        self.state.set_vgpr(1, POS_ONE_F32)
        self.state.set_vgpr(2, NEG_ONE_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(self.state.get_vgpr(0), 0x80017FFF)

    def test_sgpr_sources_execution(self) -> None:
        """Verify execution reading from scalar register file."""
        self.state.set_sgpr(10, POS_HALF_F32)
        self.state.set_sgpr(11, NEG_HALF_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.SGPR)),
            src1=VOP3Operand(Reg(11, RegType.SGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xC0004000)
        self.assertEqual(self.state.get_vgpr(3), 0xC0004000)
        # Verify SGPRs were not modified
        self.assertEqual(self.state.get_sgpr(10), POS_HALF_F32)
        self.assertEqual(self.state.get_sgpr(11), NEG_HALF_F32)

    def test_literal_sources_execution(self) -> None:
        """Verify execution with Literal32 inline constants."""
        instr = VCvtPknormI16F32Instruction(
            dst=Reg(4, RegType.VGPR),
            src0=VOP3Operand(Literal32(POS_ONE_F32)),
            src1=VOP3Operand(Literal32.from_float(0.0)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x00007FFF)
        self.assertEqual(self.state.get_vgpr(4), 0x00007FFF)

    def test_mixed_sources(self) -> None:
        """Verify execution with mixed VGPR, SGPR, and Literal32 operands."""
        self.state.set_vgpr(1, POS_ONE_F32)
        self.state.set_sgpr(2, NEG_ONE_F32)

        # VGPR and SGPR
        instr1 = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.SGPR)),
        )
        res1 = self.interp.step(instr1, self.state)
        self.assertEqual(res1, 0x80017FFF)

        # SGPR and Literal
        instr2 = VCvtPknormI16F32Instruction(
            dst=Reg(1, RegType.VGPR),
            src0=VOP3Operand(Reg(2, RegType.SGPR)),
            src1=VOP3Operand(Literal32.from_float(1.0)),
        )
        res2 = self.interp.step(instr2, self.state)
        self.assertEqual(res2, 0x7FFF8001)

    def test_input_modifiers_in_interpreter(self) -> None:
        """Verify abs and neg input modifiers evaluated properly by interpreter."""
        self.state.set_vgpr(1, NEG_ONE_F32)
        self.state.set_vgpr(2, POS_HALF_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR), abs_mod=True),
            src1=VOP3Operand(Reg(2, RegType.VGPR), neg_mod=True),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0xC0007FFF)
        self.assertEqual(self.state.get_vgpr(0), 0xC0007FFF)

    def test_clamp_modifier_in_interpreter(self) -> None:
        """Verify clamp modifier evaluated properly by interpreter."""
        self.state.set_vgpr(1, NEG_ONE_F32)
        self.state.set_vgpr(2, NEG_HALF_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            clamp=True,
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(self.state.get_vgpr(0), 0x00000000)

    def test_register_state_isolation(self) -> None:
        """Verify only destination VGPR is written and all other registers remain unchanged."""
        for i in range(256):
            self.state.set_vgpr(i, 0x11111111 * (i % 7 + 1))
            self.state.set_sgpr(i, 0x22222222 * (i % 5 + 1))

        self.state.set_vgpr(10, POS_ONE_F32)
        self.state.set_vgpr(20, POS_ONE_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(30, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(20, RegType.VGPR)),
        )

        self.interp.step(instr, self.state)
        self.assertEqual(self.state.get_vgpr(30), 0x7FFF7FFF)

        # Check all other VGPRs
        for i in range(256):
            if i in (10, 20, 30):
                continue
            expected = 0x11111111 * (i % 7 + 1)
            self.assertEqual(self.state.get_vgpr(i), expected)

        # Check all SGPRs
        for i in range(256):
            expected = 0x22222222 * (i % 5 + 1)
            self.assertEqual(self.state.get_sgpr(i), expected)

    def test_out_of_bounds_register_access(self) -> None:
        """Verify out of range register indices raise IndexError."""
        with self.assertRaises(IndexError):
            self.state.get_vgpr(256)
        with self.assertRaises(IndexError):
            self.state.set_vgpr(-1, 0)
        with self.assertRaises(IndexError):
            self.state.get_sgpr(300)
        with self.assertRaises(IndexError):
            self.state.set_sgpr(-5, 0)


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

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VCvtPknormI16F32Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        s0_vals = [
            self.resolve_lane_operand(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]
        s1_vals = [
            self.resolve_lane_operand(instr.src1, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution and writeback phase under EXEC mask
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_cvt_pknorm_i16_f32(
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


class TestValuCvtPknormI16F32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and Wave32/Wave64 EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_all_active(self) -> None:
        """Verify full 32-lane vector computation with unique values per lane."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            # Input values from -1.0 to +1.0
            val0 = -1.0 + (2.0 * lane / 31.0)
            val1 = 1.0 - (2.0 * lane / 31.0)
            state.set_vgpr(lane, 1, f32_to_bits(val0))
            state.set_vgpr(lane, 2, f32_to_bits(val1))

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = oracle_v_cvt_pknorm_i16_f32(state.get_vgpr(lane, 1), state.get_vgpr(lane, 2))
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_all_active(self) -> None:
        """Verify full 64-lane vector execution in Wave64 mode."""
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        for lane in range(64):
            val0 = math.sin(lane * 0.1)
            val1 = math.cos(lane * 0.1)
            state.set_vgpr(lane, 1, f32_to_bits(val0))
            state.set_vgpr(lane, 2, f32_to_bits(val1))

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = oracle_v_cvt_pknorm_i16_f32(state.get_vgpr(lane, 1), state.get_vgpr(lane, 2))
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave32_even_odd_divergence(self) -> None:
        """Verify even lanes active and odd lanes masked out preserve state."""
        even_mask = 0x55555555  # bits 0, 2, 4, ...
        state = SIMDState(num_lanes=32, exec_mask=even_mask)
        initial_sentinel = 0xDEADBEEF

        for lane in range(32):
            state.set_vgpr(lane, 0, initial_sentinel)
            state.set_vgpr(lane, 1, POS_ONE_F32)
            state.set_vgpr(lane, 2, NEG_ONE_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x80017FFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0x80017FFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), initial_sentinel)

    def test_wave64_divergence_high_low(self) -> None:
        """Verify low 32 lanes active and high 32 lanes masked out in Wave64."""
        low32_mask = 0x00000000FFFFFFFF
        state = SIMDState(num_lanes=64, exec_mask=low32_mask)
        sentinel = 0xBAADF00D

        for lane in range(64):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, POS_HALF_F32)
            state.set_vgpr(lane, 2, POS_HALF_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], 0x40004000)
                self.assertEqual(state.get_vgpr(lane, 0), 0x40004000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_exec_all_zero_mask(self) -> None:
        """Verify EXEC=0 skips execution across all lanes entirely."""
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        sentinel = 0x12345678

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, POS_ONE_F32)
            state.set_vgpr(lane, 2, POS_ONE_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_single_lane_active(self) -> None:
        """Verify single active lane executes while 31 other lanes remain untouched."""
        lane_target = 19
        state = SIMDState(num_lanes=32, exec_mask=(1 << lane_target))
        sentinel = 0xCAFEBABE

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, POS_ONE_F32)
            state.set_vgpr(lane, 2, POS_ONE_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane == lane_target:
                self.assertEqual(results[lane], 0x7FFF7FFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0x7FFF7FFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_shared_sgpr_broadcast_simd(self) -> None:
        """Verify uniform SGPR operand is correctly broadcast across divergent lanes."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        state.set_sgpr(5, POS_ONE_F32)  # Shared +1.0f

        for lane in range(32):
            # Divergent channel 0 values
            val0 = -1.0 if lane < 16 else 0.5
            state.set_vgpr(lane, 1, f32_to_bits(val0))

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(5, RegType.SGPR)),
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

class TestValuCvtPknormI16F32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        """Verify dst == src0 reads original src0 before destination overwrite."""
        state = ScalarMachineState()
        state.set_vgpr(0, POS_ONE_F32)   # src0 initial: +1.0f
        state.set_vgpr(1, NEG_ONE_F32)   # src1: -1.0f

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(state.get_vgpr(0), 0x80017FFF)
        self.assertEqual(state.get_vgpr(1), NEG_ONE_F32)  # src1 preserved

    def test_dst_alias_src1(self) -> None:
        """Verify dst == src1 reads original src1 before destination overwrite."""
        state = ScalarMachineState()
        state.set_vgpr(1, POS_ONE_F32)   # src0: +1.0f
        state.set_vgpr(0, NEG_ONE_F32)   # src1 initial: -1.0f

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x80017FFF)
        self.assertEqual(state.get_vgpr(0), 0x80017FFF)
        self.assertEqual(state.get_vgpr(1), POS_ONE_F32)  # src0 preserved

    def test_full_self_aliasing(self) -> None:
        """Verify dst == src0 == src1 correctly converts and duplicates channels."""
        state = ScalarMachineState()
        state.set_vgpr(7, POS_HALF_F32)  # +0.5f

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(7, RegType.VGPR),
            src0=VOP3Operand(Reg(7, RegType.VGPR)),
            src1=VOP3Operand(Reg(7, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        # Both channels convert +0.5f -> 0x4000
        self.assertEqual(res, 0x40004000)
        self.assertEqual(state.get_vgpr(7), 0x40004000)

    def test_src0_alias_src1_distinct_dst(self) -> None:
        """Verify src0 == src1 with distinct dst preserves source register."""
        state = ScalarMachineState()
        state.set_vgpr(1, NEG_ONE_F32)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x80018001)
        self.assertEqual(state.get_vgpr(0), 0x80018001)
        self.assertEqual(state.get_vgpr(1), NEG_ONE_F32)  # Source untouched

    def test_aliased_operand_with_modifiers(self) -> None:
        """Verify self-aliased register with asymmetric channel modifiers."""
        state = ScalarMachineState()
        state.set_vgpr(2, NEG_HALF_F32)  # -0.5f

        # src0: abs(-0.5) = +0.5 -> 0x4000
        # src1: neg(-0.5) = +0.5 -> 0x4000
        instr = VCvtPknormI16F32Instruction(
            dst=Reg(2, RegType.VGPR),
            src0=VOP3Operand(Reg(2, RegType.VGPR), abs_mod=True),
            src1=VOP3Operand(Reg(2, RegType.VGPR), neg_mod=True),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x40004000)
        self.assertEqual(state.get_vgpr(2), 0x40004000)

    def test_sequential_chained_aliasing(self) -> None:
        """Verify sequential dependent instructions with aliased registers."""
        state = ScalarMachineState()
        state.set_vgpr(0, POS_ONE_F32)

        # Step 1: v0 = pack(v0, 0.0) -> (0x0000, 0x7FFF) = 0x00007FFF
        instr1 = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Literal32.from_float(0.0)),
        )
        res1 = self.scalar_interp.step(instr1, state)
        self.assertEqual(res1, 0x00007FFF)
        self.assertEqual(state.get_vgpr(0), 0x00007FFF)

        # Step 2: v0 is now a subnormal float in IEEE 754 (bits 0x00007FFF).
        # v1 = pack(v0, v0). Subnormals underflow to 0.
        instr2 = VCvtPknormI16F32Instruction(
            dst=Reg(1, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
        )
        res2 = self.scalar_interp.step(instr2, state)
        self.assertEqual(res2, 0x00000000)
        self.assertEqual(state.get_vgpr(1), 0x00000000)
        self.assertEqual(state.get_vgpr(0), 0x00007FFF)

    def test_simd_aliasing_with_inactive_lanes(self) -> None:
        """Verify SIMD self-aliasing correctly preserves inactive lanes."""
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # Low 16 lanes active
        initial_val = POS_ONE_F32

        for lane in range(32):
            state.set_vgpr(lane, 0, initial_val)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
        )

        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            if lane < 16:
                self.assertEqual(results[lane], 0x7FFF7FFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0x7FFF7FFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), initial_val)


# ==============================================================================
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class CvtPknormStatusFlags:
    """Architectural condition codes and status flags for packed SNORM16 conversion."""
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


def compute_status_flags(src0_bits: int, src1_bits: int, packed_result: int) -> CvtPknormStatusFlags:
    """Compute architectural condition codes and floating-point exception flags."""
    s0_nan = is_nan_f32(src0_bits)
    s1_nan = is_nan_f32(src1_bits)
    s0_snan = is_snan_f32(src0_bits)
    s1_snan = is_snan_f32(src1_bits)

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

    # Check if either input magnitude exceeded 1.0f or was inf
    f0 = bits_to_f32(src0_bits) if not s0_nan else 0.0
    f1 = bits_to_f32(src1_bits) if not s1_nan else 0.0

    s0_clamped = is_inf_f32(src0_bits) or (abs(f0) > 1.0)
    s1_clamped = is_inf_f32(src1_bits) or (abs(f1) > 1.0)
    overflow_or_clamped = s0_clamped or s1_clamped

    return CvtPknormStatusFlags(
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


class TestValuCvtPknormI16F32StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and exception detection."""

    def test_zero_flags(self) -> None:
        """Verify zero flags for zero inputs, non-zero inputs, and mixed channels."""
        # Both channels zero
        res_zero = oracle_v_cvt_pknorm_i16_f32(POS_ZERO_F32, NEG_ZERO_F32)
        flags_zero = compute_status_flags(POS_ZERO_F32, NEG_ZERO_F32, res_zero)
        self.assertTrue(flags_zero.ch0_zero)
        self.assertTrue(flags_zero.ch1_zero)
        self.assertTrue(flags_zero.both_zero)

        # Ch0 non-zero, Ch1 zero
        res_mixed = oracle_v_cvt_pknorm_i16_f32(POS_ONE_F32, POS_ZERO_F32)
        flags_mixed = compute_status_flags(POS_ONE_F32, POS_ZERO_F32, res_mixed)
        self.assertFalse(flags_mixed.ch0_zero)
        self.assertTrue(flags_mixed.ch1_zero)
        self.assertFalse(flags_mixed.both_zero)

        # Both channels non-zero
        res_nonzero = oracle_v_cvt_pknorm_i16_f32(POS_HALF_F32, NEG_HALF_F32)
        flags_nonzero = compute_status_flags(POS_HALF_F32, NEG_HALF_F32, res_nonzero)
        self.assertFalse(flags_nonzero.ch0_zero)
        self.assertFalse(flags_nonzero.ch1_zero)
        self.assertFalse(flags_nonzero.both_zero)

    def test_negative_flags(self) -> None:
        """Verify negative flags properly distinguish signed channels."""
        # Mixed sign
        res = oracle_v_cvt_pknorm_i16_f32(POS_HALF_F32, NEG_HALF_F32)
        flags = compute_status_flags(POS_HALF_F32, NEG_HALF_F32, res)
        self.assertFalse(flags.ch0_negative)
        self.assertTrue(flags.ch1_negative)

        # Both negative
        res_neg = oracle_v_cvt_pknorm_i16_f32(NEG_ONE_F32, NEG_HALF_F32)
        flags_neg = compute_status_flags(NEG_ONE_F32, NEG_HALF_F32, res_neg)
        self.assertTrue(flags_neg.ch0_negative)
        self.assertTrue(flags_neg.ch1_negative)

    def test_clamp_and_saturation_flags(self) -> None:
        """Verify clamped flags for boundary and out-of-range inputs."""
        # Normal in-range value (no clamp)
        res_normal = oracle_v_cvt_pknorm_i16_f32(POS_HALF_F32, NEG_HALF_F32)
        flags_normal = compute_status_flags(POS_HALF_F32, NEG_HALF_F32, res_normal)
        self.assertFalse(flags_normal.overflow_or_clamped)
        self.assertFalse(flags_normal.ch0_clamped_max)
        self.assertFalse(flags_normal.ch1_clamped_min)

        # Out-of-range positive clamp (> 1.0f)
        res_pos_ovf = oracle_v_cvt_pknorm_i16_f32(POS_TWO_F32, POS_HALF_F32)
        flags_pos_ovf = compute_status_flags(POS_TWO_F32, POS_HALF_F32, res_pos_ovf)
        self.assertTrue(flags_pos_ovf.overflow_or_clamped)
        self.assertTrue(flags_pos_ovf.ch0_clamped_max)
        self.assertFalse(flags_pos_ovf.ch1_clamped_max)

        # Out-of-range negative clamp (< -1.0f)
        res_neg_ovf = oracle_v_cvt_pknorm_i16_f32(POS_HALF_F32, NEG_TWO_F32)
        flags_neg_ovf = compute_status_flags(POS_HALF_F32, NEG_TWO_F32, res_neg_ovf)
        self.assertTrue(flags_neg_ovf.overflow_or_clamped)
        self.assertFalse(flags_neg_ovf.ch0_clamped_min)
        self.assertTrue(flags_neg_ovf.ch1_clamped_min)

        # Infinity clamping
        res_inf = oracle_v_cvt_pknorm_i16_f32(POS_INF_F32, NEG_INF_F32)
        flags_inf = compute_status_flags(POS_INF_F32, NEG_INF_F32, res_inf)
        self.assertTrue(flags_inf.overflow_or_clamped)
        self.assertTrue(flags_inf.ch0_clamped_max)
        self.assertTrue(flags_inf.ch1_clamped_min)

    def test_nan_and_invalid_op_flags(self) -> None:
        """Verify quiet NaN and signaling NaN exception detection."""
        # Quiet NaN
        res_qnan = oracle_v_cvt_pknorm_i16_f32(CANONICAL_QNAN_F32, POS_ONE_F32)
        flags_qnan = compute_status_flags(CANONICAL_QNAN_F32, POS_ONE_F32, res_qnan)
        self.assertTrue(flags_qnan.nan_encountered)
        self.assertFalse(flags_qnan.invalid_op)

        # Signaling NaN
        res_snan = oracle_v_cvt_pknorm_i16_f32(POS_ONE_F32, CANONICAL_SNAN_F32)
        flags_snan = compute_status_flags(POS_ONE_F32, CANONICAL_SNAN_F32, res_snan)
        self.assertTrue(flags_snan.nan_encountered)
        self.assertTrue(flags_snan.invalid_op)

    def test_simd_vector_condition_mask(self) -> None:
        """Verify SIMD lane mask generation from per-lane status condition codes."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            val = -1.0 if lane % 2 == 0 else 1.0
            state.set_vgpr(lane, 1, f32_to_bits(val))
            state.set_vgpr(lane, 2, POS_ZERO_F32)

        interp = SIMDInterpreter()
        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = interp.step(instr, state)
        vcc_neg_mask = 0
        for lane in range(32):
            res = results[lane]
            assert res is not None
            flags = compute_status_flags(state.get_vgpr(lane, 1), state.get_vgpr(lane, 2), res)
            if flags.ch0_negative:
                vcc_neg_mask |= (1 << lane)

        # Even lanes should have ch0_negative=True (mask = 0x55555555)
        self.assertEqual(vcc_neg_mask, 0x55555555)


# ==============================================================================
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestValuCvtPknormI16F32IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, infinities, and NaNs."""

    def test_signed_zeros(self) -> None:
        """Verify +0.0 and -0.0 both convert to 0x0000 regardless of modifiers."""
        # Unmodified
        self.assertEqual(f32_to_snorm16(POS_ZERO_F32), 0)
        self.assertEqual(f32_to_snorm16(NEG_ZERO_F32), 0)

        # Modifiers on zero
        res_abs = oracle_v_cvt_pknorm_i16_f32(NEG_ZERO_F32, NEG_ZERO_F32, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0x00000000)

        res_neg = oracle_v_cvt_pknorm_i16_f32(POS_ZERO_F32, POS_ZERO_F32, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0x00000000)

    def test_subnormal_numbers(self) -> None:
        """Verify positive and negative subnormal floats underflow to 0x0000."""
        # Smallest positive subnormal: 2^-149 (approx 1.4e-45)
        self.assertEqual(f32_to_snorm16(MIN_SUBNORMAL_F32), 0)

        # Largest positive subnormal: (1 - 2^-23) * 2^-126 (approx 1.17e-38)
        self.assertEqual(f32_to_snorm16(MAX_SUBNORMAL_F32), 0)

        # Negative subnormals
        min_sub_neg = MIN_SUBNORMAL_F32 | SIGN_MASK_F32
        max_sub_neg = MAX_SUBNORMAL_F32 | SIGN_MASK_F32
        self.assertEqual(f32_to_snorm16(min_sub_neg), 0)
        self.assertEqual(f32_to_snorm16(max_sub_neg), 0)

        # Packed combination of subnormals
        res = oracle_v_cvt_pknorm_i16_f32(MIN_SUBNORMAL_F32, max_sub_neg)
        self.assertEqual(res, 0x00000000)

    def test_smallest_normal_float(self) -> None:
        """Verify smallest positive normal float 2^-126 underflows to 0."""
        min_normal_f32 = 0x00800000  # 2^-126
        min_normal_neg_f32 = 0x80800000
        self.assertEqual(f32_to_snorm16(min_normal_f32), 0)
        self.assertEqual(f32_to_snorm16(min_normal_neg_f32), 0)

    def test_infinities(self) -> None:
        """Verify +inf and -inf clamp to +32767 (0x7FFF) and -32767 (0x8001)."""
        # +inf -> 0x7FFF
        self.assertEqual(f32_to_snorm16(POS_INF_F32), 0x7FFF)
        # -inf -> 0x8001
        self.assertEqual(f32_to_snorm16(NEG_INF_F32), 0x8001)

        # +inf with neg modifier -> -inf -> 0x8001
        res_neg = oracle_v_cvt_pknorm_i16_f32(POS_INF_F32, POS_INF_F32, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0x80018001)

        # -inf with abs modifier -> +inf -> 0x7FFF
        res_abs = oracle_v_cvt_pknorm_i16_f32(NEG_INF_F32, NEG_INF_F32, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0x7FFF7FFF)

    def test_nan_payload_variations(self) -> None:
        """Verify all varieties of Quiet and Signaling NaNs convert to 0x0000."""
        qnan_payloads = [
            0x7FC00001,  # Quiet NaN with payload 1
            0x7FFFFFFF,  # Quiet NaN with all payload bits set
            0xFFC00000,  # Negative Quiet NaN
            0xFFFFFFFF,  # Negative Quiet NaN with full payload
        ]
        snan_payloads = [
            0x7F800001,  # Signaling NaN with bit 0
            0x7FBFFFFF,  # Signaling NaN with highest mantissa bit clear
            0xFF800001,  # Negative Signaling NaN
            0xFFBFFFFF,  # Negative Signaling NaN with payload
        ]

        for payload in qnan_payloads + snan_payloads:
            self.assertTrue(is_nan_f32(payload))
            self.assertEqual(f32_to_snorm16(payload), 0)

        # Mixed NaN pairs in oracle
        res = oracle_v_cvt_pknorm_i16_f32(0x7FC00001, 0xFF800001)
        self.assertEqual(res, 0x00000000)

    def test_max_finite_floats(self) -> None:
        """Verify maximum finite single-precision floats clamp to boundaries."""
        max_float = 0x7F7FFFFF       # Approx +3.4028e+38
        min_float_neg = 0xFF7FFFFF   # Approx -3.4028e+38

        self.assertEqual(f32_to_snorm16(max_float), 0x7FFF)
        self.assertEqual(f32_to_snorm16(min_float_neg), 0x8001)

        res = oracle_v_cvt_pknorm_i16_f32(max_float, min_float_neg)
        self.assertEqual(res, 0x80017FFF)

    def test_epsilon_around_one(self) -> None:
        """Verify precision right at and near the 1.0 boundary."""
        # 1.0 - 2^-24 (largest float strictly less than 1.0)
        near_one_pos = 0x3F7FFFFF
        # 1.0 + 2^-23 (smallest float strictly greater than 1.0)
        above_one_pos = 0x3F800001

        # Both round/clamp to 32767
        self.assertEqual(f32_to_snorm16(near_one_pos), 0x7FFF)
        self.assertEqual(f32_to_snorm16(above_one_pos), 0x7FFF)

        # -1.0 + 2^-24 (closest float above -1.0)
        near_one_neg = 0xBF7FFFFF
        # -1.0 - 2^-23 (closest float below -1.0)
        below_one_neg = 0xBF800001

        self.assertEqual(f32_to_snorm16(near_one_neg), 0x8001)
        self.assertEqual(f32_to_snorm16(below_one_neg), 0x8001)

    def test_snorm16_quantum_steps(self) -> None:
        """Verify smallest non-zero quantizations (1 / 32767.0)."""
        step = 1.0 / 32767.0
        step_pos = f32_to_bits(step)
        step_neg = f32_to_bits(-step)

        # 1.0 / 32767.0 rounds to 1 (0x0001)
        self.assertEqual(f32_to_snorm16(step_pos), 0x0001)
        # -1.0 / 32767.0 rounds to -1 (0xFFFF)
        self.assertEqual(f32_to_snorm16(step_neg), 0xFFFF)

        # Fractional steps: 0.4 ulp rounds to 0, 0.6 ulp rounds to 1
        sub_step = f32_to_bits(0.4 / 32767.0)
        sup_step = f32_to_bits(0.6 / 32767.0)
        self.assertEqual(f32_to_snorm16(sub_step), 0x0000)
        self.assertEqual(f32_to_snorm16(sup_step), 0x0001)


# ==============================================================================
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestValuCvtPknormI16F32NegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for syntax rejection, invalid operand handling, and mutation sensitivity."""

    def test_syntax_rejection_sgpr_destination(self) -> None:
        """Verify VALU instructions reject scalar registers (SGPR) as destination."""
        s0 = Reg(0, RegType.SGPR)
        v1 = Reg(1, RegType.VGPR)
        with self.assertRaises(TypeError):
            VCvtPknormI16F32Instruction(
                dst=s0,
                src0=VOP3Operand(v1),
                src1=VOP3Operand(v1),
            )

    def test_syntax_rejection_invalid_operand_types(self) -> None:
        """Verify instruction construction rejects non-VOP3Operand objects."""
        v0 = Reg(0, RegType.VGPR)
        v1 = Reg(1, RegType.VGPR)

        with self.assertRaises(TypeError):
            VCvtPknormI16F32Instruction(dst=v0, src0=v1, src1=VOP3Operand(v1))  # type: ignore
        with self.assertRaises(TypeError):
            VCvtPknormI16F32Instruction(dst=v0, src0=VOP3Operand(v1), src1=123)  # type: ignore
        with self.assertRaises(TypeError):
            VOP3Operand(source="not_a_register")  # type: ignore

    def test_syntax_rejection_register_bounds(self) -> None:
        """Verify registers reject out-of-bound indices."""
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_simd_invalid_configuration(self) -> None:
        """Verify SIMD machine state rejects non-standard wave sizes."""
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_mutation_incorrect_scale_factor(self) -> None:
        """Mutation detection: verify test suite detects incorrect 32768.0 scale factor."""
        def mutated_f32_to_snorm16(bits: int) -> int:
            if is_nan_f32(bits):
                return 0
            val = bits_to_f32(bits)
            clamped = max(-1.0, min(1.0, val))
            # MUTATION: using 32768.0 instead of 32767.0
            return int(round(clamped * 32768.0)) & MASK16

        # Under correct oracle, +1.0f produces 0x7FFF (32767).
        # Mutated scale factor produces 32768 (0x8000).
        correct_val = f32_to_snorm16(POS_ONE_F32)
        mutated_val = mutated_f32_to_snorm16(POS_ONE_F32)

        self.assertEqual(correct_val, 0x7FFF)
        self.assertEqual(mutated_val, 0x8000)
        self.assertNotEqual(correct_val, mutated_val)

    def test_mutation_missing_input_clamp(self) -> None:
        """Mutation detection: verify test suite detects omitted [-1.0, 1.0] clamp."""
        def mutated_f32_to_snorm16_noclamp(bits: int) -> int:
            if is_nan_f32(bits):
                return 0
            val = bits_to_f32(bits)
            # MUTATION: no clamping of input float
            return int(round(val * 32767.0)) & MASK16

        # For +2.0f, correct oracle clamps to +1.0f -> 0x7FFF
        # Mutated un-clamped function overflows
        correct_val = f32_to_snorm16(POS_TWO_F32)
        mutated_val = mutated_f32_to_snorm16_noclamp(POS_TWO_F32)

        self.assertEqual(correct_val, 0x7FFF)
        self.assertNotEqual(correct_val, mutated_val)

    def test_mutation_truncation_instead_of_round_to_even(self) -> None:
        """Mutation detection: verify test suite detects truncation (round-toward-zero)."""
        def mutated_f32_to_snorm16_trunc(bits: int) -> int:
            if is_nan_f32(bits):
                return 0
            val = max(-1.0, min(1.0, bits_to_f32(bits)))
            # MUTATION: integer truncation instead of roundTiesToEven
            return int(val * 32767.0) & MASK16

        # For +0.5f: 0.5 * 32767.0 = 16383.5.
        # roundTiesToEven rounds to 16384 (0x4000). Truncation yields 16383 (0x3FFF).
        correct_val = f32_to_snorm16(POS_HALF_F32)
        mutated_val = mutated_f32_to_snorm16_trunc(POS_HALF_F32)

        self.assertEqual(correct_val, 0x4000)
        self.assertEqual(mutated_val, 0x3FFF)
        self.assertNotEqual(correct_val, mutated_val)

    def test_mutation_inverted_channel_packing(self) -> None:
        """Mutation detection: verify test suite detects swapped destination channels."""
        def mutated_pack(s0: int, s1: int) -> int:
            ch0 = f32_to_snorm16(s0)
            ch1 = f32_to_snorm16(s1)
            # MUTATION: inverted channel placement (ch0 in high 16, ch1 in low 16)
            return ((ch0 & MASK16) << 16) | (ch1 & MASK16)

        # For (s0=+1.0f, s1=-1.0f), correct packing is 0x80017FFF.
        # Mutated packing produces 0x7FFF8001.
        correct_packed = oracle_v_cvt_pknorm_i16_f32(POS_ONE_F32, NEG_ONE_F32)
        mutated_packed = mutated_pack(POS_ONE_F32, NEG_ONE_F32)

        self.assertEqual(correct_packed, 0x80017FFF)
        self.assertEqual(mutated_packed, 0x7FFF8001)
        self.assertNotEqual(correct_packed, mutated_packed)

    def test_mutation_nan_leak(self) -> None:
        """Mutation detection: verify test suite detects failure to zero NaNs."""
        def mutated_f32_to_snorm16_nan_leak(bits: int) -> int:
            # MUTATION: NaNs convert to 0x7FFF instead of 0
            if is_nan_f32(bits):
                return 0x7FFF
            return f32_to_snorm16(bits)

        correct_val = f32_to_snorm16(CANONICAL_QNAN_F32)
        mutated_val = mutated_f32_to_snorm16_nan_leak(CANONICAL_QNAN_F32)

        self.assertEqual(correct_val, 0x0000)
        self.assertEqual(mutated_val, 0x7FFF)
        self.assertNotEqual(correct_val, mutated_val)

    def test_mutation_exec_mask_bypass(self) -> None:
        """Mutation detection: verify test suite detects inactive lane write violation."""
        state = SIMDState(num_lanes=32, exec_mask=0x00000001)  # Only lane 0 active
        sentinel = 0x55555555
        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)

        # MUTATION: interpreter incorrectly ignores EXEC mask and writes to all lanes
        def mutated_step(instr: VCvtPknormI16F32Instruction, st: SIMDState) -> None:
            for lane in range(st.num_lanes):
                # Bug: ignoring st.exec_mask
                st.set_vgpr(lane, instr.dst.index, 0x7FFF7FFF)

        instr = VCvtPknormI16F32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Literal32.from_float(1.0)),
            src1=VOP3Operand(Literal32.from_float(1.0)),
        )

        mutated_step(instr, state)
        # Lane 1 was masked off, but mutated step altered it
        self.assertNotEqual(state.get_vgpr(1, 0), sentinel)


if __name__ == "__main__":
    unittest.main()
