from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pknorm_u16_f64.

Models the AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) instruction:
    v_cvt_pknorm_u16_f64 D, S0, S1

Lowered Architectural Semantics:
    D[15:0]  = convert_f64_to_unorm16(S0)
    D[31:16] = convert_f64_to_unorm16(S1)

RDNA2 VALU Specification Rules:
  1. Packing & Destination:
     - Destination is a single 32-bit VGPR (v[d]).
     - Overwrites all 32 bits of destination VGPR atomically.
     - Sources S0 and S1 are 64-bit double-precision floats (binary64)
       represented as 64-bit register pairs (v[s:s+1] or s[s:s+1]) or 64-bit literals.
     - D[15:0] is derived from S0; D[31:16] is derived from S1.
  2. Conversion Mathematics:
     - Inputs are IEEE 754-2008 double-precision floating-point numbers (binary64).
     - Each channel is clamped to [0.0, 1.0]:
         * Values <= 0.0 (including -0.0, -Infinity) clamp to 0.0 -> 0x0000.
         * Values >= 1.0 (including +Infinity) clamp to 1.0 -> 0xFFFF (65535).
     - Normalized scaling:
         * Product = x * 65535.0
         * Rounded to nearest integer with tie-breaking to even (IEEE 754 roundTiesToEven).
  3. NaN Semantics:
     - Any NaN input (Quiet NaN or Signaling NaN, positive or negative sign, any payload)
       evaluates to 0x0000.
  4. VOP3 Source Modifiers:
     - abs (|x|): Clears sign bit (bit 63) of f64 input before conversion.
     - neg (-x): Inverts sign bit (bit 63) of f64 input before conversion.
     - neg(abs(x)): Inverts sign bit after abs, forcing negative sign -> clamps to 0x0000.
     - clamp on destination is rejected / unsupported.

Standard library Python only (python -I, struct, math, unittest). Zero game data.
"""

import dataclasses
import enum
import math
import struct
import unittest

# Bitwise Masks & Architectural Constants
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
SIGN64: int = 0x8000_0000_0000_0000
ABS64: int = 0x7FFF_FFFF_FFFF_FFFF
EXP_MASK64: int = 0x7FF0_0000_0000_0000
MANT_MASK64: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT64: int = 0x0008_0000_0000_0000
CANONICAL_QNAN64: int = 0x7FF8_0000_0000_0000

UNORM16_MIN: int = 0x0000
UNORM16_MAX: int = 0xFFFF
SCALE_U16: float = 65535.0

# IEEE 754 binary64 Bit Patterns
POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_ONE_F64: int = 0x3FF0_0000_0000_0000          # 1.0
NEG_ONE_F64: int = 0xBFF0_0000_0000_0000          # -1.0
POS_TWO_F64: int = 0x4000_0000_0000_0000          # 2.0
NEG_TWO_F64: int = 0xC000_0000_0000_0000          # -2.0
HALF_F64: int = 0x3FE0_0000_0000_0000             # 0.5
NEG_HALF_F64: int = 0xBFE0_0000_0000_0000         # -0.5
QUARTER_F64: int = 0x3FD0_0000_0000_0000          # 0.25
NEG_QUARTER_F64: int = 0xBFD0_0000_0000_0000      # -0.25
THREE_QUARTERS_F64: int = 0x3FE8_0000_0000_0000   # 0.75

POS_INF_F64: int = 0x7FF0_0000_0000_0000          # +Infinity
NEG_INF_F64: int = 0xFFF0_0000_0000_0000          # -Infinity
POS_QNAN_F64: int = 0x7FF8_0000_0000_0000         # Quiet NaN
NEG_QNAN_F64: int = 0xFFF8_0000_0000_0000         # Negative Quiet NaN
POS_SNAN_F64: int = 0x7FF0_0000_0000_0001         # Signaling NaN
NEG_SNAN_F64: int = 0xFFF0_0000_0000_0001         # Negative Signaling NaN
SUBNORM_MIN_F64: int = 0x0000_0000_0000_0001      # Smallest positive subnormal (2^-1074)
SUBNORM_MAX_F64: int = 0x000F_FFFF_FFFF_FFFF      # Largest positive subnormal
NEG_SUBNORM_MIN_F64: int = 0x8000_0000_0000_0001
NEG_SUBNORM_MAX_F64: int = 0x800F_FFFF_FFFF_FFFF


def f64_to_bits(val: float) -> int:
    """Pack Python float into IEEE 754 binary64 unsigned 64-bit integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0.0 else NEG_INF_F64


def bits_to_f64(bits: int) -> float:
    """Unpack unsigned 64-bit integer into Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_dwords(val64: int) -> tuple[int, int]:
    """Split 64-bit integer into low and high 32-bit dwords."""
    return val64 & MASK32, (val64 >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join low and high 32-bit dwords into a 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def is_nan64(bits: int) -> bool:
    """Return True if 64-bit word represents an IEEE 754 NaN."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & MANT_MASK64) != 0)


def is_qnan64(bits: int) -> bool:
    """Return True if 64-bit word represents an IEEE 754 Quiet NaN."""
    bits &= MASK64
    return is_nan64(bits) and bool(bits & QNAN_BIT64)


def is_snan64(bits: int) -> bool:
    """Return True if 64-bit word represents an IEEE 754 Signaling NaN."""
    bits &= MASK64
    return is_nan64(bits) and not bool(bits & QNAN_BIT64)


def is_inf64(bits: int) -> bool:
    """Return True if 64-bit word represents positive or negative infinity."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & MANT_MASK64) == 0)


def is_subnormal64(bits: int) -> bool:
    """Return True if 64-bit word represents an IEEE 754 subnormal double."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == 0) and ((bits & MANT_MASK64) != 0)


def is_zero64(bits: int) -> bool:
    """Return True if 64-bit word represents +0.0 or -0.0."""
    return (bits & ABS64) == 0


def apply_f64_modifiers(bits64: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 64-bit double word.

    In RDNA2 ISA, abs clears sign bit (bit 63) first, then neg inverts sign bit (bit 63).
    """
    res = bits64 & MASK64
    if abs_mod:
        res &= ABS64
    if neg_mod:
        res ^= SIGN64
    return res


def pack_u16_to_u32(lo_half: int, hi_half: int) -> int:
    """Pack two 16-bit integers into an unsigned 32-bit dword."""
    return ((hi_half & MASK16) << 16) | (lo_half & MASK16)


def extract_u16(val32: int, high_half: bool) -> int:
    """Extract low 16 bits (high_half=False) or high 16 bits (high_half=True)."""
    return ((val32 >> 16) & MASK16) if high_half else (val32 & MASK16)


def oracle_cvt_unorm16_f64(f64_bits: int) -> int:
    """Mathematical reference oracle converting a single f64 to unorm16.

    Semantics:
        - If input is NaN: returns 0x0000.
        - If input <= 0.0 (including -0.0, -inf): returns 0x0000.
        - If input >= 1.0 (including +inf): returns 0xFFFF.
        - Otherwise: round_half_even(x * 65535.0) & 0xFFFF.
    """
    if is_nan64(f64_bits):
        return UNORM16_MIN
    val = bits_to_f64(f64_bits)
    if val <= 0.0:
        return UNORM16_MIN
    if val >= 1.0 or math.isinf(val):
        return UNORM16_MAX
    scaled = val * SCALE_U16
    return round(scaled) & MASK16


def oracle_v_cvt_pknorm_u16_f64(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Architectural reference oracle for RDNA2 v_cvt_pknorm_u16_f64.

    Applies source modifiers, converts f64 inputs to unorm16,
    and packs into dst[31:0].
    """
    s0_mod = apply_f64_modifiers(src0_bits, abs_mod=abs0, neg_mod=neg0)
    s1_mod = apply_f64_modifiers(src1_bits, abs_mod=abs1, neg_mod=neg1)

    lo = oracle_cvt_unorm16_f64(s0_mod)
    hi = oracle_cvt_unorm16_f64(s1_mod)

    return pack_u16_to_u32(lo, hi)


# ==============================================================================
# Unit Tests: Architectural Reference Model & Mathematical Oracle
# ==============================================================================

class TestValuCvtPknormU16F64Oracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_cvt_pknorm_u16_f64."""

    def test_basic_normal_conversions(self) -> None:
        # 0.0 -> 0x0000, 1.0 -> 0xFFFF
        self.assertEqual(oracle_cvt_unorm16_f64(POS_ZERO_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(POS_ONE_F64), 0xFFFF)

        # 0.5 -> 32768 (32767.5 rounds to even 32768)
        self.assertEqual(oracle_cvt_unorm16_f64(HALF_F64), 32768)

        # 0.25 -> 16384 (16383.75 rounds to 16384)
        self.assertEqual(oracle_cvt_unorm16_f64(QUARTER_F64), 16384)

        # 0.75 -> 49151 (49151.25 rounds to 49151)
        self.assertEqual(oracle_cvt_unorm16_f64(THREE_QUARTERS_F64), 49151)

    def test_channel_packing_and_ordering(self) -> None:
        # Low channel (S0) gets 1.0 -> 0xFFFF
        # High channel (S1) gets 0.5 -> 32768 (0x8000)
        res = oracle_v_cvt_pknorm_u16_f64(POS_ONE_F64, HALF_F64)
        expected = (32768 << 16) | 0xFFFF
        self.assertEqual(res, expected)
        self.assertEqual(extract_u16(res, high_half=False), 0xFFFF)
        self.assertEqual(extract_u16(res, high_half=True), 32768)

    def test_clamping_out_of_range(self) -> None:
        # Negative values clamp to 0x0000
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_ONE_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_TWO_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(f64_to_bits(-100.0)), 0)

        # Values > 1.0 clamp to 0xFFFF
        self.assertEqual(oracle_cvt_unorm16_f64(POS_TWO_F64), 0xFFFF)
        self.assertEqual(oracle_cvt_unorm16_f64(f64_to_bits(1.5)), 0xFFFF)
        self.assertEqual(oracle_cvt_unorm16_f64(f64_to_bits(1000.0)), 0xFFFF)

    def test_infinities(self) -> None:
        self.assertEqual(oracle_cvt_unorm16_f64(POS_INF_F64), 0xFFFF)
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_INF_F64), 0)
        res = oracle_v_cvt_pknorm_u16_f64(POS_INF_F64, NEG_INF_F64)
        self.assertEqual(res, 0x0000FFFF)

    def test_nan_evaluates_to_zero(self) -> None:
        self.assertEqual(oracle_cvt_unorm16_f64(POS_QNAN_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_QNAN_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(POS_SNAN_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_SNAN_F64), 0)

    def test_signed_zeros(self) -> None:
        self.assertEqual(oracle_cvt_unorm16_f64(POS_ZERO_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_ZERO_F64), 0)
        self.assertEqual(oracle_v_cvt_pknorm_u16_f64(POS_ZERO_F64, NEG_ZERO_F64), 0)

    def test_round_ties_to_even_steps(self) -> None:
        # In double precision:
        # 0.5 / 65535.0 gives exact product 0.5 -> rounds to 0 (even)
        self.assertEqual(oracle_cvt_unorm16_f64(f64_to_bits(0.5 / 65535.0)), 0)
        # 1.5 / 65535.0 gives exact product 1.5 -> rounds to 2 (even)
        self.assertEqual(oracle_cvt_unorm16_f64(f64_to_bits(1.5 / 65535.0)), 2)
        # 2.5 / 65535.0 gives exact product 2.5 -> rounds to 2 (even)
        self.assertEqual(oracle_cvt_unorm16_f64(f64_to_bits(2.5 / 65535.0)), 2)
        # 3.5 / 65535.0 gives exact product 3.5 -> rounds to 4 (even)
        self.assertEqual(oracle_cvt_unorm16_f64(f64_to_bits(3.5 / 65535.0)), 4)

    def test_source_modifiers(self) -> None:
        # abs on negative input: |-0.25| = 0.25 -> 16384
        res_abs = oracle_v_cvt_pknorm_u16_f64(NEG_QUARTER_F64, POS_ZERO_F64, abs0=True)
        self.assertEqual(extract_u16(res_abs, high_half=False), 16384)

        # neg on positive input: -0.25 -> clamps to 0
        res_neg = oracle_v_cvt_pknorm_u16_f64(QUARTER_F64, POS_ZERO_F64, neg0=True)
        self.assertEqual(extract_u16(res_neg, high_half=False), 0)

        # abs then neg (-|x|): forces negative -> clamps to 0
        res_both = oracle_v_cvt_pknorm_u16_f64(QUARTER_F64, POS_ZERO_F64, abs0=True, neg0=True)
        self.assertEqual(extract_u16(res_both, high_half=False), 0)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclasses.dataclass(frozen=True)
class Reg:
    """Represents an architectural register or 64-bit aligned register pair."""
    index: int
    reg_type: RegType
    is_double: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")
        if self.is_double and self.index > 254:
            raise ValueError(f"64-bit register pair base index {self.index} exceeds 254")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        if self.is_double:
            return f"{prefix}[{self.index}:{self.index + 1}]"
        return f"{prefix}{self.index}"


@dataclasses.dataclass(frozen=True)
class Literal64:
    """Represents an immediate 64-bit float or integer constant."""
    bits: int

    @classmethod
    def from_f64(cls, val: float) -> Literal64:
        return cls(f64_to_bits(val))

    @property
    def name(self) -> str:
        return f"0x{self.bits & MASK64:016X}"


@dataclasses.dataclass(frozen=True)
class VOP3Operand64:
    """Typed operand for 64-bit source with optional modifiers (abs, neg)."""
    source: Reg | Literal64
    abs_mod: bool = False
    neg_mod: bool = False

    def disassemble(self) -> str:
        s = self.source.name
        if self.abs_mod:
            s = f"|{s}|"
        if self.neg_mod:
            s = f"-{s}"
        return s


class Opcode(enum.Enum):
    V_CVT_PKNORM_U16_F64 = "v_cvt_pknorm_u16_f64"


@dataclasses.dataclass(frozen=True)
class VCvtPknormU16F64Instruction:
    """Symbolic AST node for RDNA2 v_cvt_pknorm_u16_f64."""
    dst: Reg
    src0: VOP3Operand64
    src1: VOP3Operand64
    clamp: bool = False
    opcode: Opcode = Opcode.V_CVT_PKNORM_U16_F64

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.dst.is_double:
            raise TypeError("Destination of v_cvt_pknorm_u16_f64 must be single 32-bit VGPR")
        if self.clamp:
            raise ValueError("v_cvt_pknorm_u16_f64 destination cannot have clamp modifier")

    def disassemble(self) -> str:
        return f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"


@dataclasses.dataclass
class Program:
    """Represents an ordered sequence of instructions."""
    instructions: list[VCvtPknormU16F64Instruction] = dataclasses.field(default_factory=list)

    def append(self, instr: VCvtPknormU16F64Instruction) -> None:
        self.instructions.append(instr)

    def disassemble(self) -> str:
        return "\n".join(instr.disassemble() for instr in self.instructions)


class TestValuCvtPknormU16F64AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_creation_and_naming(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        v_pair = Reg(2, RegType.VGPR, is_double=True)
        s_pair = Reg(4, RegType.SGPR, is_double=True)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(v_pair.name, "v[2:3]")
        self.assertEqual(s_pair.name, "s[4:5]")

    def test_register_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(255, RegType.VGPR, is_double=True)

    def test_literal_operand(self) -> None:
        lit = Literal64(0x3FF0000000000000)
        self.assertEqual(lit.name, "0x3FF0000000000000")
        lit_f64 = Literal64.from_f64(1.0)
        self.assertEqual(lit_f64.bits, POS_ONE_F64)

    def test_operand_modifiers_disassembly(self) -> None:
        v_pair = Reg(2, RegType.VGPR, is_double=True)
        op_plain = VOP3Operand64(v_pair)
        self.assertEqual(op_plain.disassemble(), "v[2:3]")

        op_abs = VOP3Operand64(v_pair, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v[2:3]|")

        op_neg = VOP3Operand64(v_pair, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v[2:3]")

        op_both = VOP3Operand64(v_pair, abs_mod=True, neg_mod=True)
        self.assertEqual(op_both.disassemble(), "-|v[2:3]|")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand64(Reg(2, RegType.VGPR, is_double=True))
        src1 = VOP3Operand64(Reg(4, RegType.VGPR, is_double=True), neg_mod=True)

        instr = VCvtPknormU16F64Instruction(dst=dst, src0=src0, src1=src1)
        self.assertEqual(instr.disassemble(), "v_cvt_pknorm_u16_f64 v0, v[2:3], -v[4:5]")

    def test_instruction_validation(self) -> None:
        # Destination must be VGPR
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand64(Reg(2, RegType.VGPR, is_double=True))
        with self.assertRaises(TypeError):
            VCvtPknormU16F64Instruction(dst=dst_sgpr, src0=src, src1=src)

        # Destination cannot be double VGPR
        dst_double = Reg(0, RegType.VGPR, is_double=True)
        with self.assertRaises(TypeError):
            VCvtPknormU16F64Instruction(dst=dst_double, src0=src, src1=src)

        # Destination clamp modifier rejected
        with self.assertRaises(ValueError):
            VCvtPknormU16F64Instruction(dst=Reg(0, RegType.VGPR), src0=src, src1=src, clamp=True)

    def test_program_container(self) -> None:
        prog = Program()
        v0 = Reg(0, RegType.VGPR)
        v2 = Reg(2, RegType.VGPR, is_double=True)
        v4 = Reg(4, RegType.VGPR, is_double=True)
        prog.append(VCvtPknormU16F64Instruction(v0, VOP3Operand64(v2), VOP3Operand64(v4)))
        expected = "v_cvt_pknorm_u16_f64 v0, v[2:3], v[4:5]"
        self.assertEqual(prog.disassemble(), expected)


if __name__ == "__main__":
    unittest.main()
