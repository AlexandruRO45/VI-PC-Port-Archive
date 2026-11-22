from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pknorm_u16_f16.

Models IEEE 754-2008 binary16 half-precision floating-point to unsigned normalized 16-bit
integer (UNORM16) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU)
architectural semantics:
  v_cvt_pknorm_u16_f16 D, S0, S1 [op_sel:[s0, s1]]

Lowered Architectural Semantics:
  D[15:0]  = f16_to_unorm16(select_half(S0, op_sel[0]))
  D[31:16] = f16_to_unorm16(select_half(S1, op_sel[1]))

RDNA2 VALU Specification Rules:
  1. Packing & Destination:
     - Destination is a single 32-bit VGPR (v[d]).
     - Overwrites all 32 bits of the destination register atomically.
     - D[15:0] is derived exclusively from S0; D[31:16] is derived exclusively from S1.
  2. Operand Selection (op_sel):
     - op_sel[0] selects the low 16 bits (0) or high 16 bits (1) of S0.
     - op_sel[1] selects the low 16 bits (0) or high 16 bits (1) of S1.
  3. Conversion Mathematics:
     - Input S0 and S1 half-words are IEEE 754-2008 binary16 floats.
     - Clamped to [0.0, 1.0]:
         * Values <= 0.0 (including -0.0 and negative numbers) clamp to 0.0 -> 0x0000.
         * Values >= 1.0 clamp to 1.0 -> 0xFFFF (65535).
         * +Infinity clamps to 1.0 -> 0xFFFF.
         * -Infinity clamps to 0.0 -> 0x0000.
     - Normalized scaling:
         * Product = clamped * 65535.0
         * Rounded to nearest integer with tie-breaking to even (IEEE 754 roundTiesToEven).
  4. NaN Semantics:
     - Any NaN input (Quiet NaN or Signaling NaN, positive or negative sign, any payload)
       evaluates to 0x0000 per AMD VALU hardware specifications.
  5. VOP3 Source Modifiers:
     - abs (|x|): Clears sign bit (bit 15) of f16 input before conversion.
     - neg (-x): Inverts sign bit (bit 15) of f16 input before conversion.
     - neg(abs(x)): Inverts sign bit after abs, forcing negative sign -> clamps to 0x0000.

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
MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
CANONICAL_QNAN_F16: int = 0x7E00
CANONICAL_SNAN_F16: int = 0x7C01

UNORM16_MIN: int = 0x0000
UNORM16_MAX: int = 0xFFFF
SCALE_U16: float = 65535.0

# IEEE 754 binary16 Bit Patterns
POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_ONE_F16: int = 0x3C00              # +1.0f16
NEG_ONE_F16: int = 0xBC00              # -1.0f16
POS_HALF_F16: int = 0x3800             # +0.5f16
NEG_HALF_F16: int = 0xB800             # -0.5f16
POS_TWO_F16: int = 0x4000              # +2.0f16
NEG_TWO_F16: int = 0xC000              # -2.0f16
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
MIN_SUBNORMAL_F16: int = 0x0001        # 2^-24
MAX_SUBNORMAL_F16: int = 0x03FF
MIN_NORMAL_F16: int = 0x0400           # 2^-14
MAX_NORMAL_F16: int = 0x7BFF           # 65504.0


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 65535]."""
    return val & MASK16


def f16_to_bits(val: float) -> int:
    """Encode float to IEEE 754 half-precision (binary16) unsigned 16-bit integer."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Decode 16-bit unsigned integer to IEEE 754 half-precision float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_nan_f16(bits: int) -> bool:
    """True if 16-bit word represents an IEEE 754 half-precision NaN."""
    bits &= MASK16
    return (bits & EXP_MASK_F16) == EXP_MASK_F16 and (bits & MANT_MASK_F16) != 0


def is_qnan_f16(bits: int) -> bool:
    """True if bits represent a quiet half-precision NaN (bit 9 is 1)."""
    bits &= MASK16
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """True if bits represent a signaling half-precision NaN (bit 9 is 0)."""
    bits &= MASK16
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """True if bits represent positive or negative infinity."""
    bits &= MASK16
    return (bits & EXP_MASK_F16) == EXP_MASK_F16 and (bits & MANT_MASK_F16) == 0


def is_subnormal_f16(bits: int) -> bool:
    """True if bits represent an IEEE 754 half-precision subnormal number."""
    bits &= MASK16
    return (bits & EXP_MASK_F16) == 0 and (bits & MANT_MASK_F16) != 0


def apply_f16_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Applies VOP3 input modifiers to 16-bit float: abs (|x|) first, then neg (-x)."""
    bits &= MASK16
    if abs_mod:
        bits &= ~SIGN_MASK_F16
    if neg_mod:
        bits ^= SIGN_MASK_F16
    return bits & MASK16


def select_f16_half(src32: int, sel_high: int) -> int:
    """Select low (0) or high (1) 16-bit half of a 32-bit register."""
    src = src32 & MASK32
    return ((src >> 16) & MASK16) if sel_high else (src & MASK16)


def round_ties_to_even(val: float) -> int:
    """Round floating-point value to nearest integer using IEEE 754 roundTiesToEven."""
    return int(round(val))


def f16_to_unorm16(bits16: int) -> int:
    """Convert an IEEE 754 binary16 float to UNORM16 uint16.

    Semantics:
      - NaN converts to 0 (0x0000).
      - Clamped to [0.0, 1.0].
      - Scaled by 65535.0.
      - Rounded with roundTiesToEven.
      - Representable range: [0, 65535].
    """
    bits16 &= MASK16
    if is_nan_f16(bits16):
        return 0

    val = bits_to_f16(bits16)
    if math.isinf(val):
        val = 1.0 if val > 0 else 0.0
    else:
        val = max(0.0, min(1.0, val))

    scaled = val * SCALE_U16
    return round_ties_to_even(scaled) & MASK16


def unpack_pknorm_u16(packed_b32: int) -> Tuple[int, int]:
    """Unpack 32-bit packed word into (ch0_u16, ch1_u16) unsigned integers."""
    packed = packed_b32 & MASK32
    ch0 = packed & MASK16
    ch1 = (packed >> 16) & MASK16
    return ch0, ch1


def unorm16_to_f16(val16: int) -> float:
    """Convert unsigned 16-bit normalized integer back to float in range [0.0, 1.0]."""
    u_val = val16 & MASK16
    return max(0.0, min(1.0, u_val / SCALE_U16))


def oracle_v_cvt_pknorm_u16_f16(
    src0_bits: int,
    src1_bits: int,
    op_sel: Tuple[int, int] = (0, 0),
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pknorm_u16_f16.

    Packs two half-precision float sources into a single 32-bit destination:
      dst[15:0]  = unorm16(src0_half)
      dst[31:16] = unorm16(src1_half)
    """
    h0 = select_f16_half(src0_bits, op_sel[0])
    h1 = select_f16_half(src1_bits, op_sel[1])

    h0_mod = apply_f16_modifiers(h0, abs_mod=abs0, neg_mod=neg0)
    h1_mod = apply_f16_modifiers(h1, abs_mod=abs1, neg_mod=neg1)

    u0 = f16_to_unorm16(h0_mod)
    u1 = f16_to_unorm16(h1_mod)

    return ((u1 & MASK16) << 16) | (u0 & MASK16)


# ==============================================================================
# Commit 1: Reference Mathematical Oracle and Architectural Model Tests
# ==============================================================================

class TestValuCvtPknormU16F16ModelAndOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_normalized_boundaries(self) -> None:
        """Verify conversions at normalized boundaries: +1.0, +0.0, -0.0, -1.0."""
        # +1.0 in both channels -> 0xFFFFFFFF
        res = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_ONE_F16)
        self.assertEqual(res, 0xFFFFFFFF)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 65535)
        self.assertEqual(ch1, 65535)

        # 0.0 in both channels -> 0x00000000
        res = oracle_v_cvt_pknorm_u16_f16(POS_ZERO_F16, NEG_ZERO_F16)
        self.assertEqual(res, 0x00000000)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 0)
        self.assertEqual(ch1, 0)

        # Negative values clamp to 0x0000
        res = oracle_v_cvt_pknorm_u16_f16(NEG_ONE_F16, NEG_ONE_F16)
        self.assertEqual(res, 0x00000000)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 0)
        self.assertEqual(ch1, 0)

        # Asymmetric pairing: (+1.0, 0.0) -> 0x0000FFFF
        res = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_ZERO_F16)
        self.assertEqual(res, 0x0000FFFF)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 65535)
        self.assertEqual(ch1, 0)

    def test_fractional_values(self) -> None:
        """Verify conversions for fractional values: +0.5, +0.25, +0.75."""
        # +0.5: 0.5 * 65535.0 = 32767.5 -> roundTiesToEven -> 32768 (0x8000)
        res = oracle_v_cvt_pknorm_u16_f16(POS_HALF_F16, POS_HALF_F16)
        self.assertEqual(res, 0x80008000)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 32768)
        self.assertEqual(ch1, 32768)

        # +0.25: 0.25 * 65535.0 = 16383.75 -> 16384 (0x4000)
        q25 = f16_to_bits(0.25)
        res_q = oracle_v_cvt_pknorm_u16_f16(q25, q25)
        self.assertEqual(res_q, 0x40004000)

        # +0.75: 0.75 * 65535.0 = 49151.25 -> 49151 (0xBFFF)
        q75 = f16_to_bits(0.75)
        res_75 = oracle_v_cvt_pknorm_u16_f16(q75, q75)
        self.assertEqual(res_75, 0xBFFFBFFF)

    def test_round_ties_to_even(self) -> None:
        """Verify exact tie-breaking to nearest even integer."""
        self.assertEqual(round_ties_to_even(0.5), 0)
        self.assertEqual(round_ties_to_even(1.5), 2)
        self.assertEqual(round_ties_to_even(2.5), 2)
        self.assertEqual(round_ties_to_even(3.5), 4)

        # In f16, 0.5 is exact: 0.5 * 65535.0 = 32767.5 -> nearest even is 32768 (0x8000)
        self.assertEqual(f16_to_unorm16(POS_HALF_F16), 32768)

    def test_out_of_range_clamping(self) -> None:
        """Verify inputs outside [0.0, 1.0] are clamped to boundaries."""
        # +2.0 clamps to 1.0 -> 0xFFFF
        res_pos = oracle_v_cvt_pknorm_u16_f16(POS_TWO_F16, f16_to_bits(10.0))
        self.assertEqual(res_pos, 0xFFFFFFFF)

        # Negative numbers clamp to 0.0 -> 0x0000
        res_neg = oracle_v_cvt_pknorm_u16_f16(NEG_TWO_F16, f16_to_bits(-10.0))
        self.assertEqual(res_neg, 0x00000000)

    def test_nan_and_inf_handling(self) -> None:
        """Verify NaN converts to 0x0000 and infinities clamp to boundaries."""
        # NaNs convert to 0x0000
        res_nan = oracle_v_cvt_pknorm_u16_f16(CANONICAL_QNAN_F16, CANONICAL_SNAN_F16)
        self.assertEqual(res_nan, 0x00000000)

        # +inf clamps to 0xFFFF, -inf clamps to 0x0000
        res_inf = oracle_v_cvt_pknorm_u16_f16(POS_INF_F16, NEG_INF_F16)
        self.assertEqual(res_inf, 0x0000FFFF)

    def test_op_sel_operand_selection(self) -> None:
        """Verify op_sel selects low or high 16-bit half from 32-bit dwords."""
        # reg0: low = +1.0 (0x3C00), high = 0.0 (0x0000) -> 0x00003C00
        reg0 = (POS_ZERO_F16 << 16) | POS_ONE_F16
        # reg1: low = 0.0 (0x0000), high = +0.5 (0x3800) -> 0x38000000
        reg1 = (POS_HALF_F16 << 16) | POS_ZERO_F16

        # op_sel=[0, 0]: select low from reg0 (+1.0) and low from reg1 (0.0) -> 0x0000FFFF
        res00 = oracle_v_cvt_pknorm_u16_f16(reg0, reg1, op_sel=(0, 0))
        self.assertEqual(res00, 0x0000FFFF)

        # op_sel=[1, 1]: select high from reg0 (0.0) and high from reg1 (+0.5) -> 0x80000000
        res11 = oracle_v_cvt_pknorm_u16_f16(reg0, reg1, op_sel=(1, 1))
        self.assertEqual(res11, 0x80000000)

        # op_sel=[0, 1]: select low from reg0 (+1.0) and high from reg1 (+0.5) -> 0x8000FFFF
        res01 = oracle_v_cvt_pknorm_u16_f16(reg0, reg1, op_sel=(0, 1))
        self.assertEqual(res01, 0x8000FFFF)

    def test_input_modifiers(self) -> None:
        """Verify abs and neg modifiers applied per channel."""
        # abs on negative input: abs(-1.0) = +1.0 -> 0xFFFF
        res_abs = oracle_v_cvt_pknorm_u16_f16(NEG_ONE_F16, NEG_HALF_F16, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0x8000FFFF)

        # neg on positive input: neg(+1.0) = -1.0 -> clamps to 0x0000
        res_neg = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_HALF_F16, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0x00000000)

        # abs+neg guarantees negative -> clamps to 0x0000
        res_absneg = oracle_v_cvt_pknorm_u16_f16(NEG_ONE_F16, POS_HALF_F16, abs0=True, neg0=True, abs1=True, neg1=True)
        self.assertEqual(res_absneg, 0x00000000)

    def test_unpack_and_roundtrip(self) -> None:
        """Verify unpacking and precision round-trip within 1 ULP of UNORM16."""
        for val in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
            f_bits = f16_to_bits(val)
            packed = oracle_v_cvt_pknorm_u16_f16(f_bits, f_bits)
            ch0, ch1 = unpack_pknorm_u16(packed)
            self.assertEqual(ch0, ch1)
            reconstructed = unorm16_to_f16(ch0)
            self.assertAlmostEqual(val, reconstructed, delta=1.0 / 65535.0)


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
    """Represents a 32-bit literal or inline constant."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= 0xFFFFFFFF):
            raise ValueError(f"Literal32 bits 0x{self.bits:X} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def from_f16_pair(cls, lo: float, hi: float) -> Literal32:
        return cls(((f16_to_bits(hi) & MASK16) << 16) | (f16_to_bits(lo) & MASK16))

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
    V_CVT_PKNORM_U16_F16 = "v_cvt_pknorm_u16_f16"


@dataclass(frozen=True)
class VCvtPknormU16F16Instruction:
    """Symbolic AST node representing a RDNA2 v_cvt_pknorm_u16_f16 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    op_sel: Tuple[int, int] = (0, 0)
    opcode: Opcode = Opcode.V_CVT_PKNORM_U16_F16

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if not isinstance(self.src0, VOP3Operand):
            raise TypeError(f"src0 must be VOP3Operand, got {type(self.src0).__name__}")
        if not isinstance(self.src1, VOP3Operand):
            raise TypeError(f"src1 must be VOP3Operand, got {type(self.src1).__name__}")
        if self.op_sel[0] not in (0, 1) or self.op_sel[1] not in (0, 1):
            raise ValueError(f"op_sel values must be 0 or 1, got {self.op_sel}")

    def disassemble(self) -> str:
        s = f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.op_sel != (0, 0):
            s += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]}]"
        return s


class TestValuCvtPknormU16F16SymbolicAST(unittest.TestCase):
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
        """Verify 32-bit literal constants creation from integers and float pairs."""
        lit_hex = Literal32(0x3C003C00)
        lit_pair = Literal32.from_f16_pair(1.0, 1.0)
        self.assertEqual(lit_hex.bits, 0x3C003C00)
        self.assertEqual(lit_pair.bits, 0x3C003C00)
        self.assertEqual(lit_hex.name, "0x3C003C00")

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
        """Verify complete instruction disassembly with and without op_sel."""
        dst = Reg(5, RegType.VGPR)
        s0 = VOP3Operand(Reg(1, RegType.VGPR))
        s1 = VOP3Operand(Reg(2, RegType.SGPR), neg_mod=True)

        instr = VCvtPknormU16F16Instruction(dst=dst, src0=s0, src1=s1)
        self.assertEqual(instr.disassemble(), "v_cvt_pknorm_u16_f16 v5, v1, -s2")

        instr_opsel = VCvtPknormU16F16Instruction(dst=dst, src0=s0, src1=s1, op_sel=(1, 0))
        self.assertEqual(instr_opsel.disassemble(), "v_cvt_pknorm_u16_f16 v5, v1, -s2 op_sel:[1,0]")

    def test_ast_immutability_and_equality(self) -> None:
        """Verify frozen AST dataclasses ensure immutability and valid hashing."""
        v0 = Reg(0, RegType.VGPR)
        v1 = Reg(1, RegType.VGPR)
        instr1 = VCvtPknormU16F16Instruction(v0, VOP3Operand(v1), VOP3Operand(v1))
        instr2 = VCvtPknormU16F16Instruction(v0, VOP3Operand(v1), VOP3Operand(v1))

        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))
        with self.assertRaises((dataclasses.FrozenInstanceError, TypeError)):
            instr1.op_sel = (1, 1)  # type: ignore

    def test_ast_rejection_sgpr_destination(self) -> None:
        """Verify VALU instructions reject SGPR destinations."""
        s0 = Reg(0, RegType.SGPR)
        v1 = Reg(1, RegType.VGPR)
        with self.assertRaises(TypeError):
            VCvtPknormU16F16Instruction(dst=s0, src0=VOP3Operand(v1), src1=VOP3Operand(v1))

    def test_ast_rejection_invalid_op_sel(self) -> None:
        """Verify invalid op_sel values are rejected."""
        v0 = Reg(0, RegType.VGPR)
        v1 = Reg(1, RegType.VGPR)
        with self.assertRaises(ValueError):
            VCvtPknormU16F16Instruction(v0, VOP3Operand(v1), VOP3Operand(v1), op_sel=(2, 0))
        with self.assertRaises(ValueError):
            VCvtPknormU16F16Instruction(v0, VOP3Operand(v1), VOP3Operand(v1), op_sel=(0, -1))

    def test_operand_type_rejection(self) -> None:
        """Verify invalid operand types raise TypeError."""
        with self.assertRaises(TypeError):
            VOP3Operand("v1")  # type: ignore
        with self.assertRaises(TypeError):
            VCvtPknormU16F16Instruction(Reg(0, RegType.VGPR), "bad", VOP3Operand(Reg(1, RegType.VGPR)))  # type: ignore


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
    """Deterministic scalar instruction interpreter for v_cvt_pknorm_u16_f16."""

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

    def step(self, instr: VCvtPknormU16F16Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase via oracle
        result = oracle_v_cvt_pknorm_u16_f16(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            op_sel=instr.op_sel,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
        )

        # Writeback phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestValuCvtPknormU16F16ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_vgpr_sources_execution(self) -> None:
        """Verify execution with VGPR sources."""
        self.state.set_vgpr(1, POS_ONE_F16)
        self.state.set_vgpr(2, POS_HALF_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x8000FFFF)
        self.assertEqual(self.state.get_vgpr(0), 0x8000FFFF)

    def test_sgpr_sources_execution(self) -> None:
        """Verify execution reading from scalar register file."""
        self.state.set_sgpr(10, POS_HALF_F16)
        self.state.set_sgpr(11, POS_ZERO_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.SGPR)),
            src1=VOP3Operand(Reg(11, RegType.SGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x00008000)
        self.assertEqual(self.state.get_vgpr(3), 0x00008000)
        # Verify SGPRs unchanged
        self.assertEqual(self.state.get_sgpr(10), POS_HALF_F16)
        self.assertEqual(self.state.get_sgpr(11), POS_ZERO_F16)

    def test_literal_sources_execution(self) -> None:
        """Verify execution with Literal32 inline constants."""
        lit = Literal32.from_f16_pair(1.0, 0.5)
        instr = VCvtPknormU16F16Instruction(
            dst=Reg(4, RegType.VGPR),
            src0=VOP3Operand(lit),
            src1=VOP3Operand(Literal32(POS_ZERO_F16)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x0000FFFF)
        self.assertEqual(self.state.get_vgpr(4), 0x0000FFFF)

    def test_op_sel_in_interpreter(self) -> None:
        """Verify op_sel operand half selection evaluated in interpreter."""
        # v1: low = 0.0 (0x0000), high = 1.0 (0x3C00) -> 0x3C000000
        # v2: low = 0.5 (0x3800), high = 0.0 (0x0000) -> 0x00003800
        self.state.set_vgpr(1, (POS_ONE_F16 << 16) | POS_ZERO_F16)
        self.state.set_vgpr(2, (POS_ZERO_F16 << 16) | POS_HALF_F16)

        # Select high half from v1 (1.0) and low half from v2 (0.5)
        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            op_sel=(1, 0),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x8000FFFF)
        self.assertEqual(self.state.get_vgpr(0), 0x8000FFFF)

    def test_input_modifiers_in_interpreter(self) -> None:
        """Verify abs and neg input modifiers evaluated in interpreter."""
        self.state.set_vgpr(1, NEG_ONE_F16)
        self.state.set_vgpr(2, POS_HALF_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR), abs_mod=True),
            src1=VOP3Operand(Reg(2, RegType.VGPR), neg_mod=True),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x0000FFFF)
        self.assertEqual(self.state.get_vgpr(0), 0x0000FFFF)

    def test_register_state_isolation(self) -> None:
        """Verify only destination VGPR is written and all other registers remain unchanged."""
        for i in range(256):
            self.state.set_vgpr(i, 0x11111111 * (i % 7 + 1))
            self.state.set_sgpr(i, 0x22222222 * (i % 5 + 1))

        self.state.set_vgpr(10, POS_ONE_F16)
        self.state.set_vgpr(20, POS_HALF_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(30, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(20, RegType.VGPR)),
        )

        self.interp.step(instr, self.state)
        self.assertEqual(self.state.get_vgpr(30), 0x8000FFFF)

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

    def step(self, instr: VCvtPknormU16F16Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_cvt_pknorm_u16_f16(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    op_sel=instr.op_sel,
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                    abs1=instr.src1.abs_mod,
                    neg1=instr.src1.neg_mod,
                )
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestValuCvtPknormU16F16SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and Wave32/Wave64 EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_all_active(self) -> None:
        """Verify full 32-lane vector computation with unique values per lane."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            val0 = lane / 31.0
            val1 = 1.0 - (lane / 31.0)
            state.set_vgpr(lane, 1, f16_to_bits(val0))
            state.set_vgpr(lane, 2, f16_to_bits(val1))

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            expected = oracle_v_cvt_pknorm_u16_f16(state.get_vgpr(lane, 1), state.get_vgpr(lane, 2))
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_all_active(self) -> None:
        """Verify full 64-lane vector execution in Wave64 mode."""
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        for lane in range(64):
            val0 = max(0.0, math.sin(lane * 0.1))
            val1 = max(0.0, math.cos(lane * 0.1))
            state.set_vgpr(lane, 1, f16_to_bits(val0))
            state.set_vgpr(lane, 2, f16_to_bits(val1))

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = oracle_v_cvt_pknorm_u16_f16(state.get_vgpr(lane, 1), state.get_vgpr(lane, 2))
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave32_even_odd_divergence(self) -> None:
        """Verify even lanes active and odd lanes masked out preserve state."""
        even_mask = 0x55555555  # bits 0, 2, 4, ...
        state = SIMDState(num_lanes=32, exec_mask=even_mask)
        initial_sentinel = 0xDEADBEEF

        for lane in range(32):
            state.set_vgpr(lane, 0, initial_sentinel)
            state.set_vgpr(lane, 1, POS_ONE_F16)
            state.set_vgpr(lane, 2, POS_HALF_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x8000FFFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0x8000FFFF)
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
            state.set_vgpr(lane, 1, POS_HALF_F16)
            state.set_vgpr(lane, 2, POS_HALF_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], 0x80008000)
                self.assertEqual(state.get_vgpr(lane, 0), 0x80008000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_exec_all_zero_mask(self) -> None:
        """Verify EXEC=0 skips execution across all lanes entirely."""
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        sentinel = 0x12345678

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, POS_ONE_F16)
            state.set_vgpr(lane, 2, POS_ONE_F16)

        instr = VCvtPknormU16F16Instruction(
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
        lane_target = 13
        state = SIMDState(num_lanes=32, exec_mask=(1 << lane_target))
        sentinel = 0xCAFEBABE

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, POS_ONE_F16)
            state.set_vgpr(lane, 2, POS_ONE_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane == lane_target:
                self.assertEqual(results[lane], 0xFFFFFFFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0xFFFFFFFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_shared_sgpr_broadcast_simd(self) -> None:
        """Verify uniform SGPR operand is correctly broadcast across divergent lanes."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        state.set_sgpr(5, POS_ONE_F16)  # Shared +1.0f16

        for lane in range(32):
            val0 = 0.0 if lane < 16 else 0.5
            state.set_vgpr(lane, 1, f16_to_bits(val0))

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(5, RegType.SGPR)),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            ch0_expected = 0x0000 if lane < 16 else 0x8000
            expected = (0xFFFF << 16) | ch0_expected
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)


# ==============================================================================
# Commit 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuCvtPknormU16F16RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0(self) -> None:
        """Verify dst == src0 reads original src0 before destination overwrite."""
        state = ScalarMachineState()
        state.set_vgpr(0, POS_ONE_F16)   # src0 initial: +1.0
        state.set_vgpr(1, POS_ZERO_F16)  # src1: 0.0

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x0000FFFF)
        self.assertEqual(state.get_vgpr(0), 0x0000FFFF)
        self.assertEqual(state.get_vgpr(1), POS_ZERO_F16)  # src1 untouched

    def test_dst_alias_src1(self) -> None:
        """Verify dst == src1 reads original src1 before destination overwrite."""
        state = ScalarMachineState()
        state.set_vgpr(1, POS_ONE_F16)   # src0: +1.0
        state.set_vgpr(0, POS_ZERO_F16)  # src1 initial: 0.0

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x0000FFFF)
        self.assertEqual(state.get_vgpr(0), 0x0000FFFF)
        self.assertEqual(state.get_vgpr(1), POS_ONE_F16)  # src0 untouched

    def test_full_self_aliasing(self) -> None:
        """Verify dst == src0 == src1 correctly converts and duplicates channels."""
        state = ScalarMachineState()
        state.set_vgpr(7, POS_HALF_F16)  # +0.5f16

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(7, RegType.VGPR),
            src0=VOP3Operand(Reg(7, RegType.VGPR)),
            src1=VOP3Operand(Reg(7, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        # Both channels convert +0.5 -> 0x8000
        self.assertEqual(res, 0x80008000)
        self.assertEqual(state.get_vgpr(7), 0x80008000)

    def test_src0_alias_src1_distinct_dst(self) -> None:
        """Verify src0 == src1 with distinct dst preserves source register."""
        state = ScalarMachineState()
        state.set_vgpr(1, POS_ONE_F16)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(1, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(state.get_vgpr(0), 0xFFFFFFFF)
        self.assertEqual(state.get_vgpr(1), POS_ONE_F16)  # Source untouched

    def test_aliased_operand_with_op_sel(self) -> None:
        """Verify reading both halves of a single packed register with op_sel."""
        state = ScalarMachineState()
        # reg 5 contains: low = 1.0 (0x3C00), high = 0.5 (0x3800) -> 0x38003C00
        state.set_vgpr(5, (POS_HALF_F16 << 16) | POS_ONE_F16)

        # src0 takes low half (1.0), src1 takes high half (0.5), overwriting reg 5
        instr = VCvtPknormU16F16Instruction(
            dst=Reg(5, RegType.VGPR),
            src0=VOP3Operand(Reg(5, RegType.VGPR)),
            src1=VOP3Operand(Reg(5, RegType.VGPR)),
            op_sel=(0, 1),
        )

        res = self.scalar_interp.step(instr, state)
        # ch0 = 1.0 -> 0xFFFF, ch1 = 0.5 -> 0x8000
        self.assertEqual(res, 0x8000FFFF)
        self.assertEqual(state.get_vgpr(5), 0x8000FFFF)

    def test_aliased_operand_with_modifiers(self) -> None:
        """Verify self-aliased register with asymmetric channel modifiers."""
        state = ScalarMachineState()
        state.set_vgpr(2, NEG_ONE_F16)  # -1.0f16

        # src0: abs(-1.0) = +1.0 -> 0xFFFF
        # src1: plain -1.0 -> clamps to 0x0000
        instr = VCvtPknormU16F16Instruction(
            dst=Reg(2, RegType.VGPR),
            src0=VOP3Operand(Reg(2, RegType.VGPR), abs_mod=True),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 0x0000FFFF)
        self.assertEqual(state.get_vgpr(2), 0x0000FFFF)

    def test_simd_aliasing_with_inactive_lanes(self) -> None:
        """Verify SIMD self-aliasing correctly preserves inactive lanes."""
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # Low 16 lanes active
        initial_val = POS_ONE_F16

        for lane in range(32):
            state.set_vgpr(lane, 0, initial_val)

        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(0, RegType.VGPR)),
            src1=VOP3Operand(Reg(0, RegType.VGPR)),
        )

        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            if lane < 16:
                self.assertEqual(results[lane], 0xFFFFFFFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0xFFFFFFFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), initial_val)


# ==============================================================================
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class CvtPknormU16StatusFlags:
    """Architectural condition codes and status flags for packed UNORM16 conversion."""
    ch0_zero: bool
    ch1_zero: bool
    both_zero: bool
    ch0_saturated_max: bool
    ch1_saturated_max: bool
    both_saturated_max: bool
    ch0_clamped_min: bool
    ch1_clamped_min: bool
    ch0_clamped_max: bool
    ch1_clamped_max: bool
    overflow_or_clamped: bool
    nan_encountered: bool
    invalid_op: bool


def compute_status_flags(
    src0_bits: int,
    src1_bits: int,
    packed_result: int,
    op_sel: Tuple[int, int] = (0, 0),
) -> CvtPknormU16StatusFlags:
    """Compute architectural condition codes and floating-point exception flags."""
    h0 = select_f16_half(src0_bits, op_sel[0])
    h1 = select_f16_half(src1_bits, op_sel[1])

    s0_nan = is_nan_f16(h0)
    s1_nan = is_nan_f16(h1)
    s0_snan = is_snan_f16(h0)
    s1_snan = is_snan_f16(h1)

    ch0_u16, ch1_u16 = unpack_pknorm_u16(packed_result)

    ch0_zero = (ch0_u16 == 0)
    ch1_zero = (ch1_u16 == 0)
    both_zero = ((packed_result & MASK32) == 0)

    ch0_sat_max = (ch0_u16 == 65535)
    ch1_sat_max = (ch1_u16 == 65535)
    both_sat_max = ((packed_result & MASK32) == 0xFFFFFFFF)

    f0 = bits_to_f16(h0) if not s0_nan else 0.0
    f1 = bits_to_f16(h1) if not s1_nan else 0.0

    ch0_clamped_min = is_inf_f16(h0) and (f0 < 0) or (f0 <= 0.0 and not s0_nan)
    ch1_clamped_min = is_inf_f16(h1) and (f1 < 0) or (f1 <= 0.0 and not s1_nan)

    ch0_clamped_max = is_inf_f16(h0) and (f0 > 0) or (f0 >= 1.0 and not s0_nan)
    ch1_clamped_max = is_inf_f16(h1) and (f1 > 0) or (f1 >= 1.0 and not s1_nan)

    overflow_or_clamped = (
        is_inf_f16(h0) or is_inf_f16(h1) or
        (f0 < 0.0) or (f0 > 1.0) or (f1 < 0.0) or (f1 > 1.0)
    )

    return CvtPknormU16StatusFlags(
        ch0_zero=ch0_zero,
        ch1_zero=ch1_zero,
        both_zero=both_zero,
        ch0_saturated_max=ch0_sat_max,
        ch1_saturated_max=ch1_sat_max,
        both_saturated_max=both_sat_max,
        ch0_clamped_min=ch0_clamped_min,
        ch1_clamped_min=ch1_clamped_min,
        ch0_clamped_max=ch0_clamped_max,
        ch1_clamped_max=ch1_clamped_max,
        overflow_or_clamped=overflow_or_clamped,
        nan_encountered=s0_nan or s1_nan,
        invalid_op=s0_snan or s1_snan,
    )


class TestValuCvtPknormU16F16StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and exception detection."""

    def test_zero_flags(self) -> None:
        """Verify zero flags for zero inputs, non-zero inputs, and mixed channels."""
        # Both channels zero
        res_zero = oracle_v_cvt_pknorm_u16_f16(POS_ZERO_F16, NEG_ZERO_F16)
        flags_zero = compute_status_flags(POS_ZERO_F16, NEG_ZERO_F16, res_zero)
        self.assertTrue(flags_zero.ch0_zero)
        self.assertTrue(flags_zero.ch1_zero)
        self.assertTrue(flags_zero.both_zero)

        # Ch0 non-zero, Ch1 zero
        res_mixed = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_ZERO_F16)
        flags_mixed = compute_status_flags(POS_ONE_F16, POS_ZERO_F16, res_mixed)
        self.assertFalse(flags_mixed.ch0_zero)
        self.assertTrue(flags_mixed.ch1_zero)
        self.assertFalse(flags_mixed.both_zero)

        # Both non-zero
        res_nonzero = oracle_v_cvt_pknorm_u16_f16(POS_HALF_F16, POS_HALF_F16)
        flags_nonzero = compute_status_flags(POS_HALF_F16, POS_HALF_F16, res_nonzero)
        self.assertFalse(flags_nonzero.ch0_zero)
        self.assertFalse(flags_nonzero.ch1_zero)
        self.assertFalse(flags_nonzero.both_zero)

    def test_saturation_max_flags(self) -> None:
        """Verify saturation to 0xFFFF detected properly."""
        # Both saturated max
        res_max = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_ONE_F16)
        flags_max = compute_status_flags(POS_ONE_F16, POS_ONE_F16, res_max)
        self.assertTrue(flags_max.ch0_saturated_max)
        self.assertTrue(flags_max.ch1_saturated_max)
        self.assertTrue(flags_max.both_saturated_max)

        # Only ch0 saturated max
        res_half = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_HALF_F16)
        flags_half = compute_status_flags(POS_ONE_F16, POS_HALF_F16, res_half)
        self.assertTrue(flags_half.ch0_saturated_max)
        self.assertFalse(flags_half.ch1_saturated_max)
        self.assertFalse(flags_half.both_saturated_max)

    def test_clamped_bounds_flags(self) -> None:
        """Verify clamp flags for out-of-range inputs and infinities."""
        # Negative clamp (< 0.0)
        res_neg = oracle_v_cvt_pknorm_u16_f16(NEG_ONE_F16, POS_HALF_F16)
        flags_neg = compute_status_flags(NEG_ONE_F16, POS_HALF_F16, res_neg)
        self.assertTrue(flags_neg.overflow_or_clamped)
        self.assertTrue(flags_neg.ch0_clamped_min)
        self.assertFalse(flags_neg.ch1_clamped_min)

        # Positive clamp (> 1.0)
        res_pos = oracle_v_cvt_pknorm_u16_f16(POS_HALF_F16, POS_TWO_F16)
        flags_pos = compute_status_flags(POS_HALF_F16, POS_TWO_F16, res_pos)
        self.assertTrue(flags_pos.overflow_or_clamped)
        self.assertFalse(flags_pos.ch0_clamped_max)
        self.assertTrue(flags_pos.ch1_clamped_max)

        # Infinities clamp
        res_inf = oracle_v_cvt_pknorm_u16_f16(POS_INF_F16, NEG_INF_F16)
        flags_inf = compute_status_flags(POS_INF_F16, NEG_INF_F16, res_inf)
        self.assertTrue(flags_inf.overflow_or_clamped)
        self.assertTrue(flags_inf.ch0_clamped_max)
        self.assertTrue(flags_inf.ch1_clamped_min)

    def test_nan_and_invalid_op_flags(self) -> None:
        """Verify quiet NaN and signaling NaN exception detection."""
        # Quiet NaN
        res_qnan = oracle_v_cvt_pknorm_u16_f16(CANONICAL_QNAN_F16, POS_ONE_F16)
        flags_qnan = compute_status_flags(CANONICAL_QNAN_F16, POS_ONE_F16, res_qnan)
        self.assertTrue(flags_qnan.nan_encountered)
        self.assertFalse(flags_qnan.invalid_op)

        # Signaling NaN
        res_snan = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, CANONICAL_SNAN_F16)
        flags_snan = compute_status_flags(POS_ONE_F16, CANONICAL_SNAN_F16, res_snan)
        self.assertTrue(flags_snan.nan_encountered)
        self.assertTrue(flags_snan.invalid_op)

    def test_simd_condition_mask_generation(self) -> None:
        """Verify SIMD lane mask generation from per-lane status condition codes."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for lane in range(32):
            val = 0.0 if lane % 2 == 0 else 1.0
            state.set_vgpr(lane, 1, f16_to_bits(val))
            state.set_vgpr(lane, 2, POS_HALF_F16)

        interp = SIMDInterpreter()
        instr = VCvtPknormU16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
        )

        results = interp.step(instr, state)
        vcc_zero_mask = 0
        for lane in range(32):
            res = results[lane]
            assert res is not None
            flags = compute_status_flags(state.get_vgpr(lane, 1), state.get_vgpr(lane, 2), res)
            if flags.ch0_zero:
                vcc_zero_mask |= (1 << lane)

        # Even lanes had ch0=0.0 -> mask = 0x55555555
        self.assertEqual(vcc_zero_mask, 0x55555555)


# ==============================================================================
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestValuCvtPknormU16F16IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 binary16 boundary extremes, subnormals, infinities, and NaNs."""

    def test_signed_zeros(self) -> None:
        """Verify +0.0 and -0.0 both convert to 0x0000 regardless of modifiers."""
        # Unmodified
        self.assertEqual(f16_to_unorm16(POS_ZERO_F16), 0)
        self.assertEqual(f16_to_unorm16(NEG_ZERO_F16), 0)

        # Modifiers on zero
        res_abs = oracle_v_cvt_pknorm_u16_f16(NEG_ZERO_F16, NEG_ZERO_F16, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0x00000000)

        res_neg = oracle_v_cvt_pknorm_u16_f16(POS_ZERO_F16, POS_ZERO_F16, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0x00000000)

    def test_subnormal_numbers(self) -> None:
        """Verify behavior of positive and negative half-precision subnormals."""
        # Smallest positive subnormal: 2^-24 (5.96e-8) -> scales to 0.0039 -> 0
        self.assertEqual(f16_to_unorm16(MIN_SUBNORMAL_F16), 0)

        # Largest positive subnormal: (1023/1024) * 2^-14 (6.097e-5) -> scales to 3.996 -> 4
        self.assertEqual(f16_to_unorm16(MAX_SUBNORMAL_F16), 4)

        # Negative subnormals clamp to 0.0 -> 0x0000
        min_sub_neg = MIN_SUBNORMAL_F16 | SIGN_MASK_F16
        max_sub_neg = MAX_SUBNORMAL_F16 | SIGN_MASK_F16
        self.assertEqual(f16_to_unorm16(min_sub_neg), 0)
        self.assertEqual(f16_to_unorm16(max_sub_neg), 0)

        # Packed combination of subnormals
        res = oracle_v_cvt_pknorm_u16_f16(MIN_SUBNORMAL_F16, max_sub_neg)
        self.assertEqual(res, 0x00000000)

    def test_smallest_normal_float(self) -> None:
        """Verify smallest positive normal float 2^-14 converts to 4."""
        min_normal_f16 = 0x0400  # 2^-14
        min_normal_neg_f16 = 0x8400
        self.assertEqual(f16_to_unorm16(min_normal_f16), 4)
        self.assertEqual(f16_to_unorm16(min_normal_neg_f16), 0)  # negative clamps to 0

    def test_infinities(self) -> None:
        """Verify +inf and -inf clamp to 0xFFFF and 0x0000."""
        # +inf -> 0xFFFF
        self.assertEqual(f16_to_unorm16(POS_INF_F16), 0xFFFF)
        # -inf -> 0x0000
        self.assertEqual(f16_to_unorm16(NEG_INF_F16), 0x0000)

        # +inf with neg modifier -> -inf -> 0x0000
        res_neg = oracle_v_cvt_pknorm_u16_f16(POS_INF_F16, POS_INF_F16, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0x00000000)

        # -inf with abs modifier -> +inf -> 0xFFFF
        res_abs = oracle_v_cvt_pknorm_u16_f16(NEG_INF_F16, NEG_INF_F16, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0xFFFFFFFF)

    def test_nan_payload_variations(self) -> None:
        """Verify all varieties of Quiet and Signaling NaNs convert to 0x0000."""
        qnan_payloads = [
            0x7E01,  # Quiet NaN with payload 1
            0x7FFF,  # Quiet NaN with all payload bits set
            0xFE00,  # Negative Quiet NaN
            0xFFFF,  # Negative Quiet NaN with full payload
        ]
        snan_payloads = [
            0x7C01,  # Signaling NaN with bit 0
            0x7DFF,  # Signaling NaN with highest mantissa bit clear
            0xFC01,  # Negative Signaling NaN
            0xFDFF,  # Negative Signaling NaN with payload
        ]

        for payload in qnan_payloads + snan_payloads:
            self.assertTrue(is_nan_f16(payload))
            self.assertEqual(f16_to_unorm16(payload), 0)

        # Mixed NaN pairs in oracle
        res = oracle_v_cvt_pknorm_u16_f16(0x7E01, 0xFC01)
        self.assertEqual(res, 0x00000000)

    def test_max_finite_floats(self) -> None:
        """Verify maximum finite half-precision floats clamp to boundaries."""
        max_float = 0x7BFF       # 65504.0
        min_float_neg = 0xFBFF   # -65504.0

        self.assertEqual(f16_to_unorm16(max_float), 0xFFFF)
        self.assertEqual(f16_to_unorm16(min_float_neg), 0x0000)

        res = oracle_v_cvt_pknorm_u16_f16(max_float, min_float_neg)
        self.assertEqual(res, 0x0000FFFF)

    def test_unorm16_quantum_steps(self) -> None:
        """Verify smallest non-zero quantizations around 1.0 / 65535.0."""
        step = 1.0 / 65535.0
        step_pos = f16_to_bits(step)
        step_neg = f16_to_bits(-step)

        # 1.0 / 65535.0 rounds to 1 (0x0001)
        self.assertEqual(f16_to_unorm16(step_pos), 0x0001)
        # Negative step clamps to 0 (0x0000)
        self.assertEqual(f16_to_unorm16(step_neg), 0x0000)


if __name__ == "__main__":
    unittest.main()
