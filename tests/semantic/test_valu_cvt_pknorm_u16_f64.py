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


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclasses.dataclass
class ScalarMachineState:
    """Architectural register file state for a single lane."""
    vgprs: list[int] = dataclasses.field(default_factory=lambda: [0] * 256)
    sgprs: list[int] = dataclasses.field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32

    def get_vgpr_pair(self, index: int) -> int:
        return join_dwords(self.vgprs[index], self.vgprs[index + 1])

    def set_vgpr_pair(self, index: int, val64: int) -> None:
        lo, hi = split_dwords(val64)
        self.vgprs[index] = lo
        self.vgprs[index + 1] = hi

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32

    def get_sgpr_pair(self, index: int) -> int:
        return join_dwords(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr_pair(self, index: int, val64: int) -> None:
        lo, hi = split_dwords(val64)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_cvt_pknorm_u16_f64."""

    def resolve_operand_64(self, op: VOP3Operand64, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr_pair(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr_pair(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal64):
            return op.source.bits & MASK64
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VCvtPknormU16F64Instruction, state: ScalarMachineState) -> int:
        # Atomic operand fetch phase
        s0_raw = self.resolve_operand_64(instr.src0, state)
        s1_raw = self.resolve_operand_64(instr.src1, state)

        # Computation phase via architectural oracle
        result = oracle_v_cvt_pknorm_u16_f64(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            abs0=instr.src0.abs_mod,
            neg0=instr.src0.neg_mod,
            abs1=instr.src1.abs_mod,
            neg1=instr.src1.neg_mod,
        )

        # Atomic write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result

    def run_program(self, program: Program, state: ScalarMachineState) -> list[int]:
        results: list[int] = []
        for instr in program.instructions:
            results.append(self.step(instr, state))
        return results


class TestValuCvtPknormU16F64ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr_pair(2, POS_ONE_F64)
        self.state.set_vgpr_pair(4, HALF_F64)
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        res = self.interp.step(instr, self.state)
        # S0: 1.0 -> 0xFFFF; S1: 0.5 -> 32768
        expected = (32768 << 16) | 0xFFFF
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_step_with_sgpr_and_literal(self) -> None:
        self.state.set_sgpr_pair(2, QUARTER_F64)
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(1, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.SGPR, is_double=True)),
            src1=VOP3Operand64(Literal64.from_f64(1.0)),
        )
        res = self.interp.step(instr, self.state)
        # S0: 0.25 -> 16384; S1: 1.0 -> 0xFFFF
        expected = (0xFFFF << 16) | 16384
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(1), expected)

    def test_step_with_modifiers(self) -> None:
        self.state.set_vgpr_pair(2, NEG_QUARTER_F64)
        self.state.set_vgpr_pair(4, HALF_F64)
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True), abs_mod=True),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True), neg_mod=True),
        )
        res = self.interp.step(instr, self.state)
        # S0: |-0.25| = 0.25 -> 16384; S1: -0.5 -> 0
        expected = (0 << 16) | 16384
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_full_32bit_destination_overwrite(self) -> None:
        self.state.set_vgpr(7, 0xDEADBEEF)
        self.state.set_vgpr_pair(2, POS_ZERO_F64)
        self.state.set_vgpr_pair(4, POS_ZERO_F64)
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(7, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(self.state.get_vgpr(7), 0x00000000)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr_pair(2, HALF_F64)
        self.state.set_vgpr_pair(4, QUARTER_F64)
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(6, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)

    def test_run_program_sequential(self) -> None:
        prog = Program()
        self.state.set_vgpr_pair(2, POS_ONE_F64)
        self.state.set_vgpr_pair(4, HALF_F64)
        prog.append(VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        ))
        prog.append(VCvtPknormU16F64Instruction(
            dst=Reg(1, RegType.VGPR),
            src0=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        ))
        results = self.interp.run_program(prog, self.state)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], (32768 << 16) | 0xFFFF)
        self.assertEqual(results[1], (0xFFFF << 16) | 32768)


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclasses.dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
    num_lanes: int
    exec_mask: int | None = None
    vgprs: list[list[int]] = dataclasses.field(default_factory=list)
    sgprs: list[int] = dataclasses.field(default_factory=lambda: [0] * 256)

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

    def get_vgpr_pair(self, lane: int, index: int) -> int:
        return join_dwords(self.vgprs[lane][index], self.vgprs[lane][index + 1])

    def set_vgpr_pair(self, lane: int, index: int, val64: int) -> None:
        lo, hi = split_dwords(val64)
        self.vgprs[lane][index] = lo
        self.vgprs[lane][index + 1] = hi

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32

    def get_sgpr_pair(self, index: int) -> int:
        return join_dwords(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr_pair(self, index: int, val64: int) -> None:
        lo, hi = split_dwords(val64)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand_64(self, op: VOP3Operand64, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr_pair(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr_pair(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal64):
            return op.source.bits & MASK64
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VCvtPknormU16F64Instruction, state: SIMDState) -> list[int | None]:
        results: list[int | None] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        s0_vals = [
            self.resolve_lane_operand_64(instr.src0, state, lane)
            for lane in range(state.num_lanes)
        ]
        s1_vals = [
            self.resolve_lane_operand_64(instr.src1, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution and write-back phase governed by EXEC mask
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_cvt_pknorm_u16_f64(
                    src0_bits=s0_vals[lane],
                    src1_bits=s1_vals[lane],
                    abs0=instr.src0.abs_mod,
                    neg0=instr.src0.neg_mod,
                    abs1=instr.src1.abs_mod,
                    neg1=instr.src1.neg_mod,
                )
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestValuCvtPknormU16F64SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD: int = 0xCAFEBABE

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr_pair(lane, 2, f64_to_bits(lane / 32.0))
            state.set_vgpr_pair(lane, 4, POS_ONE_F64)

        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected_lo = oracle_cvt_unorm16_f64(f64_to_bits(lane / 32.0))
            expected = (0xFFFF << 16) | expected_lo
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr_pair(lane, 2, HALF_F64)
            state.set_vgpr_pair(lane, 4, QUARTER_F64)

        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        results = self.simd_interp.step(instr, state)
        expected = (16384 << 16) | 32768
        for lane in range(64):
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr_pair(lane, 2, POS_ONE_F64)
            state.set_vgpr_pair(lane, 4, POS_ONE_F64)

        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                self.assertEqual(results[lane], 0xFFFFFFFF)
                self.assertEqual(state.get_vgpr(lane, 0), 0xFFFFFFFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_split_halves_masking(self) -> None:
        mask = 0x0000_0000_FFFF_FFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr_pair(lane, 2, HALF_F64)
            state.set_vgpr_pair(lane, 4, POS_ZERO_F64)

        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        self.simd_interp.step(instr, state)
        expected = (0 << 16) | 32768
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), expected)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_active_lane(self) -> None:
        mask = 1 << 61
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr_pair(lane, 2, POS_ONE_F64)
            state.set_vgpr_pair(lane, 4, HALF_F64)

        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == 61:
                self.assertEqual(state.get_vgpr(lane, 0), (32768 << 16) | 0xFFFF)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_zero_exec_mask_preserves_all(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)

        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Literal64.from_f64(1.0)),
            src1=VOP3Operand64(Literal64.from_f64(1.0)),
        )
        results = self.simd_interp.step(instr, state)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_uniform_scalar_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr_pair(8, POS_ONE_F64)
        for lane in range(32):
            state.set_vgpr_pair(lane, 2, f64_to_bits(lane / 32.0))

        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(8, RegType.SGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            expected_hi = oracle_cvt_unorm16_f64(f64_to_bits(lane / 32.0))
            expected = (expected_hi << 16) | 0xFFFF
            self.assertEqual(state.get_vgpr(lane, 0), expected)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuCvtPknormU16F64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src0_low_dword(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr_pair(0, POS_ONE_F64)  # v[0:1] = 1.0
        state.set_vgpr_pair(2, HALF_F64)     # v[2:3] = 0.5
        # v0 = cvt_pknorm(v[0:1], v[2:3]) -> [0xFFFF, 32768]
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(0, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        )
        old_v1 = state.get_vgpr(1)
        res = self.scalar_interp.step(instr, state)
        expected = (32768 << 16) | 0xFFFF
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)
        # v1 (high dword of S0) must remain untouched
        self.assertEqual(state.get_vgpr(1), old_v1)
        # v[2:3] untouched
        self.assertEqual(state.get_vgpr_pair(2), HALF_F64)

    def test_dst_alias_src0_high_dword(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr_pair(0, POS_ONE_F64)  # v[0:1] = 1.0
        state.set_vgpr_pair(2, HALF_F64)     # v[2:3] = 0.5
        # v1 = cvt_pknorm(v[0:1], v[2:3]) -> [0xFFFF, 32768]
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(1, RegType.VGPR),
            src0=VOP3Operand64(Reg(0, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        )
        old_v0 = state.get_vgpr(0)
        res = self.scalar_interp.step(instr, state)
        expected = (32768 << 16) | 0xFFFF
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(1), expected)
        # v0 must remain untouched
        self.assertEqual(state.get_vgpr(0), old_v0)

    def test_dst_alias_src1_low_dword(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr_pair(0, HALF_F64)     # v[0:1] = 0.5
        state.set_vgpr_pair(2, POS_ONE_F64)  # v[2:3] = 1.0
        # v2 = cvt_pknorm(v[0:1], v[2:3])
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(2, RegType.VGPR),
            src0=VOP3Operand64(Reg(0, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        )
        old_v3 = state.get_vgpr(3)
        res = self.scalar_interp.step(instr, state)
        expected = (0xFFFF << 16) | 32768
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(2), expected)
        self.assertEqual(state.get_vgpr(3), old_v3)

    def test_dst_alias_src1_high_dword(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr_pair(0, HALF_F64)     # v[0:1] = 0.5
        state.set_vgpr_pair(2, POS_ONE_F64)  # v[2:3] = 1.0
        # v3 = cvt_pknorm(v[0:1], v[2:3])
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3Operand64(Reg(0, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        )
        old_v2 = state.get_vgpr(2)
        res = self.scalar_interp.step(instr, state)
        expected = (0xFFFF << 16) | 32768
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(3), expected)
        self.assertEqual(state.get_vgpr(2), old_v2)

    def test_src0_src1_same_register_pair(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr_pair(4, QUARTER_F64)  # 0.25 -> 16384
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        res = self.scalar_interp.step(instr, state)
        expected = (16384 << 16) | 16384
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)

    def test_chained_dependency_chain(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr_pair(0, HALF_F64)     # 0.5
        state.set_vgpr_pair(2, QUARTER_F64)  # 0.25
        state.set_vgpr_pair(4, POS_ONE_F64)  # 1.0

        # Instr 1: v0 = cvt_pknorm(v[0:1], v[2:3]) -> (16384 << 16) | 32768 = 0x40008000
        instr1 = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(0, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        )
        res1 = self.scalar_interp.step(instr1, state)
        self.assertEqual(res1, 0x40008000)

        # Now v[0:1] has low dword 0x40008000 and high dword of original 0.5 (0x3FE00000)
        # val64 = join_dwords(0x40008000, 0x3FE00000) = 0x3FE0000040008000 ~= 0.5000001 > 0
        instr2 = VCvtPknormU16F64Instruction(
            dst=Reg(6, RegType.VGPR),
            src0=VOP3Operand64(Reg(0, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(4, RegType.VGPR, is_double=True)),
        )
        res2 = self.scalar_interp.step(instr2, state)
        self.assertEqual(extract_u16(res2, high_half=True), 0xFFFF)

    def test_simd_multi_lane_aliased_inplace(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr_pair(lane, 0, POS_ONE_F64)
            state.set_vgpr_pair(lane, 2, HALF_F64)

        # Overwrite v0 in-place across all lanes
        instr = VCvtPknormU16F64Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand64(Reg(0, RegType.VGPR, is_double=True)),
            src1=VOP3Operand64(Reg(2, RegType.VGPR, is_double=True)),
        )
        self.simd_interp.step(instr, state)
        expected = (32768 << 16) | 0xFFFF
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), expected)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class CvtPknormStatusFlags:
    """Status flags and architectural condition codes for packed unorm16 channels."""
    ch0_zero: bool
    ch1_zero: bool
    both_zero: bool
    ch0_saturated: bool
    ch1_saturated: bool
    both_saturated: bool
    ch0_nan_input: bool
    ch1_nan_input: bool


def evaluate_cvt_pknorm_flags(s0_bits: int, s1_bits: int, dst32: int) -> CvtPknormStatusFlags:
    """Evaluate architectural condition codes and status flags for v_cvt_pknorm_u16_f64."""
    h0 = dst32 & MASK16
    h1 = (dst32 >> 16) & MASK16

    ch0_z = (h0 == UNORM16_MIN)
    ch1_z = (h1 == UNORM16_MIN)
    ch0_sat = (h0 == UNORM16_MAX)
    ch1_sat = (h1 == UNORM16_MAX)

    return CvtPknormStatusFlags(
        ch0_zero=ch0_z,
        ch1_zero=ch1_z,
        both_zero=ch0_z and ch1_z,
        ch0_saturated=ch0_sat,
        ch1_saturated=ch1_sat,
        both_saturated=ch0_sat and ch1_sat,
        ch0_nan_input=is_nan64(s0_bits),
        ch1_nan_input=is_nan64(s1_bits),
    )


def compute_pknorm_half_carry(val0: int, val1: int) -> tuple[int, int]:
    """Compute 16-bit unsigned carry out from half-channel addition."""
    total = (val0 & MASK16) + (val1 & MASK16)
    return total & MASK16, (total >> 16) & 1


def compute_pknorm_half_borrow(val0: int, val1: int) -> tuple[int, int]:
    """Compute 16-bit unsigned borrow out from half-channel subtraction (val0 - val1)."""
    diff = (val0 & MASK16) - (val1 & MASK16)
    borrow = 1 if diff < 0 else 0
    return diff & MASK16, borrow


def compute_packed_u16_add(dst32_a: int, dst32_b: int) -> tuple[int, tuple[int, int]]:
    """Perform SIMD packed 16-bit unsigned addition with per-channel carry generation."""
    h0_a, h1_a = dst32_a & MASK16, (dst32_a >> 16) & MASK16
    h0_b, h1_b = dst32_b & MASK16, (dst32_b >> 16) & MASK16

    s0, c0 = compute_pknorm_half_carry(h0_a, h0_b)
    s1, c1 = compute_pknorm_half_carry(h1_a, h1_b)
    return pack_u16_to_u32(s0, s1), (c0, c1)


def compute_packed_u16_sub(dst32_a: int, dst32_b: int) -> tuple[int, tuple[int, int]]:
    """Perform SIMD packed 16-bit unsigned subtraction with per-channel borrow generation."""
    h0_a, h1_a = dst32_a & MASK16, (dst32_a >> 16) & MASK16
    h0_b, h1_b = dst32_b & MASK16, (dst32_b >> 16) & MASK16

    d0, b0 = compute_pknorm_half_borrow(h0_a, h0_b)
    d1, b1 = compute_pknorm_half_borrow(h1_a, h1_b)
    return pack_u16_to_u32(d0, d1), (b0, b1)


class TestValuCvtPknormU16F64ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_zero_flags(self) -> None:
        dst_both_z = oracle_v_cvt_pknorm_u16_f64(POS_ZERO_F64, NEG_ZERO_F64)
        flags_both = evaluate_cvt_pknorm_flags(POS_ZERO_F64, NEG_ZERO_F64, dst_both_z)
        self.assertTrue(flags_both.ch0_zero)
        self.assertTrue(flags_both.ch1_zero)
        self.assertTrue(flags_both.both_zero)
        self.assertFalse(flags_both.ch0_saturated)
        self.assertFalse(flags_both.ch1_saturated)

        dst_ch1_z = oracle_v_cvt_pknorm_u16_f64(POS_ONE_F64, POS_ZERO_F64)
        flags_ch1_z = evaluate_cvt_pknorm_flags(POS_ONE_F64, POS_ZERO_F64, dst_ch1_z)
        self.assertFalse(flags_ch1_z.ch0_zero)
        self.assertTrue(flags_ch1_z.ch1_zero)
        self.assertFalse(flags_ch1_z.both_zero)

    def test_saturation_flags(self) -> None:
        dst_both_sat = oracle_v_cvt_pknorm_u16_f64(POS_ONE_F64, POS_TWO_F64)
        flags_both = evaluate_cvt_pknorm_flags(POS_ONE_F64, POS_TWO_F64, dst_both_sat)
        self.assertTrue(flags_both.ch0_saturated)
        self.assertTrue(flags_both.ch1_saturated)
        self.assertTrue(flags_both.both_saturated)

        dst_mixed = oracle_v_cvt_pknorm_u16_f64(POS_ONE_F64, HALF_F64)
        flags_mixed = evaluate_cvt_pknorm_flags(POS_ONE_F64, HALF_F64, dst_mixed)
        self.assertTrue(flags_mixed.ch0_saturated)
        self.assertFalse(flags_mixed.ch1_saturated)
        self.assertFalse(flags_mixed.both_saturated)

    def test_nan_input_status_flags(self) -> None:
        dst = oracle_v_cvt_pknorm_u16_f64(POS_QNAN_F64, HALF_F64)
        flags = evaluate_cvt_pknorm_flags(POS_QNAN_F64, HALF_F64, dst)
        self.assertTrue(flags.ch0_nan_input)
        self.assertFalse(flags.ch1_nan_input)
        self.assertTrue(flags.ch0_zero)
        self.assertFalse(flags.ch1_zero)

        dst_snan = oracle_v_cvt_pknorm_u16_f64(POS_ONE_F64, NEG_SNAN_F64)
        flags_snan = evaluate_cvt_pknorm_flags(POS_ONE_F64, NEG_SNAN_F64, dst_snan)
        self.assertFalse(flags_snan.ch0_nan_input)
        self.assertTrue(flags_snan.ch1_nan_input)
        self.assertTrue(flags_snan.ch1_zero)

    def test_carry_generation_on_packed_channels(self) -> None:
        # Summing 0x8000 + 0x8000 = 0x10000 -> carry = 1, sum = 0
        s0, c0 = compute_pknorm_half_carry(0x8000, 0x8000)
        self.assertEqual(s0, 0)
        self.assertEqual(c0, 1)

        # 0x1234 + 0x2345 = 0x3579 -> carry = 0
        s1, c1 = compute_pknorm_half_carry(0x1234, 0x2345)
        self.assertEqual(s1, 0x3579)
        self.assertEqual(c1, 0)

        # Packed SIMD add
        w_a = pack_u16_to_u32(0xFFFF, 0x1000)
        w_b = pack_u16_to_u32(0x0001, 0x2000)
        sum_packed, (car0, car1) = compute_packed_u16_add(w_a, w_b)
        self.assertEqual(sum_packed, pack_u16_to_u32(0x0000, 0x3000))
        self.assertEqual(car0, 1)
        self.assertEqual(car1, 0)

    def test_borrow_generation_on_packed_channels(self) -> None:
        # 0x0000 - 0x0001 -> diff = 0xFFFF, borrow = 1
        d0, b0 = compute_pknorm_half_borrow(0x0000, 0x0001)
        self.assertEqual(d0, 0xFFFF)
        self.assertEqual(b0, 1)

        # 0x8000 - 0x4000 = 0x4000 -> borrow = 0
        d1, b1 = compute_pknorm_half_borrow(0x8000, 0x4000)
        self.assertEqual(d1, 0x4000)
        self.assertEqual(b1, 0)

        # Packed SIMD sub
        w_a = pack_u16_to_u32(0x0002, 0x5000)
        w_b = pack_u16_to_u32(0x0005, 0x2000)
        diff_packed, (bor0, bor1) = compute_packed_u16_sub(w_a, w_b)
        self.assertEqual(diff_packed, pack_u16_to_u32(0xFFFD, 0x3000))
        self.assertEqual(bor0, 1)
        self.assertEqual(bor1, 0)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestValuCvtPknormU16F64IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormals_evaluation(self) -> None:
        # Positive subnormals in f64 (< 2^-1022) scale to << 0.5 -> round to 0
        self.assertEqual(oracle_cvt_unorm16_f64(SUBNORM_MIN_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(SUBNORM_MAX_F64), 0)
        self.assertTrue(is_subnormal64(SUBNORM_MIN_F64))
        self.assertTrue(is_subnormal64(SUBNORM_MAX_F64))

        # Negative subnormals clamp to 0
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_SUBNORM_MIN_F64), 0)
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_SUBNORM_MAX_F64), 0)

        # Packed combination of subnormals
        res = oracle_v_cvt_pknorm_u16_f64(SUBNORM_MIN_F64, SUBNORM_MAX_F64)
        self.assertEqual(res, 0x00000000)

    def test_all_nan_variants_evaluate_to_zero(self) -> None:
        nan_cases = [
            POS_QNAN_F64,
            NEG_QNAN_F64,
            POS_SNAN_F64,
            NEG_SNAN_F64,
            0x7FF8000000000001,
            0x7FFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF,
            0x7FFA000000000000,
            0xFFF0000000000002,
        ]
        for nan_word in nan_cases:
            with self.subTest(nan_pattern=hex(nan_word)):
                self.assertTrue(is_nan64(nan_word))
                self.assertEqual(oracle_cvt_unorm16_f64(nan_word), 0)
                # Test as S0 with normal S1
                res_s0 = oracle_v_cvt_pknorm_u16_f64(nan_word, POS_ONE_F64)
                self.assertEqual(res_s0, (0xFFFF << 16) | 0)
                # Test as S1 with normal S0
                res_s1 = oracle_v_cvt_pknorm_u16_f64(POS_ONE_F64, nan_word)
                self.assertEqual(res_s1, (0 << 16) | 0xFFFF)
                # Test both channels NaN
                self.assertEqual(oracle_v_cvt_pknorm_u16_f64(nan_word, nan_word), 0)

    def test_infinity_boundaries(self) -> None:
        self.assertTrue(is_inf64(POS_INF_F64))
        self.assertTrue(is_inf64(NEG_INF_F64))
        self.assertEqual(oracle_cvt_unorm16_f64(POS_INF_F64), 0xFFFF)
        self.assertEqual(oracle_cvt_unorm16_f64(NEG_INF_F64), 0)

        res_pos_neg = oracle_v_cvt_pknorm_u16_f64(POS_INF_F64, NEG_INF_F64)
        self.assertEqual(res_pos_neg, (0x0000 << 16) | 0xFFFF)

        res_neg_pos = oracle_v_cvt_pknorm_u16_f64(NEG_INF_F64, POS_INF_F64)
        self.assertEqual(res_neg_pos, (0xFFFF << 16) | 0x0000)

    def test_signed_zeros_permutations(self) -> None:
        combos = [
            (POS_ZERO_F64, POS_ZERO_F64),
            (POS_ZERO_F64, NEG_ZERO_F64),
            (NEG_ZERO_F64, POS_ZERO_F64),
            (NEG_ZERO_F64, NEG_ZERO_F64),
        ]
        for s0, s1 in combos:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertTrue(is_zero64(s0))
                self.assertTrue(is_zero64(s1))
                self.assertEqual(oracle_v_cvt_pknorm_u16_f64(s0, s1), 0x00000000)

    def test_finite_boundary_extremes(self) -> None:
        # Smallest positive normal double: 2^-1022
        dbl_min = 0x0010000000000000
        self.assertEqual(oracle_cvt_unorm16_f64(dbl_min), 0)

        # Largest double < 1.0 (1.0 - 2^-53) -> rounds to 0xFFFF
        one_minus_eps = 0x3FEFFFFFFFFFFFFF
        self.assertEqual(oracle_cvt_unorm16_f64(one_minus_eps), 0xFFFF)

        # Smallest double > 1.0 (1.0 + 2^-52) -> clamps to 0xFFFF
        one_plus_eps = 0x3FF0000000000001
        self.assertEqual(oracle_cvt_unorm16_f64(one_plus_eps), 0xFFFF)

        # Maximum finite positive double (DBL_MAX)
        dbl_max = 0x7FEFFFFFFFFFFFFF
        self.assertEqual(oracle_cvt_unorm16_f64(dbl_max), 0xFFFF)

        # Maximum finite negative double (-DBL_MAX)
        neg_dbl_max = 0xFFEFFFFFFFFFFFFF
        self.assertEqual(oracle_cvt_unorm16_f64(neg_dbl_max), 0)

    def test_modifier_interaction_with_boundaries(self) -> None:
        # abs(-inf) -> +inf -> 0xFFFF
        res_abs_inf = oracle_v_cvt_pknorm_u16_f64(NEG_INF_F64, POS_ZERO_F64, abs0=True)
        self.assertEqual(extract_u16(res_abs_inf, high_half=False), 0xFFFF)

        # neg(+inf) -> -inf -> 0x0000
        res_neg_inf = oracle_v_cvt_pknorm_u16_f64(POS_INF_F64, POS_ZERO_F64, neg0=True)
        self.assertEqual(extract_u16(res_neg_inf, high_half=False), 0x0000)

        # abs and neg on NaN still yield 0x0000
        res_mod_nan = oracle_v_cvt_pknorm_u16_f64(POS_QNAN_F64, NEG_SNAN_F64, abs0=True, neg1=True)
        self.assertEqual(res_mod_nan, 0x00000000)

        # abs(-0.0) -> +0.0 -> 0x0000
        res_abs_z = oracle_v_cvt_pknorm_u16_f64(NEG_ZERO_F64, NEG_ZERO_F64, abs0=True, abs1=True)
        self.assertEqual(res_abs_z, 0x00000000)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestValuCvtPknormU16F64NegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_register_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(255, RegType.VGPR, is_double=True)

    def test_syntax_rejection_destination_must_be_single_vgpr(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand64(Reg(2, RegType.VGPR, is_double=True))
        with self.assertRaises(TypeError):
            VCvtPknormU16F64Instruction(dst=dst_sgpr, src0=src, src1=src)

        dst_double = Reg(0, RegType.VGPR, is_double=True)
        with self.assertRaises(TypeError):
            VCvtPknormU16F64Instruction(dst=dst_double, src0=src, src1=src)

    def test_syntax_rejection_clamp_modifier_rejected(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src = VOP3Operand64(Reg(2, RegType.VGPR, is_double=True))
        with self.assertRaises(ValueError):
            VCvtPknormU16F64Instruction(dst=dst, src0=src, src1=src, clamp=True)

    def test_syntax_rejection_invalid_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_operand_order_reversal(self) -> None:
        s0, s1 = POS_ONE_F64, HALF_F64
        correct = oracle_v_cvt_pknorm_u16_f64(s0, s1)
        reversed_order = oracle_v_cvt_pknorm_u16_f64(s1, s0)
        self.assertEqual(correct, (32768 << 16) | 0xFFFF)
        self.assertEqual(reversed_order, (0xFFFF << 16) | 32768)
        self.assertNotEqual(correct, reversed_order)

    def test_negative_control_missing_high_half_shift(self) -> None:
        s0, s1 = POS_ONE_F64, HALF_F64
        correct = oracle_v_cvt_pknorm_u16_f64(s0, s1)
        lo = oracle_cvt_unorm16_f64(s0)
        hi = oracle_cvt_unorm16_f64(s1)
        buggy_unshifted = (lo & MASK16) | (hi & MASK16)
        self.assertNotEqual(correct, buggy_unshifted)

    def test_negative_control_off_by_one_shift(self) -> None:
        s0, s1 = POS_ONE_F64, HALF_F64
        correct = oracle_v_cvt_pknorm_u16_f64(s0, s1)
        lo = oracle_cvt_unorm16_f64(s0)
        hi = oracle_cvt_unorm16_f64(s1)
        shift_15 = (lo & MASK16) | ((hi & MASK16) << 15)
        shift_17 = (lo & MASK16) | (((hi & MASK16) << 17) & MASK32)
        self.assertNotEqual(correct, shift_15)
        self.assertNotEqual(correct, shift_17)

    def test_negative_control_missing_mask_truncation(self) -> None:
        dirty_s0 = 0x10001
        dirty_s1 = 0x0002
        correct = pack_u16_to_u32(dirty_s0, dirty_s1)
        buggy_leaking = (dirty_s0 | (dirty_s1 << 16)) & MASK32
        self.assertEqual(correct, 0x00020001)
        self.assertEqual(buggy_leaking, 0x00030001)
        self.assertNotEqual(correct, buggy_leaking)

    def test_negative_control_wrong_modifier_bit(self) -> None:
        val = POS_ONE_F64
        correct_neg = apply_f64_modifiers(val, neg_mod=True)
        # Inverting bit 62 instead of sign bit 63
        wrong_bit = val ^ (1 << 62)
        self.assertEqual(correct_neg, NEG_ONE_F64)
        self.assertNotEqual(correct_neg, wrong_bit)

    def test_negative_control_modifier_order(self) -> None:
        val = POS_ONE_F64
        correct = apply_f64_modifiers(val, abs_mod=True, neg_mod=True)
        buggy_neg_then_abs = (val ^ SIGN64) & ABS64
        self.assertEqual(correct, NEG_ONE_F64)
        self.assertEqual(buggy_neg_then_abs, POS_ONE_F64)
        self.assertNotEqual(correct, buggy_neg_then_abs)

    def test_mutation_wrong_scale_factor_65536(self) -> None:
        """Mutant: Multiplying by 65536.0 instead of 65535.0."""
        def mutant_cvt(bits: int) -> int:
            val = bits_to_f64(bits)
            return round(val * 65536.0) & MASK16

        correct = oracle_cvt_unorm16_f64(POS_ONE_F64)
        mutant = mutant_cvt(POS_ONE_F64)
        self.assertEqual(correct, 65535)
        self.assertEqual(mutant, 0)
        self.assertNotEqual(correct, mutant)

    def test_mutation_snorm_scale_32767(self) -> None:
        """Mutant: Using snorm scale 32767.0 instead of unorm 65535.0."""
        def mutant_cvt(bits: int) -> int:
            val = bits_to_f64(bits)
            return round(val * 32767.0) & MASK16

        correct = oracle_cvt_unorm16_f64(POS_ONE_F64)
        mutant = mutant_cvt(POS_ONE_F64)
        self.assertEqual(correct, 65535)
        self.assertEqual(mutant, 32767)
        self.assertNotEqual(correct, mutant)

    def test_mutation_truncation_rounding(self) -> None:
        """Mutant: Using math.floor/truncation instead of round-half-even."""
        def mutant_cvt(bits: int) -> int:
            val = bits_to_f64(bits)
            return int(val * 65535.0) & MASK16

        correct = oracle_cvt_unorm16_f64(QUARTER_F64)
        mutant = mutant_cvt(QUARTER_F64)
        self.assertEqual(correct, 16384)
        self.assertEqual(mutant, 16383)
        self.assertNotEqual(correct, mutant)

    def test_mutation_nan_non_zero(self) -> None:
        """Mutant: Returning 0xFFFF on NaN input instead of 0."""
        def mutant_cvt(bits: int) -> int:
            if is_nan64(bits):
                return 0xFFFF
            return oracle_cvt_unorm16_f64(bits)

        correct = oracle_cvt_unorm16_f64(POS_QNAN_F64)
        mutant = mutant_cvt(POS_QNAN_F64)
        self.assertEqual(correct, 0)
        self.assertEqual(mutant, 0xFFFF)
        self.assertNotEqual(correct, mutant)

    def test_mutation_ignore_abs_modifier(self) -> None:
        """Mutant: Neglecting abs modifier."""
        def mutant_mod(bits: int, abs_mod: bool, neg_mod: bool) -> int:
            res = bits & MASK64
            if neg_mod:
                res ^= SIGN64
            return res

        val = NEG_QUARTER_F64
        correct = apply_f64_modifiers(val, abs_mod=True, neg_mod=False)
        mutant = mutant_mod(val, abs_mod=True, neg_mod=False)
        self.assertEqual(correct, QUARTER_F64)
        self.assertEqual(mutant, NEG_QUARTER_F64)
        self.assertNotEqual(correct, mutant)

    def test_mutation_ignore_neg_modifier(self) -> None:
        """Mutant: Neglecting neg modifier."""
        def mutant_mod(bits: int, abs_mod: bool, neg_mod: bool) -> int:
            res = bits & MASK64
            if abs_mod:
                res &= ABS64
            return res

        val = POS_ONE_F64
        correct = apply_f64_modifiers(val, abs_mod=False, neg_mod=True)
        mutant = mutant_mod(val, abs_mod=False, neg_mod=True)
        self.assertEqual(correct, NEG_ONE_F64)
        self.assertEqual(mutant, POS_ONE_F64)
        self.assertNotEqual(correct, mutant)

    def test_mutation_pack_addition_instead_of_or(self) -> None:
        """Mutant: Channel packing using addition instead of bitwise OR."""
        def mutant_pack(lo: int, hi: int) -> int:
            return ((hi << 16) + lo) & MASK32

        lo, hi = 0x8000, 0x8000
        self.assertEqual(pack_u16_to_u32(lo, hi), mutant_pack(lo, hi))

        dirty_lo, dirty_hi = 0x18000, 0x8000
        self.assertNotEqual(pack_u16_to_u32(dirty_lo, dirty_hi), mutant_pack(dirty_lo, dirty_hi))


if __name__ == "__main__":
    unittest.main()
