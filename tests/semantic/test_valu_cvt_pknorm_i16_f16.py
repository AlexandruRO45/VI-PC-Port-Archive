from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pknorm_i16_f16.

Models the AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) instruction:
    v_cvt_pknorm_i16_f16 D, S0, S1 [op_sel:[s0, s1, 0, 0]]

Lowered Architectural Semantics:
    D[15:0]  = convert_f16_to_snorm16(extract_half(S0, op_sel[0]))
    D[31:16] = convert_f16_to_snorm16(extract_half(S1, op_sel[1]))

RDNA2 VALU Specification Rules:
  1. Packing & Destination:
     - Destination is a single 32-bit VGPR (v[d]).
     - Overwrites all 32 bits of destination VGPR atomically.
     - D[15:0] is derived from S0 (selected half); D[31:16] is derived from S1 (selected half).
  2. Conversion Mathematics:
     - Inputs are IEEE 754-2008 half-precision floating-point numbers (binary16).
     - Each channel is clamped to [-1.0, 1.0]:
         * Values <= -1.0 (including -Infinity) clamp to -1.0 -> -32767 (0x8001).
         * Values >= 1.0 (including +Infinity) clamp to 1.0 -> +32767 (0x7FFF).
         * -0.0 and +0.0 evaluate to 0x0000.
     - Normalized scaling:
         * Product = x * 32767.0
         * Rounded to nearest integer with tie-breaking to even (IEEE 754 roundTiesToEven).
  3. NaN Semantics:
     - Any NaN input (Quiet NaN or Signaling NaN, positive or negative sign, any payload)
       evaluates to 0x0000.
  4. VOP3/VOP3P Modifiers:
     - op_sel[0]: Selects low half (0) or high half (1) of S0.
     - op_sel[1]: Selects low half (0) or high half (1) of S1.
     - abs (|x|): Clears sign bit (bit 15) of f16 input before conversion.
     - neg (-x): Inverts sign bit (bit 15) of f16 input before conversion.
     - neg(abs(x)): Inverts sign bit after abs, forcing negative sign.
     - clamp on destination is rejected / unsupported.

Standard library Python only (python -I, struct, math, unittest). Zero game data.
"""

import dataclasses
import enum
import math
import struct
import unittest

# Bitwise Masks & Architectural Constants
MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
SIGN16: int = 0x8000
ABS16: int = 0x7FFF
EXP_MASK16: int = 0x7C00
MANT_MASK16: int = 0x03FF
QNAN_BIT16: int = 0x0200
CANONICAL_QNAN16: int = 0x7E00

SNORM16_MIN: int = 0x8001               # -32767 (signed 16-bit)
SNORM16_MAX: int = 0x7FFF               # +32767
SCALE_SNORM16: float = 32767.0

# IEEE 754 binary16 Bit Patterns
POS_ZERO_F16: int = 0x0000              # +0.0
NEG_ZERO_F16: int = 0x8000              # -0.0
POS_ONE_F16: int = 0x3C00               # 1.0
NEG_ONE_F16: int = 0xBC00               # -1.0
POS_TWO_F16: int = 0x4000               # 2.0
NEG_TWO_F16: int = 0xC000               # -2.0
HALF_F16: int = 0x3800                  # 0.5
NEG_HALF_F16: int = 0xB800              # -0.5
QUARTER_F16: int = 0x3400               # 0.25
NEG_QUARTER_F16: int = 0xB400           # -0.25

POS_INF_F16: int = 0x7C00               # +Infinity
NEG_INF_F16: int = 0xFC00               # -Infinity
POS_QNAN_F16: int = 0x7E00              # Quiet NaN
NEG_QNAN_F16: int = 0xFE00              # Negative Quiet NaN
POS_SNAN_F16: int = 0x7C01              # Signaling NaN
NEG_SNAN_F16: int = 0xFC01              # Negative Signaling NaN
SUBNORM_MIN_F16: int = 0x0001           # Smallest positive subnormal (2^-24)
SUBNORM_MAX_F16: int = 0x03FF           # Largest positive subnormal
NEG_SUBNORM_MIN_F16: int = 0x8001
NEG_SUBNORM_MAX_F16: int = 0x83FF


def f16_to_bits(val: float) -> int:
    """Pack Python float into IEEE 754 binary16 unsigned 16-bit integer."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF_F16 if val > 0.0 else NEG_INF_F16


def bits_to_f16(bits: int) -> float:
    """Unpack unsigned 16-bit integer into Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def split_f16_bits(bits: int) -> tuple[int, int, int]:
    """Unpack 16-bit word into (sign, exponent, fraction) components."""
    bits &= MASK16
    sign = (bits & SIGN16) >> 15
    exp = (bits & EXP_MASK16) >> 10
    mant = bits & MANT_MASK16
    return sign, exp, mant


def is_nan16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 NaN."""
    bits &= MASK16
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & MANT_MASK16) != 0)


def is_qnan16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 Quiet NaN."""
    bits &= MASK16
    return is_nan16(bits) and bool(bits & QNAN_BIT16)


def is_snan16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 Signaling NaN."""
    bits &= MASK16
    return is_nan16(bits) and not bool(bits & QNAN_BIT16)


def is_inf16(bits: int) -> bool:
    """Return True if 16-bit word represents positive or negative infinity."""
    bits &= MASK16
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & MANT_MASK16) == 0)


def is_subnormal16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 subnormal half float."""
    bits &= MASK16
    return ((bits & EXP_MASK16) == 0) and ((bits & MANT_MASK16) != 0)


def is_zero16(bits: int) -> bool:
    """Return True if 16-bit word represents +0.0 or -0.0."""
    return (bits & ABS16) == 0


def extract_half(val32: int, high_half: bool) -> int:
    """Extract low 16 bits (high_half=False) or high 16 bits (high_half=True)."""
    return ((val32 >> 16) & MASK16) if high_half else (val32 & MASK16)


def apply_f16_modifiers(bits16: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 16-bit float word.

    In RDNA2 ISA, abs clears sign bit (bit 15) first, then neg inverts sign bit (bit 15).
    """
    res = bits16 & MASK16
    if abs_mod:
        res &= ABS16
    if neg_mod:
        res ^= SIGN16
    return res


def pack_u16_to_u32(lo_half: int, hi_half: int) -> int:
    """Pack two 16-bit integers into an unsigned 32-bit dword."""
    return ((hi_half & MASK16) << 16) | (lo_half & MASK16)


def oracle_cvt_snorm16_f16(f16_bits: int) -> int:
    """Mathematical reference oracle converting a single f16 to snorm16.

    Semantics:
        - If input is NaN: returns 0x0000.
        - If input <= -1.0 (including -inf): returns 0x8001 (-32767).
        - If input >= 1.0 (including +inf): returns 0x7FFF (+32767).
        - If input is +-0.0: returns 0x0000.
        - Otherwise: round_half_even(x * 32767.0) & 0xFFFF.
    """
    if is_nan16(f16_bits):
        return 0x0000
    val = bits_to_f16(f16_bits)
    if val <= -1.0 or (math.isinf(val) and val < 0):
        return SNORM16_MIN
    if val >= 1.0 or (math.isinf(val) and val > 0):
        return SNORM16_MAX
    if val == 0.0:
        return 0x0000
    scaled = val * SCALE_SNORM16
    return round(scaled) & MASK16


def oracle_v_cvt_pknorm_i16_f16(
    src0_bits: int,
    src1_bits: int,
    op_sel: tuple[int, int] = (0, 0),
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Architectural reference oracle for RDNA2 v_cvt_pknorm_i16_f16.

    Extracts half words via op_sel, applies source modifiers,
    converts to snorm16, and packs into dst[31:0].
    """
    h0 = extract_half(src0_bits, bool(op_sel[0]))
    h1 = extract_half(src1_bits, bool(op_sel[1]))

    h0_mod = apply_f16_modifiers(h0, abs_mod=abs0, neg_mod=neg0)
    h1_mod = apply_f16_modifiers(h1, abs_mod=abs1, neg_mod=neg1)

    lo = oracle_cvt_snorm16_f16(h0_mod)
    hi = oracle_cvt_snorm16_f16(h1_mod)

    return pack_u16_to_u32(lo, hi)


# ==============================================================================
# Unit Tests: Architectural Reference Model & Mathematical Oracle
# ==============================================================================

class TestValuCvtPknormI16F16Oracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_cvt_pknorm_i16_f16."""

    def test_basic_normal_conversions(self) -> None:
        # 0.0 -> 0x0000
        self.assertEqual(oracle_cvt_snorm16_f16(POS_ZERO_F16), 0)
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_ZERO_F16), 0)

        # 1.0 -> 0x7FFF (+32767)
        self.assertEqual(oracle_cvt_snorm16_f16(POS_ONE_F16), 0x7FFF)

        # -1.0 -> 0x8001 (-32767)
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_ONE_F16), 0x8001)

        # 0.5 -> 16384 (0x4000), 16383.5 rounds to even 16384
        self.assertEqual(oracle_cvt_snorm16_f16(HALF_F16), 0x4000)

        # -0.5 -> -16384 (0xC000), -16383.5 rounds to even -16384
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_HALF_F16), 0xC000)

    def test_channel_packing_and_ordering(self) -> None:
        # S0 gets 0.5 -> 0x4000 in low half
        # S1 gets -1.0 -> 0x8001 in high half
        res = oracle_v_cvt_pknorm_i16_f16(HALF_F16, NEG_ONE_F16)
        expected = (0x8001 << 16) | 0x4000
        self.assertEqual(res, expected)
        self.assertEqual(extract_half(res, high_half=False), 0x4000)
        self.assertEqual(extract_half(res, high_half=True), 0x8001)

    def test_op_sel_permutations(self) -> None:
        # S0 has 1.0 in low half and 0.5 in high half
        # S1 has -1.0 in low half and -0.5 in high half
        s0 = (HALF_F16 << 16) | POS_ONE_F16
        s1 = (NEG_HALF_F16 << 16) | NEG_ONE_F16

        # op_sel: [0, 0] -> S0 low (1.0 -> 0x7FFF), S1 low (-1.0 -> 0x8001)
        self.assertEqual(oracle_v_cvt_pknorm_i16_f16(s0, s1, (0, 0)), (0x8001 << 16) | 0x7FFF)

        # op_sel: [1, 0] -> S0 high (0.5 -> 0x4000), S1 low (-1.0 -> 0x8001)
        self.assertEqual(oracle_v_cvt_pknorm_i16_f16(s0, s1, (1, 0)), (0x8001 << 16) | 0x4000)

        # op_sel: [0, 1] -> S0 low (1.0 -> 0x7FFF), S1 high (-0.5 -> 0xC000)
        self.assertEqual(oracle_v_cvt_pknorm_i16_f16(s0, s1, (0, 1)), (0xC000 << 16) | 0x7FFF)

        # op_sel: [1, 1] -> S0 high (0.5 -> 0x4000), S1 high (-0.5 -> 0xC000)
        self.assertEqual(oracle_v_cvt_pknorm_i16_f16(s0, s1, (1, 1)), (0xC000 << 16) | 0x4000)

    def test_clamping_out_of_range(self) -> None:
        # Values > 1.0 clamp to 0x7FFF (+32767)
        self.assertEqual(oracle_cvt_snorm16_f16(POS_TWO_F16), 0x7FFF)
        self.assertEqual(oracle_cvt_snorm16_f16(f16_to_bits(10.0)), 0x7FFF)

        # Values < -1.0 clamp to 0x8001 (-32767)
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_TWO_F16), 0x8001)
        self.assertEqual(oracle_cvt_snorm16_f16(f16_to_bits(-10.0)), 0x8001)

    def test_infinities(self) -> None:
        self.assertEqual(oracle_cvt_snorm16_f16(POS_INF_F16), 0x7FFF)
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_INF_F16), 0x8001)
        res = oracle_v_cvt_pknorm_i16_f16(POS_INF_F16, NEG_INF_F16)
        self.assertEqual(res, (0x8001 << 16) | 0x7FFF)

    def test_nan_evaluates_to_zero(self) -> None:
        self.assertEqual(oracle_cvt_snorm16_f16(POS_QNAN_F16), 0)
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_QNAN_F16), 0)
        self.assertEqual(oracle_cvt_snorm16_f16(POS_SNAN_F16), 0)
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_SNAN_F16), 0)

    def test_round_ties_to_even(self) -> None:
        # 0.5 * 32767 = 16383.5 -> rounds to even 16384 (0x4000)
        self.assertEqual(oracle_cvt_snorm16_f16(HALF_F16), 16384)
        # -0.5 * 32767 = -16383.5 -> rounds to even -16384 (0xC000)
        self.assertEqual(oracle_cvt_snorm16_f16(NEG_HALF_F16), (-16384 & 0xFFFF))

    def test_source_modifiers(self) -> None:
        # abs on -1.0 gives 1.0 -> 0x7FFF
        res_abs = oracle_v_cvt_pknorm_i16_f16(NEG_ONE_F16, POS_ZERO_F16, abs0=True)
        self.assertEqual(extract_half(res_abs, high_half=False), 0x7FFF)

        # neg on 1.0 gives -1.0 -> 0x8001
        res_neg = oracle_v_cvt_pknorm_i16_f16(POS_ONE_F16, POS_ZERO_F16, neg0=True)
        self.assertEqual(extract_half(res_neg, high_half=False), 0x8001)

        # abs then neg (-|x|) gives -1.0 -> 0x8001
        res_both = oracle_v_cvt_pknorm_i16_f16(POS_ONE_F16, POS_ZERO_F16, abs0=True, neg0=True)
        self.assertEqual(extract_half(res_both, high_half=False), 0x8001)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class Literal16:
    """Represents an immediate 16-bit float or integer constant."""
    bits: int

    @classmethod
    def from_f16(cls, val: float) -> Literal16:
        return cls(f16_to_bits(val))

    @property
    def name(self) -> str:
        return f"0x{self.bits & MASK16:04X}"


@dataclasses.dataclass(frozen=True)
class VOP3POperand:
    """Typed operand for packed 16-bit operations with optional modifiers."""
    source: Reg | Literal16
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
    V_CVT_PKNORM_I16_F16 = "v_cvt_pknorm_i16_f16"


@dataclasses.dataclass(frozen=True)
class VCvtPknormI16F16Instruction:
    """Symbolic AST node for RDNA2 v_cvt_pknorm_i16_f16."""
    dst: Reg
    src0: VOP3POperand
    src1: VOP3POperand
    op_sel: tuple[int, int] = (0, 0)
    clamp: bool = False
    opcode: Opcode = Opcode.V_CVT_PKNORM_I16_F16

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.clamp:
            raise ValueError("v_cvt_pknorm_i16_f16 destination cannot have clamp modifier")
        if self.op_sel[0] not in (0, 1) or self.op_sel[1] not in (0, 1):
            raise ValueError(f"op_sel values must be 0 or 1, got {self.op_sel}")

    def disassemble(self) -> str:
        s = f"{self.opcode.value} {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}"
        if self.op_sel != (0, 0):
            s += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]},0,0]"
        return s


@dataclasses.dataclass
class Program:
    """Represents an ordered sequence of instructions."""
    instructions: list[VCvtPknormI16F16Instruction] = dataclasses.field(default_factory=list)

    def append(self, instr: VCvtPknormI16F16Instruction) -> None:
        self.instructions.append(instr)

    def disassemble(self) -> str:
        return "\n".join(instr.disassemble() for instr in self.instructions)


class TestValuCvtPknormI16F16AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_creation_and_naming(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        v255 = Reg(255, RegType.VGPR)
        s12 = Reg(12, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(s12.name, "s12")

    def test_register_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(300, RegType.SGPR)

    def test_literal_operand(self) -> None:
        lit = Literal16(0x3C00)
        self.assertEqual(lit.name, "0x3C00")
        lit_float = Literal16.from_f16(1.0)
        self.assertEqual(lit_float.bits, POS_ONE_F16)
        self.assertEqual(lit_float.name, "0x3C00")

    def test_operand_modifiers_disassembly(self) -> None:
        v1 = Reg(1, RegType.VGPR)
        op_plain = VOP3POperand(v1)
        self.assertEqual(op_plain.disassemble(), "v1")

        op_abs = VOP3POperand(v1, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v1|")

        op_neg = VOP3POperand(v1, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v1")

        op_both = VOP3POperand(v1, abs_mod=True, neg_mod=True)
        self.assertEqual(op_both.disassemble(), "-|v1|")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3POperand(Reg(1, RegType.VGPR))
        src1 = VOP3POperand(Reg(2, RegType.VGPR), neg_mod=True)

        instr_default = VCvtPknormI16F16Instruction(dst=dst, src0=src0, src1=src1)
        self.assertEqual(instr_default.disassemble(), "v_cvt_pknorm_i16_f16 v0, v1, -v2")

        instr_opsel = VCvtPknormI16F16Instruction(dst=dst, src0=src0, src1=src1, op_sel=(1, 0))
        self.assertEqual(instr_opsel.disassemble(), "v_cvt_pknorm_i16_f16 v0, v1, -v2 op_sel:[1,0,0,0]")

    def test_instruction_validation(self) -> None:
        # Destination must be VGPR
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3POperand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VCvtPknormI16F16Instruction(dst=dst_sgpr, src0=src, src1=src)

        # Destination clamp modifier rejected
        with self.assertRaises(ValueError):
            VCvtPknormI16F16Instruction(dst=Reg(0, RegType.VGPR), src0=src, src1=src, clamp=True)

        # op_sel must be in {0, 1}
        with self.assertRaises(ValueError):
            VCvtPknormI16F16Instruction(dst=Reg(0, RegType.VGPR), src0=src, src1=src, op_sel=(2, 0))

    def test_program_container(self) -> None:
        prog = Program()
        v0 = Reg(0, RegType.VGPR)
        v1 = Reg(1, RegType.VGPR)
        v2 = Reg(2, RegType.VGPR)
        prog.append(VCvtPknormI16F16Instruction(v0, VOP3POperand(v1), VOP3POperand(v2)))
        prog.append(VCvtPknormI16F16Instruction(v1, VOP3POperand(v0, abs_mod=True), VOP3POperand(v2, neg_mod=True), op_sel=(1, 1)))
        expected_asm = (
            "v_cvt_pknorm_i16_f16 v0, v1, v2\n"
            "v_cvt_pknorm_i16_f16 v1, |v0|, -v2 op_sel:[1,1,0,0]"
        )
        self.assertEqual(prog.disassemble(), expected_asm)


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

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_cvt_pknorm_i16_f16."""

    def resolve_operand(self, op: VOP3POperand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal16):
            return op.source.bits & MASK16
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VCvtPknormI16F16Instruction, state: ScalarMachineState) -> int:
        # Atomic operand fetch phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)

        # Computation phase via architectural oracle
        result = oracle_v_cvt_pknorm_i16_f16(
            src0_bits=s0_raw,
            src1_bits=s1_raw,
            op_sel=instr.op_sel,
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


class TestValuCvtPknormI16F16ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, POS_ONE_F16)
        self.state.set_vgpr(2, HALF_F16)
        instr = VCvtPknormI16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(Reg(1, RegType.VGPR)),
            src1=VOP3POperand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        # S0: 1.0 -> 0x7FFF; S1: 0.5 -> 0x4000
        expected = (0x4000 << 16) | 0x7FFF
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_step_with_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(5, NEG_ONE_F16)
        instr = VCvtPknormI16F16Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3POperand(Reg(5, RegType.SGPR)),
            src1=VOP3POperand(Literal16.from_f16(1.0)),
        )
        res = self.interp.step(instr, self.state)
        # S0: -1.0 -> 0x8001; S1: 1.0 -> 0x7FFF
        expected = (0x7FFF << 16) | 0x8001
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(3), expected)

    def test_step_with_modifiers_and_opsel(self) -> None:
        # v1 has -0.5 in low half and 0.25 in high half
        # v2 has 1.0 in low half and -1.0 in high half
        self.state.set_vgpr(1, (QUARTER_F16 << 16) | NEG_HALF_F16)
        self.state.set_vgpr(2, (NEG_ONE_F16 << 16) | POS_ONE_F16)

        # src0 extracts low half of v1 (-0.5) with abs -> |-0.5| = 0.5 -> 0x4000
        # src1 extracts high half of v2 (-1.0) with neg -> -(-1.0) = 1.0 -> 0x7FFF
        instr = VCvtPknormI16F16Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3POperand(Reg(1, RegType.VGPR), abs_mod=True),
            src1=VOP3POperand(Reg(2, RegType.VGPR), neg_mod=True),
            op_sel=(0, 1),
        )
        res = self.interp.step(instr, self.state)
        expected = (0x7FFF << 16) | 0x4000
        self.assertEqual(res, expected)
        self.assertEqual(self.state.get_vgpr(0), expected)

    def test_full_32bit_destination_overwrite(self) -> None:
        self.state.set_vgpr(7, 0xDEADBEEF)
        self.state.set_vgpr(1, POS_ZERO_F16)
        self.state.set_vgpr(2, POS_ZERO_F16)
        instr = VCvtPknormI16F16Instruction(
            dst=Reg(7, RegType.VGPR),
            src0=VOP3POperand(Reg(1, RegType.VGPR)),
            src1=VOP3POperand(Reg(2, RegType.VGPR)),
        )
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(self.state.get_vgpr(7), 0x00000000)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, HALF_F16)
        self.state.set_vgpr(11, NEG_HALF_F16)
        instr = VCvtPknormI16F16Instruction(
            dst=Reg(12, RegType.VGPR),
            src0=VOP3POperand(Reg(10, RegType.VGPR)),
            src1=VOP3POperand(Reg(11, RegType.VGPR)),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)

    def test_run_program_sequential(self) -> None:
        prog = Program()
        self.state.set_vgpr(1, POS_ONE_F16)
        self.state.set_vgpr(2, HALF_F16)
        prog.append(VCvtPknormI16F16Instruction(
            dst=Reg(3, RegType.VGPR),
            src0=VOP3POperand(Reg(1, RegType.VGPR)),
            src1=VOP3POperand(Reg(2, RegType.VGPR)),
        ))
        prog.append(VCvtPknormI16F16Instruction(
            dst=Reg(4, RegType.VGPR),
            src0=VOP3POperand(Reg(2, RegType.VGPR)),
            src1=VOP3POperand(Reg(1, RegType.VGPR)),
        ))
        results = self.interp.run_program(prog, self.state)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], (0x4000 << 16) | 0x7FFF)
        self.assertEqual(results[1], (0x7FFF << 16) | 0x4000)


if __name__ == "__main__":
    unittest.main()
