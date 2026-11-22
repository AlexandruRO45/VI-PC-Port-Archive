from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mad_i32_i16.

AMD RDNA2 VALU Instruction Set Architecture:
  v_mad_i32_i16 dst, src0, src1, src2 [clamp] [op_sel:x,y]

Format:
  VOP3 (3-operand vector ALU instruction).

Operation:
  Multiplies two signed 16-bit integers, adds a 32-bit signed integer accumulator,
  and writes the 32-bit result to a vector register:
    s0_16   = (src0 >> 16) if op_sel[0] else src0
    s1_16   = (src1 >> 16) if op_sel[1] else src1
    prod32  = int16(s0_16) * int16(s1_16)
    sum32   = prod32 + int32(src2)
    dst.i32 = clamp_i32(sum32) if clamp else (sum32 & 0xFFFFFFFF)

Mathematical Properties and Bounds:
  - src0.i16 and src1.i16 are signed 16-bit integers in [-32768, 32767].
  - Product prod32 is strictly bounded in [-1073709056, 1073741824].
    Maximum positive product: (-32768) * (-32768) = 1,073,741,824 (0x40000000).
    Minimum negative product: (-32768) * 32767 = -1,073,709,056.
  - The product always fits without overflow in signed 32-bit integer arithmetic.
  - Accumulation with 32-bit signed addend wraps modulo 2^32 unless clamp is enabled.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
import struct
import unittest

# Architectural Constants
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
SIGN16: int = 0x8000
SIGN32: int = 0x8000_0000
MODULO16: int = 0x1_0000
MODULO32: int = 0x1_0000_0000

INT16_MIN: int = -0x8000
INT16_MAX: int = 0x7FFF
UINT16_MAX: int = 0xFFFF
INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF

NUM_VGPRS: int = 256
NUM_SGPRS: int = 106


def to_i16(val: int) -> int:
    """Normalize integer to signed 16-bit integer in [-32768, 32767]."""
    val = val & MASK16
    return val - MODULO16 if (val & SIGN16) else val


def to_u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit word in [0, 65535]."""
    return val & MASK16


def to_i32(val: int) -> int:
    """Normalize integer to signed 32-bit integer in [-2^31, 2^31 - 1]."""
    val = val & MASK32
    return val - MODULO32 if (val & SIGN32) else val


def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword in [0, 2^32 - 1]."""
    return val & MASK32


def clamp_i32(val: int) -> int:
    """Clamp signed integer to 32-bit signed range [INT32_MIN, INT32_MAX]."""
    if val < INT32_MIN:
        return INT32_MIN
    if val > INT32_MAX:
        return INT32_MAX
    return val


# ==============================================================================
# Reference Mathematical Oracle
# ==============================================================================

def ref_mul_i16(s0: int, s1: int) -> int:
    """Compute exact signed 16-bit * 16-bit product -> signed 32-bit integer."""
    i0 = to_i16(s0)
    i1 = to_i16(s1)
    return i0 * i1


def ref_mad_i32_i16(
    s0: int,
    s1: int,
    s2: int,
    clamp: bool = False,
    op_sel: tuple[int, int] = (0, 0),
) -> int:
    """Pure-Python reference oracle for v_mad_i32_i16.

    Returns unsigned 32-bit dword result.
    """
    s0_16 = (s0 >> 16) if op_sel[0] else s0
    s1_16 = (s1 >> 16) if op_sel[1] else s1

    prod = ref_mul_i16(s0_16, s1_16)
    acc = to_i32(s2)
    raw_sum = prod + acc

    if clamp:
        res_signed = clamp_i32(raw_sum)
        return to_u32(res_signed)

    return to_u32(raw_sum)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class Operand:
    """Represents a vector register, scalar register, inline constant, or literal."""
    reg_type: RegType
    index: int = 0
    literal_val: int | None = None

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index < NUM_VGPRS):
                raise ValueError(f"VGPR index {self.index} out of range [0, {NUM_VGPRS - 1}]")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index < NUM_SGPRS):
                raise ValueError(f"SGPR index {self.index} out of range [0, {NUM_SGPRS - 1}]")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_val is None:
                raise ValueError("Literal operand requires literal_val")
            if not (0 <= self.literal_val <= MASK32):
                raise ValueError(f"Literal value {self.literal_val:#x} out of range [0, {MASK32:#x}]")
        elif self.reg_type == RegType.INLINE_CONST:
            if self.literal_val is None:
                raise ValueError("Inline constant operand requires literal_val")

    @classmethod
    def vgpr(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index)

    @classmethod
    def sgpr(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index)

    @classmethod
    def literal(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_val=to_u32(value))

    @classmethod
    def inline_const(cls, value: int) -> Operand:
        return cls(RegType.INLINE_CONST, literal_val=value)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        if self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        if self.reg_type == RegType.LITERAL:
            assert self.literal_val is not None
            return f"0x{self.literal_val:08X}"
        if self.reg_type == RegType.INLINE_CONST:
            assert self.literal_val is not None
            return str(self.literal_val)
        raise ValueError(f"Unknown RegType: {self.reg_type}")


@dataclass(frozen=True)
class VMadI32I16Instruction:
    """Symbolic AST node for RDNA2 v_mad_i32_i16 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False
    op_sel: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        parts = [
            "v_mad_i32_i16",
            f"{self.dst.disassemble()}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}",
        ]
        if self.clamp:
            parts.append("clamp")
        if self.op_sel != (0, 0):
            parts.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]}]")
        return " ".join(parts)


def parse_v_mad_i32_i16(asm_line: str) -> VMadI32I16Instruction:
    """Parse assembly string into VMadI32I16Instruction AST node."""
    line = asm_line.strip()
    if not line.startswith("v_mad_i32_i16"):
        raise ValueError(f"Expected opcode 'v_mad_i32_i16', got '{line}'")

    rest = line[len("v_mad_i32_i16"):].strip()
    tokens = rest.split()
    clamp = False
    op_sel = (0, 0)

    operand_tokens_raw: list[str] = []
    for tok in tokens:
        if tok == "clamp":
            clamp = True
        elif tok.startswith("op_sel:[") and tok.endswith("]"):
            content = tok[len("op_sel:["):-1]
            sel_parts = content.split(",")
            op_sel = (int(sel_parts[0]), int(sel_parts[1]))
        else:
            operand_tokens_raw.append(tok)

    operands_str = " ".join(operand_tokens_raw)
    op_list = [t.strip() for t in operands_str.split(",") if t.strip()]
    if len(op_list) != 4:
        raise ValueError(f"v_mad_i32_i16 requires 4 operands, got {len(op_list)}: {op_list}")

    def parse_op(tok: str) -> Operand:
        m_vgpr = re.fullmatch(r"v(\d+)", tok)
        if m_vgpr:
            return Operand.vgpr(int(m_vgpr.group(1)))
        m_sgpr = re.fullmatch(r"s(\d+)", tok)
        if m_sgpr:
            return Operand.sgpr(int(m_sgpr.group(1)))
        if tok.startswith("0x") or tok.startswith("-0x"):
            return Operand.literal(int(tok, 16))
        try:
            val = int(tok, 10)
            if -16 <= val <= 64:
                return Operand.inline_const(val)
            return Operand.literal(val)
        except ValueError:
            pass
        raise ValueError(f"Cannot parse operand token: '{tok}'")

    dst = parse_op(op_list[0])
    src0 = parse_op(op_list[1])
    src1 = parse_op(op_list[2])
    src2 = parse_op(op_list[3])

    return VMadI32I16Instruction(dst=dst, src0=src0, src1=src1, src2=src2, clamp=clamp, op_sel=op_sel)


# ==============================================================================
# Architectural Hardware Model
# ==============================================================================

@dataclass
class RegisterFile:
    """Architectural model representing RDNA2 VGPRs and SGPRs."""
    vgprs: list[int] = field(default_factory=lambda: [0] * NUM_VGPRS)
    sgprs: list[int] = field(default_factory=lambda: [0] * NUM_SGPRS)

    def read_vgpr(self, idx: int) -> int:
        if not (0 <= idx < NUM_VGPRS):
            raise IndexError(f"VGPR index {idx} out of range [0, {NUM_VGPRS - 1}]")
        return self.vgprs[idx] & MASK32

    def write_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_VGPRS):
            raise IndexError(f"VGPR index {idx} out of range [0, {NUM_VGPRS - 1}]")
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        return self.sgprs[idx] & MASK32

    def write_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        self.sgprs[idx] = val & MASK32


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarState:
    """Scalar processor execution state."""
    rf: RegisterFile = field(default_factory=RegisterFile)

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.rf.read_vgpr(op.index)
        if op.reg_type == RegType.SGPR:
            return self.rf.read_sgpr(op.index)
        if op.reg_type in (RegType.LITERAL, RegType.INLINE_CONST):
            assert op.literal_val is not None
            return to_u32(op.literal_val)
        raise ValueError(f"Unsupported operand type: {op.reg_type}")

    def write_operand(self, op: Operand, val: int) -> None:
        if op.reg_type == RegType.VGPR:
            self.rf.write_vgpr(op.index, val)
        elif op.reg_type == RegType.SGPR:
            self.rf.write_sgpr(op.index, val)
        else:
            raise TypeError(f"Cannot write value to {op.reg_type}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_mad_i32_i16."""

    def step(self, instr: VMadI32I16Instruction, state: ScalarState) -> int:
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        s2 = state.read_operand(instr.src2)

        res = ref_mad_i32_i16(s0, s1, s2, clamp=instr.clamp, op_sel=instr.op_sel)
        state.write_operand(instr.dst, res)
        return res


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class WaveMode(Enum):
    WAVE32 = 32
    WAVE64 = 64


@dataclass
class SIMDState:
    """RDNA2 SIMD Wavefront execution state."""
    wave_mode: WaveMode = WaveMode.WAVE32
    exec_mask: int = 0xFFFF_FFFF
    num_lanes: int = 32
    vgprs: list[list[int]] = field(default_factory=list)
    sgprs: list[int] = field(default_factory=lambda: [0] * NUM_SGPRS)

    def __post_init__(self) -> None:
        self.num_lanes = self.wave_mode.value
        if not self.vgprs:
            self.vgprs = [[0] * NUM_VGPRS for _ in range(self.num_lanes)]
        if self.wave_mode == WaveMode.WAVE32:
            self.exec_mask &= MASK32
        else:
            self.exec_mask &= (1 << 64) - 1

    def read_lane(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        if op.reg_type in (RegType.LITERAL, RegType.INLINE_CONST):
            assert op.literal_val is not None
            return to_u32(op.literal_val)
        raise ValueError(f"Unsupported operand type: {op.reg_type}")

    def write_lane(self, op: Operand, lane: int, val: int) -> None:
        if op.reg_type == RegType.VGPR:
            self.vgprs[lane][op.index] = val & MASK32
        else:
            raise TypeError(f"Lane write target must be VGPR, got {op.reg_type}")


class SIMDInterpreter:
    """SIMD wavefront instruction interpreter handling divergence and EXEC masking."""

    def step(self, instr: VMadI32I16Instruction, state: SIMDState) -> list[int | None]:
        results: list[int | None] = [None] * state.num_lanes
        for lane in range(state.num_lanes):
            if bool(state.exec_mask & (1 << lane)):
                s0 = state.read_lane(instr.src0, lane)
                s1 = state.read_lane(instr.src1, lane)
                s2 = state.read_lane(instr.src2, lane)
                res = ref_mad_i32_i16(s0, s1, s2, clamp=instr.clamp, op_sel=instr.op_sel)
                state.write_lane(instr.dst, lane, res)
                results[lane] = res
        return results


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle
# ==============================================================================

class TestReferenceMathematicalOracle(unittest.TestCase):
    """Exhaustive mathematical verification of pure-Python reference oracle."""

    def test_zero_identities(self) -> None:
        self.assertEqual(ref_mad_i32_i16(0, 0, 0), 0)
        self.assertEqual(ref_mad_i32_i16(0, 100, 0), 0)
        self.assertEqual(ref_mad_i32_i16(100, 0, 0), 0)
        self.assertEqual(ref_mad_i32_i16(0, 0, 0x1234_5678), 0x1234_5678)

    def test_one_identity(self) -> None:
        self.assertEqual(ref_mad_i32_i16(1, 42, 0), 42)
        self.assertEqual(ref_mad_i32_i16(42, 1, 100), 142)

    def test_signed_multiplication_signs(self) -> None:
        # positive * positive
        self.assertEqual(ref_mad_i32_i16(50, 40, 0), 2000)
        # positive * negative
        self.assertEqual(ref_mad_i32_i16(50, -40, 0), to_u32(-2000))
        # negative * positive
        self.assertEqual(ref_mad_i32_i16(-50, 40, 0), to_u32(-2000))
        # negative * negative
        self.assertEqual(ref_mad_i32_i16(-50, -40, 0), 2000)

    def test_signed_boundary_products(self) -> None:
        # INT16_MIN * INT16_MIN = (-32768) * (-32768) = 1073741824 = 0x4000_0000
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, INT16_MIN, 0), 0x4000_0000)

        # INT16_MIN * 1 = -32768
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, 1, 0), to_u32(INT16_MIN))

        # INT16_MIN * -1 = 32768 = 0x8000
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, -1, 0), 0x8000)

        # INT16_MAX * INT16_MAX = 32767 * 32767 = 1073676289
        self.assertEqual(ref_mad_i32_i16(INT16_MAX, INT16_MAX, 0), 1073676289)

        # INT16_MIN * INT16_MAX = (-32768) * 32767 = -1073709056
        self.assertEqual(ref_mad_i32_i16(INT16_MIN, INT16_MAX, 0), to_u32(-1073709056))

    def test_accumulation_modulo32_wrapping(self) -> None:
        # (-1) * 1 + 1 = 0
        self.assertEqual(ref_mad_i32_i16(-1, 1, 1), 0)

        # INT32_MAX + 1 wraps to INT32_MIN (0x80000000)
        self.assertEqual(ref_mad_i32_i16(1, 1, INT32_MAX), 0x8000_0000)

        # INT32_MIN - 1 wraps to INT32_MAX (0x7FFFFFFF)
        self.assertEqual(ref_mad_i32_i16(-1, 1, INT32_MIN), 0x7FFF_FFFF)

    def test_clamp_modifier(self) -> None:
        # Overflow without clamp wraps; with clamp clamps to INT32_MAX
        val_unclamped = ref_mad_i32_i16(1, 10, INT32_MAX, clamp=False)
        val_clamped = ref_mad_i32_i16(1, 10, INT32_MAX, clamp=True)
        self.assertEqual(val_clamped, INT32_MAX)
        self.assertNotEqual(val_unclamped, val_clamped)

        # Underflow with clamp clamps to INT32_MIN
        val_unclamped_under = ref_mad_i32_i16(-1, 10, INT32_MIN, clamp=False)
        val_clamped_under = ref_mad_i32_i16(-1, 10, INT32_MIN, clamp=True)
        self.assertEqual(val_clamped_under, to_u32(INT32_MIN))
        self.assertNotEqual(val_unclamped_under, val_clamped_under)

    def test_op_sel_halfword_selection(self) -> None:
        # Packed dword: high 16 = 20, low 16 = 10 -> 0x0014_000A
        packed_s0 = (20 << 16) | 10
        # Packed dword: high 16 = -3, low 16 = 5 -> 0xFFFD_0005
        packed_s1 = (to_u16(-3) << 16) | 5

        # op_sel = (0, 0): uses low words (10 * 5 = 50)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(0, 0)), 50)

        # op_sel = (1, 0): uses high word of s0, low of s1 (20 * 5 = 100)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(1, 0)), 100)

        # op_sel = (0, 1): uses low of s0, high of s1 (10 * -3 = -30)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(0, 1)), to_u32(-30))

        # op_sel = (1, 1): uses high words (20 * -3 = -60)
        self.assertEqual(ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(1, 1)), to_u32(-60))

    def test_architectural_register_file(self) -> None:
        rf = RegisterFile()
        rf.write_vgpr(0, 0x1234_5678)
        self.assertEqual(rf.read_vgpr(0), 0x1234_5678)

        rf.write_sgpr(10, 0x9ABC_DEF0)
        self.assertEqual(rf.read_sgpr(10), 0x9ABC_DEF0)

        with self.assertRaises(IndexError):
            rf.read_vgpr(256)
        with self.assertRaises(IndexError):
            rf.read_sgpr(106)


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestSymbolicAST(unittest.TestCase):
    """Validation and syntax testing for AST nodes and expressions."""

    def test_operand_creation(self) -> None:
        v0 = Operand.vgpr(0)
        self.assertEqual(v0.reg_type, RegType.VGPR)
        self.assertEqual(v0.index, 0)
        self.assertEqual(v0.disassemble(), "v0")

        s5 = Operand.sgpr(5)
        self.assertEqual(s5.disassemble(), "s5")

        lit = Operand.literal(0x1234)
        self.assertEqual(lit.disassemble(), "0x00001234")

        inline = Operand.inline_const(42)
        self.assertEqual(inline.disassemble(), "42")

    def test_instruction_creation(self) -> None:
        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(100),
            clamp=True,
            op_sel=(1, 0),
        )
        self.assertEqual(instr.disassemble(), "v_mad_i32_i16 v0, v1, s2, 0x00000064 clamp op_sel:[1,0]")

    def test_instruction_destination_vgpr_validation(self) -> None:
        with self.assertRaises(TypeError):
            VMadI32I16Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_parse_assembly_roundtrip(self) -> None:
        asm1 = "v_mad_i32_i16 v0, v1, s2, v3"
        instr1 = parse_v_mad_i32_i16(asm1)
        self.assertEqual(instr1.disassemble(), asm1)

        asm2 = "v_mad_i32_i16 v10, v2, v3, 0x0000000A clamp op_sel:[1,1]"
        instr2 = parse_v_mad_i32_i16(asm2)
        self.assertEqual(instr2.disassemble(), asm2)
        self.assertTrue(instr2.clamp)
        self.assertEqual(instr2.op_sel, (1, 1))


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Execution tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarState()

    def test_scalar_interpreter_basic_registers(self) -> None:
        self.state.rf.write_vgpr(1, 10)
        self.state.rf.write_sgpr(2, to_u32(-20))
        self.state.rf.write_vgpr(3, 100)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.vgpr(3),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, to_u32(-100))
        self.assertEqual(self.state.rf.read_vgpr(0), to_u32(-100))

    def test_scalar_interpreter_with_clamp(self) -> None:
        self.state.rf.write_vgpr(1, 1000)
        self.state.rf.write_vgpr(2, 1000)
        self.state.rf.write_vgpr(3, INT32_MAX)

        instr_clamp = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
            clamp=True,
        )

        res = self.interp.step(instr_clamp, self.state)
        self.assertEqual(res, INT32_MAX)
        self.assertEqual(self.state.rf.read_vgpr(0), INT32_MAX)

    def test_scalar_interpreter_sequential_chaining(self) -> None:
        self.state.rf.write_vgpr(0, 0)
        self.state.rf.write_vgpr(1, 5)
        self.state.rf.write_vgpr(2, 6)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        self.interp.step(instr, self.state)
        self.assertEqual(self.state.rf.read_vgpr(0), 30)

        self.interp.step(instr, self.state)
        self.assertEqual(self.state.rf.read_vgpr(0), 60)


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence and EXEC Masking
# ==============================================================================

class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Multi-lane divergence and Wave32/Wave64 EXEC mask tests."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.vgprs[lane][1] = lane
            state.vgprs[lane][2] = 2
            state.vgprs[lane][3] = 100

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            expected = lane * 2 + 100
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][0], expected)

    def test_wave64_divergence(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE64, exec_mask=(1 << 64) - 1)
        for lane in range(64):
            state.vgprs[lane][1] = lane
            state.vgprs[lane][2] = -1
            state.vgprs[lane][3] = 0

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        for lane in range(64):
            expected = to_u32(-lane)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][0], expected)

    def test_partial_exec_mask_preserves_inactive_lanes(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0x0000_000F)
        sentinel = 0xDEAD_BEEF
        for lane in range(32):
            state.vgprs[lane][0] = sentinel
            state.vgprs[lane][1] = 10
            state.vgprs[lane][2] = 10
            state.vgprs[lane][3] = 0

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        for lane in range(4):
            self.assertEqual(results[lane], 100)
            self.assertEqual(state.vgprs[lane][0], 100)

        for lane in range(4, 32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.vgprs[lane][0], sentinel)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0)
        sentinel = 0xCAFE_BABE
        for lane in range(32):
            state.vgprs[lane][0] = sentinel

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.literal(5),
            src1=Operand.literal(5),
            src2=Operand.literal(5),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.vgprs[lane][0], sentinel)


# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Register aliasing, overlapping destination/source, and immutability tests."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_accumulator_aliasing_dst_equals_src2(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(0, 500)
        state.rf.write_vgpr(1, 10)
        state.rf.write_vgpr(2, 20)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 700)
        self.assertEqual(state.rf.read_vgpr(0), 700)
        self.assertEqual(state.rf.read_vgpr(1), 10)
        self.assertEqual(state.rf.read_vgpr(2), 20)

    def test_multiplier_aliasing_dst_equals_src0(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(0, 15)
        state.rf.write_vgpr(1, 4)
        state.rf.write_vgpr(2, 100)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 160)
        self.assertEqual(state.rf.read_vgpr(0), 160)
        self.assertEqual(state.rf.read_vgpr(1), 4)
        self.assertEqual(state.rf.read_vgpr(2), 100)

    def test_multiplicand_aliasing_dst_equals_src1(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(0, 7)
        state.rf.write_vgpr(1, 9)
        state.rf.write_vgpr(2, 50)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(2),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 113)
        self.assertEqual(state.rf.read_vgpr(0), 113)

    def test_total_overlap_all_operands_same_register(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(0, 6)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(0),
        )

        res = self.scalar_interp.step(instr, state)
        self.assertEqual(res, 42)
        self.assertEqual(state.rf.read_vgpr(0), 42)

    def test_unaliased_operands_immutability(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(10, 123)
        state.rf.write_vgpr(11, 456)
        state.rf.write_vgpr(12, 789)
        state.rf.write_sgpr(5, 0xCAFEBABE)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(20),
            src0=Operand.vgpr(10),
            src1=Operand.vgpr(11),
            src2=Operand.vgpr(12),
        )

        self.scalar_interp.step(instr, state)
        self.assertEqual(state.rf.read_vgpr(10), 123)
        self.assertEqual(state.rf.read_vgpr(11), 456)
        self.assertEqual(state.rf.read_vgpr(12), 789)
        self.assertEqual(state.rf.read_sgpr(5), 0xCAFEBABE)

    def test_simd_inplace_multi_iteration_accumulation(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0x00FF_00FF)
        sentinel = 0x8888_7777
        for lane in range(32):
            if bool(state.exec_mask & (1 << lane)):
                state.write_lane(Operand.vgpr(0), lane, 0)
                state.write_lane(Operand.vgpr(1), lane, lane)
                state.write_lane(Operand.vgpr(2), lane, 10)
            else:
                state.write_lane(Operand.vgpr(0), lane, sentinel)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        for it in range(1, 4):
            self.simd_interp.step(instr, state)
            for lane in range(32):
                if bool(state.exec_mask & (1 << lane)):
                    expected = (lane * 10) * it
                    self.assertEqual(state.vgprs[lane][0], expected)
                else:
                    self.assertEqual(state.vgprs[lane][0], sentinel)


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

def compute_mad_i32_i16_status(
    s0: int,
    s1: int,
    s2: int,
    clamp: bool = False,
    op_sel: tuple[int, int] = (0, 0),
) -> dict[str, bool]:
    """Compute arithmetic condition codes and status flags for v_mad_i32_i16."""
    s0_16 = (s0 >> 16) if op_sel[0] else s0
    s1_16 = (s1 >> 16) if op_sel[1] else s1

    prod_s32 = ref_mul_i16(s0_16, s1_16)
    acc_s32 = to_i32(s2)
    raw_sum = prod_s32 + acc_s32

    prod_u32 = to_u32(prod_s32)
    acc_u32 = to_u32(acc_s32)
    carry = (prod_u32 + acc_u32) >= MODULO32

    clamped = (raw_sum < INT32_MIN) or (raw_sum > INT32_MAX)
    final_val = clamp_i32(raw_sum) if clamp else raw_sum
    final_u32 = to_u32(final_val)

    zero = (final_u32 == 0)
    negative = bool(final_u32 & SIGN32)
    overflow = bool(((prod_s32 ^ final_val) & (acc_s32 ^ final_val) & SIGN32) != 0)

    return {
        "zero": zero,
        "negative": negative,
        "carry": carry,
        "overflow": overflow,
        "clamped": clamped,
        "prod_zero": (prod_s32 == 0),
    }


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Exhaustive verification of condition codes, carry bits, and status flags."""

    def test_zero_flag(self) -> None:
        status = compute_mad_i32_i16_status(-50, 20, 1000)
        self.assertTrue(status["zero"])
        self.assertFalse(status["negative"])

        status_nz = compute_mad_i32_i16_status(-50, 20, 1001)
        self.assertFalse(status_nz["zero"])

    def test_negative_flag(self) -> None:
        status_neg = compute_mad_i32_i16_status(100, 100, -20000)
        self.assertTrue(status_neg["negative"])

        status_pos = compute_mad_i32_i16_status(100, 100, 0)
        self.assertFalse(status_pos["negative"])

    def test_carry_boundary(self) -> None:
        status_carry = compute_mad_i32_i16_status(-1, 1, 1)
        self.assertTrue(status_carry["carry"])

        status_no_carry = compute_mad_i32_i16_status(0, 0, UINT32_MAX)
        self.assertFalse(status_no_carry["carry"])

    def test_overflow_and_clamp_detection(self) -> None:
        status_ov = compute_mad_i32_i16_status(INT16_MIN, INT16_MIN, 0x4000_0000)
        self.assertTrue(status_ov["overflow"])
        self.assertTrue(status_ov["negative"])
        self.assertTrue(status_ov["clamped"])


# ==============================================================================
# Unit Tests: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754EdgeCasesAndBoundaryExtremes(unittest.TestCase):
    """Raw IEEE 754 bit pattern preservation and extreme arithmetic boundaries."""

    def test_f16_signed_zero_distinction_no_flush(self) -> None:
        pos_zero = 0x0000
        neg_zero = 0x8000

        res_pos = ref_mad_i32_i16(pos_zero, 1, 0)
        res_neg = ref_mad_i32_i16(neg_zero, 1, 0)
        self.assertEqual(res_pos, 0)
        self.assertEqual(res_neg, to_u32(INT16_MIN))
        self.assertNotEqual(res_pos, res_neg)

    def test_f16_nan_and_infinity_preservation(self) -> None:
        inf_pos = 0x7C00
        inf_neg = 0xFC00
        nan_val = 0x7E00

        self.assertEqual(ref_mad_i32_i16(inf_pos, 1, 0), 31744)
        self.assertEqual(ref_mad_i32_i16(inf_neg, 1, 0), to_u32(-1024))
        self.assertEqual(ref_mad_i32_i16(nan_val, 1, 0), 32256)

    def test_f16_subnormals_treated_as_integers(self) -> None:
        self.assertEqual(ref_mad_i32_i16(0x0001, 10, 5), 15)
        self.assertEqual(ref_mad_i32_i16(0x03FF, 2, 0), 1023 * 2)

    def test_f32_float_bit_patterns_in_accumulator(self) -> None:
        qnan32 = 0x7FC0_0000
        neg_zero32 = 0x8000_0000

        self.assertEqual(ref_mad_i32_i16(0, 0, qnan32), qnan32)
        self.assertEqual(ref_mad_i32_i16(0, 0, neg_zero32), neg_zero32)

    def test_extreme_boundary_min_times_min_plus_min32(self) -> None:
        res = ref_mad_i32_i16(INT16_MIN, INT16_MIN, INT32_MIN)
        self.assertEqual(res, 0xC000_0000)

    def test_extreme_boundary_max_times_max_plus_max32_with_clamp(self) -> None:
        res_unclamped = ref_mad_i32_i16(INT16_MAX, INT16_MAX, INT32_MAX, clamp=False)
        res_clamped = ref_mad_i32_i16(INT16_MAX, INT16_MAX, INT32_MAX, clamp=True)

        expected_unclamped = to_u32(1073676289 + INT32_MAX)
        self.assertEqual(res_unclamped, expected_unclamped)
        self.assertEqual(res_clamped, INT32_MAX)

    def test_extreme_boundary_min_times_max_plus_min32_with_clamp(self) -> None:
        res_unclamped = ref_mad_i32_i16(INT16_MIN, INT16_MAX, INT32_MIN, clamp=False)
        res_clamped = ref_mad_i32_i16(INT16_MIN, INT16_MAX, INT32_MIN, clamp=True)

        expected_unclamped = to_u32(-1073709056 + INT32_MIN)
        self.assertEqual(res_unclamped, expected_unclamped)
        self.assertEqual(res_clamped, to_u32(INT32_MIN))


# ==============================================================================
# Unit Tests: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Negative controls, parser syntax rejection, and mutant kill verification."""

    def test_parser_unsupported_opcode_rejected(self) -> None:
        invalid_opcodes = [
            "v_mad_i64_i32 v0, v1, v2, v3",
            "v_add_i32 v0, v1, v2, v3",
            "s_mad_i32_i16 v0, v1, v2, v3",
        ]
        for line in invalid_opcodes:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    parse_v_mad_i32_i16(line)

    def test_parser_invalid_operand_counts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_v_mad_i32_i16("v_mad_i32_i16 v0, v1, v2")
        with self.assertRaises(ValueError):
            parse_v_mad_i32_i16("v_mad_i32_i16 v0, v1, v2, v3, v4")

    def test_ast_type_and_semantic_rejections(self) -> None:
        with self.assertRaises(TypeError):
            VMadI32I16Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

        with self.assertRaises(ValueError):
            Operand.vgpr(256)

        with self.assertRaises(ValueError):
            Operand.sgpr(106)

    def test_mutation_unsigned_multiplication_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int) -> int:
            u0 = s0 & MASK16
            u1 = s1 & MASK16
            return to_u32((u0 * u1) + to_i32(s2))

        s0, s1, s2 = -10, 20, 0
        gold = ref_mad_i32_i16(s0, s1, s2)
        mut = mutant_mad(s0, s1, s2)
        self.assertNotEqual(gold, mut)

    def test_mutation_omitted_accumulator_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int) -> int:
            return to_u32(ref_mul_i16(s0, s1))

        s0, s1, s2 = 5, 5, 100
        gold = ref_mad_i32_i16(s0, s1, s2)
        mut = mutant_mad(s0, s1, s2)
        self.assertEqual(gold, 125)
        self.assertEqual(mut, 25)
        self.assertNotEqual(gold, mut)

    def test_mutation_clamp_ignored_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int, clamp: bool = False) -> int:
            return to_u32(ref_mul_i16(s0, s1) + to_i32(s2))

        gold = ref_mad_i32_i16(1000, 1000, INT32_MAX, clamp=True)
        mut = mutant_mad(1000, 1000, INT32_MAX, clamp=True)
        self.assertEqual(gold, INT32_MAX)
        self.assertNotEqual(gold, mut)

    def test_mutation_op_sel_ignored_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int, op_sel: tuple[int, int] = (0, 0)) -> int:
            return ref_mad_i32_i16(s0, s1, s2, op_sel=(0, 0))

        packed_s0 = (100 << 16) | 10
        packed_s1 = 5
        gold = ref_mad_i32_i16(packed_s0, packed_s1, 0, op_sel=(1, 0))
        mut = mutant_mad(packed_s0, packed_s1, 0, op_sel=(1, 0))
        self.assertEqual(gold, 500)
        self.assertEqual(mut, 50)
        self.assertNotEqual(gold, mut)

    def test_mutation_simd_exec_mask_bypass_killed(self) -> None:
        class BuggySIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VMadI32I16Instruction, state: SIMDState) -> list[int | None]:
                results: list[int | None] = []
                for lane in range(state.num_lanes):
                    s0 = state.read_lane(instr.src0, lane)
                    s1 = state.read_lane(instr.src1, lane)
                    s2 = state.read_lane(instr.src2, lane)
                    res = ref_mad_i32_i16(s0, s1, s2, clamp=instr.clamp, op_sel=instr.op_sel)
                    state.write_lane(instr.dst, lane, res)
                    results.append(res)
                return results

        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0x0000_0000)
        sentinel = 0xAAAA_BBBB
        for lane in range(32):
            state.write_lane(Operand.vgpr(0), lane, sentinel)

        instr = VMadI32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.literal(1),
            src1=Operand.literal(1),
            src2=Operand.literal(1),
        )

        buggy = BuggySIMDInterpreter()
        buggy.step(instr, state)

        corrupted = [lane for lane in range(32) if state.vgprs[lane][0] != sentinel]
        self.assertEqual(len(corrupted), 32)


if __name__ == "__main__":
    unittest.main()
