from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mad_i64_i32.

AMD RDNA2 VALU Instruction Set Architecture:
  v_mad_i64_i32 dst, sdst, src0, src1, src2

Format:
  VOP3B / VOP3 (3-operand vector ALU instruction with scalar destination).

Operation:
  Multiplies two signed 32-bit integers, accumulates a 64-bit integer, and writes
  the 64-bit result to a vector register pair while generating a carry-out bitmask
  in a scalar register:
    prod64  = int64(src0.i32) * int64(src1.i32)
    sum64   = uint64(prod64) + uint64(src2.i64)
    dst.i64 = sum64 & 0xFFFFFFFFFFFFFFFF
    sdst    = 1 if (uint64(prod64) + uint64(src2.i64) >= 2^64) else 0

Operands:
  - dst: 64-bit vector register pair v[d:d+1] (d in [0, 254]).
  - sdst: Scalar condition / carry destination register (s[s:s+1] or VCC).
  - src0: 32-bit signed integer (VGPR, SGPR, literal, or inline constant).
  - src1: 32-bit signed integer (VGPR, SGPR, literal, or inline constant).
  - src2: 64-bit signed/unsigned integer (VGPR pair, SGPR pair, or literal).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
import struct
from typing import Optional, Union
import unittest

# Architectural Constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN32: int = 0x8000_0000
SIGN64: int = 0x8000_0000_0000_0000
MODULO32: int = 0x1_0000_0000
MODULO64: int = 0x1_0000_0000_0000_0000

INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
INT64_MIN: int = -0x8000_0000_0000_0000
INT64_MAX: int = 0x7FFF_FFFF_FFFF_FFFF
UINT32_MAX: int = 0xFFFF_FFFF
UINT64_MAX: int = 0xFFFF_FFFF_FFFF_FFFF

NUM_VGPRS: int = 256
NUM_SGPRS: int = 106


def to_i32(val: int) -> int:
    """Normalize integer to signed 32-bit integer in [-2^31, 2^31 - 1]."""
    val = val & MASK32
    return val - MODULO32 if (val & SIGN32) else val


def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword in [0, 2^32 - 1]."""
    return val & MASK32


def to_i64(val: int) -> int:
    """Normalize integer to signed 64-bit integer in [-2^63, 2^63 - 1]."""
    val = val & MASK64
    return val - MODULO64 if (val & SIGN64) else val


def to_u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword in [0, 2^64 - 1]."""
    return val & MASK64


def split_u64(val: int) -> tuple[int, int]:
    """Split 64-bit integer into (lo32, hi32) dwords."""
    u = val & MASK64
    return u & MASK32, (u >> 32) & MASK32


def combine_u32(lo: int, hi: int) -> int:
    """Combine two 32-bit dwords into unsigned 64-bit qword."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


# ==============================================================================
# Reference Mathematical Oracles
# ==============================================================================

def ref_mul_i32(s0: int, s1: int) -> int:
    """Compute exact signed 32-bit * 32-bit multiplication -> signed 64-bit product."""
    i0 = to_i32(s0)
    i1 = to_i32(s1)
    return i0 * i1


def ref_mad_i64_i32(s0: int, s1: int, s2: int) -> tuple[int, int]:
    """Pure-Python reference oracle for v_mad_i64_i32: (dst_u64, carry_out).

    Computes:
      prod64 = int64(s0.i32) * int64(s1.i32)
      raw_sum = (prod64 & MASK64) + (s2 & MASK64)
      dst = raw_sum & MASK64
      carry = 1 if raw_sum >= 2^64 else 0
    """
    prod = ref_mul_i32(s0, s1)
    prod_u64 = prod & MASK64
    s2_u64 = s2 & MASK64
    raw_sum = prod_u64 + s2_u64
    dst = raw_sum & MASK64
    carry = 1 if raw_sum >= MODULO64 else 0
    return dst, carry


def ref_mad_i64_i32_split(s0: int, s1: int, s2_lo: int, s2_hi: int) -> tuple[int, int, int]:
    """Reference mathematical oracle with split 32-bit dwords for s2.

    Returns (dst_lo, dst_hi, carry_out).
    """
    s2 = combine_u32(s2_lo, s2_hi)
    dst_u64, carry = ref_mad_i64_i32(s0, s1, s2)
    lo, hi = split_u64(dst_u64)
    return lo, hi, carry


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
    is_64bit: bool = False
    literal_val: int | None = None

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            max_idx = NUM_VGPRS - 2 if self.is_64bit else NUM_VGPRS - 1
            if not (0 <= self.index <= max_idx):
                raise ValueError(
                    f"VGPR {'pair ' if self.is_64bit else ''}index {self.index} out of range [0, {max_idx}]"
                )
        elif self.reg_type == RegType.SGPR:
            max_idx = NUM_SGPRS - 2 if self.is_64bit else NUM_SGPRS - 1
            if not (0 <= self.index <= max_idx):
                raise ValueError(
                    f"SGPR {'pair ' if self.is_64bit else ''}index {self.index} out of range [0, {max_idx}]"
                )
        elif self.reg_type == RegType.LITERAL:
            if self.literal_val is None:
                raise ValueError("Literal operand requires literal_val")
            limit = MASK64 if self.is_64bit else MASK32
            if not (0 <= self.literal_val <= limit):
                raise ValueError(
                    f"Literal value {self.literal_val:#x} out of range [0, {limit:#x}]"
                )
        elif self.reg_type == RegType.INLINE_CONST:
            if self.literal_val is None:
                raise ValueError("Inline constant operand requires literal_val")

    @classmethod
    def vgpr(cls, index: int, is_64bit: bool = False) -> Operand:
        return cls(RegType.VGPR, index=index, is_64bit=is_64bit)

    @classmethod
    def sgpr(cls, index: int, is_64bit: bool = False) -> Operand:
        return cls(RegType.SGPR, index=index, is_64bit=is_64bit)

    @classmethod
    def literal(cls, value: int, is_64bit: bool = False) -> Operand:
        val = to_u64(value) if is_64bit else to_u32(value)
        return cls(RegType.LITERAL, is_64bit=is_64bit, literal_val=val)

    @classmethod
    def inline_const(cls, value: int) -> Operand:
        return cls(RegType.INLINE_CONST, literal_val=value)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v[{self.index}:{self.index + 1}]" if self.is_64bit else f"v{self.index}"
        if self.reg_type == RegType.SGPR:
            return f"s[{self.index}:{self.index + 1}]" if self.is_64bit else f"s{self.index}"
        if self.reg_type == RegType.LITERAL:
            assert self.literal_val is not None
            return f"0x{self.literal_val:016X}" if self.is_64bit else f"0x{self.literal_val:08X}"
        if self.reg_type == RegType.INLINE_CONST:
            assert self.literal_val is not None
            return str(self.literal_val)
        raise ValueError(f"Unknown RegType: {self.reg_type}")


@dataclass(frozen=True)
class VMadI64I32Instruction:
    """Symbolic AST node for RDNA2 v_mad_i64_i32 instruction."""
    dst: Operand
    sdst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if not self.dst.is_64bit:
            raise ValueError("Destination VGPR must be a 64-bit register pair v[d:d+1]")
        if self.sdst.reg_type != RegType.SGPR:
            raise TypeError(f"Scalar destination sdst must be an SGPR, got {self.sdst.reg_type}")
        if self.src0.is_64bit:
            raise ValueError("src0 must be a 32-bit operand")
        if self.src1.is_64bit:
            raise ValueError("src1 must be a 32-bit operand")
        if not self.src2.is_64bit:
            raise ValueError("src2 must be a 64-bit operand (register pair or 64-bit literal)")

    def disassemble(self) -> str:
        return (
            f"v_mad_i64_i32 {self.dst.disassemble()}, {self.sdst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        )


def parse_v_mad_i64_i32(asm_line: str) -> VMadI64I32Instruction:
    """Parse assembly string into VMadI64I32Instruction AST node."""
    line = asm_line.strip()
    if not line.startswith("v_mad_i64_i32"):
        raise ValueError(f"Expected opcode 'v_mad_i64_i32', got '{line}'")

    args_part = line[len("v_mad_i64_i32"):].strip()
    tokens = [tok.strip() for tok in args_part.split(",")]
    if len(tokens) != 5:
        raise ValueError(f"v_mad_i64_i32 requires 5 operands, got {len(tokens)}: {tokens}")

    def parse_op(tok: str, expect_64bit: bool) -> Operand:
        # Check VGPR pair: v[x:y]
        m_vgpr_pair = re.fullmatch(r"v\[(\d+):(\d+)\]", tok)
        if m_vgpr_pair:
            lo, hi = int(m_vgpr_pair.group(1)), int(m_vgpr_pair.group(2))
            if hi != lo + 1:
                raise ValueError(f"Invalid VGPR pair span: {tok}")
            return Operand.vgpr(lo, is_64bit=True)

        # Single VGPR: vx
        m_vgpr = re.fullmatch(r"v(\d+)", tok)
        if m_vgpr:
            idx = int(m_vgpr.group(1))
            return Operand.vgpr(idx, is_64bit=expect_64bit)

        # Check SGPR pair: s[x:y]
        m_sgpr_pair = re.fullmatch(r"s\[(\d+):(\d+)\]", tok)
        if m_sgpr_pair:
            lo, hi = int(m_sgpr_pair.group(1)), int(m_sgpr_pair.group(2))
            if hi != lo + 1:
                raise ValueError(f"Invalid SGPR pair span: {tok}")
            return Operand.sgpr(lo, is_64bit=True)

        # Single SGPR: sx
        m_sgpr = re.fullmatch(r"s(\d+)", tok)
        if m_sgpr:
            idx = int(m_sgpr.group(1))
            return Operand.sgpr(idx, is_64bit=expect_64bit)

        # Hex literal
        if tok.startswith("0x") or tok.startswith("-0x"):
            val = int(tok, 16)
            return Operand.literal(val, is_64bit=expect_64bit)

        # Integer literal / inline constant
        try:
            val = int(tok, 10)
            if -16 <= val <= 64 and not expect_64bit:
                return Operand.inline_const(val)
            return Operand.literal(val, is_64bit=expect_64bit)
        except ValueError:
            pass

        raise ValueError(f"Cannot parse operand token: '{tok}'")

    dst = parse_op(tokens[0], expect_64bit=True)
    sdst = parse_op(tokens[1], expect_64bit=True)
    src0 = parse_op(tokens[2], expect_64bit=False)
    src1 = parse_op(tokens[3], expect_64bit=False)
    src2 = parse_op(tokens[4], expect_64bit=True)

    return VMadI64I32Instruction(dst=dst, sdst=sdst, src0=src0, src1=src1, src2=src2)


# ==============================================================================
# Architectural Hardware Model
# ==============================================================================

@dataclass
class RegisterFile:
    """Architectural model representing RDNA2 VGPRs and SGPRs for scalar simulation."""
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

    def read_vgpr_64(self, idx: int) -> int:
        if not (0 <= idx < NUM_VGPRS - 1):
            raise IndexError(f"VGPR 64-bit pair index {idx} out of range [0, {NUM_VGPRS - 2}]")
        return combine_u32(self.vgprs[idx], self.vgprs[idx + 1])

    def write_vgpr_64(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_VGPRS - 1):
            raise IndexError(f"VGPR 64-bit pair index {idx} out of range [0, {NUM_VGPRS - 2}]")
        lo, hi = split_u64(val)
        self.vgprs[idx] = lo
        self.vgprs[idx + 1] = hi

    def read_sgpr(self, idx: int) -> int:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        return self.sgprs[idx] & MASK32

    def write_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_SGPRS):
            raise IndexError(f"SGPR index {idx} out of range [0, {NUM_SGPRS - 1}]")
        self.sgprs[idx] = val & MASK32

    def read_sgpr_64(self, idx: int) -> int:
        if not (0 <= idx < NUM_SGPRS - 1):
            raise IndexError(f"SGPR 64-bit pair index {idx} out of range [0, {NUM_SGPRS - 2}]")
        return combine_u32(self.sgprs[idx], self.sgprs[idx + 1])

    def write_sgpr_64(self, idx: int, val: int) -> None:
        if not (0 <= idx < NUM_SGPRS - 1):
            raise IndexError(f"SGPR 64-bit pair index {idx} out of range [0, {NUM_SGPRS - 2}]")
        lo, hi = split_u64(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarState:
    """Scalar processor execution state."""
    rf: RegisterFile = field(default_factory=RegisterFile)

    def read_operand_32(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.rf.read_vgpr(op.index)
        if op.reg_type == RegType.SGPR:
            return self.rf.read_sgpr(op.index)
        if op.reg_type in (RegType.LITERAL, RegType.INLINE_CONST):
            assert op.literal_val is not None
            return to_u32(op.literal_val)
        raise ValueError(f"Unsupported operand type for 32-bit read: {op.reg_type}")

    def read_operand_64(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.rf.read_vgpr_64(op.index)
        if op.reg_type == RegType.SGPR:
            return self.rf.read_sgpr_64(op.index)
        if op.reg_type == RegType.LITERAL:
            assert op.literal_val is not None
            return to_u64(op.literal_val)
        if op.reg_type == RegType.INLINE_CONST:
            assert op.literal_val is not None
            return to_u64(to_i32(op.literal_val))
        raise ValueError(f"Unsupported operand type for 64-bit read: {op.reg_type}")

    def write_operand_64(self, op: Operand, val: int) -> None:
        if op.reg_type == RegType.VGPR:
            self.rf.write_vgpr_64(op.index, val)
        elif op.reg_type == RegType.SGPR:
            self.rf.write_sgpr_64(op.index, val)
        else:
            raise TypeError(f"Cannot write 64-bit value to {op.reg_type}")

    def write_operand_32(self, op: Operand, val: int) -> None:
        if op.reg_type == RegType.VGPR:
            self.rf.write_vgpr(op.index, val)
        elif op.reg_type == RegType.SGPR:
            self.rf.write_sgpr(op.index, val)
        else:
            raise TypeError(f"Cannot write 32-bit value to {op.reg_type}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_mad_i64_i32."""

    def step(self, instr: VMadI64I32Instruction, state: ScalarState) -> tuple[int, int]:
        s0 = state.read_operand_32(instr.src0)
        s1 = state.read_operand_32(instr.src1)
        s2 = state.read_operand_64(instr.src2)

        dst_u64, carry = ref_mad_i64_i32(s0, s1, s2)

        # Write 64-bit destination vector register pair
        state.write_operand_64(instr.dst, dst_u64)

        # Write scalar carry output to sdst
        if instr.sdst.is_64bit:
            state.write_operand_64(instr.sdst, carry)
        else:
            state.write_operand_32(instr.sdst, carry)

        return dst_u64, carry


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
            self.exec_mask &= MASK64

    def read_lane_32(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        if op.reg_type in (RegType.LITERAL, RegType.INLINE_CONST):
            assert op.literal_val is not None
            return to_u32(op.literal_val)
        raise ValueError(f"Unsupported operand type for 32-bit read: {op.reg_type}")

    def read_lane_64(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[lane][op.index]
            hi = self.vgprs[lane][op.index + 1]
            return combine_u32(lo, hi)
        if op.reg_type == RegType.SGPR:
            lo = self.sgprs[op.index]
            hi = self.sgprs[op.index + 1]
            return combine_u32(lo, hi)
        if op.reg_type == RegType.LITERAL:
            assert op.literal_val is not None
            return to_u64(op.literal_val)
        if op.reg_type == RegType.INLINE_CONST:
            assert op.literal_val is not None
            return to_u64(to_i32(op.literal_val))
        raise ValueError(f"Unsupported operand type for 64-bit read: {op.reg_type}")

    def write_lane_64(self, op: Operand, lane: int, val: int) -> None:
        if op.reg_type == RegType.VGPR:
            lo, hi = split_u64(val)
            self.vgprs[lane][op.index] = lo
            self.vgprs[lane][op.index + 1] = hi
        else:
            raise TypeError(f"Lane write target must be VGPR, got {op.reg_type}")

    def write_sdst_mask(self, op: Operand, mask: int) -> None:
        if op.reg_type == RegType.SGPR:
            if op.is_64bit:
                lo, hi = split_u64(mask)
                self.sgprs[op.index] = lo
                self.sgprs[op.index + 1] = hi
            else:
                self.sgprs[op.index] = mask & MASK32
        else:
            raise TypeError(f"sdst must be SGPR, got {op.reg_type}")


class SIMDInterpreter:
    """SIMD wavefront instruction interpreter handling divergence and EXEC masking."""

    def step(self, instr: VMadI64I32Instruction, state: SIMDState) -> tuple[list[int | None], int]:
        per_lane_results: list[int | None] = [None] * state.num_lanes
        carry_mask = 0

        for lane in range(state.num_lanes):
            lane_active = bool(state.exec_mask & (1 << lane))
            if lane_active:
                s0 = state.read_lane_32(instr.src0, lane)
                s1 = state.read_lane_32(instr.src1, lane)
                s2 = state.read_lane_64(instr.src2, lane)

                dst_val, carry = ref_mad_i64_i32(s0, s1, s2)
                state.write_lane_64(instr.dst, lane, dst_val)
                per_lane_results[lane] = dst_val
                if carry:
                    carry_mask |= (1 << lane)

        state.write_sdst_mask(instr.sdst, carry_mask)
        return per_lane_results, carry_mask


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle & Architectural Model
# ==============================================================================

class TestReferenceMathematicalOracle(unittest.TestCase):
    """Exhaustive mathematical verification of pure-Python reference oracle."""

    def test_zero_identities(self) -> None:
        self.assertEqual(ref_mad_i64_i32(0, 0, 0), (0, 0))
        self.assertEqual(ref_mad_i64_i32(0, 12345, 0), (0, 0))
        self.assertEqual(ref_mad_i64_i32(12345, 0, 0), (0, 0))
        self.assertEqual(ref_mad_i64_i32(0, 0, 0x1234_5678_9ABC_DEF0), (0x1234_5678_9ABC_DEF0, 0))

    def test_one_identity(self) -> None:
        self.assertEqual(ref_mad_i64_i32(1, 42, 0), (42, 0))
        self.assertEqual(ref_mad_i64_i32(42, 1, 100), (142, 0))

    def test_signed_multiplication_signs(self) -> None:
        dst, c = ref_mad_i64_i32(100, 200, 0)
        self.assertEqual(dst, 20000)
        self.assertEqual(c, 0)

        dst, c = ref_mad_i64_i32(100, -200, 0)
        expected = to_u64(-20000)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

        dst, c = ref_mad_i64_i32(-100, 200, 0)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

        dst, c = ref_mad_i64_i32(-100, -200, 0)
        self.assertEqual(dst, 20000)
        self.assertEqual(c, 0)

    def test_signed_boundary_products(self) -> None:
        dst, c = ref_mad_i64_i32(INT32_MIN, INT32_MIN, 0)
        self.assertEqual(dst, 0x4000_0000_0000_0000)
        self.assertEqual(c, 0)

        dst, c = ref_mad_i64_i32(INT32_MIN, 1, 0)
        self.assertEqual(dst, to_u64(INT32_MIN))
        self.assertEqual(c, 0)

        dst, c = ref_mad_i64_i32(INT32_MIN, -1, 0)
        self.assertEqual(dst, 0x8000_0000)
        self.assertEqual(c, 0)

        expected = (INT32_MAX * INT32_MAX) & MASK64
        dst, c = ref_mad_i64_i32(INT32_MAX, INT32_MAX, 0)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

        expected = (INT32_MIN * INT32_MAX) & MASK64
        dst, c = ref_mad_i64_i32(INT32_MIN, INT32_MAX, 0)
        self.assertEqual(dst, expected)
        self.assertEqual(c, 0)

    def test_commutativity_of_multiplication(self) -> None:
        operands = [
            (0, 0),
            (1, -1),
            (-500, 300),
            (INT32_MIN, INT32_MAX),
            (0x1234_5678, -0x1234_5678),
        ]
        s2_values = [0, 1, -1, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000]
        for s0, s1 in operands:
            for s2 in s2_values:
                with self.subTest(s0=s0, s1=s1, s2=s2):
                    res1, c1 = ref_mad_i64_i32(s0, s1, s2)
                    res2, c2 = ref_mad_i64_i32(s1, s0, s2)
                    self.assertEqual(res1, res2)
                    self.assertEqual(c1, c2)

    def test_carry_out_generation_boundary(self) -> None:
        dst, c = ref_mad_i64_i32(-1, 1, 1)
        self.assertEqual(dst, 0)
        self.assertEqual(c, 1)

        dst, c = ref_mad_i64_i32(-1, 1, 0)
        self.assertEqual(dst, MASK64)
        self.assertEqual(c, 0)

        dst, c = ref_mad_i64_i32(1, 1, UINT64_MAX)
        self.assertEqual(dst, 0)
        self.assertEqual(c, 1)

    def test_split_oracle_reconstruction(self) -> None:
        s0 = -12345
        s1 = 67890
        s2_lo = 0xDEAD_BEEF
        s2_hi = 0xCAFE_BABE
        dst_lo, dst_hi, carry = ref_mad_i64_i32_split(s0, s1, s2_lo, s2_hi)
        full_dst, full_c = ref_mad_i64_i32(s0, s1, combine_u32(s2_lo, s2_hi))
        self.assertEqual(combine_u32(dst_lo, dst_hi), full_dst)
        self.assertEqual(carry, full_c)

    def test_architectural_register_file(self) -> None:
        rf = RegisterFile()
        rf.write_vgpr(0, 0x12345678)
        rf.write_vgpr(1, 0x9ABCDEF0)
        self.assertEqual(rf.read_vgpr_64(0), 0x9ABC_DEF0_1234_5678)

        rf.write_vgpr_64(10, 0xFEDC_BA98_7654_3210)
        self.assertEqual(rf.read_vgpr(10), 0x7654_3210)
        self.assertEqual(rf.read_vgpr(11), 0xFEDC_BA98)

        rf.write_sgpr_64(2, 0x0123_4567_89AB_CDEF)
        self.assertEqual(rf.read_sgpr_64(2), 0x0123_4567_89AB_CDEF)

        with self.assertRaises(IndexError):
            rf.read_vgpr(256)
        with self.assertRaises(IndexError):
            rf.read_vgpr_64(255)
        with self.assertRaises(IndexError):
            rf.read_sgpr(106)
        with self.assertRaises(IndexError):
            rf.read_sgpr_64(105)


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestSymbolicAST(unittest.TestCase):
    """Validation and syntax testing for AST nodes and operand expressions."""

    def test_operand_creation_valid(self) -> None:
        v0_64 = Operand.vgpr(0, is_64bit=True)
        self.assertEqual(v0_64.reg_type, RegType.VGPR)
        self.assertEqual(v0_64.index, 0)
        self.assertTrue(v0_64.is_64bit)
        self.assertEqual(v0_64.disassemble(), "v[0:1]")

        s2_64 = Operand.sgpr(2, is_64bit=True)
        self.assertEqual(s2_64.disassemble(), "s[2:3]")

        v5 = Operand.vgpr(5, is_64bit=False)
        self.assertEqual(v5.disassemble(), "v5")

        lit32 = Operand.literal(0x12345678, is_64bit=False)
        self.assertEqual(lit32.disassemble(), "0x12345678")

        lit64 = Operand.literal(0x1234_5678_9ABC_DEF0, is_64bit=True)
        self.assertEqual(lit64.disassemble(), "0x123456789ABCDEF0")

        inline = Operand.inline_const(42)
        self.assertEqual(inline.disassemble(), "42")

    def test_operand_index_bounds(self) -> None:
        Operand.vgpr(254, is_64bit=True)
        with self.assertRaises(ValueError):
            Operand.vgpr(255, is_64bit=True)

        Operand.sgpr(104, is_64bit=True)
        with self.assertRaises(ValueError):
            Operand.sgpr(105, is_64bit=True)

    def test_instruction_creation_valid(self) -> None:
        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(2, is_64bit=True),
            src0=Operand.vgpr(4),
            src1=Operand.sgpr(6),
            src2=Operand.vgpr(8, is_64bit=True),
        )
        self.assertEqual(instr.disassemble(), "v_mad_i64_i32 v[0:1], s[2:3], v4, s6, v[8:9]")

    def test_instruction_destination_vgpr_validation(self) -> None:
        with self.assertRaises(TypeError):
            VMadI64I32Instruction(
                dst=Operand.sgpr(0, is_64bit=True),
                sdst=Operand.sgpr(2, is_64bit=True),
                src0=Operand.vgpr(4),
                src1=Operand.vgpr(5),
                src2=Operand.vgpr(6, is_64bit=True),
            )

        with self.assertRaises(ValueError):
            VMadI64I32Instruction(
                dst=Operand.vgpr(0, is_64bit=False),  # Not 64-bit pair
                sdst=Operand.sgpr(2, is_64bit=True),
                src0=Operand.vgpr(4),
                src1=Operand.vgpr(5),
                src2=Operand.vgpr(6, is_64bit=True),
            )

    def test_instruction_operand_width_validation(self) -> None:
        # src0 cannot be 64-bit
        with self.assertRaises(ValueError):
            VMadI64I32Instruction(
                dst=Operand.vgpr(0, is_64bit=True),
                sdst=Operand.sgpr(2, is_64bit=True),
                src0=Operand.vgpr(4, is_64bit=True),
                src1=Operand.vgpr(6),
                src2=Operand.vgpr(8, is_64bit=True),
            )
        # src2 must be 64-bit
        with self.assertRaises(ValueError):
            VMadI64I32Instruction(
                dst=Operand.vgpr(0, is_64bit=True),
                sdst=Operand.sgpr(2, is_64bit=True),
                src0=Operand.vgpr(4),
                src1=Operand.vgpr(6),
                src2=Operand.vgpr(8, is_64bit=False),
            )

    def test_parse_assembly_roundtrip(self) -> None:
        asm = "v_mad_i64_i32 v[0:1], s[2:3], v4, s6, v[8:9]"
        instr = parse_v_mad_i64_i32(asm)
        self.assertEqual(instr.disassemble(), asm)

        asm_lit = "v_mad_i64_i32 v[10:11], s[0:1], 42, -5, 0x0000000000001000"
        instr_lit = parse_v_mad_i64_i32(asm_lit)
        self.assertEqual(instr_lit.dst.index, 10)
        self.assertEqual(instr_lit.src0.literal_val, 42)
        self.assertEqual(instr_lit.src1.literal_val, -5)


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Execution tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarState()

    def test_scalar_interpreter_basic_registers(self) -> None:
        self.state.rf.write_vgpr(4, 100)
        self.state.rf.write_sgpr(6, 200)
        self.state.rf.write_vgpr_64(8, 50)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(2, is_64bit=True),
            src0=Operand.vgpr(4),
            src1=Operand.sgpr(6),
            src2=Operand.vgpr(8, is_64bit=True),
        )

        dst, carry = self.interp.step(instr, self.state)
        self.assertEqual(dst, 20050)
        self.assertEqual(carry, 0)
        self.assertEqual(self.state.rf.read_vgpr_64(0), 20050)
        self.assertEqual(self.state.rf.read_sgpr_64(2), 0)

    def test_scalar_interpreter_signed_negatives_and_literals(self) -> None:
        self.state.rf.write_vgpr(10, to_u32(-300))

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(20, is_64bit=True),
            sdst=Operand.sgpr(10, is_64bit=False),
            src0=Operand.vgpr(10),
            src1=Operand.literal(400),
            src2=Operand.literal(200000, is_64bit=True),
        )

        dst, carry = self.interp.step(instr, self.state)
        self.assertEqual(dst, 80000)
        self.assertEqual(carry, 1)
        self.assertEqual(self.state.rf.read_vgpr_64(20), 80000)
        self.assertEqual(self.state.rf.read_sgpr(10), 1)

    def test_scalar_interpreter_carry_generation(self) -> None:
        self.state.rf.write_vgpr(0, to_u32(-1))
        self.state.rf.write_vgpr(1, 1)
        self.state.rf.write_vgpr_64(2, 1)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(4, is_64bit=True),
            sdst=Operand.sgpr(0, is_64bit=True),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2, is_64bit=True),
        )

        dst, carry = self.interp.step(instr, self.state)
        self.assertEqual(dst, 0)
        self.assertEqual(carry, 1)
        self.assertEqual(self.state.rf.read_vgpr_64(4), 0)
        self.assertEqual(self.state.rf.read_sgpr_64(0), 1)

    def test_scalar_interpreter_sequential_chaining(self) -> None:
        self.state.rf.write_vgpr(2, 10)
        self.state.rf.write_vgpr(3, 20)
        self.state.rf.write_vgpr_64(0, 0)

        instr1 = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(2, is_64bit=True),
            src0=Operand.vgpr(2),
            src1=Operand.vgpr(3),
            src2=Operand.vgpr(0, is_64bit=True),
        )
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.rf.read_vgpr_64(0), 200)

        self.state.rf.write_vgpr(2, 30)
        self.state.rf.write_vgpr(3, 40)
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.rf.read_vgpr_64(0), 1400)


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence and EXEC Masking
# ==============================================================================

class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Multi-lane divergence and Wave32/Wave64 EXEC mask tests."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_all_lanes_active_divergent_operands(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.vgprs[lane][2] = lane
            state.vgprs[lane][3] = 2
            state.vgprs[lane][4] = 100
            state.vgprs[lane][5] = 0

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(0, is_64bit=False),
            src0=Operand.vgpr(2),
            src1=Operand.vgpr(3),
            src2=Operand.vgpr(4, is_64bit=True),
        )

        results, carry_mask = self.interp.step(instr, state)
        self.assertEqual(carry_mask, 0)
        for lane in range(32):
            expected = lane * 2 + 100
            self.assertEqual(results[lane], expected)
            lo = state.vgprs[lane][0]
            hi = state.vgprs[lane][1]
            self.assertEqual(combine_u32(lo, hi), expected)

    def test_wave64_divergence_and_carry_masking(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE64, exec_mask=MASK64)
        for lane in range(64):
            if lane % 2 == 0:
                state.vgprs[lane][2] = to_u32(-1)
                state.vgprs[lane][3] = 1
                state.vgprs[lane][4] = 1
                state.vgprs[lane][5] = 0
            else:
                state.vgprs[lane][2] = 1
                state.vgprs[lane][3] = 1
                state.vgprs[lane][4] = 0
                state.vgprs[lane][5] = 0

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(10, is_64bit=True),
            sdst=Operand.sgpr(2, is_64bit=True),
            src0=Operand.vgpr(2),
            src1=Operand.vgpr(3),
            src2=Operand.vgpr(4, is_64bit=True),
        )

        results, carry_mask = self.interp.step(instr, state)
        expected_mask = 0x5555_5555_5555_5555
        self.assertEqual(carry_mask, expected_mask)
        lo_carry = state.sgprs[2]
        hi_carry = state.sgprs[3]
        self.assertEqual(combine_u32(lo_carry, hi_carry), expected_mask)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0)
            else:
                self.assertEqual(results[lane], 1)

    def test_partial_exec_mask_preserves_inactive_lanes(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0x0000_000F)
        sentinel_lo = 0xDEAD_BEEF
        sentinel_hi = 0xCAFE_BABE

        for lane in range(32):
            state.vgprs[lane][0] = sentinel_lo
            state.vgprs[lane][1] = sentinel_hi
            state.vgprs[lane][2] = 10
            state.vgprs[lane][3] = 10
            state.vgprs[lane][4] = 0
            state.vgprs[lane][5] = 0

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(4, is_64bit=False),
            src0=Operand.vgpr(2),
            src1=Operand.vgpr(3),
            src2=Operand.vgpr(4, is_64bit=True),
        )

        results, carry_mask = self.interp.step(instr, state)
        for lane in range(4):
            self.assertEqual(results[lane], 100)
            self.assertEqual(state.vgprs[lane][0], 100)
            self.assertEqual(state.vgprs[lane][1], 0)

        for lane in range(4, 32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.vgprs[lane][0], sentinel_lo)
            self.assertEqual(state.vgprs[lane][1], sentinel_hi)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0x0)
        for lane in range(32):
            state.vgprs[lane][0] = 0x1111_1111
            state.vgprs[lane][1] = 0x2222_2222

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(0, is_64bit=False),
            src0=Operand.literal(5),
            src1=Operand.literal(5),
            src2=Operand.literal(5, is_64bit=True),
        )

        results, carry_mask = self.interp.step(instr, state)
        self.assertEqual(carry_mask, 0)
        self.assertEqual(state.sgprs[0], 0)
        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.vgprs[lane][0], 0x1111_1111)
            self.assertEqual(state.vgprs[lane][1], 0x2222_2222)


# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Register aliasing, overlapping destination/source, and immutability tests."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_inplace_accumulator_aliasing_dst_equals_src2(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr_64(0, 500)
        state.rf.write_vgpr(4, 50)
        state.rf.write_vgpr(5, 20)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(2, is_64bit=True),
            src0=Operand.vgpr(4),
            src1=Operand.vgpr(5),
            src2=Operand.vgpr(0, is_64bit=True),
        )

        dst, carry = self.scalar_interp.step(instr, state)
        self.assertEqual(dst, 1500)
        self.assertEqual(carry, 0)
        self.assertEqual(state.rf.read_vgpr_64(0), 1500)
        self.assertEqual(state.rf.read_vgpr(4), 50)
        self.assertEqual(state.rf.read_vgpr(5), 20)

    def test_low_word_destination_aliasing_dst_lo_equals_src0(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(0, 25)
        state.rf.write_vgpr(1, 0xAAAA_BBBB)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(0, is_64bit=False),
            src0=Operand.vgpr(0),
            src1=Operand.literal(10),
            src2=Operand.literal(1000, is_64bit=True),
        )

        dst, carry = self.scalar_interp.step(instr, state)
        self.assertEqual(dst, 1250)
        self.assertEqual(carry, 0)
        self.assertEqual(state.rf.read_vgpr(0), 1250)
        self.assertEqual(state.rf.read_vgpr(1), 0)

    def test_high_word_destination_aliasing_dst_hi_equals_src1(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(0, 0x1111_2222)
        state.rf.write_vgpr(1, 7)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(0, is_64bit=False),
            src0=Operand.literal(100),
            src1=Operand.vgpr(1),
            src2=Operand.literal(500, is_64bit=True),
        )

        dst, carry = self.scalar_interp.step(instr, state)
        self.assertEqual(dst, 1200)
        self.assertEqual(carry, 0)
        self.assertEqual(state.rf.read_vgpr(0), 1200)
        self.assertEqual(state.rf.read_vgpr(1), 0)

    def test_total_overlap_all_operands_in_dst_pair(self) -> None:
        state = ScalarState()
        v0_initial = 3
        v1_initial = to_u32(-4)
        state.rf.write_vgpr(0, v0_initial)
        state.rf.write_vgpr(1, v1_initial)

        s2_expected = combine_u32(v0_initial, v1_initial)
        expected_dst, expected_carry = ref_mad_i64_i32(3, -4, s2_expected)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(10, is_64bit=False),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(0, is_64bit=True),
        )

        dst, carry = self.scalar_interp.step(instr, state)
        self.assertEqual(dst, expected_dst)
        self.assertEqual(carry, expected_carry)
        self.assertEqual(state.rf.read_vgpr_64(0), expected_dst)

    def test_unaliased_operands_immutability(self) -> None:
        state = ScalarState()
        state.rf.write_vgpr(10, 123)
        state.rf.write_vgpr(11, 456)
        state.rf.write_vgpr_64(12, 0xCAFE_BABE_DEAD_BEEF)
        state.rf.write_sgpr(50, 0x1337_C0DE)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(20, is_64bit=True),
            sdst=Operand.sgpr(2, is_64bit=True),
            src0=Operand.vgpr(10),
            src1=Operand.vgpr(11),
            src2=Operand.vgpr(12, is_64bit=True),
        )

        self.scalar_interp.step(instr, state)
        self.assertEqual(state.rf.read_vgpr(10), 123)
        self.assertEqual(state.rf.read_vgpr(11), 456)
        self.assertEqual(state.rf.read_vgpr_64(12), 0xCAFE_BABE_DEAD_BEEF)
        self.assertEqual(state.rf.read_sgpr(50), 0x1337_C0DE)

    def test_simd_inplace_multi_iteration_accumulation(self) -> None:
        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0x00FF_00FF)
        sentinel = 0x8888_7777_6666_5555
        for lane in range(32):
            if bool(state.exec_mask & (1 << lane)):
                state.write_lane_64(Operand.vgpr(0, is_64bit=True), lane, 0)
                state.vgprs[lane][2] = lane
                state.vgprs[lane][3] = 10
            else:
                state.write_lane_64(Operand.vgpr(0, is_64bit=True), lane, sentinel)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(0, is_64bit=False),
            src0=Operand.vgpr(2),
            src1=Operand.vgpr(3),
            src2=Operand.vgpr(0, is_64bit=True),
        )

        for it in range(1, 4):
            self.simd_interp.step(instr, state)
            for lane in range(32):
                if bool(state.exec_mask & (1 << lane)):
                    expected = (lane * 10) * it
                    actual = combine_u32(state.vgprs[lane][0], state.vgprs[lane][1])
                    self.assertEqual(actual, expected)
                else:
                    actual = combine_u32(state.vgprs[lane][0], state.vgprs[lane][1])
                    self.assertEqual(actual, sentinel)


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

def compute_status_flags(s0: int, s1: int, s2: int) -> dict[str, bool]:
    """Compute condition codes and status flags for v_mad_i64_i32."""
    prod_s64 = ref_mul_i32(s0, s1)
    prod_u64 = prod_s64 & MASK64
    s2_u64 = s2 & MASK64
    raw_sum = prod_u64 + s2_u64
    dst_u64 = raw_sum & MASK64

    carry = (raw_sum >= MODULO64)
    zero = (dst_u64 == 0)
    negative = bool(dst_u64 & SIGN64)
    overflow = bool(((prod_u64 ^ dst_u64) & (s2_u64 ^ dst_u64) & SIGN64) != 0)

    prod_zero = (prod_s64 == 0)
    prod_negative = (prod_s64 < 0)

    return {
        "carry": carry,
        "zero": zero,
        "negative": negative,
        "overflow": overflow,
        "prod_zero": prod_zero,
        "prod_negative": prod_negative,
    }


def ref_mad_128_i32(s0: int, s1: int, s2_lo64: int, s3_hi64: int = 0) -> tuple[int, int]:
    """128-bit multi-precision multiply-accumulate using carry out."""
    dst_lo64, carry = ref_mad_i64_i32(s0, s1, s2_lo64)
    raw_hi = (s3_hi64 & MASK64) + carry
    dst_hi64 = raw_hi & MASK64
    return dst_lo64, dst_hi64


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Exhaustive verification of condition codes, carry bits, and status flags."""

    def test_carry_boundary_transition(self) -> None:
        flags1 = compute_status_flags(1, 1, UINT64_MAX - 1)
        self.assertFalse(flags1["carry"])
        self.assertFalse(flags1["zero"])

        flags2 = compute_status_flags(1, 1, UINT64_MAX)
        self.assertTrue(flags2["carry"])
        self.assertTrue(flags2["zero"])

    def test_zero_flag_detection(self) -> None:
        flags = compute_status_flags(-5, 5, 25)
        self.assertTrue(flags["zero"])
        self.assertFalse(flags["negative"])

        flags_nz = compute_status_flags(-5, 5, 24)
        self.assertFalse(flags_nz["zero"])

    def test_negative_sign_flag_detection(self) -> None:
        flags_neg = compute_status_flags(10, -20, 0)
        self.assertTrue(flags_neg["negative"])
        self.assertTrue(flags_neg["prod_negative"])

        flags_pos = compute_status_flags(10, 20, 0)
        self.assertFalse(flags_pos["negative"])
        self.assertFalse(flags_pos["prod_negative"])

    def test_signed_overflow_flag_detection(self) -> None:
        flags = compute_status_flags(INT32_MIN, INT32_MIN, 0x4000_0000_0000_0000)
        self.assertTrue(flags["overflow"])
        self.assertTrue(flags["negative"])
        self.assertFalse(flags["carry"])

        flags_no_ov = compute_status_flags(100, 200, 300)
        self.assertFalse(flags_no_ov["overflow"])

    def test_multi_precision_128bit_carry_propagation(self) -> None:
        s0 = -1
        s1 = 1
        s2_lo = 1
        s3_hi = 100

        dst_lo, dst_hi = ref_mad_128_i32(s0, s1, s2_lo, s3_hi)
        self.assertEqual(dst_lo, 0)
        self.assertEqual(dst_hi, 101)


# ==============================================================================
# Unit Tests: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BitPatternsAndBoundaryExtremes(unittest.TestCase):
    """Raw IEEE 754 bit pattern preservation and extreme arithmetic boundaries."""

    def test_float32_zero_distinction_no_flush(self) -> None:
        pos_zero = struct.unpack(">I", struct.pack(">f", 0.0))[0]
        neg_zero = struct.unpack(">I", struct.pack(">f", -0.0))[0]
        self.assertEqual(pos_zero, 0x0000_0000)
        self.assertEqual(neg_zero, 0x8000_0000)

        dst_pos, _ = ref_mad_i64_i32(pos_zero, 1, 0)
        dst_neg, _ = ref_mad_i64_i32(neg_zero, 1, 0)
        self.assertEqual(dst_pos, 0)
        self.assertEqual(dst_neg, to_u64(INT32_MIN))
        self.assertNotEqual(dst_pos, dst_neg)

    def test_float32_nan_and_infinity_bit_preservation(self) -> None:
        qnan_bits = 0x7FC0_0000
        snan_bits = 0x7F80_0001
        ninf_bits = 0xFF80_0000

        dst_qnan, _ = ref_mad_i64_i32(qnan_bits, 1, 0)
        self.assertEqual(dst_qnan, qnan_bits)

        dst_snan, _ = ref_mad_i64_i32(snan_bits, 1, 0)
        self.assertEqual(dst_snan, snan_bits)

        dst_ninf, _ = ref_mad_i64_i32(ninf_bits, 1, 0)
        self.assertEqual(dst_ninf, to_u64(-8388608))

    def test_float32_subnormals_treated_as_integers(self) -> None:
        subnormal_min = 0x0000_0001
        subnormal_max = 0x007F_FFFF

        dst, c = ref_mad_i64_i32(subnormal_min, 10, 5)
        self.assertEqual(dst, 15)
        self.assertEqual(c, 0)

        dst_max, _ = ref_mad_i64_i32(subnormal_max, 2, 0)
        self.assertEqual(dst_max, 8388607 * 2)

    def test_float64_double_bit_patterns_in_accumulator(self) -> None:
        double_qnan = 0x7FF8_0000_0000_0000
        double_neg_zero = 0x8000_0000_0000_0000

        dst, c = ref_mad_i64_i32(0, 0, double_qnan)
        self.assertEqual(dst, double_qnan)
        self.assertEqual(c, 0)

        dst, c = ref_mad_i64_i32(0, 0, double_neg_zero)
        self.assertEqual(dst, double_neg_zero)
        self.assertEqual(c, 0)

    def test_extreme_boundary_min_times_min_plus_min64(self) -> None:
        dst, c = ref_mad_i64_i32(INT32_MIN, INT32_MIN, INT64_MIN)
        self.assertEqual(dst, 0xC000_0000_0000_0000)
        self.assertEqual(c, 0)

    def test_extreme_boundary_max_times_max_plus_max_u64(self) -> None:
        expected_prod = INT32_MAX * INT32_MAX
        dst, c = ref_mad_i64_i32(INT32_MAX, INT32_MAX, UINT64_MAX)
        expected_dst = (expected_prod - 1) & MASK64
        self.assertEqual(dst, expected_dst)
        self.assertEqual(c, 1)

    def test_extreme_boundary_min_times_max_plus_max_i64(self) -> None:
        prod = INT32_MIN * INT32_MAX
        expected_dst = (prod + INT64_MAX) & MASK64
        dst, c = ref_mad_i64_i32(INT32_MIN, INT32_MAX, INT64_MAX)
        self.assertEqual(dst, expected_dst)
        self.assertEqual(c, 1)


# ==============================================================================
# Unit Tests: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Negative controls, parser syntax rejection, and mutant kill verification."""

    def test_parser_unsupported_opcode_rejected(self) -> None:
        invalid_opcodes = [
            "v_mad_u64_u32 v[0:1], s[0:1], v2, v3, v[4:5]",
            "v_add_i32 v[0:1], s[0:1], v2, v3, v[4:5]",
            "s_mad_i64_i32 v[0:1], s[0:1], v2, v3, v[4:5]",
        ]
        for line in invalid_opcodes:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    parse_v_mad_i64_i32(line)

    def test_parser_invalid_operand_counts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_v_mad_i64_i32("v_mad_i64_i32 v[0:1], s[0:1], v2, v3")
        with self.assertRaises(ValueError):
            parse_v_mad_i64_i32("v_mad_i64_i32 v[0:1], s[0:1], v2, v3, v[4:5], v6")

    def test_parser_invalid_register_spans_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_v_mad_i64_i32("v_mad_i64_i32 v[0:2], s[0:1], v2, v3, v[4:5]")
        with self.assertRaises(ValueError):
            parse_v_mad_i64_i32("v_mad_i64_i32 v[5:4], s[0:1], v2, v3, v[4:5]")
        with self.assertRaises(ValueError):
            parse_v_mad_i64_i32("v_mad_i64_i32 invalid, s[0:1], v2, v3, v[4:5]")

    def test_ast_type_and_semantic_rejections(self) -> None:
        with self.assertRaises(TypeError):
            VMadI64I32Instruction(
                dst=Operand.sgpr(0, is_64bit=True),
                sdst=Operand.sgpr(2, is_64bit=True),
                src0=Operand.vgpr(4),
                src1=Operand.vgpr(5),
                src2=Operand.vgpr(6, is_64bit=True),
            )

        with self.assertRaises(ValueError):
            VMadI64I32Instruction(
                dst=Operand.vgpr(0, is_64bit=False),
                sdst=Operand.sgpr(2, is_64bit=True),
                src0=Operand.vgpr(4),
                src1=Operand.vgpr(5),
                src2=Operand.vgpr(6, is_64bit=True),
            )

        with self.assertRaises(TypeError):
            VMadI64I32Instruction(
                dst=Operand.vgpr(0, is_64bit=True),
                sdst=Operand.vgpr(2, is_64bit=True),
                src0=Operand.vgpr(4),
                src1=Operand.vgpr(5),
                src2=Operand.vgpr(6, is_64bit=True),
            )

    def test_mutation_unsigned_multiplication_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int) -> tuple[int, int]:
            u0 = s0 & MASK32
            u1 = s1 & MASK32
            prod = u0 * u1
            raw_sum = (prod & MASK64) + (s2 & MASK64)
            return (raw_sum & MASK64), (1 if raw_sum >= MODULO64 else 0)

        s0, s1, s2 = -10, 20, 0
        gold_dst, gold_c = ref_mad_i64_i32(s0, s1, s2)
        mut_dst, mut_c = mutant_mad(s0, s1, s2)
        self.assertNotEqual(gold_dst, mut_dst)

    def test_mutation_truncated_product_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int) -> tuple[int, int]:
            prod = (ref_mul_i32(s0, s1)) & MASK32
            raw_sum = prod + (s2 & MASK64)
            return (raw_sum & MASK64), (1 if raw_sum >= MODULO64 else 0)

        s0, s1, s2 = INT32_MIN, INT32_MIN, 0
        gold_dst, gold_c = ref_mad_i64_i32(s0, s1, s2)
        mut_dst, mut_c = mutant_mad(s0, s1, s2)
        self.assertNotEqual(gold_dst, mut_dst)

    def test_mutation_truncated_accumulator_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int) -> tuple[int, int]:
            prod = ref_mul_i32(s0, s1) & MASK64
            raw_sum = prod + (s2 & MASK32)
            return (raw_sum & MASK64), (1 if raw_sum >= MODULO64 else 0)

        s0, s1, s2 = 1, 1, 0x1234_5678_0000_0000
        gold_dst, gold_c = ref_mad_i64_i32(s0, s1, s2)
        mut_dst, mut_c = mutant_mad(s0, s1, s2)
        self.assertNotEqual(gold_dst, mut_dst)

    def test_mutation_inverted_carry_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int) -> tuple[int, int]:
            prod = ref_mul_i32(s0, s1) & MASK64
            raw_sum = prod + (s2 & MASK64)
            carry = 0 if raw_sum >= MODULO64 else 1
            return (raw_sum & MASK64), carry

        s0, s1, s2 = -1, 1, 1
        gold_dst, gold_c = ref_mad_i64_i32(s0, s1, s2)
        mut_dst, mut_c = mutant_mad(s0, s1, s2)
        self.assertEqual(gold_c, 1)
        self.assertEqual(mut_c, 0)
        self.assertNotEqual(gold_c, mut_c)

    def test_mutation_omitted_accumulator_killed(self) -> None:
        def mutant_mad(s0: int, s1: int, s2: int) -> tuple[int, int]:
            prod = ref_mul_i32(s0, s1) & MASK64
            return prod, 0

        s0, s1, s2 = 5, 5, 100
        gold_dst, _ = ref_mad_i64_i32(s0, s1, s2)
        mut_dst, _ = mutant_mad(s0, s1, s2)
        self.assertEqual(gold_dst, 125)
        self.assertEqual(mut_dst, 25)
        self.assertNotEqual(gold_dst, mut_dst)

    def test_mutation_simd_exec_mask_bypass_killed(self) -> None:
        class BuggySIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VMadI64I32Instruction, state: SIMDState) -> tuple[list[int | None], int]:
                results: list[int | None] = []
                for lane in range(state.num_lanes):
                    s0 = state.read_lane_32(instr.src0, lane)
                    s1 = state.read_lane_32(instr.src1, lane)
                    s2 = state.read_lane_64(instr.src2, lane)
                    dst_val, _ = ref_mad_i64_i32(s0, s1, s2)
                    state.write_lane_64(instr.dst, lane, dst_val)
                    results.append(dst_val)
                return results, 0

        state = SIMDState(wave_mode=WaveMode.WAVE32, exec_mask=0x0000_0000)
        sentinel = 0xAAAA_BBBB_CCCC_DDDD
        for lane in range(32):
            state.write_lane_64(Operand.vgpr(0, is_64bit=True), lane, sentinel)

        instr = VMadI64I32Instruction(
            dst=Operand.vgpr(0, is_64bit=True),
            sdst=Operand.sgpr(0, is_64bit=False),
            src0=Operand.literal(1),
            src1=Operand.literal(1),
            src2=Operand.literal(1, is_64bit=True),
        )

        buggy = BuggySIMDInterpreter()
        buggy.step(instr, state)

        corrupted = [
            lane for lane in range(32)
            if combine_u32(state.vgprs[lane][0], state.vgprs[lane][1]) != sentinel
        ]
        self.assertEqual(len(corrupted), 32)


if __name__ == "__main__":
    unittest.main()
