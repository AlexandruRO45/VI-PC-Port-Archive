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


if __name__ == "__main__":
    unittest.main()
