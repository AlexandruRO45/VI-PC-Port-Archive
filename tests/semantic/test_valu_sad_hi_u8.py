from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_sad_hi_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_sad_hi_u8 dst, src0, src1, src2

Format:
  VOP3 (3-operand vector ALU instruction, 32-bit destination, three 32-bit inputs).

Operation:
  Computes the 4-channel sum of unsigned 8-bit absolute differences between
  packed byte lanes of src0 and src1, shifts the resulting sum left by 16 bits,
  and accumulates into the 32-bit unsigned accumulator src2:
    diff0 = |src0[7:0]   - src1[7:0]|
    diff1 = |src0[15:8]  - src1[15:8]|
    diff2 = |src0[23:16] - src1[23:16]|
    diff3 = |src0[31:24] - src1[31:24]|
    sum_diff = diff0 + diff1 + diff2 + diff3
    sum_shifted = sum_diff << 16
    dst = (sum_shifted + src2) & 0xFFFFFFFF

Mathematical Properties and Bounds:
  - Operands src0, src1 are treated as four packed 8-bit unsigned integers in [0, 255].
  - Each byte difference diff_i is in [0, 255].
  - Total byte absolute difference sum_diff is in [0, 4 * 255] = [0, 1020] (0x000 to 0x3FC).
  - sum_diff strictly fits within 10 bits (< 2^10 = 1024).
  - The shifted sum (sum_diff << 16) is in [0, 0x03FC0000], strictly occupying bits [25:16].
    Bits [15:0] of (sum_diff << 16) are strictly zero.
  - Adding unsigned 32-bit src2 produces a full sum in [0, 0x03FC0000 + 0xFFFFFFFF] = [0, 0x103FBFFFF].
  - Destination wraps modulo 2^32: dst = (sum_shifted + src2) & 0xFFFFFFFF.
  - Carry-out into the 33rd bit is: (sum_shifted + src2) >> 32 in {0, 1}.
  - Symmetry / Commutativity:
    v_sad_hi_u8(src0, src1, src2) == v_sad_hi_u8(src1, src0, src2).
  - Complementary packing with v_sad_u8:
    v_sad_u8 computes (sum_diff0 + src2) & 0xFFFFFFFF.
    If v_sad_u8 is executed with src2 = 0, it yields sum_diff0 in bits [15:0] (since sum_diff0 <= 1020).
    Feeding this result as src2 into v_sad_hi_u8 with inputs src0_b, src1_b yields:
      (sum_diff1 << 16) | sum_diff0
    packing two independent 4-element 8-bit SAD evaluations into one 32-bit register.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
BYTE_SHIFTS: Tuple[int, int, int, int] = (0, 8, 16, 24)


def u8(val: int) -> int:
    """Normalize integer to unsigned 8-bit byte."""
    return val & MASK8


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def unpack_bytes_u32(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned word into four 8-bit unsigned byte lanes (b0, b1, b2, b3)."""
    val &= MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_bytes_u32(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit unsigned dword."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_abs_diff_u8(b0: int, b1: int) -> int:
    """Pure-Python reference oracle for 8-bit unsigned absolute difference: |b0 - b1|."""
    u0 = b0 & MASK8
    u1 = b1 & MASK8
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u8_sum(s0: int, s1: int) -> int:
    """Compute the 4-channel sum of 8-bit unsigned absolute differences in [0, 1020]."""
    b0_0, b0_1, b0_2, b0_3 = unpack_bytes_u32(s0)
    b1_0, b1_1, b1_2, b1_3 = unpack_bytes_u32(s1)
    return (
        ref_abs_diff_u8(b0_0, b1_0)
        + ref_abs_diff_u8(b0_1, b1_1)
        + ref_abs_diff_u8(b0_2, b1_2)
        + ref_abs_diff_u8(b0_3, b1_3)
    )


def ref_sad_hi_u8(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_sad_hi_u8:

    dst = ((sum_abs_diff(s0, s1) << 16) + s2) & 0xFFFFFFFF
    """
    sum_diff = ref_sad_u8_sum(s0, s1)
    acc = s2 & MASK32
    return ((sum_diff << 16) + acc) & MASK32


def ref_sad_hi_u8_full(s0: int, s1: int, s2: int) -> Tuple[int, int, int]:
    """Return (dst_u32, carry_out, full_sum64) for v_sad_hi_u8."""
    sum_diff = ref_sad_u8_sum(s0, s1)
    acc = s2 & MASK32
    full = (sum_diff << 16) + acc
    dst = full & MASK32
    carry = (full >> 32) & 1
    return dst, carry, full


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    """Register and operand architectural types."""
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    """Architectural register operand or inline constant expression."""
    reg_type: RegType
    index: int = 0
    literal_value: Optional[int] = None

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            if not (0 <= self.literal_value <= MASK32):
                raise ValueError(f"Literal value {self.literal_value:#x} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def vgpr(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index)

    @classmethod
    def sgpr(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index)

    @classmethod
    def literal(cls, value: int) -> Operand:
        if not (0 <= value <= MASK32):
            raise ValueError(f"Literal value {value} out of range [0, 0xFFFFFFFF]")
        return cls(RegType.LITERAL, literal_value=value)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        if self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        assert self.literal_value is not None
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VSadHiU8Instruction:
    """Symbolic AST node for AMD RDNA2 v_sad_hi_u8 VALU instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return (
            f"v_sad_hi_u8 {self.dst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        )


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestVSadHiU8Oracle(unittest.TestCase):
    """Test suite verifying mathematical bounds, properties, and edge cases of the oracle."""

    def test_oracle_zero_inputs(self) -> None:
        """All-zero inputs yield zero result."""
        self.assertEqual(ref_sad_hi_u8(0, 0, 0), 0)
        dst, carry, full = ref_sad_hi_u8_full(0, 0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(carry, 0)
        self.assertEqual(full, 0)

    def test_oracle_identical_inputs(self) -> None:
        """When s0 == s1, absolute differences are all zero and result is s2."""
        patterns = [0x00000000, 0x12345678, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF]
        for pat in patterns:
            acc = 0xDEADBEEF
            self.assertEqual(ref_sad_hi_u8(pat, pat, acc), acc)
            self.assertEqual(ref_sad_u8_sum(pat, pat), 0)

    def test_oracle_maximum_difference(self) -> None:
        """Max difference across all 4 byte lanes (0x00 vs 0xFF): sum_diff = 1020 (0x3FC)."""
        s0 = 0x00000000
        s1 = 0xFFFFFFFF
        sum_diff = ref_sad_u8_sum(s0, s1)
        self.assertEqual(sum_diff, 1020)
        self.assertEqual(sum_diff, 0x3FC)

        # Shifted left 16: 0x03FC0000
        dst = ref_sad_hi_u8(s0, s1, 0)
        self.assertEqual(dst, 0x03FC_0000)

    def test_oracle_single_byte_lanes(self) -> None:
        """Verify each byte lane contributes independently to bits [25:16]."""
        # Lane 0 (bits 7:0)
        self.assertEqual(ref_sad_hi_u8(0x000000FF, 0, 0), 0x00FF_0000)
        # Lane 1 (bits 15:8)
        self.assertEqual(ref_sad_hi_u8(0x0000FF00, 0, 0), 0x00FF_0000)
        # Lane 2 (bits 23:16)
        self.assertEqual(ref_sad_hi_u8(0x00FF0000, 0, 0), 0x00FF_0000)
        # Lane 3 (bits 31:24)
        self.assertEqual(ref_sad_hi_u8(0xFF000000, 0, 0), 0x00FF_0000)

        # Single byte diff of 1 in each lane
        self.assertEqual(ref_sad_hi_u8(0x00000001, 0, 0), 0x0001_0000)
        self.assertEqual(ref_sad_hi_u8(0x00000100, 0, 0), 0x0001_0000)
        self.assertEqual(ref_sad_hi_u8(0x00010000, 0, 0), 0x0001_0000)
        self.assertEqual(ref_sad_hi_u8(0x01000000, 0, 0), 0x0001_0000)

    def test_oracle_commutativity(self) -> None:
        """Commutativity: ref_sad_hi_u8(a, b, c) == ref_sad_hi_u8(b, a, c)."""
        test_pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0xFF00FF00, 0x00FF00FF),
            (0x80808080, 0x01010101),
            (0xFFFFFFFF, 0x00000000),
            (0x7F7F7F7F, 0x80808080),
        ]
        acc = 0xCAFEBABE
        for s0, s1 in test_pairs:
            self.assertEqual(ref_sad_hi_u8(s0, s1, acc), ref_sad_hi_u8(s1, s0, acc))
            self.assertEqual(ref_sad_u8_sum(s0, s1), ref_sad_u8_sum(s1, s0))

    def test_oracle_accumulation_no_overflow(self) -> None:
        """Accumulator adds directly to shifted diff without wrapping when sum < 2^32."""
        s0 = 0x01010101
        s1 = 0x00000000
        # sum_diff = 4 -> shifted = 0x00040000
        acc = 0x1234
        self.assertEqual(ref_sad_hi_u8(s0, s1, acc), 0x0004_1234)

    def test_oracle_accumulation_with_wrap(self) -> None:
        """32-bit truncation when sum_shifted + s2 exceeds 0xFFFFFFFF."""
        s0 = 0xFFFFFFFF
        s1 = 0x00000000
        # sum_diff = 1020 (0x3FC) -> shifted = 0x03FC0000
        acc = 0xFFFF_0000
        # 0x03FC0000 + 0xFFFF0000 = 0x103FB0000 -> wrapped = 0x03FB0000
        dst, carry, full = ref_sad_hi_u8_full(s0, s1, acc)
        self.assertEqual(dst, 0x03FB_0000)
        self.assertEqual(carry, 1)
        self.assertEqual(full, 0x1_03FB_0000)
        self.assertEqual(ref_sad_hi_u8(s0, s1, acc), 0x03FB_0000)

    def test_oracle_bounds(self) -> None:
        """sum_diff is strictly in [0, 1020] and shifted sum is in [0, 0x03FC0000]."""
        for b0 in (0, 1, 127, 128, 254, 255):
            for b1 in (0, 1, 127, 128, 254, 255):
                s0 = pack_bytes_u32(b0, b0, b0, b0)
                s1 = pack_bytes_u32(b1, b1, b1, b1)
                diff = ref_sad_u8_sum(s0, s1)
                self.assertGreaterEqual(diff, 0)
                self.assertLessEqual(diff, 1020)
                shifted = diff << 16
                self.assertEqual(shifted & 0xFFFF, 0)
                self.assertEqual(shifted >> 26, 0)

    def test_oracle_complementary_packing_with_v_sad_u8(self) -> None:
        """v_sad_hi_u8 pairs with v_sad_u8 to pack two 16-bit SAD results into one 32-bit word."""
        # First SAD: 4 differences summing to 10
        s0_lo = pack_bytes_u32(1, 2, 3, 4)
        s1_lo = pack_bytes_u32(0, 0, 0, 0)
        sad_lo = ref_sad_u8_sum(s0_lo, s1_lo)  # 10
        self.assertEqual(sad_lo, 10)

        # Second SAD: 4 differences summing to 25
        s0_hi = pack_bytes_u32(5, 6, 7, 7)
        s1_hi = pack_bytes_u32(0, 0, 0, 0)
        sad_hi = ref_sad_u8_sum(s0_hi, s1_hi)  # 25
        self.assertEqual(sad_hi, 25)

        # Pass sad_lo as s2 to v_sad_hi_u8
        packed = ref_sad_hi_u8(s0_hi, s1_hi, sad_lo)
        self.assertEqual(packed & 0xFFFF, sad_lo)
        self.assertEqual((packed >> 16) & 0xFFFF, sad_hi)
        self.assertEqual(packed, (sad_hi << 16) | sad_lo)


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestVSadHiU8SymbolicAST(unittest.TestCase):
    """Test suite verifying symbolic AST construction, validation, and disassembly."""

    def test_ast_creation_valid_operands(self) -> None:
        """Construct AST nodes with valid VGPR, SGPR, and Literal operands."""
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(0x1234),
        )
        self.assertEqual(instr.dst.reg_type, RegType.VGPR)
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.src0.reg_type, RegType.VGPR)
        self.assertEqual(instr.src0.index, 1)
        self.assertEqual(instr.src1.reg_type, RegType.SGPR)
        self.assertEqual(instr.src1.index, 2)
        self.assertEqual(instr.src2.reg_type, RegType.LITERAL)
        self.assertEqual(instr.src2.literal_value, 0x1234)

    def test_ast_disassembly(self) -> None:
        """Disassembly formatting matches AMD RDNA ISA assembly conventions."""
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(10),
            src1=Operand.sgpr(4),
            src2=Operand.literal(0x00FF00FF),
        )
        self.assertEqual(instr.disassemble(), "v_sad_hi_u8 v5, v10, s4, 0x00FF00FF")

    def test_ast_dst_must_be_vgpr(self) -> None:
        """VALU instruction destination must be a VGPR; SGPR/Literal raises TypeError."""
        with self.assertRaises(TypeError):
            VSadHiU8Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VSadHiU8Instruction(
                dst=Operand.literal(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_ast_operand_bounds(self) -> None:
        """Operand indices and literals must be strictly bounded."""
        # Valid bounds
        Operand.vgpr(0)
        Operand.vgpr(255)
        Operand.sgpr(0)
        Operand.sgpr(105)
        Operand.literal(0)
        Operand.literal(MASK32)

        # Out-of-bounds VGPR
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)

        # Out-of-bounds SGPR
        with self.assertRaises(ValueError):
            Operand.sgpr(-1)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)

        # Out-of-bounds Literal
        with self.assertRaises(ValueError):
            Operand.literal(-1)
        with self.assertRaises(ValueError):
            Operand.literal(0x1_0000_0000)

    def test_ast_equality_and_hashing(self) -> None:
        """AST nodes are frozen and hashable."""
        instr1 = VSadHiU8Instruction(Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3), Operand.vgpr(4))
        instr2 = VSadHiU8Instruction(Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3), Operand.vgpr(4))
        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))


if __name__ == "__main__":
    unittest.main()
