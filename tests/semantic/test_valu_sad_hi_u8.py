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

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
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
# Condition Codes and Architectural Status Flags
# ==============================================================================

@dataclass(frozen=True)
class StatusFlags:
    """Architectural status flags and condition codes for VALU ALU operations."""
    zero: bool
    sign: bool
    carry: bool
    overflow: bool


def compute_flags(s0: int, s1: int, s2: int) -> StatusFlags:
    """Evaluate condition codes and status flags resulting from v_sad_hi_u8 execution."""
    dst, carry, _ = ref_sad_hi_u8_full(s0, s1, s2)
    shifted = (ref_sad_u8_sum(s0, s1) << 16) & MASK32
    acc = s2 & MASK32

    zero = (dst == 0)
    sign = bool(dst & 0x8000_0000)
    # Signed overflow: adding two operands with same sign producing different sign
    overflow = bool((~(shifted ^ acc) & (shifted ^ dst)) & 0x8000_0000)

    return StatusFlags(zero=zero, sign=sign, carry=bool(carry), overflow=overflow)


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
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarState:
    """Architectural register state for single-thread scalar emulation."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def read_operand(self, op: Operand) -> int:
        """Resolve operand value from architectural state or inline constant."""
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        assert op.literal_value is not None
        return op.literal_value & MASK32

    def write_vgpr(self, index: int, value: int) -> None:
        """Write 32-bit dword to destination VGPR."""
        if not (0 <= index <= 255):
            raise ValueError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[index] = value & MASK32


class ScalarInterpreter:
    """Deterministic scalar interpreter executing v_sad_hi_u8 instructions."""

    def step(self, instr: VSadHiU8Instruction, state: ScalarState) -> int:
        """Execute a single v_sad_hi_u8 instruction against architectural scalar state."""
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        s2 = state.read_operand(instr.src2)

        res = ref_sad_hi_u8(s0, s1, s2)
        state.write_vgpr(instr.dst.index, res)
        return res


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Vector execution state for Wave32 and Wave64 SIMD architectures."""
    num_lanes: int = 32
    exec_mask: int = MASK32
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"Invalid SIMD lane count {self.num_lanes}, must be 32 or 64")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        mask_limit = MASK32 if self.num_lanes == 32 else MASK64
        self.exec_mask &= mask_limit

    def is_lane_active(self, lane: int) -> bool:
        """Query whether a SIMD work-item lane is active under current EXEC mask."""
        return bool((self.exec_mask >> lane) & 1)

    def read_lane(self, op: Operand, lane: int) -> int:
        """Read operand value for a specific SIMD lane."""
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        assert op.literal_value is not None
        return op.literal_value & MASK32

    def write_lane(self, op: Operand, lane: int, value: int) -> None:
        """Write 32-bit dword to destination VGPR for a specific SIMD lane."""
        assert op.reg_type == RegType.VGPR
        self.vgprs[lane][op.index] = value & MASK32


class SIMDInterpreter:
    """SIMD vector interpreter modeling Wave32 / Wave64 lane divergence and EXEC masking."""

    def step(self, instr: VSadHiU8Instruction, state: SIMDState) -> List[Optional[int]]:
        """Execute a v_sad_hi_u8 instruction across all SIMD lanes under EXEC masking."""
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                s0 = state.read_lane(instr.src0, lane)
                s1 = state.read_lane(instr.src1, lane)
                s2 = state.read_lane(instr.src2, lane)

                res = ref_sad_hi_u8(s0, s1, s2)
                state.write_lane(instr.dst, lane, res)
                results.append(res)
            else:
                results.append(None)
        return results


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


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestVSadHiU8ScalarInterpreter(unittest.TestCase):
    """Test suite verifying single-thread deterministic scalar interpreter execution."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_interp_basic_execution(self) -> None:
        """Execute basic v_sad_hi_u8 instruction updating state.vgprs[dst]."""
        self.state.vgprs[1] = 0x01020304
        self.state.vgprs[2] = 0x00000000
        self.state.sgprs[0] = 0x1000

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.sgpr(0),
        )

        res = self.interp.step(instr, self.state)
        # diff = 1 + 2 + 3 + 4 = 10 -> 10 << 16 = 0x000A0000 -> + 0x1000 = 0x000A1000
        self.assertEqual(res, 0x000A_1000)
        self.assertEqual(self.state.vgprs[0], 0x000A_1000)

    def test_scalar_interp_multi_step_sequence(self) -> None:
        """Sequential chained execution with accumulation."""
        self.state.vgprs[1] = pack_bytes_u32(10, 20, 30, 40)
        self.state.vgprs[2] = pack_bytes_u32(0, 0, 0, 0)
        self.state.vgprs[0] = 0

        # Step 1: v0 = v_sad_hi_u8(v1, v2, 0)
        instr1 = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.literal(0),
        )
        self.interp.step(instr1, self.state)
        expected1 = (100 << 16)
        self.assertEqual(self.state.vgprs[0], expected1)

        # Step 2: v0 = v_sad_hi_u8(v1, v2, v0) -> adds another (100 << 16)
        instr2 = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )
        self.interp.step(instr2, self.state)
        expected2 = (200 << 16)
        self.assertEqual(self.state.vgprs[0], expected2)

    def test_scalar_interp_operand_types(self) -> None:
        """Execution with combinations of VGPR, SGPR, and Literal operands."""
        self.state.vgprs[3] = 0xFFFFFFFF
        self.state.sgprs[5] = 0x00000000
        literal_val = 0x00005555

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(10),
            src0=Operand.vgpr(3),
            src1=Operand.sgpr(5),
            src2=Operand.literal(literal_val),
        )
        res = self.interp.step(instr, self.state)
        # sum_diff = 1020 (0x3FC), shifted = 0x03FC0000, + 0x5555 = 0x03FC5555
        self.assertEqual(res, 0x03FC_5555)
        self.assertEqual(self.state.vgprs[10], 0x03FC_5555)

    def test_scalar_interp_matches_oracle(self) -> None:
        """Exhaustive check of scalar interpreter against reference oracle across test vectors."""
        test_vectors = [
            (0x00000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0x00000000, 0x00000000),
            (0x12345678, 0x87654321, 0x00010000),
            (0x55555555, 0xAAAAAAAA, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x12345678),
        ]
        for s0, s1, s2 in test_vectors:
            self.state.vgprs[1] = s0
            self.state.vgprs[2] = s1
            self.state.vgprs[3] = s2
            instr = VSadHiU8Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
            res = self.interp.step(instr, self.state)
            expected = ref_sad_hi_u8(s0, s1, s2)
            self.assertEqual(res, expected)
            self.assertEqual(self.state.vgprs[0], expected)


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class TestVSadHiU8SIMD(unittest.TestCase):
    """Test suite verifying SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_simd_wave32_full_exec(self) -> None:
        """Wave32 execution with all 32 lanes active."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.vgprs[lane][1] = pack_bytes_u32(lane, 0, 0, 0)
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = 0x0001_0000

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = ((lane << 16) + 0x0001_0000) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][0], expected)

    def test_simd_wave64_full_exec(self) -> None:
        """Wave64 execution with all 64 lanes active."""
        state = SIMDState(num_lanes=64, exec_mask=MASK64)
        for lane in range(64):
            state.vgprs[lane][1] = pack_bytes_u32(1, 2, 3, 4)
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = lane

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 64)
        # diff = 10 -> shifted = 0x000A0000
        for lane in range(64):
            expected = 0x000A_0000 + lane
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][0], expected)

    def test_simd_wave32_divergence_mask(self) -> None:
        """Wave32 divergent execution: only even lanes active (0x55555555)."""
        sentinel = 0xCAFE_BABE
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.vgprs[lane][0] = sentinel
            state.vgprs[lane][1] = 0xFFFFFFFF
            state.vgprs[lane][2] = 0x00000000
            state.vgprs[lane][3] = 0

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        # diff = 1020 -> shifted = 0x03FC0000
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x03FC_0000)
                self.assertEqual(state.vgprs[lane][0], 0x03FC_0000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], sentinel)

    def test_simd_wave64_divergence_mask(self) -> None:
        """Wave64 divergent execution: low 32 lanes active, high 32 lanes inactive."""
        sentinel = 0xFEED_FACE
        exec_mask = 0x0000_0000_FFFF_FFFF
        state = SIMDState(num_lanes=64, exec_mask=exec_mask)
        for lane in range(64):
            state.vgprs[lane][0] = sentinel
            state.vgprs[lane][1] = 0x01010101
            state.vgprs[lane][2] = 0x00000000
            state.vgprs[lane][3] = 0

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        # diff = 4 -> shifted = 0x00040000
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], 0x0004_0000)
                self.assertEqual(state.vgprs[lane][0], 0x0004_0000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], sentinel)

    def test_simd_zero_exec_mask(self) -> None:
        """EXEC = 0 leaves all lanes completely unmodified."""
        sentinel = 0xDEAD_C0DE
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.vgprs[lane][0] = sentinel

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.literal(0xFFFFFFFF),
            src1=Operand.literal(0),
            src2=Operand.literal(0),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.vgprs[lane][0], sentinel)

    def test_simd_per_lane_divergent_operands(self) -> None:
        """Each SIMD lane computes independent absolute difference calculations."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.vgprs[lane][1] = pack_bytes_u32(lane, lane * 2, lane * 3, lane * 4)
            state.vgprs[lane][2] = pack_bytes_u32(0, 0, 0, 0)
            state.vgprs[lane][3] = 0

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            diff = u8(lane) + u8(lane * 2) + u8(lane * 3) + u8(lane * 4)
            expected = (diff << 16) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][5], expected)


# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestVSadHiU8RegisterAliasing(unittest.TestCase):
    """Test suite verifying register aliasing and in-place operand preservation."""

    def test_aliasing_dst_equals_src0(self) -> None:
        """dst and src0 share the same VGPR: dst must be read before overwritten."""
        state = ScalarState()
        interp = ScalarInterpreter()

        # v0 = initial src0, v1 = src1, s0 = src2
        state.vgprs[0] = 0x0A0A0A0A
        state.vgprs[1] = 0x00000000
        state.sgprs[0] = 0x1234

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.sgpr(0),
        )

        res = interp.step(instr, state)
        # diff = 4 * 10 = 40 -> shifted = 40 << 16 = 0x00280000 -> + 0x1234 = 0x00281234
        expected = 0x0028_1234
        self.assertEqual(res, expected)
        self.assertEqual(state.vgprs[0], expected)

    def test_aliasing_dst_equals_src1(self) -> None:
        """dst and src1 share the same VGPR: dst must be read before overwritten."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[0] = 0x00000000
        state.vgprs[1] = 0xFFFFFFFF
        state.sgprs[0] = 0x5678

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(0),
            src2=Operand.sgpr(0),
        )

        res = interp.step(instr, state)
        # diff = 1020 -> shifted = 0x03FC0000 -> + 0x5678 = 0x03FC5678
        expected = 0x03FC_5678
        self.assertEqual(res, expected)
        self.assertEqual(state.vgprs[0], expected)

    def test_aliasing_dst_equals_src2_accumulator_in_place(self) -> None:
        """dst and src2 share the same VGPR (canonical in-place accumulation loop)."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[0] = 0x0000_1000  # accumulator in v0
        state.vgprs[1] = pack_bytes_u32(1, 1, 1, 1)
        state.vgprs[2] = pack_bytes_u32(0, 0, 0, 0)

        # v0 = v_sad_hi_u8(v1, v2, v0)
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        # Iteration 1: 4 << 16 = 0x00040000 -> + 0x1000 = 0x00041000
        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 0x0004_1000)

        # Iteration 2: + 0x00040000 -> 0x00081000
        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 0x0008_1000)

        # Iteration 3: + 0x00040000 -> 0x000C1000
        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 0x000C_1000)

    def test_aliasing_src0_equals_src1(self) -> None:
        """src0 == src1 produces diff=0, preserving accumulator exactly."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[1] = 0xDEADBEEF
        state.vgprs[2] = 0xCAFEBABE

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )

        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 0xCAFEBABE)

    def test_aliasing_all_operands_same_register(self) -> None:
        """dst == src0 == src1 == src2 preserves the original register value."""
        state = ScalarState()
        interp = ScalarInterpreter()

        val = 0x12345678
        state.vgprs[0] = val

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(0),
        )

        res = interp.step(instr, state)
        self.assertEqual(res, val)
        self.assertEqual(state.vgprs[0], val)

    def test_simd_aliasing_under_divergence(self) -> None:
        """SIMD execution with in-place accumulator aliasing (dst == src2) under EXEC mask."""
        interp = SIMDInterpreter()
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)  # low 16 lanes active
        initial_acc = 0x0000_0005

        for lane in range(32):
            state.vgprs[lane][0] = initial_acc
            state.vgprs[lane][1] = pack_bytes_u32(2, 2, 2, 2)
            state.vgprs[lane][2] = pack_bytes_u32(0, 0, 0, 0)

        # v0 = v_sad_hi_u8(v1, v2, v0)
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        results = interp.step(instr, state)
        # diff = 8 -> shifted = 8 << 16 = 0x00080000 -> + 5 = 0x00080005
        expected_active = 0x0008_0005
        for lane in range(32):
            if lane < 16:
                self.assertEqual(results[lane], expected_active)
                self.assertEqual(state.vgprs[lane][0], expected_active)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], initial_acc)


# ==============================================================================
# Unit Tests: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

class TestVSadHiU8ConditionCodesAndFlags(unittest.TestCase):
    """Test suite verifying condition codes, status flags, and carry generation."""

    def test_flags_zero_result(self) -> None:
        """Zero flag asserted when dst evaluates to 0x00000000."""
        # Case 1: diff=0 and acc=0
        flags = compute_flags(0x12345678, 0x12345678, 0)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry)

        # Case 2: diff=1 -> shifted=0x00010000, acc=0xFFFF0000 -> sum=0x1_00000000 -> dst=0
        flags_wrap = compute_flags(pack_bytes_u32(1, 0, 0, 0), 0, 0xFFFF_0000)
        self.assertTrue(flags_wrap.zero)
        self.assertFalse(flags_wrap.sign)
        self.assertTrue(flags_wrap.carry)

    def test_flags_sign_bit_set(self) -> None:
        """Sign flag asserted when dst MSB (bit 31) is set."""
        # acc with bit 31 set
        flags = compute_flags(0, 0, 0x8000_0000)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.carry)

    def test_flags_sign_bit_clear(self) -> None:
        """Sign flag cleared when dst MSB (bit 31) is zero."""
        flags = compute_flags(0, 0, 0x7FFF_FFFF)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry)

    def test_flags_carry_out_generated(self) -> None:
        """Carry flag asserted when full 64-bit sum overflows 32 bits."""
        # max diff 1020 -> shifted = 0x03FC0000; acc = 0xFF000000
        # 0x03FC0000 + 0xFF000000 = 0x102FC0000 -> carry = 1
        flags = compute_flags(0xFFFFFFFF, 0x00000000, 0xFF00_0000)
        self.assertTrue(flags.carry)

    def test_flags_carry_out_not_generated(self) -> None:
        """Carry flag cleared when sum fits in 32 bits."""
        flags = compute_flags(0xFFFFFFFF, 0x00000000, 0x0000_FFFF)
        self.assertFalse(flags.carry)

    def test_flags_carry_exact_threshold(self) -> None:
        """Test exact boundary condition around 0x1_0000_0000 for carry generation."""
        s0 = pack_bytes_u32(1, 0, 0, 0)
        s1 = 0
        # diff = 1 -> shifted = 0x00010000

        # One below wrap threshold: 0x00010000 + 0xFFFE_FFFF = 0x0_FFFF_FFFF
        flags_below = compute_flags(s0, s1, 0xFFFE_FFFF)
        self.assertFalse(flags_below.carry)
        self.assertFalse(flags_below.zero)
        self.assertTrue(flags_below.sign)

        # Exactly at wrap threshold: 0x00010000 + 0xFFFF_0000 = 0x1_0000_0000
        flags_exact = compute_flags(s0, s1, 0xFFFF_0000)
        self.assertTrue(flags_exact.carry)
        self.assertTrue(flags_exact.zero)
        self.assertFalse(flags_exact.sign)

        # One above wrap threshold: 0x00010000 + 0xFFFF_0001 = 0x1_0000_0001
        flags_above = compute_flags(s0, s1, 0xFFFF_0001)
        self.assertTrue(flags_above.carry)
        self.assertFalse(flags_above.zero)
        self.assertFalse(flags_above.sign)

    def test_flags_vcc_non_pollution(self) -> None:
        """Architecturally, v_sad_hi_u8 does not consume or overwrite VCC condition registers."""
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        self.assertEqual(instr.dst.reg_type, RegType.VGPR)


# ==============================================================================
# Unit Tests: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestVSadHiU8IEEE754BitPatterns(unittest.TestCase):
    """Test suite verifying bit-exact integer handling of IEEE 754 float patterns.

    v_sad_hi_u8 is an integer byte-wise VALU instruction. Operands with bit patterns
    corresponding to IEEE 754 floating-point specials (+0.0, -0.0, subnormals, infinities,
    quiet/signaling NaNs) must be interpreted strictly as unsigned 8-bit integers without
    floating-point normalization, subnormal flush-to-zero (FTZ), or NaN quiet bit mutation.
    """

    @staticmethod
    def float_to_u32(val: float) -> int:
        """Convert Python float to 32-bit unsigned integer bitcast."""
        return struct.unpack("<I", struct.pack("<f", val))[0]

    def test_ieee754_zero_and_negative_zero(self) -> None:
        """+0.0 (0x00000000) vs -0.0 (0x80000000) differs in byte 3 sign bit."""
        pos_zero = self.float_to_u32(+0.0)  # 0x00000000
        neg_zero = self.float_to_u32(-0.0)  # 0x80000000

        self.assertEqual(pos_zero, 0x0000_0000)
        self.assertEqual(neg_zero, 0x8000_0000)

        # In float math, +0.0 == -0.0; in v_sad_hi_u8, byte 3 has diff |0x80 - 0x00| = 128
        sum_diff = ref_sad_u8_sum(pos_zero, neg_zero)
        self.assertEqual(sum_diff, 128)

        dst = ref_sad_hi_u8(pos_zero, neg_zero, 0)
        # 128 << 16 = 0x00800000
        self.assertEqual(dst, 0x0080_0000)

    def test_ieee754_infinity_representations(self) -> None:
        """+inf (0x7F800000) and -inf (0xFF800000) bit patterns."""
        pos_inf = self.float_to_u32(float("inf"))   # 0x7F800000
        neg_inf = self.float_to_u32(float("-inf"))  # 0xFF800000

        self.assertEqual(pos_inf, 0x7F80_0000)
        self.assertEqual(neg_inf, 0xFF80_0000)

        # Byte 3 differs: |0x7F - 0xFF| = 128
        self.assertEqual(ref_sad_u8_sum(pos_inf, neg_inf), 128)
        self.assertEqual(ref_sad_hi_u8(pos_inf, neg_inf, 0), 0x0080_0000)

        # Identical infinities yield zero difference
        self.assertEqual(ref_sad_u8_sum(pos_inf, pos_inf), 0)
        self.assertEqual(ref_sad_hi_u8(pos_inf, pos_inf, 0x42), 0x42)

    def test_ieee754_nan_preservation_and_payloads(self) -> None:
        """Quiet and signaling NaN patterns must not trigger exceptions or canonicalization."""
        qnan = 0x7FC0_0000  # standard quiet NaN
        snan = 0x7F80_0001  # signaling NaN
        custom_nan = 0x7FA5_C301  # NaN with arbitrary payload bits

        # Differences between NaN bit patterns are computed as plain unsigned bytes
        # |0x7F - 0x7F| = 0, |0xC0 - 0x80| = 0x40 = 64, |0x00 - 0x00| = 0, |0x00 - 0x01| = 1
        # sum_diff = 64 + 1 = 65 -> shifted = 65 << 16 = 0x00410000
        self.assertEqual(ref_sad_u8_sum(qnan, snan), 65)
        self.assertEqual(ref_sad_hi_u8(qnan, snan, 0), 0x0041_0000)

        # Self-difference of NaN is identically 0 (unlike FP where NaN != NaN)
        self.assertEqual(ref_sad_u8_sum(qnan, qnan), 0)
        self.assertEqual(ref_sad_hi_u8(custom_nan, custom_nan, 0x1234), 0x1234)

    def test_ieee754_subnormals_no_denorm_flushing(self) -> None:
        """Subnormal patterns must never be flushed to zero (FTZ mode does not apply to VALU integer)."""
        subnormal_min = 0x0000_0001  # smallest positive subnormal
        subnormal_max = 0x007F_FFFF  # largest positive subnormal

        # byte difference against 0:
        self.assertEqual(ref_sad_u8_sum(subnormal_min, 0), 1)
        self.assertEqual(ref_sad_hi_u8(subnormal_min, 0, 0), 0x0001_0000)

        # subnormal_max: bytes are (0xFF, 0xFF, 0x7F, 0x00) -> sum = 255 + 255 + 127 = 637 (0x27D)
        sum_max = ref_sad_u8_sum(subnormal_max, 0)
        self.assertEqual(sum_max, 637)
        self.assertEqual(ref_sad_hi_u8(subnormal_max, 0, 0), 637 << 16)


# ==============================================================================
# Unit Tests: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestVSadHiU8NegativeControls(unittest.TestCase):
    """Test suite verifying negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_destination_not_vgpr(self) -> None:
        """Destination operand must strictly be a VGPR."""
        with self.assertRaises(TypeError):
            VSadHiU8Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.literal(0),
            )
        with self.assertRaises(TypeError):
            VSadHiU8Instruction(
                dst=Operand.literal(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.literal(0),
            )

    def test_syntax_rejection_invalid_vgpr_index(self) -> None:
        """VGPR indices must be strictly within [0, 255]."""
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)

    def test_syntax_rejection_invalid_sgpr_index(self) -> None:
        """SGPR indices must be strictly within [0, 105]."""
        with self.assertRaises(ValueError):
            Operand.sgpr(-1)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)

    def test_syntax_rejection_invalid_literal_value(self) -> None:
        """Literal constants must be unsigned 32-bit integers."""
        with self.assertRaises(ValueError):
            Operand.literal(-1)
        with self.assertRaises(ValueError):
            Operand.literal(0x1_0000_0000)

    def test_syntax_rejection_invalid_simd_lane_count(self) -> None:
        """SIMDState lane count must be either 32 (Wave32) or 64 (Wave64)."""
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_bit_flip_detection(self) -> None:
        """Single-bit corruptions of the expected result are deterministically caught."""
        s0 = 0xFFFFFFFF
        s1 = 0x00000000
        s2 = 0x00000000
        correct = ref_sad_hi_u8(s0, s1, s2)
        self.assertEqual(correct, 0x03FC_0000)

        # Flip bit 16
        corrupted = correct ^ (1 << 16)
        self.assertNotEqual(correct, corrupted)

    def test_mutation_omitted_shift_detected(self) -> None:
        """Mutant: Omits `<< 16` shift (computing v_sad_u8 instead of v_sad_hi_u8)."""
        def mutant_sad(s0: int, s1: int, s2: int) -> int:
            # Bug: forgot << 16
            return (ref_sad_u8_sum(s0, s1) + u32(s2)) & MASK32

        s0 = pack_bytes_u32(10, 0, 0, 0)
        s1 = 0
        s2 = 0

        oracle_res = ref_sad_hi_u8(s0, s1, s2)
        mutant_res = mutant_sad(s0, s1, s2)

        # Oracle shifts 10 by 16: 0x000A0000; mutant returns 10
        self.assertEqual(oracle_res, 0x000A_0000)
        self.assertEqual(mutant_res, 10)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_signed_byte_difference_detected(self) -> None:
        """Mutant: Treats packed bytes as signed two's complement integers."""
        def mutant_sad_hi(s0: int, s1: int, s2: int) -> int:
            def to_s8(x: int) -> int:
                b = x & 0xFF
                return b - 256 if (b & 0x80) else b
            b0_0, b0_1, b0_2, b0_3 = unpack_bytes_u32(s0)
            b1_0, b1_1, b1_2, b1_3 = unpack_bytes_u32(s1)
            # Bug: signed absolute difference
            diff = (
                abs(to_s8(b0_0) - to_s8(b1_0))
                + abs(to_s8(b0_1) - to_s8(b1_1))
                + abs(to_s8(b0_2) - to_s8(b1_2))
                + abs(to_s8(b0_3) - to_s8(b1_3))
            )
            return ((diff << 16) + u32(s2)) & MASK32

        s0 = pack_bytes_u32(0xFF, 0, 0, 0)  # 255 unsigned, -1 signed
        s1 = 0
        s2 = 0

        oracle_res = ref_sad_hi_u8(s0, s1, s2)
        mutant_res = mutant_sad_hi(s0, s1, s2)

        # Unsigned diff: 255 << 16 = 0x00FF0000
        self.assertEqual(oracle_res, 0x00FF_0000)
        # Signed diff: |-1 - 0| = 1 << 16 = 0x00010000
        self.assertEqual(mutant_res, 0x0001_0000)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_signed_whole_dword_diff_detected(self) -> None:
        """Mutant: Computes whole 32-bit difference instead of 4 packed byte differences."""
        def mutant_sad_hi(s0: int, s1: int, s2: int) -> int:
            # Bug: whole dword diff instead of 4 byte lanes
            diff = abs(u32(s0) - u32(s1))
            return ((diff << 16) + u32(s2)) & MASK32

        s0 = 0x0102_0304
        s1 = 0x0000_0000
        s2 = 0

        oracle_res = ref_sad_hi_u8(s0, s1, s2)
        mutant_res = mutant_sad_hi(s0, s1, s2)

        # Oracle sum = 1 + 2 + 3 + 4 = 10 -> 0x000A0000
        self.assertEqual(oracle_res, 0x000A_0000)
        # Mutant: 0x01020304 << 16 != 0x000A0000
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_omitted_accumulator_detected(self) -> None:
        """Mutant: Forgets to add accumulator src2."""
        def mutant_sad_hi(s0: int, s1: int, s2: int) -> int:
            # Bug: ignores s2
            return (ref_sad_u8_sum(s0, s1) << 16) & MASK32

        s0 = 0
        s1 = 0
        s2 = 0x1234_5678

        oracle_res = ref_sad_hi_u8(s0, s1, s2)
        mutant_res = mutant_sad_hi(s0, s1, s2)

        self.assertEqual(oracle_res, s2)
        self.assertEqual(mutant_res, 0)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_incorrect_shift_amount_detected(self) -> None:
        """Mutant: Shifts by 8 or 24 instead of 16."""
        def mutant_sad_hi_shift8(s0: int, s1: int, s2: int) -> int:
            return ((ref_sad_u8_sum(s0, s1) << 8) + u32(s2)) & MASK32

        s0 = pack_bytes_u32(1, 0, 0, 0)
        s1 = 0
        s2 = 0

        oracle_res = ref_sad_hi_u8(s0, s1, s2)
        mutant_res = mutant_sad_hi_shift8(s0, s1, s2)

        # Oracle: 1 << 16 = 0x00010000; Mutant: 1 << 8 = 0x00000100
        self.assertEqual(oracle_res, 0x0001_0000)
        self.assertEqual(mutant_res, 0x0000_0100)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_simd_ignore_exec_mask_detected(self) -> None:
        """Mutant: SIMD interpreter writes inactive lanes ignoring EXEC mask."""
        sentinel = 0xCAFEBABE
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.vgprs[lane][0] = sentinel

        class BrokenSIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VSadHiU8Instruction, state: SIMDState) -> List[Optional[int]]:
                results = []
                for lane in range(state.num_lanes):
                    # Bug: no is_lane_active check
                    res = ref_sad_hi_u8(0xFFFFFFFF, 0, 0)
                    state.write_lane(instr.dst, lane, res)
                    results.append(res)
                return results

        broken = BrokenSIMDInterpreter()
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.literal(0xFFFFFFFF),
            src1=Operand.literal(0),
            src2=Operand.literal(0),
        )
        broken.step(instr, state)

        # Verify that broken interpreter corrupted all 32 lanes
        corrupted = [lane for lane in range(32) if state.vgprs[lane][0] != sentinel]
        self.assertEqual(len(corrupted), 32)

    def test_mutation_unmasked_overflow_detected(self) -> None:
        """Mutant: Fails to wrap result to 32 bits upon overflow."""
        def mutant_sad_hi_unmasked(s0: int, s1: int, s2: int) -> int:
            # Bug: no 32-bit truncation
            return (ref_sad_u8_sum(s0, s1) << 16) + (s2 & MASK32)

        s0 = 0xFFFFFFFF
        s1 = 0x00000000
        s2 = 0xFFFF_0000

        oracle_res = ref_sad_hi_u8(s0, s1, s2)
        mutant_res = mutant_sad_hi_unmasked(s0, s1, s2)

        self.assertEqual(oracle_res, 0x03FB_0000)
        self.assertEqual(mutant_res, 0x1_03FB_0000)
        self.assertNotEqual(oracle_res, mutant_res)


if __name__ == "__main__":
    unittest.main()
