from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_sad_u16.

AMD RDNA VALU Instruction Set Architecture:
  v_sad_u16 dst, src0, src1, src2

Format:
  VOP3 (3-operand vector ALU instruction, 32-bit destination, three 32-bit inputs).

Operation:
  Computes the 2-channel sum of unsigned 16-bit absolute differences between
  packed halfword lanes of src0 and src1, and accumulates into the 32-bit
  unsigned accumulator src2:
    diff0 = |src0[15:0]  - src1[15:0]|
    diff1 = |src0[31:16] - src1[31:16]|
    sum_diff = diff0 + diff1
    dst = (sum_diff + src2) & 0xFFFFFFFF

Mathematical Properties and Bounds:
  - Operands src0, src1 are treated as two packed 16-bit unsigned integers in [0, 65535].
  - Each halfword absolute difference diff0, diff1 is in [0, 65535] (0x0000 to 0xFFFF).
  - Total sum of absolute differences sum_diff is in [0, 2 * 65535] = [0, 131070] (0x00000 to 0x1FFFE).
  - sum_diff strictly fits within 17 bits (< 2^17 = 131072).
  - Adding unsigned 32-bit src2 produces a full sum in [0, 131070 + 0xFFFFFFFF] = [0, 0x1_0001_FFFD].
  - Destination wraps modulo 2^32: dst = (sum_diff + src2) & 0xFFFFFFFF.
  - Carry-out into the 33rd bit is: (sum_diff + src2) >> 32 in {0, 1}.
  - Symmetry / Commutativity:
    v_sad_u16(src0, src1, src2) == v_sad_u16(src1, src0, src2).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def unpack_halfwords_u32(val: int) -> Tuple[int, int]:
    """Unpack a 32-bit unsigned word into two 16-bit unsigned halfword lanes (h0, h1)."""
    val &= MASK32
    return (
        val & MASK16,
        (val >> 16) & MASK16,
    )


def pack_halfwords_u32(h0: int, h1: int) -> int:
    """Pack two 16-bit unsigned integers into a 32-bit unsigned dword."""
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def ref_abs_diff_u16(h0: int, h1: int) -> int:
    """Pure-Python reference oracle for 16-bit unsigned absolute difference: |h0 - h1|."""
    u0 = h0 & MASK16
    u1 = h1 & MASK16
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u16_sum(s0: int, s1: int) -> int:
    """Compute the 2-channel sum of 16-bit unsigned absolute differences in [0, 131070]."""
    h0_0, h0_1 = unpack_halfwords_u32(s0)
    h1_0, h1_1 = unpack_halfwords_u32(s1)
    return ref_abs_diff_u16(h0_0, h1_0) + ref_abs_diff_u16(h0_1, h1_1)


def ref_sad_u16(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_sad_u16:

    dst = (sum_abs_diff_u16(s0, s1) + s2) & 0xFFFFFFFF
    """
    sum_diff = ref_sad_u16_sum(s0, s1)
    acc = s2 & MASK32
    return (sum_diff + acc) & MASK32


def ref_sad_u16_full(s0: int, s1: int, s2: int) -> Tuple[int, int, int]:
    """Return (dst_u32, carry_out, full_sum64) for v_sad_u16."""
    sum_diff = ref_sad_u16_sum(s0, s1)
    acc = s2 & MASK32
    full = sum_diff + acc
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
class VSadU16Instruction:
    """Symbolic AST node for AMD RDNA2 v_sad_u16 VALU instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return (
            f"v_sad_u16 {self.dst.disassemble()}, "
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
    """Deterministic scalar interpreter executing v_sad_u16 instructions."""

    def step(self, instr: VSadU16Instruction, state: ScalarState) -> int:
        """Execute a single v_sad_u16 instruction against architectural scalar state."""
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        s2 = state.read_operand(instr.src2)

        res = ref_sad_u16(s0, s1, s2)
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

    def step(self, instr: VSadU16Instruction, state: SIMDState) -> List[Optional[int]]:
        """Execute a v_sad_u16 instruction across all SIMD lanes under EXEC masking."""
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                s0 = state.read_lane(instr.src0, lane)
                s1 = state.read_lane(instr.src1, lane)
                s2 = state.read_lane(instr.src2, lane)

                res = ref_sad_u16(s0, s1, s2)
                state.write_lane(instr.dst, lane, res)
                results.append(res)
            else:
                results.append(None)
        return results


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestVSadU16Oracle(unittest.TestCase):
    """Test suite verifying mathematical bounds, properties, and edge cases of the oracle."""

    def test_oracle_zero_inputs(self) -> None:
        """All-zero inputs yield zero result."""
        self.assertEqual(ref_sad_u16(0, 0, 0), 0)
        dst, carry, full = ref_sad_u16_full(0, 0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(carry, 0)
        self.assertEqual(full, 0)

    def test_oracle_identical_inputs(self) -> None:
        """When s0 == s1, absolute differences are all zero and result is s2."""
        patterns = [0x00000000, 0x12345678, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF]
        for pat in patterns:
            acc = 0xDEADBEEF
            self.assertEqual(ref_sad_u16(pat, pat, acc), acc)
            self.assertEqual(ref_sad_u16_sum(pat, pat), 0)

    def test_oracle_maximum_difference(self) -> None:
        """Max difference across both 16-bit lanes (0x0000 vs 0xFFFF): sum_diff = 131070 (0x1FFFE)."""
        s0 = 0x00000000
        s1 = 0xFFFFFFFF
        sum_diff = ref_sad_u16_sum(s0, s1)
        self.assertEqual(sum_diff, 131070)
        self.assertEqual(sum_diff, 0x1FFFE)

        dst = ref_sad_u16(s0, s1, 0)
        self.assertEqual(dst, 0x1FFFE)

    def test_oracle_single_halfword_lanes(self) -> None:
        """Verify each 16-bit lane contributes independently."""
        # Lane 0 (bits 15:0)
        self.assertEqual(ref_sad_u16(0x0000FFFF, 0, 0), 0xFFFF)
        # Lane 1 (bits 31:16)
        self.assertEqual(ref_sad_u16(0xFFFF0000, 0, 0), 0xFFFF)

        # Difference of 1 in each lane
        self.assertEqual(ref_sad_u16(0x00000001, 0, 0), 1)
        self.assertEqual(ref_sad_u16(0x00010000, 0, 0), 1)

    def test_oracle_commutativity(self) -> None:
        """Commutativity: ref_sad_u16(a, b, c) == ref_sad_u16(b, a, c)."""
        test_pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0xFFFF0000, 0x0000FFFF),
            (0x80008000, 0x00010001),
            (0xFFFFFFFF, 0x00000000),
            (0x7FFF7FFF, 0x80008000),
        ]
        acc = 0xCAFEBABE
        for s0, s1 in test_pairs:
            self.assertEqual(ref_sad_u16(s0, s1, acc), ref_sad_u16(s1, s0, acc))
            self.assertEqual(ref_sad_u16_sum(s0, s1), ref_sad_u16_sum(s1, s0))

    def test_oracle_accumulation_no_overflow(self) -> None:
        """Accumulator adds directly to diff sum without wrapping when sum < 2^32."""
        s0 = pack_halfwords_u32(100, 200)
        s1 = pack_halfwords_u32(0, 0)
        # sum_diff = 300
        acc = 0x1000
        self.assertEqual(ref_sad_u16(s0, s1, acc), 0x1000 + 300)

    def test_oracle_accumulation_with_wrap(self) -> None:
        """32-bit truncation when sum_diff + s2 exceeds 0xFFFFFFFF."""
        s0 = 0xFFFFFFFF
        s1 = 0x00000000
        # sum_diff = 131070 (0x1FFFE)
        acc = 0xFFFF_0000
        # 0x1FFFE + 0xFFFF0000 = 0x1_0000FFFE -> wrapped = 0x0000FFFE
        dst, carry, full = ref_sad_u16_full(s0, s1, acc)
        self.assertEqual(dst, 0x0000_FFFE)
        self.assertEqual(carry, 1)
        self.assertEqual(full, 0x1_0000_FFFE)
        self.assertEqual(ref_sad_u16(s0, s1, acc), 0x0000_FFFE)

    def test_oracle_bounds(self) -> None:
        """sum_diff is strictly in [0, 131070]."""
        for h0 in (0, 1, 32767, 32768, 65534, 65535):
            for h1 in (0, 1, 32767, 32768, 65534, 65535):
                s0 = pack_halfwords_u32(h0, h0)
                s1 = pack_halfwords_u32(h1, h1)
                diff = ref_sad_u16_sum(s0, s1)
                self.assertGreaterEqual(diff, 0)
                self.assertLessEqual(diff, 131070)


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestVSadU16SymbolicAST(unittest.TestCase):
    """Test suite verifying symbolic AST construction, validation, and disassembly."""

    def test_ast_creation_valid_operands(self) -> None:
        """Construct AST nodes with valid VGPR, SGPR, and Literal operands."""
        instr = VSadU16Instruction(
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
        instr = VSadU16Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(10),
            src1=Operand.sgpr(4),
            src2=Operand.literal(0x00FF00FF),
        )
        self.assertEqual(instr.disassemble(), "v_sad_u16 v5, v10, s4, 0x00FF00FF")

    def test_ast_dst_must_be_vgpr(self) -> None:
        """VALU instruction destination must be a VGPR; SGPR/Literal raises TypeError."""
        with self.assertRaises(TypeError):
            VSadU16Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VSadU16Instruction(
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
        instr1 = VSadU16Instruction(Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3), Operand.vgpr(4))
        instr2 = VSadU16Instruction(Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3), Operand.vgpr(4))
        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestVSadU16ScalarInterpreter(unittest.TestCase):
    """Test suite verifying single-thread deterministic scalar interpreter execution."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_interp_basic_execution(self) -> None:
        """Execute basic v_sad_u16 instruction updating state.vgprs[dst]."""
        self.state.vgprs[1] = pack_halfwords_u32(100, 200)
        self.state.vgprs[2] = pack_halfwords_u32(0, 0)
        self.state.sgprs[0] = 0x1000

        instr = VSadU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.sgpr(0),
        )

        res = self.interp.step(instr, self.state)
        # diff = 100 + 200 = 300 -> + 0x1000 = 0x112C
        self.assertEqual(res, 0x1000 + 300)
        self.assertEqual(self.state.vgprs[0], 0x1000 + 300)

    def test_scalar_interp_multi_step_sequence(self) -> None:
        """Sequential chained execution with accumulation."""
        self.state.vgprs[1] = pack_halfwords_u32(50, 50)
        self.state.vgprs[2] = pack_halfwords_u32(0, 0)
        self.state.vgprs[0] = 0

        # Step 1: v0 = v_sad_u16(v1, v2, 0) -> diff = 100
        instr1 = VSadU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.literal(0),
        )
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.vgprs[0], 100)

        # Step 2: v0 = v_sad_u16(v1, v2, v0) -> adds another 100 -> 200
        instr2 = VSadU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )
        self.interp.step(instr2, self.state)
        self.assertEqual(self.state.vgprs[0], 200)

    def test_scalar_interp_operand_types(self) -> None:
        """Execution with combinations of VGPR, SGPR, and Literal operands."""
        self.state.vgprs[3] = 0xFFFFFFFF
        self.state.sgprs[5] = 0x00000000
        literal_val = 0x00005555

        instr = VSadU16Instruction(
            dst=Operand.vgpr(10),
            src0=Operand.vgpr(3),
            src1=Operand.sgpr(5),
            src2=Operand.literal(literal_val),
        )
        res = self.interp.step(instr, self.state)
        # sum_diff = 131070 (0x1FFFE) -> + 0x5555 = 0x25553
        expected = 131070 + literal_val
        self.assertEqual(res, expected)
        self.assertEqual(self.state.vgprs[10], expected)

    def test_scalar_interp_matches_oracle(self) -> None:
        """Check scalar interpreter against reference oracle across test vectors."""
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
            instr = VSadU16Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
            res = self.interp.step(instr, self.state)
            expected = ref_sad_u16(s0, s1, s2)
            self.assertEqual(res, expected)
            self.assertEqual(self.state.vgprs[0], expected)


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class TestVSadU16SIMD(unittest.TestCase):
    """Test suite verifying SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_simd_wave32_full_exec(self) -> None:
        """Wave32 execution with all 32 lanes active."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.vgprs[lane][1] = pack_halfwords_u32(lane, 0)
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = 0x0001_0000

        instr = VSadU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = lane + 0x0001_0000
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][0], expected)

    def test_simd_wave64_full_exec(self) -> None:
        """Wave64 execution with all 64 lanes active."""
        state = SIMDState(num_lanes=64, exec_mask=MASK64)
        for lane in range(64):
            state.vgprs[lane][1] = pack_halfwords_u32(10, 20)
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = lane

        instr = VSadU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 64)
        # diff = 30
        for lane in range(64):
            expected = 30 + lane
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

        instr = VSadU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        # diff = 65535 + 65535 = 131070 (0x1FFFE)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x1FFFE)
                self.assertEqual(state.vgprs[lane][0], 0x1FFFE)
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
            state.vgprs[lane][1] = pack_halfwords_u32(1, 1)
            state.vgprs[lane][2] = 0x00000000
            state.vgprs[lane][3] = 0

        instr = VSadU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        # diff = 2
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], 2)
                self.assertEqual(state.vgprs[lane][0], 2)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], sentinel)

    def test_simd_zero_exec_mask(self) -> None:
        """EXEC = 0 leaves all lanes completely unmodified."""
        sentinel = 0xDEAD_C0DE
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.vgprs[lane][0] = sentinel

        instr = VSadU16Instruction(
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
            state.vgprs[lane][1] = pack_halfwords_u32(lane, lane * 2)
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = 0

        instr = VSadU16Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            diff = u16(lane) + u16(lane * 2)
            self.assertEqual(results[lane], diff)
            self.assertEqual(state.vgprs[lane][5], diff)


if __name__ == "__main__":
    unittest.main()
