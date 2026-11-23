from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for AMD RDNA2 v_sad_i16.

Architecture & Instruction Specifications (AMD RDNA2 / GFX10 Vector ALU):
- v_sad_i16 (VOP3 3-operand vector ALU instruction):
    Computes the 2-channel sum of signed 16-bit absolute differences between
    packed signed halfword lanes of src0 and src1, and accumulates into the 32-bit
    unsigned accumulator src2:
        h0_0 = to_s16(src0 & 0xFFFF)
        h0_1 = to_s16((src0 >> 16) & 0xFFFF)
        h1_0 = to_s16(src1 & 0xFFFF)
        h1_1 = to_s16((src1 >> 16) & 0xFFFF)
        diff0 = abs(h0_0 - h1_0)
        diff1 = abs(h0_1 - h1_1)
        sum_diff = diff0 + diff1
        dst.u32 = (sum_diff + src2.u32) mod 2^32
        carry_out = 1 if (sum_diff + src2.u32 >= 2^32) else 0

    Outputs:
        Vector Destination (VDST): 32-bit VGPR storing dst.u32.
        Carry generation: Tracks whether (sum_diff + src2) exceeded 2^32.

    Mathematical Bounds:
        - h0_i, h1_i in [-32768, 32767] (signed two's complement int16).
        - diff_i = |h0_i - h1_i| in [0, 65535] (0x0000 to 0xFFFF).
        - Maximum difference per lane: |32767 - (-32768)| = 65535.
        - Total sum_diff = diff0 + diff1 in [0, 131070] (0x00000 to 0x1FFFE).
        - src2.u32 in [0, 2^32 - 1] (0 to 0xFFFFFFFF).
        - Maximum full sum: 131070 + 0xFFFFFFFF = 0x1_0001_FFFD.
        - Therefore, carry_out is strictly binary {0, 1}.

    Differences from v_sad_u16:
        - In v_sad_u16, 0xFFFF is unsigned 65535, giving |65535 - 0| = 65535.
        - In v_sad_i16, 0xFFFF is signed -1, giving |-1 - 0| = 1.
        - In v_sad_i16, 0x8000 (-32768) vs 0x7FFF (32767) gives max difference 65535.

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and limits
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def to_s16(val: int) -> int:
    """Convert integer to signed 16-bit integer [-32768, 32767]."""
    val &= MASK16
    return val - 65536 if (val & 0x8000) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def unpack_signed_halfwords(val: int) -> Tuple[int, int]:
    """Unpack 32-bit word into two signed 16-bit integers (h0, h1)."""
    val &= MASK32
    return to_s16(val & MASK16), to_s16((val >> 16) & MASK16)


def pack_halfwords(h0: int, h1: int) -> int:
    """Pack two 16-bit integers into an unsigned 32-bit dword."""
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def ref_abs_diff_s16(h0: int, h1: int) -> int:
    """Signed 16-bit absolute difference: |to_s16(h0) - to_s16(h1)|."""
    s0 = to_s16(h0)
    s1 = to_s16(h1)
    return abs(s0 - s1)


def ref_sad_i16_sum(s0: int, s1: int) -> int:
    """Compute the 2-channel sum of signed 16-bit absolute differences in [0, 131070]."""
    h0_0, h0_1 = unpack_signed_halfwords(s0)
    h1_0, h1_1 = unpack_signed_halfwords(s1)
    return abs(h0_0 - h1_0) + abs(h0_1 - h1_1)


def ref_sad_i16(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_sad_i16:

    dst = (sum_abs_diff_s16(s0, s1) + s2) & 0xFFFFFFFF
    """
    sum_diff = ref_sad_i16_sum(s0, s1)
    acc = s2 & MASK32
    return (sum_diff + acc) & MASK32


def ref_sad_i16_full(s0: int, s1: int, s2: int) -> Tuple[int, int, int]:
    """Return (dst_u32, carry_out, full_sum64) for v_sad_i16."""
    sum_diff = ref_sad_i16_sum(s0, s1)
    acc = s2 & MASK32
    full = sum_diff + acc
    carry = (full >> 32) & 1
    dst = full & MASK32
    return dst, carry, full


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestVSadI16Oracle(unittest.TestCase):
    """Test suite verifying mathematical bounds, properties, and edge cases of the oracle."""

    def test_oracle_zero_inputs(self) -> None:
        """All-zero inputs yield zero result."""
        self.assertEqual(ref_sad_i16(0, 0, 0), 0)
        dst, carry, full = ref_sad_i16_full(0, 0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(carry, 0)
        self.assertEqual(full, 0)

    def test_oracle_identical_inputs(self) -> None:
        """When s0 == s1, absolute differences are all zero and result is s2."""
        patterns = [0x00000000, 0x12345678, 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF]
        for pat in patterns:
            acc = 0xDEADBEEF
            self.assertEqual(ref_sad_i16(pat, pat, acc), acc)
            self.assertEqual(ref_sad_i16_sum(pat, pat), 0)

    def test_oracle_signed_boundary_differences(self) -> None:
        """Max difference between min negative (-32768, 0x8000) and max positive (32767, 0x7FFF)."""
        s0 = pack_halfwords(0x8000, 0x8000)  # -32768, -32768
        s1 = pack_halfwords(0x7FFF, 0x7FFF)  # 32767, 32767

        # |32767 - (-32768)| = 65535 per lane -> total sum = 131070 (0x1FFFE)
        sum_diff = ref_sad_i16_sum(s0, s1)
        self.assertEqual(sum_diff, 131070)
        self.assertEqual(sum_diff, 0x1FFFE)
        self.assertEqual(ref_sad_i16(s0, s1, 0), 0x1FFFE)

    def test_oracle_negative_one_representation(self) -> None:
        """0xFFFF represents signed -1, differing from 0 by exactly 1."""
        s0 = pack_halfwords(0xFFFF, 0xFFFF)  # -1, -1
        s1 = pack_halfwords(0x0000, 0x0000)  # 0, 0

        # In signed math, |-1 - 0| = 1 per lane -> total sum = 2
        sum_diff = ref_sad_i16_sum(s0, s1)
        self.assertEqual(sum_diff, 2)
        self.assertEqual(ref_sad_i16(s0, s1, 0), 2)

    def test_oracle_single_halfword_lanes(self) -> None:
        """Verify each 16-bit lane contributes independently."""
        # Lane 0: -32768 vs 0 -> 32768
        self.assertEqual(ref_sad_i16(pack_halfwords(0x8000, 0), 0, 0), 32768)
        # Lane 1: -32768 vs 0 -> 32768
        self.assertEqual(ref_sad_i16(pack_halfwords(0, 0x8000), 0, 0), 32768)

    def test_oracle_commutativity(self) -> None:
        """Commutativity: ref_sad_i16(a, b, c) == ref_sad_i16(b, a, c)."""
        test_pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0xFFFF0000, 0x0000FFFF),
            (0x80008000, 0x7FFF7FFF),
            (0xFFFFFFFF, 0x00000000),
        ]
        acc = 0xCAFEBABE
        for s0, s1 in test_pairs:
            self.assertEqual(ref_sad_i16(s0, s1, acc), ref_sad_i16(s1, s0, acc))
            self.assertEqual(ref_sad_i16_sum(s0, s1), ref_sad_i16_sum(s1, s0))

    def test_oracle_accumulation_with_wrap(self) -> None:
        """32-bit truncation when sum_diff + s2 exceeds 0xFFFFFFFF."""
        s0 = pack_halfwords(0x8000, 0x8000)
        s1 = pack_halfwords(0x7FFF, 0x7FFF)
        # sum_diff = 131070 (0x1FFFE); acc = 0xFFFF_0001
        # 0x1FFFE + 0xFFFF0001 = 0x1_0000FFFF -> wrapped = 0x0000FFFF
        dst, carry, full = ref_sad_i16_full(s0, s1, 0xFFFF_0001)
        self.assertEqual(dst, 0x0000_FFFF)
        self.assertEqual(carry, 1)
        self.assertEqual(full, 0x1_0000_FFFF)




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
class VSadI16Instruction:
    """Symbolic AST node for AMD RDNA2 v_sad_i16 VALU instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return (
            f"v_sad_i16 {self.dst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        )


class TestVSadI16SymbolicAST(unittest.TestCase):
    """Test suite verifying symbolic AST construction, validation, and disassembly."""

    def test_ast_creation_valid_operands(self) -> None:
        """Construct AST nodes with valid operands."""
        instr = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(0x1234),
        )
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.src0.index, 1)
        self.assertEqual(instr.src1.index, 2)
        self.assertEqual(instr.src2.literal_value, 0x1234)

    def test_ast_disassembly(self) -> None:
        """Disassembly formatting matches AMD RDNA ISA assembly conventions."""
        instr = VSadI16Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(10),
            src1=Operand.sgpr(4),
            src2=Operand.literal(0x00FF00FF),
        )
        self.assertEqual(instr.disassemble(), "v_sad_i16 v5, v10, s4, 0x00FF00FF")

    def test_ast_dst_must_be_vgpr(self) -> None:
        """VALU instruction destination must be a VGPR; SGPR/Literal raises TypeError."""
        with self.assertRaises(TypeError):
            VSadI16Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_ast_operand_bounds(self) -> None:
        """Operand indices and literals must be strictly bounded."""
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.literal(-1)
        with self.assertRaises(ValueError):
            Operand.literal(0x1_0000_0000)

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
    """Deterministic scalar interpreter executing v_sad_i16 instructions."""

    def step(self, instr: VSadI16Instruction, state: ScalarState) -> int:
        """Execute a single v_sad_i16 instruction against architectural scalar state."""
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        s2 = state.read_operand(instr.src2)

        res = ref_sad_i16(s0, s1, s2)
        state.write_vgpr(instr.dst.index, res)
        return res


class TestVSadI16ScalarInterpreter(unittest.TestCase):
    """Test suite verifying single-thread deterministic scalar interpreter execution."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_interp_basic_execution(self) -> None:
        """Execute basic v_sad_i16 instruction updating state.vgprs[dst]."""
        self.state.vgprs[1] = pack_halfwords(-100, 200)
        self.state.vgprs[2] = pack_halfwords(0, 0)
        self.state.sgprs[0] = 0x1000

        instr = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.sgpr(0),
        )

        res = self.interp.step(instr, self.state)
        # diff = |-100 - 0| + |200 - 0| = 300 -> + 0x1000 = 0x112C
        self.assertEqual(res, 0x1000 + 300)
        self.assertEqual(self.state.vgprs[0], 0x1000 + 300)

    def test_scalar_interp_multi_step_sequence(self) -> None:
        """Sequential chained execution with accumulation."""
        self.state.vgprs[1] = pack_halfwords(-50, -50)
        self.state.vgprs[2] = pack_halfwords(0, 0)
        self.state.vgprs[0] = 0

        # Step 1: v0 = v_sad_i16(v1, v2, 0) -> diff = 50 + 50 = 100
        instr1 = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.literal(0),
        )
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.vgprs[0], 100)

        # Step 2: v0 = v_sad_i16(v1, v2, v0) -> adds another 100 -> 200
        instr2 = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )
        self.interp.step(instr2, self.state)
        self.assertEqual(self.state.vgprs[0], 200)

    def test_scalar_interp_matches_oracle(self) -> None:
        """Check scalar interpreter against reference oracle across test vectors."""
        test_vectors = [
            (0x00000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0x00000000, 0x00000000),
            (0x80007FFF, 0x7FFF8000, 0x00010000),
            (0x55555555, 0xAAAAAAAA, 0xFFFFFFFF),
        ]
        for s0, s1, s2 in test_vectors:
            self.state.vgprs[1] = s0
            self.state.vgprs[2] = s1
            self.state.vgprs[3] = s2
            instr = VSadI16Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
            res = self.interp.step(instr, self.state)
            expected = ref_sad_i16(s0, s1, s2)
            self.assertEqual(res, expected)
            self.assertEqual(self.state.vgprs[0], expected)

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

    def step(self, instr: VSadI16Instruction, state: SIMDState) -> List[Optional[int]]:
        """Execute a v_sad_i16 instruction across all SIMD lanes under EXEC masking."""
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                s0 = state.read_lane(instr.src0, lane)
                s1 = state.read_lane(instr.src1, lane)
                s2 = state.read_lane(instr.src2, lane)

                res = ref_sad_i16(s0, s1, s2)
                state.write_lane(instr.dst, lane, res)
                results.append(res)
            else:
                results.append(None)
        return results


class TestVSadI16SIMD(unittest.TestCase):
    """Test suite verifying SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_simd_wave32_full_exec(self) -> None:
        """Wave32 execution with all 32 lanes active."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.vgprs[lane][1] = pack_halfwords(-lane, 0)
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = 0x0001_0000

        instr = VSadI16Instruction(
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
            state.vgprs[lane][1] = pack_halfwords(-10, 20)
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = lane

        instr = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 64)
        # diff = 10 + 20 = 30
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
            state.vgprs[lane][1] = pack_halfwords(0x8000, 0x8000)
            state.vgprs[lane][2] = pack_halfwords(0x7FFF, 0x7FFF)
            state.vgprs[lane][3] = 0

        instr = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 131070)
                self.assertEqual(state.vgprs[lane][0], 131070)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], sentinel)

# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestVSadI16RegisterAliasing(unittest.TestCase):
    """Test suite verifying register aliasing and in-place operand preservation."""

    def test_aliasing_dst_equals_src0(self) -> None:
        """dst and src0 share the same VGPR: dst must be read before overwritten."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[0] = pack_halfwords(-100, 200)
        state.vgprs[1] = pack_halfwords(0, 0)
        state.sgprs[0] = 0x1234

        instr = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.sgpr(0),
        )

        res = interp.step(instr, state)
        expected = 300 + 0x1234
        self.assertEqual(res, expected)
        self.assertEqual(state.vgprs[0], expected)

    def test_aliasing_dst_equals_src2_accumulator_in_place(self) -> None:
        """dst and src2 share the same VGPR (canonical in-place accumulation loop)."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[0] = 1000  # accumulator in v0
        state.vgprs[1] = pack_halfwords(-10, 20)
        state.vgprs[2] = pack_halfwords(0, 0)

        # v0 = v_sad_i16(v1, v2, v0)
        instr = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        # Iteration 1: + 30 -> 1030
        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 1030)

        # Iteration 2: + 30 -> 1060
        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 1060)

    def test_aliasing_src0_equals_src1(self) -> None:
        """src0 == src1 produces diff=0, preserving accumulator exactly."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[1] = 0xDEADBEEF
        state.vgprs[2] = 0xCAFEBABE

        instr = VSadI16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )

        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 0xCAFEBABE)

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
    """Evaluate condition codes and status flags resulting from v_sad_i16 execution."""
    dst, carry, _ = ref_sad_i16_full(s0, s1, s2)
    sum_diff = ref_sad_i16_sum(s0, s1) & MASK32
    acc = s2 & MASK32

    zero = (dst == 0)
    sign = bool(dst & 0x8000_0000)
    overflow = bool((~(sum_diff ^ acc) & (sum_diff ^ dst)) & 0x8000_0000)

    return StatusFlags(zero=zero, sign=sign, carry=bool(carry), overflow=overflow)


class TestVSadI16ConditionCodesAndFlags(unittest.TestCase):
    """Test suite verifying condition codes, status flags, and carry generation."""

    def test_flags_zero_result(self) -> None:
        """Zero flag asserted when dst evaluates to 0x00000000."""
        flags = compute_flags(0x12345678, 0x12345678, 0)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry)

        # diff = 1, acc = 0xFFFFFFFF -> sum = 0x1_00000000 -> dst = 0
        flags_wrap = compute_flags(pack_halfwords(-1, 0), pack_halfwords(0, 0), 0xFFFF_FFFF)
        self.assertTrue(flags_wrap.zero)
        self.assertFalse(flags_wrap.sign)
        self.assertTrue(flags_wrap.carry)

    def test_flags_sign_bit_set(self) -> None:
        """Sign flag asserted when dst MSB (bit 31) is set."""
        flags = compute_flags(0, 0, 0x8000_0000)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.carry)

    def test_flags_carry_out_generated(self) -> None:
        """Carry flag asserted when full 64-bit sum overflows 32 bits."""
        s0 = pack_halfwords(0x8000, 0x8000)
        s1 = pack_halfwords(0x7FFF, 0x7FFF)
        # sum_diff = 131070 (0x1FFFE); acc = 0xFFFF_0001
        flags = compute_flags(s0, s1, 0xFFFF_0001)
        self.assertTrue(flags.carry)

if __name__ == "__main__":
    unittest.main()
