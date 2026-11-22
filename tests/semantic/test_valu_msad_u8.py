from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_msad_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_msad_u8 dst, src0, src1, src2

Operation:
  Performs 4-channel masked sum of absolute differences across packed 8-bit
  unsigned byte lanes within 32-bit dwords, with 32-bit unsigned accumulation:
    For each byte lane i in {0, 1, 2, 3}:
      s0_i = (src0 >> (8 * i)) & 0xFF
      s1_i = (src1 >> (8 * i)) & 0xFF
      diff_i = 0 if s1_i == 0 else |s0_i - s1_i|
    sad_sum = diff_0 + diff_1 + diff_2 + diff_3
    dst = (src2 + sad_sum) mod 2^32

Mathematical and Architectural Properties:
  - Masking semantics:
    - Byte lane i is masked out (omitted) if and only if reference byte s1_i == 0.
    - If s1_i == 0, diff_i is strictly 0 regardless of s0_i (even if s0_i == 0xFF).
    - If s1_i != 0, diff_i is the unsigned absolute difference |s0_i - s1_i| in [0, 255].
  - Asymmetry / Non-commutativity:
    - Unlike unmasked v_sad_u8, v_msad_u8 is NOT commutative with respect to src0 and src1:
      the mask is dictated solely by src1.
  - Range and bounds:
    - Each unmasked byte difference is in [0, 255].
    - Maximum sad_sum across 4 lanes is 4 * 255 = 1020 (0x3FC).
    - Minimum sad_sum is 0 (all lanes masked, or all active lanes equal).
  - Accumulator wrapping:
    - src2 is an unsigned 32-bit integer.
    - The addition (src2 + sad_sum) wraps modulo 2^32 with standard unsigned arithmetic.
  - Clamp modifier:
    - In AMD RDNA2 ISA / Leonida translator contract, clamp modifier is unsupported
      and rejected rather than guessed.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
BYTE_SHIFTS: Tuple[int, int, int, int] = (0, 8, 16, 24)
MAX_SAD_SUM: int = 4 * 255  # 1020 (0x3FC)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def unpack_bytes_u32(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned word into 4 byte lanes (b0, b1, b2, b3)."""
    val &= MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_bytes_u32(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit unsigned word."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_msad_byte(s0_byte: int, s1_byte: int) -> int:
    """Compute single byte-lane masked absolute difference.

    If s1_byte is 0, the lane is masked out and contributes 0.
    Otherwise, returns the unsigned absolute difference |s0_byte - s1_byte|.
    """
    b0 = s0_byte & MASK8
    b1 = s1_byte & MASK8
    if b1 == 0:
        return 0
    return (b0 - b1) if b0 >= b1 else (b1 - b0)


def oracle_v_msad_u8(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_msad_u8.

    Unpacks 4 byte lanes from src0 and src1, computes masked absolute differences
    conditioned on src1 byte non-zero, sums the differences, and adds accumulator src2.
    Result wraps modulo 2^32.
    """
    s0_lanes = unpack_bytes_u32(src0)
    s1_lanes = unpack_bytes_u32(src1)

    diff0 = ref_msad_byte(s0_lanes[0], s1_lanes[0])
    diff1 = ref_msad_byte(s0_lanes[1], s1_lanes[1])
    diff2 = ref_msad_byte(s0_lanes[2], s1_lanes[2])
    diff3 = ref_msad_byte(s0_lanes[3], s1_lanes[3])

    sad_sum = diff0 + diff1 + diff2 + diff3
    acc = src2 & MASK32
    return (acc + sad_sum) & MASK32


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg32:
    """Represents a 32-bit architectural register (VGPR or SGPR)."""
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
    """Represents a 32-bit integer literal or inline constant."""
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= MASK32):
            raise ValueError(f"Literal32 value {self.value:#x} out of range [0, 0xFFFFFFFF]")

    @property
    def name(self) -> str:
        return f"{self.value:#010x}" if self.value > 9 else str(self.value)


@dataclass(frozen=True)
class VMsadOperand:
    """Typed expression operand representing a register or literal."""
    source: Union[Reg32, Literal32]

    def disassemble(self) -> str:
        return self.source.name


@dataclass(frozen=True)
class VMsadU8Instruction:
    """Symbolic AST node representing an RDNA2 v_msad_u8 VALU instruction."""
    dst: Reg32
    src0: VMsadOperand
    src1: VMsadOperand
    src2: VMsadOperand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.clamp:
            raise ValueError("clamp modifier is unsupported on v_msad_u8")

    def disassemble(self) -> str:
        return f"v_msad_u8 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single execution thread."""
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
    """Deterministic scalar instruction interpreter for v_msad_u8."""

    def resolve_operand(self, op: VMsadOperand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VMsadU8Instruction, state: ScalarMachineState) -> int:
        # Atomic operand read phase
        s0 = self.resolve_operand(instr.src0, state)
        s1 = self.resolve_operand(instr.src1, state)
        s2 = self.resolve_operand(instr.src2, state)

        # Computation phase via reference oracle
        result = oracle_v_msad_u8(s0, s1, s2)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
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
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane {lane} out of range [0, {self.num_lanes - 1}]")
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane {lane} out of range [0, {self.num_lanes - 1}]")
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[lane][index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VMsadOperand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VMsadU8Instruction, state: SIMDState) -> List[Optional[int]]:
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
        s2_vals = [
            self.resolve_lane_operand(instr.src2, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution and selective write-back phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_msad_u8(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class VMsadStatusFlags:
    """Architectural condition codes, status flags, and carry generation for v_msad_u8."""
    zero: bool
    sign: bool
    carry_out: bool
    masked_lanes_count: int
    active_lanes_count: int
    lane_masked: Tuple[bool, bool, bool, bool]
    lane_diff: Tuple[int, int, int, int]
    diff_sum: int


def evaluate_v_msad_u8_flags(src0: int, src1: int, src2: int, result: int) -> VMsadStatusFlags:
    """Evaluate architectural condition codes, masking status, and 32-bit carry generation."""
    s0_lanes = unpack_bytes_u32(src0)
    s1_lanes = unpack_bytes_u32(src1)
    acc = src2 & MASK32

    lane_masked: List[bool] = []
    lane_diff: List[int] = []

    for i in range(4):
        b0 = s0_lanes[i]
        b1 = s1_lanes[i]
        if b1 == 0:
            lane_masked.append(True)
            lane_diff.append(0)
        else:
            lane_masked.append(False)
            lane_diff.append(abs(b0 - b1))

    diff_sum = sum(lane_diff)
    full_sum = acc + diff_sum
    carry_out = full_sum > MASK32
    res32 = result & MASK32

    zero = (res32 == 0)
    sign = bool(res32 & 0x8000_0000)
    masked_count = sum(1 for m in lane_masked if m)
    active_count = 4 - masked_count

    return VMsadStatusFlags(
        zero=zero,
        sign=sign,
        carry_out=carry_out,
        masked_lanes_count=masked_count,
        active_lanes_count=active_count,
        lane_masked=(lane_masked[0], lane_masked[1], lane_masked[2], lane_masked[3]),
        lane_diff=(lane_diff[0], lane_diff[1], lane_diff[2], lane_diff[3]),
        diff_sum=diff_sum,
    )




# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMsadU8ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_msad_u8 architectural bit model and reference oracle."""

    def test_byte_unpack_pack_roundtrip(self) -> None:
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55A3C7E,
            0x01020304,
            0x80402010,
        ]
        for pat in patterns:
            with self.subTest(pattern=hex(pat)):
                b0, b1, b2, b3 = unpack_bytes_u32(pat)
                repacked = pack_bytes_u32(b0, b1, b2, b3)
                self.assertEqual(repacked, pat)

    def test_all_lanes_masked_when_s1_is_zero(self) -> None:
        # When src1 is 0, all 4 byte lanes are masked out -> diff = 0 regardless of src0
        for s0 in [0x00000000, 0xFFFFFFFF, 0x12345678, 0x80808080]:
            with self.subTest(s0=hex(s0)):
                self.assertEqual(oracle_v_msad_u8(s0, 0x00000000, 0), 0)
                self.assertEqual(oracle_v_msad_u8(s0, 0x00000000, 100), 100)
                self.assertEqual(oracle_v_msad_u8(s0, 0x00000000, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_all_lanes_unmasked_known_values(self) -> None:
        # s0 = [10, 20, 30, 40], s1 = [15, 20, 25, 50], s2 = 100
        # diffs: |10-15|=5, |20-20|=0, |30-25|=5, |40-50|=10 -> sum = 20
        # result = 100 + 20 = 120
        s0 = pack_bytes_u32(10, 20, 30, 40)
        s1 = pack_bytes_u32(15, 20, 25, 50)
        self.assertEqual(oracle_v_msad_u8(s0, s1, 100), 120)

    def test_partial_masking_selective_skips(self) -> None:
        # S0 = [0xFF, 0x20, 0xFF, 0x40]
        # S1 = [0x00, 0x20, 0x00, 0x30]  (lanes 0 and 2 masked out)
        # diffs: lane 0: masked (0), lane 1: |0x20 - 0x20|=0, lane 2: masked (0), lane 3: |0x40 - 0x30|=0x10 (16)
        # Total sad_sum = 16
        s0 = 0x40FF20FF
        s1 = 0x30002000
        self.assertEqual(oracle_v_msad_u8(s0, s1, 0), 16)
        self.assertEqual(oracle_v_msad_u8(s0, s1, 1000), 1016)

    def test_masking_asymmetry_non_commutativity(self) -> None:
        # When zeros appear, swapping s0 and s1 gives different results:
        # Case 1: s0 has 0, s1 has non-zero -> S1 != 0, so diff = |0 - X| = X
        # Case 2: s0 has non-zero, s1 has 0 -> S1 == 0, so diff = 0 (masked!)
        s0 = pack_bytes_u32(0, 10, 0, 20)
        s1 = pack_bytes_u32(10, 0, 20, 0)
        res_0_1 = oracle_v_msad_u8(s0, s1, 0)
        res_1_0 = oracle_v_msad_u8(s1, s0, 0)

        # In res_0_1: s1 has [10, 0, 20, 0] -> lanes 0, 2 active (|0-10|=10, |0-20|=20) -> sum = 30
        self.assertEqual(res_0_1, 30)
        # In res_1_0: s0 in s1 position has [0, 10, 0, 20] -> lanes 1, 3 active (|0-10|=10, |0-20|=20) -> sum = 30
        # If we pick asymmetric values:
        s_a = pack_bytes_u32(50, 0, 0, 0)
        s_b = pack_bytes_u32(0, 50, 0, 0)
        # v_msad(s_a, s_b, 0): s_b has lane 1 = 50 != 0 -> |0 - 50| = 50. Total = 50
        # v_msad(s_b, s_a, 0): s_a has lane 0 = 50 != 0 -> |0 - 50| = 50.
        # Now asymmetrical:
        s_x = pack_bytes_u32(100, 200, 0, 0)
        s_y = pack_bytes_u32(0, 50, 0, 0)
        # msad(x, y, 0): y has lane 0 = 0 (masked), lane 1 = 50 (|200-50|=150) -> sum = 150
        # msad(y, x, 0): x has lane 0 = 100 (|0-100|=100), lane 1 = 200 (|50-200|=150) -> sum = 250
        self.assertEqual(oracle_v_msad_u8(s_x, s_y, 0), 150)
        self.assertEqual(oracle_v_msad_u8(s_y, s_x, 0), 250)
        self.assertNotEqual(oracle_v_msad_u8(s_x, s_y, 0), oracle_v_msad_u8(s_y, s_x, 0))

    def test_maximum_delta_accumulation(self) -> None:
        # Max delta: all 4 lanes max difference: |0xFF - 0x00| is not allowed for s1 (s1=0 masks it),
        # but |0x00 - 0xFF| = 255, with s1 = 0xFF!
        # Or |0xFF - 0x01| = 254 with s1 = 0x01.
        # Max difference per lane is |0x00 - 0xFF| = 255 when s0 = 0, s1 = 0xFF.
        # Across 4 lanes: 4 * 255 = 1020 (0x3FC).
        s0_zero = 0x00000000
        s1_max = 0xFFFFFFFF
        sad_max = oracle_v_msad_u8(s0_zero, s1_max, 0)
        self.assertEqual(sad_max, MAX_SAD_SUM)
        self.assertEqual(sad_max, 1020)

    def test_accumulator_32bit_wraparound(self) -> None:
        # When acc + diff exceeds 0xFFFFFFFF, wraps modulo 2^32
        s0 = 0x00000000
        s1 = 0xFFFFFFFF  # diff = 1020
        # acc = 0xFFFFFFFF -> (0xFFFFFFFF + 1020) mod 2^32 = 1019
        self.assertEqual(oracle_v_msad_u8(s0, s1, 0xFFFFFFFF), 1019)

        # Exact wrap boundary: acc = 0xFFFF_FC04 (4294966276) + 1020 = 0x1_0000_0000 -> 0
        acc_exact = 0xFFFF_FC04
        self.assertEqual(oracle_v_msad_u8(s0, s1, acc_exact), 0)

    def test_identity_when_equal_and_nonzero(self) -> None:
        # When src0 == src1 and all bytes != 0, diff = 0 for all lanes, so result = src2
        val = 0x01020304
        self.assertEqual(oracle_v_msad_u8(val, val, 0x12345678), 0x12345678)
        self.assertEqual(oracle_v_msad_u8(0xFFFFFFFF, 0xFFFFFFFF, 42), 42)


class TestValuMsadU8SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_validation_and_naming(self) -> None:
        v0 = Reg32(0, RegType.VGPR)
        v255 = Reg32(255, RegType.VGPR)
        s12 = Reg32(12, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(s12.name, "s12")

    def test_register_bounds_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(1000, RegType.SGPR)

    def test_literal_values_and_naming(self) -> None:
        lit_small = Literal32(1)
        lit_hex = Literal32(0x10203040)
        self.assertEqual(lit_small.name, "1")
        self.assertEqual(lit_hex.name, "0x10203040")

        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_instruction_ast_construction_and_disassembly(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src0 = VMsadOperand(Reg32(1, RegType.VGPR))
        src1 = VMsadOperand(Reg32(4, RegType.SGPR))
        src2 = VMsadOperand(Literal32(100))

        instr = VMsadU8Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(instr.disassemble(), "v_msad_u8 v0, v1, s4, 0x00000064")

    def test_destination_must_be_vgpr(self) -> None:
        dst_sgpr = Reg32(0, RegType.SGPR)
        src = VMsadOperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VMsadU8Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)

    def test_clamp_modifier_rejection_in_ast(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src = VMsadOperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(ValueError):
            VMsadU8Instruction(dst=dst, src0=src, src1=src, src2=src, clamp=True)


class TestValuMsadU8ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_state_get_set(self) -> None:
        self.assertEqual(self.state.get_vgpr(0), 0)
        self.assertEqual(self.state.get_sgpr(0), 0)

        self.state.set_vgpr(10, 0x12345678)
        self.assertEqual(self.state.get_vgpr(10), 0x12345678)

        self.state.set_sgpr(5, 0xABCDEF01)
        self.assertEqual(self.state.get_sgpr(5), 0xABCDEF01)

        # 32-bit truncation masking
        self.state.set_vgpr(1, 0x1_0000_0042)
        self.assertEqual(self.state.get_vgpr(1), 0x42)

        with self.assertRaises(IndexError):
            self.state.get_vgpr(-1)
        with self.assertRaises(IndexError):
            self.state.get_vgpr(256)
        with self.assertRaises(IndexError):
            self.state.get_sgpr(-1)
        with self.assertRaises(IndexError):
            self.state.get_sgpr(256)

    def test_scalar_step_vgpr_sources(self) -> None:
        # s0 = [10, 20, 30, 40], s1 = [10, 0, 30, 50], s2 = 100
        # lane 1 is masked (s1 byte == 0)
        # diffs: |10-10|=0, 0 (masked), |30-30|=0, |40-50|=10 -> sum = 10
        # result = 100 + 10 = 110
        s0_val = pack_bytes_u32(10, 20, 30, 40)
        s1_val = pack_bytes_u32(10, 0, 30, 50)
        s2_val = 100

        self.state.set_vgpr(1, s0_val)
        self.state.set_vgpr(2, s1_val)
        self.state.set_vgpr(3, s2_val)

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Reg32(3, RegType.VGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 110)
        self.assertEqual(self.state.get_vgpr(0), 110)

    def test_scalar_step_sgpr_and_literal_sources(self) -> None:
        s0_val = pack_bytes_u32(0, 0, 0, 0)
        s1_val = 0xFFFFFFFF  # max diff 1020

        self.state.set_vgpr(1, s0_val)
        self.state.set_sgpr(4, s1_val)

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(4, RegType.SGPR)),
            src2=VMsadOperand(Literal32(500)),
        )

        res = self.interp.step(instr, self.state)
        # 500 + 1020 = 1520
        self.assertEqual(res, 1520)
        self.assertEqual(self.state.get_vgpr(0), 1520)

    def test_sequential_instruction_flow(self) -> None:
        # Step 1: v0 = v_msad_u8(v1, v2, 0)
        # Step 2: v3 = v_msad_u8(v1, v2, v0)  (accumulating with previous result)
        self.state.set_vgpr(1, pack_bytes_u32(1, 2, 3, 4))
        self.state.set_vgpr(2, pack_bytes_u32(2, 3, 4, 5))

        # diffs: |1-2|=1, |2-3|=1, |3-4|=1, |4-5|=1 -> sum = 4
        instr1 = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Literal32(0)),
        )
        res1 = self.interp.step(instr1, self.state)
        self.assertEqual(res1, 4)
        self.assertEqual(self.state.get_vgpr(0), 4)

        instr2 = VMsadU8Instruction(
            dst=Reg32(3, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Reg32(0, RegType.VGPR)),
        )
        res2 = self.interp.step(instr2, self.state)
        # 4 + 4 = 8
        self.assertEqual(res2, 8)
        self.assertEqual(self.state.get_vgpr(3), 8)


class TestValuMsadU8SIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        for lane in range(32):
            s0_val = pack_bytes_u32(lane * 2, lane * 3, lane * 4, lane * 5)
            s1_val = pack_bytes_u32(10, 20, 30, 40)
            s2_val = 100 + lane
            state.set_vgpr(lane, 1, s0_val)
            state.set_vgpr(lane, 2, s1_val)
            state.set_vgpr(lane, 3, s2_val)

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Reg32(3, RegType.VGPR)),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = oracle_v_msad_u8(
                state.get_vgpr(lane, 1),
                state.get_vgpr(lane, 2),
                state.get_vgpr(lane, 3),
            )
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=64)
        interp = SIMDInterpreter()

        for lane in range(64):
            state.set_vgpr(lane, 1, pack_bytes_u32(lane, lane ^ 0x33, 100, 200))
            state.set_vgpr(lane, 2, pack_bytes_u32(50, 60, 70, 80))

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Literal32(10)),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = oracle_v_msad_u8(
                state.get_vgpr(lane, 1),
                state.get_vgpr(lane, 2),
                10,
            )
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_partial_exec_mask_even_lanes(self) -> None:
        sentinel = 0xDEADBEEF
        state = SIMDState(num_lanes=32, exec_mask=0x55555555)  # only even lanes active
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, 0x10203040)
            state.set_vgpr(lane, 2, 0x50607080)

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Literal32(0)),
        )

        results = interp.step(instr, state)
        expected = oracle_v_msad_u8(0x10203040, 0x50607080, 0)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_zero_exec_mask_no_modification(self) -> None:
        sentinel = 0xCAFEBABE
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Literal32(0xFFFFFFFF)),
            src1=VMsadOperand(Literal32(0xFFFFFFFF)),
            src2=VMsadOperand(Literal32(100)),
        )

        results = interp.step(instr, state)
        self.assertEqual(results, [None] * 32)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_sgpr_scalar_broadcast_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        state.set_sgpr(5, 0x20202020)  # broadcast to all lanes
        for lane in range(32):
            state.set_vgpr(lane, 1, pack_bytes_u32(lane, lane, lane, lane))

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(5, RegType.SGPR)),
            src2=VMsadOperand(Literal32(0)),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            expected = oracle_v_msad_u8(
                pack_bytes_u32(lane, lane, lane, lane),
                0x20202020,
                0,
            )
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_divergent_masking_pattern_across_lanes(self) -> None:
        # Test divergent masking where odd lanes have S1 with zeros and even lanes do not
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_vgpr(lane, 1, 0xFFFFFFFF)
            if lane % 2 == 0:
                state.set_vgpr(lane, 2, 0xFFFFFFFF)  # no masking: diffs all 0
            else:
                state.set_vgpr(lane, 2, 0x00000000)  # all masked: diffs 0

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Literal32(42)),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(results[lane], 42)
            self.assertEqual(state.get_vgpr(lane, 0), 42)


class TestValuMsadU8RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x10203040
        val_b = 0x30405060
        acc_val = 1000

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, acc_val)

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(0, RegType.VGPR)),  # Aliased with dst
            src1=VMsadOperand(Reg32(1, RegType.VGPR)),
            src2=VMsadOperand(Reg32(2, RegType.VGPR)),
        )

        expected = oracle_v_msad_u8(val_a, val_b, acc_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)
        # Verify src1 and src2 remain intact
        self.assertEqual(state.get_vgpr(1), val_b)
        self.assertEqual(state.get_vgpr(2), acc_val)

    def test_dst_aliases_src1(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x10203040
        val_b = 0x30005000  # lanes 0 and 2 masked
        acc_val = 500

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, acc_val)

        instr = VMsadU8Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VMsadOperand(Reg32(0, RegType.VGPR)),
            src1=VMsadOperand(Reg32(1, RegType.VGPR)),  # Aliased with dst
            src2=VMsadOperand(Reg32(2, RegType.VGPR)),
        )

        expected = oracle_v_msad_u8(val_a, val_b, acc_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(1), expected)
        self.assertEqual(state.get_vgpr(0), val_a)
        self.assertEqual(state.get_vgpr(2), acc_val)

    def test_dst_aliases_src2(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x10203040
        val_b = 0x30405060
        acc_val = 200

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, acc_val)

        instr = VMsadU8Instruction(
            dst=Reg32(2, RegType.VGPR),
            src0=VMsadOperand(Reg32(0, RegType.VGPR)),
            src1=VMsadOperand(Reg32(1, RegType.VGPR)),
            src2=VMsadOperand(Reg32(2, RegType.VGPR)),  # Aliased with dst
        )

        expected = oracle_v_msad_u8(val_a, val_b, acc_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(2), expected)
        self.assertEqual(state.get_vgpr(0), val_a)
        self.assertEqual(state.get_vgpr(1), val_b)

    def test_dst_aliases_src0_and_src1_identity_zero_diff(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val = 0x55AA00CC  # note byte 1 is 0
        acc = 1234
        state.set_vgpr(0, val)
        state.set_vgpr(1, acc)

        # When src0 == src1, diff is 0 for all non-zero bytes (|x-x|=0),
        # and 0 for zero bytes (masked). So sad_sum is strictly 0.
        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(0, RegType.VGPR)),
            src1=VMsadOperand(Reg32(0, RegType.VGPR)),
            src2=VMsadOperand(Reg32(1, RegType.VGPR)),
        )

        res = interp.step(instr, state)
        self.assertEqual(res, acc)
        self.assertEqual(state.get_vgpr(0), acc)

    def test_dst_aliases_all_operands(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val = 0x7E8000FF
        state.set_vgpr(0, val)

        # v_msad_u8 v0, v0, v0, v0:
        # sad_sum(v0, v0) is 0, acc is v0 -> result is 0 + v0 = v0
        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(0, RegType.VGPR)),
            src1=VMsadOperand(Reg32(0, RegType.VGPR)),
            src2=VMsadOperand(Reg32(0, RegType.VGPR)),
        )

        res = interp.step(instr, state)
        self.assertEqual(res, val)
        self.assertEqual(state.get_vgpr(0), val)

    def test_simd_multi_lane_aliased_dst_preservation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # lanes 0..15 active, 16..31 inactive
        interp = SIMDInterpreter()

        original_v0 = [0x10000000 | lane for lane in range(32)]
        for lane in range(32):
            state.set_vgpr(lane, 0, original_v0[lane])
            state.set_vgpr(lane, 1, 0x20202020)

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(0, RegType.VGPR)),  # Aliased
            src1=VMsadOperand(Reg32(1, RegType.VGPR)),
            src2=VMsadOperand(Literal32(0)),
        )

        interp.step(instr, state)

        for lane in range(32):
            if lane < 16:
                expected = oracle_v_msad_u8(original_v0[lane], 0x20202020, 0)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), original_v0[lane])

    def test_chained_in_place_accumulation(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        # Step 0: v0 starts at 0
        state.set_vgpr(0, 0)
        state.set_vgpr(1, pack_bytes_u32(10, 20, 30, 40))
        state.set_vgpr(2, pack_bytes_u32(15, 25, 35, 45))  # diff = 5*4 = 20

        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Reg32(1, RegType.VGPR)),
            src1=VMsadOperand(Reg32(2, RegType.VGPR)),
            src2=VMsadOperand(Reg32(0, RegType.VGPR)),  # accum from v0
        )

        # Run 5 iterations: v0 should accumulate 20 each time
        for i in range(1, 6):
            res = interp.step(instr, state)
            self.assertEqual(res, 20 * i)
            self.assertEqual(state.get_vgpr(0), 20 * i)


class TestValuMsadU8ConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_zero_flag_when_result_is_zero(self) -> None:
        # Case 1: 0 + 0 = 0
        res = oracle_v_msad_u8(0, 0, 0)
        flags = evaluate_v_msad_u8_flags(0, 0, 0, res)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry_out)
        self.assertEqual(flags.masked_lanes_count, 4)
        self.assertEqual(flags.active_lanes_count, 0)

        # Case 2: exact 32-bit overflow wrapping to 0:
        # acc = 0xFFFF_FC04, diff_sum = 1020 -> full sum = 0x1_0000_0000 -> res = 0
        s0 = 0x00000000
        s1 = 0xFFFFFFFF
        acc = 0xFFFF_FC04
        res_wrap = oracle_v_msad_u8(s0, s1, acc)
        self.assertEqual(res_wrap, 0)
        flags_wrap = evaluate_v_msad_u8_flags(s0, s1, acc, res_wrap)
        self.assertTrue(flags_wrap.zero)
        self.assertFalse(flags_wrap.sign)
        self.assertTrue(flags_wrap.carry_out)  # Overflowed!
        self.assertEqual(flags_wrap.diff_sum, 1020)

    def test_sign_flag_high_bit_set(self) -> None:
        # acc has bit 31 set
        acc_neg = 0x8000_0000
        res = oracle_v_msad_u8(0, 0, acc_neg)
        flags = evaluate_v_msad_u8_flags(0, 0, acc_neg, res)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.carry_out)

        # acc just below bit 31 (0x7FFF_FFFF) + diff 1 -> 0x8000_0000
        s0 = pack_bytes_u32(1, 0, 0, 0)
        s1 = pack_bytes_u32(2, 0, 0, 0)  # diff = 1
        res_trans = oracle_v_msad_u8(s0, s1, 0x7FFF_FFFF)
        self.assertEqual(res_trans, 0x8000_0000)
        flags_trans = evaluate_v_msad_u8_flags(s0, s1, 0x7FFF_FFFF, res_trans)
        self.assertTrue(flags_trans.sign)

    def test_carry_out_32bit_overflow(self) -> None:
        # 0xFFFFFFFF + diff 1 -> carry_out True
        s0 = pack_bytes_u32(1, 0, 0, 0)
        s1 = pack_bytes_u32(2, 0, 0, 0)  # diff = 1
        res = oracle_v_msad_u8(s0, s1, 0xFFFFFFFF)
        self.assertEqual(res, 0)
        flags = evaluate_v_msad_u8_flags(s0, s1, 0xFFFFFFFF, res)
        self.assertTrue(flags.carry_out)
        self.assertTrue(flags.zero)

    def test_carry_out_no_overflow(self) -> None:
        # 0x00001000 + diff 100 -> no overflow
        res = oracle_v_msad_u8(0, 0, 0x1000)
        flags = evaluate_v_msad_u8_flags(0, 0, 0x1000, res)
        self.assertFalse(flags.carry_out)

    def test_masked_lanes_counting(self) -> None:
        # 0 masked lanes: all bytes non-zero
        s1_none = pack_bytes_u32(1, 2, 3, 4)
        flags_none = evaluate_v_msad_u8_flags(0, s1_none, 0, 0)
        self.assertEqual(flags_none.masked_lanes_count, 0)
        self.assertEqual(flags_none.active_lanes_count, 4)
        self.assertEqual(flags_none.lane_masked, (False, False, False, False))

        # 2 masked lanes: bytes 0 and 2 are 0
        s1_two = pack_bytes_u32(0, 5, 0, 10)
        flags_two = evaluate_v_msad_u8_flags(0, s1_two, 0, 0)
        self.assertEqual(flags_two.masked_lanes_count, 2)
        self.assertEqual(flags_two.active_lanes_count, 2)
        self.assertEqual(flags_two.lane_masked, (True, False, True, False))

        # 4 masked lanes: all 0
        flags_four = evaluate_v_msad_u8_flags(0, 0, 0, 0)
        self.assertEqual(flags_four.masked_lanes_count, 4)
        self.assertEqual(flags_four.active_lanes_count, 0)
        self.assertEqual(flags_four.lane_masked, (True, True, True, True))

    def test_per_lane_diff_tracking(self) -> None:
        # Lane 0: s0=10, s1=20 -> diff 10
        # Lane 1: s0=50, s1=0  -> masked -> diff 0
        # Lane 2: s0=100, s1=75 -> diff 25
        # Lane 3: s0=0, s1=50  -> diff 50
        s0 = pack_bytes_u32(10, 50, 100, 0)
        s1 = pack_bytes_u32(20, 0, 75, 50)
        res = oracle_v_msad_u8(s0, s1, 0)
        self.assertEqual(res, 10 + 0 + 25 + 50)  # 85
        flags = evaluate_v_msad_u8_flags(s0, s1, 0, res)
        self.assertEqual(flags.lane_diff, (10, 0, 25, 50))
        self.assertEqual(flags.diff_sum, 85)


class TestValuMsadU8IEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 float bit pattern handling, byte extremes, and boundary limits."""

    def test_ieee754_special_float_bitpatterns_treated_as_raw_bytes(self) -> None:
        # v_msad_u8 is an integer byte VALU instruction.
        # When shaders feed bit-casts of IEEE 754 floats (+0, -0, +inf, -inf, qnan, snan),
        # hardware treats them strictly as raw byte fields without FPU traps, flushing, or quietening.
        pos_zero = 0x0000_0000
        neg_zero = 0x8000_0000
        pos_inf = 0x7F80_0000
        neg_inf = 0xFF80_0000
        qnan = 0x7FC0_0000
        snan = 0x7F80_0001

        special_floats = [pos_zero, neg_zero, pos_inf, neg_inf, qnan, snan]
        for f0 in special_floats:
            for f1 in special_floats:
                with self.subTest(f0=hex(f0), f1=hex(f1)):
                    res0 = oracle_v_msad_u8(f0, f1, 0)
                    res1 = oracle_v_msad_u8(f0, f1, 100)

                    # Compute expected via byte unpack
                    b0_lanes = unpack_bytes_u32(f0)
                    b1_lanes = unpack_bytes_u32(f1)
                    expected_diff = sum(
                        0 if b1 == 0 else abs(b0 - b1)
                        for b0, b1 in zip(b0_lanes, b1_lanes)
                    )
                    self.assertEqual(res0, expected_diff)
                    self.assertEqual(res1, 100 + expected_diff)

    def test_negative_zero_float_bit_pattern_as_reference(self) -> None:
        # -0.0f is 0x8000_0000:
        # bytes 0, 1, 2 are 0x00 (masked!)
        # byte 3 is 0x80 (active!)
        neg_zero = 0x8000_0000
        s0 = pack_bytes_u32(0xFF, 0xFF, 0xFF, 0x00)

        # When s1 = neg_zero:
        # lanes 0, 1, 2 have s1 = 0 -> masked out (0 contribution each)
        # lane 3 has s1 = 0x80 -> |0x00 - 0x80| = 128
        res = oracle_v_msad_u8(s0, neg_zero, 0)
        self.assertEqual(res, 128)

    def test_subnormal_float_bits_no_flush_to_zero(self) -> None:
        # Smallest positive subnormal: 0x00000001 (byte 0 is 1, bytes 1..3 are 0)
        # Subnormal float 3: 0x00000003 (byte 0 is 3, bytes 1..3 are 0)
        # If byte 0 were flushed to zero, lane 0 would be masked.
        # But in integer VALU, byte 0 is 1 and 3, so lane 0 is NOT masked!
        sub1 = 0x0000_0001
        sub3 = 0x0000_0003
        res = oracle_v_msad_u8(sub3, sub1, 0)
        # |3 - 1| = 2 for lane 0; lanes 1..3 are masked (s1 byte == 0)
        self.assertEqual(res, 2)

    def test_nan_payload_treated_as_raw_byte_data(self) -> None:
        nan_payload = 0x7FA5_5A3C
        # Bytes: b0=0x3C, b1=0x5A, b2=0xA5, b3=0x7F (all non-zero)
        res = oracle_v_msad_u8(0, nan_payload, 0)
        # |0 - b| = b for each byte:
        expected = 0x3C + 0x5A + 0xA5 + 0x7F
        self.assertEqual(res, expected)

    def test_extreme_byte_boundary_combinations(self) -> None:
        extreme_bytes = [0x00, 0x01, 0x02, 0x7E, 0x7F, 0x80, 0x81, 0xFE, 0xFF]
        for b0 in extreme_bytes:
            for b1 in extreme_bytes:
                val = ref_msad_byte(b0, b1)
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 255)
                if b1 == 0:
                    self.assertEqual(val, 0)
                else:
                    self.assertEqual(val, abs(b0 - b1))

    def test_accumulator_boundary_extremes(self) -> None:
        s0 = pack_bytes_u32(0, 0, 0, 0)
        s1 = pack_bytes_u32(1, 1, 1, 1)  # diff = 4
        boundaries = [0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFB, 0xFFFF_FFFF]
        for acc in boundaries:
            with self.subTest(acc=hex(acc)):
                res = oracle_v_msad_u8(s0, s1, acc)
                self.assertEqual(res, (acc + 4) & MASK32)


class TestValuMsadU8NegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(999, RegType.SGPR)

    def test_syntax_rejection_valu_destination_must_be_vgpr(self) -> None:
        dst_sgpr = Reg32(0, RegType.SGPR)
        src = VMsadOperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VMsadU8Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)

    def test_syntax_rejection_literal_as_destination(self) -> None:
        dst_lit = Literal32(0)  # type: ignore[arg-type]
        src = VMsadOperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(AttributeError):
            VMsadU8Instruction(dst=dst_lit, src0=src, src1=src, src2=src)  # type: ignore

    def test_syntax_rejection_clamp_modifier(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src = VMsadOperand(Reg32(1, RegType.VGPR))
        with self.assertRaises(ValueError):
            VMsadU8Instruction(dst=dst, src0=src, src1=src, src2=src, clamp=True)

    def test_syntax_rejection_invalid_simd_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_bit_flip_detection(self) -> None:
        s0 = 0x10203040
        s1 = 0x50607080
        s2 = 100
        correct = oracle_v_msad_u8(s0, s1, s2)

        # Corrupt single bits across different positions
        for bit_pos in (0, 7, 8, 15, 16, 23, 24, 31):
            corrupted = correct ^ (1 << bit_pos)
            self.assertNotEqual(correct, corrupted)

    def test_mutation_unmasked_sad_detected(self) -> None:
        """Mutant: Omits mask check and calculates SAD for all lanes regardless of s1."""
        def mutant_unmasked_msad(s0: int, s1: int, s2: int) -> int:
            b0_lanes = unpack_bytes_u32(s0)
            b1_lanes = unpack_bytes_u32(s1)
            # Bug: fails to check b1 == 0
            diff = sum(abs(b0 - b1) for b0, b1 in zip(b0_lanes, b1_lanes))
            return (s2 + diff) & MASK32

        s0 = pack_bytes_u32(10, 20, 30, 40)
        s1 = pack_bytes_u32(0, 0, 0, 0)  # All masked in correct oracle
        s2 = 0

        oracle_res = oracle_v_msad_u8(s0, s1, s2)
        mutant_res = mutant_unmasked_msad(s0, s1, s2)

        self.assertEqual(oracle_res, 0)
        self.assertEqual(mutant_res, 100)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_reverse_mask_operand_detected(self) -> None:
        """Mutant: Checks s0 == 0 instead of s1 == 0 for masking."""
        def mutant_reverse_mask(s0: int, s1: int, s2: int) -> int:
            b0_lanes = unpack_bytes_u32(s0)
            b1_lanes = unpack_bytes_u32(s1)
            # Bug: checks b0 instead of b1
            diff = sum(0 if b0 == 0 else abs(b0 - b1) for b0, b1 in zip(b0_lanes, b1_lanes))
            return (s2 + diff) & MASK32

        s0 = pack_bytes_u32(0, 0, 0, 0)
        s1 = pack_bytes_u32(5, 5, 5, 5)
        s2 = 0

        oracle_res = oracle_v_msad_u8(s0, s1, s2)
        mutant_res = mutant_reverse_mask(s0, s1, s2)

        # In correct oracle, s1 != 0, so diff is 4 * 5 = 20
        self.assertEqual(oracle_res, 20)
        # In mutant, s0 == 0 masks everything -> 0
        self.assertEqual(mutant_res, 0)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_signed_byte_interpretation_detected(self) -> None:
        """Mutant: Casts bytes to signed int8 (-128..127) before difference."""
        def mutant_signed_msad_byte(b0: int, b1: int) -> int:
            if b1 == 0:
                return 0
            sb0 = (b0 & 0xFF) if (b0 & 0x80) == 0 else (b0 & 0xFF) - 256
            sb1 = (b1 & 0xFF) if (b1 & 0x80) == 0 else (b1 & 0xFF) - 256
            return abs(sb0 - sb1)

        b0 = 0xFF  # 255 unsigned, -1 signed
        b1 = 0x01  # 1

        oracle_diff = ref_msad_byte(b0, b1)
        mutant_diff = mutant_signed_msad_byte(b0, b1)

        # Unsigned diff: |255 - 1| = 254
        self.assertEqual(oracle_diff, 254)
        # Signed diff: |-1 - 1| = 2
        self.assertEqual(mutant_diff, 2)
        self.assertNotEqual(oracle_diff, mutant_diff)

    def test_mutation_ignore_simd_exec_mask_detected(self) -> None:
        """Mutant: SIMD interpreter writes to inactive lanes."""
        sentinel = 0xCAFEBABE
        state = SIMDState(num_lanes=32, exec_mask=0x00000000)  # all inactive
        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)

        class BrokenSIMDInterpreter(SIMDInterpreter):
            def step(self, instr: VMsadU8Instruction, state: SIMDState) -> List[Optional[int]]:
                results = []
                for lane in range(state.num_lanes):
                    # Bug: no exec_mask check
                    res = oracle_v_msad_u8(0, 0, 0)
                    state.set_vgpr(lane, instr.dst.index, res)
                    results.append(res)
                return results

        broken = BrokenSIMDInterpreter()
        instr = VMsadU8Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadOperand(Literal32(0)),
            src1=VMsadOperand(Literal32(0)),
            src2=VMsadOperand(Literal32(0)),
        )
        broken.step(instr, state)

        # All 32 lanes were corrupted
        corrupted = [lane for lane in range(32) if state.get_vgpr(lane, 0) != sentinel]
        self.assertEqual(len(corrupted), 32)

    def test_mutation_diff_sum_8bit_truncation_detected(self) -> None:
        """Mutant: Truncates intermediate SAD sum to 8 bits before accumulation."""
        def mutant_truncated_msad(s0: int, s1: int, s2: int) -> int:
            b0_lanes = unpack_bytes_u32(s0)
            b1_lanes = unpack_bytes_u32(s1)
            diff = sum(0 if b1 == 0 else abs(b0 - b1) for b0, b1 in zip(b0_lanes, b1_lanes))
            # Bug: truncates diff to 8 bits
            diff_8 = diff & 0xFF
            return (s2 + diff_8) & MASK32

        s0 = pack_bytes_u32(0, 0, 0, 0)
        s1 = pack_bytes_u32(255, 255, 255, 255)  # diff = 4 * 255 = 1020

        oracle_res = oracle_v_msad_u8(s0, s1, 0)
        mutant_res = mutant_truncated_msad(s0, s1, 0)

        # 1020 vs (1020 & 0xFF = 252)
        self.assertEqual(oracle_res, 1020)
        self.assertEqual(mutant_res, 252)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_cross_byte_bleed_detected(self) -> None:
        """Mutant: Performs whole 32-bit subtraction instead of per-byte SAD."""
        def mutant_whole_word_sad(s0: int, s1: int, s2: int) -> int:
            # Bug: whole word diff instead of per-byte isolation
            diff = abs((s0 & MASK32) - (s1 & MASK32))
            return (s2 + diff) & MASK32

        s0 = 0x0000_0100  # byte 0 = 0, byte 1 = 1
        s1 = 0x0000_0001  # byte 0 = 1, byte 1 = 0
        s2 = 0

        # Oracle: byte 0 |0 - 1| = 1; byte 1 is masked (s1 byte 1 == 0) -> diff = 1
        oracle_res = oracle_v_msad_u8(s0, s1, s2)
        mutant_res = mutant_whole_word_sad(s0, s1, s2)

        self.assertEqual(oracle_res, 1)
        # Mutant: |0x100 - 0x1| = 0xFF = 255
        self.assertEqual(mutant_res, 255)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_shift_byte_sum_hi_detected(self) -> None:
        """Mutant: Shifts byte sum left by 16 bits (erroneously mimicking v_sad_hi_u8)."""
        def mutant_sad_hi_msad(s0: int, s1: int, s2: int) -> int:
            b0_lanes = unpack_bytes_u32(s0)
            b1_lanes = unpack_bytes_u32(s1)
            diff = sum(0 if b1 == 0 else abs(b0 - b1) for b0, b1 in zip(b0_lanes, b1_lanes))
            # Bug: shifts left by 16
            return (s2 + (diff << 16)) & MASK32

        s0 = pack_bytes_u32(1, 1, 1, 1)
        s1 = pack_bytes_u32(2, 2, 2, 2)  # diff = 4
        s2 = 0

        oracle_res = oracle_v_msad_u8(s0, s1, s2)
        mutant_res = mutant_sad_hi_msad(s0, s1, s2)

        self.assertEqual(oracle_res, 4)
        self.assertEqual(mutant_res, 4 << 16)
        self.assertNotEqual(oracle_res, mutant_res)


if __name__ == "__main__":
    unittest.main()
