from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_msad_u32.

AMD RDNA VALU Instruction Set Architecture:
  v_msad_u32 dst, src0, src1, src2

Operation:
  Performs 32-bit masked sum of absolute difference with 32-bit unsigned accumulation:
    diff = 0 if src1 == 0 else |src0.u32 - src1.u32|
    dst = (src2.u32 + diff) mod 2^32

Mathematical and Architectural Properties:
  - Masking semantics:
    - Operation is masked out (omitted) if and only if reference word src1 == 0.
    - If src1 == 0, diff is strictly 0 regardless of src0 (even if src0 == 0xFFFFFFFF).
    - If src1 != 0, diff is the unsigned absolute difference |src0 - src1| in [0, 2^32 - 1].
  - Asymmetry / Non-commutativity:
    - Unlike unmasked v_sad_u32, v_msad_u32 is NOT commutative with respect to src0 and src1:
      the mask is dictated solely by src1.
  - Range and bounds:
    - Unmasked difference is in [0, 2^32 - 1] (0xFFFFFFFF).
    - Minimum diff is 0 (masked or equal operands).
  - Accumulator wrapping:
    - src2 is an unsigned 32-bit integer.
    - The addition (src2 + diff) wraps modulo 2^32 with standard unsigned arithmetic.
  - Clamp modifier:
    - In AMD RDNA2 ISA / Leonida translator contract, clamp modifier is unsupported
      and rejected rather than guessed.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MAX_DIFF_U32: int = MASK32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def ref_abs_diff_u32(s0: int, s1: int) -> int:
    """Compute strictly unsigned 32-bit absolute difference |s0 - s1|."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def oracle_v_msad_u32(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_msad_u32.

    Computes masked absolute difference conditioned on src1 non-zero,
    and adds accumulator src2. Result wraps modulo 2^32.
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    acc = src2 & MASK32

    if u1 == 0:
        diff = 0
    else:
        diff = (u0 - u1) if u0 >= u1 else (u1 - u0)

    return (acc + diff) & MASK32


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
class VMsadU32Operand:
    """Typed expression operand representing a register or literal."""
    source: Union[Reg32, Literal32]

    def disassemble(self) -> str:
        return self.source.name


@dataclass(frozen=True)
class VMsadU32Instruction:
    """Symbolic AST node representing an RDNA2 v_msad_u32 VALU instruction."""
    dst: Reg32
    src0: VMsadU32Operand
    src1: VMsadU32Operand
    src2: VMsadU32Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.clamp:
            raise ValueError("clamp modifier is unsupported on v_msad_u32")

    def disassemble(self) -> str:
        return f"v_msad_u32 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


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
    """Deterministic scalar instruction interpreter for v_msad_u32."""

    def resolve_operand(self, op: VMsadU32Operand, state: ScalarMachineState) -> int:
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

    def step(self, instr: VMsadU32Instruction, state: ScalarMachineState) -> int:
        # Atomic operand read phase
        s0 = self.resolve_operand(instr.src0, state)
        s1 = self.resolve_operand(instr.src1, state)
        s2 = self.resolve_operand(instr.src2, state)

        # Computation phase via reference oracle
        result = oracle_v_msad_u32(s0, s1, s2)

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

    def resolve_lane_operand(self, op: VMsadU32Operand, state: SIMDState, lane: int) -> int:
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

    def step(self, instr: VMsadU32Instruction, state: SIMDState) -> List[Optional[int]]:
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
                res = oracle_v_msad_u32(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class VMsadU32StatusFlags:
    """Architectural condition codes, status flags, and carry generation for v_msad_u32."""
    zero: bool
    sign: bool
    carry_out: bool
    masked: bool
    diff: int


def evaluate_v_msad_u32_flags(src0: int, src1: int, src2: int, result: int) -> VMsadU32StatusFlags:
    """Evaluate architectural condition codes, masking status, and 32-bit carry generation."""
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    acc = src2 & MASK32

    masked = (u1 == 0)
    if masked:
        diff = 0
    else:
        diff = (u0 - u1) if u0 >= u1 else (u1 - u0)

    full_sum = acc + diff
    carry_out = full_sum > MASK32
    res32 = result & MASK32

    zero = (res32 == 0)
    sign = bool(res32 & 0x8000_0000)

    return VMsadU32StatusFlags(
        zero=zero,
        sign=sign,
        carry_out=carry_out,
        masked=masked,
        diff=diff,
    )




# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMsadU32ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_msad_u32 architectural bit model and reference oracle."""

    def test_masked_when_s1_is_zero(self) -> None:
        # When src1 is 0, masked out -> diff = 0 regardless of src0
        for s0 in [0x00000000, 0xFFFFFFFF, 0x12345678, 0x80000000]:
            with self.subTest(s0=hex(s0)):
                self.assertEqual(oracle_v_msad_u32(s0, 0, 0), 0)
                self.assertEqual(oracle_v_msad_u32(s0, 0, 500), 500)
                self.assertEqual(oracle_v_msad_u32(s0, 0, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_unmasked_known_values(self) -> None:
        # s0 = 1000, s1 = 1500, s2 = 100
        # diff = |1000 - 1500| = 500
        # result = 100 + 500 = 600
        self.assertEqual(oracle_v_msad_u32(1000, 1500, 100), 600)
        self.assertEqual(oracle_v_msad_u32(2000, 1000, 0), 1000)

    def test_masking_asymmetry_non_commutativity(self) -> None:
        # s0 has 1000, s1 has 0 -> masked (result = 0)
        # s0 has 0, s1 has 1000 -> active (|0 - 1000| = 1000)
        res_s0_s1 = oracle_v_msad_u32(1000, 0, 0)
        res_s1_s0 = oracle_v_msad_u32(0, 1000, 0)

        self.assertEqual(res_s0_s1, 0)
        self.assertEqual(res_s1_s0, 1000)
        self.assertNotEqual(res_s0_s1, res_s1_s0)

    def test_maximum_delta_accumulation(self) -> None:
        # Max delta: |0 - 0xFFFFFFFF| = 0xFFFFFFFF with s1 = 0xFFFFFFFF!
        sad_max = oracle_v_msad_u32(0, 0xFFFFFFFF, 0)
        self.assertEqual(sad_max, MAX_DIFF_U32)
        self.assertEqual(sad_max, 0xFFFFFFFF)

    def test_accumulator_32bit_wraparound(self) -> None:
        # diff = 0xFFFFFFFF, acc = 1 -> wraps to 0
        self.assertEqual(oracle_v_msad_u32(0, 0xFFFFFFFF, 1), 0)
        # diff = 0xFFFFFFFF, acc = 2 -> wraps to 1
        self.assertEqual(oracle_v_msad_u32(0, 0xFFFFFFFF, 2), 1)

    def test_identity_when_equal_and_nonzero(self) -> None:
        # When src0 == src1 and src1 != 0, diff = 0, so result = src2
        val = 0x12345678
        self.assertEqual(oracle_v_msad_u32(val, val, 0xABCDEF01), 0xABCDEF01)
        self.assertEqual(oracle_v_msad_u32(0xFFFFFFFF, 0xFFFFFFFF, 42), 42)


class TestValuMsadU32SymbolicAST(unittest.TestCase):
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
        lit_small = Literal32(7)
        lit_hex = Literal32(0x10203040)
        self.assertEqual(lit_small.name, "7")
        self.assertEqual(lit_hex.name, "0x10203040")

        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_instruction_ast_construction_and_disassembly(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src0 = VMsadU32Operand(Reg32(1, RegType.VGPR))
        src1 = VMsadU32Operand(Reg32(4, RegType.SGPR))
        src2 = VMsadU32Operand(Literal32(500))

        instr = VMsadU32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(instr.disassemble(), "v_msad_u32 v0, v1, s4, 0x000001f4")

    def test_destination_must_be_vgpr(self) -> None:
        dst_sgpr = Reg32(0, RegType.SGPR)
        src = VMsadU32Operand(Reg32(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VMsadU32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)

    def test_clamp_modifier_rejection_in_ast(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src = VMsadU32Operand(Reg32(1, RegType.VGPR))
        with self.assertRaises(ValueError):
            VMsadU32Instruction(dst=dst, src0=src, src1=src, src2=src, clamp=True)


class TestValuMsadU32ScalarInterpreter(unittest.TestCase):
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
        # s0 = 1000, s1 = 0, s2 = 500
        # masked because s1 == 0 -> diff = 0 -> result = 500
        self.state.set_vgpr(1, 1000)
        self.state.set_vgpr(2, 0)
        self.state.set_vgpr(3, 500)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(3, RegType.VGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 500)
        self.assertEqual(self.state.get_vgpr(0), 500)

    def test_scalar_step_sgpr_and_literal_sources(self) -> None:
        self.state.set_vgpr(1, 1000)
        self.state.set_sgpr(4, 2500)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(4, RegType.SGPR)),
            src2=VMsadU32Operand(Literal32(100)),
        )

        res = self.interp.step(instr, self.state)
        # 100 + |1000 - 2500| = 1600
        self.assertEqual(res, 1600)
        self.assertEqual(self.state.get_vgpr(0), 1600)

    def test_sequential_instruction_flow(self) -> None:
        # Step 1: v0 = v_msad_u32(v1, v2, 0)
        # Step 2: v3 = v_msad_u32(v1, v2, v0)  (accumulating with previous result)
        self.state.set_vgpr(1, 100)
        self.state.set_vgpr(2, 250)  # diff = 150

        instr1 = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Literal32(0)),
        )
        res1 = self.interp.step(instr1, self.state)
        self.assertEqual(res1, 150)
        self.assertEqual(self.state.get_vgpr(0), 150)

        instr2 = VMsadU32Instruction(
            dst=Reg32(3, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(0, RegType.VGPR)),
        )
        res2 = self.interp.step(instr2, self.state)
        # 150 + 150 = 300
        self.assertEqual(res2, 300)
        self.assertEqual(self.state.get_vgpr(3), 300)


class TestValuMsadU32SIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        for lane in range(32):
            s0_val = lane * 100
            s1_val = 500
            s2_val = 1000 + lane
            state.set_vgpr(lane, 1, s0_val)
            state.set_vgpr(lane, 2, s1_val)
            state.set_vgpr(lane, 3, s2_val)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(3, RegType.VGPR)),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = oracle_v_msad_u32(
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
            state.set_vgpr(lane, 1, lane * 50)
            state.set_vgpr(lane, 2, 2000)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Literal32(10)),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = oracle_v_msad_u32(
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
            state.set_vgpr(lane, 1, 1000)
            state.set_vgpr(lane, 2, 3000)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Literal32(0)),
        )

        results = interp.step(instr, state)
        expected = oracle_v_msad_u32(1000, 3000, 0)
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

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Literal32(0xFFFFFFFF)),
            src1=VMsadU32Operand(Literal32(0xFFFFFFFF)),
            src2=VMsadU32Operand(Literal32(100)),
        )

        results = interp.step(instr, state)
        self.assertEqual(results, [None] * 32)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_sgpr_scalar_broadcast_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        state.set_sgpr(5, 5000)  # broadcast to all lanes
        for lane in range(32):
            state.set_vgpr(lane, 1, lane * 200)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(5, RegType.SGPR)),
            src2=VMsadU32Operand(Literal32(0)),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            expected = oracle_v_msad_u32(lane * 200, 5000, 0)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_divergent_masking_pattern_across_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_vgpr(lane, 1, 0xFFFFFFFF)
            if lane % 2 == 0:
                state.set_vgpr(lane, 2, 0xFFFFFFFF)  # no masking: diff = 0
            else:
                state.set_vgpr(lane, 2, 0x00000000)  # masked: diff = 0

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Literal32(77)),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(results[lane], 77)
            self.assertEqual(state.get_vgpr(lane, 0), 77)


class TestValuMsadU32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x1000_2000
        val_b = 0x3000_4000
        acc_val = 5000

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, acc_val)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(0, RegType.VGPR)),  # Aliased with dst
            src1=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(2, RegType.VGPR)),
        )

        expected = oracle_v_msad_u32(val_a, val_b, acc_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(0), expected)
        # Verify src1 and src2 remain intact
        self.assertEqual(state.get_vgpr(1), val_b)
        self.assertEqual(state.get_vgpr(2), acc_val)

    def test_dst_aliases_src1(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x1000_2000
        val_b = 0x0000_0000  # masked
        acc_val = 1500

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, acc_val)

        instr = VMsadU32Instruction(
            dst=Reg32(1, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(0, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(1, RegType.VGPR)),  # Aliased with dst
            src2=VMsadU32Operand(Reg32(2, RegType.VGPR)),
        )

        expected = oracle_v_msad_u32(val_a, val_b, acc_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(1), expected)
        self.assertEqual(state.get_vgpr(0), val_a)
        self.assertEqual(state.get_vgpr(2), acc_val)

    def test_dst_aliases_src2(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val_a = 0x1000_2000
        val_b = 0x3000_4000
        acc_val = 2000

        state.set_vgpr(0, val_a)
        state.set_vgpr(1, val_b)
        state.set_vgpr(2, acc_val)

        instr = VMsadU32Instruction(
            dst=Reg32(2, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(0, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(2, RegType.VGPR)),  # Aliased with dst
        )

        expected = oracle_v_msad_u32(val_a, val_b, acc_val)
        res = interp.step(instr, state)
        self.assertEqual(res, expected)
        self.assertEqual(state.get_vgpr(2), expected)
        self.assertEqual(state.get_vgpr(0), val_a)
        self.assertEqual(state.get_vgpr(1), val_b)

    def test_dst_aliases_src0_and_src1_identity_zero_diff(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val = 0x5555_0000
        acc = 9999
        state.set_vgpr(0, val)
        state.set_vgpr(1, acc)

        # When src0 == src1, diff is 0, so result is accumulator
        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(0, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(0, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(1, RegType.VGPR)),
        )

        res = interp.step(instr, state)
        self.assertEqual(res, acc)
        self.assertEqual(state.get_vgpr(0), acc)

    def test_dst_aliases_all_operands(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        val = 0x7E80_0000
        state.set_vgpr(0, val)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(0, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(0, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(0, RegType.VGPR)),
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
            state.set_vgpr(lane, 1, 0x20002000)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(0, RegType.VGPR)),  # Aliased
            src1=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src2=VMsadU32Operand(Literal32(0)),
        )

        interp.step(instr, state)

        for lane in range(32):
            if lane < 16:
                expected = oracle_v_msad_u32(original_v0[lane], 0x20002000, 0)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), original_v0[lane])

    def test_chained_in_place_accumulation(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, 0)
        state.set_vgpr(1, 1000)
        state.set_vgpr(2, 1100)  # diff = 100

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(0, RegType.VGPR)),  # accum from v0
        )

        # Run 5 iterations: v0 should accumulate 100 each time
        for i in range(1, 6):
            res = interp.step(instr, state)
            self.assertEqual(res, 100 * i)
            self.assertEqual(state.get_vgpr(0), 100 * i)


class TestValuMsadU32ConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_zero_flag_when_result_is_zero(self) -> None:
        # Case 1: 0 + 0 = 0
        res = oracle_v_msad_u32(0, 0, 0)
        flags = evaluate_v_msad_u32_flags(0, 0, 0, res)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry_out)
        self.assertTrue(flags.masked)
        self.assertEqual(flags.diff, 0)

        # Case 2: exact 32-bit overflow wrapping to 0:
        # acc = 1, diff = 0xFFFFFFFF -> full sum = 0x1_0000_0000 -> res = 0
        res_wrap = oracle_v_msad_u32(0, 0xFFFFFFFF, 1)
        self.assertEqual(res_wrap, 0)
        flags_wrap = evaluate_v_msad_u32_flags(0, 0xFFFFFFFF, 1, res_wrap)
        self.assertTrue(flags_wrap.zero)
        self.assertFalse(flags_wrap.sign)
        self.assertTrue(flags_wrap.carry_out)  # Overflowed!
        self.assertEqual(flags_wrap.diff, 0xFFFFFFFF)

    def test_sign_flag_high_bit_set(self) -> None:
        acc_neg = 0x8000_0000
        res = oracle_v_msad_u32(0, 0, acc_neg)
        flags = evaluate_v_msad_u32_flags(0, 0, acc_neg, res)
        self.assertTrue(flags.sign)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.carry_out)

        # acc just below bit 31 (0x7FFF_FFFF) + diff 1 -> 0x8000_0000
        res_trans = oracle_v_msad_u32(1, 2, 0x7FFF_FFFF)
        self.assertEqual(res_trans, 0x8000_0000)
        flags_trans = evaluate_v_msad_u32_flags(1, 2, 0x7FFF_FFFF, res_trans)
        self.assertTrue(flags_trans.sign)

    def test_carry_out_32bit_overflow(self) -> None:
        # 0xFFFFFFFF + diff 1 -> carry_out True
        res = oracle_v_msad_u32(1, 2, 0xFFFFFFFF)
        self.assertEqual(res, 0)
        flags = evaluate_v_msad_u32_flags(1, 2, 0xFFFFFFFF, res)
        self.assertTrue(flags.carry_out)
        self.assertTrue(flags.zero)

    def test_carry_out_no_overflow(self) -> None:
        # 0x00001000 + diff 100 -> no overflow
        res = oracle_v_msad_u32(0, 0, 0x1000)
        flags = evaluate_v_msad_u32_flags(0, 0, 0x1000, res)
        self.assertFalse(flags.carry_out)

    def test_masking_detection(self) -> None:
        # Unmasked: src1 != 0
        flags_unmasked = evaluate_v_msad_u32_flags(100, 200, 0, 100)
        self.assertFalse(flags_unmasked.masked)
        self.assertEqual(flags_unmasked.diff, 100)

        # Masked: src1 == 0
        flags_masked = evaluate_v_msad_u32_flags(100, 0, 0, 0)
        self.assertTrue(flags_masked.masked)
        self.assertEqual(flags_masked.diff, 0)


class TestValuMsadU32IEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 float bit pattern handling, 32-bit extremes, and boundary limits."""

    def test_ieee754_special_float_bitpatterns_treated_as_raw_dwords(self) -> None:
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
                    res0 = oracle_v_msad_u32(f0, f1, 0)
                    res1 = oracle_v_msad_u32(f0, f1, 100)

                    expected_diff = 0 if f1 == 0 else (f0 - f1 if f0 >= f1 else f1 - f0)
                    self.assertEqual(res0, expected_diff)
                    self.assertEqual(res1, (100 + expected_diff) & MASK32)

    def test_subnormal_float_bits_no_flush_to_zero(self) -> None:
        sub1 = 0x0000_0001
        sub3 = 0x0000_0003
        res = oracle_v_msad_u32(sub3, sub1, 0)
        self.assertEqual(res, 2)

    def test_nan_payload_treated_as_raw_dword_data(self) -> None:
        nan_payload = 0x7FA5_5A3C
        res = oracle_v_msad_u32(0, nan_payload, 0)
        self.assertEqual(res, nan_payload)

    def test_extreme_dword_boundary_combinations(self) -> None:
        extremes = [0x0000_0000, 0x0000_0001, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFFF]
        for s0 in extremes:
            for s1 in extremes:
                val = oracle_v_msad_u32(s0, s1, 0)
                if s1 == 0:
                    self.assertEqual(val, 0)
                else:
                    self.assertEqual(val, ref_abs_diff_u32(s0, s1))

    def test_accumulator_boundary_extremes(self) -> None:
        s0 = 0
        s1 = 1  # diff = 1
        boundaries = [0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFFF]
        for acc in boundaries:
            with self.subTest(acc=hex(acc)):
                res = oracle_v_msad_u32(s0, s1, acc)
                self.assertEqual(res, (acc + 1) & MASK32)


if __name__ == "__main__":
    unittest.main()
