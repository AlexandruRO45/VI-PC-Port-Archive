from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_lshl_add_u32.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_lshl_add_u32 D, S0, S1, S2
Operation:
    Performs a 32-bit logical shift left of S0 by the shift count specified in S1,
    then adds S2 to the result modulo 2^32:
        dst = ((S0 << (S1 & 0x1F)) + S2) & 0xFFFFFFFF

Modifiers and Semantics:
    - S0: 32-bit unsigned integer base operand to shift.
    - S1: 32-bit unsigned integer shift count; only the low 5 bits [4:0] are used (modulo 32).
    - S2: 32-bit unsigned integer addend added to the shifted result.
    - Full 32-bit modulo wrapping arithmetic on addition without saturation.
    - Bits shifted left past bit 31 are discarded (zero-fill on the right).

Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs).
    The destination is always a 32-bit vector register (VGPR).
    Unlike carry-generating instructions, v_lshl_add_u32 does not write to VCC.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
SHIFT_MASK: int = 0x1F


def oracle_v_lshl_add_u32(src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_lshl_add_u32."""
    s0 = src0_bits & MASK32
    shift = src1_bits & SHIFT_MASK
    s2 = src2_bits & MASK32

    shifted = (s0 << shift) & MASK32
    return (shifted + s2) & MASK32


class TestValuLshlAddU32ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_lshl_add_u32."""

    def test_zero_shift_identity(self) -> None:
        # (s0 << 0) + s2 == s0 + s2
        res = oracle_v_lshl_add_u32(100, 0, 50)
        self.assertEqual(res, 150)

    def test_zero_add_identity(self) -> None:
        # (s0 << s1) + 0 == s0 << s1
        res = oracle_v_lshl_add_u32(0x00000001, 16, 0)
        self.assertEqual(res, 0x00010000)

    def test_shift_count_modulo_32(self) -> None:
        # S1 values 32, 64, 96 mask to shift count 0
        self.assertEqual(oracle_v_lshl_add_u32(0x1234, 32, 0), 0x1234)
        self.assertEqual(oracle_v_lshl_add_u32(0x1234, 64, 0), 0x1234)
        # S1 value 33 masks to shift count 1
        self.assertEqual(oracle_v_lshl_add_u32(0x0001, 33, 0), 0x0002)
        # S1 value 0xFFFFFFFF masks to 31
        self.assertEqual(oracle_v_lshl_add_u32(0x0001, 0xFFFFFFFF, 0), 0x80000000)

    def test_modulo_32_addition_overflow(self) -> None:
        # Addition wraps modulo 2^32
        res = oracle_v_lshl_add_u32(0x80000000, 0, 0x80000000)
        self.assertEqual(res, 0)

        # Shifted value + addend overflows 32 bits
        res2 = oracle_v_lshl_add_u32(0x40000000, 1, 0x00000001)
        self.assertEqual(res2, 0x80000001)

    def test_high_bits_discarded_on_shift(self) -> None:
        # Shifting 0x80000000 left by 1 results in 0 (overflow discarded)
        self.assertEqual(oracle_v_lshl_add_u32(0x80000000, 1, 0x00000005), 0x00000005)

    def test_combined_shift_and_add(self) -> None:
        # (0x00000003 << 4) + 0x0000000F = 0x00000030 + 0x0000000F = 0x0000003F
        res = oracle_v_lshl_add_u32(3, 4, 15)
        self.assertEqual(res, 0x3F)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit literal integer constant."""
    bits: int

    @property
    def name(self) -> str:
        return f"0x{self.bits & MASK32:08X}"


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand (Register or Literal32)."""
    source: Union[Reg, Literal32]

    def disassemble(self) -> str:
        return self.source.name


class Opcode(Enum):
    V_LSHL_ADD_U32 = "v_lshl_add_u32"


@dataclass(frozen=True)
class VLshlAddU32Instruction:
    """Symbolic AST node representing a RDNA2 v_lshl_add_u32 instruction."""
    dst: Reg
    src0: VOP3Operand
    src1: VOP3Operand
    src2: VOP3Operand
    opcode: Opcode = Opcode.V_LSHL_ADD_U32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        return (
            f"{self.opcode.value} {self.dst.name}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"
        )


class TestValuLshlAddU32SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_creation_and_disassembly(self) -> None:
        v0 = Reg(0, RegType.VGPR)
        s5 = Reg(5, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(s5.name, "s5")

    def test_register_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Reg(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg(256, RegType.VGPR)

    def test_literal_operand(self) -> None:
        lit = Literal32(0x12345678)
        self.assertEqual(lit.name, "0x12345678")

    def test_instruction_disassembly(self) -> None:
        dst = Reg(0, RegType.VGPR)
        src0 = VOP3Operand(Reg(1, RegType.VGPR))
        src1 = VOP3Operand(Literal32(3))
        src2 = VOP3Operand(Reg(2, RegType.SGPR))

        inst = VLshlAddU32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(inst.disassemble(), "v_lshl_add_u32 v0, v1, 0x00000003, s2")

    def test_instruction_validation(self) -> None:
        dst_sgpr = Reg(0, RegType.SGPR)
        src = VOP3Operand(Reg(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VLshlAddU32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)
    vcc: int = 0

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for v_lshl_add_u32."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VLshlAddU32Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s0_raw = self.resolve_operand(instr.src0, state)
        s1_raw = self.resolve_operand(instr.src1, state)
        s2_raw = self.resolve_operand(instr.src2, state)

        # Computation phase
        result = oracle_v_lshl_add_u32(s0_raw, s1_raw, s2_raw)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


class TestValuLshlAddU32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_basic_step_vgprs(self) -> None:
        self.state.set_vgpr(1, 0x00000005)
        self.state.set_vgpr(2, 3)
        self.state.set_vgpr(3, 0x00000010)
        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        # (5 << 3) + 16 = 40 + 16 = 56 (0x38)
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 56)
        self.assertEqual(self.state.get_vgpr(0), 56)

    def test_step_sgpr_and_literal(self) -> None:
        self.state.set_sgpr(4, 0x00000100)
        self.state.set_vgpr(1, 0x00000020)
        instr = VLshlAddU32Instruction(
            dst=Reg(5, RegType.VGPR),
            src0=VOP3Operand(Reg(4, RegType.SGPR)),
            src1=VOP3Operand(Literal32(2)),
            src2=VOP3Operand(Reg(1, RegType.VGPR)),
        )
        # (0x100 << 2) + 0x20 = 0x400 + 0x20 = 0x420
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 0x420)
        self.assertEqual(self.state.get_vgpr(5), 0x420)

    def test_deterministic_repeatability(self) -> None:
        self.state.set_vgpr(10, 0x12345678)
        self.state.set_vgpr(11, 4)
        self.state.set_vgpr(12, 0x0000ABCD)
        instr = VLshlAddU32Instruction(
            dst=Reg(13, RegType.VGPR),
            src0=VOP3Operand(Reg(10, RegType.VGPR)),
            src1=VOP3Operand(Reg(11, RegType.VGPR)),
            src2=VOP3Operand(Reg(12, RegType.VGPR)),
        )
        res0 = self.interp.step(instr, self.state)
        for _ in range(50):
            res_n = self.interp.step(instr, self.state)
            self.assertEqual(res0, res_n)

    def test_unaffected_registers_preserved(self) -> None:
        self.state.set_vgpr(1, 10)
        self.state.set_vgpr(2, 1)
        self.state.set_vgpr(3, 5)
        self.state.set_vgpr(7, 0xDEADBEEF)
        self.state.set_sgpr(9, 0xCAFEBABE)
        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        self.interp.step(instr, self.state)
        self.assertEqual(self.state.get_vgpr(7), 0xDEADBEEF)
        self.assertEqual(self.state.get_sgpr(9), 0xCAFEBABE)


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
    vcc: int = 0

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index] & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, Reg):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.bits & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VLshlAddU32Instruction, state: SIMDState) -> List[Optional[int]]:
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

        # Execution phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_lshl_add_u32(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.set_vgpr(lane, instr.dst.index, res)
                results[lane] = res

        return results


class TestValuLshlAddU32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD: int = 0xDEADBEEF

    def setUp(self) -> None:
        self.simd_interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr(lane, 1, lane)
            state.set_vgpr(lane, 2, 2)
            state.set_vgpr(lane, 3, 100)

        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(32):
            expected = ((lane << 2) + 100) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr(lane, 1, 1)
            state.set_vgpr(lane, 2, lane % 32)
            state.set_vgpr(lane, 3, 0)

        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            expected = (1 << (lane % 32)) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self) -> None:
        mask = 0xAAAA_AAAA_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 10)
            state.set_vgpr(lane, 2, 1)
            state.set_vgpr(lane, 3, 5)

        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            is_active = bool((mask >> lane) & 1)
            if is_active:
                expected = (10 << 1) + 5
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_split_halves_masking(self) -> None:
        mask = 0x0000_0000_FFFF_FFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x10)
            state.set_vgpr(lane, 2, 4)
            state.set_vgpr(lane, 3, 0x20)

        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0x120)
        for lane in range(32, 64):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_single_lane_active(self) -> None:
        target_lane = 19
        mask = 1 << target_lane
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 0x40)
            state.set_vgpr(lane, 2, 2)
            state.set_vgpr(lane, 3, 0x11)

        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(results[lane], 0x111)
                self.assertEqual(state.get_vgpr(lane, 0), 0x111)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_all_lanes_disabled(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, 100)
            state.set_vgpr(lane, 2, 2)
            state.set_vgpr(lane, 3, 50)

        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(3, RegType.VGPR)),
        )
        results = self.simd_interp.step(instr, state)
        for lane in range(64):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)

    def test_shared_sgpr_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000_FFFF)
        state.set_sgpr(5, 0x00001000)
        for lane in range(32):
            state.set_vgpr(lane, 0, self.CANARY_OLD)
            state.set_vgpr(lane, 1, lane)
            state.set_vgpr(lane, 2, 1)

        instr = VLshlAddU32Instruction(
            dst=Reg(0, RegType.VGPR),
            src0=VOP3Operand(Reg(1, RegType.VGPR)),
            src1=VOP3Operand(Reg(2, RegType.VGPR)),
            src2=VOP3Operand(Reg(5, RegType.SGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(16):
            expected = ((lane << 1) + 0x1000) & MASK32
            self.assertEqual(state.get_vgpr(lane, 0), expected)
        for lane in range(16, 32):
            self.assertEqual(state.get_vgpr(lane, 0), self.CANARY_OLD)





if __name__ == "__main__":
    unittest.main()
