from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_dot2_i32_i16.

AMD RDNA VALU Instruction Set Architecture:
  v_dot2_i32_i16 dst, src0, src1, src2 [clamp]

Format:
  VOP3P / VOP3 (3-operand vector ALU instruction with signed 32-bit output).

Operands:
  dst:  32-bit vector register (v[dst]) storing a two's-complement signed 32-bit integer.
  src0: 32-bit register (v[s0] or s[s0]) storing a packed pair of signed 16-bit integers:
        - src0.i16[0]: low 16 bits [15:0]
        - src0.i16[1]: high 16 bits [31:16]
  src1: 32-bit register (v[s1] or s[s1]) storing a packed pair of signed 16-bit integers:
        - src1.i16[0]: low 16 bits [15:0]
        - src1.i16[1]: high 16 bits [31:16]
  src2: 32-bit register (v[s2], s[s2], or literal) storing a signed 32-bit integer accumulator.

Operation:
  Sign-extends each 16-bit halfword to a 32-bit signed integer, computes two products,
  and sums them with the 32-bit signed accumulator:
    prod0   = int32(src0.i16[0]) * int32(src1.i16[0])
    prod1   = int32(src0.i16[1]) * int32(src1.i16[1])
    full    = prod0 + prod1 + int32(src2)

Clamp Modifier:
  When clamp is set, the final result saturates to signed 32-bit bounds [-2^31, 2^31 - 1]:
    if full > 0x7FFFFFFF:       res = 0x7FFFFFFF
    elif full < -0x80000000:    res = 0x80000000
    else:                       res = full & 0xFFFFFFFF
  When unclamped, the full sum wraps modulo 2^32.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF

INT32_MIN: int = -0x8000_0000
INT32_MAX: int = 0x7FFF_FFFF
UINT32_INT_MIN: int = 0x8000_0000
UINT32_INT_MAX: int = 0x7FFF_FFFF


def s16(val: int) -> int:
    """Sign-extend a 16-bit integer to Python signed integer."""
    val &= MASK16
    return val - 0x1_0000 if (val & 0x8000) else val


def s32(val: int) -> int:
    """Sign-extend a 32-bit integer to Python signed integer."""
    val &= MASK32
    return val - 0x1_0000_0000 if (val & 0x8000_0000) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def pack_i16x2(lo: int, hi: int) -> int:
    """Pack two signed 16-bit integers into an unsigned 32-bit dword."""
    return (lo & MASK16) | ((hi & MASK16) << 16)


def unpack_i16x2(packed32: int) -> Tuple[int, int]:
    """Unpack an unsigned 32-bit dword into two signed 16-bit integers (lo, hi)."""
    return s16(packed32 & MASK16), s16((packed32 >> 16) & MASK16)


def ref_dot2_i32_i16(
    s0_packed: int,
    s1_packed: int,
    s2_i32: int,
    *,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_dot2_i32_i16 returning uint32 bits."""
    a0, a1 = unpack_i16x2(s0_packed)
    b0, b1 = unpack_i16x2(s1_packed)
    c = s32(s2_i32)

    full = (a0 * b0) + (a1 * b1) + c

    if clamp:
        if full > INT32_MAX:
            return UINT32_INT_MAX
        if full < INT32_MIN:
            return UINT32_INT_MIN
        return u32(full)
    return u32(full)


class TestValuDot2I32I16ReferenceOracle(unittest.TestCase):
    """Architectural reference tests validating mathematical model for v_dot2_i32_i16."""

    def test_basic_positive_dot_product(self) -> None:
        """(2, 3) . (4, 5) + 6 = 8 + 15 + 6 = 29"""
        s0 = pack_i16x2(2, 3)
        s1 = pack_i16x2(4, 5)
        s2 = 6
        res = ref_dot2_i32_i16(s0, s1, s2)
        self.assertEqual(res, 29)

    def test_signed_negative_elements(self) -> None:
        """(-2, 3) . (4, -5) + 10 = -8 + (-15) + 10 = -13"""
        s0 = pack_i16x2(-2, 3)
        s1 = pack_i16x2(4, -5)
        s2 = 10
        res = ref_dot2_i32_i16(s0, s1, s2)
        self.assertEqual(res, u32(-13))

    def test_zero_operands(self) -> None:
        """(0, 0) . (100, -200) + 12345 = 12345"""
        s0 = pack_i16x2(0, 0)
        s1 = pack_i16x2(100, -200)
        s2 = 12345
        res = ref_dot2_i32_i16(s0, s1, s2)
        self.assertEqual(res, 12345)

    def test_unclamped_modulo_wrapping_overflow(self) -> None:
        """Two maximum products (-32768 * -32768 * 2) = 2^31 = 2147483648 -> wraps to 0x80000000."""
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(-32768, -32768)
        s2 = 0
        # (-32768)^2 + (-32768)^2 = 2^30 + 2^30 = 2^31 = 0x80000000
        res_unclamped = ref_dot2_i32_i16(s0, s1, s2, clamp=False)
        self.assertEqual(res_unclamped, 0x8000_0000)

    def test_clamp_positive_saturation(self) -> None:
        """Positive overflow saturates to 0x7FFFFFFF (INT32_MAX) when clamp=True."""
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(-32768, -32768)
        s2 = 0
        res_clamped = ref_dot2_i32_i16(s0, s1, s2, clamp=True)
        self.assertEqual(res_clamped, UINT32_INT_MAX)

    def test_clamp_negative_saturation(self) -> None:
        """Negative underflow saturates to 0x80000000 (INT32_MIN) when clamp=True."""
        # (-32768 * 32767) * 2 = -2 * 1073709056 = -2147418112
        # Adding accumulator -100000 causes signed underflow < -2147483648
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(32767, 32767)
        s2 = u32(-100_000)
        res_clamped = ref_dot2_i32_i16(s0, s1, s2, clamp=True)
        self.assertEqual(res_clamped, UINT32_INT_MIN)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    """Represents a register operand or inline literal constant."""
    reg_type: RegType
    index: int = 0
    count: int = 1
    literal_value: int | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"Operand count must be >= 1, got {self.count}")
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
            if self.index + self.count > 256:
                raise ValueError(f"VGPR range exceeds 255")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
            if self.index + self.count > 106:
                raise ValueError(f"SGPR range exceeds 105")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            if not (0 <= self.literal_value <= MASK32):
                raise ValueError(f"Literal value {self.literal_value:#x} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def vgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.VGPR, index=index, count=count)

    @classmethod
    def sgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.SGPR, index=index, count=count)

    @classmethod
    def literal(cls, value: int, count: int = 1) -> Operand:
        return cls(RegType.LITERAL, count=count, literal_value=value & MASK32)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            if self.count == 1:
                return f"v{self.index}"
            return f"v[{self.index}:{self.index + self.count - 1}]"
        if self.reg_type == RegType.SGPR:
            if self.count == 1:
                return f"s{self.index}"
            return f"s[{self.index}:{self.index + self.count - 1}]"
        assert self.literal_value is not None
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VDot2I32I16Instruction:
    """Symbolic AST node for AMD RDNA2 v_dot2_i32_i16 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be a 32-bit VGPR (count=1), got {self.dst.count}")
        if self.src0.count != 1:
            raise ValueError(f"src0 must be a 32-bit operand (count=1), got {self.src0.count}")
        if self.src1.count != 1:
            raise ValueError(f"src1 must be a 32-bit operand (count=1), got {self.src1.count}")
        if self.src2.count != 1:
            raise ValueError(f"src2 must be a 32-bit operand (count=1), got {self.src2.count}")

    def disassemble(self) -> str:
        base = (
            f"v_dot2_i32_i16 {self.dst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, "
            f"{self.src2.disassemble()}"
        )
        return f"{base} clamp" if self.clamp else base


class TestValuDot2I32I16SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self) -> None:
        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.vgpr(3),
        )
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.src0.index, 1)
        self.assertEqual(instr.src1.index, 2)
        self.assertEqual(instr.src2.index, 3)
        self.assertEqual(
            instr.disassemble(),
            "v_dot2_i32_i16 v0, v1, s2, v3",
        )

    def test_ast_disassembly_with_clamp_and_literal(self) -> None:
        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(6),
            src1=Operand.literal(0x12345678),
            src2=Operand.vgpr(7),
            clamp=True,
        )
        self.assertEqual(
            instr.disassemble(),
            "v_dot2_i32_i16 v5, v6, 0x12345678, v7 clamp",
        )

    def test_ast_immutability(self) -> None:
        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        with self.assertRaises(Exception):
            instr.dst = Operand.vgpr(4)  # type: ignore[misc]

    def test_ast_operand_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarRegisterState:
    """Architectural register state for a single scalar thread."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def read_operand_32(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        assert op.literal_value is not None
        return op.literal_value & MASK32

    def write_vgpr_32(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar interpreter executing v_dot2_i32_i16 instructions."""

    def step(self, instr: VDot2I32I16Instruction, state: ScalarRegisterState) -> int:
        s0 = state.read_operand_32(instr.src0)
        s1 = state.read_operand_32(instr.src1)
        s2 = state.read_operand_32(instr.src2)
        res = ref_dot2_i32_i16(s0, s1, s2, clamp=instr.clamp)
        state.write_vgpr_32(instr.dst.index, res)
        return res


class TestValuDot2I32I16ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def setUp(self) -> None:
        self.state = ScalarRegisterState()
        self.interp = ScalarInterpreter()

    def test_scalar_execution_from_vgprs(self) -> None:
        # s0 in v1: (10, -20)
        self.state.vgprs[1] = pack_i16x2(10, -20)
        # s1 in v2: (30, 40)
        self.state.vgprs[2] = pack_i16x2(30, 40)
        # s2 in v3: 500
        self.state.vgprs[3] = 500

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        res = self.interp.step(instr, self.state)
        # 10*30 + (-20)*40 + 500 = 300 - 800 + 500 = 0
        self.assertEqual(res, 0)
        self.assertEqual(self.state.vgprs[0], 0)

    def test_scalar_execution_with_sgpr_and_literal(self) -> None:
        self.state.vgprs[0] = pack_i16x2(-100, 200)
        self.state.sgprs[4] = pack_i16x2(5, -2)

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(1),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(4),
            src2=Operand.literal(1000),
        )

        res = self.interp.step(instr, self.state)
        # (-100)*5 + 200*(-2) + 1000 = -500 - 400 + 1000 = 100
        self.assertEqual(res, 100)
        self.assertEqual(self.state.vgprs[1], 100)

    def test_multi_step_chained_accumulation(self) -> None:
        self.state.vgprs[1] = pack_i16x2(3, 4)
        self.state.vgprs[2] = pack_i16x2(5, 6)
        self.state.vgprs[3] = 0

        # Step 1: v0 = (3,4).(5,6) + 0 = 15 + 24 = 39
        instr1 = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.vgprs[0], 39)

        # Step 2: v0 = (10,10).(2,2) + v0 = 20 + 20 + 39 = 79
        self.state.vgprs[4] = pack_i16x2(10, 10)
        self.state.vgprs[5] = pack_i16x2(2, 2)
        instr2 = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(4),
            src1=Operand.vgpr(5),
            src2=Operand.vgpr(0),
        )
        self.interp.step(instr2, self.state)
        self.assertEqual(self.state.vgprs[0], 79)

    def test_scalar_determinism(self) -> None:
        state_a = ScalarRegisterState()
        state_b = ScalarRegisterState()
        val_s0 = pack_i16x2(-1234, 5678)
        val_s1 = pack_i16x2(987, -654)
        state_a.vgprs[1] = val_s0
        state_b.vgprs[1] = val_s0
        state_a.sgprs[0] = val_s1
        state_b.sgprs[0] = val_s1

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(0),
            src2=Operand.literal(42),
        )
        res_a = self.interp.step(instr, state_a)
        res_b = self.interp.step(instr, state_b)
        self.assertEqual(res_a, res_b)
        self.assertEqual(state_a.vgprs, state_b.vgprs)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDWaveState:
    """Architectural SIMD state for Wave32 and Wave64 execution modes."""
    wave_size: int = 32
    exec_mask: int = MASK32
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size {self.wave_size}, must be 32 or 64")
        if self.wave_size == 32:
            self.exec_mask &= MASK32
        else:
            self.exec_mask &= 0xFFFF_FFFF_FFFF_FFFF
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        return bool((self.exec_mask >> lane) & 1)

    def read_lane_operand_32(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        assert op.literal_value is not None
        return op.literal_value & MASK32

    def write_lane_vgpr_32(self, index: int, lane: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32


class SIMDInterpreter:
    """SIMD vector interpreter executing instructions with lane-divergence and EXEC masking."""

    def step(self, instr: VDot2I32I16Instruction, state: SIMDWaveState) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for lane in range(state.wave_size):
            if not state.is_lane_active(lane):
                results.append(None)
                continue
            s0 = state.read_lane_operand_32(instr.src0, lane)
            s1 = state.read_lane_operand_32(instr.src1, lane)
            s2 = state.read_lane_operand_32(instr.src2, lane)
            res = ref_dot2_i32_i16(s0, s1, s2, clamp=instr.clamp)
            state.write_lane_vgpr_32(instr.dst.index, lane, res)
            results.append(res)
        return results


class TestValuDot2I32I16SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDWaveState(wave_size=32, exec_mask=MASK32)
        interp = SIMDInterpreter()

        # Shared query in s2: (10, -5)
        state.sgprs[2] = pack_i16x2(10, -5)

        for lane in range(32):
            val_s0 = pack_i16x2(lane, 2)
            state.write_lane_vgpr_32(0, lane, val_s0)
            state.write_lane_vgpr_32(1, lane, 100)

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(2),
            src2=Operand.vgpr(1),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = lane * 10 + 2 * (-5) + 100
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][3], expected)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDWaveState(wave_size=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        interp = SIMDInterpreter()
        state.sgprs[0] = pack_i16x2(3, 4)

        for lane in range(64):
            val = pack_i16x2(lane, -lane)
            state.write_lane_vgpr_32(0, lane, val)

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(1),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(0),
            src2=Operand.literal(10),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = u32(lane * 3 + (-lane) * 4 + 10)
            self.assertEqual(results[lane], expected)

    def test_wave32_partial_exec_mask_inactive_lanes_unmodified(self) -> None:
        mask = 0x5555_5555  # Even active
        state = SIMDWaveState(wave_size=32, exec_mask=mask)
        interp = SIMDInterpreter()

        sentinel = 0xDEAD_BEEF
        s0_val = pack_i16x2(5, 5)
        s1_val = pack_i16x2(2, 2)

        for lane in range(32):
            state.write_lane_vgpr_32(0, lane, sentinel)
            state.write_lane_vgpr_32(1, lane, s0_val)
            state.write_lane_vgpr_32(2, lane, s1_val)

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.literal(0),
        )

        results = interp.step(instr, state)
        expected_active = 5 * 2 + 5 * 2  # 20
        for lane in range(32):
            if (mask >> lane) & 1:
                self.assertEqual(results[lane], expected_active)
                self.assertEqual(state.vgprs[lane][0], expected_active)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], sentinel)

    def test_wave64_isolated_lane_divergence(self) -> None:
        mask = 1 << 53
        state = SIMDWaveState(wave_size=64, exec_mask=mask)
        interp = SIMDInterpreter()

        sentinel = 0xCAFE_BABE
        for lane in range(64):
            state.write_lane_vgpr_32(0, lane, sentinel)
            state.write_lane_vgpr_32(1, lane, pack_i16x2(7, 8))

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.literal(pack_i16x2(2, 3)),
            src2=Operand.literal(5),
        )

        results = interp.step(instr, state)
        # 7*2 + 8*3 + 5 = 14 + 24 + 5 = 43
        expected = 43
        for lane in range(64):
            if lane == 53:
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.vgprs[lane][0], expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], sentinel)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuDot2I32I16RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.state = ScalarRegisterState()
        self.interp = ScalarInterpreter()

    def test_aliasing_dst_equals_src2_inplace_accumulator(self) -> None:
        """v0 = v_dot2_i32_i16 v1, v2, v0 (in-place accumulator)."""
        acc_initial = 100
        self.state.vgprs[0] = acc_initial
        self.state.vgprs[1] = pack_i16x2(4, 5)
        self.state.vgprs[2] = pack_i16x2(6, 7)

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        res = self.interp.step(instr, self.state)
        # 4*6 + 5*7 + 100 = 24 + 35 + 100 = 159
        self.assertEqual(res, 159)
        self.assertEqual(self.state.vgprs[0], 159)

    def test_aliasing_dst_equals_src0_vector_overwrite(self) -> None:
        """v0 = v_dot2_i32_i16 v0, v1, v2 (destination overwrites src0)."""
        s0_val = pack_i16x2(-10, 20)
        self.state.vgprs[0] = s0_val
        self.state.vgprs[1] = pack_i16x2(3, 2)
        self.state.vgprs[2] = 50

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )

        res = self.interp.step(instr, self.state)
        # (-10)*3 + 20*2 + 50 = -30 + 40 + 50 = 60
        self.assertEqual(res, 60)
        self.assertEqual(self.state.vgprs[0], 60)

    def test_aliasing_src0_equals_src1_self_dot_product(self) -> None:
        """v0 = v_dot2_i32_i16 v1, v1, v2 (squared magnitude |v1|^2 + v2)."""
        val_v1 = pack_i16x2(30, -40)
        self.state.vgprs[1] = val_v1
        self.state.vgprs[2] = 100

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )

        res = self.interp.step(instr, self.state)
        # 30^2 + (-40)^2 + 100 = 900 + 1600 + 100 = 2600
        self.assertEqual(res, 2600)
        self.assertEqual(self.state.vgprs[1], val_v1)

    def test_aliasing_quadruple_collision(self) -> None:
        """v0 = v_dot2_i32_i16 v0, v0, v0."""
        # v0 holds (2, 3) -> 2 | (3 << 16) = 0x00030002
        # As 32-bit signed int: 0x00030002 = 196610
        val = pack_i16x2(2, 3)
        self.state.vgprs[0] = val

        instr = VDot2I32I16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(0),
        )

        expected = ref_dot2_i32_i16(val, val, val)
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.vgprs[0], expected)


# ==============================================================================
# Condition Codes, Status Flags, and Overflow Generation
# ==============================================================================

@dataclass(frozen=True)
class IntegerStatusFlags:
    """Architectural status flags for signed 32-bit integer arithmetic."""
    is_zero: bool
    is_negative: bool
    overflow: bool
    carry: bool


def compute_integer_status_flags(
    s0_packed: int,
    s1_packed: int,
    s2_i32: int,
    *,
    clamp: bool = False,
) -> IntegerStatusFlags:
    """Compute architectural integer condition flags for v_dot2_i32_i16."""
    a0, a1 = unpack_i16x2(s0_packed)
    b0, b1 = unpack_i16x2(s1_packed)
    c = s32(s2_i32)

    full = (a0 * b0) + (a1 * b1) + c
    overflow = (full > INT32_MAX) or (full < INT32_MIN)

    if clamp:
        if full > INT32_MAX:
            res = UINT32_INT_MAX
        elif full < INT32_MIN:
            res = UINT32_INT_MIN
        else:
            res = u32(full)
    else:
        res = u32(full)

    is_zero = (res == 0)
    is_negative = bool(res & 0x8000_0000)
    carry = bool((full >> 32) & 1)

    return IntegerStatusFlags(
        is_zero=is_zero,
        is_negative=is_negative,
        overflow=overflow,
        carry=carry,
    )


class TestValuDot2I32I16ConditionCodesAndFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and overflow generation."""

    def test_zero_flag(self) -> None:
        flags = compute_integer_status_flags(0, 0, 0)
        self.assertTrue(flags.is_zero)
        self.assertFalse(flags.is_negative)
        self.assertFalse(flags.overflow)

    def test_negative_flag(self) -> None:
        s0 = pack_i16x2(5, 0)
        s1 = pack_i16x2(-10, 0)
        s2 = 0
        flags = compute_integer_status_flags(s0, s1, s2)
        self.assertTrue(flags.is_negative)
        self.assertFalse(flags.is_zero)

    def test_signed_overflow_detection(self) -> None:
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(-32768, -32768)
        s2 = 0
        flags = compute_integer_status_flags(s0, s1, s2)
        self.assertTrue(flags.overflow)

    def test_clamped_clears_negative_flag_on_positive_overflow(self) -> None:
        s0 = pack_i16x2(-32768, -32768)
        s1 = pack_i16x2(-32768, -32768)
        s2 = 0
        # Unclamped: wraps to 0x80000000 (negative)
        flags_unclamped = compute_integer_status_flags(s0, s1, s2, clamp=False)
        self.assertTrue(flags_unclamped.is_negative)

        # Clamped: saturates to 0x7FFFFFFF (positive)
        flags_clamped = compute_integer_status_flags(s0, s1, s2, clamp=True)
        self.assertFalse(flags_clamped.is_negative)
        self.assertTrue(flags_clamped.overflow)


# ==============================================================================
# IEEE 754 Boundary Extremes and Raw Bit Patterns
# ==============================================================================

class TestValuDot2I32I16IEEE754BitPatterns(unittest.TestCase):
    """Verifies integer interpretation of IEEE 754 float bit patterns in registers."""

    POS_ZERO_F16 = 0x0000
    NEG_ZERO_F16 = 0x8000  # As signed i16: -32768
    POS_INF_F16 = 0x7C00   # As signed i16: 31744
    NEG_INF_F16 = 0xFC00   # As signed i16: -1024
    QNAN_F16 = 0x7E00      # As signed i16: 32256

    def test_signed_zero_float_bits_interpreted_as_int16(self) -> None:
        """+0.0 (0x0000) is 0, -0.0 (0x8000) is -32768."""
        s0 = pack_i16x2(self.NEG_ZERO_F16, self.POS_ZERO_F16)
        s1 = pack_i16x2(1, 0)
        s2 = 0
        res = ref_dot2_i32_i16(s0, s1, s2)
        # (-32768) * 1 + 0 = -32768
        self.assertEqual(res, u32(-32768))

    def test_infinity_float_bits_arithmetic(self) -> None:
        """+inf bitfield (0x7C00 = 31744) multiplied as signed int16."""
        s0 = pack_i16x2(self.POS_INF_F16, 0)
        s1 = pack_i16x2(2, 0)
        s2 = 0
        res = ref_dot2_i32_i16(s0, s1, s2)
        # 31744 * 2 = 63488
        self.assertEqual(res, 63488)

    def test_nan_float_bits_arithmetic(self) -> None:
        """qNaN bitfield (0x7E00 = 32256) multiplied without exception."""
        s0 = pack_i16x2(self.QNAN_F16, 0)
        s1 = pack_i16x2(3, 0)
        s2 = 100
        res = ref_dot2_i32_i16(s0, s1, s2)
        # 32256 * 3 + 100 = 96768 + 100 = 96868
        self.assertEqual(res, 96868)


if __name__ == "__main__":
    unittest.main()
