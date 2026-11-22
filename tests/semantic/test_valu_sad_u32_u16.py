from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for AMD RDNA2 v_sad_u32_u16.

Architecture & Instruction Specifications (AMD RDNA2 / GFX10 Vector ALU):
- v_sad_u32_u16 (VOP3 3-operand vector ALU instruction):
    Computes an unsigned 32-bit sum of absolute differences with mixed 16-bit and 32-bit operands:
        s0.u16 = (src0 >> (16 if op_sel[0] else 0)) & 0xFFFF
        s1.u16 = (src1 >> (16 if op_sel[1] else 0)) & 0xFFFF
        diff.u32 = abs(s0.u16 - s1.u16)
        full_sum = diff.u32 + (src2.u32 & 0xFFFFFFFF)
        dst.u32 = min(0xFFFFFFFF, full_sum) if clamp else (full_sum mod 2^32)
        carry_out = 1 if (full_sum >= 2^32) else 0

    Outputs:
        Vector Destination (VDST): 32-bit VGPR storing dst.u32.
        Carry generation: Tracks whether full_sum exceeded 2^32 (overflowing 32-bit unsigned range).

    Mathematical Bounds:
        - s0.u16 in [0, 65535] (0x0000 to 0xFFFF)
        - s1.u16 in [0, 65535] (0x0000 to 0xFFFF)
        - src2.u32 in [0, 2^32 - 1] (0 to 0xFFFFFFFF)
        - Maximum difference: |65535 - 0| = 65535 = 0xFFFF
        - Maximum full sum: 0xFFFF + 0xFFFFFFFF = 0x10000FFFE
        - Therefore, carry_out is strictly binary {0, 1}.

    Modifiers:
        - op_sel[0]: Selects low (0) or high (1) 16 bits of src0 register.
        - op_sel[1]: Selects low (0) or high (1) 16 bits of src1 register.
        - clamp: When enabled, saturates the destination result to 0xFFFFFFFF on overflow.

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


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def ref_extract_u16(val: int, high_halfword: bool = False) -> int:
    """Extract selected 16-bit unsigned halfword from a 32-bit word."""
    shift = 16 if high_halfword else 0
    return (val >> shift) & MASK16


def ref_abs_diff_u16(s0: int, s1: int) -> int:
    """Pure-Python reference oracle for 16-bit unsigned absolute difference: |s0 - s1|."""
    u0 = s0 & MASK16
    u1 = s1 & MASK16
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u32_u16(
    s0: int,
    s1: int,
    s2: int,
    op_sel0: bool = False,
    op_sel1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_sad_u32_u16:

    dst = min(0xFFFFFFFF, |s0.u16 - s1.u16| + s2.u32) if clamp else (|s0.u16 - s1.u16| + s2.u32) & 0xFFFFFFFF
    """
    u0 = ref_extract_u16(s0, op_sel0)
    u1 = ref_extract_u16(s1, op_sel1)
    diff = ref_abs_diff_u16(u0, u1)
    acc = s2 & MASK32
    full = diff + acc
    if clamp and full > MASK32:
        return MASK32
    return full & MASK32


def ref_sad_u32_u16_full(
    s0: int,
    s1: int,
    s2: int,
    op_sel0: bool = False,
    op_sel1: bool = False,
    clamp: bool = False,
) -> Tuple[int, int, int]:
    """Return (dst_u32, carry_out, full_sum64) for v_sad_u32_u16."""
    u0 = ref_extract_u16(s0, op_sel0)
    u1 = ref_extract_u16(s1, op_sel1)
    diff = ref_abs_diff_u16(u0, u1)
    acc = s2 & MASK32
    full = diff + acc
    carry = (full >> 32) & 1
    dst = MASK32 if (clamp and full > MASK32) else (full & MASK32)
    return dst, carry, full


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestVSadU32U16Oracle(unittest.TestCase):
    """Test suite verifying mathematical bounds, properties, and edge cases of the oracle."""

    def test_oracle_zero_inputs(self) -> None:
        """All-zero inputs yield zero result."""
        self.assertEqual(ref_sad_u32_u16(0, 0, 0), 0)
        dst, carry, full = ref_sad_u32_u16_full(0, 0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(carry, 0)
        self.assertEqual(full, 0)

    def test_oracle_identical_inputs(self) -> None:
        """When s0 == s1, absolute difference is zero and result is accumulator s2."""
        patterns = [0x00000000, 0x00001234, 0x0000FFFF, 0x12345678, 0xFFFFFFFF]
        for pat in patterns:
            acc = 0xCAFEBABE
            self.assertEqual(ref_sad_u32_u16(pat, pat, acc), acc)

    def test_oracle_maximum_difference(self) -> None:
        """Max difference between 16-bit operands (|0xFFFF - 0x0000| = 65535)."""
        s0 = 0x0000FFFF
        s1 = 0x00000000
        self.assertEqual(ref_sad_u32_u16(s0, s1, 0), 65535)

    def test_oracle_op_sel_halfword_selection(self) -> None:
        """Verify op_sel modifiers properly select low vs high halfwords."""
        s0 = 0x1234_5678
        s1 = 0x1234_0000

        # op_sel0=False (low: 0x5678), op_sel1=True (high: 0x1234)
        # diff = |0x5678 - 0x1234| = 0x4444 = 17476
        res = ref_sad_u32_u16(s0, s1, 0, op_sel0=False, op_sel1=True)
        self.assertEqual(res, 0x4444)

        # op_sel0=True (high: 0x1234), op_sel1=True (high: 0x1234) -> diff = 0
        res_zero = ref_sad_u32_u16(s0, s1, 0, op_sel0=True, op_sel1=True)
        self.assertEqual(res_zero, 0)

    def test_oracle_commutativity(self) -> None:
        """Commutativity: ref_sad_u32_u16(s0, s1, s2) == ref_sad_u32_u16(s1, s0, s2)."""
        test_pairs = [
            (0x1234, 0x5678),
            (0xFFFF, 0x0000),
            (0x8000, 0x0001),
            (0x7FFF, 0x8000),
        ]
        acc = 0x1000
        for s0, s1 in test_pairs:
            self.assertEqual(ref_sad_u32_u16(s0, s1, acc), ref_sad_u32_u16(s1, s0, acc))

    def test_oracle_accumulation_with_wrap(self) -> None:
        """32-bit truncation when diff + s2 exceeds 0xFFFFFFFF."""
        s0 = 0xFFFF
        s1 = 0x0000
        # diff = 65535 (0xFFFF); acc = 0xFFFFFFFF
        # 0xFFFF + 0xFFFFFFFF = 0x1_0000_FFFE -> wrapped = 0x0000FFFE
        dst, carry, full = ref_sad_u32_u16_full(s0, s1, 0xFFFF_FFFF)
        self.assertEqual(dst, 0x0000_FFFE)
        self.assertEqual(carry, 1)
        self.assertEqual(full, 0x1_0000_FFFE)

    def test_oracle_clamp_modifier(self) -> None:
        """Clamp modifier saturates to 0xFFFFFFFF on overflow instead of wrapping."""
        s0 = 0xFFFF
        s1 = 0x0000
        acc = 0xFFFF_FFFF
        # Without clamp -> 0x0000FFFE
        self.assertEqual(ref_sad_u32_u16(s0, s1, acc, clamp=False), 0x0000_FFFE)
        # With clamp -> saturated to 0xFFFFFFFF
        self.assertEqual(ref_sad_u32_u16(s0, s1, acc, clamp=True), 0xFFFF_FFFF)




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
class VSadU32U16Instruction:
    """Symbolic AST node for AMD RDNA2 v_sad_u32_u16 VALU instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    op_sel0: bool = False
    op_sel1: bool = False
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        mods = []
        if self.op_sel0 or self.op_sel1:
            mods.append(f"op_sel:[{int(self.op_sel0)},{int(self.op_sel1)},0]")
        if self.clamp:
            mods.append("clamp")
        mod_str = (" " + " ".join(mods)) if mods else ""
        return (
            f"v_sad_u32_u16 {self.dst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}{mod_str}"
        )


class TestVSadU32U16SymbolicAST(unittest.TestCase):
    """Test suite verifying symbolic AST construction, validation, and disassembly."""

    def test_ast_creation_valid_operands(self) -> None:
        """Construct AST nodes with valid operands and modifiers."""
        instr = VSadU32U16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(0x1234),
            op_sel0=True,
            clamp=True,
        )
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.src0.index, 1)
        self.assertEqual(instr.src1.index, 2)
        self.assertEqual(instr.src2.literal_value, 0x1234)
        self.assertTrue(instr.op_sel0)
        self.assertTrue(instr.clamp)

    def test_ast_disassembly(self) -> None:
        """Disassembly formatting matches AMD RDNA ISA assembly conventions."""
        instr = VSadU32U16Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(10),
            src1=Operand.sgpr(4),
            src2=Operand.literal(0x00FF00FF),
            op_sel0=True,
            clamp=True,
        )
        self.assertEqual(
            instr.disassemble(),
            "v_sad_u32_u16 v5, v10, s4, 0x00FF00FF op_sel:[1,0,0] clamp",
        )

    def test_ast_dst_must_be_vgpr(self) -> None:
        """VALU instruction destination must be a VGPR; SGPR/Literal raises TypeError."""
        with self.assertRaises(TypeError):
            VSadU32U16Instruction(
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
    """Deterministic scalar interpreter executing v_sad_u32_u16 instructions."""

    def step(self, instr: VSadU32U16Instruction, state: ScalarState) -> int:
        """Execute a single v_sad_u32_u16 instruction against architectural scalar state."""
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        s2 = state.read_operand(instr.src2)

        res = ref_sad_u32_u16(
            s0,
            s1,
            s2,
            op_sel0=instr.op_sel0,
            op_sel1=instr.op_sel1,
            clamp=instr.clamp,
        )
        state.write_vgpr(instr.dst.index, res)
        return res


class TestVSadU32U16ScalarInterpreter(unittest.TestCase):
    """Test suite verifying single-thread deterministic scalar interpreter execution."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_interp_basic_execution(self) -> None:
        """Execute basic v_sad_u32_u16 instruction updating state.vgprs[dst]."""
        self.state.vgprs[1] = 0x0000_0300
        self.state.vgprs[2] = 0x0000_0100
        self.state.sgprs[0] = 0x1000

        instr = VSadU32U16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.sgpr(0),
        )

        res = self.interp.step(instr, self.state)
        # diff = 0x300 - 0x100 = 0x200 -> + 0x1000 = 0x1200
        self.assertEqual(res, 0x1200)
        self.assertEqual(self.state.vgprs[0], 0x1200)

    def test_scalar_interp_multi_step_sequence(self) -> None:
        """Sequential chained execution with accumulation."""
        self.state.vgprs[1] = 100
        self.state.vgprs[2] = 0
        self.state.vgprs[0] = 0

        # Step 1: v0 = v_sad_u32_u16(v1, v2, 0)
        instr1 = VSadU32U16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.literal(0),
        )
        self.interp.step(instr1, self.state)
        self.assertEqual(self.state.vgprs[0], 100)

        # Step 2: v0 = v_sad_u32_u16(v1, v2, v0) -> adds another 100 -> 200
        instr2 = VSadU32U16Instruction(
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
            (0x0000FFFF, 0x00000000, 0x00000000),
            (0x12345678, 0x87654321, 0x00010000),
            (0x00005555, 0x0000AAAA, 0xFFFFFFFF),
        ]
        for s0, s1, s2 in test_vectors:
            self.state.vgprs[1] = s0
            self.state.vgprs[2] = s1
            self.state.vgprs[3] = s2
            instr = VSadU32U16Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
            res = self.interp.step(instr, self.state)
            expected = ref_sad_u32_u16(s0, s1, s2)
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

    def step(self, instr: VSadU32U16Instruction, state: SIMDState) -> List[Optional[int]]:
        """Execute a v_sad_u32_u16 instruction across all SIMD lanes under EXEC masking."""
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                s0 = state.read_lane(instr.src0, lane)
                s1 = state.read_lane(instr.src1, lane)
                s2 = state.read_lane(instr.src2, lane)

                res = ref_sad_u32_u16(
                    s0,
                    s1,
                    s2,
                    op_sel0=instr.op_sel0,
                    op_sel1=instr.op_sel1,
                    clamp=instr.clamp,
                )
                state.write_lane(instr.dst, lane, res)
                results.append(res)
            else:
                results.append(None)
        return results


class TestVSadU32U16SIMD(unittest.TestCase):
    """Test suite verifying SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_simd_wave32_full_exec(self) -> None:
        """Wave32 execution with all 32 lanes active."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.vgprs[lane][1] = lane
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = 0x0001_0000

        instr = VSadU32U16Instruction(
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
            state.vgprs[lane][1] = 50
            state.vgprs[lane][2] = 0
            state.vgprs[lane][3] = lane

        instr = VSadU32U16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            expected = 50 + lane
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.vgprs[lane][0], expected)

    def test_simd_wave32_divergence_mask(self) -> None:
        """Wave32 divergent execution: only even lanes active (0x55555555)."""
        sentinel = 0xCAFE_BABE
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.vgprs[lane][0] = sentinel
            state.vgprs[lane][1] = 0x0000_FFFF
            state.vgprs[lane][2] = 0x0000_0000
            state.vgprs[lane][3] = 0

        instr = VSadU32U16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0xFFFF)
                self.assertEqual(state.vgprs[lane][0], 0xFFFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.vgprs[lane][0], sentinel)

# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestVSadU32U16RegisterAliasing(unittest.TestCase):
    """Test suite verifying register aliasing and in-place operand preservation."""

    def test_aliasing_dst_equals_src0(self) -> None:
        """dst and src0 share the same VGPR: dst must be read before overwritten."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[0] = 300
        state.vgprs[1] = 0
        state.sgprs[0] = 0x1234

        instr = VSadU32U16Instruction(
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

        state.vgprs[0] = 500  # accumulator in v0
        state.vgprs[1] = 10
        state.vgprs[2] = 0

        # v0 = v_sad_u32_u16(v1, v2, v0)
        instr = VSadU32U16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        # Iteration 1: + 10 -> 510
        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 510)

        # Iteration 2: + 10 -> 520
        interp.step(instr, state)
        self.assertEqual(state.vgprs[0], 520)

    def test_aliasing_src0_equals_src1(self) -> None:
        """src0 == src1 produces diff=0, preserving accumulator exactly."""
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vgprs[1] = 0xDEADBEEF
        state.vgprs[2] = 0xCAFEBABE

        instr = VSadU32U16Instruction(
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


def compute_flags(
    s0: int,
    s1: int,
    s2: int,
    op_sel0: bool = False,
    op_sel1: bool = False,
    clamp: bool = False,
) -> StatusFlags:
    """Evaluate condition codes and status flags resulting from v_sad_u32_u16 execution."""
    dst, carry, _ = ref_sad_u32_u16_full(s0, s1, s2, op_sel0=op_sel0, op_sel1=op_sel1, clamp=clamp)
    u0 = ref_extract_u16(s0, op_sel0)
    u1 = ref_extract_u16(s1, op_sel1)
    diff = ref_abs_diff_u16(u0, u1)
    acc = s2 & MASK32

    zero = (dst == 0)
    sign = bool(dst & 0x8000_0000)
    overflow = bool((~(diff ^ acc) & (diff ^ dst)) & 0x8000_0000)

    return StatusFlags(zero=zero, sign=sign, carry=bool(carry), overflow=overflow)


class TestVSadU32U16ConditionCodesAndFlags(unittest.TestCase):
    """Test suite verifying condition codes, status flags, and carry generation."""

    def test_flags_zero_result(self) -> None:
        """Zero flag asserted when dst evaluates to 0x00000000."""
        flags = compute_flags(0x1234, 0x1234, 0)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.carry)

        # diff=1, acc=0xFFFFFFFF -> sum=0x1_00000000 -> dst=0
        flags_wrap = compute_flags(1, 0, 0xFFFF_FFFF)
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
        # max diff 65535 (0xFFFF); acc = 0xFFFF_0001
        # 0xFFFF + 0xFFFF0001 = 0x1_00000000 -> carry = 1
        flags = compute_flags(0xFFFF, 0x0000, 0xFFFF_0001)
        self.assertTrue(flags.carry)

# ==============================================================================
# Unit Tests: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestVSadU32U16IEEE754BitPatterns(unittest.TestCase):
    """Test suite verifying bit-exact integer handling of IEEE 754 float patterns.

    v_sad_u32_u16 operates strictly on integer bitfields. Halfword patterns corresponding
    to IEEE 754 half-precision (f16) or single-precision (f32) specials (+0.0, -0.0,
    subnormals, infinities, NaNs) must be evaluated as unsigned binary integers without
    floating-point normalization or subnormal flush-to-zero (FTZ).
    """

    @staticmethod
    def float_to_u32(val: float) -> int:
        """Convert Python float to 32-bit unsigned integer bitcast."""
        return struct.unpack("<I", struct.pack("<f", val))[0]

    def test_ieee754_zero_and_negative_zero(self) -> None:
        """+0.0 (0x00000000) vs -0.0 (0x80000000) in single precision."""
        pos_zero = self.float_to_u32(+0.0)  # 0x00000000
        neg_zero = self.float_to_u32(-0.0)  # 0x80000000

        # Low halfword diff: |0x0000 - 0x0000| = 0
        self.assertEqual(ref_sad_u32_u16(pos_zero, neg_zero, 0), 0)

        # High halfword diff: |0x0000 - 0x8000| = 32768
        self.assertEqual(
            ref_sad_u32_u16(pos_zero, neg_zero, 0, op_sel0=True, op_sel1=True),
            32768,
        )

    def test_ieee754_f16_half_precision_specials(self) -> None:
        """Verify packed IEEE 754 half-precision (f16) values."""
        f16_pos_inf = 0x7C00
        f16_neg_inf = 0xFC00
        f16_qnan = 0x7E00

        # Diff between +inf and -inf: |0x7C00 - 0xFC00| = 32768 (0x8000)
        self.assertEqual(ref_sad_u32_u16(f16_pos_inf, f16_neg_inf, 0), 32768)

        # Diff between NaN and NaN is identically 0
        self.assertEqual(ref_sad_u32_u16(f16_qnan, f16_qnan, 0x55), 0x55)

    def test_ieee754_subnormals_no_denorm_flushing(self) -> None:
        """16-bit subnormal bit patterns are never flushed to zero."""
        f16_subnorm_min = 0x0001
        self.assertEqual(ref_sad_u32_u16(f16_subnorm_min, 0, 0), 1)

if __name__ == "__main__":
    unittest.main()
