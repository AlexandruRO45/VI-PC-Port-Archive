"""Semantic hardware-level and translator reference tests for AMD RDNA2 v_lshrrev_b16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture & Semantics:
  In AMD RDNA2 (GFX10 / GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  `v_lshrrev_b16` is a 16-bit unsigned logical shift right reverse instruction.

  Instruction:
    v_lshrrev_b16 vd, src0, src1

  Mathematical Specification:
    shift = src0 & 15
    val = src1 & 0xFFFF
    D.u16 = (val >> shift) & 0xFFFF

  RDNA2 Hardware Rules:
    - Reversed operand order: src0 is the shift count, src1 is the data value.
    - Shift amount is masked to low 4 bits (src0 & 15).
    - Bits shifted into MSB are zeroes (logical shift right, zero-extended).
    - High 16 bits of destination VGPR follow RDNA2 VOP2 / VOP3 encoding semantics.
    - Flags:
        Neither SCC nor VCC is modified by this bitwise instruction.
    - Execution Masking:
        Inactive lanes under EXEC mask do not write to their destination VGPR.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def ref_lshrrev_b16(shift_raw: int, val_raw: int) -> int:
    """Pure bitwise reference mathematical oracle for RDNA2 v_lshrrev_b16.

    Computes:
      shift = shift_raw & 15
      val = val_raw & 0xFFFF
      (val >> shift) & 0xFFFF
    """
    shift = shift_raw & 15
    val = val_raw & MASK16
    return (val >> shift) & MASK16


# =============================================================================
# Symbolic AST & Instruction Representation
# =============================================================================

@dataclasses.dataclass(frozen=True)
class RegVGPR:
    idx: int

    def __post_init__(self) -> None:
        if not (0 <= self.idx <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.idx}")


@dataclasses.dataclass(frozen=True)
class ImmU16:
    val: int

    def __post_init__(self) -> None:
        if not (0 <= self.val <= MASK16):
            raise ValueError(f"16-bit constant out of range: {self.val}")


@dataclasses.dataclass(frozen=True)
class SourceOperand:
    val: Union[RegVGPR, ImmU16]


@dataclasses.dataclass(frozen=True)
class VLshrrevB16Instruction:
    vdst: RegVGPR
    src0: SourceOperand  # shift count
    src1: SourceOperand  # value to shift

    @staticmethod
    def parse(line: str) -> VLshrrevB16Instruction:
        tokens = line.replace(",", " ").split()
        if not tokens or tokens[0] != "v_lshrrev_b16":
            raise ValueError(f"Invalid opcode for v_lshrrev_b16: {line}")
        if len(tokens) != 4:
            raise ValueError(f"Expected 3 operands for v_lshrrev_b16: {line}")
        vdst = VLshrrevB16Instruction._parse_reg(tokens[1])
        src0 = VLshrrevB16Instruction._parse_src(tokens[2])
        src1 = VLshrrevB16Instruction._parse_src(tokens[3])
        return VLshrrevB16Instruction(vdst=vdst, src0=src0, src1=src1)

    @staticmethod
    def _parse_reg(text: str) -> RegVGPR:
        t = text.strip().lower()
        if not (t.startswith("v") and t[1:].isdigit()):
            raise ValueError(f"Malformed VGPR register: {text}")
        return RegVGPR(int(t[1:]))

    @staticmethod
    def _parse_src(text: str) -> SourceOperand:
        t = text.strip()
        if t.lower().startswith("v") and t[1:].isdigit():
            return SourceOperand(val=RegVGPR(int(t[1:])))
        else:
            val_int = int(t, 0)
            return SourceOperand(val=ImmU16(val_int & MASK16))


# =============================================================================
# Architectural Interpreter
# =============================================================================

class ScalarInterpreter:
    """Deterministic scalar register file and execution engine for v_lshrrev_b16."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.vcc: int = 0
        self.scc: int = 0

    def read_u16(self, reg: RegVGPR) -> int:
        return self.vgprs[reg.idx] & MASK16

    def write_u16(self, reg: RegVGPR, val16: int) -> None:
        # VOP2 zero-extends to 32 bits on GFX10
        self.vgprs[reg.idx] = val16 & MASK16

    def evaluate_src(self, src: SourceOperand) -> int:
        if isinstance(src.val, RegVGPR):
            return self.read_u16(src.val)
        elif isinstance(src.val, ImmU16):
            return src.val.val
        else:
            raise TypeError(f"Unknown operand type: {type(src.val)}")

    def execute(self, instr: VLshrrevB16Instruction) -> int:
        sh = self.evaluate_src(instr.src0)
        v = self.evaluate_src(instr.src1)
        res = ref_lshrrev_b16(sh, v)
        self.write_u16(instr.vdst, res)
        return res


class SIMDInterpreter:
    """SIMD execution engine supporting 32 and 64 lanes with divergence masking."""

    def __init__(self, num_lanes: int = 64) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count: {num_lanes}")
        self.num_lanes = num_lanes
        self.lanes: List[ScalarInterpreter] = [ScalarInterpreter() for _ in range(num_lanes)]

    def execute_wave(self, instr: VLshrrevB16Instruction, exec_mask: int) -> List[int]:
        results: List[int] = []
        for lane_id in range(self.num_lanes):
            lane_active = bool(exec_mask & (1 << lane_id))
            interp = self.lanes[lane_id]
            if lane_active:
                results.append(interp.execute(instr))
            else:
                results.append(interp.read_u16(instr.vdst))
        return results


# =============================================================================
# Unit Tests: Mathematical Oracle & Edge Cases
# =============================================================================

class TestValuLshrrevB16MathematicalOracle(unittest.TestCase):
    """Tests covering reference mathematical oracle and 16-bit logical right shifts."""

    def test_basic_shifts(self) -> None:
        # shift = 4, val = 0x1230 -> 0x0123
        self.assertEqual(ref_lshrrev_b16(4, 0x1230), 0x0123)
        # shift = 1, val = 0x0002 -> 0x0001
        self.assertEqual(ref_lshrrev_b16(1, 0x0002), 0x0001)
        # shift = 8, val = 0xFF00 -> 0x00FF
        self.assertEqual(ref_lshrrev_b16(8, 0xFF00), 0x00FF)

    def test_zero_shift(self) -> None:
        # shift = 0 -> identity
        self.assertEqual(ref_lshrrev_b16(0, 0xABCD), 0xABCD)
        self.assertEqual(ref_lshrrev_b16(0, 0x0000), 0x0000)
        self.assertEqual(ref_lshrrev_b16(0, 0xFFFF), 0xFFFF)

    def test_max_shift_15(self) -> None:
        # shift = 15 extracts MSB
        self.assertEqual(ref_lshrrev_b16(15, 0x8000), 1)
        self.assertEqual(ref_lshrrev_b16(15, 0x7FFF), 0)
        self.assertEqual(ref_lshrrev_b16(15, 0xFFFF), 1)

    def test_logical_zero_fill(self) -> None:
        # MSB is 1; logical shift must fill with 0s (not sign-extend)
        self.assertEqual(ref_lshrrev_b16(1, 0x8000), 0x4000)
        self.assertEqual(ref_lshrrev_b16(2, 0x8000), 0x2000)
        self.assertEqual(ref_lshrrev_b16(4, 0xF000), 0x0F00)

    def test_shift_amount_wrap(self) -> None:
        # shift count is masked with & 15
        # 16 & 15 == 0 -> no shift
        self.assertEqual(ref_lshrrev_b16(16, 0x1234), 0x1234)
        # 20 & 15 == 4 -> shift by 4
        self.assertEqual(ref_lshrrev_b16(20, 0x1230), 0x0123)
        # 31 & 15 == 15 -> shift by 15
        self.assertEqual(ref_lshrrev_b16(31, 0x8000), 1)


# =============================================================================
# Unit Tests: Symbolic AST & Parser
# =============================================================================

class TestValuLshrrevB16SymbolicAST(unittest.TestCase):
    """Tests covering AST creation, operand typing, and syntax parsing."""

    def test_instruction_parsing_basic(self) -> None:
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v2, v0, v1")
        self.assertEqual(instr.vdst.idx, 2)
        self.assertEqual(instr.src0.val, RegVGPR(0))
        self.assertEqual(instr.src1.val, RegVGPR(1))

    def test_instruction_parsing_constants(self) -> None:
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v3, 4, 0x1230")
        self.assertEqual(instr.vdst.idx, 3)
        self.assertIsInstance(instr.src0.val, ImmU16)
        self.assertEqual(instr.src0.val.val, 4)
        self.assertIsInstance(instr.src1.val, ImmU16)
        self.assertEqual(instr.src1.val.val, 0x1230)


# =============================================================================
# Unit Tests: Deterministic Scalar Interpreter
# =============================================================================

class TestValuLshrrevB16ScalarInterpreter(unittest.TestCase):
    """Tests covering deterministic register execution in a single thread."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_execution(self) -> None:
        self.interp.vgprs[0] = 4       # shift count
        self.interp.vgprs[1] = 0x4320  # value to shift
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v2, v0, v1")
        got = self.interp.execute(instr)
        self.assertEqual(got, 0x0432)
        self.assertEqual(self.interp.read_u16(RegVGPR(2)), 0x0432)


# =============================================================================
# Unit Tests: SIMD Multi-Lane & EXEC Masking
# =============================================================================

class TestValuLshrrevB16SIMDDivergence(unittest.TestCase):
    """Tests covering SIMD multi-lane divergence across Wave32 and Wave64."""

    def test_wave32_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=32)
        sentinel = 0xDEAD
        for lane_id in range(32):
            simd.lanes[lane_id].vgprs[0] = lane_id % 16  # shift
            simd.lanes[lane_id].vgprs[1] = 0x8000        # val
            simd.lanes[lane_id].vgprs[2] = sentinel

        # Active only on even lanes
        exec_mask = 0x55555555
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v2, v0, v1")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(32):
            if lane_id % 2 == 0:
                expected = ref_lshrrev_b16(lane_id % 16, 0x8000)
                self.assertEqual(results[lane_id], expected)
                self.assertEqual(simd.lanes[lane_id].read_u16(RegVGPR(2)), expected)
            else:
                self.assertEqual(results[lane_id], sentinel)
                self.assertEqual(simd.lanes[lane_id].read_u16(RegVGPR(2)), sentinel)

    def test_wave64_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=64)
        sentinel = 0xBEEF
        for lane_id in range(64):
            simd.lanes[lane_id].vgprs[0] = 4
            simd.lanes[lane_id].vgprs[1] = 0xF000
            simd.lanes[lane_id].vgprs[2] = sentinel

        # Upper 32 lanes active only
        exec_mask = 0xFFFFFFFF00000000
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v2, v0, v1")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(64):
            if lane_id >= 32:
                self.assertEqual(results[lane_id], 0x0F00)
            else:
                self.assertEqual(results[lane_id], sentinel)


# =============================================================================
# Unit Tests: Register Aliasing & Preservation
# =============================================================================

class TestValuLshrrevB16RegisterAliasing(unittest.TestCase):
    """Tests covering destination overwrites and register aliasing."""

    def test_vdst_equals_src0(self) -> None:
        interp = ScalarInterpreter()
        interp.vgprs[0] = 4
        interp.vgprs[1] = 0x1230
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v0, v0, v1")
        got = interp.execute(instr)
        self.assertEqual(got, 0x0123)
        self.assertEqual(interp.read_u16(RegVGPR(0)), 0x0123)
        # Source 1 should be untouched
        self.assertEqual(interp.read_u16(RegVGPR(1)), 0x1230)

    def test_vdst_equals_src1(self) -> None:
        interp = ScalarInterpreter()
        interp.vgprs[0] = 8
        interp.vgprs[1] = 0xAB00
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v1, v0, v1")
        got = interp.execute(instr)
        self.assertEqual(got, 0x00AB)
        self.assertEqual(interp.read_u16(RegVGPR(1)), 0x00AB)
        self.assertEqual(interp.read_u16(RegVGPR(0)), 8)

    def test_all_operands_identical(self) -> None:
        interp = ScalarInterpreter()
        interp.vgprs[0] = 0x0004
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v0, v0, v0")
        got = interp.execute(instr)
        # 4 >> 4 = 0
        self.assertEqual(got, 0)
        self.assertEqual(interp.read_u16(RegVGPR(0)), 0)


# =============================================================================
# Unit Tests: Condition Codes & Flags
# =============================================================================

class TestValuLshrrevB16ConditionCodes(unittest.TestCase):
    """Tests verifying SCC and VCC flags remain completely isolated."""

    def test_flags_preserved(self) -> None:
        interp = ScalarInterpreter()
        interp.scc = 1
        interp.vcc = 0xCAFEBABE12345678
        interp.vgprs[0] = 4
        interp.vgprs[1] = 0x1230
        instr = VLshrrevB16Instruction.parse("v_lshrrev_b16 v2, v0, v1")
        interp.execute(instr)
        self.assertEqual(interp.scc, 1)
        self.assertEqual(interp.vcc, 0xCAFEBABE12345678)


# =============================================================================
# Unit Tests: Negative Controls & Mutation Detection
# =============================================================================

class TestValuLshrrevB16NegativeControls(unittest.TestCase):
    """Negative controls testing syntax rejection and mutation detection."""

    def test_syntax_rejection(self) -> None:
        with self.assertRaises(ValueError):
            VLshrrevB16Instruction.parse("v_lshlrev_b16 v2, v0, v1")

        with self.assertRaises(ValueError):
            # Missing operand
            VLshrrevB16Instruction.parse("v_lshrrev_b16 v2, v0")

    def test_mutation_detector_non_reversed_operands(self) -> None:
        # Mutation: faulty implementation shifting src0 by src1 instead of src1 by src0
        got = ref_lshrrev_b16(4, 0x1230)
        mutated = ref_lshrrev_b16(0x1230, 4)  # 0x1230 & 15 == 0 -> 4 >> 0 = 4
        self.assertEqual(got, 0x0123)
        self.assertNotEqual(got, mutated)

    def test_mutation_detector_arithmetic_shift(self) -> None:
        # Mutation: faulty implementation performing arithmetic shift (sign-extension)
        got = ref_lshrrev_b16(1, 0x8000)
        # Arithmetic shift would produce 0xC000
        self.assertEqual(got, 0x4000)
        self.assertNotEqual(got, 0xC000)


# =============================================================================
# Unit Tests: Translator Integration (tools/rdna2spirv.py)
# =============================================================================

class TestValuLshrrevB16Translator(unittest.TestCase):
    """End-to-end integration tests exercising tools/rdna2spirv.py."""

    def test_translator_invocation_and_execution(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 4",       # shift count = 4
            "v_mov_b32 v1, 0x1230",  # value to shift
            "v_lshrrev_b16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_lshrrev_b16_test", asm)
        self.assertIn("OpShiftRightLogical", spvasm)

        words = support.mrt0_words(spvasm)
        # 0x1230 >> 4 = 0x0123
        self.assertEqual(words, [0x0123, 0x0123, 0x0123, 0x0123])

    def test_translator_shift_zero_and_large(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 0x0000",
            "v_mov_b32 v1, 0xABCD",
            "v_lshrrev_b16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_lshrrev_b16_zero", asm)
        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [0xABCD, 0xABCD, 0xABCD, 0xABCD])


if __name__ == "__main__":
    unittest.main()
