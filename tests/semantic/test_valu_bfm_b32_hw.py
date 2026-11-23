"""Semantic hardware-level and translator reference tests for AMD RDNA2 v_bfm_b32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture & Semantics:
  In AMD RDNA2 (GFX10 / GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  `v_bfm_b32` is a 32-bit bitfield mask generation (Bitfield Make) instruction.

  Instruction:
    v_bfm_b32 vd, src0, src1

  Mathematical Specification:
    width = src0 & 31
    offset = src1 & 31
    D.u32 = (((1 << width) - 1) << offset) & 0xFFFFFFFF

  RDNA2 Hardware Rules:
    - src0 is width (bits [4:0]), src1 is offset (bits [4:0]).
    - If width == 0, (1 << 0) - 1 = 0, so the result is 0.
    - If width == 31, (1 << 31) - 1 = 0x7FFFFFFF.
    - Any bits shifted beyond bit 31 are truncated (standard 32-bit unsigned wrapping).
    - Unused higher bits [31:5] of both src0 and src1 are ignored due to the & 31 mask.
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

MASK32: int = 0xFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def ref_bfm_b32(width_raw: int, offset_raw: int) -> int:
    """Pure bitwise reference mathematical oracle for RDNA2 v_bfm_b32.

    Computes:
      w = width_raw & 31
      o = offset_raw & 31
      ((1 << w) - 1) << o (mod 2^32)
    """
    w = width_raw & 31
    o = offset_raw & 31
    mask = (1 << w) - 1
    res = (mask << o) & MASK32
    return res


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
class ImmU32:
    val: int

    def __post_init__(self) -> None:
        if not (0 <= self.val <= MASK32):
            raise ValueError(f"32-bit constant out of range: {self.val}")


@dataclasses.dataclass(frozen=True)
class SourceOperand:
    val: Union[RegVGPR, ImmU32]


@dataclasses.dataclass(frozen=True)
class VBFMB32Instruction:
    vdst: RegVGPR
    src0: SourceOperand  # width
    src1: SourceOperand  # offset

    @staticmethod
    def parse(line: str) -> VBFMB32Instruction:
        tokens = line.replace(",", " ").split()
        if not tokens or tokens[0] != "v_bfm_b32":
            raise ValueError(f"Invalid opcode for v_bfm_b32: {line}")
        if len(tokens) != 4:
            raise ValueError(f"Expected 3 operands for v_bfm_b32: {line}")
        vdst = VBFMB32Instruction._parse_reg(tokens[1])
        src0 = VBFMB32Instruction._parse_src(tokens[2])
        src1 = VBFMB32Instruction._parse_src(tokens[3])
        return VBFMB32Instruction(vdst=vdst, src0=src0, src1=src1)

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
            return SourceOperand(val=ImmU32(val_int & MASK32))


# =============================================================================
# Architectural Interpreter
# =============================================================================

class ScalarInterpreter:
    """Deterministic scalar register file and execution engine for v_bfm_b32."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.vcc: int = 0
        self.scc: int = 0

    def read_u32(self, reg: RegVGPR) -> int:
        return self.vgprs[reg.idx] & MASK32

    def write_u32(self, reg: RegVGPR, val32: int) -> None:
        self.vgprs[reg.idx] = val32 & MASK32

    def evaluate_src(self, src: SourceOperand) -> int:
        if isinstance(src.val, RegVGPR):
            return self.read_u32(src.val)
        elif isinstance(src.val, ImmU32):
            return src.val.val
        else:
            raise TypeError(f"Unknown operand type: {type(src.val)}")

    def execute(self, instr: VBFMB32Instruction) -> int:
        w = self.evaluate_src(instr.src0)
        o = self.evaluate_src(instr.src1)
        res = ref_bfm_b32(w, o)
        self.write_u32(instr.vdst, res)
        return res


class SIMDInterpreter:
    """SIMD execution engine supporting 32 and 64 lanes with divergence masking."""

    def __init__(self, num_lanes: int = 64) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count: {num_lanes}")
        self.num_lanes = num_lanes
        self.lanes: List[ScalarInterpreter] = [ScalarInterpreter() for _ in range(num_lanes)]

    def execute_wave(self, instr: VBFMB32Instruction, exec_mask: int) -> List[int]:
        results: List[int] = []
        for lane_id in range(self.num_lanes):
            lane_active = bool(exec_mask & (1 << lane_id))
            interp = self.lanes[lane_id]
            if lane_active:
                results.append(interp.execute(instr))
            else:
                results.append(interp.read_u32(instr.vdst))
        return results


# =============================================================================
# Unit Tests: Mathematical Oracle & Edge Cases
# =============================================================================

class TestValuBfmB32MathematicalOracle(unittest.TestCase):
    """Tests covering reference mathematical oracle and standard bitfield make rules."""

    def test_basic_masks(self) -> None:
        # width = 4, offset = 2 -> ((1<<4)-1) << 2 = 15 << 2 = 60 (0x3C)
        self.assertEqual(ref_bfm_b32(4, 2), 0x3C)
        # width = 8, offset = 0 -> 0xFF
        self.assertEqual(ref_bfm_b32(8, 0), 0xFF)
        # width = 8, offset = 8 -> 0xFF00
        self.assertEqual(ref_bfm_b32(8, 8), 0xFF00)
        # width = 16, offset = 16 -> 0xFFFF0000
        self.assertEqual(ref_bfm_b32(16, 16), 0xFFFF0000)

    def test_zero_width(self) -> None:
        # (1 << 0) - 1 = 0
        self.assertEqual(ref_bfm_b32(0, 0), 0)
        self.assertEqual(ref_bfm_b32(0, 10), 0)
        self.assertEqual(ref_bfm_b32(0, 31), 0)

    def test_max_width_31(self) -> None:
        # (1 << 31) - 1 = 0x7FFFFFFF
        self.assertEqual(ref_bfm_b32(31, 0), 0x7FFFFFFF)
        # shifted by 1: 0xFFFFFFFE
        self.assertEqual(ref_bfm_b32(31, 1), 0xFFFFFFFE)

    def test_overflow_truncation(self) -> None:
        # width = 16, offset = 24 -> 0xFFFF << 24 = 0xFF000000 (truncated to 32 bits)
        self.assertEqual(ref_bfm_b32(16, 24), 0xFF000000)
        # width = 31, offset = 31 -> (0x7FFFFFFF << 31) & 0xFFFFFFFF = 0x80000000
        self.assertEqual(ref_bfm_b32(31, 31), 0x80000000)

    def test_high_bits_ignored(self) -> None:
        # width 36 & 31 == 4, offset 34 & 31 == 2 -> same as (4, 2)
        self.assertEqual(ref_bfm_b32(36, 34), 0x3C)
        self.assertEqual(ref_bfm_b32(0xFFFFFFFF, 0xFFFFFFFF), ref_bfm_b32(31, 31))


# =============================================================================
# Unit Tests: Symbolic AST & Parser
# =============================================================================

class TestValuBfmB32SymbolicAST(unittest.TestCase):
    """Tests covering AST creation, operand typing, and syntax parsing."""

    def test_instruction_parsing_basic(self) -> None:
        instr = VBFMB32Instruction.parse("v_bfm_b32 v2, v0, v1")
        self.assertEqual(instr.vdst.idx, 2)
        self.assertEqual(instr.src0.val, RegVGPR(0))
        self.assertEqual(instr.src1.val, RegVGPR(1))

    def test_instruction_parsing_constants(self) -> None:
        instr = VBFMB32Instruction.parse("v_bfm_b32 v3, 8, 16")
        self.assertEqual(instr.vdst.idx, 3)
        self.assertIsInstance(instr.src0.val, ImmU32)
        self.assertEqual(instr.src0.val.val, 8)
        self.assertIsInstance(instr.src1.val, ImmU32)
        self.assertEqual(instr.src1.val.val, 16)


# =============================================================================
# Unit Tests: Deterministic Scalar Interpreter
# =============================================================================

class TestValuBfmB32ScalarInterpreter(unittest.TestCase):
    """Tests covering deterministic register execution in a single thread."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_execution(self) -> None:
        self.interp.vgprs[0] = 8
        self.interp.vgprs[1] = 4
        instr = VBFMB32Instruction.parse("v_bfm_b32 v2, v0, v1")
        got = self.interp.execute(instr)
        # 0xFF << 4 = 0x0FF0
        self.assertEqual(got, 0x0FF0)
        self.assertEqual(self.interp.read_u32(RegVGPR(2)), 0x0FF0)


# =============================================================================
# Unit Tests: SIMD Multi-Lane & EXEC Masking
# =============================================================================

class TestValuBfmB32SIMDDivergence(unittest.TestCase):
    """Tests covering SIMD multi-lane divergence across Wave32 and Wave64."""

    def test_wave32_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=32)
        sentinel = 0xDEADBEEF
        for lane_id in range(32):
            simd.lanes[lane_id].vgprs[0] = lane_id  # width = lane_id
            simd.lanes[lane_id].vgprs[1] = 0        # offset = 0
            simd.lanes[lane_id].vgprs[2] = sentinel

        # Active only on even lanes
        exec_mask = 0x55555555
        instr = VBFMB32Instruction.parse("v_bfm_b32 v2, v0, v1")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(32):
            if lane_id % 2 == 0:
                expected = ref_bfm_b32(lane_id, 0)
                self.assertEqual(results[lane_id], expected)
                self.assertEqual(simd.lanes[lane_id].read_u32(RegVGPR(2)), expected)
            else:
                self.assertEqual(results[lane_id], sentinel)
                self.assertEqual(simd.lanes[lane_id].read_u32(RegVGPR(2)), sentinel)

    def test_wave64_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=64)
        sentinel = 0xCAFEBABE
        for lane_id in range(64):
            simd.lanes[lane_id].vgprs[0] = 4
            simd.lanes[lane_id].vgprs[1] = lane_id % 32
            simd.lanes[lane_id].vgprs[2] = sentinel

        # Upper 32 lanes active only
        exec_mask = 0xFFFFFFFF00000000
        instr = VBFMB32Instruction.parse("v_bfm_b32 v2, v0, v1")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(64):
            if lane_id >= 32:
                expected = ref_bfm_b32(4, lane_id % 32)
                self.assertEqual(results[lane_id], expected)
            else:
                self.assertEqual(results[lane_id], sentinel)


# =============================================================================
# Unit Tests: Register Aliasing & Preservation
# =============================================================================

class TestValuBfmB32RegisterAliasing(unittest.TestCase):
    """Tests covering destination overwrites and register aliasing."""

    def test_vdst_equals_src0(self) -> None:
        interp = ScalarInterpreter()
        interp.vgprs[0] = 4
        interp.vgprs[1] = 2
        instr = VBFMB32Instruction.parse("v_bfm_b32 v0, v0, v1")
        got = interp.execute(instr)
        self.assertEqual(got, 0x3C)
        self.assertEqual(interp.read_u32(RegVGPR(0)), 0x3C)
        # Source 1 should be untouched
        self.assertEqual(interp.read_u32(RegVGPR(1)), 2)

    def test_vdst_equals_src1(self) -> None:
        interp = ScalarInterpreter()
        interp.vgprs[0] = 8
        interp.vgprs[1] = 0
        instr = VBFMB32Instruction.parse("v_bfm_b32 v1, v0, v1")
        got = interp.execute(instr)
        self.assertEqual(got, 0xFF)
        self.assertEqual(interp.read_u32(RegVGPR(1)), 0xFF)
        self.assertEqual(interp.read_u32(RegVGPR(0)), 8)

    def test_all_operands_identical(self) -> None:
        interp = ScalarInterpreter()
        interp.vgprs[0] = 4
        instr = VBFMB32Instruction.parse("v_bfm_b32 v0, v0, v0")
        got = interp.execute(instr)
        # width = 4, offset = 4 -> 0xF << 4 = 0xF0
        self.assertEqual(got, 0xF0)
        self.assertEqual(interp.read_u32(RegVGPR(0)), 0xF0)


# =============================================================================
# Unit Tests: Condition Codes & Flags
# =============================================================================

class TestValuBfmB32ConditionCodes(unittest.TestCase):
    """Tests verifying SCC and VCC flags remain completely isolated."""

    def test_flags_preserved(self) -> None:
        interp = ScalarInterpreter()
        interp.scc = 1
        interp.vcc = 0xDEADBEEF01234567
        interp.vgprs[0] = 4
        interp.vgprs[1] = 2
        instr = VBFMB32Instruction.parse("v_bfm_b32 v2, v0, v1")
        interp.execute(instr)
        self.assertEqual(interp.scc, 1)
        self.assertEqual(interp.vcc, 0xDEADBEEF01234567)


# =============================================================================
# Unit Tests: Negative Controls & Mutation Detection
# =============================================================================

class TestValuBfmB32NegativeControls(unittest.TestCase):
    """Negative controls testing syntax rejection and mutation detection."""

    def test_syntax_rejection(self) -> None:
        with self.assertRaises(ValueError):
            VBFMB32Instruction.parse("v_bfi_b32 v2, v0, v1")

        with self.assertRaises(ValueError):
            # Missing operand
            VBFMB32Instruction.parse("v_bfm_b32 v2, v0")

    def test_mutation_detector_missing_mask(self) -> None:
        # Mutation: implementation shifting 1 << 36 without & 31 mask
        got = ref_bfm_b32(36, 2)
        # width 36 & 31 = 4 -> 0x3C
        self.assertEqual(got, 0x3C)

    def test_mutation_detector_zero_width(self) -> None:
        # Mutation: implementation returning all-ones instead of 0 for width 0
        got = ref_bfm_b32(0, 5)
        self.assertEqual(got, 0)
        self.assertNotEqual(got, 0xFFFFFFFF)


# =============================================================================
# Unit Tests: Translator Integration (tools/rdna2spirv.py)
# =============================================================================

class TestValuBfmB32Translator(unittest.TestCase):
    """End-to-end integration tests exercising tools/rdna2spirv.py."""

    def test_translator_invocation_and_execution(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 4",  # width = 4
            "v_mov_b32 v1, 2",  # offset = 2
            "v_bfm_b32 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_bfm_b32_test", asm)
        self.assertIn("OpShiftLeftLogical", spvasm)

        words = support.mrt0_words(spvasm)
        # (1 << 4 - 1) << 2 = 15 << 2 = 60 (0x3C)
        self.assertEqual(words, [0x3C, 0x3C, 0x3C, 0x3C])

    def test_translator_width_8_offset_8(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 8",
            "v_mov_b32 v1, 8",
            "v_bfm_b32 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_bfm_b32_byte1", asm)
        words = support.mrt0_words(spvasm)
        # (1 << 8 - 1) << 8 = 0xFF00
        self.assertEqual(words, [0xFF00, 0xFF00, 0xFF00, 0xFF00])


if __name__ == "__main__":
    unittest.main()
