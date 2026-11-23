"""Semantic hardware-level and translator reference tests for AMD RDNA2 v_med3_u32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture & Semantics:
  In AMD RDNA2 (GFX10 / GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  `v_med3_u32` computes the unsigned median of three 32-bit integer values.

  Instruction:
    v_med3_u32 vd, src0, src1, src2

  Mathematical Specification:
    D.u32 = clamp(src2, min(src0, src1), max(src0, src1))
          = sorted([src0, src1, src2])[1] (unsigned 32-bit ordering)

  RDNA2 Hardware Rules:
    - Treats all three operands as unsigned 32-bit integers in [0, 2^32 - 1].
    - Returns the middle value among src0, src1, and src2 under unsigned comparison.
    - If two or three operands are equal, that value is returned.
    - Flags:
        Neither SCC nor VCC is modified by this bitwise/arithmetic instruction.
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
_TRANSLATE_ID: int = 0


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def ref_med3_u32(s0: int, s1: int, s2: int) -> int:
    """Pure bitwise reference mathematical oracle for RDNA2 v_med3_u32.

    Returns the median of three unsigned 32-bit integers.
    """
    v0 = s0 & MASK32
    v1 = s1 & MASK32
    v2 = s2 & MASK32
    return sorted([v0, v1, v2])[1]


def run_translated_med3_u32(s0: int, s1: int, s2: int) -> int:
    """Execute v_med3_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        "v_med3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_med3_u32_trans_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0] & MASK32


def run_translated_med3_u32_vec4(vec: list[tuple[int, int, int]]) -> list[int]:
    """Execute four distinct median evaluations exported to MRT0 (x, y, z, w)."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = []
    assert len(vec) == 4
    for idx, (s0, s1, s2) in enumerate(vec):
        d = idx
        v0 = 4 + idx * 3
        v1 = v0 + 1
        v2 = v0 + 2
        lines.append(f"v_mov_b32 v{v0}, {s0 & MASK32:#x}")
        lines.append(f"v_mov_b32 v{v1}, {s1 & MASK32:#x}")
        lines.append(f"v_mov_b32 v{v2}, {s2 & MASK32:#x}")
        lines.append(f"v_med3_u32 v{d}, v{v0}, v{v1}, v{v2}")
    lines.append("exp mrt0 v0, v1, v2, v3 done vm")
    lines.append("s_endpgm",)
    spv = support.translate_lines(f"ps_med3_u32_vec4_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return [w & MASK32 for w in words[:4]]


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
class VMed3U32Instruction:
    vdst: RegVGPR
    src0: SourceOperand
    src1: SourceOperand
    src2: SourceOperand

    @staticmethod
    def parse(line: str) -> VMed3U32Instruction:
        tokens = line.replace(",", " ").split()
        if not tokens or tokens[0] != "v_med3_u32":
            raise ValueError(f"Invalid opcode for v_med3_u32: {line}")
        if len(tokens) != 5:
            raise ValueError(f"Expected 4 operands for v_med3_u32: {line}")
        vdst = VMed3U32Instruction._parse_reg(tokens[1])
        src0 = VMed3U32Instruction._parse_src(tokens[2])
        src1 = VMed3U32Instruction._parse_src(tokens[3])
        src2 = VMed3U32Instruction._parse_src(tokens[4])
        return VMed3U32Instruction(vdst=vdst, src0=src0, src1=src1, src2=src2)

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
    """Deterministic scalar register file and execution engine for v_med3_u32."""

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

    def execute(self, instr: VMed3U32Instruction) -> int:
        s0 = self.evaluate_src(instr.src0)
        s1 = self.evaluate_src(instr.src1)
        s2 = self.evaluate_src(instr.src2)
        res = ref_med3_u32(s0, s1, s2)
        self.write_u32(instr.vdst, res)
        return res


class SIMDInterpreter:
    """SIMD execution engine supporting 32 and 64 lanes with divergence masking."""

    def __init__(self, num_lanes: int = 64) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count: {num_lanes}")
        self.num_lanes = num_lanes
        self.lanes: List[ScalarInterpreter] = [ScalarInterpreter() for _ in range(num_lanes)]

    def execute_wave(self, instr: VMed3U32Instruction, exec_mask: int) -> List[int]:
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

class TestValuMed3U32MathematicalOracle(unittest.TestCase):
    """Tests covering reference mathematical oracle and unsigned 3-way median rules."""

    def test_permutations_of_three_values(self) -> None:
        # For any permutation of (10, 20, 30), median is 20
        cases = [
            (10, 20, 30),
            (10, 30, 20),
            (20, 10, 30),
            (20, 30, 10),
            (30, 10, 20),
            (30, 20, 10),
        ]
        for a, b, c in cases:
            with self.subTest(a=a, b=b, c=c):
                expected = ref_med3_u32(a, b, c)
                self.assertEqual(expected, 20)
                actual = run_translated_med3_u32(a, b, c)
                self.assertEqual(actual, 20, f"Translator mismatch on permutation ({a}, {b}, {c})")

    def test_unsigned_extremes(self) -> None:
        cases = [
            (0, 100, 0xFFFFFFFF, 100),
            (0xFFFFFFFF, 100, 0, 100),
            (0x80000000, 0x10, 0xFFFFFFFF, 0x80000000),
            (0, 0x80000000, 0xFFFFFFFF, 0x80000000),
            (0x7FFFFFFF, 0x80000000, 0x7FFFFFFE, 0x7FFFFFFF),
        ]
        for a, b, c, exp in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(ref_med3_u32(a, b, c), exp)
                actual = run_translated_med3_u32(a, b, c)
                self.assertEqual(actual, exp, f"Translator mismatch on extremes ({a:#x}, {b:#x}, {c:#x})")

    def test_duplicate_values(self) -> None:
        cases = [
            (5, 5, 10, 5),
            (5, 10, 5, 5),
            (10, 5, 5, 5),
            (5, 10, 10, 10),
            (10, 10, 10, 10),
            (0, 0, 0xFFFFFFFF, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF),
        ]
        for a, b, c, exp in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(ref_med3_u32(a, b, c), exp)
                actual = run_translated_med3_u32(a, b, c)
                self.assertEqual(actual, exp, f"Translator mismatch on duplicates ({a}, {b}, {c})")

    def test_all_zeros_and_all_ones(self) -> None:
        self.assertEqual(ref_med3_u32(0, 0, 0), 0)
        self.assertEqual(run_translated_med3_u32(0, 0, 0), 0)

        self.assertEqual(ref_med3_u32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(run_translated_med3_u32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)


# =============================================================================
# Unit Tests: Symbolic AST & Parser
# =============================================================================

class TestValuMed3U32SymbolicAST(unittest.TestCase):
    """Tests covering AST creation, operand typing, and syntax parsing."""

    def test_instruction_parsing_basic(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        instr = VMed3U32Instruction.parse("v_med3_u32 v3, v0, v1, v2")
        self.assertEqual(instr.vdst.idx, 3)
        self.assertEqual(instr.src0.val, RegVGPR(0))
        self.assertEqual(instr.src1.val, RegVGPR(1))
        self.assertEqual(instr.src2.val, RegVGPR(2))

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 15",
            "v_mov_b32 v1, 45",
            "v_mov_b32 v2, 30",
            "v_med3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_med3_ast_{_TRANSLATE_ID}", lines)
        self.assertIn("UMin", spv)
        self.assertIn("UMax", spv)
        self.assertIn("UClamp", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 30)

    def test_instruction_parsing_constants(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        instr = VMed3U32Instruction.parse("v_med3_u32 v0, 10, 20, 30")
        self.assertEqual(instr.vdst.idx, 0)
        self.assertIsInstance(instr.src0.val, ImmU32)
        self.assertEqual(instr.src0.val.val, 10)
        self.assertIsInstance(instr.src1.val, ImmU32)
        self.assertEqual(instr.src1.val.val, 20)
        self.assertIsInstance(instr.src2.val, ImmU32)
        self.assertEqual(instr.src2.val.val, 30)

        # Translated cross-check
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_med3_u32 v0, v1, v2, 30",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_med3_const_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 20)


# =============================================================================
# Unit Tests: Deterministic Scalar Interpreter
# =============================================================================

class TestValuMed3U32ScalarInterpreter(unittest.TestCase):
    """Tests covering deterministic register execution in a single thread."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_execution(self) -> None:
        self.interp.vgprs[0] = 50
        self.interp.vgprs[1] = 10
        self.interp.vgprs[2] = 30
        instr = VMed3U32Instruction.parse("v_med3_u32 v3, v0, v1, v2")
        got = self.interp.execute(instr)
        self.assertEqual(got, 30)
        self.assertEqual(self.interp.read_u32(RegVGPR(3)), 30)

        # Cross-check through translator
        actual = run_translated_med3_u32(50, 10, 30)
        self.assertEqual(actual, 30)


# =============================================================================
# Unit Tests: SIMD Multi-Lane & EXEC Masking
# =============================================================================

class TestValuMed3U32SIMDDivergence(unittest.TestCase):
    """Tests covering SIMD multi-lane divergence across Wave32 and Wave64."""

    def test_wave32_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=32)
        sentinel = 0xDEADBEEF
        for lane_id in range(32):
            simd.lanes[lane_id].vgprs[0] = lane_id
            simd.lanes[lane_id].vgprs[1] = 10
            simd.lanes[lane_id].vgprs[2] = 20
            simd.lanes[lane_id].vgprs[3] = sentinel

        # Active only on odd lanes
        exec_mask = 0xAAAAAAAA
        instr = VMed3U32Instruction.parse("v_med3_u32 v3, v0, v1, v2")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(32):
            if lane_id % 2 == 1:
                expected = ref_med3_u32(lane_id, 10, 20)
                self.assertEqual(results[lane_id], expected)
                self.assertEqual(simd.lanes[lane_id].read_u32(RegVGPR(3)), expected)
            else:
                self.assertEqual(results[lane_id], sentinel)
                self.assertEqual(simd.lanes[lane_id].read_u32(RegVGPR(3)), sentinel)

        # Cross-check representative active lane 5 (5, 10, 20 -> 10) and lane 25 (25, 10, 20 -> 20)
        self.assertEqual(run_translated_med3_u32(5, 10, 20), 10)
        self.assertEqual(run_translated_med3_u32(25, 10, 20), 20)

    def test_wave64_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=64)
        sentinel = 0xCAFEBABE
        for lane_id in range(64):
            simd.lanes[lane_id].vgprs[0] = lane_id * 2
            simd.lanes[lane_id].vgprs[1] = 50
            simd.lanes[lane_id].vgprs[2] = 100
            simd.lanes[lane_id].vgprs[3] = sentinel

        # Lower 32 lanes active only
        exec_mask = 0x00000000FFFFFFFF
        instr = VMed3U32Instruction.parse("v_med3_u32 v3, v0, v1, v2")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(64):
            if lane_id < 32:
                expected = ref_med3_u32(lane_id * 2, 50, 100)
                self.assertEqual(results[lane_id], expected)
            else:
                self.assertEqual(results[lane_id], sentinel)

        # Cross-check lane 0 (0, 50, 100 -> 50) and lane 30 (60, 50, 100 -> 60)
        self.assertEqual(run_translated_med3_u32(0, 50, 100), 50)
        self.assertEqual(run_translated_med3_u32(60, 50, 100), 60)


# =============================================================================
# Unit Tests: Register Aliasing & Preservation
# =============================================================================

class TestValuMed3U32RegisterAliasing(unittest.TestCase):
    """Tests covering destination overwrites and register aliasing."""

    def test_vdst_equals_src0(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        interp = ScalarInterpreter()
        interp.vgprs[0] = 10
        interp.vgprs[1] = 30
        interp.vgprs[2] = 20
        instr = VMed3U32Instruction.parse("v_med3_u32 v0, v0, v1, v2")
        got = interp.execute(instr)
        self.assertEqual(got, 20)
        self.assertEqual(interp.read_u32(RegVGPR(0)), 20)
        self.assertEqual(interp.read_u32(RegVGPR(1)), 30)
        self.assertEqual(interp.read_u32(RegVGPR(2)), 20)

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 20",
            "v_med3_u32 v0, v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_med3_alias0_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 20)

    def test_vdst_equals_src1(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        interp = ScalarInterpreter()
        interp.vgprs[0] = 80
        interp.vgprs[1] = 10
        interp.vgprs[2] = 40
        instr = VMed3U32Instruction.parse("v_med3_u32 v1, v0, v1, v2")
        got = interp.execute(instr)
        self.assertEqual(got, 40)
        self.assertEqual(interp.read_u32(RegVGPR(1)), 40)

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 80",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 40",
            "v_med3_u32 v1, v0, v1, v2",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_med3_alias1_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 40)

    def test_vdst_equals_src2(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        interp = ScalarInterpreter()
        interp.vgprs[0] = 50
        interp.vgprs[1] = 10
        interp.vgprs[2] = 90
        instr = VMed3U32Instruction.parse("v_med3_u32 v2, v0, v1, v2")
        got = interp.execute(instr)
        self.assertEqual(got, 50)
        self.assertEqual(interp.read_u32(RegVGPR(2)), 50)

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 90",
            "v_med3_u32 v2, v0, v1, v2",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_med3_alias2_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 50)

    def test_all_operands_identical(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        interp = ScalarInterpreter()
        interp.vgprs[0] = 42
        instr = VMed3U32Instruction.parse("v_med3_u32 v0, v0, v0, v0")
        got = interp.execute(instr)
        self.assertEqual(got, 42)
        self.assertEqual(interp.read_u32(RegVGPR(0)), 42)

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 42",
            "v_med3_u32 v0, v0, v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_med3_alias_all_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 42)


# =============================================================================
# Unit Tests: Condition Codes & Flags
# =============================================================================

class TestValuMed3U32ConditionCodes(unittest.TestCase):
    """Tests verifying SCC and VCC flags remain completely isolated."""

    def test_flags_preserved(self) -> None:
        interp = ScalarInterpreter()
        interp.scc = 1
        interp.vcc = 0xDEADBEEFCAFE0000
        interp.vgprs[0] = 10
        interp.vgprs[1] = 20
        interp.vgprs[2] = 30
        instr = VMed3U32Instruction.parse("v_med3_u32 v3, v0, v1, v2")
        interp.execute(instr)
        self.assertEqual(interp.scc, 1)
        self.assertEqual(interp.vcc, 0xDEADBEEFCAFE0000)


# =============================================================================
# Unit Tests: Negative Controls & Mutation Detection
# =============================================================================

class TestValuMed3U32NegativeControls(unittest.TestCase):
    """Negative controls testing syntax rejection and mutation detection."""

    def test_syntax_rejection(self) -> None:
        with self.assertRaises(ValueError):
            VMed3U32Instruction.parse("v_med3_i32 v3, v0, v1, v2")

        with self.assertRaises(ValueError):
            # Missing operand
            VMed3U32Instruction.parse("v_med3_u32 v3, v0, v1")

    def test_mutation_detector_signed_vs_unsigned(self) -> None:
        # Under signed comparison, -1 (0xFFFFFFFF) < 0 < 10, so median is 0
        # Under unsigned comparison, 0 < 10 < 0xFFFFFFFF, so median is 10
        got = ref_med3_u32(0, 10, 0xFFFFFFFF)
        self.assertEqual(got, 10)
        self.assertNotEqual(got, 0)

        # Cross-check through translated execution to ensure rdna2spirv emits UClamp and not SClamp
        trans_got = run_translated_med3_u32(0, 10, 0xFFFFFFFF)
        self.assertEqual(trans_got, 10)
        self.assertNotEqual(trans_got, 0)

    def test_mutation_detector_min_or_max_substitute(self) -> None:
        # Mutation: mistakenly computing min or max instead of median
        got = ref_med3_u32(10, 20, 30)
        self.assertEqual(got, 20)
        self.assertNotEqual(got, 10)
        self.assertNotEqual(got, 30)

        trans_got = run_translated_med3_u32(10, 20, 30)
        self.assertEqual(trans_got, 20)
        self.assertNotEqual(trans_got, 10)
        self.assertNotEqual(trans_got, 30)

    def test_mutation_detector_low_clamp_min_to_max(self) -> None:
        # If low clamp 'min(x, y)' is mutated to 'max(x, y)', the median lowers to:
        # clamp(z, max(x, y), max(x, y)) = max(x, y).
        # Whenever z < max(x, y), the mutant returns max(x, y) instead of median.
        cases = [
            (10, 30, 20, 20, 30),      # median 20, mutant max(10, 30) = 30
            (30, 10, 20, 20, 30),      # median 20, mutant max(30, 10) = 30
            (100, 200, 50, 100, 200),  # median 100, mutant max(100, 200) = 200
            (0, 0xFFFFFFFF, 100, 100, 0xFFFFFFFF),  # median 100, mutant 0xFFFFFFFF
        ]
        for x, y, z, expected_median, mutant_value in cases:
            actual = run_translated_med3_u32(x, y, z)
            self.assertEqual(actual, expected_median)
            self.assertNotEqual(actual, mutant_value)


# =============================================================================
# Unit Tests: Translator Integration & Multi-Vector Boundaries
# =============================================================================

class TestValuMed3U32Translator(unittest.TestCase):
    """End-to-end integration tests exercising tools/rdna2spirv.py."""

    def test_translator_invocation_and_execution(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 20",
            "v_med3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_med3_u32_test", asm)
        self.assertIn("UMin", spvasm)
        self.assertIn("UMax", spvasm)
        self.assertIn("UClamp", spvasm)
        self.assertNotIn("SMin", spvasm)
        self.assertNotIn("SMax", spvasm)
        self.assertNotIn("SClamp", spvasm)

        words = support.mrt0_words(spvasm)
        self.assertEqual(words, [20, 20, 20, 20])

    def test_translator_unsigned_extremes(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_med3_u32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_med3_u32_extremes", asm)
        words = support.mrt0_words(spvasm)
        # 0 < 100 < 0xFFFFFFFF -> median is 100
        self.assertEqual(words, [100, 100, 100, 100])

    def test_translator_multi_vector_median_boundaries(self) -> None:
        """Verify multi-vector 4-lane median evaluation on key numeric boundary conditions."""
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        # Bundle 1: Extreme values and identity cases
        bundle1 = [
            (0, 0, 0),
            (MASK32, MASK32, MASK32),
            (0, MASK32, 1),
            (MASK32, 0, MASK32 - 1),
        ]
        exp1 = [ref_med3_u32(a, b, c) for a, b, c in bundle1]
        self.assertEqual(exp1, [0, MASK32, 1, MASK32 - 1])
        act1 = run_translated_med3_u32_vec4(bundle1)
        self.assertEqual(act1, exp1)

        # Bundle 2: Sign-bit boundary patterns (distinguishing unsigned UClamp vs signed SClamp)
        bundle2 = [
            (0x80000000, 0x7FFFFFFF, 0x80000001),
            (0x7FFFFFFF, 0x80000000, 0),
            (0, 0xFFFFFFFF, 0x80000000),
            (0x80000000, 0, 0xFFFFFFFF),
        ]
        exp2 = [ref_med3_u32(a, b, c) for a, b, c in bundle2]
        self.assertEqual(exp2, [0x80000000, 0x7FFFFFFF, 0x80000000, 0x80000000])
        act2 = run_translated_med3_u32_vec4(bundle2)
        self.assertEqual(act2, exp2)

        # Bundle 3: Lower-clamp sensitivity cases where z < max(x, y)
        bundle3 = [
            (10, 50, 20),
            (50, 10, 20),
            (1000, 2000, 500),
            (0xFFFFFFFF, 100, 50),
        ]
        exp3 = [ref_med3_u32(a, b, c) for a, b, c in bundle3]
        self.assertEqual(exp3, [20, 20, 1000, 100])
        act3 = run_translated_med3_u32_vec4(bundle3)
        self.assertEqual(act3, exp3)

    def test_translator_sgpr_and_literal_operands(self) -> None:
        """Verify v_med3_u32 with scalar SGPRs and literal constants."""
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "s_mov_b32 s0, 200",
            "v_mov_b32 v1, 100",
            "v_med3_u32 v0, s0, v1, 300",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_med3_sgpr_lit_{_TRANSLATE_ID}", lines)
        self.assertIn("UMin", spv)
        self.assertIn("UMax", spv)
        self.assertIn("UClamp", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 200)


if __name__ == "__main__":
    unittest.main()
