from __future__ import annotations

"""Semantic and hardware reference tests for RDNA2 v_sub_co_u32.

Validates 32-bit unsigned vector subtraction with carry-out / borrow:
  vdst = (src0 - src1) & 0xFFFFFFFF
  sdst / vcc_lo = 0xFFFFFFFF if (src0 < src1) else 0

Exercises reference mathematical oracle, symbolic AST definitions, deterministic
scalar interpreter, SIMD multi-lane divergence with EXEC masking, register
aliasing, status flags with carry/borrow, IEEE 754 bit-pattern invariance, and
negative controls. All stages cross-checked with rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

_TRANSLATE_ID = 0


def oracle_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Mathematical reference oracle for v_sub_co_u32.

    Returns:
        (diff, borrow) where:
          diff = (src0 - src1) & 0xFFFFFFFF
          borrow = 1 if (src0 & 0xFFFFFFFF) < (src1 & 0xFFFFFFFF) else 0
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    diff = (u0 - u1) & MASK32
    borrow = 1 if u0 < u1 else 0
    return diff, borrow


def run_translated_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Execute v_sub_co_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        "v_sub_co_u32 v0, vcc_lo, v1, v2",
        "v_cndmask_b32 v3, 0, 1, vcc_lo",
        "exp mrt0 v0, v3, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_sub_co_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32, words[1] & 1

# ---------------------------------------------------------------------------
# Stage 1: Reference mathematical oracle and architectural model
# ---------------------------------------------------------------------------

class TestValuSubCoU32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_sub_co_u32."""

    def test_oracle_basic_subtraction(self) -> None:
        cases = [(10, 3), (100, 100), (0, 0), (1000, 400)]
        for a, b in cases:
            expected = oracle_sub_co_u32(a, b)
            actual = run_translated_sub_co_u32(a, b)
            self.assertEqual(actual, expected, f"Mismatch on {a} - {b}")

    def test_oracle_borrow_subtraction(self) -> None:
        cases = [(3, 10), (0, 1), (0, MASK32), (MASK32, 0)]
        for a, b in cases:
            expected = oracle_sub_co_u32(a, b)
            actual = run_translated_sub_co_u32(a, b)
            self.assertEqual(actual, expected, f"Borrow mismatch on {a} - {b}")

    def test_oracle_matches_translated_execution(self) -> None:
        pairs = [
            (10, 3),
            (3, 10),
            (0, 0),
            (0, 1),
            (1, 0),
            (MASK32, 1),
            (1, MASK32),
            (0x12345678, 0x11111111),
            (0x11111111, 0x12345678),
            (0x80000000, 1),
            (0x7FFFFFFF, 0x80000000),
            (0xAAAAAAAA, 0x55555555),
            (0x55555555, 0xAAAAAAAA),
        ]
        for a, b in pairs:
            expected = oracle_sub_co_u32(a, b)
            actual = run_translated_sub_co_u32(a, b)
            self.assertEqual(actual, expected, f"Mismatch on {a:#x} - {b:#x}")

# ---------------------------------------------------------------------------
# Stage 2: Symbolic AST definitions and typed expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()
    SPECIAL = enum.auto()


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index < 256):
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.kind == RegKind.SGPR and not (0 <= self.index < 106):
            raise ValueError(f"SGPR index out of range: {self.index}")

    def __str__(self) -> str:
        if self.name:
            return self.name
        if self.kind == RegKind.VGPR:
            return f"v{self.index}"
        if self.kind == RegKind.SGPR:
            return f"s{self.index}"
        return f"special_{self.index}"


class Opcode(enum.Enum):
    V_SUB_CO_U32 = "v_sub_co_u32"


@dataclasses.dataclass(frozen=True)
class VOP3COInstruction:
    opcode: Opcode
    vdst: Register
    sdst: Register
    src0: Register | int
    src1: Register | int

    def __post_init__(self) -> None:
        if self.vdst.kind != RegKind.VGPR:
            raise ValueError(f"Destination vdst must be VGPR, got {self.vdst.kind}")
        if self.sdst.kind not in (RegKind.SGPR, RegKind.SPECIAL):
            raise ValueError(f"Destination sdst must be SGPR or SPECIAL, got {self.sdst.kind}")


class TestValuSubCoU32ASTDefinitions(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_sub_co_u32."""

    def test_ast_construction_and_formatting(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        vdst = Register(RegKind.VGPR, 0)
        sdst = Register(RegKind.SPECIAL, 0, "vcc_lo")
        src0 = Register(RegKind.VGPR, 1)
        src1 = Register(RegKind.VGPR, 2)
        inst = VOP3COInstruction(Opcode.V_SUB_CO_U32, vdst, sdst, src0, src1)

        self.assertEqual(inst.opcode, Opcode.V_SUB_CO_U32)
        self.assertEqual(str(inst.vdst), "v0")
        self.assertEqual(str(inst.sdst), "vcc_lo")
        self.assertEqual(str(inst.src0), "v1")
        self.assertEqual(str(inst.src1), "v2")

        # Verify through translator
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 3",
            f"{inst.opcode.value} {inst.vdst}, {inst.sdst}, {inst.src0}, {inst.src1}",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_co_ast_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 7)

    def test_ast_with_literal_and_sgpr(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        vdst = Register(RegKind.VGPR, 5)
        sdst = Register(RegKind.SGPR, 4)
        src0 = Register(RegKind.SGPR, 2)
        inst = VOP3COInstruction(Opcode.V_SUB_CO_U32, vdst, sdst, src0, 42)

        self.assertEqual(str(inst.vdst), "v5")
        self.assertEqual(str(inst.sdst), "s4")
        self.assertEqual(str(inst.src0), "s2")
        self.assertEqual(inst.src1, 42)

        # Verify through translator
        lines = [
            "s_mov_b32 s2, 50",
            f"{inst.opcode.value} {inst.vdst}, {inst.sdst}, {inst.src0}, {inst.src1}",
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_co_ast_lit_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 8)

# ---------------------------------------------------------------------------
# Stage 3: Deterministic instruction interpreter
# ---------------------------------------------------------------------------

class ValuSubCoInterpreter:
    """Scalar architectural interpreter for v_sub_co_u32."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.vcc_lo = 0

    def resolve(self, operand: Register | int) -> int:
        if isinstance(operand, int):
            return operand & MASK32
        if operand.kind == RegKind.VGPR:
            return self.vgprs[operand.index]
        if operand.kind == RegKind.SGPR:
            return self.sgprs[operand.index]
        if operand.name == "vcc_lo":
            return self.vcc_lo
        return 0

    def step(self, inst: VOP3COInstruction) -> tuple[int, int]:
        s0 = self.resolve(inst.src0)
        s1 = self.resolve(inst.src1)
        diff, borrow = oracle_sub_co_u32(s0, s1)
        self.vgprs[inst.vdst.index] = diff
        borrow_mask = MASK32 if borrow else 0
        if inst.sdst.kind == RegKind.SPECIAL and inst.sdst.name == "vcc_lo":
            self.vcc_lo = borrow_mask
        elif inst.sdst.kind == RegKind.SGPR:
            self.sgprs[inst.sdst.index] = borrow_mask
        return diff, borrow


class TestValuSubCoU32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_sub_co_u32."""

    def test_interpreter_vgpr_subtraction(self) -> None:
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 50
        interp.vgprs[2] = 20

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 30)
        self.assertEqual(borrow, 0)
        self.assertEqual(interp.vgprs[0], 30)
        self.assertEqual(interp.vcc_lo, 0)

        # Cross-check through translator
        t_diff, t_borrow = run_translated_sub_co_u32(50, 20)
        self.assertEqual(t_diff, diff)
        self.assertEqual(t_borrow, borrow)

    def test_interpreter_borrow_into_sgpr(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 10
        interp.vgprs[2] = 25

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 3),
            Register(RegKind.SGPR, 8),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, (10 - 25) & MASK32)
        self.assertEqual(borrow, 1)
        self.assertEqual(interp.vgprs[3], (10 - 25) & MASK32)
        self.assertEqual(interp.sgprs[8], MASK32)

        # Cross-check with translated execution routing borrow into SGPR s8
        lines = [
            "s_mov_b32 s8, 0",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 25",
            "v_sub_co_u32 v3, s8, v1, v2",
            "v_cndmask_b32 v4, 0, 1, s8",
            "exp mrt0 v3, v4, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_sgpr_borrow_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, diff)
        self.assertEqual(words[1] & 1, 1)

    def test_interpreter_literal_subtraction(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 100

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 1),
            30,
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 70)
        self.assertEqual(borrow, 0)

        # Cross-check with translated execution
        lines = [
            "v_mov_b32 v1, 100",
            "v_sub_co_u32 v0, vcc_lo, v1, 30",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_lit_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 70)

# ---------------------------------------------------------------------------
# Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking
# ---------------------------------------------------------------------------

class SIMDWavefront:
    """SIMD Wavefront multi-lane model with 32-bit or 64-bit EXEC masking."""

    def __init__(self, wave_size: int = 32, exec_mask: int | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.mask_limit = (1 << wave_size) - 1
        self.exec_mask = self.mask_limit if exec_mask is None else (exec_mask & self.mask_limit)

    def execute(self, src0_lanes: list[int], src1_lanes: list[int]) -> tuple[list[int], int]:
        diff_lanes: list[int] = [0] * self.wave_size
        vcc_mask = 0

        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                diff, borrow = oracle_sub_co_u32(src0_lanes[lane], src1_lanes[lane])
                diff_lanes[lane] = diff
                if borrow:
                    vcc_mask |= (1 << lane)
            else:
                diff_lanes[lane] = 0

        return diff_lanes, vcc_mask


class TestValuSubCoU32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and EXEC masking for v_sub_co_u32."""

    def test_wave32_all_active(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        s0 = [lane * 10 for lane in range(32)]
        s1 = [5 for _ in range(32)]

        diffs, vcc = wave.execute(s0, s1)
        # Lane 0: 0 - 5 => borrow
        self.assertEqual(diffs[0], (0 - 5) & MASK32)
        self.assertEqual(vcc & 1, 1)
        # Lane 1+: >= 10 - 5 => no borrow
        for lane in range(1, 32):
            self.assertEqual(diffs[lane], lane * 10 - 5)
            self.assertEqual((vcc >> lane) & 1, 0)

        # Cross-check lane 0 and lane 31 through translated execution
        t0_diff, t0_borrow = run_translated_sub_co_u32(s0[0], s1[0])
        self.assertEqual(t0_diff, diffs[0])
        self.assertEqual(t0_borrow, 1)

        t31_diff, t31_borrow = run_translated_sub_co_u32(s0[31], s1[31])
        self.assertEqual(t31_diff, diffs[31])
        self.assertEqual(t31_borrow, 0)

    def test_wave64_divergence_masking(self) -> None:
        # Alternating even lanes enabled
        mask = 0x5555_5555_5555_5555
        wave = SIMDWavefront(wave_size=64, exec_mask=mask)
        s0 = [0] * 64
        s1 = [1] * 64

        diffs, vcc = wave.execute(s0, s1)
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(diffs[lane], MASK32)
                self.assertEqual((vcc >> lane) & 1, 1)
            else:
                self.assertEqual(diffs[lane], 0)
                self.assertEqual((vcc >> lane) & 1, 0)
        self.assertEqual(vcc, mask)

        # Cross-check active lane through translator
        t_diff, t_borrow = run_translated_sub_co_u32(0, 1)
        self.assertEqual(t_diff, MASK32)
        self.assertEqual(t_borrow, 1)

# ---------------------------------------------------------------------------
# Stage 5: Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuSubCoU32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_sub_co_u32."""

    def test_inplace_dst_equals_src0(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # v0 = v0 - v1
        interp = ValuSubCoInterpreter()
        interp.vgprs[0] = 100
        interp.vgprs[1] = 40

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 60)
        self.assertEqual(borrow, 0)
        self.assertEqual(interp.vgprs[0], 60)
        self.assertEqual(interp.vgprs[1], 40)

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 40",
            "v_sub_co_u32 v0, vcc_lo, v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_co_alias_s0_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 60)

    def test_inplace_dst_equals_src1(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # v1 = v0 - v1
        interp = ValuSubCoInterpreter()
        interp.vgprs[0] = 50
        interp.vgprs[1] = 80

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 1),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, (50 - 80) & MASK32)
        self.assertEqual(borrow, 1)
        self.assertEqual(interp.vgprs[1], (50 - 80) & MASK32)
        self.assertEqual(interp.vgprs[0], 50)

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 80",
            "v_sub_co_u32 v1, vcc_lo, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v1, v3, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_co_alias_s1_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, (50 - 80) & MASK32)
        self.assertEqual(words[1] & 1, 1)

    def test_inplace_all_aliased(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # v0 = v0 - v0
        interp = ValuSubCoInterpreter()
        interp.vgprs[0] = 0xDEADBEEF

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 0),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 0)
        self.assertEqual(borrow, 0)
        self.assertEqual(interp.vgprs[0], 0)

        # Translated cross-check
        lines = [
            "v_mov_b32 v0, 0xdeadbeef",
            "v_sub_co_u32 v0, vcc_lo, v0, v0",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_co_alias_all_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0)
        self.assertEqual(words[1] & 1, 0)

    def test_aliased_translated_execution(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 35",
            "v_sub_co_u32 v0, vcc_lo, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_co_alias_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 65)
        self.assertEqual(words[1] & 1, 0)

# ---------------------------------------------------------------------------
# Stage 6: Condition codes, status flags, and carry/borrow generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SubStatusFlags:
    zero: bool
    negative: bool
    borrow: bool
    overflow: bool


def compute_sub_status_flags(src0: int, src1: int) -> SubStatusFlags:
    """Compute ALU status flags for 32-bit subtraction src0 - src1."""
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    diff = (u0 - u1) & MASK32
    zero = (diff == 0)
    negative = bool((diff >> 31) & 1)
    borrow = (u0 < u1)
    # Signed overflow: operands have different signs and result sign differs from src0
    overflow = bool(((u0 ^ u1) & (u0 ^ diff) & 0x8000_0000) != 0)
    return SubStatusFlags(zero=zero, negative=negative, borrow=borrow, overflow=overflow)


class TestValuSubCoU32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and carry/borrow for v_sub_co_u32."""

    def test_zero_flag(self) -> None:
        flags = compute_sub_status_flags(0x1234, 0x1234)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.overflow)

        # Cross-check through translator
        diff, borrow = run_translated_sub_co_u32(0x1234, 0x1234)
        self.assertEqual(diff, 0)
        self.assertEqual(borrow, 0)

    def test_negative_and_borrow(self) -> None:
        flags = compute_sub_status_flags(5, 10)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.overflow)

        # Cross-check through translator
        diff, borrow = run_translated_sub_co_u32(5, 10)
        self.assertEqual(diff, (5 - 10) & MASK32)
        self.assertEqual(borrow, 1)

    def test_signed_overflow_without_unsigned_borrow(self) -> None:
        # 0x8000_0000 (-2147483648) - 1 => 0x7FFF_FFFF (+2147483647)
        # Unsigned: 0x8000_0000 >= 1 => no borrow
        # Signed: -2147483648 - 1 => overflow
        flags = compute_sub_status_flags(0x8000_0000, 1)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.borrow)
        self.assertTrue(flags.overflow)

        # Cross-check through translator
        diff, borrow = run_translated_sub_co_u32(0x8000_0000, 1)
        self.assertEqual(diff, 0x7FFF_FFFF)
        self.assertEqual(borrow, 0)

    def test_multi_precision_64bit_borrow_chain(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # Compute 64-bit subtraction using two 32-bit steps:
        # A = 0x0000_0002_0000_0001, B = 0x0000_0001_0000_0005
        # Lower: 1 - 5 = 0xFFFF_FFFC with borrow = 1
        # Upper: 2 - 1 - borrow = 0 with borrow = 0
        a_lo, a_hi = 1, 2
        b_lo, b_hi = 5, 1

        diff_lo, borrow_lo = oracle_sub_co_u32(a_lo, b_lo)
        self.assertEqual(diff_lo, 0xFFFF_FFFC)
        self.assertEqual(borrow_lo, 1)

        diff_hi, borrow_hi = oracle_sub_co_u32(a_hi - borrow_lo, b_hi)
        self.assertEqual(diff_hi, 0)
        self.assertEqual(borrow_hi, 0)

        res64 = (diff_hi << 32) | diff_lo
        expected64 = (((a_hi << 32) | a_lo) - ((b_hi << 32) | b_lo)) & MASK64
        self.assertEqual(res64, expected64)

        # Verify through translator-backed multi-precision subtraction chain
        lines = [
            f"v_mov_b32 v0, {a_lo}",
            f"v_mov_b32 v1, {a_hi}",
            f"v_mov_b32 v2, {b_lo}",
            f"v_mov_b32 v3, {b_hi}",
            "v_sub_co_u32 v4, vcc_lo, v0, v2",
            "v_sub_co_ci_u32 v5, vcc_lo, v1, v3, vcc_lo",
            "exp mrt0 v4, v5, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_chain_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0xFFFF_FFFC)
        self.assertEqual(words[1] & MASK32, 0)

# ---------------------------------------------------------------------------
# Stage 7: IEEE 754 boundary extremes, integer boundaries, and bitwise limits
# ---------------------------------------------------------------------------

class TestValuSubCoU32IEEE754EdgeCases(unittest.TestCase):
    """Stage 7: Boundary extremes, integer limits, and bit-pattern invariance for v_sub_co_u32."""

    def test_integer_boundary_extremes(self) -> None:
        boundary_cases = [
            (MASK32, MASK32, 0, 0),
            (MASK32, 0, MASK32, 0),
            (0, MASK32, 1, 1),
            (0x8000_0000, 1, 0x7FFF_FFFF, 0),
            (0x7FFF_FFFF, 0x8000_0000, MASK32, 1),
        ]
        for src0, src1, exp_diff, exp_borrow in boundary_cases:
            self.assertEqual(oracle_sub_co_u32(src0, src1), (exp_diff, exp_borrow))
            # Cross-check through translator
            t_diff, t_borrow = run_translated_sub_co_u32(src0, src1)
            self.assertEqual(t_diff, exp_diff, f"Diff mismatch for {src0:#x} - {src1:#x}")
            self.assertEqual(t_borrow, exp_borrow, f"Borrow mismatch for {src0:#x} - {src1:#x}")

    def test_float_bitpatterns_treated_as_integers(self) -> None:
        pos_inf = struct.unpack(">I", struct.pack(">f", float("inf")))[0]
        neg_inf = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]
        qnan = struct.unpack(">I", struct.pack(">f", float("nan")))[0]

        res = oracle_sub_co_u32(pos_inf, neg_inf)
        expected = ((pos_inf - neg_inf) & MASK32, 1 if pos_inf < neg_inf else 0)
        self.assertEqual(res, expected)

        res_trans = run_translated_sub_co_u32(pos_inf, neg_inf)
        self.assertEqual(res_trans, expected)

        # qnan and 0
        res_nan = oracle_sub_co_u32(qnan, 0)
        self.assertEqual(res_nan, (qnan, 0))
        self.assertEqual(run_translated_sub_co_u32(qnan, 0), (qnan, 0))

# ---------------------------------------------------------------------------
# Stage 8: Negative controls, syntax rejection, and mutation detection
# ---------------------------------------------------------------------------

class TestValuSubCoU32NegativeControls(unittest.TestCase):
    """Stage 8: Negative controls, syntax rejection, and mutation detection for v_sub_co_u32."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106)

    def test_invalid_destination_kind(self) -> None:
        with self.assertRaises(ValueError):
            VOP3COInstruction(
                Opcode.V_SUB_CO_U32,
                Register(RegKind.SGPR, 0),
                Register(RegKind.SPECIAL, 0, "vcc_lo"),
                Register(RegKind.VGPR, 1),
                Register(RegKind.VGPR, 2),
            )

    def test_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=128)

    def test_mutation_detection_and_oracle_rejection(self) -> None:
        a, b = 15, 25
        correct = oracle_sub_co_u32(a, b)

        def mutated_oracle_inverted_borrow(x: int, y: int) -> tuple[int, int]:
            return (x - y) & MASK32, 1 if x > y else 0

        def mutated_oracle_reversed_sub(x: int, y: int) -> tuple[int, int]:
            return (y - x) & MASK32, 1 if y < x else 0

        self.assertNotEqual(mutated_oracle_inverted_borrow(a, b), correct)
        self.assertNotEqual(mutated_oracle_reversed_sub(a, b), correct)

    def test_translator_compliance_guard(self) -> None:
        test_path = os.path.join(HERE, "test_valu_sub_co_u32_hw.py")
        self.assertTrue(os.path.isfile(test_path))
        with open(test_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("import support", content)
        self.assertIn("support.translate_lines", content)

# ---------------------------------------------------------------------------
# Stage 9: Translator lowering, SPIR-V opcode assertions, and modifier verification
# ---------------------------------------------------------------------------

class TestValuSubCoU32TranslatorLowering(unittest.TestCase):
    """Stage 9: Direct translator lowering tests asserting OpISubBorrow generation and modifiers."""

    def test_translator_emits_op_isub_borrow(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 5",
            "v_sub_co_u32 v0, vcc_lo, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_lowering_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        self.assertNotIn("OpIAddCarry", spv)
        self.assertIn("OpCompositeExtract %uint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 15)

    def test_translator_clamp_modifier_saturates_to_zero_on_borrow(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # 10 - 20 with clamp => saturates to 0, vcc_lo borrow is 1
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_sub_co_u32 v0, vcc_lo, v1, v2 clamp",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_clamp_borrow_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0, "Clamped subtraction with borrow must saturate to 0")
        self.assertEqual(words[1] & 1, 1, "Carry/borrow flag must still reflect true borrow under clamp")

    def test_translator_clamp_modifier_without_borrow(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # 30 - 12 with clamp => 18, vcc_lo borrow is 0
        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 12",
            "v_sub_co_u32 v0, vcc_lo, v1, v2 clamp",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_clamp_noborrow_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 18)
        self.assertEqual(words[1] & 1, 0)

    def test_translator_null_sdst(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # When sdst is 'null', borrow is discarded and only vdst is written
        lines = [
            "v_mov_b32 v1, 45",
            "v_mov_b32 v2, 15",
            "v_sub_co_u32 v0, null, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_null_sdst_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 30)

    def test_translator_sgpr_src0_and_literal_src1(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "s_mov_b32 s2, 75",
            "v_sub_co_u32 v0, vcc_lo, s2, 25",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_sgpr_src_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 50)
        self.assertEqual(words[1] & 1, 0)

if __name__ == "__main__":
    unittest.main()
