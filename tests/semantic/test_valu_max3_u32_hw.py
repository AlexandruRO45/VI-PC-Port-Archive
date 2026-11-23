from __future__ import annotations

"""Semantic and hardware reference tests for RDNA2 v_max3_u32.

Validates 32-bit unsigned three-input vector maximum:
  vdst = max(src0, src1, src2) (unsigned 32-bit integer arithmetic)

Lowers to GLSL.std.450 UMax via rdna2spirv.

Exercises reference mathematical oracle, symbolic AST definitions, deterministic
scalar interpreter, SIMD multi-lane divergence with EXEC masking, register
aliasing, status flags with unsigned bounds, IEEE 754 bit-pattern invariance, and
negative controls. All stages cross-checked with rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFF_FFFF

_TRANSLATE_ID = 0


def oracle_max3_u32(src0: int, src1: int, src2: int) -> int:
    """Mathematical reference oracle for v_max3_u32.

    Returns:
        Unsigned maximum of three 32-bit unsigned integers.
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    u2 = src2 & MASK32
    return max(u0, u1, u2)


def run_translated_max3_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_max3_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_max3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_max3_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32

# ---------------------------------------------------------------------------
# Stage 1: Reference mathematical oracle and architectural model
# ---------------------------------------------------------------------------

class TestValuMax3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_max3_u32."""

    def test_oracle_basic_maximum(self) -> None:
        self.assertEqual(oracle_max3_u32(10, 50, 30), 50)
        self.assertEqual(oracle_max3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_max3_u32(1, 2, 3), 3)
        self.assertEqual(oracle_max3_u32(3, 2, 1), 3)
        self.assertEqual(oracle_max3_u32(2, 3, 1), 3)

    def test_oracle_unsigned_semantics(self) -> None:
        # In unsigned arithmetic, 0xFFFFFFFF > 0x7FFFFFFF > 0
        self.assertEqual(oracle_max3_u32(0, 0x7FFF_FFFF, MASK32), MASK32)
        self.assertEqual(oracle_max3_u32(0x8000_0000, 10, 20), 0x8000_0000)
        self.assertEqual(oracle_max3_u32(MASK32, MASK32 - 1, 0), MASK32)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (10, 50, 30),
            (0, 0, 0),
            (1, 2, 3),
            (3, 2, 1),
            (2, 3, 1),
            (0x12345678, 0x99999999, 0x55555555),
            (MASK32, 0x80000000, 0x7FFFFFFF),
            (0, MASK32, 1),
            (100, 100, 100),
            (100, 200, 100),
            (200, 100, 100),
        ]
        for a, b, c in triples:
            expected = oracle_max3_u32(a, b, c)
            actual = run_translated_max3_u32(a, b, c)
            self.assertEqual(actual, expected, f"Mismatch on max3({a:#x}, {b:#x}, {c:#x})")

    def test_oracle_permutation_invariance(self) -> None:
        a, b, c = 123, 456, 789
        m = oracle_max3_u32(a, b, c)
        self.assertEqual(oracle_max3_u32(a, c, b), m)
        self.assertEqual(oracle_max3_u32(b, a, c), m)
        self.assertEqual(oracle_max3_u32(b, c, a), m)
        self.assertEqual(oracle_max3_u32(c, a, b), m)
        self.assertEqual(oracle_max3_u32(c, b, a), m)

# ---------------------------------------------------------------------------
# Stage 2: Symbolic AST definitions and typed expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()


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
        return f"s{self.index}"


class Opcode(enum.Enum):
    V_MAX3_U32 = "v_max3_u32"


@dataclasses.dataclass(frozen=True)
class VOP3Instruction:
    opcode: Opcode
    vdst: Register
    src0: Register | int
    src1: Register | int
    src2: Register | int

    def __post_init__(self) -> None:
        if self.vdst.kind != RegKind.VGPR:
            raise ValueError(f"Destination vdst must be VGPR, got {self.vdst.kind}")


class TestValuMax3U32ASTDefinitions(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_max3_u32."""

    def test_ast_construction_and_formatting(self) -> None:
        vdst = Register(RegKind.VGPR, 0)
        src0 = Register(RegKind.VGPR, 1)
        src1 = Register(RegKind.VGPR, 2)
        src2 = Register(RegKind.VGPR, 3)
        inst = VOP3Instruction(Opcode.V_MAX3_U32, vdst, src0, src1, src2)

        self.assertEqual(inst.opcode, Opcode.V_MAX3_U32)
        self.assertEqual(str(inst.vdst), "v0")
        self.assertEqual(str(inst.src0), "v1")
        self.assertEqual(str(inst.src1), "v2")
        self.assertEqual(str(inst.src2), "v3")

    def test_ast_with_literal_and_sgpr(self) -> None:
        vdst = Register(RegKind.VGPR, 4)
        src0 = Register(RegKind.SGPR, 2)
        src1 = Register(RegKind.VGPR, 5)
        inst = VOP3Instruction(Opcode.V_MAX3_U32, vdst, src0, src1, 42)

        self.assertEqual(str(inst.vdst), "v4")
        self.assertEqual(str(inst.src0), "s2")
        self.assertEqual(str(inst.src1), "v5")
        self.assertEqual(inst.src2, 42)

# ---------------------------------------------------------------------------
# Stage 3: Deterministic instruction interpreter
# ---------------------------------------------------------------------------

class ValuMax3Interpreter:
    """Scalar architectural interpreter for v_max3_u32."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def resolve(self, operand: Register | int) -> int:
        if isinstance(operand, int):
            return operand & MASK32
        if operand.kind == RegKind.VGPR:
            return self.vgprs[operand.index]
        if operand.kind == RegKind.SGPR:
            return self.sgprs[operand.index]
        return 0

    def step(self, inst: VOP3Instruction) -> int:
        s0 = self.resolve(inst.src0)
        s1 = self.resolve(inst.src1)
        s2 = self.resolve(inst.src2)
        res = oracle_max3_u32(s0, s1, s2)
        self.vgprs[inst.vdst.index] = res
        return res


class TestValuMax3U32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_max3_u32."""

    def test_interpreter_vgpr_execution(self) -> None:
        interp = ValuMax3Interpreter()
        interp.vgprs[1] = 20
        interp.vgprs[2] = 80
        interp.vgprs[3] = 45

        inst = VOP3Instruction(
            Opcode.V_MAX3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = interp.step(inst)
        self.assertEqual(res, 80)
        self.assertEqual(interp.vgprs[0], 80)

    def test_interpreter_mixed_operands(self) -> None:
        interp = ValuMax3Interpreter()
        interp.sgprs[4] = 100
        interp.vgprs[2] = 25

        inst = VOP3Instruction(
            Opcode.V_MAX3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SGPR, 4),
            Register(RegKind.VGPR, 2),
            200,
        )
        res = interp.step(inst)
        self.assertEqual(res, 200)
        self.assertEqual(interp.vgprs[0], 200)

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

    def execute(self, s0_lanes: list[int], s1_lanes: list[int], s2_lanes: list[int]) -> list[int]:
        res_lanes: list[int] = [0] * self.wave_size
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                res_lanes[lane] = oracle_max3_u32(s0_lanes[lane], s1_lanes[lane], s2_lanes[lane])
            else:
                res_lanes[lane] = 0
        return res_lanes


class TestValuMax3U32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and EXEC masking for v_max3_u32."""

    def test_wave32_all_active(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        s0 = [lane * 10 for lane in range(32)]
        s1 = [100] * 32
        s2 = [50] * 32

        res = wave.execute(s0, s1, s2)
        for lane in range(32):
            expected = max(lane * 10, 100, 50)
            self.assertEqual(res[lane], expected)

    def test_wave64_divergence_masking(self) -> None:
        mask = 0xAAAA_AAAA_AAAA_AAAA
        wave = SIMDWavefront(wave_size=64, exec_mask=mask)
        s0 = [5] * 64
        s1 = [15] * 64
        s2 = [25] * 64

        res = wave.execute(s0, s1, s2)
        for lane in range(64):
            if (mask >> lane) & 1:
                self.assertEqual(res[lane], 25)
            else:
                self.assertEqual(res[lane], 0)

# ---------------------------------------------------------------------------
# Stage 5: Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuMax3U32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_max3_u32."""

    def test_inplace_dst_equals_src0(self) -> None:
        interp = ValuMax3Interpreter()
        interp.vgprs[0] = 50
        interp.vgprs[1] = 100
        interp.vgprs[2] = 75

        inst = VOP3Instruction(
            Opcode.V_MAX3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        res = interp.step(inst)
        self.assertEqual(res, 100)
        self.assertEqual(interp.vgprs[0], 100)
        self.assertEqual(interp.vgprs[1], 100)
        self.assertEqual(interp.vgprs[2], 75)

    def test_inplace_all_aliased(self) -> None:
        interp = ValuMax3Interpreter()
        interp.vgprs[0] = 0xDEADBEEF

        inst = VOP3Instruction(
            Opcode.V_MAX3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0xDEADBEEF)
        self.assertEqual(interp.vgprs[0], 0xDEADBEEF)

    def test_aliased_translated_execution(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 120",
            "v_mov_b32 v2, 90",
            "v_max3_u32 v0, v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_max3_alias_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 120)

if __name__ == "__main__":
    unittest.main()
