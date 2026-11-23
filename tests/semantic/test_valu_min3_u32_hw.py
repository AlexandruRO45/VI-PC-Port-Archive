from __future__ import annotations

"""Semantic reference and translator regression suite for RDNA2 v_min3_u32.

Tests v_min3_u32 (three-operand 32-bit unsigned integer minimum) against mathematical reference
semantics, SIMD divergence execution, and translated SPIR-V shaders via rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def oracle_min3_u32(a: int, b: int, c: int) -> int:
    """Reference mathematical oracle for RDNA2 v_min3_u32: min(unsigned a, unsigned b, unsigned c)."""
    return min(u32(a), u32(b), u32(c))


_TRANSLATE_ID: int = 0


def run_translated_min3_u32(src0: int, src1: int, src2: int, name: Optional[str] = None) -> int:
    """Execute v_min3_u32 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    prog_name = name or f"ps_min3_trans_{_TRANSLATE_ID}"
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_min3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(prog_name, lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


class TestValuMin3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_min3_u32."""

    def test_oracle_basic_minimum(self) -> None:
        cases = [
            (10, 20, 30, 10),
            (30, 10, 20, 10),
            (20, 30, 10, 10),
            (0, 0, 0, 0),
            (1, 2, 3, 1),
            (100, 200, 300, 100),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(oracle_min3_u32(a, b, c), expected)
                hw = run_translated_min3_u32(a, b, c)
                self.assertEqual(hw, expected)

    def test_oracle_unsigned_vs_signed_divergence(self) -> None:
        # In signed math, 0xFFFFFFFF (-1) < 10, but in unsigned math 0xFFFFFFFF is 4294967295 > 10.
        self.assertEqual(oracle_min3_u32(MASK32, 10, 20), 10)
        hw1 = run_translated_min3_u32(MASK32, 10, 20)
        self.assertEqual(hw1, 10)
        self.assertNotEqual(hw1, MASK32)  # SMin would return MASK32 (-1)

        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x7FFF_FFFF, 0), 0)
        hw2 = run_translated_min3_u32(0x8000_0000, 0x7FFF_FFFF, 0)
        self.assertEqual(hw2, 0)

        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x9000_0000, MASK32), 0x8000_0000)
        hw3 = run_translated_min3_u32(0x8000_0000, 0x9000_0000, MASK32)
        self.assertEqual(hw3, 0x8000_0000)
        self.assertNotEqual(hw3, MASK32)  # SMin would return MASK32 (-1)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (0, 0, 0),
            (10, 20, 30),
            (1000, 2000, 3000),
            (0x12345678, 0x11111111, 0x22222222),
            (MASK32, 10, 20),
            (MASK32, MASK32, 1),
            (0x7FFFFFFF, 1, 0x80000000),
            (0xAAAAAAAA, 0x55555555, 1),
            (0, MASK32, 0x80000000),
            (MASK32, 0, MASK32),
            (MASK32, MASK32, MASK32),
            (1, 1, 1),
        ]
        for a, b, c in triples:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                expected = oracle_min3_u32(a, b, c)
                actual = run_translated_min3_u32(a, b, c)
                self.assertEqual(actual, expected)

    def test_oracle_permutations_and_idempotence(self) -> None:
        a, b, c = 0x1234, 0x5678, 0x9ABC
        res1 = run_translated_min3_u32(a, b, c)
        res2 = run_translated_min3_u32(c, a, b)
        res3 = run_translated_min3_u32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)
        self.assertEqual(res1, a)

        hw_same = run_translated_min3_u32(a, a, a)
        self.assertEqual(hw_same, a)


class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
        if self.kind == RegKind.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of range [0, 105]")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


class Opcode(enum.Enum):
    V_MIN3_U32 = "v_min3_u32"


@dataclasses.dataclass
class VOP3Instruction:
    opcode: Opcode
    dst: Register
    src0: Union[Register, int]
    src1: Union[Register, int]
    src2: Union[Register, int]

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VOP3 destination must be VGPR")
        if self.opcode != Opcode.V_MIN3_U32:
            raise ValueError(f"Invalid opcode {self.opcode}")

    def to_asm(self) -> str:
        s0 = f"{self.src0:#x}" if isinstance(self.src0, int) else str(self.src0)
        s1 = f"{self.src1:#x}" if isinstance(self.src1, int) else str(self.src1)
        s2 = f"{self.src2:#x}" if isinstance(self.src2, int) else str(self.src2)
        return f"{self.opcode.value} {self.dst}, {s0}, {s1}, {s2}"


class TestValuMin3U32AST(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_min3_u32."""

    def test_instruction_creation_and_disasm(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        s0 = Register(RegKind.VGPR, 1)
        s1 = Register(RegKind.VGPR, 2)
        s2 = Register(RegKind.VGPR, 3)
        inst = VOP3Instruction(Opcode.V_MIN3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_min3_u32 v0, v1, v2, v3")

        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 10",
            "v_mov_b32 v3, 20",
            inst.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_disasm", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 10)

    def test_instruction_with_literals_and_sgprs(self) -> None:
        dst = Register(RegKind.VGPR, 4)
        s0 = Register(RegKind.SGPR, 0)
        s1 = 0x100
        s2 = Register(RegKind.VGPR, 5)
        inst = VOP3Instruction(Opcode.V_MIN3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_min3_u32 v4, s0, 0x100, v5")

        lines = [
            "s_mov_b32 s0, 200",
            "v_mov_b32 v5, 300",
            inst.to_asm(),
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_sgpr_lit", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 200)

    def test_ast_emits_valid_spirv(self) -> None:
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 200",
            inst.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_min3", lines)
        self.assertIn("UMin", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)


class ScalarMachineState:
    """Architectural state for executing VALU scalar instructions."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[Register, int]) -> int:
        if isinstance(op, int):
            return op & MASK32
        if op.kind == RegKind.VGPR:
            return self.vgprs[op.index]
        return self.sgprs[op.index]

    def write_reg(self, reg: Register, val: int) -> None:
        val &= MASK32
        if reg.kind == RegKind.VGPR:
            self.vgprs[reg.index] = val
        else:
            self.sgprs[reg.index] = val

    def step(self, inst: VOP3Instruction) -> int:
        val0 = self.read_operand(inst.src0)
        val1 = self.read_operand(inst.src1)
        val2 = self.read_operand(inst.src2)
        res = oracle_min3_u32(val0, val1, val2)
        self.write_reg(inst.dst, res)
        return res


class TestValuMin3U32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_min3_u32."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_scalar_step_basic(self) -> None:
        self.state.vgprs[1] = 50
        self.state.vgprs[2] = 20
        self.state.vgprs[3] = 80
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 20)
        self.assertEqual(self.state.vgprs[0], 20)

        hw = run_translated_min3_u32(50, 20, 80)
        self.assertEqual(hw, 20)

    def test_scalar_step_sgpr_and_imm(self) -> None:
        self.state.sgprs[4] = 100
        self.state.vgprs[1] = 50
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SGPR, 4),
            25,
            Register(RegKind.VGPR, 1),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 25)
        self.assertEqual(self.state.vgprs[0], 25)

        lines = [
            "s_mov_b32 s4, 100",
            "v_mov_b32 v1, 50",
            "v_min3_u32 v0, s4, 25, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_sgpr_imm", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 25)

    def test_scalar_matches_translated_shader(self) -> None:
        test_cases = [
            (10, 20, 30),
            (100, 20, 300),
            (0xFFFF_0000, 0x0000_FFFF, 0x1234),
            (0x8000_0000, 0x4000_0000, 0x7000_0000),
            (0, 0, 0),
            (MASK32, MASK32, 0),
        ]
        for a, b, c in test_cases:
            self.state.vgprs[1] = a
            self.state.vgprs[2] = b
            self.state.vgprs[3] = c
            inst = VOP3Instruction(
                Opcode.V_MIN3_U32,
                Register(RegKind.VGPR, 0),
                Register(RegKind.VGPR, 1),
                Register(RegKind.VGPR, 2),
                Register(RegKind.VGPR, 3),
            )
            scalar_res = self.state.step(inst)
            trans_res = run_translated_min3_u32(a, b, c)
            self.assertEqual(scalar_res, trans_res)


class SIMDWavefront:
    """Wavefront model simulating 32-lane or 64-lane SIMD execution."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"wave_size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]

    def read_operand(self, op: Union[Register, int], lane: int) -> int:
        if isinstance(op, int):
            return op & MASK32
        return self.vgprs[op.index][lane]

    def write_vgpr(self, reg: Register, lane: int, val: int) -> None:
        self.vgprs[reg.index][lane] = val & MASK32

    def step(self, inst: VOP3Instruction) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                v0 = self.read_operand(inst.src0, lane)
                v1 = self.read_operand(inst.src1, lane)
                v2 = self.read_operand(inst.src2, lane)
                res = oracle_min3_u32(v0, v1, v2)
                self.write_vgpr(inst.dst, lane, res)


class TestValuMin3U32SIMD(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for v_min3_u32."""

    def test_simd32_all_lanes_active(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        for lane in range(32):
            wave.vgprs[1][lane] = 100 + lane
            wave.vgprs[2][lane] = 50 + lane * 2
            wave.vgprs[3][lane] = 200 - lane
        wave.step(inst)
        for lane in range(32):
            expected = min(100 + lane, 50 + lane * 2, 200 - lane)
            self.assertEqual(wave.vgprs[0][lane], expected)

        # Cross-check through translator
        hw = run_translated_min3_u32(100, 50, 200)
        self.assertEqual(hw, 50)

    def test_simd64_divergence_exec_mask(self) -> None:
        wave = SIMDWavefront(wave_size=64)
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        wave.exec_mask = 0x5555_5555_5555_5555  # even lanes active
        for lane in range(64):
            wave.vgprs[0][lane] = 0xCAFE_BABE
            wave.vgprs[1][lane] = 10
            wave.vgprs[2][lane] = 20
            wave.vgprs[3][lane] = 30
        wave.step(inst)
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[0][lane], 10)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0xCAFE_BABE)

        hw = run_translated_min3_u32(10, 20, 30)
        self.assertEqual(hw, 10)

    def test_simd_inactive_lanes_unmodified(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        wave.exec_mask = 0x0000_0001  # only lane 0 active
        for lane in range(32):
            wave.vgprs[0][lane] = 0x1111_1111 * lane
            wave.vgprs[1][lane] = 5
            wave.vgprs[2][lane] = 15
            wave.vgprs[3][lane] = 25
        wave.step(inst)
        self.assertEqual(wave.vgprs[0][0], 5)
        for lane in range(1, 32):
            self.assertEqual(wave.vgprs[0][lane], 0x1111_1111 * lane)

        hw = run_translated_min3_u32(5, 15, 25)
        self.assertEqual(hw, 5)

    def test_simd_translator_active_mask_preservation(self) -> None:
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        lines = [
            "v_mov_b32 v1, 15",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 25",
            inst.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_simd_min3", lines)
        self.assertIn("OpExtInst", spv)
        self.assertIn("UMin", spv)
        self.assertEqual(support.mrt0_words(spv)[0], 5)


class TestValuMin3U32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_min3_u32."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_in_place_dst_equals_src0(self) -> None:
        self.state.vgprs[1] = 100
        self.state.vgprs[2] = 20
        self.state.vgprs[3] = 300
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 20)
        self.assertEqual(self.state.vgprs[1], 20)

        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 300",
            "v_min3_u32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src0", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 20)

    def test_in_place_dst_equals_src1(self) -> None:
        self.state.vgprs[1] = 100
        self.state.vgprs[2] = 200
        self.state.vgprs[3] = 30
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 30)
        self.assertEqual(self.state.vgprs[2], 30)

        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v2, v1, v2, v3",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src1", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 30)

    def test_in_place_dst_equals_src2(self) -> None:
        self.state.vgprs[1] = 10
        self.state.vgprs[2] = 200
        self.state.vgprs[3] = 300
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 3),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 10)
        self.assertEqual(self.state.vgprs[3], 10)

        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 200",
            "v_mov_b32 v3, 300",
            "v_min3_u32 v3, v1, v2, v3",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src2", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 10)

    def test_in_place_all_operands_aliased(self) -> None:
        self.state.vgprs[1] = 42
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 42)
        self.assertEqual(self.state.vgprs[1], 42)

        lines = [
            "v_mov_b32 v1, 42",
            "v_min3_u32 v1, v1, v1, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_all", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 42)

    def test_in_place_translated_shader(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 300",
            "v_min3_u32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inplace_min3", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_surrounding_registers_unmodified(self) -> None:
        self.state.vgprs[0] = 0xAAAA_AAAA
        self.state.vgprs[4] = 0xBBBB_BBBB
        self.state.vgprs[1] = 10
        self.state.vgprs[2] = 20
        self.state.vgprs[3] = 30
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[0], 0xAAAA_AAAA)
        self.assertEqual(self.state.vgprs[2], 10)
        self.assertEqual(self.state.vgprs[4], 0xBBBB_BBBB)

        lines = [
            "v_mov_b32 v0, 0xaaaaaaaa",
            "v_mov_b32 v4, 0xbbbbbbbb",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v2, v1, v2, v3",
            "exp mrt0 v0, v2, v4, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_surround_unmod", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xAAAAAAAA)
        self.assertEqual(words[1], 10)
        self.assertEqual(words[2], 0xBBBBBBBB)


@dataclasses.dataclass(frozen=True)
class Min3StatusFlags:
    """Status flags and operand analysis for 3-operand minimum."""
    zero: bool
    negative: bool
    is_min_src0: bool
    is_min_src1: bool
    is_min_src2: bool
    tie_count: int
    parity: bool


def compute_min3_status_flags(a: int, b: int, c: int) -> Min3StatusFlags:
    res = oracle_min3_u32(a, b, c)
    ua, ub, uc = u32(a), u32(b), u32(c)
    is0 = (res == ua)
    is1 = (res == ub)
    is2 = (res == uc)
    tie_count = int(is0) + int(is1) + int(is2)
    zero = (res == 0)
    negative = bool(res & SIGN_BIT32)
    parity = (bin(res).count("1") % 2 == 0)
    return Min3StatusFlags(
        zero=zero,
        negative=negative,
        is_min_src0=is0,
        is_min_src1=is1,
        is_min_src2=is2,
        tie_count=tie_count,
        parity=parity,
    )


class TestValuMin3U32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and extrema boundary tests for v_min3_u32."""

    def test_status_flags_zero(self) -> None:
        hw = run_translated_min3_u32(0, 10, 20)
        self.assertEqual(hw, 0)
        flags = compute_min3_status_flags(0, 10, 20)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.is_min_src0)
        self.assertFalse(flags.is_min_src1)
        self.assertEqual(flags.tie_count, 1)
        self.assertTrue(flags.parity)

    def test_status_flags_negative_msb(self) -> None:
        hw = run_translated_min3_u32(0x8000_0000, 0x9000_0000, 0xA000_0000)
        self.assertEqual(hw, 0x8000_0000)
        flags = compute_min3_status_flags(0x8000_0000, 0x9000_0000, 0xA000_0000)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.is_min_src0)
        self.assertEqual(flags.tie_count, 1)

    def test_status_flags_two_way_tie(self) -> None:
        hw = run_translated_min3_u32(15, 15, 30)
        self.assertEqual(hw, 15)
        flags = compute_min3_status_flags(15, 15, 30)
        self.assertTrue(flags.is_min_src0)
        self.assertTrue(flags.is_min_src1)
        self.assertFalse(flags.is_min_src2)
        self.assertEqual(flags.tie_count, 2)

    def test_status_flags_three_way_tie(self) -> None:
        hw = run_translated_min3_u32(42, 42, 42)
        self.assertEqual(hw, 42)
        flags = compute_min3_status_flags(42, 42, 42)
        self.assertTrue(flags.is_min_src0)
        self.assertTrue(flags.is_min_src1)
        self.assertTrue(flags.is_min_src2)
        self.assertEqual(flags.tie_count, 3)

    def test_extrema_boundary_status(self) -> None:
        hw = run_translated_min3_u32(MASK32, 0, MASK32)
        self.assertEqual(hw, 0)
        flags = compute_min3_status_flags(MASK32, 0, MASK32)
        self.assertTrue(flags.zero)
        self.assertTrue(flags.is_min_src1)
        self.assertEqual(flags.tie_count, 1)


class TestValuMin3U32IEEE754EdgeCases(unittest.TestCase):
    """Stage 7: Boundary extremes, subnormals, and NaN handling for v_min3_u32."""

    def test_integer_boundary_extremes(self) -> None:
        extremes = [
            (0, 1, 2, 0),
            (MASK32, MASK32 - 1, MASK32, MASK32 - 1),
            (0x8000_0000, 0x7FFF_FFFF, 0x8000_0001, 0x7FFF_FFFF),
            (0x1000_0000, 0x2000_0000, 0x4000_0000, 0x1000_0000),
            (0, 0, MASK32, 0),
            (MASK32, MASK32, MASK32, MASK32),
            (0, MASK32, 1, 0),
        ]
        for a, b, c, expected in extremes:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(oracle_min3_u32(a, b, c), expected)
                hw = run_translated_min3_u32(a, b, c)
                self.assertEqual(hw, expected)

    def test_float_bitpatterns_treated_as_integers(self) -> None:
        # In RDNA2, v_min3_u32 treats IEEE 754 bit-patterns strictly as raw unsigned integers
        pos_inf = struct.unpack(">I", struct.pack(">f", float("inf")))[0]   # 0x7F80_0000
        neg_inf = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]  # 0xFF80_0000
        qnan = struct.unpack(">I", struct.pack(">f", float("nan")))[0]      # 0x7FC0_0000

        # In unsigned integer order: 0x7F80_0000 < 0x7FC0_0000 < 0xFF80_0000
        self.assertEqual(oracle_min3_u32(pos_inf, neg_inf, qnan), pos_inf)
        res_trans = run_translated_min3_u32(pos_inf, neg_inf, qnan)
        self.assertEqual(res_trans, pos_inf)

    def test_float_signed_zeros_treated_as_integers(self) -> None:
        pos_zero = 0x0000_0000  # +0.0
        neg_zero = 0x8000_0000  # -0.0
        # In unsigned integer order: 0x0000_0000 (0) < 0x8000_0000 (2147483648)
        self.assertEqual(oracle_min3_u32(pos_zero, neg_zero, 100), pos_zero)
        self.assertEqual(run_translated_min3_u32(pos_zero, neg_zero, 100), pos_zero)

    def test_subnormal_bitpatterns_treated_as_integers(self) -> None:
        subnormal_min = 1
        subnormal_max = 0x007F_FFFF
        normal_min = 0x0080_0000
        self.assertEqual(oracle_min3_u32(subnormal_min, subnormal_max, normal_min), subnormal_min)
        self.assertEqual(run_translated_min3_u32(subnormal_min, subnormal_max, normal_min), subnormal_min)


class TestValuMin3U32NegativeControls(unittest.TestCase):
    """Stage 8: Negative controls, syntax rejection, and mutation detection for v_min3_u32."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106)

    def test_invalid_destination_kind(self) -> None:
        with self.assertRaises(ValueError):
            VOP3Instruction(
                Opcode.V_MIN3_U32,
                Register(RegKind.SGPR, 0),
                Register(RegKind.VGPR, 1),
                Register(RegKind.VGPR, 2),
                Register(RegKind.VGPR, 3),
            )

    def test_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=128)

    def test_mutation_detection_and_oracle_rejection(self) -> None:
        a, b, c = MASK32, 10, 20
        # In unsigned min3: min(4294967295, 10, 20) = 10
        hw = run_translated_min3_u32(a, b, c)
        self.assertEqual(hw, 10)
        # Verify execution detects max mutation
        self.assertNotEqual(hw, MASK32)

        # Verify execution detects signed min mutation (where 0xFFFFFFFF = -1 < 10)
        self.assertNotEqual(hw, s32(a) & MASK32)

        # Another case distinguishing signed vs unsigned min:
        # Unsigned: min(0x8000_0000, 100, 200) = 100
        # Signed: min(-2147483648, 100, 200) = -2147483648 (0x8000_0000)
        hw2 = run_translated_min3_u32(0x8000_0000, 100, 200)
        self.assertEqual(hw2, 100)
        self.assertNotEqual(hw2, 0x8000_0000)

    def test_translator_structural_umin_lowering(self) -> None:
        """Verify that translator lowers v_min3_u32 to OpExtInst UMin and not SMin or UMax."""
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_struct", lines)
        self.assertIn("UMin", spv)
        self.assertNotIn("SMin", spv)
        self.assertNotIn("UMax", spv)
        self.assertNotIn("SMax", spv)
        self.assertEqual(spv.count("UMin"), 2)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 10)

    def test_translator_syntax_rejection(self) -> None:
        prog = [
            "v_min3_u32 v0, v1, v2",
            "s_endpgm",
        ]
        with self.assertRaises((ValueError, rdna2spirv.Unsupported, IndexError)):
            support.translate_lines("ps_min3_missing_op", prog)

    def test_unsupported_64bit_register_rejected(self) -> None:
        prog = [
            "v_min3_u32 v0, v[1:2], v3, v4",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min3_64bit", prog)


if __name__ == "__main__":
    unittest.main()
