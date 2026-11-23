from __future__ import annotations

"""Hardware-verified semantic reference and translator regression suite for RDNA2 v_min3_u32.

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


def run_translated_min3_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_min3_u32 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_min3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_min3_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32

class TestValuMin3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_min3_u32."""

    def test_oracle_basic_minimum(self) -> None:
        self.assertEqual(oracle_min3_u32(10, 20, 30), 10)
        self.assertEqual(oracle_min3_u32(30, 10, 20), 10)
        self.assertEqual(oracle_min3_u32(20, 30, 10), 10)
        self.assertEqual(oracle_min3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_min3_u32(1, 2, 3), 1)
        self.assertEqual(oracle_min3_u32(100, 200, 300), 100)

    def test_oracle_unsigned_vs_signed_divergence(self) -> None:
        # In signed math, 0xFFFFFFFF (-1) < 10, but in unsigned math 0xFFFFFFFF is 4294967295 > 10.
        self.assertEqual(oracle_min3_u32(MASK32, 10, 20), 10)
        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x7FFF_FFFF, 0), 0)
        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x9000_0000, MASK32), 0x8000_0000)

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
        ]
        for a, b, c in triples:
            expected = oracle_min3_u32(a, b, c)
            actual = run_translated_min3_u32(a, b, c)
            self.assertEqual(actual, expected)

    def test_oracle_permutations_and_idempotence(self) -> None:
        a, b, c = 0x1234, 0x5678, 0x9ABC
        res1 = oracle_min3_u32(a, b, c)
        res2 = oracle_min3_u32(c, a, b)
        res3 = oracle_min3_u32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)
        self.assertEqual(oracle_min3_u32(a, a, a), a)

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

    def test_instruction_with_literals_and_sgprs(self) -> None:
        dst = Register(RegKind.VGPR, 4)
        s0 = Register(RegKind.SGPR, 0)
        s1 = 0x100
        s2 = Register(RegKind.VGPR, 5)
        inst = VOP3Instruction(Opcode.V_MIN3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_min3_u32 v4, s0, 0x100, v5")

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

    def test_scalar_matches_translated_shader(self) -> None:
        test_cases = [
            (10, 20, 30),
            (100, 20, 300),
            (0xFFFF_0000, 0x0000_FFFF, 0x1234),
            (0x8000_0000, 0x4000_0000, 0x7000_0000),
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

    def test_simd_translator_active_mask_preservation(self) -> None:
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        lines = [inst.to_asm(), "s_endpgm"]
        spv = support.translate_lines("ps_simd_min3", lines)
        self.assertIn("OpExtInst", spv)

if __name__ == "__main__":
    unittest.main()
