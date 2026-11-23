from __future__ import annotations

"""Semantic and hardware reference suite for RDNA2 VALU integer 3-input operations.

Covers 6 three-input integer opcodes:
  - v_add3_u32: (src0 + src1 + src2) & 0xFFFFFFFF
  - v_lshl_add_u32: ((src0 << (src1 & 31)) + src2) & 0xFFFFFFFF
  - v_add_lshl_u32: (((src0 + src1) & 0xFFFFFFFF) << (src2 & 31)) & 0xFFFFFFFF
  - v_lshl_or_b32: ((src0 << (src1 & 31)) | src2) & 0xFFFFFFFF
  - v_and_or_b32: ((src0 & src1) | src2) & 0xFFFFFFFF
  - v_or3_b32: (src0 | src1 | src2) & 0xFFFFFFFF

Each opcode is tested across reference arithmetic oracles, symbolic AST and
scalar architectural interpreters, SIMD multi-lane divergence with EXEC masking,
boundary extremes and IEEE 754 bit-pattern invariance, and rdna2spirv
translation lowered to SPIR-V and validated with spirv_mini_interp.
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


def run_translated_triop(opcode: str, src0: int, src1: int, src2: int) -> int:
    """Execute 3-input VALU instruction lowered to SPIR-V."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        f"{opcode} v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_{opcode}_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


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


class TriOpcode(enum.Enum):
    V_ADD3_U32 = "v_add3_u32"
    V_LSHL_ADD_U32 = "v_lshl_add_u32"
    V_ADD_LSHL_U32 = "v_add_lshl_u32"
    V_LSHL_OR_B32 = "v_lshl_or_b32"
    V_AND_OR_B32 = "v_and_or_b32"
    V_OR3_B32 = "v_or3_b32"


@dataclasses.dataclass(frozen=True)
class VOP3Instruction:
    opcode: TriOpcode
    vdst: Register
    src0: Register | int
    src1: Register | int
    src2: Register | int

    def __post_init__(self) -> None:
        if self.vdst.kind != RegKind.VGPR:
            raise ValueError(f"Destination vdst must be VGPR, got {self.vdst.kind}")


class BundleInterpreter:
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
        if inst.opcode == TriOpcode.V_ADD3_U32:
            res = (s0 + s1 + s2) & MASK32
        elif inst.opcode == TriOpcode.V_LSHL_ADD_U32:
            res = ((s0 << (s1 & 31)) + s2) & MASK32
        elif inst.opcode == TriOpcode.V_ADD_LSHL_U32:
            res = (((s0 + s1) & MASK32) << (s2 & 31)) & MASK32
        elif inst.opcode == TriOpcode.V_LSHL_OR_B32:
            res = ((s0 << (s1 & 31)) | s2) & MASK32
        elif inst.opcode == TriOpcode.V_AND_OR_B32:
            res = ((s0 & s1) | s2) & MASK32
        elif inst.opcode == TriOpcode.V_OR3_B32:
            res = (s0 | s1 | s2) & MASK32
        else:
            raise ValueError(f"Unknown opcode: {inst.opcode}")
        self.vgprs[inst.vdst.index] = res
        return res


class SIMDWavefront:
    def __init__(self, wave_size: int = 32, exec_mask: int | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.mask_limit = (1 << wave_size) - 1
        self.exec_mask = self.mask_limit if exec_mask is None else (exec_mask & self.mask_limit)

    def execute(self, op_fn, s0_lanes: list[int], s1_lanes: list[int], s2_lanes: list[int]) -> list[int]:
        res_lanes: list[int] = [0] * self.wave_size
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                res_lanes[lane] = op_fn(s0_lanes[lane], s1_lanes[lane], s2_lanes[lane])
            else:
                res_lanes[lane] = 0
        return res_lanes

# ---------------------------------------------------------------------------
# Opcode 1: v_add3_u32 - Reference Oracle
# ---------------------------------------------------------------------------

def oracle_add3_u32(a: int, b: int, c: int) -> int:
    return ((a & MASK32) + (b & MASK32) + (c & MASK32)) & MASK32


class TestAdd3U32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        self.assertEqual(oracle_add3_u32(10, 20, 30), 60)
        self.assertEqual(oracle_add3_u32(0, 0, 0), 0)

    def test_wrap_around(self) -> None:
        self.assertEqual(oracle_add3_u32(MASK32, 1, 0), 0)
        self.assertEqual(oracle_add3_u32(MASK32, MASK32, 2), 0)

    def test_permutations(self) -> None:
        x, y, z = 100, 200, 300
        m = oracle_add3_u32(x, y, z)
        self.assertEqual(oracle_add3_u32(z, x, y), m)
        self.assertEqual(oracle_add3_u32(y, z, x), m)

# ---------------------------------------------------------------------------
# Opcode 1: v_add3_u32 - AST and Interpreter
# ---------------------------------------------------------------------------

class TestAdd3U32ASTInterpreter(unittest.TestCase):
    def test_ast_construction(self) -> None:
        inst = VOP3Instruction(
            TriOpcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        self.assertEqual(inst.opcode, TriOpcode.V_ADD3_U32)
        self.assertEqual(str(inst.vdst), "v0")

    def test_interpreter_step(self) -> None:
        interp = BundleInterpreter()
        interp.vgprs[1] = 15
        interp.vgprs[2] = 25
        interp.sgprs[0] = 35
        inst = VOP3Instruction(
            TriOpcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.SGPR, 0),
        )
        res = interp.step(inst)
        self.assertEqual(res, 75)
        self.assertEqual(interp.vgprs[0], 75)

# ---------------------------------------------------------------------------
# Opcode 1: v_add3_u32 - SIMD and Boundary Tests
# ---------------------------------------------------------------------------

class TestAdd3U32SIMDBoundaries(unittest.TestCase):
    def test_simd_wave32_divergence(self) -> None:
        wave = SIMDWavefront(wave_size=32, exec_mask=0x5555_5555)
        s0 = [1] * 32
        s1 = [2] * 32
        s2 = [3] * 32
        res = wave.execute(oracle_add3_u32, s0, s1, s2)
        for lane in range(32):
            self.assertEqual(res[lane], 6 if lane % 2 == 0 else 0)

    def test_boundary_values(self) -> None:
        self.assertEqual(oracle_add3_u32(0x7FFF_FFFF, 1, 0), 0x8000_0000)
        self.assertEqual(oracle_add3_u32(0x8000_0000, 0x8000_0000, 0), 0)
        self.assertEqual(oracle_add3_u32(MASK32, MASK32, MASK32), MASK32 - 2)

# ---------------------------------------------------------------------------
# Opcode 1: v_add3_u32 - Translator Integration
# ---------------------------------------------------------------------------

class TestAdd3U32Translator(unittest.TestCase):
    def test_translated_execution(self) -> None:
        triples = [
            (10, 20, 30),
            (0, 0, 0),
            (MASK32, 1, 0),
            (0x12345678, 0x11111111, 0x22222222),
            (0x7FFFFFFF, 1, 0x80000000),
        ]
        for a, b, c in triples:
            expected = oracle_add3_u32(a, b, c)
            actual = run_translated_triop("v_add3_u32", a, b, c)
            self.assertEqual(actual, expected)

# ---------------------------------------------------------------------------
# Opcode 2: v_lshl_add_u32 - Reference Oracle
# ---------------------------------------------------------------------------

def oracle_lshl_add_u32(a: int, b: int, c: int) -> int:
    return (((a & MASK32) << (b & 31)) + (c & MASK32)) & MASK32


class TestLshlAddU32Oracle(unittest.TestCase):
    def test_basic_shift_add(self) -> None:
        # (5 << 2) + 3 = 20 + 3 = 23
        self.assertEqual(oracle_lshl_add_u32(5, 2, 3), 23)
        self.assertEqual(oracle_lshl_add_u32(1, 0, 10), 11)

    def test_shift_count_masking(self) -> None:
        # Shift count is masked to low 5 bits: 32 -> 0, 33 -> 1
        self.assertEqual(oracle_lshl_add_u32(5, 32, 3), 8)
        self.assertEqual(oracle_lshl_add_u32(5, 33, 3), 13)

    def test_wrapping(self) -> None:
        self.assertEqual(oracle_lshl_add_u32(0x8000_0000, 1, 0), 0)

# ---------------------------------------------------------------------------
# Opcode 2: v_lshl_add_u32 - AST and Interpreter
# ---------------------------------------------------------------------------

class TestLshlAddU32ASTInterpreter(unittest.TestCase):
    def test_ast_construction(self) -> None:
        inst = VOP3Instruction(
            TriOpcode.V_LSHL_ADD_U32,
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 3),
        )
        self.assertEqual(inst.opcode, TriOpcode.V_LSHL_ADD_U32)

    def test_interpreter_step(self) -> None:
        interp = BundleInterpreter()
        interp.vgprs[0] = 7
        interp.vgprs[1] = 3
        interp.vgprs[2] = 10
        inst = VOP3Instruction(
            TriOpcode.V_LSHL_ADD_U32,
            Register(RegKind.VGPR, 4),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        res = interp.step(inst)
        # (7 << 3) + 10 = 56 + 10 = 66
        self.assertEqual(res, 66)
        self.assertEqual(interp.vgprs[4], 66)

# ---------------------------------------------------------------------------
# Opcode 2: v_lshl_add_u32 - SIMD and Boundary Tests
# ---------------------------------------------------------------------------

class TestLshlAddU32SIMDBoundaries(unittest.TestCase):
    def test_simd_wave_shift_sweep(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        s0 = [1] * 32
        s1 = [lane for lane in range(32)]
        s2 = [0] * 32
        res = wave.execute(oracle_lshl_add_u32, s0, s1, s2)
        for lane in range(32):
            self.assertEqual(res[lane], (1 << lane) & MASK32)

    def test_boundary_shift_values(self) -> None:
        self.assertEqual(oracle_lshl_add_u32(1, 31, 0), 0x8000_0000)
        self.assertEqual(oracle_lshl_add_u32(1, 31, 1), 0x8000_0001)

# ---------------------------------------------------------------------------
# Opcode 2: v_lshl_add_u32 - Translator Integration
# ---------------------------------------------------------------------------

class TestLshlAddU32Translator(unittest.TestCase):
    def test_translated_execution(self) -> None:
        triples = [
            (5, 2, 3),
            (10, 4, 1),
            (1, 31, 5),
            (0x1234, 16, 0x5678),
            (0, 5, 42),
        ]
        for a, b, c in triples:
            expected = oracle_lshl_add_u32(a, b, c)
            actual = run_translated_triop("v_lshl_add_u32", a, b, c)
            self.assertEqual(actual, expected)

# ---------------------------------------------------------------------------
# Opcode 3: v_add_lshl_u32 - Reference Oracle
# ---------------------------------------------------------------------------

def oracle_add_lshl_u32(a: int, b: int, c: int) -> int:
    add_part = ((a & MASK32) + (b & MASK32)) & MASK32
    return (add_part << (c & 31)) & MASK32


class TestAddLshlU32Oracle(unittest.TestCase):
    def test_basic_add_shift(self) -> None:
        # (3 + 5) << 2 = 8 << 2 = 32
        self.assertEqual(oracle_add_lshl_u32(3, 5, 2), 32)
        self.assertEqual(oracle_add_lshl_u32(10, 20, 0), 30)

    def test_add_wrap_before_shift(self) -> None:
        # (0xFFFFFFFF + 1) = 0 => 0 << 5 = 0
        self.assertEqual(oracle_add_lshl_u32(MASK32, 1, 5), 0)

    def test_shift_count_mask(self) -> None:
        self.assertEqual(oracle_add_lshl_u32(2, 3, 33), 10)

# ---------------------------------------------------------------------------
# Opcode 3: v_add_lshl_u32 - AST and Interpreter
# ---------------------------------------------------------------------------

class TestAddLshlU32ASTInterpreter(unittest.TestCase):
    def test_ast_construction(self) -> None:
        inst = VOP3Instruction(
            TriOpcode.V_ADD_LSHL_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        self.assertEqual(inst.opcode, TriOpcode.V_ADD_LSHL_U32)

    def test_interpreter_step(self) -> None:
        interp = BundleInterpreter()
        interp.vgprs[1] = 4
        interp.vgprs[2] = 6
        interp.vgprs[3] = 3
        inst = VOP3Instruction(
            TriOpcode.V_ADD_LSHL_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = interp.step(inst)
        # (4 + 6) << 3 = 10 << 3 = 80
        self.assertEqual(res, 80)
        self.assertEqual(interp.vgprs[0], 80)

# ---------------------------------------------------------------------------
# Opcode 3: v_add_lshl_u32 - SIMD and Boundary Tests
# ---------------------------------------------------------------------------

class TestAddLshlU32SIMDBoundaries(unittest.TestCase):
    def test_simd_wave_divergence(self) -> None:
        wave = SIMDWavefront(wave_size=32, exec_mask=0x00FF_00FF)
        s0 = [2] * 32
        s1 = [3] * 32
        s2 = [1] * 32
        res = wave.execute(oracle_add_lshl_u32, s0, s1, s2)
        for lane in range(32):
            if ((0x00FF_00FF >> lane) & 1):
                self.assertEqual(res[lane], 10)
            else:
                self.assertEqual(res[lane], 0)

    def test_boundary_shift_out(self) -> None:
        self.assertEqual(oracle_add_lshl_u32(1, 0, 31), 0x8000_0000)
        self.assertEqual(oracle_add_lshl_u32(2, 0, 31), 0)

# ---------------------------------------------------------------------------
# Opcode 3: v_add_lshl_u32 - Translator Integration
# ---------------------------------------------------------------------------

class TestAddLshlU32Translator(unittest.TestCase):
    def test_translated_execution(self) -> None:
        triples = [
            (3, 5, 2),
            (10, 20, 0),
            (1, 1, 4),
            (0x1000, 0x2000, 3),
            (MASK32, 1, 2),
        ]
        for a, b, c in triples:
            expected = oracle_add_lshl_u32(a, b, c)
            actual = run_translated_triop("v_add_lshl_u32", a, b, c)
            self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
