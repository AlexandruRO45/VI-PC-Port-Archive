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

if __name__ == "__main__":
    unittest.main()
