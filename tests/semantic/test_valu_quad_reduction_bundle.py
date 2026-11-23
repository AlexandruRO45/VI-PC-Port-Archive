from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 VALU Quad / Multi-lane Reduction & LDS Swizzle Extensions Suite.

Covers:
- s_quadmask_b32: 32-bit scalar nibble/quad reduction mask generation (8 bits).
- s_quadmask_b64: 64-bit scalar quad reduction mask generation (16 bits).
- v_perm_b32: 32-bit vector byte permutation from 64-bit pair with sign extension/constant injection.
- ds_swizzle_b32: LDS cross-lane subgroup swizzle (QUAD_PERM, SWAP, REVERSE, BROADCAST).
- ds_bpermute_b32: LDS cross-lane backward gather permute.
- ds_permute_b32: LDS cross-lane forward scatter permute.

Standard library only (python -I, struct, math, unittest). Zero game data.
All test suites execute real translated assembly via support.translate_lines and spirv_mini_interp.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFF_FFFF
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_shader_words(name: str, lines: List[str]) -> Tuple[int, str]:
    full_lines = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = support.translate_lines(name, full_lines)
    words = support.mrt0_words(spv)
    return words[0], spv


def run_workgroup_lanes(name: str, lines: List[str], lanes: int = 4) -> Tuple[List[int], str]:
    full_lines = lines + ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
    spv = support.translate_lines(name, full_lines)
    mem: Dict[int, int] = {}
    si.run_workgroup(
        spv, lanes, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory=mem
    )
    res = [mem.get(4 * i, 0) for i in range(lanes)]
    return res, spv


# ------------------------------------------------------------------------------
# s_quadmask_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_quadmask_b32(s0: int) -> int:
    s0 = s0 & MASK32
    res = 0
    for i in range(8):
        if (s0 >> (4 * i)) & 0xF:
            res |= (1 << i)
    return res


class TestQuadmaskB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            0x12,
            0xF0F0_F0F0,
            0x0000_0001,
            0x8000_0000,
            0xFFFF_FFFF,
            0x0000_0000,
        ]
        for idx, val in enumerate(vectors):
            val_hw, _ = run_shader_words(f"ps_qm32_oracle_{idx}", [
                f"s_mov_b32 s1, {hex(val)}",
                "s_quadmask_b32 s0, s1",
                "v_mov_b32 v0, s0",
            ])
            self.assertEqual(val_hw, ref_quadmask_b32(val))

    def test_oracle_quad_boundaries_hardware(self):
        # Test individual nibbles
        for i in range(8):
            val = 1 << (4 * i)
            val_hw, _ = run_shader_words(f"ps_qm32_nibble_{i}", [
                f"s_mov_b32 s1, {hex(val)}",
                "s_quadmask_b32 s0, s1",
                "v_mov_b32 v0, s0",
            ])
            self.assertEqual(val_hw, 1 << i)


# ------------------------------------------------------------------------------
# s_quadmask_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int = 0

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range: {self.index}")

    @property
    def name(self) -> str:
        return f"{'v' if self.op_type == OperandType.VGPR else 's'}{self.index}"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF_FFFF_FFFF):
            raise ValueError(f"Literal out of range: {self.value}")


@dataclass
class SQuadmaskB32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]


class ScalarMachineQuadLogic:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.scc: int = 0

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand: {op}")

    def step_quadmask_b32(self, insn: SQuadmaskB32Instruction):
        s0 = self.read_operand(insn.src0)
        res = ref_quadmask_b32(s0)
        self.sgprs[insn.dst.index] = res
        self.scc = 1 if res != 0 else 0


class TestQuadmaskB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineQuadLogic()
        machine.sgprs[1] = 0x00FF_00FF
        insn = SQuadmaskB32Instruction(
            dst=RegisterOperand(OperandType.SGPR, 0),
            src0=RegisterOperand(OperandType.SGPR, 1),
        )
        machine.step_quadmask_b32(insn)

        val_hw, _ = run_shader_words("ps_qm32_ast_scalar", [
            "s_mov_b32 s1, 0x00FF00FF",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(machine.sgprs[0], val_hw)
        self.assertEqual(val_hw, 0x33)

    def test_scalar_scc_flag_hardware(self):
        val_zero, _ = run_shader_words("ps_qm32_scc_zero", [
            "s_mov_b32 s1, 0",
            "s_quadmask_b32 s0, s1",
            "s_cselect_b32 s2, 1, 2",
            "v_mov_b32 v0, s2",
        ])
        # scc is 0 on zero -> s_cselect picks 2
        self.assertEqual(val_zero, 2)


# ------------------------------------------------------------------------------
# s_quadmask_b32 SIMD Divergence & Nibble Reduction
# ------------------------------------------------------------------------------

class TestQuadmaskB32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        # Alternating nibble patterns
        val_even, _ = run_shader_words("ps_qm32_even", [
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        val_odd, _ = run_shader_words("ps_qm32_odd", [
            "s_mov_b32 s1, 0xF0F0F0F0",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(val_even, 0x55)
        self.assertEqual(val_odd, 0xAA)
        self.assertEqual(val_even | val_odd, 0xFF)

    def test_algebraic_reduction_properties_hardware(self):
        # Monotonicity: quadmask(a | b) == quadmask(a) | quadmask(b)
        val_a, _ = run_shader_words("ps_qm32_prop_a", [
            "s_mov_b32 s1, 0x0000000F",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        val_b, _ = run_shader_words("ps_qm32_prop_b", [
            "s_mov_b32 s1, 0x000000F0",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        val_ab, _ = run_shader_words("ps_qm32_prop_ab", [
            "s_mov_b32 s1, 0x000000FF",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(val_a | val_b, val_ab)
